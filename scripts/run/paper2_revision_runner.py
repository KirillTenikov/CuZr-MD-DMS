#!/usr/bin/env python3
"""Paper-2 revision launcher for reproducible Stage-02 branching.

This wrapper deliberately leaves ``run_mddms_pilot.py`` unchanged.  It uses
that historical generator to create LAMMPS inputs, but then:

* ``prepare`` executes only stages 00--02 (melt/quench, NPT, NVT), writes a
  hashed Stage-02 branchpoint manifest, and never runs MD-DMS;
* ``branch`` copies one saved Stage-02 NVT glass into a new directory,
  creates a Stage-03-only P20 or P50 MD-DMS job, and writes a provenance
  manifest linking the branch to its parent glass.

The script is intentionally restricted to the Paper-2 protocol.  It does not
change the potential, preparation trajectory, thermostat, deformation rule,
or analysis convention.  Revision v2 adds non-invasive Stage-03 restart checkpoints, a restart-aware resume input, and a portable external-backup helper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

STAGE_INPUTS = {
    "00": "00_prepare_melt_quench.in",
    "01": "01_relax_npt.in",
    "02": "02_equilibrate_nvt.in",
    "03": "03_mddms_shear.in",
}
PREP_STAGES = [STAGE_INPUTS[name] for name in ("00", "01", "02")]
MDDMS_STAGE = STAGE_INPUTS["03"]
CHECKPOINTED_STAGE = "03_mddms_shear.in"
BASE_STAGE = "03_mddms_shear_uncheckpointed.in"
RESUME_STAGE = "03_resume_mddms.in"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_GLOB = "restart.*"
DEFAULT_CHECKPOINT_EVERY_STEPS = 10_000


@dataclass(frozen=True)
class Protocol:
    """Arguments mirrored into the historical generator."""

    preset: str
    model_alias: str
    natoms: int
    cu_fraction: float
    density_g_cm3: float
    seed: int
    timestep_ps: float
    temperature_high_K: float
    temperature_low_K: float
    pressure_bar: float
    strain_amplitude: float
    tdamp_ps: float
    pdamp_ps: float
    mddms_period_ps: float
    mddms_cycles: int
    thermo_every_steps: int | None
    stress_every_steps: int | None
    dump_every_steps: int
    stress_sign: float
    lmp_command: str
    checkpoint_every_steps: int


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict | None:
    if not path.exists():
        return None
    return {
        "name": path.name,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_info(repo: Path) -> dict:
    def run_git(arguments: list[str]) -> str | None:
        try:
            output = subprocess.check_output(
                ["git", *arguments],
                cwd=repo,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return output.strip()

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "status_short": run_git(["status", "--short"]),
    }


def ensure_empty_new_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing to write into non-empty directory: {path}. "
            "Use a new run name; do not overwrite a realization."
        )
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing required JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Malformed JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected an object in {path}")
    return value


def write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def runtime_shell_prefix() -> list[str]:
    """The same environment setup used by the historical generated shell script."""
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "source /opt/venv/bin/activate || true",
        "if [ -f /workspace/cuzr_mddms_runtime.env ]; then",
        "  source /workspace/cuzr_mddms_runtime.env",
        "fi",
        "if [ -f /workspace/cuzr_runtime.env ]; then",
        "  source /workspace/cuzr_runtime.env",
        "fi",
        "",
    ]


def write_stage_runner(run_dir: Path, lmp_command: str, stage_inputs: Sequence[str]) -> None:
    lines = runtime_shell_prefix()
    lines.extend(
        [
            f"LMP_CMD={shlex.quote(lmp_command)}",
            "",
            "run_stage() {",
            "  local input=\"$1\"",
            "  local log=\"${input%.in}.log\"",
            "  echo \"[run] $input -> $log\"",
            "  ${LMP_CMD} -log \"$log\" -in \"$input\"",
            "}",
            "",
        ]
    )
    lines.extend(f"run_stage {shlex.quote(stage)}" for stage in stage_inputs)
    runner = run_dir / "run_lammps.sh"
    runner.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(runner, 0o755)


def run_lammps_stage(run_dir: Path, stage_input: str, lmp_command: str, dry_run: bool) -> None:
    input_path = run_dir / stage_input
    if not input_path.exists():
        raise FileNotFoundError(f"Missing LAMMPS input {input_path}")
    command = shlex.split(lmp_command) + ["-log", stage_input.replace(".in", ".log"), "-in", stage_input]
    print("[execute]", " ".join(shlex.quote(item) for item in command), flush=True)
    if dry_run:
        return
    # Match the environment bootstrap used by the historical generated runner.
    shell_lines = runtime_shell_prefix()[3:]
    shell_lines.append("exec " + shlex.join(command))
    subprocess.run(["bash", "-c", "\n".join(shell_lines)], cwd=run_dir, check=True)


def protocol_from_args(args: argparse.Namespace) -> Protocol:
    return Protocol(
        preset=args.preset,
        model_alias=args.model_alias,
        natoms=args.natoms,
        cu_fraction=args.cu_fraction,
        density_g_cm3=args.density_g_cm3,
        seed=args.seed,
        timestep_ps=args.timestep_ps,
        temperature_high_K=args.temperature_high_K,
        temperature_low_K=args.temperature_low_K,
        pressure_bar=args.pressure_bar,
        strain_amplitude=args.strain_amplitude,
        tdamp_ps=args.tdamp_ps,
        pdamp_ps=args.pdamp_ps,
        mddms_period_ps=args.mddms_period_ps,
        mddms_cycles=args.mddms_cycles,
        thermo_every_steps=args.thermo_every_steps,
        stress_every_steps=args.stress_every_steps,
        dump_every_steps=args.dump_every_steps,
        stress_sign=args.stress_sign,
        lmp_command=args.lmp_command,
        checkpoint_every_steps=args.checkpoint_every_steps,
    )


def generator_command(
    runner_path: Path,
    run_root: Path,
    run_name: str,
    protocol: Protocol,
) -> list[str]:
    """Build a no-execute invocation of the historical generator."""
    cmd = [
        sys.executable,
        str(runner_path),
        "--run-root",
        str(run_root),
        "--run-name",
        run_name,
        "--preset",
        protocol.preset,
        "--model-alias",
        protocol.model_alias,
        "--natoms",
        str(protocol.natoms),
        "--cu-fraction",
        str(protocol.cu_fraction),
        "--density-g-cm3",
        str(protocol.density_g_cm3),
        "--seed",
        str(protocol.seed),
        "--timestep-ps",
        str(protocol.timestep_ps),
        "--temperature-high-K",
        str(protocol.temperature_high_K),
        "--temperature-low-K",
        str(protocol.temperature_low_K),
        "--pressure-bar",
        str(protocol.pressure_bar),
        "--strain-amplitude",
        str(protocol.strain_amplitude),
        "--tdamp-ps",
        str(protocol.tdamp_ps),
        "--pdamp-ps",
        str(protocol.pdamp_ps),
        "--mddms-period-ps",
        str(protocol.mddms_period_ps),
        "--mddms-cycles",
        str(protocol.mddms_cycles),
        "--dump-trajectory",
        "--dump-every-steps",
        str(protocol.dump_every_steps),
        "--stress-sign",
        str(protocol.stress_sign),
        "--lmp-command",
        protocol.lmp_command,
    ]
    if protocol.thermo_every_steps is not None:
        cmd.extend(["--thermo-every-steps", str(protocol.thermo_every_steps)])
    if protocol.stress_every_steps is not None:
        cmd.extend(["--stress-every-steps", str(protocol.stress_every_steps)])
    return cmd


def run_generator(
    runner_path: Path,
    run_root: Path,
    run_name: str,
    protocol: Protocol,
    dry_run: bool,
) -> None:
    command = generator_command(runner_path, run_root, run_name, protocol)
    print("[generate]", " ".join(shlex.quote(item) for item in command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def model_record_from_metadata(run_dir: Path) -> dict:
    metadata = read_json(run_dir / "metadata.json")
    run_cfg = metadata.get("run_config")
    if not isinstance(run_cfg, dict):
        raise RuntimeError(f"metadata.json in {run_dir} has no run_config")
    model_path = Path(str(run_cfg["model_file"]))
    return {
        "alias": run_cfg.get("model_alias"),
        "kind": run_cfg.get("model_kind"),
        "path": str(model_path.resolve()),
        "sha256": sha256(model_path),
    }


def write_stage02_manifest(run_dir: Path, protocol: Protocol) -> Path:
    data_file = run_dir / "02_after_equilibrate_nvt.data"
    restart_file = run_dir / "02_after_equilibrate_nvt.restart"
    if not data_file.exists():
        raise FileNotFoundError(f"Preparation did not produce {data_file}")
    manifest = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "kind": "paper2_stage02_branchpoint",
        "run_dir": str(run_dir.resolve()),
        "protocol": asdict(protocol),
        "stage02": {
            "data": file_record(data_file),
            "restart": file_record(restart_file),
        },
        "model": model_record_from_metadata(run_dir),
        "reproducibility": {
            "revision_wrapper_sha256": sha256(Path(__file__).resolve()),
            "git": git_info(Path.cwd().resolve()),
        },
        "notes": [
            "This Stage-02 configuration is the sole parent for both P20 and P50 branches of this seed.",
            "The historical Stage-03 input recreates velocities with seed + 404 after reading the saved data file.",
        ],
    }
    destination = run_dir / "stage02_branchpoint.json"
    write_json(destination, manifest)
    return destination


def assert_parent_compatible(parent_metadata: dict, protocol: Protocol) -> None:
    parent_cfg = parent_metadata.get("run_config")
    if not isinstance(parent_cfg, dict):
        raise RuntimeError("Parent metadata.json has no run_config")

    checks = {
        "model_kind": protocol.model_alias == "mace_c" and "mace" or parent_cfg.get("model_kind"),
        "natoms": protocol.natoms,
        "seed": protocol.seed,
        "timestep_ps": protocol.timestep_ps,
        "temperature_low_K": protocol.temperature_low_K,
    }
    for key, expected in checks.items():
        observed = parent_cfg.get(key)
        if observed is None:
            continue
        if isinstance(expected, float):
            same = abs(float(observed) - expected) <= 1.0e-12
        else:
            same = observed == expected
        if not same:
            raise ValueError(
                f"Parent branchpoint mismatch for {key}: parent={observed!r}, requested={expected!r}"
            )

    if protocol.model_alias != parent_cfg.get("model_alias"):
        raise ValueError(
            "Parent and branch use different model aliases: "
            f"parent={parent_cfg.get('model_alias')!r}, requested={protocol.model_alias!r}"
        )
    parent_cu = parent_cfg.get("cu_fraction")
    if parent_cu is not None and abs(float(parent_cu) - protocol.cu_fraction) > 1.0e-12:
        raise ValueError("Parent and branch use different Cu fractions")


def remove_unused_preparation_files(branch_dir: Path) -> None:
    """A Stage-03 branch should not look like an independently prepared glass."""
    for name in [
        "initial.data",
        "00_prepare_melt_quench.in",
        "01_relax_npt.in",
        "02_equilibrate_nvt.in",
    ]:
        path = branch_dir / name
        if path.exists():
            path.unlink()



def mddms_total_steps(protocol: Protocol) -> int:
    """Return the fixed total number of Stage-03 integration steps."""
    return int(round(protocol.mddms_period_ps * protocol.mddms_cycles / protocol.timestep_ps))


def _mace_or_eam_block(model: dict) -> str:
    """Build the pair/neighbor commands for a restart input from the recorded model."""
    kind = str(model.get("kind"))
    raw_path = model.get("path")
    if not raw_path:
        raise RuntimeError("Model path is missing from the branch manifest")
    model_path = Path(str(raw_path))
    if kind == "mace":
        pair = f"pair_style mliap unified {model_path} 0\npair_coeff * * Cu Zr"
    elif kind == "eam":
        pair = f"pair_style eam/fs\npair_coeff * * {model_path} Cu Zr"
    else:
        raise RuntimeError(f"Unsupported model kind for restart input: {kind!r}")
    return "\n".join([pair, "neighbor 2.0 bin", "neigh_modify delay 0 every 1 check yes"])


def checkpoint_files(run_dir: Path) -> list[Path]:
    """Return unique-timestep restart files in numerical order."""
    folder = run_dir / CHECKPOINT_DIR
    if not folder.exists():
        return []
    candidates: list[tuple[int, Path]] = []
    for path in folder.glob(CHECKPOINT_GLOB):
        match = re.fullmatch(r"restart\.(\d+)", path.name)
        if match and path.is_file() and path.stat().st_size > 0:
            candidates.append((int(match.group(1)), path))
    return [path for _, path in sorted(candidates)]


def write_checkpoint_manifest(
    run_dir: Path,
    protocol: Protocol,
    *,
    base_input: Path,
    checkpointed_input: Path,
    resume_input: Path,
    parent_manifest: Path | None = None,
) -> Path:
    """Write deterministic checkpoint/restart provenance for one Stage-03 branch."""
    manifest: dict = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "kind": "paper2_stage03_checkpoint_plan",
        "run_dir": str(run_dir.resolve()),
        "checkpoint": {
            "directory": CHECKPOINT_DIR,
            "filename_pattern": f"{CHECKPOINT_DIR}/{CHECKPOINT_GLOB}",
            "every_steps": protocol.checkpoint_every_steps,
            "every_ps": protocol.checkpoint_every_steps * protocol.timestep_ps,
            "total_steps": mddms_total_steps(protocol),
            "resume_input": RESUME_STAGE,
            "note": "Each wildcard restart filename includes its completed timestep; the highest complete timestep is the preferred resume point.",
        },
        "inputs": {
            "historical_generated_stage03": file_record(base_input),
            "checkpointed_stage03": file_record(checkpointed_input),
            "resume_stage03": file_record(resume_input),
        },
        "protocol": asdict(protocol),
        "reproducibility": {
            "revision_wrapper_sha256": sha256(Path(__file__).resolve()),
            "git": git_info(Path.cwd().resolve()),
        },
        "notes": [
            "Checkpoint writes do not change the MD-DMS force field, thermostat, deformation variables, timestep, period, or cycle count.",
            "A restart must be resumed with the same Docker image, LAMMPS binary/build, GPU execution layout, and MACE model recorded for this run.",
            "A local restart is not an off-machine backup. Run checkpoint_sync.py against a persistent external destination while production is running.",
        ],
    }
    if parent_manifest is not None and parent_manifest.exists():
        manifest["parent_branch_manifest"] = file_record(parent_manifest)
    destination = run_dir / "checkpoint_manifest.json"
    write_json(destination, manifest)
    return destination


def patch_stage03_for_checkpoints(run_dir: Path, protocol: Protocol, model: dict) -> tuple[Path, Path, Path]:
    """Preserve the generated Stage-03 input, add restart writes, and create a resume input.

    The historical generator remains unchanged.  The first-run input differs only by
    periodic binary restart output.  The resume input reads one of those restarts and
    reissues the same potential, NVT, and variable-form xy deformation definitions.
    """
    stage_path = run_dir / CHECKPOINTED_STAGE
    if not stage_path.exists():
        raise FileNotFoundError(f"Missing generated Stage-03 input: {stage_path}")
    if protocol.checkpoint_every_steps <= 0:
        raise ValueError("--checkpoint-every-steps must be positive")
    if protocol.checkpoint_every_steps % protocol.dump_every_steps != 0:
        raise ValueError(
            "checkpoint cadence must be an integer multiple of dump cadence so every restart aligns with a saved trajectory frame"
        )
    original = stage_path.read_text(encoding="utf-8")
    if "\nrestart " in original:
        raise RuntimeError(f"{stage_path} already contains restart directives; refusing to patch twice")
    if "reset_timestep 0" not in original or "velocity all create" not in original:
        raise RuntimeError("Unexpected Stage-03 structure: expected a fresh Stage-03 input generated by the historical runner")

    base_path = run_dir / BASE_STAGE
    if base_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing preserved input: {base_path}")
    shutil.copy2(stage_path, base_path)

    expected_steps = mddms_total_steps(protocol)
    run_pattern = re.compile(r"(?m)^(?P<indent>\s*)run\s+(?P<steps>\d+)\s*$")
    matches = list(run_pattern.finditer(original))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one LAMMPS run command in Stage-03 input; found {len(matches)}")
    match = matches[0]
    observed_steps = int(match.group("steps"))
    if observed_steps != expected_steps:
        raise RuntimeError(
            f"Generated Stage-03 run length ({observed_steps}) does not match protocol ({expected_steps})"
        )
    checkpoint_line = (
        "# Paper-2 revision safety: binary restart after every completed checkpoint interval.\n"
        f"restart {protocol.checkpoint_every_steps} {CHECKPOINT_DIR}/{CHECKPOINT_GLOB}\n"
    )
    replacement = checkpoint_line + match.group(0) + "\nrestart 0"
    checkpointed = original[: match.start()] + replacement + original[match.end() :]
    stage_path.write_text(checkpointed, encoding="utf-8")

    metadata = read_json(run_dir / "metadata.json")
    cfg = metadata.get("run_config")
    preset = metadata.get("preset")
    if not isinstance(cfg, dict) or not isinstance(preset, dict):
        raise RuntimeError("Generated metadata.json is missing run_config or effective preset")
    if cfg.get("dump_atomic") or cfg.get("dump_voronoi"):
        raise RuntimeError("Paper-2 revision branches must not enable atomic-stress or Voronoi dumps")
    dump_trajectory = bool(cfg.get("dump_trajectory"))
    dump_every = int(cfg.get("dump_every_steps") or preset.get("dump_every_steps"))
    stress_every = int(preset.get("stress_every_steps"))
    thermo_every = int(preset.get("thermo_every_steps"))
    dt = float(cfg.get("timestep_ps"))
    temperature = float(cfg.get("temperature_low_K"))
    tdamp = float(cfg.get("tdamp_ps"))
    gamma0 = float(cfg.get("strain_amplitude"))
    period = float(preset.get("mddms_period_ps"))

    trajectory_lines = ""
    if dump_trajectory:
        trajectory_lines = (
            f"dump traj all custom {dump_every} trajectory.lammpstrj id type x y z ix iy iz\n"
            "dump_modify traj sort id append yes\n"
        )
    resume_text = f"""# Generated by paper2_revision_runner.py. Do not edit by hand.
# Resume the checkpointed Stage-03 MD-DMS trajectory without resetting time,
# velocities, or the sinusoidal deformation phase.
units metal
atom_style atomic
boundary p p p
newton on
read_restart ${{restart_file}}
{_mace_or_eam_block(model)}
timestep {dt:.8f}
thermo {thermo_every}
thermo_style custom step time temp pe ke etotal press pxx pyy pzz pxy lx ly lz xy
# IMPORTANT: this resume input preserves the current timestep, velocities, and deformation phase.
variable gamma0 equal {gamma0:.12f}
variable period equal {period:.12f}
variable omega equal 2.0*{3.14159265358979323846:.20f}/v_period
variable gamma equal v_gamma0*sin(v_omega*time)
variable gammadot equal v_gamma0*v_omega*cos(v_omega*time)
variable xy_target equal v_gamma*ly
variable xy_rate equal v_gammadot*ly
fix thermostat all nvt temp {temperature:.6f} {temperature:.6f} {tdamp:.6f}
fix deform all deform 1 xy variable v_xy_target v_xy_rate remap x
variable time_ps equal time
variable pxy_bar equal pxy
variable temp_K equal temp
variable pe_eV equal pe
variable ke_eV equal ke
variable press_bar equal press
variable xy_A equal xy
variable ly_A equal ly
fix ts all ave/time {stress_every} 1 {stress_every} v_time_ps v_gamma v_pxy_bar v_temp_K v_pe_eV v_ke_eV v_press_bar v_xy_A v_ly_A append stress_timeseries.dat
{trajectory_lines}restart {protocol.checkpoint_every_steps} {CHECKPOINT_DIR}/{CHECKPOINT_GLOB}
run {expected_steps} upto
restart 0
unfix ts
unfix deform
unfix thermostat
write_data 03_after_mddms.data
write_restart 03_after_mddms.restart
"""
    resume_path = run_dir / RESUME_STAGE
    resume_path.write_text(resume_text, encoding="utf-8")
    (run_dir / CHECKPOINT_DIR).mkdir(exist_ok=True)
    return base_path, stage_path, resume_path


def write_checkpoint_sync_helper(run_dir: Path) -> Path:
    """Create a portable Python copier for persistent/off-machine checkpoint backup.

    It intentionally uses only the Python standard library so it works inside the
    project container without requiring rsync.  It writes every destination file
    atomically through a temporary name, so an interrupted copy leaves the prior
    verified destination file intact.
    """
    destination = run_dir / "checkpoint_sync.py"
    source = r"""#!/usr/bin/env python3
# Mirror a live Paper-2 Stage-03 run to an external persistent destination.
# Examples:
#   python checkpoint_sync.py --backup-dir /mnt/persistent/Paper2 --once
#   python checkpoint_sync.py --backup-dir /mnt/persistent/Paper2 --loop --interval-sec 60 --trajectory-interval-sec 300
# The backup destination must be outside the disposable VM/container filesystem.
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomically_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    with src.open("rb") as inp, tmp.open("wb") as out:
        shutil.copyfileobj(inp, out, length=1024 * 1024)
        out.flush()
        os.fsync(out.fileno())
    shutil.copystat(src, tmp, follow_symlinks=True)
    os.replace(tmp, dst)


def files_to_copy(include_trajectory: bool) -> list[Path]:
    names = [
        "03_mddms_shear.in", "03_mddms_shear_uncheckpointed.in", "03_resume_mddms.in",
        "metadata.json", "branch_manifest.json", "checkpoint_manifest.json",
        "stress_timeseries.dat", "03_mddms_shear.log", "run_lammps.sh",
    ]
    selected = [RUN_DIR / name for name in names if (RUN_DIR / name).is_file()]
    selected.extend(sorted((RUN_DIR / "checkpoints").glob("restart.*")) if (RUN_DIR / "checkpoints").exists() else [])
    if include_trajectory and (RUN_DIR / "trajectory.lammpstrj").is_file():
        selected.append(RUN_DIR / "trajectory.lammpstrj")
    return selected


def sync_once(backup_base: Path, include_trajectory: bool) -> dict:
    target = backup_base.resolve() / RUN_DIR.name
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in files_to_copy(include_trajectory):
        rel = src.relative_to(RUN_DIR)
        dst = target / rel
        atomically_copy(src, dst)
        copied.append({"path": str(rel), "bytes": dst.stat().st_size, "sha256": sha256(dst)})
    status = {
        "updated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_run_dir": str(RUN_DIR),
        "backup_run_dir": str(target),
        "trajectory_copied": include_trajectory,
        "files": copied,
    }
    (target / "checkpoint_backup_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Mirror live checkpoints to an external persistent backup destination.")
    parser.add_argument("--backup-dir", default=os.environ.get("MDDMS_BACKUP_DIR"), help="External persistent directory; never the run directory itself.")
    parser.add_argument("--once", action="store_true", help="Copy once and exit.")
    parser.add_argument("--loop", action="store_true", help="Continue copying until interrupted.")
    parser.add_argument("--interval-sec", type=float, default=60.0)
    parser.add_argument("--trajectory-interval-sec", type=float, default=300.0)
    args = parser.parse_args()
    if bool(args.once) == bool(args.loop):
        parser.error("choose exactly one of --once or --loop")
    if not args.backup_dir:
        parser.error("--backup-dir or MDDMS_BACKUP_DIR is required")
    backup = Path(args.backup_dir).expanduser().resolve()
    if backup == RUN_DIR or RUN_DIR in backup.parents:
        parser.error("backup destination must be outside the live run directory")
    if args.interval_sec <= 0 or args.trajectory_interval_sec <= 0:
        parser.error("intervals must be positive")
    last_trajectory = 0.0
    while True:
        now = time.monotonic()
        include_trajectory = now - last_trajectory >= args.trajectory_interval_sec
        sync_once(backup, include_trajectory)
        if include_trajectory:
            last_trajectory = now
        if args.once:
            return 0
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    raise SystemExit(main())
"""
    destination.write_text(source, encoding="utf-8")
    os.chmod(destination, 0o755)
    return destination


def latest_checkpoint(run_dir: Path, explicit: str | None = None) -> Path:
    if explicit:
        selected = Path(explicit)
        if not selected.is_absolute():
            selected = run_dir / selected
        selected = selected.resolve()
        if not selected.exists() or selected.stat().st_size == 0:
            raise FileNotFoundError(f"Requested restart is missing or empty: {selected}")
        return selected
    files = checkpoint_files(run_dir)
    if not files:
        raise FileNotFoundError(f"No completed checkpoint files found under {run_dir / CHECKPOINT_DIR}")
    return files[-1]


def run_lammps_resume(run_dir: Path, restart_file: Path, lmp_command: str, dry_run: bool) -> None:
    input_path = run_dir / RESUME_STAGE
    if not input_path.exists():
        raise FileNotFoundError(f"Missing resume input: {input_path}")
    command = shlex.split(lmp_command) + [
        "-var", "restart_file", str(restart_file),
        "-log", "03_resume_mddms.log",
        "-in", RESUME_STAGE,
    ]
    print("[resume]", " ".join(shlex.quote(item) for item in command), flush=True)
    if dry_run:
        return
    shell_lines = runtime_shell_prefix()[3:]
    shell_lines.append("exec " + shlex.join(command))
    subprocess.run(["bash", "-c", "\n".join(shell_lines)], cwd=run_dir, check=True)


def resume(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "checkpoint_manifest.json"
    manifest = read_json(manifest_path)
    protocol_data = manifest.get("protocol")
    if not isinstance(protocol_data, dict):
        raise RuntimeError(f"Checkpoint manifest has no protocol: {manifest_path}")
    protocol = Protocol(**protocol_data)
    restart = latest_checkpoint(run_dir, args.restart_file)
    resume_log = run_dir / "resume_history.json"
    previous: list[dict] = []
    if resume_log.exists():
        existing = read_json(resume_log)
        prior = existing.get("attempts")
        if isinstance(prior, list):
            previous = prior
    previous.append({
        "requested_utc": utc_now(),
        "restart": file_record(restart),
        "lmp_command": protocol.lmp_command,
        "dry_run": bool(args.dry_run),
    })
    write_json(resume_log, {"schema_version": 1, "run_dir": str(run_dir), "attempts": previous})
    run_lammps_resume(run_dir, restart, protocol.lmp_command, args.dry_run)
    return 0

def prepare(args: argparse.Namespace) -> int:
    protocol = protocol_from_args(args)
    run_root = Path(args.run_root)
    run_dir = (run_root / args.run_name).resolve()
    runner_path = Path(args.runner_path).resolve()
    if not runner_path.exists():
        raise FileNotFoundError(f"Historical runner not found: {runner_path}")
    ensure_empty_new_dir(run_dir)
    # The historical generator requires the target directory not to exist yet.
    run_dir.rmdir()
    run_generator(runner_path, run_root, args.run_name, protocol, args.dry_run)
    if args.dry_run:
        return 0

    # The historical generator creates the run directory.  It is now safe to run 00--02 only.
    for stage in PREP_STAGES:
        run_lammps_stage(run_dir, stage, protocol.lmp_command, dry_run=False)
    write_stage_runner(run_dir, protocol.lmp_command, PREP_STAGES)
    manifest = write_stage02_manifest(run_dir, protocol)
    print(f"[ok] Stage-02 branchpoint prepared: {run_dir}")
    print(f"[ok] Manifest: {manifest}")
    return 0


def branch(args: argparse.Namespace) -> int:
    protocol = protocol_from_args(args)
    parent_dir = Path(args.parent_run_dir).resolve()
    parent_data = parent_dir / "02_after_equilibrate_nvt.data"
    parent_restart = parent_dir / "02_after_equilibrate_nvt.restart"
    parent_metadata_path = parent_dir / "metadata.json"
    if not parent_data.exists():
        raise FileNotFoundError(f"Parent is missing {parent_data}")
    parent_metadata = read_json(parent_metadata_path)
    assert_parent_compatible(parent_metadata, protocol)

    run_root = Path(args.run_root)
    branch_dir = (run_root / args.run_name).resolve()
    runner_path = Path(args.runner_path).resolve()
    if not runner_path.exists():
        raise FileNotFoundError(f"Historical runner not found: {runner_path}")
    ensure_empty_new_dir(branch_dir)
    branch_dir.rmdir()
    run_generator(runner_path, run_root, args.run_name, protocol, args.dry_run)
    if args.dry_run:
        return 0

    shutil.copy2(parent_data, branch_dir / parent_data.name)
    if parent_restart.exists():
        shutil.copy2(parent_restart, branch_dir / parent_restart.name)
    shutil.copy2(parent_metadata_path, branch_dir / "parent_metadata.json")
    parent_branchpoint = parent_dir / "stage02_branchpoint.json"
    if parent_branchpoint.exists():
        shutil.copy2(parent_branchpoint, branch_dir / "parent_stage02_branchpoint.json")
    remove_unused_preparation_files(branch_dir)
    metadata = read_json(branch_dir / "metadata.json")
    model = model_record_from_metadata(branch_dir)
    base_stage, checkpointed_stage, resume_stage = patch_stage03_for_checkpoints(branch_dir, protocol, model)
    sync_helper = write_checkpoint_sync_helper(branch_dir)
    write_stage_runner(branch_dir, protocol.lmp_command, [MDDMS_STAGE])

    metadata = read_json(branch_dir / "metadata.json")
    metadata["run_type"] = "paper2_stage03_branch"
    metadata["initial_structure"] = {
        "not_used": True,
        "reason": "The branch starts from the copied parent Stage-02 NVT state.",
    }
    metadata["parent_run_dir"] = str(parent_dir)
    metadata["branch_manifest"] = "branch_manifest.json"
    metadata["checkpoint_manifest"] = "checkpoint_manifest.json"
    metadata["checkpoint_every_steps"] = protocol.checkpoint_every_steps
    metadata["checkpoint_resume_input"] = RESUME_STAGE
    metadata["checkpoint_sync_helper"] = sync_helper.name
    write_json(branch_dir / "metadata.json", metadata)

    manifest = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "kind": "paper2_stage03_branch",
        "parent_run_dir": str(parent_dir),
        "parent_stage02": {
            "data": file_record(parent_data),
            "restart": file_record(parent_restart),
            "metadata": file_record(parent_metadata_path),
            "branchpoint_manifest": file_record(parent_branchpoint),
        },
        "copied_stage02": {
            "data": file_record(branch_dir / parent_data.name),
            "restart": file_record(branch_dir / parent_restart.name),
        },
        "protocol": asdict(protocol),
        "model": model_record_from_metadata(branch_dir),
        "reproducibility": {
            "revision_wrapper_sha256": sha256(Path(__file__).resolve()),
            "git": git_info(Path.cwd().resolve()),
        },
        "notes": [
            "Only Stage 03 is present in this branch; melt, quench, NPT, and NVT were not rerun.",
            "P20 and P50 are comparable branches only when their parent_stage02.data SHA-256 is identical.",
        ],
    }
    branch_manifest_path = branch_dir / "branch_manifest.json"
    write_json(branch_manifest_path, manifest)
    checkpoint_manifest = write_checkpoint_manifest(
        branch_dir,
        protocol,
        base_input=base_stage,
        checkpointed_input=checkpointed_stage,
        resume_input=resume_stage,
        parent_manifest=branch_manifest_path,
    )

    print(f"[ok] Stage-03 branch generated: {branch_dir}")
    print(f"[ok] Checkpoints: every {protocol.checkpoint_every_steps} steps in {branch_dir / CHECKPOINT_DIR}")
    print(f"[ok] Resume input: {resume_stage}")
    print(f"[ok] External backup helper: {sync_helper}")
    print(f"[ok] Checkpoint manifest: {checkpoint_manifest}")
    if args.execute:
        run_lammps_stage(branch_dir, MDDMS_STAGE, protocol.lmp_command, dry_run=False)
    return 0


def add_protocol_arguments(parser: argparse.ArgumentParser, *, include_period: bool = True) -> None:
    parser.add_argument("--runner-path", default="scripts/run/run_mddms_pilot.py")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preset", default="pressure_relaxed")
    parser.add_argument("--model-alias", default="mace_c", choices=["mace_c"])
    parser.add_argument("--natoms", type=int, default=4000)
    parser.add_argument("--cu-fraction", type=float, default=0.64)
    parser.add_argument("--density-g-cm3", type=float, default=7.20)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timestep-ps", type=float, default=0.001)
    parser.add_argument("--temperature-high-K", type=float, default=3000.0)
    parser.add_argument("--temperature-low-K", type=float, default=300.0)
    parser.add_argument("--pressure-bar", type=float, default=0.0)
    parser.add_argument("--strain-amplitude", type=float, default=0.01)
    parser.add_argument("--tdamp-ps", type=float, default=0.1)
    parser.add_argument("--pdamp-ps", type=float, default=1.0)
    if include_period:
        parser.add_argument("--mddms-period-ps", type=float, default=50.0)
    parser.add_argument("--mddms-cycles", type=int, default=6)
    parser.add_argument("--thermo-every-steps", type=int, default=None)
    parser.add_argument("--stress-every-steps", type=int, default=None)
    parser.add_argument("--dump-every-steps", type=int, default=1000)
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY_STEPS,
        help="Write a binary Stage-03 restart at this interval; must align with dump cadence.",
    )
    parser.add_argument("--stress-sign", type=float, default=-1.0)
    parser.add_argument(
        "--lmp-command",
        default="lmp -k on g 1 -sf kk -pk kokkos newton on neigh half",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and branch the Paper-2 five-realization MACE revision campaign.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Generate and execute stages 00--02 only.")
    add_protocol_arguments(prepare_parser)
    prepare_parser.add_argument("--dry-run", action="store_true")
    prepare_parser.set_defaults(function=prepare)

    branch_parser = subparsers.add_parser("branch", help="Create a Stage-03-only MD-DMS branch.")
    add_protocol_arguments(branch_parser)
    branch_parser.add_argument("--parent-run-dir", required=True)
    branch_parser.add_argument("--execute", action="store_true", help="Run Stage 03 after generating the branch.")
    branch_parser.add_argument("--dry-run", action="store_true")
    branch_parser.set_defaults(function=branch)

    resume_parser = subparsers.add_parser("resume", help="Resume a checkpointed Stage-03 branch from its latest completed restart.")
    resume_parser.add_argument("--run-dir", required=True, help="Existing Stage-03 branch directory.")
    resume_parser.add_argument("--restart-file", default=None, help="Optional explicit restart path; default is the highest completed checkpoint timestep.")
    resume_parser.add_argument("--dry-run", action="store_true", help="Show the LAMMPS resume command without executing it.")
    resume_parser.set_defaults(function=resume)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"prepare", "branch"}:
        if args.mddms_cycles != 6:
            parser.error("This Paper-2 revision wrapper is restricted to six MD-DMS cycles.")
        if args.natoms != 4000 or abs(args.cu_fraction - 0.64) > 1.0e-12:
            parser.error("This Paper-2 revision wrapper is restricted to Cu64Zr36 with 4000 atoms.")
        if abs(args.temperature_low_K - 300.0) > 1.0e-12 or abs(args.strain_amplitude - 0.01) > 1.0e-12:
            parser.error("This Paper-2 revision wrapper is restricted to 300 K and gamma0 = 0.01.")
        if args.command == "branch" and args.mddms_period_ps not in {20.0, 50.0}:
            parser.error("Paper-2 Stage-03 branches must use --mddms-period-ps 20 or 50.")
        if args.checkpoint_every_steps <= 0:
            parser.error("--checkpoint-every-steps must be positive.")
        if args.checkpoint_every_steps % args.dump_every_steps != 0:
            parser.error("--checkpoint-every-steps must be an integer multiple of --dump-every-steps.")
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
