#!/usr/bin/env python3
"""Generic Stage-02/Stage-03 branching for Cu-Zr MD-DMS.

This runner intentionally reuses the already-tested Paper-2 revision helpers
instead of copying them.  ``run_mddms_pilot.py`` remains the source of truth
for LAMMPS input generation; this script adds flexible branching controls for
later campaigns such as Paper 3.

Two Stage-03 start modes are supported:

* ``data-recreate-velocities``: historical behavior used by Papers 2/3;
* ``restart-preserve-velocities``: read the Stage-02 restart and keep the
  equilibrated velocities.

The historical Paper-2 runner is left untouched, so old commands remain valid.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import paper2_revision_runner as legacy

START_DATA_RECREATE = "data-recreate-velocities"
START_RESTART_PRESERVE = "restart-preserve-velocities"
START_MODES = (START_DATA_RECREATE, START_RESTART_PRESERVE)
GENERATED_STAGE = "03_mddms_shear_generated.in"


def _kind(slug: str, suffix: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", slug.strip()) or "mddms"
    return f"{clean}_{suffix}"


def _protocol_from_args(args: argparse.Namespace) -> legacy.Protocol:
    return legacy.Protocol(
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


def _generator_command_with_preparation_overrides(
    runner_path: Path,
    run_root: Path,
    run_name: str,
    protocol: legacy.Protocol,
    args: argparse.Namespace,
) -> list[str]:
    cmd = legacy.generator_command(runner_path, run_root, run_name, protocol)
    for flag, attr in [
        ("--melt-ps", "melt_ps"),
        ("--quench-rate-K-per-ps", "quench_rate_K_per_ps"),
        ("--relax-ps", "relax_ps"),
        ("--equilibrate-ps", "equilibrate_ps"),
    ]:
        value = getattr(args, attr, None)
        if value is not None:
            cmd.extend([flag, str(value)])
    return cmd


def _run_generator_with_overrides(
    runner_path: Path,
    run_root: Path,
    run_name: str,
    protocol: legacy.Protocol,
    args: argparse.Namespace,
) -> None:
    cmd = _generator_command_with_preparation_overrides(
        runner_path, run_root, run_name, protocol, args
    )
    print("[generate]", " ".join(shlex.quote(x) for x in cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, check=True)


def _write_generic_stage02_manifest(
    run_dir: Path,
    protocol: legacy.Protocol,
    campaign_slug: str,
) -> Path:
    data_file = run_dir / "02_after_equilibrate_nvt.data"
    restart_file = run_dir / "02_after_equilibrate_nvt.restart"
    if not data_file.exists():
        raise FileNotFoundError(f"Preparation did not produce {data_file}")
    manifest = {
        "schema_version": 2,
        "created_utc": legacy.utc_now(),
        "kind": _kind(campaign_slug, "stage02_branchpoint"),
        "run_dir": str(run_dir.resolve()),
        "protocol": asdict(protocol),
        "stage02": {
            "data": legacy.file_record(data_file),
            "restart": legacy.file_record(restart_file),
        },
        "model": legacy.model_record_from_metadata(run_dir),
        "notes": [
            "Stage 02 is the reusable parent state for Stage-03 controls.",
            "Historical mode recreates Stage-03 velocities after read_data.",
            "Restart-preserving mode keeps the equilibrated Stage-02 velocities.",
        ],
    }
    destination = run_dir / "stage02_branchpoint.json"
    legacy.write_json(destination, manifest)
    return destination


def _apply_start_mode(branch_dir: Path, start_mode: str) -> Path:
    stage = branch_dir / legacy.MDDMS_STAGE
    generated = branch_dir / GENERATED_STAGE
    if generated.exists():
        raise FileExistsError(f"Refusing to overwrite {generated}")
    shutil.copy2(stage, generated)

    text = stage.read_text(encoding="utf-8")
    if start_mode == START_DATA_RECREATE:
        if "read_data       02_after_equilibrate_nvt.data" not in text and "read_data 02_after_equilibrate_nvt.data" not in text:
            raise RuntimeError("Historical start mode expected read_data for Stage 02")
        if "velocity        all create" not in text and "velocity all create" not in text:
            raise RuntimeError("Historical start mode expected velocity recreation")
        return generated

    if start_mode != START_RESTART_PRESERVE:
        raise ValueError(f"Unknown start mode: {start_mode}")

    restart = branch_dir / "02_after_equilibrate_nvt.restart"
    if not restart.exists():
        raise FileNotFoundError(
            "restart-preserve-velocities requires 02_after_equilibrate_nvt.restart"
        )

    text, n_read = re.subn(
        r"(?m)^read_data\s+02_after_equilibrate_nvt\.data\s*$",
        "read_restart    02_after_equilibrate_nvt.restart",
        text,
        count=1,
    )
    if n_read != 1:
        raise RuntimeError("Expected exactly one Stage-02 read_data command")

    text, n_velocity = re.subn(
        r"(?m)^velocity\s+all\s+create\s+.*$",
        "# Stage-02 velocities preserved from restart; no velocity recreation.",
        text,
        count=1,
    )
    if n_velocity != 1:
        raise RuntimeError("Expected exactly one Stage-03 velocity creation command")
    stage.write_text(text, encoding="utf-8")
    return generated


def _patch_restart_mode_for_checkpoints(
    run_dir: Path,
    protocol: legacy.Protocol,
    model: dict,
) -> tuple[Path, Path, Path]:
    """Checkpoint patch equivalent to the legacy helper, without requiring velocity create."""
    stage = run_dir / legacy.MDDMS_STAGE
    original = stage.read_text(encoding="utf-8")
    if "read_restart    02_after_equilibrate_nvt.restart" not in original:
        raise RuntimeError("Restart-preserving Stage 03 does not read the Stage-02 restart")
    if re.search(r"(?m)^velocity\s+all\s+create\b", original):
        raise RuntimeError("Restart-preserving Stage 03 must not recreate velocities")

    base = run_dir / legacy.BASE_STAGE
    if base.exists():
        raise FileExistsError(f"Refusing to overwrite {base}")
    shutil.copy2(stage, base)

    expected_steps = legacy.mddms_total_steps(protocol)
    pattern = re.compile(r"(?m)^(?P<indent>\s*)run\s+(?P<steps>\d+)\s*$")
    matches = list(pattern.finditer(original))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Stage-03 run command; found {len(matches)}")
    match = matches[0]
    if int(match.group("steps")) != expected_steps:
        raise RuntimeError("Generated Stage-03 run length does not match the protocol")
    checkpoint = (
        "# Generic branch safety: periodic binary restart.\n"
        f"restart {protocol.checkpoint_every_steps} "
        f"{legacy.CHECKPOINT_DIR}/{legacy.CHECKPOINT_GLOB}\n"
    )
    stage.write_text(
        original[: match.start()]
        + checkpoint
        + match.group(0)
        + "\nrestart 0"
        + original[match.end() :],
        encoding="utf-8",
    )

    metadata = legacy.read_json(run_dir / "metadata.json")
    cfg = metadata["run_config"]
    preset = metadata["preset"]
    dump_trajectory = bool(cfg.get("dump_trajectory"))
    dump_every = int(cfg.get("dump_every_steps") or preset["dump_every_steps"])
    stress_every = int(preset["stress_every_steps"])
    thermo_every = int(preset["thermo_every_steps"])
    dt = float(cfg["timestep_ps"])
    temperature = float(cfg["temperature_low_K"])
    tdamp = float(cfg["tdamp_ps"])
    gamma0 = float(cfg["strain_amplitude"])
    period = float(preset["mddms_period_ps"])

    trajectory = ""
    if dump_trajectory:
        trajectory = (
            f"dump traj all custom {dump_every} trajectory.lammpstrj id type x y z ix iy iz\n"
            "dump_modify traj sort id append yes\n"
        )

    resume = run_dir / legacy.RESUME_STAGE
    resume.write_text(
        f"""# Generated by mddms_branch_runner.py. Do not edit by hand.\nunits metal\natom_style atomic\nboundary p p p\nnewton on\nread_restart ${{restart_file}}\n{legacy._mace_or_eam_block(model)}\ntimestep {dt:.8f}\nthermo {thermo_every}\nthermo_style custom step time temp pe ke etotal press pxx pyy pzz pxy lx ly lz xy\nvariable gamma0 equal {gamma0:.12f}\nvariable period equal {period:.12f}\nvariable omega equal 2.0*3.14159265358979323846/v_period\nvariable gamma equal v_gamma0*sin(v_omega*time)\nvariable gammadot equal v_gamma0*v_omega*cos(v_omega*time)\nvariable xy_target equal v_gamma*ly\nvariable xy_rate equal v_gammadot*ly\nfix thermostat all nvt temp {temperature:.6f} {temperature:.6f} {tdamp:.6f}\nfix deform all deform 1 xy variable v_xy_target v_xy_rate remap x\nvariable time_ps equal time\nvariable pxy_bar equal pxy\nvariable temp_K equal temp\nvariable pe_eV equal pe\nvariable ke_eV equal ke\nvariable press_bar equal press\nvariable xy_A equal xy\nvariable ly_A equal ly\nfix ts all ave/time {stress_every} 1 {stress_every} v_time_ps v_gamma v_pxy_bar v_temp_K v_pe_eV v_ke_eV v_press_bar v_xy_A v_ly_A append stress_timeseries.dat\n{trajectory}restart {protocol.checkpoint_every_steps} {legacy.CHECKPOINT_DIR}/{legacy.CHECKPOINT_GLOB}\nrun {expected_steps} upto\nrestart 0\nunfix ts\nunfix deform\nunfix thermostat\nwrite_data 03_after_mddms.data\nwrite_restart 03_after_mddms.restart\n""",
        encoding="utf-8",
    )
    (run_dir / legacy.CHECKPOINT_DIR).mkdir(exist_ok=True)
    return base, stage, resume


def _write_generic_checkpoint_manifest(
    run_dir: Path,
    protocol: legacy.Protocol,
    campaign_slug: str,
    start_mode: str,
    generated: Path,
    base: Path,
    stage: Path,
    resume: Path,
    branch_manifest: Path,
) -> None:
    legacy.write_json(
        run_dir / "checkpoint_manifest.json",
        {
            "schema_version": 2,
            "created_utc": legacy.utc_now(),
            "kind": _kind(campaign_slug, "stage03_checkpoint_plan"),
            "start_mode": start_mode,
            "protocol": asdict(protocol),
            "checkpoint": {
                "directory": legacy.CHECKPOINT_DIR,
                "filename_pattern": f"{legacy.CHECKPOINT_DIR}/{legacy.CHECKPOINT_GLOB}",
                "every_steps": protocol.checkpoint_every_steps,
                "every_ps": protocol.checkpoint_every_steps * protocol.timestep_ps,
                "total_steps": legacy.mddms_total_steps(protocol),
                "resume_input": legacy.RESUME_STAGE,
            },
            "inputs": {
                "historical_generator_output": legacy.file_record(generated),
                "effective_uncheckpointed_stage03": legacy.file_record(base),
                "checkpointed_stage03": legacy.file_record(stage),
                "resume_stage03": legacy.file_record(resume),
            },
            "parent_branch_manifest": legacy.file_record(branch_manifest),
        },
    )


def prepare(args: argparse.Namespace) -> int:
    protocol = _protocol_from_args(args)
    run_root = Path(args.run_root)
    run_dir = (run_root / args.run_name).resolve()
    runner = Path(args.runner_path).resolve()
    legacy.ensure_empty_new_dir(run_dir)
    run_dir.rmdir()
    _run_generator_with_overrides(runner, run_root, args.run_name, protocol, args)
    if args.dry_run:
        return 0
    for stage in legacy.PREP_STAGES:
        legacy.run_lammps_stage(run_dir, stage, protocol.lmp_command, dry_run=False)
    legacy.write_stage_runner(run_dir, protocol.lmp_command, legacy.PREP_STAGES)
    manifest = _write_generic_stage02_manifest(run_dir, protocol, args.campaign_slug)
    print(f"[ok] Stage-02 branchpoint: {manifest}")
    return 0


def branch(args: argparse.Namespace) -> int:
    protocol = _protocol_from_args(args)
    parent = Path(args.parent_run_dir).resolve()
    parent_data = parent / "02_after_equilibrate_nvt.data"
    parent_restart = parent / "02_after_equilibrate_nvt.restart"
    parent_metadata = parent / "metadata.json"
    if not parent_data.exists():
        raise FileNotFoundError(parent_data)
    if args.start_mode == START_RESTART_PRESERVE and not parent_restart.exists():
        raise FileNotFoundError(parent_restart)
    legacy.assert_parent_compatible(legacy.read_json(parent_metadata), protocol)

    run_root = Path(args.run_root)
    branch_dir = (run_root / args.run_name).resolve()
    runner = Path(args.runner_path).resolve()
    legacy.ensure_empty_new_dir(branch_dir)
    branch_dir.rmdir()
    legacy.run_generator(runner, run_root, args.run_name, protocol, args.dry_run)
    if args.dry_run:
        return 0

    shutil.copy2(parent_data, branch_dir / parent_data.name)
    if parent_restart.exists():
        shutil.copy2(parent_restart, branch_dir / parent_restart.name)
    shutil.copy2(parent_metadata, branch_dir / "parent_metadata.json")
    parent_manifest = parent / "stage02_branchpoint.json"
    if parent_manifest.exists():
        shutil.copy2(parent_manifest, branch_dir / "parent_stage02_branchpoint.json")
    legacy.remove_unused_preparation_files(branch_dir)

    generated = _apply_start_mode(branch_dir, args.start_mode)
    model = legacy.model_record_from_metadata(branch_dir)
    if args.start_mode == START_DATA_RECREATE:
        base, stage, resume = legacy.patch_stage03_for_checkpoints(branch_dir, protocol, model)
    else:
        base, stage, resume = _patch_restart_mode_for_checkpoints(branch_dir, protocol, model)
    sync_helper = legacy.write_checkpoint_sync_helper(branch_dir)
    legacy.write_stage_runner(branch_dir, protocol.lmp_command, [legacy.MDDMS_STAGE])

    metadata = legacy.read_json(branch_dir / "metadata.json")
    metadata["run_type"] = _kind(args.campaign_slug, "stage03_branch")
    metadata["stage03_start_mode"] = args.start_mode
    metadata["parent_run_dir"] = str(parent)
    metadata["checkpoint_sync_helper"] = sync_helper.name
    legacy.write_json(branch_dir / "metadata.json", metadata)

    branch_manifest = branch_dir / "branch_manifest.json"
    legacy.write_json(
        branch_manifest,
        {
            "schema_version": 2,
            "created_utc": legacy.utc_now(),
            "kind": _kind(args.campaign_slug, "stage03_branch"),
            "start_mode": args.start_mode,
            "parent_run_dir": str(parent),
            "parent_stage02": {
                "data": legacy.file_record(parent_data),
                "restart": legacy.file_record(parent_restart),
                "metadata": legacy.file_record(parent_metadata),
            },
            "protocol": asdict(protocol),
            "model": legacy.model_record_from_metadata(branch_dir),
        },
    )
    _write_generic_checkpoint_manifest(
        branch_dir,
        protocol,
        args.campaign_slug,
        args.start_mode,
        generated,
        base,
        stage,
        resume,
        branch_manifest,
    )

    print(f"[ok] Stage-03 branch: {branch_dir}")
    print(f"[ok] start mode: {args.start_mode}")
    if args.execute:
        legacy.run_lammps_stage(branch_dir, legacy.MDDMS_STAGE, protocol.lmp_command, dry_run=False)
    return 0


def resume(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    manifest = legacy.read_json(run_dir / "checkpoint_manifest.json")
    protocol = legacy.Protocol(**manifest["protocol"])
    restart = legacy.latest_checkpoint(run_dir, args.restart_file)
    legacy.run_lammps_resume(run_dir, restart, protocol.lmp_command, args.dry_run)
    return 0


def _add_protocol_args(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument("--mddms-period-ps", type=float, default=50.0)
    parser.add_argument("--mddms-cycles", type=int, default=6)
    parser.add_argument("--thermo-every-steps", type=int, default=None)
    parser.add_argument("--stress-every-steps", type=int, default=None)
    parser.add_argument("--dump-every-steps", type=int, default=1000)
    parser.add_argument("--checkpoint-every-steps", type=int, default=legacy.DEFAULT_CHECKPOINT_EVERY_STEPS)
    parser.add_argument("--stress-sign", type=float, default=-1.0)
    parser.add_argument("--campaign-slug", default="mddms")
    parser.add_argument(
        "--lmp-command",
        default="lmp -k on g 1 -sf kk -pk kokkos newton on neigh half",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generic reproducible MD-DMS preparation/branch runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare")
    _add_protocol_args(p_prepare)
    p_prepare.add_argument("--melt-ps", type=float, default=None)
    p_prepare.add_argument("--quench-rate-K-per-ps", type=float, default=None)
    p_prepare.add_argument("--relax-ps", type=float, default=None)
    p_prepare.add_argument("--equilibrate-ps", type=float, default=None)
    p_prepare.add_argument("--dry-run", action="store_true")
    p_prepare.set_defaults(function=prepare)

    p_branch = sub.add_parser("branch")
    _add_protocol_args(p_branch)
    p_branch.add_argument("--parent-run-dir", required=True)
    p_branch.add_argument("--start-mode", choices=START_MODES, default=START_DATA_RECREATE)
    p_branch.add_argument("--execute", action="store_true")
    p_branch.add_argument("--dry-run", action="store_true")
    p_branch.set_defaults(function=branch)

    p_resume = sub.add_parser("resume")
    p_resume.add_argument("--run-dir", required=True)
    p_resume.add_argument("--restart-file", default=None)
    p_resume.add_argument("--dry-run", action="store_true")
    p_resume.set_defaults(function=resume)
    return parser


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.command not in {"prepare", "branch"}:
        return
    if args.natoms <= 0 or args.timestep_ps <= 0 or args.mddms_period_ps <= 0:
        parser.error("natoms, timestep, and period must be positive")
    if not (0.0 <= args.cu_fraction <= 1.0):
        parser.error("--cu-fraction must be in [0, 1]")
    if args.mddms_cycles <= 0 or args.strain_amplitude <= 0:
        parser.error("cycles and strain amplitude must be positive")
    if args.checkpoint_every_steps <= 0 or args.dump_every_steps <= 0:
        parser.error("checkpoint and dump cadence must be positive")
    if args.checkpoint_every_steps % args.dump_every_steps != 0:
        parser.error("checkpoint cadence must be a multiple of dump cadence")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate(parser, args)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
