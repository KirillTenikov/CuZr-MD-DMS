#!/usr/bin/env python3
"""
CuZr-MD-DMS pilot runner.

This script is the first "working horse" for the Paper 2 MD-DMS workflow.

It can:
  1. generate a Cu-Zr initial LAMMPS data file,
  2. generate LAMMPS inputs for:
       00_prepare_melt_quench
       01_relax_npt
       02_equilibrate_nvt
       03_mddms_shear
  3. optionally execute the stages with LAMMPS,
  4. optionally fit the system stress response and extract G', G''.

Design choices:
  - No pyiron.
  - Python controls workflow.
  - LAMMPS input files are generated and saved for reproducibility.
  - MACE is used through LAMMPS ML-IAP unified interface.
  - EAM is also supported for baseline/sanity runs.

This first version is intended for technical pilots.
Do not treat the default "pilot" preset as final production science.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

AMU_TO_G = 1.66053906660e-24
CM3_TO_A3 = 1.0e24
CU_MASS = 63.546
ZR_MASS = 91.224


@dataclass(frozen=True)
class Preset:
    name: str
    melt_ps: float
    quench_rate_K_per_ps: float
    relax_ps: float
    equilibrate_ps: float
    mddms_period_ps: float
    mddms_cycles: int
    thermo_every_steps: int
    stress_every_steps: int
    dump_every_steps: int


PRESETS: dict[str, Preset] = {
    # Very small sanity run: checks that LAMMPS, potential, deformation,
    # and output mechanics work. Not physically meaningful.
    "tiny": Preset(
        name="tiny",
        melt_ps=1.0,
        quench_rate_K_per_ps=2700.0,  # 3000 -> 300 K in 1 ps
        relax_ps=1.0,
        equilibrate_ps=1.0,
        mddms_period_ps=1.0,
        mddms_cycles=2,
        thermo_every_steps=100,
        stress_every_steps=10,
        dump_every_steps=100,
    ),
    # Practical first cloud test.
    "pilot": Preset(
        name="pilot",
        melt_ps=20.0,
        quench_rate_K_per_ps=135.0,  # 3000 -> 300 K in 20 ps
        relax_ps=10.0,
        equilibrate_ps=10.0,
        mddms_period_ps=10.0,
        mddms_cycles=3,
        thermo_every_steps=500,
        stress_every_steps=20,
        dump_every_steps=1000,
    ),
    # Thesis Chapter 3-like baseline. Expensive with MACE.
    "ch3": Preset(
        name="ch3",
        melt_ps=1000.0,
        quench_rate_K_per_ps=1.0,    # 3000 -> 300 K in 2700 ps
        relax_ps=100.0,
        equilibrate_ps=0.0,
        mddms_period_ps=50.0,
        mddms_cycles=30,
        thermo_every_steps=1000,
        stress_every_steps=10,
        dump_every_steps=1000,
    ),
}


@dataclass
class RunConfig:
    run_name: str
    run_dir: str
    preset: str
    model_kind: str
    model_file: str
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
    dump_atomic: bool
    stress_sign: float
    lmp_command: str


def ps_to_steps(ps: float, timestep_ps: float) -> int:
    return max(0, int(round(ps / timestep_ps)))


def composition_counts(natoms: int, cu_fraction: float) -> tuple[int, int]:
    n_cu = int(round(natoms * cu_fraction))
    n_cu = min(max(n_cu, 0), natoms)
    n_zr = natoms - n_cu
    return n_cu, n_zr


def estimate_box_length_a(n_cu: int, n_zr: int, density_g_cm3: float) -> float:
    mass_g = (n_cu * CU_MASS + n_zr * ZR_MASS) * AMU_TO_G
    volume_cm3 = mass_g / density_g_cm3
    volume_a3 = volume_cm3 * CM3_TO_A3
    return volume_a3 ** (1.0 / 3.0)


def write_initial_data(
    path: Path,
    natoms: int,
    cu_fraction: float,
    density_g_cm3: float,
    seed: int,
) -> dict:
    """Write a simple lattice-like random-alloy starting structure.

    We intentionally avoid fully random positions because atom overlaps can
    make the first MD steps unstable. The structure is not meant to be physical;
    it will be melted and quenched.
    """
    rng = random.Random(seed)
    n_cu, n_zr = composition_counts(natoms, cu_fraction)
    L = estimate_box_length_a(n_cu, n_zr, density_g_cm3)

    ngrid = math.ceil(natoms ** (1.0 / 3.0))
    spacing = L / ngrid

    positions: list[tuple[float, float, float]] = []
    for ix in range(ngrid):
        for iy in range(ngrid):
            for iz in range(ngrid):
                if len(positions) >= natoms:
                    break
                # Small jitter avoids perfect symmetry but keeps distances safe.
                jitter = 0.12 * spacing
                x = (ix + 0.5) * spacing + rng.uniform(-jitter, jitter)
                y = (iy + 0.5) * spacing + rng.uniform(-jitter, jitter)
                z = (iz + 0.5) * spacing + rng.uniform(-jitter, jitter)
                positions.append((x % L, y % L, z % L))
            if len(positions) >= natoms:
                break
        if len(positions) >= natoms:
            break

    atom_types = [1] * n_cu + [2] * n_zr
    rng.shuffle(atom_types)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("Cu-Zr initial structure generated by run_mddms_pilot.py\n\n")
        f.write(f"{natoms} atoms\n")
        f.write("2 atom types\n\n")
        f.write(f"0.0 {L:.10f} xlo xhi\n")
        f.write(f"0.0 {L:.10f} ylo yhi\n")
        f.write(f"0.0 {L:.10f} zlo zhi\n\n")
        f.write("Masses\n\n")
        f.write(f"1 {CU_MASS:.8f} # Cu\n")
        f.write(f"2 {ZR_MASS:.8f} # Zr\n\n")
        f.write("Atoms # atomic\n\n")
        for atom_id, (atype, (x, y, z)) in enumerate(zip(atom_types, positions), start=1):
            f.write(f"{atom_id} {atype} {x:.10f} {y:.10f} {z:.10f}\n")

    return {
        "natoms": natoms,
        "n_cu": n_cu,
        "n_zr": n_zr,
        "cu_fraction_actual": n_cu / natoms,
        "zr_fraction_actual": n_zr / natoms,
        "density_g_cm3": density_g_cm3,
        "box_length_A": L,
        "grid": ngrid,
        "spacing_A": spacing,
    }


def potential_block(model_kind: Literal["mace", "eam"], model_file: Path) -> str:
    model_file = model_file.resolve()
    if model_kind == "mace":
        return f"""# MACE via LAMMPS ML-IAP unified interface
pair_style      mliap unified {model_file} 0
pair_coeff      * * Cu Zr
"""
    if model_kind == "eam":
        return f"""# EAM baseline
pair_style      eam/alloy
pair_coeff      * * {model_file} Cu Zr
"""
    raise ValueError(f"Unsupported model_kind: {model_kind}")


def common_header(data_file: str, model_kind: str, model_file: Path) -> str:
    return f"""units           metal
atom_style      atomic
boundary        p p p
newton          on

read_data       {data_file}

{potential_block(model_kind, model_file)}

neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes
"""


def write_prepare_input(run_dir: Path, cfg: RunConfig, preset: Preset) -> None:
    melt_steps = ps_to_steps(preset.melt_ps, cfg.timestep_ps)
    quench_ps = (cfg.temperature_high_K - cfg.temperature_low_K) / preset.quench_rate_K_per_ps
    quench_steps = ps_to_steps(quench_ps, cfg.timestep_ps)

    text = f"""{common_header("initial.data", cfg.model_kind, Path(cfg.model_file))}
timestep        {cfg.timestep_ps:.8f}

thermo          {preset.thermo_every_steps}
thermo_style    custom step time temp pe ke etotal press pxx pyy pzz pxy lx ly lz

# Gentle cleanup before the high-temperature melt.
minimize        1.0e-6 1.0e-8 1000 10000

velocity        all create {cfg.temperature_high_K:.6f} {cfg.seed + 101} mom yes rot yes dist gaussian

fix             melt all nvt temp {cfg.temperature_high_K:.6f} {cfg.temperature_high_K:.6f} {cfg.tdamp_ps:.6f}
run             {melt_steps}
unfix           melt

fix             quench all nvt temp {cfg.temperature_high_K:.6f} {cfg.temperature_low_K:.6f} {cfg.tdamp_ps:.6f}
run             {quench_steps}
unfix           quench

write_data      00_after_quench.data
write_restart   00_after_quench.restart
"""
    (run_dir / "00_prepare_melt_quench.in").write_text(text, encoding="utf-8")


def write_relax_input(run_dir: Path, cfg: RunConfig, preset: Preset) -> None:
    relax_steps = ps_to_steps(preset.relax_ps, cfg.timestep_ps)
    text = f"""{common_header("00_after_quench.data", cfg.model_kind, Path(cfg.model_file))}
timestep        {cfg.timestep_ps:.8f}

thermo          {preset.thermo_every_steps}
thermo_style    custom step time temp pe ke etotal press pxx pyy pzz pxy lx ly lz

velocity        all create {cfg.temperature_low_K:.6f} {cfg.seed + 202} mom yes rot yes dist gaussian

fix             relax all npt temp {cfg.temperature_low_K:.6f} {cfg.temperature_low_K:.6f} {cfg.tdamp_ps:.6f} iso {cfg.pressure_bar:.6f} {cfg.pressure_bar:.6f} {cfg.pdamp_ps:.6f}
run             {relax_steps}
unfix           relax

write_data      01_after_relax_npt.data
write_restart   01_after_relax_npt.restart
"""
    (run_dir / "01_relax_npt.in").write_text(text, encoding="utf-8")


def write_equilibrate_input(run_dir: Path, cfg: RunConfig, preset: Preset) -> None:
    eq_steps = ps_to_steps(preset.equilibrate_ps, cfg.timestep_ps)
    text = f"""{common_header("01_after_relax_npt.data", cfg.model_kind, Path(cfg.model_file))}
timestep        {cfg.timestep_ps:.8f}

thermo          {preset.thermo_every_steps}
thermo_style    custom step time temp pe ke etotal press pxx pyy pzz pxy lx ly lz

velocity        all create {cfg.temperature_low_K:.6f} {cfg.seed + 303} mom yes rot yes dist gaussian

fix             eq all nvt temp {cfg.temperature_low_K:.6f} {cfg.temperature_low_K:.6f} {cfg.tdamp_ps:.6f}
run             {eq_steps}
unfix           eq

write_data      02_after_equilibrate_nvt.data
write_restart   02_after_equilibrate_nvt.restart
"""
    (run_dir / "02_equilibrate_nvt.in").write_text(text, encoding="utf-8")


def write_mddms_input(run_dir: Path, cfg: RunConfig, preset: Preset) -> None:
    mddms_ps = preset.mddms_period_ps * preset.mddms_cycles
    mddms_steps = ps_to_steps(mddms_ps, cfg.timestep_ps)

    atomic_dump = ""
    if cfg.dump_atomic:
        atomic_dump = f"""
# Atomic-level stress output.
# c_atomstress[4] is the xy component of LAMMPS stress/atom.
# c_atomvol[1] is Voronoi volume. Later analysis may use -c_atomstress[4]/c_atomvol[1].
compute         atomstress all stress/atom NULL
compute         atomvol all voronoi/atom
dump            atomdump all custom {preset.dump_every_steps} atom_stress.lammpstrj id type x y z c_atomstress[4] c_atomvol[1]
dump_modify     atomdump sort id
"""

    text = f"""{common_header("02_after_equilibrate_nvt.data", cfg.model_kind, Path(cfg.model_file))}
timestep        {cfg.timestep_ps:.8f}

# The box must be triclinic before applying xy shear.
change_box      all triclinic

reset_timestep  0
velocity        all create {cfg.temperature_low_K:.6f} {cfg.seed + 404} mom yes rot yes dist gaussian

thermo          {preset.thermo_every_steps}
thermo_style    custom step time temp pe ke etotal press pxx pyy pzz pxy lx ly lz xy

# Sinusoidal shear strain:
# gamma(t) = gamma0 * sin(2*pi*t/T)
# LAMMPS xy tilt has units of length, so xy(t) = gamma(t) * Ly.
variable        gamma0 equal {cfg.strain_amplitude:.12f}
variable        period equal {preset.mddms_period_ps:.12f}
variable        omega equal 2.0*{math.pi:.16f}/v_period
variable        gamma equal v_gamma0*sin(v_omega*time)
variable        gammadot equal v_gamma0*v_omega*cos(v_omega*time)
variable        xy_target equal v_gamma*ly
variable        xy_rate equal v_gammadot*ly

# Thermostat during small-amplitude oscillatory shear.
# This first implementation uses affine remapping of coordinates.
fix             thermostat all nvt temp {cfg.temperature_low_K:.6f} {cfg.temperature_low_K:.6f} {cfg.tdamp_ps:.6f}
fix             deform all deform 1 xy variable v_xy_target v_xy_rate remap x

# System-level time series.
variable        time_ps equal time
variable        pxy_bar equal pxy
variable        temp_K equal temp
variable        pe_eV equal pe
variable        ke_eV equal ke
variable        press_bar equal press
variable        xy_A equal xy
variable        ly_A equal ly

fix             ts all ave/time {preset.stress_every_steps} 1 {preset.stress_every_steps} v_time_ps v_gamma v_pxy_bar v_temp_K v_pe_eV v_ke_eV v_press_bar v_xy_A v_ly_A file stress_timeseries.dat
{atomic_dump}
run             {mddms_steps}

unfix           ts
unfix           deform
unfix           thermostat

write_data      03_after_mddms.data
write_restart   03_after_mddms.restart
"""
    (run_dir / "03_mddms_shear.in").write_text(text, encoding="utf-8")


def write_run_shell_script(run_dir: Path, cfg: RunConfig) -> None:
    stage_files = [
        "00_prepare_melt_quench.in",
        "01_relax_npt.in",
        "02_equilibrate_nvt.in",
        "03_mddms_shear.in",
    ]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "source /opt/venv/bin/activate || true",
        "if [ -f /workspace/cuzr_mddms_runtime.env ]; then",
        "  source /workspace/cuzr_mddms_runtime.env",
        "fi",
        "",
        f"LMP_CMD={shlex.quote(cfg.lmp_command)}",
        "",
        "run_stage() {",
        "  local input=\"$1\"",
        "  local log=\"${input%.in}.log\"",
        "  echo \"[run] $input -> $log\"",
        "  ${LMP_CMD} -log \"$log\" -in \"$input\"",
        "}",
        "",
    ]
    for stage in stage_files:
        lines.append(f"run_stage {shlex.quote(stage)}")
    lines.append("")
    (run_dir / "run_lammps.sh").write_text("\n".join(lines), encoding="utf-8")
    os.chmod(run_dir / "run_lammps.sh", 0o755)


def default_lmp_command(model_kind: str) -> str:
    if model_kind == "mace":
        return "lmp -k on g 1 -sf kk -pk kokkos newton on neigh half"
    return "lmp"


def run_lammps_stage(run_dir: Path, input_file: str, lmp_command: str) -> None:
    input_path = run_dir / input_file
    log_name = input_file.replace(".in", ".log")
    cmd = shlex.split(lmp_command) + ["-log", log_name, "-in", input_path.name]
    print("[execute]", " ".join(shlex.quote(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=run_dir, check=True)


def execute_stages(run_dir: Path, cfg: RunConfig) -> None:
    stages = [
        "00_prepare_melt_quench.in",
        "01_relax_npt.in",
        "02_equilibrate_nvt.in",
        "03_mddms_shear.in",
    ]
    for stage in stages:
        run_lammps_stage(run_dir, stage, cfg.lmp_command)


def parse_fix_ave_time(path: Path) -> tuple[list[str], list[list[float]]]:
    """Parse LAMMPS fix ave/time file.

    Expected data columns:
      timestep time_ps gamma pxy_bar temp_K pe_eV ke_eV press_bar xy_A ly_A
    """
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue

    columns = [
        "timestep",
        "time_ps",
        "gamma",
        "pxy_bar",
        "temp_K",
        "pe_eV",
        "ke_eV",
        "press_bar",
        "xy_A",
        "ly_A",
    ]
    return columns, rows


def fit_mddms_response(
    stress_file: Path,
    period_ps: float,
    strain_amplitude: float,
    stress_sign: float,
) -> dict:
    import numpy as np

    columns, rows = parse_fix_ave_time(stress_file)
    if not rows:
        raise RuntimeError(f"No numeric rows found in {stress_file}")

    arr = np.asarray(rows, dtype=float)
    time_ps = arr[:, 1]
    gamma = arr[:, 2]
    pxy_bar = arr[:, 3]

    # Convert bar to GPa. Sign is configurable because LAMMPS pressure tensor
    # sign conventions can be opposite to engineering stress conventions.
    stress_gpa = stress_sign * pxy_bar * 1.0e-4

    omega = 2.0 * math.pi / period_ps
    sin_t = np.sin(omega * time_ps)
    cos_t = np.cos(omega * time_ps)
    X = np.column_stack([sin_t, cos_t, np.ones_like(time_ps)])

    coeff, *_ = np.linalg.lstsq(X, stress_gpa, rcond=None)
    a_sin, b_cos, offset = coeff

    stress_amp_gpa = float(math.sqrt(a_sin * a_sin + b_cos * b_cos))
    phase_rad = float(math.atan2(b_cos, a_sin))

    # If gamma(t)=gamma0*sin(wt), then stress=sigma0*sin(wt+delta).
    # G' = sigma0/gamma0 cos(delta), G'' = sigma0/gamma0 sin(delta).
    g_storage_gpa = float((stress_amp_gpa / strain_amplitude) * math.cos(phase_rad))
    g_loss_gpa = float((stress_amp_gpa / strain_amplitude) * math.sin(phase_rad))

    fitted = X @ coeff
    residual = stress_gpa - fitted
    rmse_gpa = float(np.sqrt(np.mean(residual * residual)))

    return {
        "stress_file": str(stress_file),
        "n_points": int(len(time_ps)),
        "period_ps": float(period_ps),
        "strain_amplitude": float(strain_amplitude),
        "stress_sign": float(stress_sign),
        "fit_model": "stress_GPa = a*sin(omega*t) + b*cos(omega*t) + c",
        "a_sin_GPa": float(a_sin),
        "b_cos_GPa": float(b_cos),
        "offset_GPa": float(offset),
        "stress_amplitude_GPa": stress_amp_gpa,
        "phase_rad": phase_rad,
        "phase_deg": float(phase_rad * 180.0 / math.pi),
        "G_storage_GPa": g_storage_gpa,
        "G_loss_GPa": g_loss_gpa,
        "rmse_GPa": rmse_gpa,
        "mean_gamma": float(np.mean(gamma)),
        "max_abs_gamma": float(np.max(np.abs(gamma))),
        "mean_stress_GPa": float(np.mean(stress_gpa)),
    }


def generate_run(cfg: RunConfig) -> None:
    if cfg.preset not in PRESETS:
        raise ValueError(f"Unknown preset {cfg.preset!r}. Available: {sorted(PRESETS)}")

    preset = PRESETS[cfg.preset]
    run_dir = Path(cfg.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    model_file = Path(cfg.model_file)
    if not model_file.exists():
        print(f"[warning] Model file does not exist yet: {model_file}", file=sys.stderr)
        print("[warning] This is okay if you are generating files locally before copying models on cloud.", file=sys.stderr)

    structure_meta = write_initial_data(
        run_dir / "initial.data",
        cfg.natoms,
        cfg.cu_fraction,
        cfg.density_g_cm3,
        cfg.seed,
    )

    write_prepare_input(run_dir, cfg, preset)
    write_relax_input(run_dir, cfg, preset)
    write_equilibrate_input(run_dir, cfg, preset)
    write_mddms_input(run_dir, cfg, preset)
    write_run_shell_script(run_dir, cfg)

    quench_ps = (cfg.temperature_high_K - cfg.temperature_low_K) / preset.quench_rate_K_per_ps

    metadata = {
        "run_config": asdict(cfg),
        "preset": asdict(preset),
        "derived": {
            "melt_steps": ps_to_steps(preset.melt_ps, cfg.timestep_ps),
            "quench_ps": quench_ps,
            "quench_steps": ps_to_steps(quench_ps, cfg.timestep_ps),
            "relax_steps": ps_to_steps(preset.relax_ps, cfg.timestep_ps),
            "equilibrate_steps": ps_to_steps(preset.equilibrate_ps, cfg.timestep_ps),
            "mddms_total_ps": preset.mddms_period_ps * preset.mddms_cycles,
            "mddms_steps": ps_to_steps(preset.mddms_period_ps * preset.mddms_cycles, cfg.timestep_ps),
        },
        "initial_structure": structure_meta,
        "notes": [
            "This is a generated technical-pilot workflow.",
            "LAMMPS input files are saved for reproducibility.",
            "For MACE, this script assumes a converted ML-IAP model file: *-mliap_lammps.pt.",
            "For production, validate thermostat/deformation settings and output cadence.",
        ],
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[ok] Generated run directory: {run_dir}")
    print("[ok] Main files:")
    for name in [
        "initial.data",
        "00_prepare_melt_quench.in",
        "01_relax_npt.in",
        "02_equilibrate_nvt.in",
        "03_mddms_shear.in",
        "run_lammps.sh",
        "metadata.json",
    ]:
        print(f"  - {run_dir / name}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate and optionally run a Cu-Zr MD-DMS pilot workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--run-name", default="pilot_001")
    p.add_argument("--run-root", default="runs")
    p.add_argument("--preset", choices=sorted(PRESETS), default="pilot")

    p.add_argument("--model-kind", choices=["mace", "eam"], default="mace")
    p.add_argument(
        "--model-file",
        default="models/mace/mace_C.model-mliap_lammps.pt",
        help="MACE ML-IAP .pt file or EAM .alloy file.",
    )

    p.add_argument("--natoms", type=int, default=4000)
    p.add_argument("--cu-fraction", type=float, default=0.64)
    p.add_argument("--density-g-cm3", type=float, default=7.20)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--timestep-ps", type=float, default=0.001, help="LAMMPS metal timestep. 0.001 ps = 1 fs.")
    p.add_argument("--temperature-high-K", type=float, default=3000.0)
    p.add_argument("--temperature-low-K", type=float, default=300.0)
    p.add_argument("--pressure-bar", type=float, default=0.0)

    p.add_argument("--strain-amplitude", type=float, default=0.01)
    p.add_argument("--tdamp-ps", type=float, default=0.1)
    p.add_argument("--pdamp-ps", type=float, default=1.0)

    p.add_argument(
        "--dump-atomic",
        action="store_true",
        help="Dump per-atom xy stress and Voronoi volume during MD-DMS. Can create large files.",
    )
    p.add_argument(
        "--stress-sign",
        type=float,
        default=-1.0,
        help="Multiplier for pxy when fitting stress. Use -1 or +1 depending on convention.",
    )
    p.add_argument(
        "--lmp-command",
        default=None,
        help="LAMMPS command without -in. If omitted, uses optimized default for MACE and simple lmp for EAM.",
    )

    p.add_argument("--execute", action="store_true", help="Actually run LAMMPS stages after generating files.")
    p.add_argument("--analyze", action="store_true", help="Fit stress_timeseries.dat after execution or existing run.")

    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    lmp_command = args.lmp_command or default_lmp_command(args.model_kind)
    run_dir = str(Path(args.run_root) / args.run_name)

    cfg = RunConfig(
        run_name=args.run_name,
        run_dir=run_dir,
        preset=args.preset,
        model_kind=args.model_kind,
        model_file=args.model_file,
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
        dump_atomic=args.dump_atomic,
        stress_sign=args.stress_sign,
        lmp_command=lmp_command,
    )

    generate_run(cfg)

    run_dir_path = Path(run_dir).resolve()

    if args.execute:
        execute_stages(run_dir_path, cfg)

    if args.analyze:
        preset = PRESETS[cfg.preset]
        stress_file = run_dir_path / "stress_timeseries.dat"
        if not stress_file.exists():
            raise FileNotFoundError(f"Cannot analyze: missing {stress_file}")
        fit = fit_mddms_response(
            stress_file=stress_file,
            period_ps=preset.mddms_period_ps,
            strain_amplitude=cfg.strain_amplitude,
            stress_sign=cfg.stress_sign,
        )
        fit_path = run_dir_path / "mddms_fit.json"
        fit_path.write_text(json.dumps(fit, indent=2), encoding="utf-8")
        print(f"[ok] Wrote fit: {fit_path}")
        print(json.dumps(fit, indent=2))

    print()
    print("Next commands:")
    print(f"  cd {run_dir_path}")
    print("  bash run_lammps.sh")
    print()
    print("Or generate + execute directly:")
    print("  python scripts/run/run_mddms_pilot.py --execute --analyze ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
