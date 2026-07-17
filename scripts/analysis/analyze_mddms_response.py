#!/usr/bin/env python3
"""Standalone post-processing for Cu-Zr MD-DMS stress time series.

Historical Paper-2 convention
-----------------------------
The accepted response model is

    sigma_xy(t) = a sin(omega t) + b cos(omega t) + c,

with stress converted from LAMMPS metal-unit pressure (bar) as

    sigma_xy [GPa] = stress_sign * pxy [bar] * 1e-4.

For strain gamma(t) = gamma0 sin(omega t):

    G'  = a / gamma0
    G'' = b / gamma0
    tan(delta) = G'' / G'
    delta = atan2(G'', G').

The primary fit uses every stored point from all cycles, including the first
cycle, to preserve comparability with the submitted Paper-2 manuscript.
Cycle-resolved fits use half-open intervals [start, end), except that the last
cycle includes its final endpoint. This exactly reproduces the historical
Paper-2 cycle table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

COLUMN_NAMES = (
    "step",
    "time_ps",
    "gamma",
    "pxy_bar",
    "temperature_K",
    "pe_eV",
    "ke_eV",
    "pressure_bar",
    "xy_A",
    "ly_A",
)
FIT_MODEL = "stress_GPa = a*sin(omega*t) + b*cos(omega*t) + c"
SCHEMA_VERSION = 1


class AnalysisError(RuntimeError):
    """Raised when inputs are incomplete or internally inconsistent."""


@dataclass(frozen=True)
class Protocol:
    period_ps: float
    cycles: int
    strain_amplitude: float
    stress_sign: float
    source: dict[str, str]


@dataclass(frozen=True)
class StressSeries:
    step: np.ndarray
    time_ps: np.ndarray
    gamma: np.ndarray
    pxy_bar: np.ndarray
    temperature_K: np.ndarray
    pe_eV: np.ndarray
    ke_eV: np.ndarray
    pressure_bar: np.ndarray
    xy_A: np.ndarray
    ly_A: np.ndarray
    sigma_xy_GPa: np.ndarray

    @property
    def n_points(self) -> int:
        return int(self.time_ps.size)


@dataclass(frozen=True)
class FitResult:
    a_sin_GPa: float
    b_cos_GPa: float
    offset_GPa: float
    stress_amplitude_GPa: float
    phase_rad: float
    phase_deg: float
    tan_delta: float
    G_storage_GPa: float
    G_loss_GPa: float
    rmse_GPa: float
    n_points: int
    time_start_ps: float
    time_end_ps: float


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"Expected a JSON object in {path}")
    return value


def nested_get(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def infer_run_dir(stress_file: Path, run_dir: Path | None) -> Path:
    if run_dir is not None:
        resolved = run_dir.resolve()
        if not resolved.is_dir():
            raise AnalysisError(f"Run directory does not exist: {resolved}")
        return resolved
    return stress_file.resolve().parent


def discover_stress_file(run_dir: Path) -> Path:
    candidates = (
        run_dir / "stress_timeseries_stitched_final.dat",
        run_dir / "stress_timeseries.dat",
    )
    found = first_existing(candidates)
    if found is None:
        raise AnalysisError(
            f"No accepted stress series found in {run_dir}; expected one of: "
            + ", ".join(path.name for path in candidates)
        )
    return found


def parse_lammps_input(path: Path) -> dict[str, float | int]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    result: dict[str, float | int] = {}
    patterns: dict[str, str] = {
        "period_ps": r"^\s*variable\s+period\s+equal\s+([0-9.eE+\-]+)",
        "strain_amplitude": r"^\s*variable\s+gamma0\s+equal\s+([0-9.eE+\-]+)",
        "run_steps": r"^\s*run\s+(\d+)\b",
        "timestep_ps": r"^\s*timestep\s+([0-9.eE+\-]+)",
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        if not matches:
            continue
        raw = matches[-1]
        result[key] = int(raw) if key == "run_steps" else float(raw)
    return result


def resolve_protocol(
    run_dir: Path,
    *,
    period_ps: float | None,
    cycles: int | None,
    strain_amplitude: float | None,
    stress_sign: float | None,
    metadata_path: Path | None = None,
    input_path: Path | None = None,
) -> tuple[Protocol, dict[str, Any] | None, Path | None, Path | None]:
    metadata_file = metadata_path or first_existing((run_dir / "metadata.json",))
    input_file = input_path or first_existing(
        (
            run_dir / "03_mddms_shear.in",
            run_dir / "03_mddms_shear_retry_traj_only.in",
        )
    )

    metadata: dict[str, Any] | None = read_json(metadata_file) if metadata_file else None
    input_values = parse_lammps_input(input_file) if input_file else {}
    source: dict[str, str] = {}

    def choose(
        name: str,
        explicit: Any,
        metadata_candidates: Sequence[tuple[str, ...]],
        input_key: str | None,
        default: Any = None,
    ) -> Any:
        if explicit is not None:
            source[name] = "command_line"
            return explicit
        if metadata is not None:
            for key_path in metadata_candidates:
                value = nested_get(metadata, *key_path)
                if value is not None:
                    source[name] = "metadata.json:" + ".".join(key_path)
                    return value
        if input_key is not None and input_key in input_values:
            source[name] = f"{input_file.name}:{input_key}" if input_file else input_key
            return input_values[input_key]
        if default is not None:
            source[name] = "default"
            return default
        raise AnalysisError(
            f"Could not determine {name}. Provide it explicitly or supply usable metadata/input files."
        )

    resolved_period = float(
        choose(
            "period_ps",
            period_ps,
            (("preset", "mddms_period_ps"), ("run_config", "mddms_period_ps")),
            "period_ps",
        )
    )
    resolved_gamma0 = float(
        choose(
            "strain_amplitude",
            strain_amplitude,
            (("run_config", "strain_amplitude"),),
            "strain_amplitude",
        )
    )
    resolved_sign = float(
        choose(
            "stress_sign",
            stress_sign,
            (("run_config", "stress_sign"),),
            None,
            -1.0,
        )
    )

    if cycles is not None:
        resolved_cycles = int(cycles)
        source["cycles"] = "command_line"
    else:
        cycle_value = None
        if metadata is not None:
            for key_path in (("preset", "mddms_cycles"), ("run_config", "mddms_cycles")):
                candidate = nested_get(metadata, *key_path)
                if candidate is not None:
                    cycle_value = int(candidate)
                    source["cycles"] = "metadata.json:" + ".".join(key_path)
                    break
        if cycle_value is None and "run_steps" in input_values:
            timestep = input_values.get("timestep_ps")
            if timestep is None and metadata is not None:
                timestep = nested_get(metadata, "run_config", "timestep_ps")
            if timestep is not None:
                calculated = float(input_values["run_steps"]) * float(timestep) / resolved_period
                rounded = int(round(calculated))
                if not math.isclose(calculated, rounded, rel_tol=0.0, abs_tol=1e-8):
                    raise AnalysisError(
                        f"Input run length implies a non-integer number of cycles: {calculated}"
                    )
                cycle_value = rounded
                source["cycles"] = f"{input_file.name}:run_steps*timestep/period"
        if cycle_value is None:
            raise AnalysisError("Could not determine number of MD-DMS cycles")
        resolved_cycles = cycle_value

    if not (resolved_period > 0.0 and math.isfinite(resolved_period)):
        raise AnalysisError(f"period_ps must be finite and positive, got {resolved_period}")
    if resolved_cycles < 1:
        raise AnalysisError(f"cycles must be >= 1, got {resolved_cycles}")
    if not (resolved_gamma0 > 0.0 and math.isfinite(resolved_gamma0)):
        raise AnalysisError(
            f"strain_amplitude must be finite and positive, got {resolved_gamma0}"
        )
    if not math.isfinite(resolved_sign) or resolved_sign == 0.0:
        raise AnalysisError(f"stress_sign must be finite and nonzero, got {resolved_sign}")

    return (
        Protocol(
            period_ps=resolved_period,
            cycles=resolved_cycles,
            strain_amplitude=resolved_gamma0,
            stress_sign=resolved_sign,
            source=source,
        ),
        metadata,
        metadata_file,
        input_file,
    )


def _load_whitespace(path: Path) -> np.ndarray:
    try:
        array = np.loadtxt(path, comments="#", dtype=float)
    except (OSError, ValueError) as exc:
        raise AnalysisError(f"Cannot parse whitespace stress file {path}: {exc}") from exc
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[1] < len(COLUMN_NAMES):
        raise AnalysisError(
            f"Expected at least {len(COLUMN_NAMES)} columns in {path}, got {array.shape[1]}"
        )
    return array[:, : len(COLUMN_NAMES)]


def _load_csv(path: Path) -> np.ndarray:
    aliases = {
        "step": ("step",),
        "time_ps": ("time_ps", "time"),
        "gamma": ("gamma", "strain"),
        "pxy_bar": ("pxy_bar", "pxy"),
        "temperature_K": ("temperature_K", "temp", "temperature"),
        "pe_eV": ("pe_eV", "pe"),
        "ke_eV": ("ke_eV", "ke"),
        "pressure_bar": ("pressure_bar", "press", "pressure"),
        "xy_A": ("xy_A", "xy"),
        "ly_A": ("ly_A", "ly"),
    }
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AnalysisError(f"CSV file has no header: {path}")
        fieldnames = set(reader.fieldnames)
        selected: dict[str, str] = {}
        for canonical, names in aliases.items():
            match = next((name for name in names if name in fieldnames), None)
            if match is None:
                raise AnalysisError(f"CSV file {path} lacks required column {canonical}")
            selected[canonical] = match
        for row_number, row in enumerate(reader, start=2):
            try:
                rows.append([float(row[selected[name]]) for name in COLUMN_NAMES])
            except (TypeError, ValueError) as exc:
                raise AnalysisError(f"Invalid numeric value in {path}, row {row_number}") from exc
    if not rows:
        raise AnalysisError(f"CSV file contains no data rows: {path}")
    return np.asarray(rows, dtype=float)


def load_stress_series(path: Path, stress_sign: float) -> StressSeries:
    path = path.resolve()
    if not path.is_file():
        raise AnalysisError(f"Stress file does not exist: {path}")

    first_data_line = ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                first_data_line = stripped
                break
    if not first_data_line:
        raise AnalysisError(f"Stress file is empty: {path}")

    is_csv = "," in first_data_line and any(char.isalpha() for char in first_data_line)
    array = _load_csv(path) if is_csv else _load_whitespace(path)
    columns = {name: array[:, index].copy() for index, name in enumerate(COLUMN_NAMES)}
    sigma = float(stress_sign) * columns["pxy_bar"] * 1.0e-4
    return StressSeries(**columns, sigma_xy_GPa=sigma)


def validate_series(series: StressSeries, protocol: Protocol) -> dict[str, Any]:
    arrays = [getattr(series, name) for name in COLUMN_NAMES] + [series.sigma_xy_GPa]
    if series.n_points < 4:
        raise AnalysisError("At least four samples are required for a three-parameter fit")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise AnalysisError("Stress series contains NaN or infinite values")

    step_differences = np.diff(series.step)
    time_differences = np.diff(series.time_ps)
    if np.any(step_differences <= 0):
        raise AnalysisError("Step column must be strictly increasing; duplicates/gaps need repair first")
    if np.any(time_differences <= 0):
        raise AnalysisError("time_ps column must be strictly increasing")

    expected_duration = protocol.period_ps * protocol.cycles
    observed_duration = float(series.time_ps[-1] - series.time_ps[0])
    median_dt = float(np.median(time_differences))
    duration_tolerance = max(1.0e-8, 1.1 * median_dt)
    if not math.isclose(observed_duration, expected_duration, rel_tol=0.0, abs_tol=duration_tolerance):
        raise AnalysisError(
            "Stress-series duration is inconsistent with protocol: "
            f"observed {observed_duration:.12g} ps, expected {expected_duration:.12g} ps"
        )

    max_abs_gamma = float(np.max(np.abs(series.gamma)))
    if not math.isclose(
        max_abs_gamma,
        protocol.strain_amplitude,
        rel_tol=2.0e-3,
        abs_tol=max(1.0e-10, protocol.strain_amplitude * 2.0e-3),
    ):
        raise AnalysisError(
            f"Maximum |gamma|={max_abs_gamma:.12g} does not match gamma0="
            f"{protocol.strain_amplitude:.12g}"
        )

    step_stride = float(np.median(step_differences))
    irregular_steps = int(np.count_nonzero(~np.isclose(step_differences, step_stride)))
    irregular_times = int(np.count_nonzero(~np.isclose(time_differences, median_dt)))

    return {
        "n_points": series.n_points,
        "first_step": float(series.step[0]),
        "last_step": float(series.step[-1]),
        "step_stride_median": step_stride,
        "irregular_step_intervals": irregular_steps,
        "time_start_ps": float(series.time_ps[0]),
        "time_end_ps": float(series.time_ps[-1]),
        "duration_ps": observed_duration,
        "time_stride_ps_median": median_dt,
        "irregular_time_intervals": irregular_times,
        "mean_gamma": float(np.mean(series.gamma)),
        "max_abs_gamma": max_abs_gamma,
        "mean_temperature_K": float(np.mean(series.temperature_K)),
        "mean_pressure_bar": float(np.mean(series.pressure_bar)),
    }


def fit_response(
    time_ps: np.ndarray,
    sigma_xy_GPa: np.ndarray,
    *,
    period_ps: float,
    strain_amplitude: float,
) -> tuple[FitResult, np.ndarray]:
    if time_ps.size != sigma_xy_GPa.size:
        raise AnalysisError("time and stress arrays have different lengths")
    if time_ps.size < 4:
        raise AnalysisError("A fit segment must contain at least four points")

    omega = 2.0 * math.pi / period_ps
    design = np.column_stack(
        (
            np.sin(omega * time_ps),
            np.cos(omega * time_ps),
            np.ones_like(time_ps),
        )
    )
    coefficients, _, rank, _ = np.linalg.lstsq(design, sigma_xy_GPa, rcond=None)
    if rank < 3:
        raise AnalysisError("Harmonic design matrix is rank deficient")
    a_sin, b_cos, offset = (float(value) for value in coefficients)
    fitted = design @ coefficients
    residual = sigma_xy_GPa - fitted
    rmse = float(np.sqrt(np.mean(residual * residual)))
    g_storage = a_sin / strain_amplitude
    g_loss = b_cos / strain_amplitude
    tan_delta = g_loss / g_storage if g_storage != 0.0 else math.nan
    phase_rad = math.atan2(b_cos, a_sin)

    return (
        FitResult(
            a_sin_GPa=a_sin,
            b_cos_GPa=b_cos,
            offset_GPa=offset,
            stress_amplitude_GPa=math.hypot(a_sin, b_cos),
            phase_rad=phase_rad,
            phase_deg=math.degrees(phase_rad),
            tan_delta=tan_delta,
            G_storage_GPa=g_storage,
            G_loss_GPa=g_loss,
            rmse_GPa=rmse,
            n_points=int(time_ps.size),
            time_start_ps=float(time_ps[0]),
            time_end_ps=float(time_ps[-1]),
        ),
        fitted,
    )


def cycle_masks(time_ps: np.ndarray, protocol: Protocol) -> list[np.ndarray]:
    t0 = float(time_ps[0])
    epsilon = max(1.0e-10, protocol.period_ps * 1.0e-12)
    masks: list[np.ndarray] = []
    for index in range(protocol.cycles):
        start = t0 + index * protocol.period_ps
        end = t0 + (index + 1) * protocol.period_ps
        if index < protocol.cycles - 1:
            mask = (time_ps >= start - epsilon) & (time_ps < end - epsilon)
        else:
            mask = (time_ps >= start - epsilon) & (time_ps <= end + epsilon)
        if int(np.count_nonzero(mask)) < 4:
            raise AnalysisError(f"Cycle {index + 1} contains fewer than four data points")
        masks.append(mask)
    return masks


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))  # pragma: no cover - compatibility with older NumPy


def fit_cycles(
    series: StressSeries, protocol: Protocol
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cycle_results: list[dict[str, Any]] = []
    loop_results: list[dict[str, Any]] = []
    for index, mask in enumerate(cycle_masks(series.time_ps, protocol), start=1):
        fit, _ = fit_response(
            series.time_ps[mask],
            series.sigma_xy_GPa[mask],
            period_ps=protocol.period_ps,
            strain_amplitude=protocol.strain_amplitude,
        )
        row = asdict(fit)
        row.update(
            {
                "cycle": index,
                "t_start_ps": float(series.time_ps[0] + (index - 1) * protocol.period_ps),
                "t_end_ps": float(series.time_ps[0] + index * protocol.period_ps),
            }
        )
        cycle_results.append(row)

        numeric_area = trapezoid(series.sigma_xy_GPa[mask], series.gamma[mask])
        expected_area = math.pi * protocol.strain_amplitude**2 * fit.G_loss_GPa
        loop_results.append(
            {
                "cycle": index,
                "loop_area_GPa": numeric_area,
                "abs_loop_area_GPa": abs(numeric_area),
                "fit_expected_area_GPa": expected_area,
                "n_points": int(np.count_nonzero(mask)),
            }
        )
    return cycle_results, loop_results


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise AnalysisError(f"Refusing to write an empty CSV: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compare_fit(stored_path: Path, current: FitResult) -> dict[str, Any]:
    stored = read_json(stored_path)
    mapping = {
        "a_sin_GPa": current.a_sin_GPa,
        "b_cos_GPa": current.b_cos_GPa,
        "offset_GPa": current.offset_GPa,
        "stress_amplitude_GPa": current.stress_amplitude_GPa,
        "phase_rad": current.phase_rad,
        "phase_deg": current.phase_deg,
        "tan_delta": current.tan_delta,
        "G_storage_GPa": current.G_storage_GPa,
        "G_loss_GPa": current.G_loss_GPa,
        "rmse_GPa": current.rmse_GPa,
    }
    deltas: dict[str, Any] = {"stored_fit_file": str(stored_path.resolve())}
    max_abs_delta = 0.0
    for key, observed in mapping.items():
        expected = stored.get(key)
        if expected is None:
            continue
        delta = observed - float(expected)
        deltas[f"stored_{key}"] = float(expected)
        deltas[f"recomputed_{key}"] = observed
        deltas[f"delta_{key}"] = delta
        max_abs_delta = max(max_abs_delta, abs(delta))
    deltas["max_abs_numeric_delta"] = max_abs_delta
    return deltas


def plot_results(
    output_dir: Path,
    series: StressSeries,
    fitted_stress: np.ndarray,
    cycle_rows: Sequence[Mapping[str, Any]],
    loop_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    figures: list[str] = []

    def save(name: str) -> None:
        path = output_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        figures.append(name)

    plt.figure(figsize=(8.0, 4.8))
    plt.plot(series.time_ps, series.sigma_xy_GPa, linewidth=0.7, label="MD stress")
    plt.plot(series.time_ps, fitted_stress, linewidth=1.2, label="harmonic fit")
    plt.xlabel("time (ps)")
    plt.ylabel(r"$\sigma_{xy}$ (GPa)")
    plt.legend()
    save("stress_vs_time_with_fit.png")

    plt.figure(figsize=(5.8, 5.2))
    plt.plot(series.gamma, series.sigma_xy_GPa, linewidth=0.7)
    plt.xlabel(r"shear strain $\gamma$")
    plt.ylabel(r"$\sigma_{xy}$ (GPa)")
    save("hysteresis_all_cycles.png")

    cycles = [int(row["cycle"]) for row in cycle_rows]
    for key, ylabel, filename in (
        ("G_storage_GPa", r"$G'$ (GPa)", "cycle_G_storage.png"),
        ("G_loss_GPa", r"$G''$ (GPa)", "cycle_G_loss.png"),
        ("tan_delta", r"$\tan\delta$", "cycle_tan_delta.png"),
        ("phase_deg", r"phase $\delta$ (deg)", "cycle_phase.png"),
    ):
        plt.figure(figsize=(6.2, 4.4))
        plt.plot(cycles, [float(row[key]) for row in cycle_rows], marker="o")
        plt.xlabel("cycle")
        plt.ylabel(ylabel)
        plt.xticks(cycles)
        save(filename)

    plt.figure(figsize=(6.2, 4.4))
    plt.plot(
        cycles,
        [float(row["loop_area_GPa"]) for row in loop_rows],
        marker="o",
        label=r"numerical $\oint\sigma\,d\gamma$",
    )
    plt.plot(
        cycles,
        [float(row["fit_expected_area_GPa"]) for row in loop_rows],
        marker="s",
        label=r"$\pi\gamma_0^2G''$",
    )
    plt.xlabel("cycle")
    plt.ylabel("loop area (GPa)")
    plt.xticks(cycles)
    plt.legend()
    save("cycle_loop_area.png")

    return figures


def analyze(
    *,
    stress_file: Path,
    run_dir: Path | None,
    output_dir: Path,
    period_ps: float | None = None,
    cycles: int | None = None,
    strain_amplitude: float | None = None,
    stress_sign: float | None = None,
    metadata_path: Path | None = None,
    input_path: Path | None = None,
    compare_fit_path: Path | None = None,
    make_plots: bool = True,
) -> dict[str, Any]:
    stress_file = stress_file.resolve()
    resolved_run_dir = infer_run_dir(stress_file, run_dir)
    protocol, metadata, metadata_file, input_file = resolve_protocol(
        resolved_run_dir,
        period_ps=period_ps,
        cycles=cycles,
        strain_amplitude=strain_amplitude,
        stress_sign=stress_sign,
        metadata_path=metadata_path,
        input_path=input_path,
    )
    series = load_stress_series(stress_file, protocol.stress_sign)
    validation = validate_series(series, protocol)
    overall, fitted_stress = fit_response(
        series.time_ps,
        series.sigma_xy_GPa,
        period_ps=protocol.period_ps,
        strain_amplitude=protocol.strain_amplitude,
    )
    cycle_rows, loop_rows = fit_cycles(series, protocol)

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fit_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "stress_file": str(stress_file),
        "stress_file_sha256": sha256(stress_file),
        "n_points": overall.n_points,
        "period_ps": protocol.period_ps,
        "cycles": protocol.cycles,
        "strain_amplitude": protocol.strain_amplitude,
        "stress_sign": protocol.stress_sign,
        "fit_model": FIT_MODEL,
        "a_sin_GPa": overall.a_sin_GPa,
        "b_cos_GPa": overall.b_cos_GPa,
        "offset_GPa": overall.offset_GPa,
        "stress_amplitude_GPa": overall.stress_amplitude_GPa,
        "phase_rad": overall.phase_rad,
        "phase_deg": overall.phase_deg,
        "tan_delta": overall.tan_delta,
        "G_storage_GPa": overall.G_storage_GPa,
        "G_loss_GPa": overall.G_loss_GPa,
        "rmse_GPa": overall.rmse_GPa,
        "mean_gamma": float(np.mean(series.gamma)),
        "max_abs_gamma": float(np.max(np.abs(series.gamma))),
        "mean_stress_GPa": float(np.mean(series.sigma_xy_GPa)),
        "protocol_source": protocol.source,
        "validation": validation,
        "provenance": {
            "run_dir": str(resolved_run_dir),
            "metadata_file": str(metadata_file.resolve()) if metadata_file else None,
            "metadata_file_sha256": sha256(metadata_file) if metadata_file else None,
            "input_file": str(input_file.resolve()) if input_file else None,
            "input_file_sha256": sha256(input_file) if input_file else None,
        },
    }
    if metadata is not None:
        fit_payload["run_metadata"] = {
            "run_name": nested_get(metadata, "run_config", "run_name"),
            "model_alias": nested_get(metadata, "run_config", "model_alias"),
            "model_kind": nested_get(metadata, "run_config", "model_kind"),
            "seed": nested_get(metadata, "run_config", "seed"),
            "natoms": nested_get(metadata, "run_config", "natoms"),
            "temperature_K": nested_get(metadata, "run_config", "temperature_low_K"),
        }

    stored_candidate = compare_fit_path
    if stored_candidate is None:
        candidate = resolved_run_dir / "mddms_fit.json"
        if candidate.is_file() and candidate.resolve() != (output_dir / "mddms_fit.json").resolve():
            stored_candidate = candidate
    comparison = compare_fit(stored_candidate, overall) if stored_candidate else None
    if comparison is not None:
        fit_payload["stored_fit_comparison"] = comparison

    write_json(output_dir / "mddms_fit.json", fit_payload)
    write_csv(output_dir / "mddms_cycles.csv", cycle_rows)
    write_json(output_dir / "mddms_cycles.json", cycle_rows)
    write_csv(output_dir / "mddms_loop_areas.csv", loop_rows)

    timeseries_rows = [
        {
            "step": float(series.step[index]),
            "time_ps": float(series.time_ps[index]),
            "gamma": float(series.gamma[index]),
            "pxy_bar": float(series.pxy_bar[index]),
            "sigma_xy_GPa": float(series.sigma_xy_GPa[index]),
            "sigma_fit_GPa": float(fitted_stress[index]),
            "residual_GPa": float(series.sigma_xy_GPa[index] - fitted_stress[index]),
            "temperature_K": float(series.temperature_K[index]),
            "pressure_bar": float(series.pressure_bar[index]),
        }
        for index in range(series.n_points)
    ]
    write_csv(output_dir / "stress_timeseries_with_fit.csv", timeseries_rows)

    figures = plot_results(output_dir, series, fitted_stress, cycle_rows, loop_rows) if make_plots else []
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
        "figures": figures,
        "primary_fit": {
            "G_storage_GPa": overall.G_storage_GPa,
            "G_loss_GPa": overall.G_loss_GPa,
            "tan_delta": overall.tan_delta,
            "phase_deg": overall.phase_deg,
            "rmse_GPa": overall.rmse_GPa,
        },
    }
    write_json(output_dir / "analysis_manifest.json", manifest)
    return fit_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit global and cycle-resolved MD-DMS response from a LAMMPS stress time series."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="Run directory containing stress_timeseries.dat")
    source.add_argument("--stress-file", type=Path, help="Explicit stress time-series file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for analysis outputs")
    parser.add_argument("--metadata", type=Path, help="Explicit metadata.json")
    parser.add_argument("--input-file", type=Path, help="Explicit 03_mddms_shear.in")
    parser.add_argument("--period-ps", type=float, help="Override oscillation period")
    parser.add_argument("--cycles", type=int, help="Override number of cycles")
    parser.add_argument("--strain-amplitude", type=float, help="Override gamma0")
    parser.add_argument("--stress-sign", type=float, help="Override pxy-to-sigma sign")
    parser.add_argument(
        "--compare-fit-json",
        type=Path,
        help="Compare recomputed fit with a historical mddms_fit.json",
    )
    parser.add_argument("--no-plots", action="store_true", help="Do not generate PNG diagnostics")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.run_dir is not None:
            run_dir = args.run_dir.resolve()
            stress_file = discover_stress_file(run_dir)
        else:
            stress_file = args.stress_file.resolve()
            run_dir = None
        result = analyze(
            stress_file=stress_file,
            run_dir=run_dir,
            output_dir=args.output_dir,
            period_ps=args.period_ps,
            cycles=args.cycles,
            strain_amplitude=args.strain_amplitude,
            stress_sign=args.stress_sign,
            metadata_path=args.metadata,
            input_path=args.input_file,
            compare_fit_path=args.compare_fit_json,
            make_plots=not args.no_plots,
        )
    except AnalysisError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Fit complete: "
        f"G'={result['G_storage_GPa']:.12g} GPa, "
        f"G''={result['G_loss_GPa']:.12g} GPa, "
        f"tan(delta)={result['tan_delta']:.12g}, "
        f"RMSE={result['rmse_GPa']:.12g} GPa"
    )
    print(f"Outputs: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
