# MD-DMS postprocessing

This document describes the standalone analysis script

```text
scripts/analysis/analyze_mddms_response.py
```

The script fits the global and cycle-resolved mechanical response from a
LAMMPS MD-DMS shear-stress time series.

## Requirements

Required:

```bash
python -m pip install numpy
```

Optional, for diagnostic figures:

```bash
python -m pip install matplotlib
```

When Matplotlib is unavailable, the numerical analysis can still be run with
`--no-plots`.

## Accepted input

The script accepts either:

1. a run directory containing an accepted stress time series; or
2. an explicitly specified stress file.

Run-directory mode:

```bash
python scripts/analysis/analyze_mddms_response.py \
  --run-dir runs/example_run \
  --output-dir analysis_outputs/example_run
```

The script searches the run directory in this order:

```text
stress_timeseries_stitched_final.dat
stress_timeseries.dat
```

Explicit-file mode:

```bash
python scripts/analysis/analyze_mddms_response.py \
  --stress-file /path/to/stress_timeseries.dat \
  --output-dir analysis_outputs/example_run \
  --period-ps 20 \
  --cycles 6 \
  --strain-amplitude 0.01 \
  --stress-sign -1
```

## Stress-series format

A whitespace-delimited stress file must contain at least the following ten
columns in this order:

```text
step
time_ps
gamma
pxy_bar
temperature_K
pe_eV
ke_eV
pressure_bar
xy_A
ly_A
```

Comment lines beginning with `#` are allowed.

A CSV file with a header is also accepted. The canonical column names above
are preferred; several short aliases such as `time`, `strain`, `pxy`,
`temperature`, `pe`, `ke`, `pressure`, `xy`, and `ly` are recognized.

## Protocol discovery

Protocol parameters are resolved in the following order:

1. explicit command-line arguments;
2. `metadata.json`;
3. `03_mddms_shear.in` or `03_mddms_shear_retry_traj_only.in`;
4. the documented default, when one exists.

The oscillation period, number of cycles, and strain amplitude must be
recoverable from these sources. The default stress sign is

```text
stress_sign = -1
```

The conversion from LAMMPS metal-unit pressure to shear stress is

\[
\sigma_{xy}\,[\mathrm{GPa}]
=
s\,p_{xy}\,[\mathrm{bar}]
\times 10^{-4},
\]

where \(s\) is `stress_sign`.

## Response model

For a sinusoidal strain

\[
\gamma(t)=\gamma_0\sin(\omega t),
\qquad
\omega=\frac{2\pi}{T},
\]

the stress is fitted by ordinary linear least squares:

\[
\sigma_{xy}(t)
=
a\sin(\omega t)
+
b\cos(\omega t)
+
c.
\]

The reported quantities are

\[
G'=\frac{a}{\gamma_0},
\qquad
G''=\frac{b}{\gamma_0},
\]

\[
\tan\delta=\frac{G''}{G'},
\qquad
\delta=\operatorname{atan2}(G'',G').
\]

The root-mean-square residual is reported as `rmse_GPa`.

## Historical Paper 2 convention

The primary fit uses every stored point from all cycles, including the first
cycle. This convention is retained to ensure direct comparability with the
submitted Paper 2 analysis.

Cycle-resolved fits use half-open time intervals

```text
[start, end)
```

for cycles 1 through \(N-1\). The final cycle includes its endpoint.

## Validation checks

Before fitting, the script checks that:

- all required values are finite;
- the stress series contains enough points;
- step and time columns are strictly increasing;
- the total duration agrees with period × number of cycles;
- the maximum absolute strain agrees with \(\gamma_0\);
- the harmonic design matrix has full rank.

The analysis stops with an error when these conditions are not satisfied.

Irregular step or time intervals are counted and written to the validation
metadata. They do not automatically invalidate a series when the time axis
remains complete and strictly increasing.

## Outputs

The output directory contains numerical results and, unless `--no-plots` is
used, diagnostic figures.

Numerical files:

```text
mddms_fit.json
mddms_cycles.csv
mddms_cycles.json
mddms_loop_areas.csv
stress_timeseries_with_fit.csv
analysis_manifest.json
```

Typical figures:

```text
stress_vs_time_with_fit.png
hysteresis_all_cycles.png
cycle_G_storage.png
cycle_G_loss.png
cycle_tan_delta.png
cycle_phase.png
cycle_loop_area.png
```

### `mddms_fit.json`

Contains the global fit, resolved protocol, validation summary, provenance,
input-file SHA-256 hashes, and optional comparison with a historical
`mddms_fit.json`.

### `mddms_cycles.csv`

Contains \(G'\), \(G''\), phase, \(\tan\delta\), offset, RMSE, and sample
information for each oscillation cycle.

### `mddms_loop_areas.csv`

Contains the numerical hysteresis-loop integral and the harmonic-fit
expectation

\[
\pi\gamma_0^2G''
\]

for each cycle.

### `stress_timeseries_with_fit.csv`

Contains the original stress series together with the fitted stress and
pointwise residual.

## Useful commands

Show all options:

```bash
python scripts/analysis/analyze_mddms_response.py --help
```

Analyze a complete run directory:

```bash
python scripts/analysis/analyze_mddms_response.py \
  --run-dir runs/rev_mace_seed46_P50 \
  --output-dir analysis_outputs/rev_mace_seed46_P50
```

Analyze an old archive after extraction:

```bash
python scripts/analysis/analyze_mddms_response.py \
  --stress-file extracted_run/stress_timeseries.dat \
  --output-dir analysis_outputs/old_run \
  --period-ps 50 \
  --cycles 6 \
  --strain-amplitude 0.01 \
  --stress-sign -1
```

Compare a recomputed fit with a stored historical fit:

```bash
python scripts/analysis/analyze_mddms_response.py \
  --run-dir runs/example_run \
  --output-dir analysis_outputs/example_run \
  --compare-fit-json runs/example_run/mddms_fit.json
```

Generate numerical outputs without PNG files:

```bash
python scripts/analysis/analyze_mddms_response.py \
  --run-dir runs/example_run \
  --output-dir analysis_outputs/example_run \
  --no-plots
```

## Reproducibility note

The standalone implementation was checked against the historical global and
cycle-resolved MACE and EAM analyses used for Paper 2. The public repository
contains the validated analysis script; internal regression materials and
development audit files are not part of the source distribution.
