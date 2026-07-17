# CuZr-MD-DMS

Reproducible molecular-dynamics dynamical mechanical spectroscopy (MD-DMS)
workflows for studying fast mechanical relaxation and NCL-like loss in
Cu–Zr metallic glasses.

The project compares a DFT-trained MACE interatomic potential with a
classical Mendelev EAM potential under the same oscillatory shear protocol.
The principal observables are the storage modulus \(G'\), loss modulus
\(G''\), loss tangent \(\tan\delta\), phase lag, hysteresis-loop area, and
cycle-to-cycle variability.

## Scientific objective

The central question is whether a fast dissipative mechanical response in
Cu\(_{64}\)Zr\(_{36}\) metallic glass remains present when the interatomic
description is changed from an empirical EAM potential to a more flexible
DFT-trained machine-learning potential.

The current Paper 2 protocol uses:

- 4000 atoms;
- Cu fraction 0.64;
- temperature 300 K;
- sinusoidal shear strain with amplitude \(\gamma_0 = 0.01\);
- periods of 20 and 50 ps;
- six oscillation cycles;
- a 1 fs molecular-dynamics timestep;
- independent amorphous realizations generated with different random seeds.

## Workflow

The computational workflow is organized into four stages:

1. preparation, melting, and quenching;
2. NPT relaxation;
3. NVT equilibration;
4. MD-DMS production under sinusoidal shear.

The shear-stress time series is analyzed with the standalone script

```text
scripts/analysis/analyze_mddms_response.py
```

The response is fitted as

\[
\sigma_{xy}(t)
=
a\sin(\omega t)
+
b\cos(\omega t)
+
c,
\]

with

\[
G'=\frac{a}{\gamma_0},
\qquad
G''=\frac{b}{\gamma_0},
\qquad
\tan\delta=\frac{G''}{G'}.
\]

The primary fit uses all stored points from all oscillation cycles, including
the first cycle. Cycle-resolved fits are also generated.

Detailed usage is documented in
[`docs/postprocessing.md`](docs/postprocessing.md).

## Repository structure

```text
CuZr-MD-DMS/
├── docs/                   project and workflow documentation
├── env/                    environment-related files
├── models/                 model placeholders and lightweight metadata
├── runs/                   generated run directories; not version-controlled
├── scripts/
│   ├── analysis/           standalone postprocessing utilities
│   ├── cloud/              cloud setup and runtime scripts
│   └── run/                simulation preparation and launch scripts
├── src/cuzr_mddms/         Python package source
├── .gitignore
├── pyproject.toml
└── README.md
```

Additional documentation:

- [`docs/README_mddms_pilot_runner.md`](docs/README_mddms_pilot_runner.md)
- [`docs/runtime_policy.md`](docs/runtime_policy.md)
- [`docs/project_plan.md`](docs/project_plan.md)
- [`docs/data_policy.md`](docs/data_policy.md)

## Postprocessing example

When a run directory contains an accepted stress series together with
`metadata.json` and/or `03_mddms_shear.in`:

```bash
python scripts/analysis/analyze_mddms_response.py \
  --run-dir runs/example_run \
  --output-dir analysis_outputs/example_run
```

For a standalone stress file with explicit protocol parameters:

```bash
python scripts/analysis/analyze_mddms_response.py \
  --stress-file /path/to/stress_timeseries.dat \
  --output-dir analysis_outputs/example_run \
  --period-ps 50 \
  --cycles 6 \
  --strain-amplitude 0.01 \
  --stress-sign -1
```

Use `--no-plots` when only numerical tables and JSON outputs are needed.

## Software

The main workflow uses:

- Python;
- NumPy;
- Matplotlib for optional diagnostic plots;
- LAMMPS;
- MACE through the LAMMPS ML-IAP/Kokkos workflow;
- the Mendelev Cu–Zr EAM potential for the classical comparison.

Runtime and cloud-build details are described in
[`docs/runtime_policy.md`](docs/runtime_policy.md).

## Data availability

Raw trajectories, stress series, restart files, and complete run archives are
not stored in the Git source tree. Selected datasets intended for public
distribution are provided separately as GitHub Release assets.

The local `results/` directory is a working archive and is intentionally
ignored by Git. Generated analyses, figures, and manuscript files are also
kept outside version control.

See [`docs/data_policy.md`](docs/data_policy.md) for the complete policy.

## Reproducibility

The postprocessing script records input-file hashes, resolved protocol
parameters, validation diagnostics, and an analysis manifest. This allows
numerical results to be traced to the exact stress series and analysis
configuration used to produce them.
