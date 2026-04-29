# MD-DMS pilot runner

This is the first practical working-horse script for the CuZr-MD-DMS project.

## Location in repo

Place it here:

```bash
mkdir -p scripts/run
cp ~/Downloads/run_mddms_pilot.py scripts/run/run_mddms_pilot.py
chmod +x scripts/run/run_mddms_pilot.py
```

## Generate a technical pilot without running LAMMPS

```bash
python scripts/run/run_mddms_pilot.py \
  --run-name pilot_mace_c_001 \
  --preset pilot \
  --model-kind mace \
  --model-file models/mace/mace_C.model-mliap_lammps.pt
```

This creates:

```text
runs/pilot_mace_c_001/
  initial.data
  00_prepare_melt_quench.in
  01_relax_npt.in
  02_equilibrate_nvt.in
  03_mddms_shear.in
  run_lammps.sh
  metadata.json
```

## Run on cloud after LAMMPS is built

```bash
source /workspace/cuzr_mddms_runtime.env

python scripts/run/run_mddms_pilot.py \
  --run-name pilot_mace_c_001 \
  --preset tiny \
  --model-kind mace \
  --model-file /workspace/CuZr-MD-DMS/models/mace/mace_C.model-mliap_lammps.pt \
  --execute \
  --analyze
```

## Presets

- `tiny`: technical smoke test only, not physical.
- `pilot`: first practical cloud test.
- `ch3`: expensive Chapter-3-like baseline.

## MACE command

By default MACE runs use:

```bash
lmp -k on g 1 -sf kk -pk kokkos newton on neigh half
```

You can override it:

```bash
python scripts/run/run_mddms_pilot.py \
  --lmp-command "lmp -k on g 1 -sf kk -pk kokkos newton on neigh half"
```

## Atomic stress output

Atomic stress output is disabled by default because files can become huge.

Enable it with:

```bash
--dump-atomic
```

This adds:

```text
atom_stress.lammpstrj
```

with:

```text
id type x y z c_atomstress[4] c_atomvol[1]
```

Later analysis can use approximately:

```text
atomic_sxy = -c_atomstress[4] / c_atomvol[1]
```

but this sign convention should be verified against the system stress.
