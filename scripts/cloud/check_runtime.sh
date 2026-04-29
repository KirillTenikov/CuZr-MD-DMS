#!/usr/bin/env bash
set -euo pipefail

# CuZr-MD-DMS runtime check script

source /opt/venv/bin/activate || true

if [ -f /workspace/cuzr_mddms_runtime.env ]; then
  source /workspace/cuzr_mddms_runtime.env
fi

echo "[Python]"
which python
python - <<'PY'
import torch
import mace

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
print("mace:", mace.__file__)

try:
    import cuequivariance
    import cuequivariance_torch
    print("cuequivariance: OK")
except Exception as exc:
    print("cuequivariance: FAILED", repr(exc))
PY

echo
echo "[LAMMPS]"
if command -v lmp >/dev/null 2>&1; then
  which lmp
  lmp -h | grep -E "ML-IAP|KOKKOS|PYTHON|MANYBODY|VORONOI" || true
else
  echo "lmp not found in PATH"
fi
