#!/usr/bin/env bash
set -euo pipefail

# Runtime check for using the Paper-1 CuZr MD Docker image with CuZr-MD-DMS.

export WORKSPACE="${WORKSPACE:-/workspace}"

if [ -f /opt/cuzr_python_prebuilt.env ]; then
  source /opt/cuzr_python_prebuilt.env
fi
if [ -f /etc/profile.d/cuzr-env.sh ]; then
  source /etc/profile.d/cuzr-env.sh
fi
if [ -f "${WORKSPACE}/cuzr_mddms_runtime.env" ]; then
  source "${WORKSPACE}/cuzr_mddms_runtime.env"
fi

export CUZR_ENV_PREFIX="${CUZR_ENV_PREFIX:-/opt/cuzr-mamba}"
export PYTHON_BIN="${PYTHON_BIN:-${CUZR_ENV_PREFIX}/bin/python}"

echo "[Environment]"
echo "WORKSPACE=${WORKSPACE}"
echo "CUZR_ENV_PREFIX=${CUZR_ENV_PREFIX}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo

echo "[GPU]"
nvidia-smi || true
echo

echo "[Python stack]"
"${PYTHON_BIN}" - <<'PY'
import importlib.util
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

for name in ["numpy", "mace", "cupy", "cuequivariance", "cuequivariance_torch"]:
    spec = importlib.util.find_spec(name)
    print(f"{name}:", "OK" if spec else "MISSING")

try:
    import mace
    print("mace file:", mace.__file__)
except Exception as exc:
    print("mace import failed:", repr(exc))
PY
echo

echo "[LAMMPS]"
if command -v lmp >/dev/null 2>&1; then
  which lmp
  lmp -h | grep -E "ML-IAP|ML-SNAP|KOKKOS|PYTHON|MANYBODY" || true
else
  echo "lmp not found in PATH"
fi

echo
echo "[Suggested MACE command]"
echo "${LMP_MACE_KOKKOS_CMD:-lmp -k on g 1 -sf kk -pk kokkos newton on neigh half} -in input.in"
