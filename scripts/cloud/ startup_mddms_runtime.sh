#!/usr/bin/env bash
set -euo pipefail

# CuZr-MD-DMS startup for the Paper-1 CuZr MD Docker image.
#
# This script uses the existing Paper-1 Docker image as the runtime base for
# Paper-2 MD-DMS. It activates /opt/cuzr-mamba, checks/installs cuEquivariance,
# builds LAMMPS on the cloud GPU machine with ML-IAP + Kokkos CUDA, and writes
# /workspace/cuzr_mddms_runtime.env.
#
# Typical use inside container:
#   cd /workspace/CuZr-MD-DMS
#   bash scripts/cloud/startup_paper1_image_mddms.sh
#
# A100 default:
#   KOKKOS_ARCH_FLAG=Kokkos_ARCH_AMPERE80
#
# H100 later:
#   export KOKKOS_ARCH_FLAG=Kokkos_ARCH_HOPPER90
#
# Force clean rebuild:
#   export FORCE_REBUILD=1
#   bash scripts/cloud/startup_paper1_image_mddms.sh

echo "[CuZr-MD-DMS] Startup for Paper-1 Docker image"

export WORKSPACE="${WORKSPACE:-/workspace}"
export CUZR_ENV_PREFIX="${CUZR_ENV_PREFIX:-/opt/cuzr-mamba}"
export PYTHON_BIN="${PYTHON_BIN:-${CUZR_ENV_PREFIX}/bin/python}"
export PIP_BIN="${PIP_BIN:-${CUZR_ENV_PREFIX}/bin/pip}"

if [ -f /opt/cuzr_python_prebuilt.env ]; then
  source /opt/cuzr_python_prebuilt.env
fi
if [ -f /etc/profile.d/cuzr-env.sh ]; then
  source /etc/profile.d/cuzr-env.sh
fi

export PATH="${CUZR_ENV_PREFIX}/bin:${PATH}"

export LAMMPS_DIR="${LAMMPS_DIR:-/opt/lammps}"
export LAMMPS_REF="${LAMMPS_REF:-develop}"
export LAMMPS_ROOT="${LAMMPS_ROOT:-${WORKSPACE}/lammps_mddms}"
export LAMMPS_BUILD_DIR="${LAMMPS_BUILD_DIR:-${LAMMPS_ROOT}/build-mliap-kokkos}"
export LAMMPS_INSTALL_DIR="${LAMMPS_INSTALL_DIR:-${LAMMPS_ROOT}/install-mliap-kokkos}"

export KOKKOS_ARCH_FLAG="${KOKKOS_ARCH_FLAG:-Kokkos_ARCH_AMPERE80}"
export BUILD_MPI="${BUILD_MPI:-ON}"
export LAMMPS_BUILD_JOBS="${LAMMPS_BUILD_JOBS:-$(nproc)}"
export FORCE_REBUILD="${FORCE_REBUILD:-0}"

export INSTALL_CUEQ_IF_MISSING="${INSTALL_CUEQ_IF_MISSING:-1}"
export CUEQ_VERSION="${CUEQ_VERSION:-0.9.1}"
export CUPY_VERSION="${CUPY_VERSION:-13.6.0}"

mkdir -p "${WORKSPACE}" "${LAMMPS_ROOT}"
LOG_DIR="${WORKSPACE}/logs/runtime"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/startup_paper1_image_mddms_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[Info] Log file: ${LOG_FILE}"
echo "[Info] WORKSPACE: ${WORKSPACE}"
echo "[Info] CUZR_ENV_PREFIX: ${CUZR_ENV_PREFIX}"
echo "[Info] PYTHON_BIN: ${PYTHON_BIN}"
echo "[Info] LAMMPS_DIR: ${LAMMPS_DIR}"
echo "[Info] LAMMPS_BUILD_DIR: ${LAMMPS_BUILD_DIR}"
echo "[Info] LAMMPS_INSTALL_DIR: ${LAMMPS_INSTALL_DIR}"
echo "[Info] KOKKOS_ARCH_FLAG: ${KOKKOS_ARCH_FLAG}"
echo "[Info] BUILD_MPI: ${BUILD_MPI}"
echo "[Info] LAMMPS_BUILD_JOBS: ${LAMMPS_BUILD_JOBS}"
echo

echo "[1/8] GPU check"
nvidia-smi || true
echo

echo "[2/8] Python/MACE/cuEquivariance check"
"${PYTHON_BIN}" - <<'PY'
import importlib.util
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

for name in ["mace", "cupy", "cuequivariance", "cuequivariance_torch"]:
    spec = importlib.util.find_spec(name)
    print(f"{name}:", "OK" if spec else "MISSING")
PY

if [ "${INSTALL_CUEQ_IF_MISSING}" = "1" ]; then
  echo
  echo "[3/8] Install missing cuEquivariance packages if needed"
  if ! "${PYTHON_BIN}" - <<'PY'
import importlib.util
missing = [
    name for name in ["cupy", "cuequivariance", "cuequivariance_torch"]
    if importlib.util.find_spec(name) is None
]
raise SystemExit(1 if missing else 0)
PY
  then
    echo "[Info] Some cuEquivariance packages are missing; installing into ${CUZR_ENV_PREFIX}"
    "${PIP_BIN}" install --no-cache-dir       "cupy-cuda12x==${CUPY_VERSION}"       "cuequivariance==${CUEQ_VERSION}"       "cuequivariance-torch==${CUEQ_VERSION}"       "cuequivariance-ops-torch-cu12==${CUEQ_VERSION}"
  else
    echo "[Info] cuEquivariance packages already available."
  fi
else
  echo
  echo "[3/8] Skipping cuEquivariance auto-install because INSTALL_CUEQ_IF_MISSING=0"
fi

echo
echo "[4/8] Final Python acceleration check"
"${PYTHON_BIN}" - <<'PY'
import torch
import mace
import cupy
import cuequivariance as cue
import cuequivariance_torch as cuet

print("mace:", mace.__file__)
print("cupy:", cupy.__version__)
print("cuequivariance:", getattr(cue, "__version__", "imported"))
print("cuequivariance_torch:", cuet.__file__)
print("cuda available:", torch.cuda.is_available())
PY

echo
echo "[5/8] Prepare LAMMPS source"
if [ -d "${LAMMPS_DIR}" ]; then
  echo "[Info] Using existing LAMMPS source: ${LAMMPS_DIR}"
  if [ -d "${LAMMPS_DIR}/.git" ]; then
    echo "[Info] LAMMPS source is a git checkout; updating to ${LAMMPS_REF}"
    cd "${LAMMPS_DIR}"
    git fetch --all --tags
    git checkout "${LAMMPS_REF}"
  else
    echo "[Info] LAMMPS source is not a git checkout; using it as-is."
  fi
else
  echo "[Info] No LAMMPS source found at ${LAMMPS_DIR}; cloning ${LAMMPS_REF}"
  git clone --branch "${LAMMPS_REF}" https://github.com/lammps/lammps.git "${LAMMPS_DIR}"
fi

if [ ! -d "${LAMMPS_DIR}/cmake" ]; then
  echo "ERROR: ${LAMMPS_DIR}/cmake not found. LAMMPS source is invalid."
  exit 1
fi

echo
echo "[6/8] Check existing LAMMPS install"
if [ "${FORCE_REBUILD}" = "1" ]; then
  echo "[Info] FORCE_REBUILD=1; removing old build/install directories"
  rm -rf "${LAMMPS_BUILD_DIR}" "${LAMMPS_INSTALL_DIR}"
fi

if [ -x "${LAMMPS_INSTALL_DIR}/bin/lmp" ]; then
  echo "[Info] Existing LAMMPS binary found:"
  "${LAMMPS_INSTALL_DIR}/bin/lmp" -h | head -n 20 || true
else
  echo "[Info] Building LAMMPS now."

  KOKKOS_WRAPPER="${LAMMPS_DIR}/lib/kokkos/bin/nvcc_wrapper"
  if [ ! -x "${KOKKOS_WRAPPER}" ]; then
    echo "ERROR: Kokkos nvcc_wrapper not found at ${KOKKOS_WRAPPER}"
    exit 1
  fi

  if [ "${BUILD_MPI}" = "ON" ]; then
    export NVCC_WRAPPER_DEFAULT_COMPILER="$(command -v mpicxx)"
  else
    export NVCC_WRAPPER_DEFAULT_COMPILER="$(command -v g++)"
  fi

  echo
  echo "[7/8] Configure/build/install LAMMPS"
  cmake -S "${LAMMPS_DIR}/cmake" -B "${LAMMPS_BUILD_DIR}"     -D CMAKE_BUILD_TYPE=Release     -D CMAKE_INSTALL_PREFIX="${LAMMPS_INSTALL_DIR}"     -D CMAKE_CXX_COMPILER="${KOKKOS_WRAPPER}"     -D CMAKE_CXX_STANDARD=17     -D BUILD_MPI="${BUILD_MPI}"     -D BUILD_SHARED_LIBS=ON     -D PKG_ML-IAP=ON     -D PKG_ML-SNAP=ON     -D MLIAP_ENABLE_PYTHON=ON     -D PKG_PYTHON=ON     -D PKG_KOKKOS=ON     -D Kokkos_ENABLE_CUDA=ON     -D "${KOKKOS_ARCH_FLAG}=ON"     -D PKG_MANYBODY=ON     -D PKG_EXTRA-COMPUTE=ON     -D PKG_EXTRA-FIX=ON     -D PKG_MISC=ON     -D Python_EXECUTABLE="${PYTHON_BIN}"     -D Python3_EXECUTABLE="${PYTHON_BIN}"

  cmake --build "${LAMMPS_BUILD_DIR}" -j "${LAMMPS_BUILD_JOBS}"
  cmake --install "${LAMMPS_BUILD_DIR}"

  cmake --build "${LAMMPS_BUILD_DIR}" --target install-python || true
fi

echo
echo "[8/8] Write runtime environment"
cat > "${WORKSPACE}/cuzr_mddms_runtime.env" <<RUNTIME_ENV
export WORKSPACE="${WORKSPACE}"
export CUZR_ENV_PREFIX="${CUZR_ENV_PREFIX}"
export PYTHON_BIN="${PYTHON_BIN}"
export PIP_BIN="${PIP_BIN}"
export LAMMPS_DIR="${LAMMPS_DIR}"
export LAMMPS_BUILD_DIR="${LAMMPS_BUILD_DIR}"
export LAMMPS_INSTALL_DIR="${LAMMPS_INSTALL_DIR}"
export PATH="${LAMMPS_INSTALL_DIR}/bin:${CUZR_ENV_PREFIX}/bin:\$PATH"
export LD_LIBRARY_PATH="${LAMMPS_INSTALL_DIR}/lib:\${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${LAMMPS_INSTALL_DIR}/lib/python:\${PYTHONPATH:-}"
export OMP_NUM_THREADS="\${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="\${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="\${OPENBLAS_NUM_THREADS:-1}"
export MACE_TIME="\${MACE_TIME:-false}"
export MACE_PROFILE="\${MACE_PROFILE:-false}"
export LMP_MACE_KOKKOS_CMD="lmp -k on g 1 -sf kk -pk kokkos newton on neigh half"
RUNTIME_ENV

source "${WORKSPACE}/cuzr_mddms_runtime.env"

echo
echo "[CuZr-MD-DMS] Runtime check"
which python
"${PYTHON_BIN}" - <<'PY'
import torch
import mace
import cupy
import cuequivariance
import cuequivariance_torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
print("mace:", mace.__file__)
print("cupy:", cupy.__version__)
print("cuequivariance: OK")
PY

which lmp
lmp -h | grep -E "ML-IAP|ML-SNAP|KOKKOS|PYTHON|MANYBODY" || true

echo
echo "[CuZr-MD-DMS] Done."
echo "Activate later with:"
echo "  source ${WORKSPACE}/cuzr_mddms_runtime.env"
echo
echo "Recommended MACE/Kokkos run command:"
echo "  \${LMP_MACE_KOKKOS_CMD} -in input.in"
