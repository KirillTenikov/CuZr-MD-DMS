#!/usr/bin/env bash
set -euo pipefail

# CuZr-MD-DMS runtime startup script.
#
# Run manually after the QData instance/container starts:
#
#   cd /workspace/CuZr-MD-DMS
#   export LAMMPS_BUILD_JOBS=6
#   export KOKKOS_ARCH_FLAG=Kokkos_ARCH_AMPERE80
#   bash scripts/cloud/startup_mddms_runtime.sh
#
# Expected GitHub Release assets:
#   https://github.com/KirillTenikov/CuZr-MD-DMS/releases/download/Models/MACE_C.model
#   https://github.com/KirillTenikov/CuZr-MD-DMS/releases/download/Models/Cu-Zr_4.eam.fs

echo "[CuZr-MD-DMS] runtime startup started"

# -----------------------------
# Configuration
# -----------------------------
export WORKSPACE="${WORKSPACE:-/workspace}"
export PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"

export CUZR_ENV_PREFIX="${CUZR_ENV_PREFIX:-/opt/cuzr-mamba}"
export PYTHON_BIN="${PYTHON_BIN:-${CUZR_ENV_PREFIX}/bin/python}"
export PIP_BIN="${PIP_BIN:-${CUZR_ENV_PREFIX}/bin/pip}"

export MODELS_BASE_URL="${MODELS_BASE_URL:-https://github.com/KirillTenikov/CuZr-MD-DMS/releases/download/Models}"
export MACE_C_ASSET="${MACE_C_ASSET:-MACE_C.model}"
export EAM_ASSET="${EAM_ASSET:-Cu-Zr_4.eam.fs}"

export MACE_DIR="${MACE_DIR:-${PROJECT_DIR}/models/mace}"
export EAM_DIR="${EAM_DIR:-${PROJECT_DIR}/models/eam}"

export MACE_C_RAW="${MACE_C_RAW:-${MACE_DIR}/${MACE_C_ASSET}}"
export MACE_C_RAW_LINK="${MACE_C_RAW_LINK:-${MACE_DIR}/mace_C_raw.model}"
export MACE_C_MLIAP_LINK="${MACE_C_MLIAP_LINK:-${MACE_DIR}/mace_C.model-mliap_lammps.pt}"

export EAM_FILE="${EAM_FILE:-${EAM_DIR}/${EAM_ASSET}}"
export EAM_LINK="${EAM_LINK:-${EAM_DIR}/cuzr_eam.fs}"

export LAMMPS_DIR="${LAMMPS_DIR:-/opt/lammps}"
export LAMMPS_REF="${LAMMPS_REF:-develop}"
export LAMMPS_ROOT="${LAMMPS_ROOT:-${WORKSPACE}/lammps_mddms}"
export LAMMPS_BUILD_DIR="${LAMMPS_BUILD_DIR:-${LAMMPS_ROOT}/build-mliap-kokkos}"
export LAMMPS_INSTALL_DIR="${LAMMPS_INSTALL_DIR:-${LAMMPS_ROOT}/install-mliap-kokkos}"

export KOKKOS_ARCH_FLAG="${KOKKOS_ARCH_FLAG:-Kokkos_ARCH_AMPERE80}"
export BUILD_MPI="${BUILD_MPI:-ON}"
export LAMMPS_BUILD_JOBS="${LAMMPS_BUILD_JOBS:-$(nproc)}"

export CHECK_ONLY="${CHECK_ONLY:-0}"
export FETCH_MODELS="${FETCH_MODELS:-1}"
export CONVERT_MACE="${CONVERT_MACE:-1}"
export BUILD_LAMMPS="${BUILD_LAMMPS:-1}"
export RUN_TINY_TEST="${RUN_TINY_TEST:-0}"

export FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
export FORCE_CONVERT="${FORCE_CONVERT:-0}"
export FORCE_REBUILD="${FORCE_REBUILD:-0}"

export INSTALL_CUEQ_IF_MISSING="${INSTALL_CUEQ_IF_MISSING:-1}"
export CUEQ_VERSION="${CUEQ_VERSION:-0.9.1}"
export CUPY_VERSION="${CUPY_VERSION:-13.6.0}"

mkdir -p "${WORKSPACE}" "${PROJECT_DIR}" "${MACE_DIR}" "${EAM_DIR}" "${LAMMPS_ROOT}"

LOG_DIR="${WORKSPACE}/logs/runtime"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/startup_mddms_runtime_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

section() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

activate_python_env() {
  if [ -f /opt/cuzr_python_prebuilt.env ]; then
    # shellcheck disable=SC1091
    source /opt/cuzr_python_prebuilt.env
  fi

  if [ -f /etc/profile.d/cuzr-env.sh ]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/cuzr-env.sh
  fi

  export PATH="${CUZR_ENV_PREFIX}/bin:${PATH}"

  if [ ! -x "${PYTHON_BIN}" ]; then
    echo "[warning] ${PYTHON_BIN} not found, falling back to python from PATH"
    export PYTHON_BIN="$(command -v python)"
    export PIP_BIN="$(command -v pip)"
  fi
}

print_config() {
  section "[0/6] Configuration"
  echo "WORKSPACE=${WORKSPACE}"
  echo "PROJECT_DIR=${PROJECT_DIR}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
  echo "MODELS_BASE_URL=${MODELS_BASE_URL}"
  echo "MACE_C_ASSET=${MACE_C_ASSET}"
  echo "EAM_ASSET=${EAM_ASSET}"
  echo "LAMMPS_DIR=${LAMMPS_DIR}"
  echo "LAMMPS_BUILD_DIR=${LAMMPS_BUILD_DIR}"
  echo "LAMMPS_INSTALL_DIR=${LAMMPS_INSTALL_DIR}"
  echo "KOKKOS_ARCH_FLAG=${KOKKOS_ARCH_FLAG}"
  echo "BUILD_MPI=${BUILD_MPI}"
  echo "LAMMPS_BUILD_JOBS=${LAMMPS_BUILD_JOBS}"
  echo "FETCH_MODELS=${FETCH_MODELS}"
  echo "CONVERT_MACE=${CONVERT_MACE}"
  echo "BUILD_LAMMPS=${BUILD_LAMMPS}"
  echo "RUN_TINY_TEST=${RUN_TINY_TEST}"
  echo "LOG_FILE=${LOG_FILE}"
}

check_python_stack() {
  section "[1/6] Check GPU / Python / MACE / cuEquivariance"

  nvidia-smi || true

  "${PYTHON_BIN}" - <<'PY'
import importlib.util
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

for name in ["numpy", "scipy", "mace", "cupy", "cuequivariance", "cuequivariance_torch"]:
    spec = importlib.util.find_spec(name)
    print(f"{name}:", "OK" if spec else "MISSING")
PY

  if [ "${INSTALL_CUEQ_IF_MISSING}" = "1" ]; then
    if ! "${PYTHON_BIN}" - <<'PY'
import importlib.util
missing = [
    name for name in ["cupy", "cuequivariance", "cuequivariance_torch"]
    if importlib.util.find_spec(name) is None
]
raise SystemExit(1 if missing else 0)
PY
    then
      echo "[info] installing missing cuEquivariance stack"
      "${PIP_BIN}" install --no-cache-dir \
        "cupy-cuda12x==${CUPY_VERSION}" \
        "cuequivariance==${CUEQ_VERSION}" \
        "cuequivariance-torch==${CUEQ_VERSION}" \
        "cuequivariance-ops-torch-cu12==${CUEQ_VERSION}"
    else
      echo "[info] cuEquivariance stack already available"
    fi
  fi
}

fetch_models() {
  section "[2/6] Fetch model files"

  if [ "${FETCH_MODELS}" != "1" ]; then
    echo "[info] FETCH_MODELS=0, skipping"
    return
  fi

  if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl is required"
    exit 1
  fi

  local mace_url="${MODELS_BASE_URL}/${MACE_C_ASSET}"
  local eam_url="${MODELS_BASE_URL}/${EAM_ASSET}"

  echo "MACE_C URL: ${mace_url}"
  echo "EAM URL:    ${eam_url}"

  if [ ! -f "${MACE_C_RAW}" ] || [ "${FORCE_DOWNLOAD}" = "1" ]; then
    curl -fL --retry 3 --retry-delay 5 -o "${MACE_C_RAW}" "${mace_url}"
  else
    echo "[info] already present: ${MACE_C_RAW}"
  fi

  if [ ! -f "${EAM_FILE}" ] || [ "${FORCE_DOWNLOAD}" = "1" ]; then
    curl -fL --retry 3 --retry-delay 5 -o "${EAM_FILE}" "${eam_url}"
  else
    echo "[info] already present: ${EAM_FILE}"
  fi

  rm -f "${MACE_C_RAW_LINK}"
  ln -s "$(basename "${MACE_C_RAW}")" "${MACE_C_RAW_LINK}"

  rm -f "${EAM_LINK}"
  ln -s "$(basename "${EAM_FILE}")" "${EAM_LINK}"

  echo "MACE directory:"
  ls -lh "${MACE_DIR}"
  echo "EAM directory:"
  ls -lh "${EAM_DIR}"
}

convert_mace_c() {
  section "[3/6] Convert MACE_C to LAMMPS ML-IAP format"

  if [ "${CONVERT_MACE}" != "1" ]; then
    echo "[info] CONVERT_MACE=0, skipping"
    return
  fi

  if [ -e "${MACE_C_MLIAP_LINK}" ] && [ "${FORCE_CONVERT}" != "1" ]; then
    echo "[info] converted ML-IAP model already exists:"
    ls -lh "${MACE_C_MLIAP_LINK}"
    return
  fi

  if [ ! -f "${MACE_C_RAW}" ]; then
    echo "ERROR: raw MACE_C model is missing: ${MACE_C_RAW}"
    exit 1
  fi

  "${PYTHON_BIN}" - <<'PY'
import torch
print("CUDA visible for MACE conversion:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: MACE ML-IAP conversion is recommended on GPU.")
PY

  echo "Converting ${MACE_C_RAW}"
  "${PYTHON_BIN}" -m mace.cli.create_lammps_model "${MACE_C_RAW}" --format=mliap

  local expected="${MACE_C_RAW}-mliap_lammps.pt"
  local converted=""

  if [ -f "${expected}" ]; then
    converted="${expected}"
  else
    converted="$(find "${MACE_DIR}" -maxdepth 1 -type f -name "*mliap_lammps.pt" -print0 \
      | xargs -0 ls -t 2>/dev/null \
      | head -n 1 || true)"
  fi

  if [ -z "${converted}" ] || [ ! -f "${converted}" ]; then
    echo "ERROR: conversion finished, but no *mliap_lammps.pt file was found"
    ls -lh "${MACE_DIR}"
    exit 1
  fi

  rm -f "${MACE_C_MLIAP_LINK}"
  ln -s "$(basename "${converted}")" "${MACE_C_MLIAP_LINK}"

  echo "Stable ML-IAP model:"
  ls -lh "${MACE_C_MLIAP_LINK}"
}

build_lammps() {
  section "[4/6] Build LAMMPS with ML-IAP + Kokkos CUDA"

  if [ "${BUILD_LAMMPS}" != "1" ]; then
    echo "[info] BUILD_LAMMPS=0, skipping"
    return
  fi

  if [ -d "${LAMMPS_DIR}" ]; then
    echo "[info] using LAMMPS source: ${LAMMPS_DIR}"
    if [ -d "${LAMMPS_DIR}/.git" ]; then
      cd "${LAMMPS_DIR}"
      git fetch --all --tags
      git checkout "${LAMMPS_REF}"
    else
      echo "[info] LAMMPS source is not a git checkout, using as-is"
    fi
  else
    git clone --branch "${LAMMPS_REF}" https://github.com/lammps/lammps.git "${LAMMPS_DIR}"
  fi

  if [ ! -d "${LAMMPS_DIR}/cmake" ]; then
    echo "ERROR: invalid LAMMPS source; ${LAMMPS_DIR}/cmake not found"
    exit 1
  fi

  if [ "${FORCE_REBUILD}" = "1" ]; then
    rm -rf "${LAMMPS_BUILD_DIR}" "${LAMMPS_INSTALL_DIR}"
  fi

  if [ -x "${LAMMPS_INSTALL_DIR}/bin/lmp" ]; then
    echo "[info] existing LAMMPS binary found:"
    "${LAMMPS_INSTALL_DIR}/bin/lmp" -h | head -n 20 || true
    return
  fi

  local kokkos_wrapper="${LAMMPS_DIR}/lib/kokkos/bin/nvcc_wrapper"
  if [ ! -x "${kokkos_wrapper}" ]; then
    echo "ERROR: Kokkos nvcc_wrapper not found: ${kokkos_wrapper}"
    exit 1
  fi

  if [ "${BUILD_MPI}" = "ON" ]; then
    export NVCC_WRAPPER_DEFAULT_COMPILER="$(command -v mpicxx)"
  else
    export NVCC_WRAPPER_DEFAULT_COMPILER="$(command -v g++)"
  fi

  cmake -S "${LAMMPS_DIR}/cmake" -B "${LAMMPS_BUILD_DIR}" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX="${LAMMPS_INSTALL_DIR}" \
    -D CMAKE_CXX_COMPILER="${kokkos_wrapper}" \
    -D CMAKE_CXX_STANDARD=17 \
    -D BUILD_MPI="${BUILD_MPI}" \
    -D BUILD_SHARED_LIBS=ON \
    -D PKG_ML-IAP=ON \
    -D PKG_ML-SNAP=ON \
    -D MLIAP_ENABLE_PYTHON=ON \
    -D PKG_PYTHON=ON \
    -D PKG_KOKKOS=ON \
    -D Kokkos_ENABLE_CUDA=ON \
    -D "${KOKKOS_ARCH_FLAG}=ON" \
    -D PKG_MANYBODY=ON \
    -D PKG_EXTRA-COMPUTE=ON \
    -D PKG_EXTRA-FIX=ON \
    -D PKG_MISC=ON \
    -D Python_EXECUTABLE="${PYTHON_BIN}" \
    -D Python3_EXECUTABLE="${PYTHON_BIN}"

  cmake --build "${LAMMPS_BUILD_DIR}" -j "${LAMMPS_BUILD_JOBS}"
  cmake --install "${LAMMPS_BUILD_DIR}"
  cmake --build "${LAMMPS_BUILD_DIR}" --target install-python || true
}

write_runtime_env() {
  section "[5/6] Write runtime environment"

  cat > "${WORKSPACE}/cuzr_mddms_runtime.env" <<RUNTIME_ENV
export WORKSPACE="${WORKSPACE}"
export PROJECT_DIR="${PROJECT_DIR}"
export CUZR_ENV_PREFIX="${CUZR_ENV_PREFIX}"
export PYTHON_BIN="${PYTHON_BIN}"
export PIP_BIN="${PIP_BIN}"
export MACE_DIR="${MACE_DIR}"
export EAM_DIR="${EAM_DIR}"
export MACE_C_MODEL="${MACE_C_MLIAP_LINK}"
export EAM_CUZR_MODEL="${EAM_LINK}"
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
export LMP_EAM_CMD="lmp"
RUNTIME_ENV

  # shellcheck disable=SC1091
  source "${WORKSPACE}/cuzr_mddms_runtime.env"
  echo "Runtime env written: ${WORKSPACE}/cuzr_mddms_runtime.env"
}

final_check() {
  section "[6/6] Final runtime check"

  # shellcheck disable=SC1091
  source "${WORKSPACE}/cuzr_mddms_runtime.env" || true

  echo "[Python]"
  which python || true
  "${PYTHON_BIN}" - <<'PY'
import importlib.util
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
for name in ["mace", "cupy", "cuequivariance", "cuequivariance_torch"]:
    print(f"{name}:", "OK" if importlib.util.find_spec(name) else "MISSING")
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
  echo "[Models]"
  ls -lh "${MACE_DIR}" || true
  ls -lh "${EAM_DIR}" || true
}

run_tiny_test() {
  if [ "${RUN_TINY_TEST}" != "1" ]; then
    return
  fi

  section "[Optional] Run tiny MACE_C MD-DMS test"

  # shellcheck disable=SC1091
  source "${WORKSPACE}/cuzr_mddms_runtime.env"
  cd "${PROJECT_DIR}"

  "${PYTHON_BIN}" scripts/run/run_mddms_pilot.py \
    --run-name tiny_mace_c_001 \
    --preset tiny \
    --model-alias mace_c \
    --execute \
    --analyze
}

activate_python_env
print_config
check_python_stack

if [ "${CHECK_ONLY}" = "1" ]; then
  write_runtime_env
  final_check
  echo "[CuZr-MD-DMS] CHECK_ONLY=1 finished"
  exit 0
fi

fetch_models
convert_mace_c
build_lammps
write_runtime_env
final_check
run_tiny_test

echo
echo "[CuZr-MD-DMS] startup finished successfully"
echo
echo "Next MACE test:"
echo "  source ${WORKSPACE}/cuzr_mddms_runtime.env"
echo "  cd ${PROJECT_DIR}"
echo "  ${PYTHON_BIN} scripts/run/run_mddms_pilot.py --run-name tiny_mace_c_001 --preset tiny --model-alias mace_c --execute --analyze"
echo
echo "Next EAM test:"
echo "  ${PYTHON_BIN} scripts/run/run_mddms_pilot.py --run-name tiny_eam_cuzr_001 --preset tiny --model-alias eam_cuzr --execute --analyze"
