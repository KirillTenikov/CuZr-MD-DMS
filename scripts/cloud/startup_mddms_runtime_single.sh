#!/usr/bin/env bash
set -euo pipefail

# CuZr-MD-DMS single runtime startup script
#
# Purpose:
#   One manually-run startup script for the cloud/QData instance.
#
# It does four things:
#   1. checks Python / CUDA / MACE / cuEquivariance,
#   2. downloads models from the CuZr-MD-DMS GitHub release,
#   3. converts MACE_C raw model to LAMMPS ML-IAP format if needed,
#   4. builds LAMMPS with ML-IAP + Kokkos CUDA and writes runtime env.
#
# Typical use after QData instance starts:
#
#   cd /workspace/CuZr-MD-DMS
#   export LAMMPS_BUILD_JOBS=6
#   export KOKKOS_ARCH_FLAG=Kokkos_ARCH_AMPERE80
#   bash scripts/cloud/startup_mddms_runtime.sh
#
# Optional flags through environment variables:
#
#   FETCH_MODELS=0       skip model download
#   CONVERT_MACE=0       skip MACE_C conversion
#   BUILD_LAMMPS=0       skip LAMMPS build
#   CHECK_ONLY=1         only check current environment
#   RUN_TINY_TEST=1      run tiny MD-DMS test after setup
#   FORCE_DOWNLOAD=1     redownload release assets
#   FORCE_CONVERT=1      reconvert MACE_C even if ML-IAP file exists
#   FORCE_REBUILD=1      rebuild LAMMPS from scratch
#
# GPU architecture:
#   A100: Kokkos_ARCH_AMPERE80
#   H100: Kokkos_ARCH_HOPPER90

echo "[CuZr-MD-DMS] Single runtime startup started"

# -----------------------------
# Global configuration
# -----------------------------
export WORKSPACE="${WORKSPACE:-/workspace}"
export PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"

export CUZR_ENV_PREFIX="${CUZR_ENV_PREFIX:-/opt/cuzr-mamba}"
export PYTHON_BIN="${PYTHON_BIN:-${CUZR_ENV_PREFIX}/bin/python}"
export PIP_BIN="${PIP_BIN:-${CUZR_ENV_PREFIX}/bin/pip}"

export MODELS_REPO="${MODELS_REPO:-KirillTenikov/CuZr-MD-DMS}"
export MODELS_RELEASE_TAG="${MODELS_RELEASE_TAG:-Models}"

export MACE_DIR="${MACE_DIR:-${PROJECT_DIR}/models/mace}"
export EAM_DIR="${EAM_DIR:-${PROJECT_DIR}/models/eam}"
export MODELS_TMP_DIR="${MODELS_TMP_DIR:-${PROJECT_DIR}/.tmp_models_${MODELS_RELEASE_TAG}}"

export LAMMPS_DIR="${LAMMPS_DIR:-/opt/lammps}"
export LAMMPS_REF="${LAMMPS_REF:-develop}"
export LAMMPS_ROOT="${LAMMPS_ROOT:-${WORKSPACE}/lammps_mddms}"
export LAMMPS_BUILD_DIR="${LAMMPS_BUILD_DIR:-${LAMMPS_ROOT}/build-mliap-kokkos}"
export LAMMPS_INSTALL_DIR="${LAMMPS_INSTALL_DIR:-${LAMMPS_ROOT}/install-mliap-kokkos}"

export KOKKOS_ARCH_FLAG="${KOKKOS_ARCH_FLAG:-Kokkos_ARCH_AMPERE80}"
export BUILD_MPI="${BUILD_MPI:-ON}"
export LAMMPS_BUILD_JOBS="${LAMMPS_BUILD_JOBS:-$(nproc)}"

export FETCH_MODELS="${FETCH_MODELS:-1}"
export CONVERT_MACE="${CONVERT_MACE:-1}"
export BUILD_LAMMPS="${BUILD_LAMMPS:-1}"
export CHECK_ONLY="${CHECK_ONLY:-0}"
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

echo "[Info] Log file: ${LOG_FILE}"
echo "[Info] WORKSPACE=${WORKSPACE}"
echo "[Info] PROJECT_DIR=${PROJECT_DIR}"
echo "[Info] CUZR_ENV_PREFIX=${CUZR_ENV_PREFIX}"
echo "[Info] PYTHON_BIN=${PYTHON_BIN}"
echo "[Info] MODELS_REPO=${MODELS_REPO}"
echo "[Info] MODELS_RELEASE_TAG=${MODELS_RELEASE_TAG}"
echo "[Info] LAMMPS_DIR=${LAMMPS_DIR}"
echo "[Info] LAMMPS_BUILD_DIR=${LAMMPS_BUILD_DIR}"
echo "[Info] LAMMPS_INSTALL_DIR=${LAMMPS_INSTALL_DIR}"
echo "[Info] KOKKOS_ARCH_FLAG=${KOKKOS_ARCH_FLAG}"
echo "[Info] BUILD_MPI=${BUILD_MPI}"
echo "[Info] LAMMPS_BUILD_JOBS=${LAMMPS_BUILD_JOBS}"
echo "[Info] FETCH_MODELS=${FETCH_MODELS}"
echo "[Info] CONVERT_MACE=${CONVERT_MACE}"
echo "[Info] BUILD_LAMMPS=${BUILD_LAMMPS}"
echo "[Info] CHECK_ONLY=${CHECK_ONLY}"
echo "[Info] RUN_TINY_TEST=${RUN_TINY_TEST}"
echo

# -----------------------------
# Helpers
# -----------------------------
section() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

ensure_python_env() {
  if [ -f /opt/cuzr_python_prebuilt.env ]; then
    # Existing CuZr MD image helper, if present.
    source /opt/cuzr_python_prebuilt.env
  fi
  if [ -f /etc/profile.d/cuzr-env.sh ]; then
    source /etc/profile.d/cuzr-env.sh
  fi

  export PATH="${CUZR_ENV_PREFIX}/bin:${PATH}"

  if [ ! -x "${PYTHON_BIN}" ]; then
    echo "WARNING: ${PYTHON_BIN} not found. Falling back to python from PATH."
    export PYTHON_BIN="$(command -v python)"
    export PIP_BIN="$(command -v pip)"
  fi
}

check_python_stack() {
  section "[1/6] GPU / Python / MACE / cuEquivariance check"

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
      echo "[Info] cuEquivariance stack missing; installing into ${CUZR_ENV_PREFIX}"
      "${PIP_BIN}" install --no-cache-dir \
        "cupy-cuda12x==${CUPY_VERSION}" \
        "cuequivariance==${CUEQ_VERSION}" \
        "cuequivariance-torch==${CUEQ_VERSION}" \
        "cuequivariance-ops-torch-cu12==${CUEQ_VERSION}"
    else
      echo "[Info] cuEquivariance stack already available."
    fi
  fi

  "${PYTHON_BIN}" - <<'PY'
import torch
import mace
print("Final Python check:")
print("  mace:", mace.__file__)
print("  cuda available:", torch.cuda.is_available())
try:
    import cupy
    import cuequivariance
    import cuequivariance_torch
    print("  cupy:", cupy.__version__)
    print("  cuequivariance: OK")
except Exception as exc:
    print("  cuequivariance import failed:", repr(exc))
    raise
PY
}

fetch_models() {
  section "[2/6] Fetch models from GitHub release"

  if [ "${FETCH_MODELS}" != "1" ]; then
    echo "[Info] FETCH_MODELS=0, skipping model download."
    return
  fi

  if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl is required."
    exit 1
  fi

  mkdir -p "${MODELS_TMP_DIR}" "${MACE_DIR}" "${EAM_DIR}"

  ASSETS_JSON="${MODELS_TMP_DIR}/release.json"
  URLS_FILE="${MODELS_TMP_DIR}/download_urls.tsv"

  echo "[Info] Querying release: https://github.com/${MODELS_REPO}/releases/tag/${MODELS_RELEASE_TAG}"
  curl -fsSL "https://api.github.com/repos/${MODELS_REPO}/releases/tags/${MODELS_RELEASE_TAG}" -o "${ASSETS_JSON}"

  "${PYTHON_BIN}" - "${ASSETS_JSON}" "${URLS_FILE}" <<'PY'
import json
import sys
from pathlib import Path

assets_json = Path(sys.argv[1])
urls_file = Path(sys.argv[2])

data = json.loads(assets_json.read_text())
assets = data.get("assets", [])

rows = []
for a in assets:
    name = a.get("name", "")
    url = a.get("browser_download_url", "")
    size = a.get("size", 0)
    if name and url:
        rows.append((name, url, size))

urls_file.write_text(
    "\n".join(f"{name}\t{url}\t{size}" for name, url, size in rows) + ("\n" if rows else "")
)

print(f"Found {len(rows)} release assets:")
for name, _, size in rows:
    print(f"  {name} ({size/1024/1024:.2f} MB)")
PY

  while IFS="$(printf '\t')" read -r name url size; do
    [ -n "${name:-}" ] || continue
    dest="${MODELS_TMP_DIR}/${name}"

    if [ -f "${dest}" ] && [ "${FORCE_DOWNLOAD}" != "1" ]; then
      echo "Already present: ${name}"
    else
      echo "Downloading: ${name}"
      curl -fL --retry 3 --retry-delay 5 -o "${dest}" "${url}"
    fi
  done < "${URLS_FILE}"

  if [ -f "${MODELS_TMP_DIR}/SHA256SUMS.txt" ]; then
    echo "[Info] Verifying SHA256SUMS.txt"
    (cd "${MODELS_TMP_DIR}" && sha256sum -c SHA256SUMS.txt)
  else
    echo "[Info] No SHA256SUMS.txt found; skipping checksum verification."
  fi

  echo "[Info] Linking model files into runtime directories"

  find "${MODELS_TMP_DIR}" -maxdepth 1 -type f \( -iname "*.pt" -o -iname "*.model" -o -iname "*.pth" \) -print0 |
  while IFS= read -r -d '' f; do
    ln -sf "${f}" "${MACE_DIR}/$(basename "${f}")"
  done

  find "${MODELS_TMP_DIR}" -maxdepth 1 -type f \( -iname "*.alloy" -o -iname "*.eam" -o -iname "*.fs" -o -iname "*.setfl" \) -print0 |
  while IFS= read -r -d '' f; do
    ln -sf "${f}" "${EAM_DIR}/$(basename "${f}")"
  done

  "${PYTHON_BIN}" - "${MACE_DIR}" "${EAM_DIR}" <<'PY'
from pathlib import Path
import re
import sys

mace_dir = Path(sys.argv[1])
eam_dir = Path(sys.argv[2])

def symlink_to_name(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.name)
    print(f"  {dst.name} -> {src.name}")

print("Convenience symlinks:")

mace_files = [p for p in mace_dir.iterdir() if p.exists() or p.is_symlink()]
eam_files = [p for p in eam_dir.iterdir() if p.exists() or p.is_symlink()]

mace_c_mliap = [
    p for p in mace_files
    if p.suffix.lower() == ".pt"
    and "mliap" in p.name.lower()
    and re.search(r"mace[_-]?c|mace.*_c_|mace.*c", p.name.lower())
]
if mace_c_mliap:
    symlink_to_name(sorted(mace_c_mliap, key=lambda p: len(p.name))[0],
                    mace_dir / "mace_C.model-mliap_lammps.pt")
else:
    print("  No recognizable MACE_C ML-IAP .pt found yet.")

mace_c_raw = [
    p for p in mace_files
    if p.suffix.lower() == ".model"
    and re.search(r"mace[_-]?c|mace.*_c_|mace.*c", p.name.lower())
]
if mace_c_raw:
    symlink_to_name(sorted(mace_c_raw, key=lambda p: len(p.name))[0],
                    mace_dir / "mace_C_raw.model")
else:
    print("  No recognizable raw MACE_C .model found.")

eam_2019 = [
    p for p in eam_files
    if p.suffix.lower() in [".alloy", ".eam", ".fs", ".setfl"]
    and any(x in p.name.lower() for x in ["2019", "mendelev"])
]
if eam_2019:
    eam_2019 = sorted(eam_2019, key=lambda p: (("2019" not in p.name.lower()), len(p.name)))
    symlink_to_name(eam_2019[0], eam_dir / "eam_mendelev_2019.eam.alloy")
else:
    print("  No recognizable EAM 2019 file found.")
PY

  echo "[Info] MACE directory:"
  ls -lh "${MACE_DIR}" || true
  echo "[Info] EAM directory:"
  ls -lh "${EAM_DIR}" || true
}

convert_mace_c() {
  section "[3/6] Convert MACE_C to LAMMPS ML-IAP format"

  if [ "${CONVERT_MACE}" != "1" ]; then
    echo "[Info] CONVERT_MACE=0, skipping conversion."
    return
  fi

  MLIAP_LINK="${MACE_DIR}/mace_C.model-mliap_lammps.pt"
  RAW_MODEL="${RAW_MODEL:-${MACE_DIR}/mace_C_raw.model}"

  if [ -e "${MLIAP_LINK}" ] && [ "${FORCE_CONVERT}" != "1" ]; then
    echo "[Info] ML-IAP model already exists:"
    ls -lh "${MLIAP_LINK}"
    return
  fi

  if [ ! -f "${RAW_MODEL}" ]; then
    echo "WARNING: raw MACE_C model not found:"
    echo "  ${RAW_MODEL}"
    echo "Skipping conversion. If the release already contains ML-IAP .pt, this is okay."
    return
  fi

  echo "[Info] Converting raw model:"
  ls -lh "${RAW_MODEL}"

  "${PYTHON_BIN}" - <<'PY'
import torch
print("CUDA visible for conversion:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: MACE docs recommend ML-IAP conversion on GPU.")
PY

  "${PYTHON_BIN}" -m mace.cli.create_lammps_model "${RAW_MODEL}" --format=mliap

  EXPECTED="${RAW_MODEL}-mliap_lammps.pt"
  if [ -f "${EXPECTED}" ]; then
    CONVERTED="${EXPECTED}"
  else
    CONVERTED="$(find "${MACE_DIR}" -maxdepth 1 -type f -name "*mliap_lammps.pt" -print0 \
      | xargs -0 ls -t 2>/dev/null \
      | head -n 1 || true)"
  fi

  if [ -z "${CONVERTED}" ] || [ ! -f "${CONVERTED}" ]; then
    echo "ERROR: conversion finished, but ML-IAP file was not found."
    ls -lh "${MACE_DIR}"
    exit 1
  fi

  rm -f "${MLIAP_LINK}"
  ln -s "$(basename "${CONVERTED}")" "${MLIAP_LINK}"

  echo "[Info] Stable ML-IAP model:"
  ls -lh "${MLIAP_LINK}"
}

build_lammps() {
  section "[4/6] Build LAMMPS with ML-IAP + Kokkos CUDA"

  if [ "${BUILD_LAMMPS}" != "1" ]; then
    echo "[Info] BUILD_LAMMPS=0, skipping LAMMPS build."
    return
  fi

  if [ -d "${LAMMPS_DIR}" ]; then
    echo "[Info] Using existing LAMMPS source: ${LAMMPS_DIR}"
    if [ -d "${LAMMPS_DIR}/.git" ]; then
      cd "${LAMMPS_DIR}"
      git fetch --all --tags
      git checkout "${LAMMPS_REF}"
    else
      echo "[Info] LAMMPS source is not a git checkout; using it as-is."
    fi
  else
    echo "[Info] Cloning LAMMPS ${LAMMPS_REF} into ${LAMMPS_DIR}"
    git clone --branch "${LAMMPS_REF}" https://github.com/lammps/lammps.git "${LAMMPS_DIR}"
  fi

  if [ ! -d "${LAMMPS_DIR}/cmake" ]; then
    echo "ERROR: ${LAMMPS_DIR}/cmake not found. Invalid LAMMPS source."
    exit 1
  fi

  if [ "${FORCE_REBUILD}" = "1" ]; then
    echo "[Info] FORCE_REBUILD=1; removing old build/install dirs."
    rm -rf "${LAMMPS_BUILD_DIR}" "${LAMMPS_INSTALL_DIR}"
  fi

  if [ -x "${LAMMPS_INSTALL_DIR}/bin/lmp" ]; then
    echo "[Info] Existing LAMMPS binary found:"
    "${LAMMPS_INSTALL_DIR}/bin/lmp" -h | head -n 20 || true
    return
  fi

  KOKKOS_WRAPPER="${LAMMPS_DIR}/lib/kokkos/bin/nvcc_wrapper"
  if [ ! -x "${KOKKOS_WRAPPER}" ]; then
    echo "ERROR: Kokkos nvcc_wrapper not found:"
    echo "  ${KOKKOS_WRAPPER}"
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
    -D CMAKE_CXX_COMPILER="${KOKKOS_WRAPPER}" \
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

  echo "[Info] Runtime env:"
  echo "  ${WORKSPACE}/cuzr_mddms_runtime.env"
}

final_check() {
  section "[6/6] Final runtime check"

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

  if [ -e "${MACE_DIR}/mace_C.model-mliap_lammps.pt" ]; then
    echo "[OK] MACE_C ML-IAP model is available."
  else
    echo "[WARNING] MACE_C ML-IAP model is missing."
  fi
}

run_tiny_test() {
  if [ "${RUN_TINY_TEST}" != "1" ]; then
    return
  fi

  section "[Optional] Run tiny MD-DMS test"

  source "${WORKSPACE}/cuzr_mddms_runtime.env"

  cd "${PROJECT_DIR}"

  "${PYTHON_BIN}" scripts/run/run_mddms_pilot.py \
    --run-name tiny_mace_c_001 \
    --preset tiny \
    --model-kind mace \
    --model-file "${MACE_DIR}/mace_C.model-mliap_lammps.pt" \
    --execute \
    --analyze
}

# -----------------------------
# Main
# -----------------------------
ensure_python_env
check_python_stack

if [ "${CHECK_ONLY}" = "1" ]; then
  write_runtime_env
  final_check
  echo "[CuZr-MD-DMS] CHECK_ONLY=1 finished."
  exit 0
fi

fetch_models
convert_mace_c
build_lammps
write_runtime_env
final_check
run_tiny_test

echo
echo "[CuZr-MD-DMS] Startup finished successfully."
echo
echo "Next:"
echo "  source ${WORKSPACE}/cuzr_mddms_runtime.env"
echo "  cd ${PROJECT_DIR}"
echo "  ${PYTHON_BIN} scripts/run/run_mddms_pilot.py \\"
echo "    --run-name tiny_mace_c_001 \\"
echo "    --preset tiny \\"
echo "    --model-kind mace \\"
echo "    --model-file ${MACE_DIR}/mace_C.model-mliap_lammps.pt \\"
echo "    --execute --analyze"
