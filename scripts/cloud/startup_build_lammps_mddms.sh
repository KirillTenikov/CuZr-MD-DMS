#!/usr/bin/env bash
set -euo pipefail

# CuZr-MD-DMS cloud startup script
#
# Purpose:
#   Build LAMMPS on the actual cloud GPU machine.
#
# Important:
#   This script is intended to run inside the CuZr-MD-DMS Docker base image.
#   The Docker image does not contain prebuilt LAMMPS.
#
# Default target:
#   NVIDIA A100, Kokkos_ARCH_AMPERE80.
#
# For H100 later:
#   export KOKKOS_ARCH_FLAG=Kokkos_ARCH_HOPPER90
#
# To force clean rebuild:
#   export FORCE_REBUILD=1
#   bash scripts/cloud/startup_build_lammps_mddms.sh

echo "[CuZr-MD-DMS] Starting cloud LAMMPS build"

# Activate Python environment from the Docker base image.
source /opt/venv/bin/activate || true

export WORKSPACE="${WORKSPACE:-/workspace}"
export LAMMPS_DIR="${LAMMPS_DIR:-/opt/lammps}"
export LAMMPS_REF="${LAMMPS_REF:-develop}"
export LAMMPS_BUILD_DIR="${LAMMPS_BUILD_DIR:-/opt/lammps/build-mliap-kokkos}"
export LAMMPS_INSTALL_DIR="${LAMMPS_INSTALL_DIR:-/opt/lammps/install-mliap-kokkos}"

# A100 default. Change to Kokkos_ARCH_HOPPER90 for H100.
export KOKKOS_ARCH_FLAG="${KOKKOS_ARCH_FLAG:-Kokkos_ARCH_AMPERE80}"

# For the first Paper 2 pilots, single-GPU non-MPI is simpler.
# Later, test BUILD_MPI=ON separately.
export BUILD_MPI="${BUILD_MPI:-OFF}"

export LAMMPS_BUILD_JOBS="${LAMMPS_BUILD_JOBS:-$(nproc)}"
export FORCE_REBUILD="${FORCE_REBUILD:-0}"

mkdir -p "$WORKSPACE" "$LAMMPS_INSTALL_DIR"

LOG_DIR="${WORKSPACE}/logs/runtime"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/build_lammps_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[Info] Log file: $LOG_FILE"
echo "[Info] WORKSPACE: $WORKSPACE"
echo "[Info] LAMMPS_DIR: $LAMMPS_DIR"
echo "[Info] LAMMPS_REF: $LAMMPS_REF"
echo "[Info] LAMMPS_BUILD_DIR: $LAMMPS_BUILD_DIR"
echo "[Info] LAMMPS_INSTALL_DIR: $LAMMPS_INSTALL_DIR"
echo "[Info] KOKKOS_ARCH_FLAG: $KOKKOS_ARCH_FLAG"
echo "[Info] BUILD_MPI: $BUILD_MPI"
echo "[Info] LAMMPS_BUILD_JOBS: $LAMMPS_BUILD_JOBS"
echo

echo "[1/7] GPU and Python check"
nvidia-smi || true

python - <<'PY'
import torch
import mace

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
print("mace:", mace.__file__)
PY

echo
echo "[2/7] cuEquivariance import check"
python - <<'PY'
try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet
    print("cuequivariance: OK")
    print("cuequivariance_torch:", cuet.__file__)
except Exception as exc:
    print("cuequivariance check failed:", repr(exc))
    print("LAMMPS build can continue, but MACE acceleration may be unavailable.")
PY

echo
echo "[3/7] Get LAMMPS source"
if [ ! -d "$LAMMPS_DIR/.git" ]; then
  git clone https://github.com/lammps/lammps.git "$LAMMPS_DIR"
fi

cd "$LAMMPS_DIR"
git fetch --all --tags
git checkout "$LAMMPS_REF"

echo
echo "[4/7] Check existing LAMMPS build"
if [ "$FORCE_REBUILD" = "1" ]; then
  echo "[Info] FORCE_REBUILD=1, removing old build/install directories"
  rm -rf "$LAMMPS_BUILD_DIR" "$LAMMPS_INSTALL_DIR"
fi

if [ -x "$LAMMPS_INSTALL_DIR/bin/lmp" ]; then
  echo "[Info] LAMMPS already exists:"
  "$LAMMPS_INSTALL_DIR/bin/lmp" -h | head -n 20 || true
else
  echo "[Info] LAMMPS not found. Building now."

  KOKKOS_WRAPPER="$LAMMPS_DIR/lib/kokkos/bin/nvcc_wrapper"

  if [ ! -x "$KOKKOS_WRAPPER" ]; then
    echo "ERROR: Kokkos nvcc_wrapper not found at:"
    echo "  $KOKKOS_WRAPPER"
    exit 1
  fi

  if [ "$BUILD_MPI" = "ON" ]; then
    export NVCC_WRAPPER_DEFAULT_COMPILER="$(command -v mpicxx)"
  else
    export NVCC_WRAPPER_DEFAULT_COMPILER="$(command -v g++)"
  fi

  echo
  echo "[5/7] Configure LAMMPS"
  cmake -S "$LAMMPS_DIR/cmake" -B "$LAMMPS_BUILD_DIR" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX="$LAMMPS_INSTALL_DIR" \
    -D CMAKE_CXX_COMPILER="$KOKKOS_WRAPPER" \
    -D CMAKE_CXX_STANDARD=17 \
    -D BUILD_MPI="$BUILD_MPI" \
    -D BUILD_SHARED_LIBS=ON \
    -D PKG_ML-IAP=ON \
    -D MLIAP_ENABLE_PYTHON=ON \
    -D PKG_PYTHON=ON \
    -D PKG_KOKKOS=ON \
    -D Kokkos_ENABLE_CUDA=ON \
    -D "${KOKKOS_ARCH_FLAG}=ON" \
    -D PKG_MANYBODY=ON \
    -D PKG_VORONOI=ON \
    -D PKG_EXTRA-COMPUTE=ON \
    -D PKG_EXTRA-FIX=ON \
    -D PKG_MISC=ON \
    -D Python_EXECUTABLE="$(command -v python)" \
    -D Python3_EXECUTABLE="$(command -v python)"

  echo
  echo "[6/7] Build and install LAMMPS"
  cmake --build "$LAMMPS_BUILD_DIR" -j "$LAMMPS_BUILD_JOBS"
  cmake --install "$LAMMPS_BUILD_DIR"
fi

echo
echo "[7/7] Write runtime environment"
cat > "$WORKSPACE/cuzr_mddms_runtime.env" <<RUNTIME_ENV
export LAMMPS_DIR="$LAMMPS_DIR"
export LAMMPS_BUILD_DIR="$LAMMPS_BUILD_DIR"
export LAMMPS_INSTALL_DIR="$LAMMPS_INSTALL_DIR"
export PATH="$LAMMPS_INSTALL_DIR/bin:\$PATH"
export LD_LIBRARY_PATH="$LAMMPS_INSTALL_DIR/lib:\${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$LAMMPS_INSTALL_DIR/lib/python:\${PYTHONPATH:-}"
RUNTIME_ENV

source "$WORKSPACE/cuzr_mddms_runtime.env"

echo
echo "[CuZr-MD-DMS] Runtime check"
which lmp
lmp -h | grep -E "ML-IAP|KOKKOS|PYTHON|MANYBODY|VORONOI" || true

echo
echo "[CuZr-MD-DMS] Done"
echo
echo "To activate this runtime later:"
echo "  source $WORKSPACE/cuzr_mddms_runtime.env"
