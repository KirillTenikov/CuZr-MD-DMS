#!/usr/bin/env bash
set -Eeuo pipefail

# CuZr-MD-DMS runtime startup script, safe production/branching version.
# Goals:
#   - prepare Python/MACE/LAMMPS runtime;
#   - avoid known MACE/Kokkos atom_stress crash by default;
#   - keep LAMMPS source/build stable unless explicitly requested;
#   - write helper commands for full safe runs, stage03-only branches, checks, and archives.

trap 'echo "[ERROR] line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

echo "[CuZr-MD-DMS] runtime startup started"

# -----------------------------
# Configuration
# -----------------------------
export WORKSPACE="${WORKSPACE:-/workspace}"
export PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
export REPO_URL="${REPO_URL:-https://github.com/KirillTenikov/CuZr-MD-DMS.git}"
export AUTO_CLONE_PROJECT="${AUTO_CLONE_PROJECT:-0}"

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
export LAMMPS_UPDATE_SOURCE="${LAMMPS_UPDATE_SOURCE:-0}"
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

# Production/branching policy.
export MD_DMS_RUN_ROOT="${MD_DMS_RUN_ROOT:-${WORKSPACE}/outputs/md_dms}"
export SAFE_NO_ATOMIC_STRESS="${SAFE_NO_ATOMIC_STRESS:-1}"
export SAFE_DUMP_TRAJECTORY="${SAFE_DUMP_TRAJECTORY:-1}"
export SAFE_DUMP_EVERY_STEPS="${SAFE_DUMP_EVERY_STEPS:-1000}"
export SAFE_STRESS_EVERY_STEPS="${SAFE_STRESS_EVERY_STEPS:-10}"
export SAFE_PRESET="${SAFE_PRESET:-pressure_relaxed}"
export SAFE_NATOMS="${SAFE_NATOMS:-4000}"
export SAFE_TEMPERATURE_LOW_K="${SAFE_TEMPERATURE_LOW_K:-300}"
export SAFE_STRAIN_AMPLITUDE="${SAFE_STRAIN_AMPLITUDE:-0.01}"
export SAFE_PERIOD_PS="${SAFE_PERIOD_PS:-50}"
export SAFE_CYCLES="${SAFE_CYCLES:-6}"

mkdir -p "${WORKSPACE}" "${PROJECT_DIR}" "${MACE_DIR}" "${EAM_DIR}" "${LAMMPS_ROOT}" "${MD_DMS_RUN_ROOT}"
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

die() {
  echo "ERROR: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

normalize_kokkos_arch_flag() {
  # Kokkos CMake options are case-sensitive. Normalize common typos
  # such as KokkOS_ARCH_HOPPER90 or KOKKOS_ARCH_HOPPER90 to the
  # exact spelling Kokkos_ARCH_HOPPER90 expected by Kokkos.
  case "${KOKKOS_ARCH_FLAG}" in
    KokkOS_ARCH_*)
      KOKKOS_ARCH_FLAG="Kokkos_ARCH_${KOKKOS_ARCH_FLAG#KokkOS_ARCH_}"
      ;;
    KOKKOS_ARCH_*)
      KOKKOS_ARCH_FLAG="Kokkos_ARCH_${KOKKOS_ARCH_FLAG#KOKKOS_ARCH_}"
      ;;
  esac

  case "${KOKKOS_ARCH_FLAG}" in
    Kokkos_ARCH_*)
      ;;
    *)
      die "KOKKOS_ARCH_FLAG must look like Kokkos_ARCH_AMPERE80 or Kokkos_ARCH_HOPPER90; got: ${KOKKOS_ARCH_FLAG}"
      ;;
  esac
  export KOKKOS_ARCH_FLAG
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

  # Prefer the mamba/conda runtime libraries over older system libraries.
  # This prevents scipy/e3nn/mace from loading /usr/lib/.../libstdc++.so.6
  # without newer CXXABI symbols required by compiled Python extensions.
  export LD_LIBRARY_PATH="${CUZR_ENV_PREFIX}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${CUZR_ENV_PREFIX}/lib:${LIBRARY_PATH:-}"
  export PYTHONNOUSERSITE=1
  hash -r

  if [ ! -x "${PYTHON_BIN}" ]; then
    echo "[warning] ${PYTHON_BIN} not found, falling back to python from PATH"
    export PYTHON_BIN="$(command -v python || true)"
    export PIP_BIN="$(command -v pip || true)"
  fi
  [ -n "${PYTHON_BIN}" ] && [ -x "${PYTHON_BIN}" ] || die "no usable python found"
  [ -n "${PIP_BIN}" ] || export PIP_BIN="$(command -v pip || true)"
}

ensure_project_dir() {
  section "[0a/7] Check project directory"
  if [ -f "${PROJECT_DIR}/scripts/run/run_mddms_pilot.py" ]; then
    echo "[info] project found: ${PROJECT_DIR}"
    return
  fi

  if [ "${AUTO_CLONE_PROJECT}" = "1" ]; then
    need_cmd git
    echo "[info] project not found; cloning ${REPO_URL} -> ${PROJECT_DIR}"
    rm -rf "${PROJECT_DIR}"
    git clone "${REPO_URL}" "${PROJECT_DIR}"
    return
  fi

  die "PROJECT_DIR does not look like CuZr-MD-DMS: ${PROJECT_DIR}. cd to repo or set AUTO_CLONE_PROJECT=1."
}

print_config() {
  section "[0b/7] Configuration"
  cat <<CFG
WORKSPACE=${WORKSPACE}
PROJECT_DIR=${PROJECT_DIR}
PYTHON_BIN=${PYTHON_BIN}
MODELS_BASE_URL=${MODELS_BASE_URL}
MACE_C_ASSET=${MACE_C_ASSET}
EAM_ASSET=${EAM_ASSET}
LAMMPS_DIR=${LAMMPS_DIR}
LAMMPS_REF=${LAMMPS_REF}
LAMMPS_UPDATE_SOURCE=${LAMMPS_UPDATE_SOURCE}
LAMMPS_BUILD_DIR=${LAMMPS_BUILD_DIR}
LAMMPS_INSTALL_DIR=${LAMMPS_INSTALL_DIR}
KOKKOS_ARCH_FLAG=${KOKKOS_ARCH_FLAG}
BUILD_MPI=${BUILD_MPI}
LAMMPS_BUILD_JOBS=${LAMMPS_BUILD_JOBS}
FETCH_MODELS=${FETCH_MODELS}
CONVERT_MACE=${CONVERT_MACE}
BUILD_LAMMPS=${BUILD_LAMMPS}
RUN_TINY_TEST=${RUN_TINY_TEST}
MD_DMS_RUN_ROOT=${MD_DMS_RUN_ROOT}
SAFE_NO_ATOMIC_STRESS=${SAFE_NO_ATOMIC_STRESS}
SAFE_DUMP_TRAJECTORY=${SAFE_DUMP_TRAJECTORY}
SAFE_DUMP_EVERY_STEPS=${SAFE_DUMP_EVERY_STEPS}
SAFE_STRESS_EVERY_STEPS=${SAFE_STRESS_EVERY_STEPS}
SAFE_PRESET=${SAFE_PRESET}
LOG_FILE=${LOG_FILE}
CFG
}

check_python_stack() {
  section "[1/7] Check GPU / Python / MACE / cuEquivariance"
  nvidia-smi || true

  "${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch import/check failed:", repr(exc))
    sys.exit(1)
# Import scipy submodules explicitly because scipy.signal/fft pull compiled
# extensions that expose libstdc++/CXXABI problems early.
try:
    import scipy
    import scipy.fft
    import scipy.signal
    print("scipy import check: OK", scipy.__version__)
except Exception as exc:
    print("scipy import check failed:", repr(exc))
    sys.exit(1)

for name in ["numpy", "mace", "cupy", "cuequivariance", "cuequivariance_torch"]:
    spec = importlib.util.find_spec(name)
    print(f"{name}:", "OK" if spec else "MISSING")
PY

  if [ "${INSTALL_CUEQ_IF_MISSING}" = "1" ]; then
    if ! "${PYTHON_BIN}" - <<'PY'
import importlib.util
missing = [name for name in ["cupy", "cuequivariance", "cuequivariance_torch"] if importlib.util.find_spec(name) is None]
raise SystemExit(1 if missing else 0)
PY
    then
      [ -n "${PIP_BIN}" ] || die "pip is missing; cannot install cuEquivariance stack"
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

fetch_one() {
  local url="$1"
  local dest="$2"
  local tmp="${dest}.tmp"

  if [ -f "${dest}" ] && [ "${FORCE_DOWNLOAD}" != "1" ]; then
    echo "[info] already present: ${dest}"
    return
  fi

  echo "[download] ${url} -> ${dest}"
  rm -f "${tmp}"
  curl -fL --retry 5 --retry-delay 5 --connect-timeout 20 -o "${tmp}" "${url}"
  [ -s "${tmp}" ] || die "download produced empty file: ${tmp}"
  mv -f "${tmp}" "${dest}"
}

fetch_models() {
  section "[2/7] Fetch model files"
  if [ "${FETCH_MODELS}" != "1" ]; then
    echo "[info] FETCH_MODELS=0, skipping"
    return
  fi
  need_cmd curl

  mkdir -p "${MACE_DIR}" "${EAM_DIR}"
  fetch_one "${MODELS_BASE_URL}/${MACE_C_ASSET}" "${MACE_C_RAW}"
  fetch_one "${MODELS_BASE_URL}/${EAM_ASSET}" "${EAM_FILE}"

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
  section "[3/7] Convert MACE_C to LAMMPS ML-IAP format"
  if [ "${CONVERT_MACE}" != "1" ]; then
    echo "[info] CONVERT_MACE=0, skipping"
    return
  fi

  if [ -e "${MACE_C_MLIAP_LINK}" ] && [ "${FORCE_CONVERT}" != "1" ]; then
    echo "[info] converted ML-IAP model already exists:"
    ls -lh "${MACE_C_MLIAP_LINK}"
    return
  fi

  [ -f "${MACE_C_RAW}" ] || die "raw MACE_C model is missing: ${MACE_C_RAW}"

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

  [ -n "${converted}" ] && [ -f "${converted}" ] || {
    ls -lh "${MACE_DIR}"
    die "conversion finished, but no *mliap_lammps.pt file was found"
  }

  rm -f "${MACE_C_MLIAP_LINK}"
  ln -s "$(basename "${converted}")" "${MACE_C_MLIAP_LINK}"
  echo "Stable ML-IAP model:"
  ls -lh "${MACE_C_MLIAP_LINK}"
}

build_lammps() {
  section "[4/7] Build LAMMPS with ML-IAP + Kokkos CUDA"
  if [ "${BUILD_LAMMPS}" != "1" ]; then
    echo "[info] BUILD_LAMMPS=0, skipping"
    return
  fi

  if [ "${FORCE_REBUILD}" = "1" ]; then
    echo "[info] FORCE_REBUILD=1; removing old build/install dirs"
    rm -rf "${LAMMPS_BUILD_DIR}" "${LAMMPS_INSTALL_DIR}"
  fi

  if [ -x "${LAMMPS_INSTALL_DIR}/bin/lmp" ]; then
    echo "[info] existing LAMMPS binary found:"
    "${LAMMPS_INSTALL_DIR}/bin/lmp" -h | head -n 25 || true
    return
  fi

  need_cmd git
  need_cmd cmake

  if [ -d "${LAMMPS_DIR}" ]; then
    echo "[info] using LAMMPS source: ${LAMMPS_DIR}"
    if [ -d "${LAMMPS_DIR}/.git" ] && [ "${LAMMPS_UPDATE_SOURCE}" = "1" ]; then
      echo "[info] LAMMPS_UPDATE_SOURCE=1; updating checkout to ${LAMMPS_REF}"
      git -C "${LAMMPS_DIR}" fetch --all --tags
      git -C "${LAMMPS_DIR}" checkout "${LAMMPS_REF}"
    else
      echo "[info] not updating existing LAMMPS source; set LAMMPS_UPDATE_SOURCE=1 to change it"
    fi
  else
    echo "[info] cloning LAMMPS ${LAMMPS_REF} -> ${LAMMPS_DIR}"
    git clone --branch "${LAMMPS_REF}" https://github.com/lammps/lammps.git "${LAMMPS_DIR}"
  fi

  [ -d "${LAMMPS_DIR}/cmake" ] || die "invalid LAMMPS source; ${LAMMPS_DIR}/cmake not found"

  local kokkos_wrapper="${LAMMPS_DIR}/lib/kokkos/bin/nvcc_wrapper"
  [ -x "${kokkos_wrapper}" ] || die "Kokkos nvcc_wrapper not found: ${kokkos_wrapper}"

  if [ "${BUILD_MPI}" = "ON" ]; then
    need_cmd mpicxx
    export NVCC_WRAPPER_DEFAULT_COMPILER="$(command -v mpicxx)"
  else
    need_cmd g++
    export NVCC_WRAPPER_DEFAULT_COMPILER="$(command -v g++)"
  fi

  normalize_kokkos_arch_flag

  # Kokkos enforces exact CMake option case. If a previous attempt cached
  # KokkOS_ARCH_* or KOKKOS_ARCH_* instead of Kokkos_ARCH_*, delete the
  # build directory so the next configure starts cleanly.
  if [ -f "${LAMMPS_BUILD_DIR}/CMakeCache.txt" ]; then
    if grep -qE 'KokkOS_ARCH_|KOKKOS_ARCH_' "${LAMMPS_BUILD_DIR}/CMakeCache.txt"; then
      echo "[warning] stale/wrong-case Kokkos arch option found in CMakeCache; removing ${LAMMPS_BUILD_DIR}"
      rm -rf "${LAMMPS_BUILD_DIR}"
    fi
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
  section "[5/7] Write runtime environment"
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
export MD_DMS_RUN_ROOT="${MD_DMS_RUN_ROOT}"
export PATH="${LAMMPS_INSTALL_DIR}/bin:${CUZR_ENV_PREFIX}/bin:\$PATH"
export LD_LIBRARY_PATH="${LAMMPS_INSTALL_DIR}/lib:${CUZR_ENV_PREFIX}/lib:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CUZR_ENV_PREFIX}/lib:\${LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${LAMMPS_INSTALL_DIR}/lib/python:\${PYTHONPATH:-}"
export OMP_NUM_THREADS="\${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="\${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="\${OPENBLAS_NUM_THREADS:-1}"
export MACE_TIME="\${MACE_TIME:-false}"
export MACE_PROFILE="\${MACE_PROFILE:-false}"
export LMP_MACE_KOKKOS_CMD="lmp -k on g 1 -sf kk -pk kokkos newton on neigh half"
export LMP_EAM_CMD="lmp"
export SAFE_NO_ATOMIC_STRESS="${SAFE_NO_ATOMIC_STRESS}"
export SAFE_DUMP_TRAJECTORY="${SAFE_DUMP_TRAJECTORY}"
export SAFE_DUMP_EVERY_STEPS="${SAFE_DUMP_EVERY_STEPS}"
export SAFE_STRESS_EVERY_STEPS="${SAFE_STRESS_EVERY_STEPS}"
export SAFE_PRESET="${SAFE_PRESET}"
export SAFE_NATOMS="${SAFE_NATOMS}"
export SAFE_TEMPERATURE_LOW_K="${SAFE_TEMPERATURE_LOW_K}"
export SAFE_STRAIN_AMPLITUDE="${SAFE_STRAIN_AMPLITUDE}"
RUNTIME_ENV

  # shellcheck disable=SC1091
  source "${WORKSPACE}/cuzr_mddms_runtime.env"
  echo "Runtime env written: ${WORKSPACE}/cuzr_mddms_runtime.env"
}

write_helper_scripts() {
  section "[6/7] Write safe production/branch helper scripts"
  local bin_dir="${WORKSPACE}/bin"
  mkdir -p "${bin_dir}"

  cat > "${bin_dir}/mddms_full_run_safe.sh" <<'EOS'
#!/usr/bin/env bash
set -Eeuo pipefail
source /workspace/cuzr_mddms_runtime.env

RUN_NAME="${1:?usage: mddms_full_run_safe.sh RUN_NAME SEED PERIOD_PS CYCLES}"
SEED="${2:?usage: mddms_full_run_safe.sh RUN_NAME SEED PERIOD_PS CYCLES}"
PERIOD_PS="${3:?usage: mddms_full_run_safe.sh RUN_NAME SEED PERIOD_PS CYCLES}"
CYCLES="${4:?usage: mddms_full_run_safe.sh RUN_NAME SEED PERIOD_PS CYCLES}"

cd "${PROJECT_DIR}"
mkdir -p "${MD_DMS_RUN_ROOT}"

args=(
  scripts/run/run_mddms_pilot.py
  --run-root "${MD_DMS_RUN_ROOT}"
  --run-name "${RUN_NAME}"
  --preset "${SAFE_PRESET:-pressure_relaxed}"
  --model-alias mace_c
  --natoms "${SAFE_NATOMS:-4000}"
  --seed "${SEED}"
  --temperature-low-K "${SAFE_TEMPERATURE_LOW_K:-300}"
  --strain-amplitude "${SAFE_STRAIN_AMPLITUDE:-0.01}"
  --mddms-period-ps "${PERIOD_PS}"
  --mddms-cycles "${CYCLES}"
  --stress-every-steps "${SAFE_STRESS_EVERY_STEPS:-10}"
  --dump-every-steps "${SAFE_DUMP_EVERY_STEPS:-1000}"
  --lmp-command "${LMP_MACE_KOKKOS_CMD}"
  --execute
  --analyze
  --save-artifacts
)

if [ "${SAFE_DUMP_TRAJECTORY:-1}" = "1" ]; then
  args+=(--dump-trajectory)
fi

# Intentionally do NOT pass --dump-atomic-stress here.
echo "[run] ${PYTHON_BIN} ${args[*]}"
"${PYTHON_BIN}" "${args[@]}"
EOS
  chmod +x "${bin_dir}/mddms_full_run_safe.sh"

  cat > "${bin_dir}/mddms_branch_stage03_safe.sh" <<'EOS'
#!/usr/bin/env bash
set -Eeuo pipefail
source /workspace/cuzr_mddms_runtime.env

BASE_RUN="${1:?usage: mddms_branch_stage03_safe.sh BASE_RUN_OR_DIR BRANCH_RUN PERIOD_PS CYCLES SEED}"
BRANCH_RUN="${2:?usage: mddms_branch_stage03_safe.sh BASE_RUN_OR_DIR BRANCH_RUN PERIOD_PS CYCLES SEED}"
PERIOD_PS="${3:?usage: mddms_branch_stage03_safe.sh BASE_RUN_OR_DIR BRANCH_RUN PERIOD_PS CYCLES SEED}"
CYCLES="${4:?usage: mddms_branch_stage03_safe.sh BASE_RUN_OR_DIR BRANCH_RUN PERIOD_PS CYCLES SEED}"
SEED="${5:?usage: mddms_branch_stage03_safe.sh BASE_RUN_OR_DIR BRANCH_RUN PERIOD_PS CYCLES SEED}"

if [[ "${BASE_RUN}" = /* ]]; then
  BASE_DIR="${BASE_RUN}"
else
  BASE_DIR="${MD_DMS_RUN_ROOT}/${BASE_RUN}"
fi
OUT_DIR="${MD_DMS_RUN_ROOT}/${BRANCH_RUN}"

[ -d "${BASE_DIR}" ] || { echo "ERROR: base run directory not found: ${BASE_DIR}" >&2; exit 1; }
mkdir -p "${OUT_DIR}"

find_stage02_file() {
  local name="$1"
  local p
  for p in \
    "${BASE_DIR}/${name}" \
    "${BASE_DIR}/02_equilibrate_nvt/${name}" \
    "${BASE_DIR}/stage02_equilibrate_nvt/${name}" \
    "${BASE_DIR}/stage02/${name}"; do
    if [ -f "${p}" ]; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}

DATA_SRC="$(find_stage02_file 02_after_equilibrate_nvt.data || true)"
RESTART_SRC="$(find_stage02_file 02_after_equilibrate_nvt.restart || true)"
[ -n "${DATA_SRC}" ] || { echo "ERROR: cannot find 02_after_equilibrate_nvt.data under ${BASE_DIR}" >&2; exit 1; }

cd "${PROJECT_DIR}"

# Generate metadata and stage03 input using the official project generator,
# but execute only stage03 after copying the equilibrated structure from the base run.
gen_args=(
  scripts/run/run_mddms_pilot.py
  --run-root "${MD_DMS_RUN_ROOT}"
  --run-name "${BRANCH_RUN}"
  --preset "${SAFE_PRESET:-pressure_relaxed}"
  --model-alias mace_c
  --natoms "${SAFE_NATOMS:-4000}"
  --seed "${SEED}"
  --temperature-low-K "${SAFE_TEMPERATURE_LOW_K:-300}"
  --strain-amplitude "${SAFE_STRAIN_AMPLITUDE:-0.01}"
  --mddms-period-ps "${PERIOD_PS}"
  --mddms-cycles "${CYCLES}"
  --stress-every-steps "${SAFE_STRESS_EVERY_STEPS:-10}"
  --dump-every-steps "${SAFE_DUMP_EVERY_STEPS:-1000}"
  --lmp-command "${LMP_MACE_KOKKOS_CMD}"
  --save-artifacts
)
if [ "${SAFE_DUMP_TRAJECTORY:-1}" = "1" ]; then
  gen_args+=(--dump-trajectory)
fi
# Intentionally do NOT pass --dump-atomic-stress.
echo "[generate] ${PYTHON_BIN} ${gen_args[*]}"
"${PYTHON_BIN}" "${gen_args[@]}"

cp -f "${DATA_SRC}" "${OUT_DIR}/02_after_equilibrate_nvt.data"
if [ -n "${RESTART_SRC}" ]; then
  cp -f "${RESTART_SRC}" "${OUT_DIR}/02_after_equilibrate_nvt.restart"
fi
if [ -f "${BASE_DIR}/metadata.json" ]; then
  cp -f "${BASE_DIR}/metadata.json" "${OUT_DIR}/base_metadata.json"
fi

cd "${OUT_DIR}"
[ -f 03_mddms_shear.in ] || { echo "ERROR: missing generated 03_mddms_shear.in in ${OUT_DIR}" >&2; exit 1; }

if grep -q "atomstress\|stress/atom\|atom_stress" 03_mddms_shear.in; then
  echo "ERROR: unsafe atomic stress output detected in 03_mddms_shear.in; refusing to run" >&2
  exit 1
fi

echo "[run stage03 only] ${LMP_MACE_KOKKOS_CMD} -log 03_mddms_shear.log -in 03_mddms_shear.in"
${LMP_MACE_KOKKOS_CMD} -log 03_mddms_shear.log -in 03_mddms_shear.in

cd "${PROJECT_DIR}"
ana_args=(
  scripts/run/run_mddms_pilot.py
  --run-root "${MD_DMS_RUN_ROOT}"
  --run-name "${BRANCH_RUN}"
  --preset "${SAFE_PRESET:-pressure_relaxed}"
  --model-alias mace_c
  --natoms "${SAFE_NATOMS:-4000}"
  --seed "${SEED}"
  --temperature-low-K "${SAFE_TEMPERATURE_LOW_K:-300}"
  --strain-amplitude "${SAFE_STRAIN_AMPLITUDE:-0.01}"
  --mddms-period-ps "${PERIOD_PS}"
  --mddms-cycles "${CYCLES}"
  --stress-every-steps "${SAFE_STRESS_EVERY_STEPS:-10}"
  --dump-every-steps "${SAFE_DUMP_EVERY_STEPS:-1000}"
  --lmp-command "${LMP_MACE_KOKKOS_CMD}"
  --analyze
  --save-artifacts
)
if [ "${SAFE_DUMP_TRAJECTORY:-1}" = "1" ]; then
  ana_args+=(--dump-trajectory)
fi

echo "[analyze] ${PYTHON_BIN} ${ana_args[*]}"
"${PYTHON_BIN}" "${ana_args[@]}"

cd "${OUT_DIR}"
for f in stress_timeseries.dat 03_after_mddms.data 03_after_mddms.restart mddms_fit.json pressure_summary.json; do
  [ -s "${f}" ] || { echo "ERROR: missing/empty ${OUT_DIR}/${f}" >&2; exit 1; }
done
if [ "${SAFE_DUMP_TRAJECTORY:-1}" = "1" ]; then
  [ -s trajectory.lammpstrj ] || { echo "ERROR: missing/empty ${OUT_DIR}/trajectory.lammpstrj" >&2; exit 1; }
fi

echo "[ok] branch complete: ${OUT_DIR}"
EOS
  chmod +x "${bin_dir}/mddms_branch_stage03_safe.sh"

  cat > "${bin_dir}/mddms_check_run.sh" <<'EOS'
#!/usr/bin/env bash
set -Eeuo pipefail
source /workspace/cuzr_mddms_runtime.env
RUN_NAME="${1:?usage: mddms_check_run.sh RUN_NAME_OR_DIR}"
if [[ "${RUN_NAME}" = /* ]]; then
  RUN_DIR="${RUN_NAME}"
else
  RUN_DIR="${MD_DMS_RUN_ROOT}/${RUN_NAME}"
fi
[ -d "${RUN_DIR}" ] || { echo "ERROR: run directory not found: ${RUN_DIR}" >&2; exit 1; }
cd "${RUN_DIR}"
echo "[run dir] ${RUN_DIR}"
ls -lh stress_timeseries.dat trajectory.lammpstrj 03_after_mddms.data 03_after_mddms.restart mddms_fit.json pressure_summary.json 2>/dev/null || true
echo
for log in 03_mddms_shear.log stage03_mddms_shear/log.lammps log.lammps; do
  if [ -f "${log}" ]; then
    echo "[tail] ${log}"
    tail -n 40 "${log}"
    echo
  fi
done
if [ -f metadata.json ]; then
  echo "[metadata MD-DMS fields]"
  grep -E 'mddms_period_ps|mddms_cycles|mddms_total_ps|mddms_steps|dump_trajectory|dump_atomic|dump_every_steps' metadata.json || true
fi
EOS
  chmod +x "${bin_dir}/mddms_check_run.sh"

  cat > "${bin_dir}/mddms_archive_run.sh" <<'EOS'
#!/usr/bin/env bash
set -Eeuo pipefail
source /workspace/cuzr_mddms_runtime.env
RUN_NAME="${1:?usage: mddms_archive_run.sh RUN_NAME_OR_DIR}"
if [[ "${RUN_NAME}" = /* ]]; then
  RUN_DIR="${RUN_NAME}"
  PARENT="$(dirname "${RUN_DIR}")"
  BASE="$(basename "${RUN_DIR}")"
else
  PARENT="${MD_DMS_RUN_ROOT}"
  BASE="${RUN_NAME}"
  RUN_DIR="${PARENT}/${BASE}"
fi
[ -d "${RUN_DIR}" ] || { echo "ERROR: run directory not found: ${RUN_DIR}" >&2; exit 1; }
ARCHIVE="${PARENT}/${BASE}.tar.gz"
tar -czf "${ARCHIVE}" -C "${PARENT}" "${BASE}"
tar -tzf "${ARCHIVE}" >/dev/null
echo "[ok] archive: ${ARCHIVE}"
ls -lh "${ARCHIVE}"
EOS
  chmod +x "${bin_dir}/mddms_archive_run.sh"

  echo "Helper scripts written:"
  ls -lh "${bin_dir}"/mddms_*_*.sh "${bin_dir}/mddms_check_run.sh" || true
}

final_check() {
  section "[7/7] Final runtime check"
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

  echo
  echo "[Safe helpers]"
  ls -lh "${WORKSPACE}/bin"/mddms_* 2>/dev/null || true
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
    --run-root "${MD_DMS_RUN_ROOT}" \
    --run-name tiny_mace_c_001 \
    --preset tiny \
    --model-alias mace_c \
    --dump-trajectory \
    --lmp-command "${LMP_MACE_KOKKOS_CMD}" \
    --execute \
    --analyze
}

activate_python_env
normalize_kokkos_arch_flag
ensure_project_dir
print_config
check_python_stack

if [ "${CHECK_ONLY}" = "1" ]; then
  write_runtime_env
  write_helper_scripts
  final_check
  echo "[CuZr-MD-DMS] CHECK_ONLY=1 finished"
  exit 0
fi

fetch_models
convert_mace_c
build_lammps
write_runtime_env
write_helper_scripts
final_check
run_tiny_test

echo
echo "[CuZr-MD-DMS] startup finished successfully"
echo
echo "Recommended current-machine refresh without touching working build/model files:"
echo "  CHECK_ONLY=1 FETCH_MODELS=0 CONVERT_MACE=0 BUILD_LAMMPS=0 bash scripts/cloud/startup_mddms_runtime.sh"
echo
echo "Safe full production run example:"
echo "  source ${WORKSPACE}/cuzr_mddms_runtime.env"
echo "  ${WORKSPACE}/bin/mddms_full_run_safe.sh prod_mace_c_4000_h100_seed44_T300_period50_cyc6_true_001 44 50 6"
echo
echo "Safe period20 branch example from an existing period50 base run:"
echo "  ${WORKSPACE}/bin/mddms_branch_stage03_safe.sh prod_mace_c_4000_h100_seed42_T300_period50_cyc6_true_001_retry_traj_only prod_mace_c_4000_h100_seed42_T300_period20_cyc6_branch_001 20 6 42"
echo
echo "Check/archive examples:"
echo "  ${WORKSPACE}/bin/mddms_check_run.sh RUN_NAME"
echo "  ${WORKSPACE}/bin/mddms_archive_run.sh RUN_NAME"
