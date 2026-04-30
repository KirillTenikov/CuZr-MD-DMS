#!/usr/bin/env bash
set -euo pipefail

# Fetch potential/model files for CuZr-MD-DMS from GitHub Releases.
#
# Default source:
#   https://github.com/KirillTenikov/CuZr-MD-DMS/releases/tag/Models
#
# This script does not commit models to git. It only populates runtime folders:
#   models/mace/
#   models/eam/
#
# Usage from repo root:
#   bash scripts/cloud/fetch_models.sh
#
# Optional:
#   RELEASE_TAG=Models bash scripts/cloud/fetch_models.sh
#   FORCE_DOWNLOAD=1 bash scripts/cloud/fetch_models.sh

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
REPO="${REPO:-KirillTenikov/CuZr-MD-DMS}"
RELEASE_TAG="${RELEASE_TAG:-Models}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

MACE_DIR="${PROJECT_DIR}/models/mace"
EAM_DIR="${PROJECT_DIR}/models/eam"
TMP_DIR="${PROJECT_DIR}/.tmp_models_${RELEASE_TAG}"

mkdir -p "${MACE_DIR}" "${EAM_DIR}" "${TMP_DIR}"

echo "[fetch_models] PROJECT_DIR=${PROJECT_DIR}"
echo "[fetch_models] REPO=${REPO}"
echo "[fetch_models] RELEASE_TAG=${RELEASE_TAG}"
echo "[fetch_models] MACE_DIR=${MACE_DIR}"
echo "[fetch_models] EAM_DIR=${EAM_DIR}"
echo

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl is required."
  exit 1
fi

ASSETS_JSON="${TMP_DIR}/release.json"
URLS_FILE="${TMP_DIR}/download_urls.tsv"

echo "[1/5] Query GitHub release API"
curl -fsSL "https://api.github.com/repos/${REPO}/releases/tags/${RELEASE_TAG}" -o "${ASSETS_JSON}"

python - "${ASSETS_JSON}" "${URLS_FILE}" <<'PY'
import json
import sys
from pathlib import Path

assets_json = Path(sys.argv[1])
urls_file = Path(sys.argv[2])

data = json.loads(assets_json.read_text())
assets = data.get("assets", [])

if not assets:
    print("WARNING: no assets found in release JSON")

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
    print(f"  {name}  ({size/1024/1024:.2f} MB)")
PY

echo
echo "[2/5] Download release assets"
while IFS="$(printf '\t')" read -r name url size; do
  [ -n "${name:-}" ] || continue
  dest="${TMP_DIR}/${name}"

  if [ -f "${dest}" ] && [ "${FORCE_DOWNLOAD}" != "1" ]; then
    echo "Already present: ${name}"
  else
    echo "Downloading: ${name}"
    curl -fL --retry 3 --retry-delay 5 -o "${dest}" "${url}"
  fi
done < "${URLS_FILE}"

echo
echo "[3/5] Verify SHA256SUMS.txt if present"
if [ -f "${TMP_DIR}/SHA256SUMS.txt" ]; then
  (
    cd "${TMP_DIR}"
    sha256sum -c SHA256SUMS.txt
  )
else
  echo "No SHA256SUMS.txt found; skipping checksum verification."
fi

echo
echo "[4/5] Populate runtime model directories"

# Put MACE-like files into models/mace.
find "${TMP_DIR}" -maxdepth 1 -type f \( \
    -iname "*.pt" -o \
    -iname "*.model" -o \
    -iname "*.pth" \
  \) -print0 |
while IFS= read -r -d '' f; do
  base="$(basename "$f")"
  ln -sf "$f" "${MACE_DIR}/${base}"
done

# Put EAM-like files into models/eam.
find "${TMP_DIR}" -maxdepth 1 -type f \( \
    -iname "*.alloy" -o \
    -iname "*.eam" -o \
    -iname "*.fs" -o \
    -iname "*.setfl" \
  \) -print0 |
while IFS= read -r -d '' f; do
  base="$(basename "$f")"
  ln -sf "$f" "${EAM_DIR}/${base}"
done

# Create stable convenience symlinks when recognizable.
python - "${MACE_DIR}" "${EAM_DIR}" <<'PY'
from pathlib import Path
import re
import sys

mace_dir = Path(sys.argv[1])
eam_dir = Path(sys.argv[2])

def symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.name)
    print(f"  {dst.name} -> {src.name}")

print("Convenience symlinks:")

mace_files = [p for p in mace_dir.iterdir() if p.is_file() or p.is_symlink()]
eam_files = [p for p in eam_dir.iterdir() if p.is_file() or p.is_symlink()]

# Prefer an ML-IAP MACE_C .pt.
mace_c_mliap = [
    p for p in mace_files
    if p.suffix.lower() == ".pt"
    and "mliap" in p.name.lower()
    and re.search(r"mace[_-]?c|mace.*_c_|mace.*c", p.name.lower())
]
if mace_c_mliap:
    symlink(sorted(mace_c_mliap, key=lambda p: len(p.name))[0], mace_dir / "mace_C.model-mliap_lammps.pt")
else:
    print("  No recognizable MACE_C ML-IAP .pt file found.")

# Raw MACE_C model.
mace_c_raw = [
    p for p in mace_files
    if p.suffix.lower() == ".model"
    and re.search(r"mace[_-]?c|mace.*_c_|mace.*c", p.name.lower())
]
if mace_c_raw:
    symlink(sorted(mace_c_raw, key=lambda p: len(p.name))[0], mace_dir / "mace_C_raw.model")

# EAM 2019.
eam_2019 = [
    p for p in eam_files
    if any(x in p.name.lower() for x in ["2019", "mendelev"])
    and p.suffix.lower() in [".alloy", ".eam", ".fs", ".setfl"]
]
if eam_2019:
    # If multiple, prefer one with 2019 in name.
    eam_2019 = sorted(eam_2019, key=lambda p: (("2019" not in p.name.lower()), len(p.name)))
    symlink(eam_2019[0], eam_dir / "eam_mendelev_2019.eam.alloy")
else:
    print("  No recognizable EAM 2019 file found.")
PY

echo
echo "[5/5] Final directories"
echo "MACE:"
ls -lh "${MACE_DIR}" || true
echo
echo "EAM:"
ls -lh "${EAM_DIR}" || true

echo
echo "[fetch_models] Done."

if [ ! -e "${MACE_DIR}/mace_C.model-mliap_lammps.pt" ]; then
  echo
  echo "WARNING: models/mace/mace_C.model-mliap_lammps.pt is missing."
  echo "The first MACE MD-DMS run expects this file."
  echo "If only a raw .model file is available, convert it on the GPU machine."
fi
