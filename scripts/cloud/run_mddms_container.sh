#!/usr/bin/env bash
set -euo pipefail

# Helper script to start the existing Paper-1 CuZr MD Docker image for Paper-2 MD-DMS.
#
# Usage:
#   IMAGE=kseina89/YOUR_EXISTING_PAPER1_IMAGE:TAG bash scripts/cloud/run_paper1_image_container.sh
#
# Or pass image as first argument:
#   bash scripts/cloud/run_paper1_image_container.sh kseina89/YOUR_EXISTING_PAPER1_IMAGE:TAG

IMAGE="${1:-${IMAGE:-}}"
if [ -z "${IMAGE}" ]; then
  echo "ERROR: provide image name:"
  echo "  IMAGE=kseina89/YOUR_EXISTING_PAPER1_IMAGE:TAG bash $0"
  echo "or:"
  echo "  bash $0 kseina89/YOUR_EXISTING_PAPER1_IMAGE:TAG"
  exit 1
fi

WORKSPACE_HOST="${WORKSPACE_HOST:-/workspace}"
WORKSPACE_CONTAINER="${WORKSPACE_CONTAINER:-/workspace}"
CONTAINER_NAME="${CONTAINER_NAME:-cuzr-mddms-paper1-runtime}"

mkdir -p "${WORKSPACE_HOST}"

docker run --gpus all -it --rm   --name "${CONTAINER_NAME}"   -v "${WORKSPACE_HOST}:${WORKSPACE_CONTAINER}"   -w "${WORKSPACE_CONTAINER}"   "${IMAGE}"   /bin/bash
