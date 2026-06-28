#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-outputs/runs/baseline_simple}"
shift || true
source "$(dirname "$0")/conda_run.sh"

"${CONDA_RUN[@]}" python -m overcooked_project.evaluate --run-dir "$RUN_DIR" "$@"
