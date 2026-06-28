#!/usr/bin/env bash
set -euo pipefail

EGO_RUN_DIR="${1:-outputs/runs/baseline_simple}"
shift || true
source "$(dirname "$0")/conda_run.sh"

"${CONDA_RUN[@]}" python -m overcooked_project.evaluate_matrix --ego-run-dir "$EGO_RUN_DIR" "$@"
