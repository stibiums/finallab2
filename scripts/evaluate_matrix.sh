#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-overcooked-marl}"
EGO_RUN_DIR="${1:-outputs/runs/baseline_simple}"
shift || true
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
CONDA_RUN=(conda run --no-capture-output -n "$ENV_NAME")

"${CONDA_RUN[@]}" python -m overcooked_project.evaluate_matrix --ego-run-dir "$EGO_RUN_DIR" "$@"
