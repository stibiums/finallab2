#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-overcooked-marl}"
RUN_DIR="${1:-outputs/runs/baseline_simple}"
shift || true
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
CONDA_RUN=(conda run --no-capture-output -n "$ENV_NAME")

"${CONDA_RUN[@]}" python -m overcooked_project.record_demo --run-dir "$RUN_DIR" "$@"
