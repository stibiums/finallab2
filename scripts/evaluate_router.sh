#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-overcooked-marl}"
CONFIG_PATH="${1:-configs/router_simple_random0.json}"
shift || true
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
CONDA_RUN=(conda run --no-capture-output -n "$ENV_NAME")

"${CONDA_RUN[@]}" python -m overcooked_project.evaluate_router --config "$CONFIG_PATH" "$@"
