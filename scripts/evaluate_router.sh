#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/router_simple_random0.json}"
shift || true
source "$(dirname "$0")/conda_run.sh"

"${CONDA_RUN[@]}" python -m overcooked_project.evaluate_router --config "$CONFIG_PATH" "$@"
