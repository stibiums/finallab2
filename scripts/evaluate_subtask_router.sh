#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/small_corridor_subtask_router_jitter_bc_delivery.json}"
shift || true
source "$(dirname "$0")/conda_run.sh"

"${CONDA_RUN[@]}" python -m overcooked_project.evaluate_subtask_router --config "$CONFIG" "$@"
