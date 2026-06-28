#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/curriculum_simple_random0.json}"
shift || true
source "$(dirname "$0")/conda_run.sh"

"${CONDA_RUN[@]}" python -m overcooked_project.train_curriculum --config "$CONFIG" "$@"
