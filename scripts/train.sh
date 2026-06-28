#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/baseline_simple.json}"
shift || true
source "$(dirname "$0")/conda_run.sh"

"${CONDA_RUN[@]}" python -m overcooked_project.train --config "$CONFIG" "$@"
