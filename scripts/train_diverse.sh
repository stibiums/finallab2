#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/partner_diversity_simple.json}"
shift || true
source "$(dirname "$0")/conda_run.sh"

"${CONDA_RUN[@]}" python -m overcooked_project.train_diverse --config "$CONFIG" "$@"
