#!/usr/bin/env bash
set -euo pipefail

EGO_RUN_DIR="${1:-outputs/runs/partner_conditioned_random1_four_partners}"
shift || true
source "$(dirname "$0")/conda_run.sh"

"${CONDA_RUN[@]}" python -m overcooked_project.evaluate_conditioned_unknown "$EGO_RUN_DIR" "$@"
