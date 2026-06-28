#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/conda_run.sh"

"${CONDA_RUN[@]}" python -m overcooked_project.train --config configs/smoke.json
"${CONDA_RUN[@]}" python -m overcooked_project.evaluate --run-dir outputs/runs/smoke_simple --episodes 2
"${CONDA_RUN[@]}" python -m overcooked_project.record_demo --run-dir outputs/runs/smoke_simple --max-steps 40
