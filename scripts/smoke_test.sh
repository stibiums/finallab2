#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-overcooked-marl}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
CONDA_RUN=(conda run --no-capture-output -n "$ENV_NAME")

"${CONDA_RUN[@]}" python -m overcooked_project.train --config configs/smoke.json
"${CONDA_RUN[@]}" python -m overcooked_project.evaluate --run-dir outputs/runs/smoke_simple --episodes 2
"${CONDA_RUN[@]}" python -m overcooked_project.record_demo --run-dir outputs/runs/smoke_simple --max-steps 40
