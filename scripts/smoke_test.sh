#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-overcooked-marl}"

conda run -n "$ENV_NAME" python -m overcooked_project.train --config configs/smoke.json
conda run -n "$ENV_NAME" python -m overcooked_project.evaluate --run-dir outputs/runs/smoke_simple --episodes 2
conda run -n "$ENV_NAME" python -m overcooked_project.record_demo --run-dir outputs/runs/smoke_simple --max-steps 40
