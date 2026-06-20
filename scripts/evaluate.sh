#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-overcooked-marl}"
RUN_DIR="${1:-outputs/runs/baseline_simple}"
shift || true

conda run -n "$ENV_NAME" python -m overcooked_project.evaluate --run-dir "$RUN_DIR" "$@"
