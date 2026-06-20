#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-overcooked-marl}"
CONFIG="${1:-configs/baseline_simple.json}"
shift || true

conda run -n "$ENV_NAME" python -m overcooked_project.train --config "$CONFIG" "$@"
