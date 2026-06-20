#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-overcooked-marl}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
CONDA_RUN=(conda run --no-capture-output -n "$ENV_NAME")

for config in \
  configs/no_shaping_simple.json \
  configs/baseline_simple.json \
  configs/distance_shaping_simple.json
do
  "${CONDA_RUN[@]}" python -m overcooked_project.train --config "$config"
done
