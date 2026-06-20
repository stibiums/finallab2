#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-overcooked-marl}"

for config in \
  configs/no_shaping_simple.json \
  configs/baseline_simple.json \
  configs/distance_shaping_simple.json
do
  conda run -n "$ENV_NAME" python -m overcooked_project.train --config "$config"
done
