#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/conda_run.sh"

for config in \
  configs/no_shaping_simple.json \
  configs/baseline_simple.json \
  configs/distance_shaping_simple.json
do
  "${CONDA_RUN[@]}" python -m overcooked_project.train --config "$config"
done
