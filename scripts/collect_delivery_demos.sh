#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/conda_run.sh"

"${CONDA_RUN[@]}" python -m overcooked_project.collect_delivery_demos "$@"
