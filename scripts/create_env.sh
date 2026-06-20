#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-overcooked-marl}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
CONDA_RUN=(conda run --no-capture-output -n "$ENV_NAME")

cd "$ROOT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found on PATH." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update -n "$ENV_NAME" -f environment.yml
else
  conda env create -f environment.yml
fi

git submodule update --init --recursive

"${CONDA_RUN[@]}" python -m pip install "pip<24" "setuptools==65.5.0" "wheel<0.40.0"
"${CONDA_RUN[@]}" python -m pip install -r requirements.txt
"${CONDA_RUN[@]}" python -m pip install --no-deps \
  -e external/PantheonRL \
  -e external/PantheonRL/overcookedgym/human_aware_rl/overcooked_ai \
  -e .

echo "Environment ready: $ENV_NAME"
