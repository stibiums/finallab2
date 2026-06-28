#!/usr/bin/env bash

ENV_NAME="${ENV_NAME:-overcooked-marl}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ -n "${ENV_PREFIX:-}" ]]; then
  CONDA_RUN=(conda run --no-capture-output -p "$ENV_PREFIX")
else
  CONDA_RUN=(conda run --no-capture-output -n "$ENV_NAME")
fi
