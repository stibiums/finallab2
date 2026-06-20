# Overcooked MARL Course Project

本项目用于完成“胡闹厨房 Overcooked”多智能体强化学习课程作业。目标是训练两个智能体在 Overcooked 环境中合作完成送餐任务，并比较不同 reward shaping 设计对协作表现的影响。

## Project Route

- Environment: `OvercookedMultiEnv-v0` from PantheonRL / Overcooked-AI.
- Baseline: two PPO agents trained with PantheonRL's simultaneous-agent interface.
- Main ablation: no shaping vs default shaping vs distance shaping.
- Outputs: trained models, TensorBoard logs, evaluation CSV/JSON, GIF demo.

## Setup

```bash
git submodule update --init --recursive
bash scripts/create_env.sh
```

The conda environment is named `overcooked-marl`. The install script intentionally installs PantheonRL with `--no-deps` because its upstream `setup.py` pins TensorFlow, which is unnecessary for the PPO experiments here and can be fragile on Apple Silicon.

## Smoke Test

```bash
bash scripts/smoke_test.sh
```

Expected artifacts:

- `outputs/runs/smoke_simple/models/ego.zip`
- `outputs/runs/smoke_simple/models/alt.zip`
- `outputs/runs/smoke_simple/metrics/eval_metrics.json`
- `outputs/runs/smoke_simple/demo/demo.gif`

## Training

Run the baseline:

```bash
bash scripts/train.sh configs/baseline_simple.json
```

Run the reward-shaping ablation:

```bash
bash scripts/run_ablation.sh
```

You can override common options:

```bash
bash scripts/train.sh configs/baseline_simple.json --run-name baseline_seed11 --seed 11 --timesteps 200000
```

All shell scripts use `conda run --no-capture-output` and `PYTHONUNBUFFERED=1`, so training logs stream live instead of being buffered until the process exits.

## Evaluation And Demo

```bash
bash scripts/evaluate.sh outputs/runs/baseline_simple --episodes 20
bash scripts/record_demo.sh outputs/runs/baseline_simple --max-steps 400
```

Evaluation writes per-episode CSV and summary JSON under `metrics/`. The demo script writes a lightweight top-down GIF under `demo/`.

## Suggested Experiment Table

| Run | Config | Purpose |
| --- | --- | --- |
| no shaping | `configs/no_shaping_simple.json` | Test sparse-reward difficulty |
| default shaping | `configs/baseline_simple.json` | Course baseline for shaped rewards |
| distance shaping | `configs/distance_shaping_simple.json` | Test whether extra dense guidance helps |

Useful report metrics:

- mean episode reward
- mean sparse reward
- soups delivered, computed as `sparse_reward / 20`
- training curve from TensorBoard
- qualitative GIF behavior

## Notes

The local M4 machine is enough for smoke tests, debugging, plotting, and short runs. Use the 4060 Linux machine for long multi-seed experiments.
