# Overcooked MARL Course Project

本项目用于完成“胡闹厨房 Overcooked”多智能体强化学习课程作业。目标是训练两个智能体在 Overcooked 环境中合作完成送餐任务，并比较不同 reward shaping 设计对协作表现的影响。

## Project Route

- Environment: `OvercookedMultiEnv-v0` from PantheonRL / Overcooked-AI.
- Baseline: two PPO agents trained with PantheonRL's simultaneous-agent interface.
- Main ablation: no shaping vs default shaping vs distance shaping.
- Follow-up experiments: partner diversity, zero-shot layout transfer, and multi-layout curriculum.
- Outputs: trained models, TensorBoard logs, evaluation CSV/JSON, GIF demo.
- Detailed execution plan: [docs/project_plan.md](docs/project_plan.md).

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

Train the partner-diversity improvement after preparing several fixed partner runs:

```bash
bash scripts/train.sh configs/baseline_simple.json --run-name baseline_seed11 --seed 11
bash scripts/train.sh configs/baseline_simple.json --run-name baseline_seed12 --seed 12
bash scripts/train_diverse.sh configs/partner_diversity_simple.json
```

Train the first multi-layout curriculum:

```bash
bash scripts/train.sh configs/multi_layout_curriculum.json
```

This samples one layout per episode from `simple`, `small_corridor`, `random0`, and `random1`. Tomato layouts are excluded because this Overcooked-AI wrapper currently raises `KeyError: 'tomato'` when stepping tomato maps.

Fine-tune from an existing checkpoint with staged curriculum evaluation:

```bash
bash scripts/train_curriculum.sh configs/curriculum_simple_random0.json
```

This loads `outputs/runs/baseline_simple`, trains on `simple + random0`, evaluates both fixed layouts every 50k steps, and keeps the best checkpoint as `models/ego.zip` / `models/alt.zip`. The final checkpoint is also saved separately as `final_ego.zip` / `final_alt.zip`.

Train a single-layout `random0` diagnostic expert:

```bash
bash scripts/train.sh configs/baseline_random0.json
```

Train the stronger long-budget `random0` specialist:

```bash
bash scripts/train.sh configs/baseline_random0_long.json
```

Train Phase 2 layout specialists:

```bash
bash scripts/train.sh configs/baseline_small_corridor.json
bash scripts/train.sh configs/small_corridor_shaping_v1.json
bash scripts/train.sh configs/baseline_random1.json
bash scripts/train.sh configs/baseline_unident_s.json
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

Evaluation writes per-episode CSV and summary JSON under `metrics/`. The demo script writes a lightweight top-down GIF under `demo/`. For multi-layout runs, pass `--layout` to force the demo to a specific map:

```bash
bash scripts/record_demo.sh outputs/runs/multi_layout_curriculum --layout random0 --output-name random0_demo --max-steps 400
```

Run cross-layout and cross-partner evaluation:

```bash
bash scripts/evaluate_matrix.sh outputs/runs/baseline_simple \
  --partner-run-dir outputs/runs/baseline_simple \
  --layout simple --layout simple_tomato --layout small_corridor --layout random0 --layout random1 --layout unident_s \
  --output-name zero_shot_layouts

bash scripts/evaluate_matrix.sh outputs/runs/partner_diversity_simple \
  --partner-run-dir outputs/runs/baseline_simple \
  --partner-run-dir outputs/runs/baseline_seed11 \
  --partner-run-dir outputs/runs/baseline_seed12 \
  --layout simple \
  --output-name partner_matrix

bash scripts/evaluate_matrix.sh outputs/runs/multi_layout_curriculum \
  --partner-run-dir outputs/runs/multi_layout_curriculum \
  --layout simple --layout small_corridor --layout random0 --layout random1 --layout unident_s --layout simple_tomato \
  --output-name zero_shot_layouts
```

Matrix evaluation records unsupported or failing layout/partner combinations as `status=error` rows instead of aborting the whole sweep.

Evaluate the layout-router baseline:

```bash
bash scripts/evaluate_router.sh configs/router_simple_random0.json
```

This routes `simple` to `outputs/runs/curriculum_simple_random0` and `random0` to `outputs/runs/baseline_random0`. Layouts without a configured specialist are recorded as `status=skipped`, which makes the current coverage explicit.

Evaluate the expanded onion-layout router:

```bash
bash scripts/evaluate_router.sh configs/router_simple_random0.json \
  --route random0=outputs/runs/baseline_random0_long \
  --route random1=outputs/runs/baseline_random1 \
  --route unident_s=outputs/runs/baseline_unident_s \
  --output-dir outputs/runs/router_onion_layouts \
  --output-name router_eval
```

This keeps `simple` from the router config, replaces `random0` with the stronger long-budget specialist, and adds the successful `random1` and `unident_s` specialists. `small_corridor` is intentionally left unrouted until a nonzero specialist exists.

## Suggested Experiment Table

| Run | Config | Purpose |
| --- | --- | --- |
| no shaping | `configs/no_shaping_simple.json` | Test sparse-reward difficulty |
| default shaping | `configs/baseline_simple.json` | Course baseline for shaped rewards |
| distance shaping | `configs/distance_shaping_simple.json` | Test whether extra dense guidance helps |
| partner diversity | `configs/partner_diversity_simple.json` | Train one ego policy against several fixed partners |
| multi-layout curriculum | `configs/multi_layout_curriculum.json` | Test whether naive episode-level layout mixing improves transfer |
| staged simple+random0 | `configs/curriculum_simple_random0.json` | Fine-tune from `simple` baseline while periodically evaluating both layouts |
| random0 expert | `configs/baseline_random0.json` | Test whether `random0` is learnable as a single-layout specialist |
| random0 long expert | `configs/baseline_random0_long.json` | Test whether more budget makes the hard-layout specialist reliable |
| small corridor expert | `configs/baseline_small_corridor.json` | Test whether `small_corridor` is learnable with default specialist training |
| small corridor shaping | `configs/small_corridor_shaping_v1.json` | Test whether distance shaping opens the failed `small_corridor` layout |
| random1 expert | `configs/baseline_random1.json` | Add a successful `random1` specialist for router coverage |
| unident_s expert | `configs/baseline_unident_s.json` | Add a successful `unident_s` specialist for router coverage |
| simple+random0 router | `configs/router_simple_random0.json` | Compose map specialists and measure routed coverage |

Useful report metrics:

- mean episode reward
- mean sparse reward
- soups delivered, computed as `sparse_reward / 20`
- training curve from TensorBoard
- qualitative GIF behavior

## Notes

The local M4 machine is enough for smoke tests, debugging, plotting, and short-to-medium runs. The 500k-step multi-layout run completed locally in about 249 seconds. Use the 4060 Linux machine for long multi-seed experiments.
