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

Train structured `small_corridor` shaping diagnostics:

```bash
bash scripts/train.sh configs/small_corridor_structured_shaping_v1.json
bash scripts/train.sh configs/small_corridor_structured_shaping_v2.json
bash scripts/train.sh configs/small_corridor_structured_shaping_v3.json
```

Fine-tune the structured v3 checkpoint on the isolated `small_corridor` delivery segment:

```bash
bash scripts/train_curriculum.sh configs/small_corridor_delivery_warmstart_from_v3.json
```

Collect scripted delivery demonstrations for behavior cloning or supervised pretraining:

```bash
bash scripts/collect_delivery_demos.sh \
  --episodes 100 \
  --max-steps 40 \
  --output outputs/demos/small_corridor_delivery_scripted.json
```

Train a delivery behavior-cloning policy from the scripted demonstrations:

```bash
bash scripts/train_delivery_bc.sh \
  --run-name small_corridor_delivery_bc_from_v3 \
  --epochs 40 \
  --batch-size 128 \
  --learning-rate 0.001
```

Check the delivery BC policy on warm-start and standard-start evaluation:

```bash
bash scripts/evaluate.sh outputs/runs/small_corridor_delivery_bc_from_v3 \
  --episodes 20 \
  --output-name eval_delivery_warmstart

bash scripts/evaluate.sh outputs/runs/small_corridor_delivery_bc_from_v3 \
  --episodes 20 \
  --standard-start \
  --horizon 400 \
  --output-name eval_standard_start_h400
```

Train held-out hard-layout partner seeds:

```bash
bash scripts/train.sh configs/baseline_random0_long_seed52.json
bash scripts/train.sh configs/baseline_random1_seed71.json
bash scripts/train.sh configs/baseline_unident_s_seed81.json
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

Trace a single episode for debugging:

```bash
bash scripts/trace_episode.sh outputs/runs/baseline_small_corridor --layout small_corridor --output-name small_corridor_trace --max-steps 400
```

Trace files are written under `outputs/runs/<run_name>/traces/` and include per-step actions, rewards, player positions, held objects, world objects, and state strings.
For configs with `event_reward_shaping`, traces also split shaped reward into base Overcooked shaping and custom event/progress shaping.

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

Evaluate the improved onion-layout router:

```bash
bash scripts/evaluate_router.sh configs/router_simple_random0.json \
  --route simple=outputs/runs/curriculum_simple_random0 \
  --route random0=outputs/runs/baseline_random0_long_seed52 \
  --route random1=outputs/runs/baseline_random1 \
  --route unident_s=outputs/runs/baseline_unident_s \
  --output-dir outputs/runs/router_onion_layouts_seed52_random0 \
  --output-name router_eval
```

Run hard-layout partner-robustness matrices:

```bash
bash scripts/evaluate_matrix.sh outputs/runs/baseline_random1 \
  --partner-run-dir outputs/runs/baseline_random1 \
  --partner-run-dir outputs/runs/baseline_random1_seed71 \
  --layout random1 \
  --output-name partner_matrix_hard_random1
```

Repeat the same pattern for `baseline_random0_long` / `baseline_random0_long_seed52` on `random0` and `baseline_unident_s` / `baseline_unident_s_seed81` on `unident_s`.

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
| random0 long held-out seed | `configs/baseline_random0_long_seed52.json` | Improve the `random0` route and test hard-layout partner compatibility |
| small corridor expert | `configs/baseline_small_corridor.json` | Test whether `small_corridor` is learnable with default specialist training |
| small corridor shaping | `configs/small_corridor_shaping_v1.json` | Test whether distance shaping opens the failed `small_corridor` layout |
| small corridor structured shaping v1 | `configs/small_corridor_structured_shaping_v1.json` | Add wrapper-level pickup and progress shaping after the original distance reward path proved inactive |
| small corridor structured shaping v2 | `configs/small_corridor_structured_shaping_v2.json` | Add orientation-aware progress and switch empty-agent target from onion source to dish source after pot progress |
| small corridor structured shaping v3 | `configs/small_corridor_structured_shaping_v3.json` | Add anti-farming guards; reaches soup pickup but still fails final serving |
| small corridor delivery warm start | `configs/small_corridor_delivery_warmstart_from_v3.json` | Continue from structured v3 on soup-held delivery states; PPO still fails sparse delivery |
| scripted delivery demos | `scripts/collect_delivery_demos.sh` | Generate clean final-delivery trajectories for later behavior cloning |
| delivery behavior cloning | `scripts/train_delivery_bc.sh` | Train PPO policy heads from scripted delivery observations/actions; solves delivery warm-start but not full standard start |
| random1 expert | `configs/baseline_random1.json` | Add a successful `random1` specialist for router coverage |
| random1 held-out seed | `configs/baseline_random1_seed71.json` | Test whether `random1` self-play success survives partner mismatch |
| unident_s expert | `configs/baseline_unident_s.json` | Add a successful `unident_s` specialist for router coverage |
| unident_s held-out seed | `configs/baseline_unident_s_seed81.json` | Test whether `unident_s` is robust across independently trained partners |
| simple+random0 router | `configs/router_simple_random0.json` | Compose map specialists and measure routed coverage |

Useful report metrics:

- mean episode reward
- mean sparse reward
- soups delivered, computed as `sparse_reward / 20`
- training curve from TensorBoard
- qualitative GIF behavior

## Notes

The local M4 machine is enough for smoke tests, debugging, plotting, and short-to-medium runs. The 500k-step multi-layout run completed locally in about 249 seconds. Use the 4060 Linux machine for long multi-seed experiments.
