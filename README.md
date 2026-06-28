# Overcooked MARL Course Project

本项目用于完成“胡闹厨房 Overcooked”多智能体强化学习课程作业。目标是训练两个智能体在 Overcooked 环境中合作完成送餐任务，并比较不同 reward shaping 设计对协作表现的影响。

## Project Route

- Environment: `OvercookedMultiEnv-v0` from PantheonRL / Overcooked-AI.
- Baseline: two PPO agents trained with PantheonRL's simultaneous-agent interface.
- Main ablation: no shaping vs default shaping vs distance shaping.
- Follow-up experiments: partner diversity, zero-shot layout transfer, and multi-layout curriculum.
- Outputs: trained models, TensorBoard logs, evaluation CSV/JSON, GIF demo.
- Detailed execution plan: [docs/project_plan.md](docs/project_plan.md).
- Report-facing tables and demo manifest: [docs/report_materials.md](docs/report_materials.md).
- Draft report text: [docs/report_draft.md](docs/report_draft.md).
- Next algorithm comparison plan: [docs/algorithm_comparison_plan.md](docs/algorithm_comparison_plan.md).
- Final submission checklist: [docs/submission_manifest.md](docs/submission_manifest.md).
- Report SVG assets: [docs/assets](docs/assets).
- Final report/slides exports and demo recording script: [report](report).
- Demo video draft: `report/demo_video_draft.mp4`, rebuild with `python scripts/build_demo_video.py --force`.
- Submission archive helper: `python scripts/package_submission.py --name 学号+姓名 --dry-run`.
- Submission preflight helper: `python scripts/check_submission_ready.py --name 学号+姓名`.
- Optional metadata helper: `python scripts/apply_submission_metadata.py --metadata report/submission_metadata.json --export`.

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

Collect full-chain `small_corridor` demonstrations from the standard start:

```bash
bash scripts/collect_delivery_demos.sh \
  --config configs/baseline_small_corridor.json \
  --mode full_chain \
  --episodes 100 \
  --max-steps 180 \
  --output outputs/demos/small_corridor_full_chain_scripted.json
```

Train a full-chain behavior-cloning specialist from those demonstrations:

```bash
bash scripts/train_delivery_bc.sh \
  --config configs/baseline_small_corridor.json \
  --init-run-dir outputs/runs/small_corridor_structured_shaping_v3 \
  --demo-path outputs/demos/small_corridor_full_chain_scripted.json \
  --run-name small_corridor_full_chain_bc_from_v3 \
  --epochs 60 \
  --batch-size 256 \
  --learning-rate 0.001
```

Evaluate and trace the full-chain BC policy from the standard start:

```bash
bash scripts/evaluate.sh outputs/runs/small_corridor_full_chain_bc_from_v3 \
  --episodes 20 \
  --horizon 400 \
  --output-name eval_standard_start_h400

bash scripts/trace_episode.sh outputs/runs/small_corridor_full_chain_bc_from_v3 \
  --horizon 400 \
  --output-name standard_start_h400_trace \
  --max-steps 400
```

Fine-tune the full-chain BC specialist with PPO:

```bash
bash scripts/train_curriculum.sh configs/baseline_small_corridor.json \
  --run-name small_corridor_full_chain_bc_ppo_finetune \
  --init-run-dir outputs/runs/small_corridor_full_chain_bc_from_v3 \
  --timesteps 50000 \
  --eval-interval 25000 \
  --eval-episodes 20
```

Collect and train a 3-cycle full-chain `small_corridor` BC specialist:

```bash
bash scripts/collect_delivery_demos.sh \
  --config configs/baseline_small_corridor.json \
  --mode full_chain \
  --full-chain-cycles 3 \
  --episodes 100 \
  --max-steps 400 \
  --output outputs/demos/small_corridor_full_chain_3cycle_scripted.json

bash scripts/train_delivery_bc.sh \
  --config configs/baseline_small_corridor.json \
  --init-run-dir outputs/runs/small_corridor_structured_shaping_v3 \
  --demo-path outputs/demos/small_corridor_full_chain_3cycle_scripted.json \
  --run-name small_corridor_full_chain_3cycle_bc_from_v3 \
  --epochs 60 \
  --batch-size 256 \
  --learning-rate 0.001

bash scripts/evaluate.sh outputs/runs/small_corridor_full_chain_3cycle_bc_from_v3 \
  --episodes 20 \
  --horizon 400 \
  --output-name eval_standard_start_h400

bash scripts/train_curriculum.sh configs/baseline_small_corridor.json \
  --run-name small_corridor_full_chain_3cycle_bc_ppo_finetune \
  --init-run-dir outputs/runs/small_corridor_full_chain_3cycle_bc_from_v3 \
  --timesteps 50000 \
  --eval-interval 25000 \
  --eval-episodes 20
```

Collect perturbed 3-cycle demos and checkpoint-select a PPO fine-tune:

```bash
ENV_PREFIX=/Volumes/data/conda_envs/overcooked-marl bash scripts/collect_delivery_demos.sh \
  --config configs/baseline_small_corridor.json \
  --mode full_chain \
  --full-chain-cycles 3 \
  --full-chain-wait-jitter 3 \
  --seed 91 \
  --episodes 100 \
  --max-steps 420 \
  --output outputs/demos/small_corridor_full_chain_3cycle_jitter3_scripted.json

ENV_PREFIX=/Volumes/data/conda_envs/overcooked-marl bash scripts/train_delivery_bc.sh \
  --config configs/baseline_small_corridor.json \
  --init-run-dir outputs/runs/small_corridor_structured_shaping_v3 \
  --demo-path outputs/demos/small_corridor_full_chain_3cycle_jitter3_scripted.json \
  --run-name small_corridor_full_chain_3cycle_jitter3_bc_from_v3 \
  --epochs 60 \
  --batch-size 256 \
  --learning-rate 0.001

ENV_PREFIX=/Volumes/data/conda_envs/overcooked-marl bash scripts/train_curriculum.sh configs/baseline_small_corridor.json \
  --run-name small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune \
  --init-run-dir outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_from_v3 \
  --timesteps 50000 \
  --eval-interval 25000 \
  --eval-episodes 20
```

Train held-out hard-layout partner seeds:

```bash
bash scripts/train.sh configs/baseline_random0_long_seed52.json
bash scripts/train.sh configs/baseline_random1_seed71.json
bash scripts/train.sh configs/baseline_random1_seed73.json
bash scripts/train.sh configs/baseline_unident_s_seed81.json
```

You can override common options:

```bash
bash scripts/train.sh configs/baseline_simple.json --run-name baseline_seed11 --seed 11 --timesteps 200000
```

All shell scripts use `conda run --no-capture-output` and `PYTHONUNBUFFERED=1`, so training logs stream live instead of being buffered until the process exits. By default they use `ENV_NAME=overcooked-marl`; set `ENV_PREFIX=/Volumes/data/conda_envs/overcooked-marl` if conda cannot find a writable named-env directory.

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

This keeps `simple` from the router config, replaces `random0` with the stronger long-budget specialist, and adds the successful `random1` and `unident_s` specialists. `small_corridor` is intentionally left unrouted in this PPO-only comparison.

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

Evaluate the broadest current onion-layout router, including the `small_corridor` full-chain BC specialist:

```bash
bash scripts/evaluate_router.sh configs/router_simple_random0.json \
  --route simple=outputs/runs/curriculum_simple_random0 \
  --route random0=outputs/runs/baseline_random0_long_seed52 \
  --route random1=outputs/runs/baseline_random1 \
  --route unident_s=outputs/runs/baseline_unident_s \
  --route small_corridor=outputs/runs/small_corridor_full_chain_bc_from_v3 \
  --output-dir outputs/runs/router_onion_layouts_with_small_corridor_bc \
  --output-name router_eval
```

Evaluate the current broadest onion-layout router with the 3-cycle `small_corridor` BC specialist:

```bash
bash scripts/evaluate_router.sh configs/router_simple_random0.json \
  --route simple=outputs/runs/curriculum_simple_random0 \
  --route random0=outputs/runs/baseline_random0_long_seed52 \
  --route random1=outputs/runs/baseline_random1 \
  --route unident_s=outputs/runs/baseline_unident_s \
  --route small_corridor=outputs/runs/small_corridor_full_chain_3cycle_bc_from_v3 \
  --output-dir outputs/runs/router_onion_layouts_with_small_corridor_3cycle_bc \
  --output-name router_eval
```

Evaluate the current strongest onion-layout router, using the checkpoint-selected perturbed `small_corridor` BC+PPO specialist:

```bash
ENV_PREFIX=/Volumes/data/conda_envs/overcooked-marl bash scripts/evaluate_router.sh configs/router_simple_random0.json \
  --route simple=outputs/runs/curriculum_simple_random0 \
  --route random0=outputs/runs/baseline_random0_long_seed52 \
  --route random1=outputs/runs/baseline_random1 \
  --route unident_s=outputs/runs/baseline_unident_s \
  --route small_corridor=outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune \
  --output-dir outputs/runs/router_onion_layouts_with_small_corridor_jitter3_bc_ppo \
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
| full-chain small corridor BC | `scripts/collect_delivery_demos.sh --mode full_chain` + `scripts/train_delivery_bc.sh` | Behavior-clone one complete standard-start cooking chain; reaches 1 soup where PPO stays at 0 |
| 3-cycle full-chain small corridor BC | `scripts/collect_delivery_demos.sh --mode full_chain --full-chain-cycles 3` + `scripts/train_delivery_bc.sh` | Behavior-clone a repeated cooking loop; improves `small_corridor` to 1.90 average soups, though still brittle |
| perturbed 3-cycle small corridor BC+PPO | `scripts/collect_delivery_demos.sh --full-chain-wait-jitter 3` + `scripts/train_curriculum.sh` | Adds wait perturbations and uses checkpoint selection; reaches 3.00 soups on `small_corridor` at the 25k checkpoint |
| role-balanced small corridor BC diagnostic | `configs/small_corridor_full_chain_3cycle_jitter3_role_balanced_bc_from_v3.json` + `scripts/train_delivery_bc.sh --role-balanced` | Trains each policy on combined p0/p1 role data; reaches 2.40 soups, so it is a negative diagnostic rather than a new best route |
| small corridor subtask router diagnostic | `configs/small_corridor_subtask_router_jitter_bc_delivery.json` + `scripts/evaluate_subtask_router.sh` | Switches from full-chain BC to delivery BC when soup is held; slightly improves BC-only to 2.60 soups but does not beat checkpoint-selected BC+PPO |
| role-specific small corridor router diagnostic | `configs/small_corridor_subtask_router_*_holder.json` + `scripts/evaluate_subtask_router.sh` | Tests ego-holder versus partner-holder switching; partner-holder reproduces held-soup routing, ego-holder never switches, and neither beats the best checkpoint |
| random1 expert | `configs/baseline_random1.json` | Add a successful `random1` specialist for router coverage |
| random1 held-out seed | `configs/baseline_random1_seed71.json` | Test whether `random1` self-play success survives partner mismatch |
| random1 held-out seed72 | `configs/baseline_random1_seed72.json` | Add a third `random1` partner for held-out compatibility testing |
| random1 partner diversity | `configs/partner_diversity_random1.json` | Train one `random1` ego against two fixed partners; improves in-pool compatibility but remains weak on held-out seed72 |
| random1 held-out seed73 | `configs/baseline_random1_seed73.json` | Add a fourth `random1` partner; self-play remains weak and cross-play stays brittle |
| random1 unknown seed76 | `configs/baseline_random1_seed76.json` | Add an unseen `random1` partner probe; self-play reaches 1.55 soups but all tested prior egos score 0 with it |
| random1 unknown seeds77/78 | `configs/baseline_random1_seed77.json`, `configs/baseline_random1_seed78.json` | Larger held-out validation; seed77 self-play reaches 5.25 soups and seed78 reaches 1.50, but prior egos still score 0 and partner-id conditioning remains at 0.00 to 0.05 |
| random1 three-partner diversity | `configs/partner_diversity_random1_three_partners.json` | Train one `random1` ego against three fixed partners; improves the seen-partner minimum but does not solve held-out seed73 |
| random1 fixed+learned partner mix | `configs/partner_diversity_random1_three_partners_selfplay_mix.json` | Train against three fixed partners plus one learned on-policy partner; negative result with 0.69 average soups over four fixed partners |
| random1 partner-id conditioned | `configs/partner_conditioned_random1_four_partners.json` | Append partner id one-hot to ego observation; improves four-known-partner average to 2.34 soups and minimum to 0.80, but unknown seeds 76/77/78 remain 0.00 to 0.05 under fixed and inferred ids |
| random1 six-partner conditioned | `configs/partner_conditioned_random1_six_partners_holdout78.json` | Larger fixed-pool control; train-pool average is only 0.66 soups with 0.00 minimum, and held-out seed78 stays at 0.00 |
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
