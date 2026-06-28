# Overcooked MARL Experiment Log

Date: 2026-06-20
Last updated: 2026-06-27

Project path: `/Volumes/share/pku/26_spring/多智能体/finallab2`

## Goal

This log records the current Overcooked multi-agent reinforcement learning experiments. The main question is whether a simple PPO baseline on the `simple` map learns a usable cooperative policy, and what its main weaknesses are under reward ablation, partner changes, zero-shot layout changes, and multi-layout curriculum training.

## Environment

- Conda environment: `overcooked-marl`
- Algorithm: PPO from Stable-Baselines3
- Environment wrapper: PantheonRL simultaneous-agent interface around Overcooked-AI
- Main training map: `simple`
- Horizon: 400
- Evaluation: 20 deterministic episodes unless noted
- Primary metric: soups delivered, computed as sparse reward / 20

## Commands

```bash
bash scripts/train.sh configs/baseline_simple.json
bash scripts/train.sh configs/no_shaping_simple.json
bash scripts/train.sh configs/distance_shaping_simple.json

bash scripts/train.sh configs/baseline_simple.json --run-name baseline_seed11 --seed 11
bash scripts/train.sh configs/baseline_simple.json --run-name baseline_seed12 --seed 12
bash scripts/train_diverse.sh configs/partner_diversity_simple.json

bash scripts/evaluate_matrix.sh outputs/runs/baseline_simple \
  --partner-run-dir outputs/runs/baseline_simple \
  --partner-run-dir outputs/runs/baseline_seed11 \
  --partner-run-dir outputs/runs/baseline_seed12 \
  --layout simple \
  --output-name partner_matrix_baseline

bash scripts/evaluate_matrix.sh outputs/runs/partner_diversity_simple \
  --partner-run-dir outputs/runs/baseline_simple \
  --partner-run-dir outputs/runs/baseline_seed11 \
  --partner-run-dir outputs/runs/baseline_seed12 \
  --layout simple \
  --output-name partner_matrix

bash scripts/evaluate_matrix.sh outputs/runs/baseline_simple \
  --partner-run-dir outputs/runs/baseline_simple \
  --layout simple --layout simple_tomato --layout small_corridor --layout random0 --layout random1 --layout unident_s \
  --output-name zero_shot_layouts

bash scripts/evaluate_matrix.sh outputs/runs/partner_diversity_simple \
  --partner-run-dir outputs/runs/baseline_simple \
  --layout simple --layout simple_tomato --layout small_corridor --layout random0 --layout random1 --layout unident_s \
  --output-name zero_shot_layouts

bash scripts/train.sh configs/multi_layout_curriculum.json

bash scripts/evaluate_matrix.sh outputs/runs/multi_layout_curriculum \
  --partner-run-dir outputs/runs/multi_layout_curriculum \
  --layout simple --layout small_corridor --layout random0 --layout random1 --layout unident_s --layout simple_tomato \
  --output-name zero_shot_layouts

bash scripts/record_demo.sh outputs/runs/multi_layout_curriculum --layout random0 --output-name random0_demo --max-steps 400
bash scripts/record_demo.sh outputs/runs/multi_layout_curriculum --layout simple --output-name simple_demo --max-steps 400

bash scripts/train_curriculum.sh configs/curriculum_simple_random0.json

bash scripts/evaluate_matrix.sh outputs/runs/curriculum_simple_random0 \
  --partner-run-dir outputs/runs/curriculum_simple_random0 \
  --layout simple --layout random0 --layout small_corridor --layout random1 --layout unident_s --layout simple_tomato \
  --output-name zero_shot_layouts

bash scripts/train.sh configs/baseline_random0.json

bash scripts/evaluate_matrix.sh outputs/runs/baseline_random0 \
  --partner-run-dir outputs/runs/baseline_random0 \
  --layout random0 --layout simple --layout small_corridor --layout random1 --layout unident_s --layout simple_tomato \
  --output-name zero_shot_layouts

bash scripts/evaluate_router.sh configs/router_simple_random0.json

bash scripts/train.sh configs/baseline_random0_long.json

bash scripts/evaluate_matrix.sh outputs/runs/baseline_random0_long \
  --partner-run-dir outputs/runs/baseline_random0_long \
  --layout random0 --layout simple --layout small_corridor --layout random1 --layout unident_s --layout simple_tomato \
  --output-name zero_shot_layouts

bash scripts/evaluate_router.sh configs/router_simple_random0.json \
  --route random0=outputs/runs/baseline_random0_long \
  --output-dir outputs/runs/router_simple_random0_long \
  --output-name router_eval

bash scripts/record_demo.sh outputs/runs/baseline_random0_long --layout random0 --output-name random0_long_demo --max-steps 400

bash scripts/train.sh configs/baseline_small_corridor.json

bash scripts/evaluate_matrix.sh outputs/runs/baseline_small_corridor \
  --partner-run-dir outputs/runs/baseline_small_corridor \
  --layout small_corridor --layout simple --layout random0 --layout random1 --layout unident_s --layout simple_tomato \
  --output-name zero_shot_layouts

bash scripts/train.sh configs/small_corridor_shaping_v1.json

bash scripts/evaluate_matrix.sh outputs/runs/small_corridor_shaping_v1 \
  --partner-run-dir outputs/runs/small_corridor_shaping_v1 \
  --layout small_corridor --layout simple --layout random0 --layout random1 --layout unident_s --layout simple_tomato \
  --output-name zero_shot_layouts

bash scripts/train.sh configs/baseline_random1.json

bash scripts/evaluate_matrix.sh outputs/runs/baseline_random1 \
  --partner-run-dir outputs/runs/baseline_random1 \
  --layout random1 --layout simple --layout random0 --layout small_corridor --layout unident_s --layout simple_tomato \
  --output-name zero_shot_layouts

bash scripts/train.sh configs/baseline_unident_s.json

bash scripts/evaluate_matrix.sh outputs/runs/baseline_unident_s \
  --partner-run-dir outputs/runs/baseline_unident_s \
  --layout unident_s --layout simple --layout random0 --layout random1 --layout small_corridor --layout simple_tomato \
  --output-name zero_shot_layouts

bash scripts/evaluate_router.sh configs/router_simple_random0.json \
  --route random0=outputs/runs/baseline_random0_long \
  --route random1=outputs/runs/baseline_random1 \
  --route unident_s=outputs/runs/baseline_unident_s \
  --output-dir outputs/runs/router_onion_layouts \
  --output-name router_eval

bash scripts/record_demo.sh outputs/runs/baseline_random1 --layout random1 --output-name random1_demo --max-steps 400
bash scripts/record_demo.sh outputs/runs/baseline_unident_s --layout unident_s --output-name unident_s_demo --max-steps 400

bash scripts/trace_episode.sh outputs/runs/baseline_small_corridor --layout small_corridor --output-name small_corridor_trace --max-steps 400
bash scripts/trace_episode.sh outputs/runs/small_corridor_shaping_v1 --layout small_corridor --output-name small_corridor_shaping_trace --max-steps 400
bash scripts/trace_episode.sh outputs/runs/baseline_random1 --layout random1 --output-name random1_trace --max-steps 400

bash scripts/train.sh configs/baseline_random0_long_seed52.json
bash scripts/train.sh configs/baseline_random1_seed71.json
bash scripts/train.sh configs/baseline_unident_s_seed81.json

bash scripts/evaluate_matrix.sh outputs/runs/baseline_random0_long \
  --partner-run-dir outputs/runs/baseline_random0_long \
  --partner-run-dir outputs/runs/baseline_random0_long_seed52 \
  --layout random0 \
  --output-name partner_matrix_hard_random0
bash scripts/evaluate_matrix.sh outputs/runs/baseline_random0_long_seed52 \
  --partner-run-dir outputs/runs/baseline_random0_long \
  --partner-run-dir outputs/runs/baseline_random0_long_seed52 \
  --layout random0 \
  --output-name partner_matrix_hard_random0

bash scripts/evaluate_matrix.sh outputs/runs/baseline_random1 \
  --partner-run-dir outputs/runs/baseline_random1 \
  --partner-run-dir outputs/runs/baseline_random1_seed71 \
  --layout random1 \
  --output-name partner_matrix_hard_random1
bash scripts/evaluate_matrix.sh outputs/runs/baseline_random1_seed71 \
  --partner-run-dir outputs/runs/baseline_random1 \
  --partner-run-dir outputs/runs/baseline_random1_seed71 \
  --layout random1 \
  --output-name partner_matrix_hard_random1

bash scripts/evaluate_matrix.sh outputs/runs/baseline_unident_s \
  --partner-run-dir outputs/runs/baseline_unident_s \
  --partner-run-dir outputs/runs/baseline_unident_s_seed81 \
  --layout unident_s \
  --output-name partner_matrix_hard_unident_s
bash scripts/evaluate_matrix.sh outputs/runs/baseline_unident_s_seed81 \
  --partner-run-dir outputs/runs/baseline_unident_s \
  --partner-run-dir outputs/runs/baseline_unident_s_seed81 \
  --layout unident_s \
  --output-name partner_matrix_hard_unident_s

bash scripts/evaluate_router.sh configs/router_simple_random0.json \
  --route simple=outputs/runs/curriculum_simple_random0 \
  --route random0=outputs/runs/baseline_random0_long_seed52 \
  --route random1=outputs/runs/baseline_random1 \
  --route unident_s=outputs/runs/baseline_unident_s \
  --output-dir outputs/runs/router_onion_layouts_seed52_random0 \
  --output-name router_eval

bash scripts/train.sh configs/small_corridor_structured_shaping_v1.json
bash scripts/train.sh configs/small_corridor_structured_shaping_v2.json
bash scripts/train.sh configs/small_corridor_structured_shaping_v3.json

bash scripts/trace_episode.sh outputs/runs/small_corridor_structured_shaping_v3 \
  --layout small_corridor \
  --output-name small_corridor_structured_trace \
  --max-steps 400

conda run --no-capture-output -n overcooked-marl python -m overcooked_project.evaluate \
  --run-dir outputs/runs/small_corridor_structured_shaping_v3 \
  --episodes 20 \
  --stochastic \
  --output-name eval_stochastic

bash scripts/trace_episode.sh outputs/runs/small_corridor_structured_shaping_v3 \
  --layout small_corridor \
  --output-name small_corridor_structured_stochastic_trace \
  --max-steps 400 \
  --stochastic

bash scripts/train_curriculum.sh configs/small_corridor_delivery_warmstart_from_v3.json

bash scripts/collect_delivery_demos.sh \
  --episodes 100 \
  --max-steps 40 \
  --output outputs/demos/small_corridor_delivery_scripted.json

bash scripts/train_delivery_bc.sh \
  --run-name small_corridor_delivery_bc_from_v3 \
  --epochs 40 \
  --batch-size 128 \
  --learning-rate 0.001

bash scripts/evaluate.sh outputs/runs/small_corridor_delivery_bc_from_v3 \
  --episodes 20 \
  --output-name eval_delivery_warmstart

bash scripts/evaluate.sh outputs/runs/small_corridor_delivery_bc_from_v3 \
  --episodes 20 \
  --standard-start \
  --horizon 400 \
  --output-name eval_standard_start_h400

bash scripts/train_curriculum.sh configs/small_corridor_delivery_warmstart_from_v3.json \
  --run-name small_corridor_delivery_bc_ppo_finetune \
  --init-run-dir outputs/runs/small_corridor_delivery_bc_from_v3 \
  --timesteps 50000 \
  --eval-interval 25000 \
  --eval-episodes 20

bash scripts/collect_delivery_demos.sh \
  --config configs/baseline_small_corridor.json \
  --mode full_chain \
  --episodes 100 \
  --max-steps 180 \
  --output outputs/demos/small_corridor_full_chain_scripted.json

bash scripts/train_delivery_bc.sh \
  --config configs/baseline_small_corridor.json \
  --init-run-dir outputs/runs/small_corridor_structured_shaping_v3 \
  --demo-path outputs/demos/small_corridor_full_chain_scripted.json \
  --run-name small_corridor_full_chain_bc_from_v3 \
  --epochs 60 \
  --batch-size 256 \
  --learning-rate 0.001

bash scripts/evaluate.sh outputs/runs/small_corridor_full_chain_bc_from_v3 \
  --episodes 20 \
  --horizon 400 \
  --output-name eval_standard_start_h400

bash scripts/trace_episode.sh outputs/runs/small_corridor_full_chain_bc_from_v3 \
  --horizon 400 \
  --output-name standard_start_h400_trace \
  --max-steps 400

bash scripts/train_curriculum.sh configs/baseline_small_corridor.json \
  --run-name small_corridor_full_chain_bc_ppo_finetune \
  --init-run-dir outputs/runs/small_corridor_full_chain_bc_from_v3 \
  --timesteps 50000 \
  --eval-interval 25000 \
  --eval-episodes 20

bash scripts/evaluate_router.sh configs/router_simple_random0.json \
  --route simple=outputs/runs/curriculum_simple_random0 \
  --route random0=outputs/runs/baseline_random0_long_seed52 \
  --route random1=outputs/runs/baseline_random1 \
  --route unident_s=outputs/runs/baseline_unident_s \
  --route small_corridor=outputs/runs/small_corridor_full_chain_bc_from_v3 \
  --output-dir outputs/runs/router_onion_layouts_with_small_corridor_bc \
  --output-name router_eval
```

## Run Summary

| Run | Seed | Timesteps | Train seconds | Mean soups | Mean sparse reward | Mean episode reward | Demo soups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_simple` | 10 | 200000 | 80.13 | 7.50 | 150.0 | 285.40 | 7.0 |
| `no_shaping_simple` | 10 | 200000 | 79.38 | 0.00 | 0.0 | 0.00 | 0.0 |
| `distance_shaping_simple` | 10 | 200000 | 80.57 | 7.50 | 150.0 | 285.40 | 7.0 |
| `baseline_seed11` | 11 | 200000 | 86.07 | 7.85 | 157.0 | 289.75 | - |
| `baseline_seed12` | 12 | 200000 | 87.21 | 10.30 | 206.0 | 385.15 | - |
| `partner_diversity_simple` | 20 | 200000 | 70.77 | 7.30 | 146.0 | 277.10 | 8.0 |
| `multi_layout_curriculum` | 30 | 500000 | 249.45 | 0.00 | 0.0 | 6.25 | 0.0 |
| `curriculum_simple_random0` | 40 | 300000 | 165.43 | 9.85 / 0.00 | 197.0 / 0.0 | 373.70 / 0.00 | 10.0 / 0.0 |
| `baseline_random0` | 50 | 300000 | 127.26 | 0.85 | 17.0 | 47.70 | 0.0 |
| `router_simple_random0` | - | 0 | - | 9.55 / 0.85 | 191.0 / 17.0 | 360.40 / 47.70 | - |
| `baseline_random0_long` | 50 | 800000 | 306.26 | 6.30 | 126.0 | 244.45 | 5.0 |
| `baseline_random0_long_seed52` | 52 | 800000 | 319.96 | 8.85 | 177.0 | 339.45 | - |
| `router_simple_random0_long` | - | 0 | - | 9.55 / 6.30 | 191.0 / 126.0 | 360.40 / 244.45 | - |
| `baseline_small_corridor` | 60 | 300000 | 118.54 | 0.00 | 0.0 | 0.00 | - |
| `small_corridor_shaping_v1` | 61 | 300000 | 116.26 | 0.00 | 0.0 | 0.00 | - |
| `small_corridor_structured_shaping_v1` | 62 | 300000 | 120.12 | 0.00 | 0.0 | 134.15 | - |
| `small_corridor_structured_shaping_v2` | 63 | 300000 | 120.98 | 0.00 | 0.0 | 18.41 | - |
| `small_corridor_structured_shaping_v3` | 64 | 300000 | 122.02 | 0.00 | 0.0 | 93.46 | - |
| `small_corridor_delivery_warmstart_from_v3` | 65 | 100000 | 46.51 | 0.00 | 0.0 | 73.76 | - |
| `small_corridor_delivery_bc_from_v3` | 65 | BC 40 epochs | - | 1.00 warm-start / 0.00 standard | 20.0 / 0.0 | 20.99 / 0.00 | - |
| `small_corridor_delivery_bc_ppo_finetune` | 65 | 50000 | 22.53 | 1.00 warm-start / 0.00 standard | 20.0 / 0.0 | 20.99 / 0.00 | - |
| `small_corridor_full_chain_bc_from_v3` | 60 | BC 60 epochs | - | 1.00 standard | 20.0 | 34.00 | 1.0 |
| `small_corridor_full_chain_bc_ppo_finetune` | 60 | 50000 | 26.46 | 1.00 best / 0.90 final | 20.0 / 18.0 | 34.00 / 31.50 | - |
| `small_corridor_full_chain_3cycle_bc_from_v3` | 60 | BC 60 epochs | - | 1.90 standard | 38.0 | 71.05 | 3.0 |
| `small_corridor_full_chain_3cycle_bc_ppo_finetune` | 60 | 50000 | 26.46 | 1.90 best / 1.55 final | 38.0 / 31.0 | 71.05 / 62.45 | - |
| `small_corridor_full_chain_3cycle_jitter3_bc_from_v3` | 60 | BC 60 epochs | - | 2.50 standard | 50.0 | 93.25 | 2.0 |
| `small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune` | 60 | 50000 | 30.50 | 3.00 best / 0.85 final | 60.0 / 17.0 | 111.00 / 29.65 | 3.0 |
| `baseline_random1` | 70 | 300000 | 116.94 | 5.80 | 116.0 | 225.10 | 6.0 |
| `baseline_random1_seed71` | 71 | 300000 | 116.99 | 5.20 | 104.0 | 202.80 | - |
| `baseline_unident_s` | 80 | 300000 | 118.35 | 12.70 | 254.0 | 481.25 | 13.0 |
| `baseline_unident_s_seed81` | 81 | 300000 | 118.88 | 12.60 | 252.0 | 470.30 | - |
| `router_onion_layouts` | - | 0 | - | 9.55 / 6.30 / 5.80 / 12.70 | 191.0 / 126.0 / 116.0 / 254.0 | 360.40 / 244.45 / 225.10 / 481.25 | - |
| `router_onion_layouts_seed52_random0` | - | 0 | - | 9.55 / 8.85 / 5.80 / 12.70 | 191.0 / 177.0 / 116.0 / 254.0 | 360.40 / 339.45 / 225.10 / 481.25 | - |
| `router_onion_layouts_with_small_corridor_bc` | - | 0 | - | 9.55 / 8.85 / 1.00 / 5.80 / 12.70 | 191.0 / 177.0 / 20.0 / 116.0 / 254.0 | 360.40 / 339.45 / 34.00 / 225.10 / 481.25 | - |
| `router_onion_layouts_with_small_corridor_3cycle_bc` | - | 0 | - | 9.55 / 8.85 / 1.90 / 5.80 / 12.70 | 191.0 / 177.0 / 38.0 / 116.0 / 254.0 | 360.40 / 339.45 / 71.05 / 225.10 / 481.25 | - |
| `router_onion_layouts_with_small_corridor_jitter3_bc_ppo` | - | 0 | - | 9.55 / 8.85 / 3.00 / 5.80 / 12.70 | 191.0 / 177.0 / 60.0 / 116.0 / 254.0 | 360.40 / 339.45 / 111.00 / 225.10 / 481.25 | - |

## Step 1: Reward-Shaping Ablation

| Run | Shaping setting | Mean soups | Interpretation |
| --- | --- | ---: | --- |
| `no_shaping_simple` | all shaping rewards disabled | 0.00 | Pure sparse reward is too hard at this training budget. The agents do not discover soup delivery. |
| `baseline_simple` | default shaping | 7.50 | Default shaping solves the basic `simple` layout reasonably well. |
| `distance_shaping_simple` | default shaping plus small distance rewards | 7.50 | Extra distance rewards did not improve final sparse performance over the default baseline. |

Conclusion: reward shaping is necessary for this short local training regime, but the tested distance shaping does not add measurable improvement.

## Step 2: Partner-Diversity Experiment

Fixed partner pool:

- `baseline_simple`
- `baseline_seed11`
- `baseline_seed12`

Evaluation matrix on `simple`:

| Ego | Partner | Mean soups | Mean sparse reward | Status |
| --- | --- | ---: | ---: | --- |
| `baseline_simple` | `baseline_simple` | 7.50 | 150.0 | ok |
| `baseline_simple` | `baseline_seed11` | 5.60 | 112.0 | ok |
| `baseline_simple` | `baseline_seed12` | 0.00 | 0.0 | ok |
| `partner_diversity_simple` | `baseline_simple` | 7.30 | 146.0 | ok |
| `partner_diversity_simple` | `baseline_seed11` | 6.50 | 130.0 | ok |
| `partner_diversity_simple` | `baseline_seed12` | 0.05 | 1.0 | ok |

Conclusion: partner-diversity training slightly improves compatibility with `baseline_seed11`, but it does not solve compatibility with the much stronger/different `baseline_seed12` partner. It also slightly lowers performance with the original baseline partner.

## Step 3: Zero-Shot Layout Generalization

Baseline ego with baseline partner:

| Layout | Mean soups | Mean sparse reward | Status | Error |
| --- | ---: | ---: | --- | --- |
| `simple` | 7.50 | 150.0 | ok | |
| `simple_tomato` | - | - | error | `KeyError: 'tomato'` |
| `small_corridor` | 0.00 | 0.0 | ok | |
| `random0` | 0.00 | 0.0 | ok | |
| `random1` | 0.00 | 0.0 | ok | |
| `unident_s` | 0.00 | 0.0 | ok | |

Partner-diversity ego with baseline partner:

| Layout | Mean soups | Mean sparse reward | Status | Error |
| --- | ---: | ---: | --- | --- |
| `simple` | 7.30 | 146.0 | ok | |
| `simple_tomato` | - | - | error | `KeyError: 'tomato'` |
| `small_corridor` | 0.00 | 0.0 | ok | |
| `random0` | 0.00 | 0.0 | ok | |
| `random1` | 0.00 | 0.0 | ok | |
| `unident_s` | 0.00 | 0.0 | ok | |

Conclusion: neither policy generalizes zero-shot beyond the training layout. The `simple_tomato` failure is not a learned-policy failure; it is an environment-stack limitation where the current Overcooked-AI featurizer used by this wrapper raises `KeyError: 'tomato'`.

## Step 4: Multi-Layout Curriculum

Training distribution:

- `simple`: 0.4
- `small_corridor`: 0.2
- `random0`: 0.2
- `random1`: 0.2

The environment wrapper samples one layout per episode. All selected layouts have the same 62-dimensional featurized observation shape, so one PPO policy can be shared across them. Tomato layouts are intentionally excluded from training because the current stack raises `KeyError: 'tomato'`.

Mixed-distribution evaluation from `train.sh`:

| Run | Mean soups | Mean sparse reward | Mean episode reward |
| --- | ---: | ---: | ---: |
| `multi_layout_curriculum` | 0.00 | 0.0 | 6.25 |

Fixed-layout self-play matrix:

| Layout | Mean soups | Mean sparse reward | Mean episode reward | Status | Error |
| --- | ---: | ---: | ---: | --- | --- |
| `simple` | 0.00 | 0.0 | 0.15 | ok | |
| `small_corridor` | 0.00 | 0.0 | 0.00 | ok | |
| `random0` | 0.10 | 2.0 | 19.20 | ok | |
| `random1` | 0.00 | 0.0 | 17.90 | ok | |
| `unident_s` | 0.00 | 0.0 | 0.00 | ok | |
| `simple_tomato` | - | - | - | error | `KeyError: 'tomato'` |

Conclusion: this first multi-layout curriculum did not solve layout generalization. It also regressed badly on `simple` compared with the single-layout baseline. The training curve reached high shaped reward near the end, but sparse delivery stayed almost zero. This suggests the current reward shaping and curriculum are teaching partial behaviors, not a reliable full delivery sequence.

## Step 5: Staged Simple + Random0 Fine-Tuning

This run loads `outputs/runs/baseline_simple`, fine-tunes on `simple + random0`, and evaluates both fixed layouts every 50k steps. The curriculum score is `min_soups_across_layouts + 0.05 * mean_soups_across_layouts`, so any improvement on the weaker layout is prioritized, with average soups as a tie-breaker. The best checkpoint is kept as `models/ego.zip` and `models/alt.zip`; the final checkpoint is saved separately.

Training distribution:

- `simple`: 0.8
- `random0`: 0.2

Curriculum checkpoints:

| Trained steps | Simple soups | Random0 soups | Score |
| ---: | ---: | ---: | ---: |
| 0 | 7.50 | 0.00 | 0.1875 |
| 50000 | 8.60 | 0.00 | 0.2150 |
| 100000 | 3.15 | 0.00 | 0.0788 |
| 150000 | 9.85 | 0.00 | 0.2463 |
| 200000 | 9.55 | 0.00 | 0.2388 |
| 250000 | 9.75 | 0.00 | 0.2438 |
| 300000 | 9.85 | 0.00 | 0.2463 |

Best checkpoint fixed-layout matrix:

| Layout | Mean soups | Mean sparse reward | Mean episode reward | Status | Error |
| --- | ---: | ---: | ---: | --- | --- |
| `simple` | 9.55 | 191.0 | 360.40 | ok | |
| `random0` | 0.00 | 0.0 | 0.00 | ok | |
| `small_corridor` | 0.00 | 0.0 | 0.00 | ok | |
| `random1` | 0.00 | 0.0 | 0.00 | ok | |
| `unident_s` | 0.00 | 0.0 | 0.90 | ok | |
| `simple_tomato` | - | - | - | error | `KeyError: 'tomato'` |

Conclusion: staged fine-tuning from the `simple` expert improves `simple` from 7.5 to about 9.6-9.9 soups, but it still never opens sparse reward on `random0`. This means direct fine-tuning from the `simple` policy does not discover the `random0` delivery routine, even when `random0` appears in the training distribution.

## Step 6: Random0 Single-Layout Diagnostic

This run trains a new self-play specialist directly on `random0`.

| Layout | Mean soups | Mean sparse reward | Mean episode reward | Status | Error |
| --- | ---: | ---: | ---: | --- | --- |
| `random0` | 0.85 | 17.0 | 47.70 | ok | |
| `simple` | 0.00 | 0.0 | 3.60 | ok | |
| `small_corridor` | 0.00 | 0.0 | 0.00 | ok | |
| `random1` | 0.00 | 0.0 | 0.30 | ok | |
| `unident_s` | 0.00 | 0.0 | 1.05 | ok | |
| `simple_tomato` | - | - | - | error | `KeyError: 'tomato'` |

Conclusion: `random0` is learnable with the current environment and default shaping, but it needs map-specific training. The problem is not that `random0` has no signal; the problem is that mixing it into a `simple` expert does not transfer the right behavior. The next optimization should use map specialists as curriculum anchors instead of expecting one expert to absorb a new layout by naive mixing.

## Step 7: Layout-Router Baseline

This run does not train a new neural policy. It evaluates a simple policy-selection baseline: choose a specialist run from the current layout name, then run that specialist's ego and partner policies on the selected layout. This is useful as a specialist-composition baseline, not as evidence that one unified policy generalizes.

Routes:

| Layout | Selected run | Reason |
| --- | --- | --- |
| `simple` | `outputs/runs/curriculum_simple_random0` | Best current `simple` score from staged curriculum |
| `random0` | `outputs/runs/baseline_random0` | Only current specialist with nonzero `random0` sparse reward |

Router evaluation:

| Layout | Selected run | Mean soups | Mean sparse reward | Mean episode reward | Status | Error |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `simple` | `curriculum_simple_random0` | 9.55 | 191.0 | 360.40 | ok | |
| `random0` | `baseline_random0` | 0.85 | 17.0 | 47.70 | ok | |
| `small_corridor` | - | - | - | - | skipped | `NoRoute` |
| `random1` | - | - | - | - | skipped | `NoRoute` |
| `unident_s` | - | - | - | - | skipped | `NoRoute` |
| `simple_tomato` | - | - | - | - | skipped | `NoRoute` |

Conclusion: the router is the first evaluated setup that gets nonzero sparse reward on both `simple` and `random0`, with a supported-layout average of 5.20 soups and a supported-layout minimum of 0.85 soups. This confirms that the immediate bottleneck is not evaluation infrastructure but policy coverage: we need stronger specialists or a real layout-conditioned policy before claiming broad layout competence.

## Step 8: Longer Random0 Specialist

This run tests the Phase 1 plan: keep the same `random0` PPO setup, same seed, and same default shaping, but increase training from 300k to 800k steps. This isolates whether the weak `random0` result was mostly a training-budget issue.

Training setup:

| Run | Layout | Seed | Timesteps | Train seconds |
| --- | --- | ---: | ---: | ---: |
| `baseline_random0_long` | `random0` | 50 | 800000 | 306.26 |

Self-play evaluation:

| Layout | Mean soups | Mean sparse reward | Mean episode reward | Status | Error |
| --- | ---: | ---: | ---: | --- | --- |
| `random0` | 6.30 | 126.0 | 244.45 | ok | |
| `simple` | 0.00 | 0.0 | 0.30 | ok | |
| `small_corridor` | 0.00 | 0.0 | 0.00 | ok | |
| `random1` | 0.00 | 0.0 | 0.00 | ok | |
| `unident_s` | 0.00 | 0.0 | 0.00 | ok | |
| `simple_tomato` | - | - | - | error | `KeyError: 'tomato'` |

Updated router evaluation:

| Layout | Selected run | Mean soups | Mean sparse reward | Mean episode reward | Status | Error |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `simple` | `curriculum_simple_random0` | 9.55 | 191.0 | 360.40 | ok | |
| `random0` | `baseline_random0_long` | 6.30 | 126.0 | 244.45 | ok | |
| `small_corridor` | - | - | - | - | skipped | `NoRoute` |
| `random1` | - | - | - | - | skipped | `NoRoute` |
| `unident_s` | - | - | - | - | skipped | `NoRoute` |
| `simple_tomato` | - | - | - | - | skipped | `NoRoute` |

Conclusion: extra training budget strongly improves the `random0` specialist, from 0.85 to 6.30 soups. The updated router's supported-layout average rises from 5.20 to 7.925 soups, and its supported-layout minimum rises from 0.85 to 6.30 soups. This clears the Phase 1 success criterion, so the next step should move to Phase 2: train specialists for `small_corridor`, `random1`, and `unident_s`.

## Step 9: Phase 2 Specialist Coverage

This step trains the remaining planned onion-layout specialists. The goal is not to force one unified policy to generalize, but to check which layouts are learnable as single-layout self-play specialists and then expand the router only with successful specialists.

Training setup:

| Run | Layout | Seed | Timesteps | Train seconds | Shaping change |
| --- | --- | ---: | ---: | ---: | --- |
| `baseline_small_corridor` | `small_corridor` | 60 | 300000 | 118.54 | default shaping |
| `small_corridor_shaping_v1` | `small_corridor` | 61 | 300000 | 116.26 | adds distance shaping 0.1 |
| `baseline_random1` | `random1` | 70 | 300000 | 116.94 | default shaping |
| `baseline_unident_s` | `unident_s` | 80 | 300000 | 118.35 | default shaping |

Self-play evaluation:

| Run | Layout | Mean soups | Mean sparse reward | Mean episode reward | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `baseline_small_corridor` | `small_corridor` | 0.00 | 0.0 | 0.00 | Failed; do not route. |
| `small_corridor_shaping_v1` | `small_corridor` | 0.00 | 0.0 | 0.00 | Failed; distance shaping did not open exploration. |
| `baseline_random1` | `random1` | 5.80 | 116.0 | 225.10 | Success; add to router. |
| `baseline_unident_s` | `unident_s` | 12.70 | 254.0 | 481.25 | Success; add to router. |

Zero-shot pattern:

| Ego run | Training layout score | Other onion layouts | Interpretation |
| --- | ---: | --- | --- |
| `baseline_small_corridor` | 0.00 on `small_corridor` | 0.00 everywhere | No useful specialist emerged. |
| `small_corridor_shaping_v1` | 0.00 on `small_corridor` | 0.00 everywhere | Distance shaping did not help this layout. |
| `baseline_random1` | 5.80 on `random1` | 0.00 on `random0`, `small_corridor`, `unident_s` | Strong specialist, no zero-shot transfer. |
| `baseline_unident_s` | 12.70 on `unident_s` | 0.00 on `simple`, `random0`, `random1`, `small_corridor` | Strong specialist, no meaningful transfer. |

Expanded router evaluation:

| Layout | Selected run | Mean soups | Mean sparse reward | Mean episode reward | Status | Error |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `simple` | `curriculum_simple_random0` | 9.55 | 191.0 | 360.40 | ok | |
| `random0` | `baseline_random0_long` | 6.30 | 126.0 | 244.45 | ok | |
| `small_corridor` | - | - | - | - | skipped | `NoRoute` |
| `random1` | `baseline_random1` | 5.80 | 116.0 | 225.10 | ok | |
| `unident_s` | `baseline_unident_s` | 12.70 | 254.0 | 481.25 | ok | |
| `simple_tomato` | - | - | - | - | skipped | `NoRoute` |

The supported-layout average is 8.59 soups and the supported-layout minimum is 5.80 soups over `simple`, `random0`, `random1`, and `unident_s`.

Conclusion: Phase 2 strongly supports the specialist-routing story. `random1` and `unident_s` are learnable with the same 300k default PPO setup, while `small_corridor` remains the hard unresolved layout even after adding simple distance shaping. The router now covers four useful layouts with nonzero sparse reward, but the project should be honest that `small_corridor` still needs a different optimization strategy.

## Step 10: Small Corridor Trace Diagnosis

This step adds a lightweight episode-tracing tool and compares two failed `small_corridor` policies against a successful `random1` specialist. The trace records actions, rewards, player positions, held objects, world objects, and text-state snapshots for each step.

Trace commands:

```bash
bash scripts/trace_episode.sh outputs/runs/baseline_small_corridor --layout small_corridor --output-name small_corridor_trace --max-steps 400
bash scripts/trace_episode.sh outputs/runs/small_corridor_shaping_v1 --layout small_corridor --output-name small_corridor_shaping_trace --max-steps 400
bash scripts/trace_episode.sh outputs/runs/baseline_random1 --layout random1 --output-name random1_trace --max-steps 400
```

Trace summary:

| Trace | Reward events | First reward step | Total reward | Sparse reward | Soups | Key behavior |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline_small_corridor/traces/small_corridor_trace.json` | 0 | - | 0.0 | 0.0 | 0.0 | Ego only moves left/down; no ego `interact`; final state has no held objects and no world objects. |
| `small_corridor_shaping_v1/traces/small_corridor_shaping_trace.json` | 0 | - | 0.0 | 0.0 | 0.0 | Ego uses `interact` 219 times, but still never picks up or places an object. |
| `baseline_random1/traces/random1_trace.json` | 37 | 7 | 227.0 | 120.0 | 6.0 | Successful policy gets shaped reward by step 7 and completes 6 soups. |

Action-count details:

| Trace | Ego action counts |
| --- | --- |
| `baseline_small_corridor` | `(-1, 0)`: 118, `(0, 1)`: 282 |
| `small_corridor_shaping_v1` | `interact`: 219, `(0, 0)`: 117, `(-1, 0)`: 44, `(0, -1)`: 20 |
| `baseline_random1` | `(-1, 0)`: 79, `(0, -1)`: 102, `(0, 1)`: 84, `(1, 0)`: 47, `interact`: 87, `(0, 0)`: 1 |

Conclusion: the `small_corridor` failure happens before the soup-delivery chain even begins. The default policy never interacts, while the distance-shaping policy interacts frequently but not at useful stations. This points to an exploration/subgoal-discovery problem, not merely insufficient final-task reward. The next `small_corridor` attempt should use a more structured approach: scripted warm start, curriculum over starting positions/subtasks, or a stronger layout-specific shaping signal around first valid pickup and first valid placement.

## Step 11: Hard-Layout Partner Robustness

This step trains held-out partner seeds for the successful hard layouts and evaluates reciprocal partner matrices. It also re-evaluates the onion-layout router after replacing the `random0` route with the stronger seed-52 long specialist.

Additional specialist seeds:

| Run | Layout | Seed | Timesteps | Mean soups | Mean sparse reward | Mean episode reward | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline_random0_long_seed52` | `random0` | 52 | 800000 | 8.85 | 177.0 | 339.45 | Stronger than `baseline_random0_long`; use as the preferred `random0` route. |
| `baseline_random1_seed71` | `random1` | 71 | 300000 | 5.20 | 104.0 | 202.80 | Similar self-play to `baseline_random1`; useful held-out partner. |
| `baseline_unident_s_seed81` | `unident_s` | 81 | 300000 | 12.60 | 252.0 | 470.30 | Matches `baseline_unident_s`; useful held-out partner. |

Reciprocal partner matrices:

| Layout | Ego | Partner | Mean soups | Mean sparse reward | Interpretation |
| --- | --- | --- | ---: | ---: | --- |
| `random0` | `baseline_random0_long` | `baseline_random0_long` | 6.30 | 126.0 | Original self-play baseline. |
| `random0` | `baseline_random0_long` | `baseline_random0_long_seed52` | 1.65 | 33.0 | Old ego is brittle with the new partner. |
| `random0` | `baseline_random0_long_seed52` | `baseline_random0_long` | 7.70 | 154.0 | New ego remains strong with the old partner. |
| `random0` | `baseline_random0_long_seed52` | `baseline_random0_long_seed52` | 8.85 | 177.0 | New self-play is the best `random0` result so far. |
| `random1` | `baseline_random1` | `baseline_random1` | 5.80 | 116.0 | Original self-play baseline. |
| `random1` | `baseline_random1` | `baseline_random1_seed71` | 0.25 | 5.0 | Strong collapse under partner mismatch. |
| `random1` | `baseline_random1_seed71` | `baseline_random1` | 0.00 | 0.0 | Mismatch collapse is reciprocal. |
| `random1` | `baseline_random1_seed71` | `baseline_random1_seed71` | 5.20 | 104.0 | Both seeds work in self-play but not cross-play. |
| `unident_s` | `baseline_unident_s` | `baseline_unident_s` | 12.70 | 254.0 | Original self-play baseline. |
| `unident_s` | `baseline_unident_s` | `baseline_unident_s_seed81` | 12.60 | 252.0 | Robust to held-out partner. |
| `unident_s` | `baseline_unident_s_seed81` | `baseline_unident_s` | 12.65 | 253.0 | Robust in the opposite direction too. |
| `unident_s` | `baseline_unident_s_seed81` | `baseline_unident_s_seed81` | 12.60 | 252.0 | Held-out self-play matches the original. |

Updated onion-router evaluation:

| Layout | Selected run | Mean soups | Mean sparse reward | Status |
| --- | --- | ---: | ---: | --- |
| `simple` | `curriculum_simple_random0` | 9.55 | 191.0 | ok |
| `random0` | `baseline_random0_long_seed52` | 8.85 | 177.0 | ok |
| `small_corridor` | - | - | - | skipped `NoRoute` |
| `random1` | `baseline_random1` | 5.80 | 116.0 | ok |
| `unident_s` | `baseline_unident_s` | 12.70 | 254.0 | ok |
| `simple_tomato` | - | - | - | skipped `NoRoute` |

The supported-layout average improves from 8.59 to 9.23 soups, while the supported-layout minimum stays 5.80 soups because `random1` is now the weakest routed layout.

Conclusion: partner robustness is layout-specific, not guaranteed by good self-play. `unident_s` is robust across two strong seeds, `random1` is strongly partner-brittle, and `random0` is asymmetric: the new seed-52 ego is robust to the old partner, but the old ego is not robust to the new partner. For the main router result, `baseline_random0_long_seed52` should replace `baseline_random0_long`.

## Step 12: Structured Small Corridor Shaping

This step tests a more explicit `small_corridor` shaping signal after the trace diagnosis showed that the original specialists did not discover useful pickups or placements. The first important implementation finding is that the upstream Overcooked-AI distance reward path is not active in this checkout: `calculate_distance_based_shaped_reward` exists, but the call in `overcooked_mdp.py` is commented out. Therefore, the earlier `small_corridor_shaping_v1` config did not actually add distance reward despite setting `DISH_DISP_DISTANCE_REW`, `POT_DISTANCE_REW`, and `SOUP_DISTANCE_REW`.

Code changes:

- Added optional `event_reward_shaping` to the environment wrapper without changing default configs.
- Added source-pickup reward, pot-placement reward, and layout-aware distance-progress reward.
- Added orientation-aware interaction distance and a dynamic empty-handed target that switches from onion source to dish source once a pot has enough ingredients.
- Added anti-farming switches to skip counter pickup rewards and skip target-switch progress on held-object changes.
- Extended traces with `base_shaped_reward` and `event_shaped_reward` so shaped reward can be audited.

Training setup:

| Run | Seed | Shaping idea | Anti-farming | Timesteps |
| --- | ---: | --- | --- | ---: |
| `small_corridor_structured_shaping_v1` | 62 | pickup reward plus positive distance progress to onion/dish/pot/serve | no | 300000 |
| `small_corridor_structured_shaping_v2` | 63 | orientation-aware distance and dish target after pot has 3 onions | no | 300000 |
| `small_corridor_structured_shaping_v3` | 64 | v2 plus source-only pickup reward and no progress reward on held-object changes | yes | 300000 |

Evaluation summary:

| Run | Deterministic soups | Deterministic episode reward | Stochastic soups | Stochastic episode reward | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `small_corridor_structured_shaping_v1` | 0.00 | 134.15 | 0.00 | 251.44 | High shaped reward, but no pot progress; deterministic policy mostly spams `interact`. |
| `small_corridor_structured_shaping_v2` | 0.00 | 18.41 | 0.00 | 389.82 | Deterministic policy collapses to moving right; stochastic policy farms onion/progress reward without soup delivery. |
| `small_corridor_structured_shaping_v3` | 0.00 | 93.46 | 0.00 | 95.74 | Cleaner reward after anti-farming; reaches soup pickup but still never serves soup. |

Trace diagnosis:

| Trace | Total reward | Base shaped | Event shaped | Sparse reward | Key behavior |
| --- | ---: | ---: | ---: | ---: | --- |
| `small_corridor_structured_shaping_v1/traces/small_corridor_structured_trace.json` | 89.3 | - | - | 0.0 | Deterministic ego uses `interact` 398 times; dishes appear, but no pot or soup states appear. |
| `small_corridor_structured_shaping_v1/traces/small_corridor_structured_stochastic_trace.json` | 208.9 | - | - | 0.0 | Stochastic policy can pick up onion and dish, but still never creates pot progress. |
| `small_corridor_structured_shaping_v2/traces/small_corridor_structured_trace.json` | 0.0 | 0.0 | 0.0 | 0.0 | Deterministic ego moves right for all 400 steps. |
| `small_corridor_structured_shaping_v2/traces/small_corridor_structured_stochastic_trace_v2breakdown.json` | 451.6 | 0.0 | 451.6 | 0.0 | Reward is entirely custom event shaping; both agents pick onions but do not produce soup. |
| `small_corridor_structured_shaping_v3/traces/small_corridor_structured_trace.json` | 98.9 | 68.0 | 30.9 | 0.0 | Both agents pick onions, pots reach 3 onions, partner picks up soup at step 93, but soup is dropped or carried around the top corridor instead of served. |
| `small_corridor_structured_shaping_v3/traces/small_corridor_structured_stochastic_trace.json` | 89.5 | 57.0 | 32.5 | 0.0 | Stochastic policy also reaches soup pickup at step 82, then fails the long delivery to the left-side serving station. |

Conclusion: structured shaping improves the failure mode, but it does not solve `small_corridor`. The bottleneck moved from "no useful subgoal" to "soup pickup without serving." Continuing to tune generic reward coefficients is likely low-yield. The next attempt should be a delivery-stage curriculum or scripted warm start, for example by training from states where a soup has already been picked up, collecting scripted trajectories for the final delivery segment, or using behavior cloning/subtask pretraining before PPO.

## Step 13: Small Corridor Delivery Warm Start And Scripted Demos

This step turns the Step 12 diagnosis into a targeted delivery-stage experiment. The environment wrapper now supports an optional `start_state_mode`. For `small_corridor_delivery`, each episode starts with one randomly selected agent holding a cooked onion soup somewhere along the delivery corridor at `y=3`. This includes positions near the serving station and positions near the pots, so the subtask ranges from final turn-and-interact to the full right-to-left delivery walk.

The training run uses `train_curriculum.py` to continue from `small_corridor_structured_shaping_v3`, rather than starting from scratch.

Training setup:

| Run | Init run | Start mode | Horizon | Timesteps | Reward signal |
| --- | --- | --- | ---: | ---: | --- |
| `small_corridor_delivery_warmstart_from_v3` | `small_corridor_structured_shaping_v3` | `small_corridor_delivery` | 120 | 100000 | sparse delivery plus held-soup progress to serving |

Warm-start curriculum evaluation:

| Trained steps | Mean episode reward | Mean sparse reward | Mean soups | Decision |
| ---: | ---: | ---: | ---: | --- |
| 0 | 34.07 | 0.0 | 0.00 | Initial v3 checkpoint does not deliver even when starting with soup. |
| 25000 | 41.26 | 0.0 | 0.00 | Progress reward rises, sparse still absent. |
| 50000 | 56.44 | 0.0 | 0.00 | More shaped/progress reward, still no delivery. |
| 75000 | 77.02 | 0.0 | 0.00 | Best warm-start episode reward, still no sparse reward. |
| 100000 | 73.76 | 0.0 | 0.00 | PPO warm start fails the final serving action. |

Conclusion from PPO warm start: starting closer to the delivery subtask is still not enough. The agents learn to increase progress-shaped reward, but deterministic evaluation never triggers a soup delivery. This is a useful negative result because it rules out a simple "just start from soup-held states" fix.

Scripted delivery demos:

| Artifact | Episodes | Successes | Success rate | Mean success steps | Meaning |
| --- | ---: | ---: | ---: | ---: | --- |
| `outputs/demos/small_corridor_delivery_scripted.json` | 100 | 100 | 1.00 | 6.90 | The delivery subtask is mechanically feasible and can be scripted reliably. |

The scripted controller is intentionally simple: the soup-holding agent moves west along the corridor to `(1, 3)`, turns south toward the serving station, then interacts. The other agent stays still. This creates a clean demonstration dataset for the exact failure segment observed in Step 12 and Step 13.

Next decision: stop relying on PPO exploration for the final delivery segment. The next concrete implementation should use `outputs/demos/small_corridor_delivery_scripted.json` for behavior cloning or supervised action pretraining, then fine-tune with PPO from that policy.

## Step 14: Behavior Cloning For The Delivery Segment

This step turns the scripted delivery dataset into supervised policy pretraining. The demo collector now records each player's 62-dimensional Overcooked featurized observation before the scripted action, so the dataset can be used directly to train the PPO policy network with a negative log-likelihood loss over discrete actions.

BC setup:

| Run | Init run | Demo file | Epochs | Dataset steps | Ego val acc | Alt val acc |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `small_corridor_delivery_bc_from_v3` | `small_corridor_structured_shaping_v3` | `outputs/demos/small_corridor_delivery_scripted.json` | 40 | 690 | 1.00 | 1.00 |

Behavioral evaluation:

| Run | Evaluation start | Horizon | Mean soups | Mean sparse reward | Mean episode reward | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `small_corridor_delivery_bc_from_v3` | `small_corridor_delivery` | 120 | 1.00 | 20.0 | 20.99 | BC successfully learns the isolated final delivery segment. |
| `small_corridor_delivery_bc_from_v3` | standard `small_corridor` | 400 | 0.00 | 0.0 | 0.00 | Delivery BC alone does not recover the earlier onion/pot/dish stages. |
| `small_corridor_delivery_bc_ppo_finetune` | `small_corridor_delivery` | 120 | 1.00 | 20.0 | 20.99 | PPO fine-tuning preserves the delivery skill. |
| `small_corridor_delivery_bc_ppo_finetune` | standard `small_corridor` | 400 | 0.00 | 0.0 | 0.00 | Fine-tuning on delivery states still does not transfer to the full start state. |

Trace diagnosis:

| Trace | Key behavior |
| --- | --- |
| `small_corridor_delivery_bc_from_v3/traces/delivery_warmstart_trace.json` | The soup holder delivers one soup. Sparse reward is 20.0 and the trace reaches 1.0 soup. |
| `small_corridor_delivery_bc_from_v3/traces/standard_start_h400_trace.json` | Ego selects `stay` for all 400 steps; no held objects, no world objects, no reward events. |
| `small_corridor_delivery_bc_ppo_finetune/traces/standard_start_h400_trace.json` | Same standard-start failure: no object pickup or placement happens. |

Conclusion: behavior cloning is effective for the exact demonstrated delivery subtask, and it proves that supervised subtask pretraining can put a useful skill into the PPO policy. However, narrow delivery-only BC overwrites or fails to activate the earlier cooking behavior from standard start. The next small-corridor attempt should train on a broader scripted subtask mixture: onion pickup, pot placement, dish pickup, soup pickup, and delivery, or use a hierarchical/subtask router that calls the delivery policy only after soup pickup.

## Step 15: Full-Chain Small Corridor Behavior Cloning

This step follows the Step 14 conclusion by collecting a full standard-start `small_corridor` trajectory rather than only the final delivery segment. The scripted plan assigns agent 0 to collect three onions and place them into the pot, while agent 1 gets a dish, waits for cooking, picks up soup, walks to the left serving station, turns south, and serves.

Scripted full-chain demos:

| Artifact | Episodes | Successes | Success rate | Mean success steps | Meaning |
| --- | ---: | ---: | ---: | ---: | --- |
| `outputs/demos/small_corridor_full_chain_scripted.json` | 100 | 100 | 1.00 | 122.0 | One full standard-start soup chain is mechanically feasible and can be demonstrated reliably. |

BC setup:

| Run | Init run | Demo file | Epochs | Dataset steps | Ego val acc | Alt val acc |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `small_corridor_full_chain_bc_from_v3` | `small_corridor_structured_shaping_v3` | `outputs/demos/small_corridor_full_chain_scripted.json` | 60 | 12200 | 1.00 | 0.9873 |

Standard-start evaluation:

| Run | Horizon | Mean soups | Mean sparse reward | Mean episode reward | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `small_corridor_full_chain_bc_from_v3` | 400 | 1.00 | 20.0 | 34.00 | Full-chain BC solves the first standard-start soup. |
| `small_corridor_full_chain_bc_ppo_finetune` best | 400 | 1.00 | 20.0 | 34.00 | The best checkpoint remains the initial BC policy. |
| `small_corridor_full_chain_bc_ppo_finetune` final | 400 | 0.90 | 18.0 | 31.50 | PPO fine-tuning slightly degrades the deterministic policy by 50k steps. |

Trace and demo:

| Artifact | Result |
| --- | --- |
| `outputs/runs/small_corridor_full_chain_bc_from_v3/traces/standard_start_h400_trace.json` | 1.00 soup, sparse reward 20.0, shaped reward 14.0, first sparse reward at step 122. |
| `outputs/runs/small_corridor_full_chain_bc_ppo_finetune/traces/standard_start_h400_trace.json` | Same best-checkpoint behavior: 1.00 soup and sparse reward 20.0. |
| `outputs/runs/small_corridor_full_chain_bc_from_v3/demo/small_corridor_full_chain_bc_demo.gif` | Qualitative demo for the first standard-start soup chain. |

Router impact:

| Layout | Selected run | Mean soups | Mean sparse reward | Mean episode reward | Status |
| --- | --- | ---: | ---: | ---: | --- |
| `simple` | `curriculum_simple_random0` | 9.55 | 191.0 | 360.40 | ok |
| `random0` | `baseline_random0_long_seed52` | 8.85 | 177.0 | 339.45 | ok |
| `small_corridor` | `small_corridor_full_chain_bc_from_v3` | 1.00 | 20.0 | 34.00 | ok |
| `random1` | `baseline_random1` | 5.80 | 116.0 | 225.10 | ok |
| `unident_s` | `baseline_unident_s` | 12.70 | 254.0 | 481.25 | ok |
| `simple_tomato` | - | - | - | - | skipped `NoRoute` |

The five-onion-layout average is 7.58 soups and the minimum is 1.00 soup. This is the broadest current coverage result, but it mixes PPO specialists with one scripted-BC specialist. The PPO-only router remains the cleaner pure-RL comparison over four routed onion layouts: 9.23 average soups and 5.80 minimum soups, with `small_corridor` skipped.

Conclusion: full-chain BC changes `small_corridor` from a zero-sparse failure into a one-soup standard-start specialist. The important limitation is that this is imitation of one complete chain, not autonomous discovery of a repeatable cooking loop. PPO fine-tuning did not improve it at 50k steps, so the next optimization should either collect multi-cycle and perturbed full-chain demos, or treat this as a hierarchical/specialist component rather than pretending it is a pure PPO solution.

## Step 16: Multi-Cycle Small Corridor Behavior Cloning

This step follows the Step 15 limitation directly. The scripted full-chain collector now supports `--full-chain-cycles`, so the same standard-start plan can demonstrate repeated cooking and serving loops instead of stopping after the first soup.

Scripted 3-cycle demos:

| Artifact | Episodes | Successes | Success rate | Mean success steps | Meaning |
| --- | ---: | ---: | ---: | ---: | --- |
| `outputs/demos/small_corridor_full_chain_3cycle_scripted.json` | 100 | 100 | 1.00 | 376.0 | Three full standard-start soup cycles are mechanically feasible and can be demonstrated reliably. |

BC setup:

| Run | Init run | Demo file | Epochs | Dataset steps | Ego val acc | Alt val acc |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `small_corridor_full_chain_3cycle_bc_from_v3` | `small_corridor_structured_shaping_v3` | `outputs/demos/small_corridor_full_chain_3cycle_scripted.json` | 60 | 37600 | 1.00 | 0.9762 |

Standard-start evaluation:

| Run | Horizon | Mean soups | Mean sparse reward | Mean episode reward | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `small_corridor_full_chain_3cycle_bc_from_v3` | 400 | 1.90 | 38.0 | 71.05 | Multi-cycle BC improves over the one-soup BC and sometimes completes all three cycles. |
| `small_corridor_full_chain_3cycle_bc_ppo_finetune` best | 400 | 1.90 | 38.0 | 71.05 | The best checkpoint remains the initial BC policy. |
| `small_corridor_full_chain_3cycle_bc_ppo_finetune` 25k | 400 | 1.00 | 20.0 | 34.00 | PPO fine-tuning initially collapses back to a one-soup policy. |
| `small_corridor_full_chain_3cycle_bc_ppo_finetune` final 50k | 400 | 1.55 | 31.0 | 62.45 | PPO recovers some multi-cycle behavior but still underperforms BC-only. |

Episode-level distribution:

| Run/checkpoint | Soup count distribution over 20 deterministic episodes |
| --- | --- |
| `small_corridor_full_chain_3cycle_bc_from_v3` | 10 episodes with 1 soup, 2 with 2 soups, 8 with 3 soups |
| `small_corridor_full_chain_3cycle_bc_ppo_finetune` 25k | 20 episodes with 1 soup |
| `small_corridor_full_chain_3cycle_bc_ppo_finetune` final 50k | 4 episodes with 0 soups, 2 with 1 soup, 13 with 2 soups, 1 with 3 soups |

Trace and demo:

| Artifact | Result |
| --- | --- |
| `outputs/runs/small_corridor_full_chain_3cycle_bc_from_v3/traces/standard_start_h400_trace.json` | 3.00 soups in one deterministic trace; sparse rewards at steps 122, 249, and 384. |
| `outputs/runs/small_corridor_full_chain_3cycle_bc_from_v3/demo/small_corridor_full_chain_3cycle_bc_demo.gif` | Qualitative demo for a 3-cycle standard-start chain. |

Router impact:

| Layout | Selected run | Mean soups | Mean sparse reward | Mean episode reward | Status |
| --- | --- | ---: | ---: | ---: | --- |
| `simple` | `curriculum_simple_random0` | 9.55 | 191.0 | 360.40 | ok |
| `random0` | `baseline_random0_long_seed52` | 8.85 | 177.0 | 339.45 | ok |
| `small_corridor` | `small_corridor_full_chain_3cycle_bc_from_v3` | 1.90 | 38.0 | 71.05 | ok |
| `random1` | `baseline_random1` | 5.80 | 116.0 | 225.10 | ok |
| `unident_s` | `baseline_unident_s` | 12.70 | 254.0 | 481.25 | ok |
| `simple_tomato` | - | - | - | - | skipped `NoRoute` |

The five-onion-layout average improves from 7.58 to 7.76 soups, and the minimum improves from 1.00 to 1.90 soups. This is now the broadest current coverage result. It still mixes PPO specialists with one scripted-BC specialist, so the PPO-only four-layout router remains the clean pure-RL comparison.

Conclusion: multi-cycle BC is a real improvement over one-cycle BC: it raises `small_corridor` from 1.00 to 1.90 average soups and produces at least one clean 3-soup trace. The remaining issue is robustness rather than feasibility. The 3-cycle policy sometimes falls out of the repeated loop, and PPO fine-tuning does not fix that at 50k steps. The next `small_corridor` attempt should add perturbed demonstrations, role-balanced variants, or an explicit subtask/controller layer instead of expecting plain PPO fine-tuning to stabilize the loop.

## Step 17: Perturbed 3-Cycle Small Corridor BC And Checkpoint Selection

This step follows the Step 16 conclusion by widening the scripted data distribution. The demo collector now supports `--full-chain-wait-jitter`, which inserts a small random number of extra `stay` actions at safe synchronization points in the full-chain script. The goal is not to create new routes, but to teach the policy to recover from small timing offsets instead of replaying one exact clocked trajectory.

Scripted perturbed demos:

| Artifact | Episodes | Successes | Success rate | Mean success steps | Perturbation |
| --- | ---: | ---: | ---: | ---: | --- |
| `outputs/demos/small_corridor_full_chain_3cycle_jitter3_scripted.json` | 100 | 100 | 1.00 | 385.09 | `full_chain_cycles=3`, `full_chain_wait_jitter=3`, seed 91 |

BC setup:

| Run | Demo file | Epochs | Dataset steps | Ego val acc | Alt val acc |
| --- | --- | ---: | ---: | ---: | ---: |
| `small_corridor_full_chain_3cycle_jitter3_bc_from_v3` | `outputs/demos/small_corridor_full_chain_3cycle_jitter3_scripted.json` | 60 | 38509 | 1.00 | 0.9657 |

Standard-start evaluation:

| Run/checkpoint | Horizon | Mean soups | Mean sparse reward | Mean episode reward | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `small_corridor_full_chain_3cycle_jitter3_bc_from_v3` | 400 | 2.50 | 50.0 | 93.25 | Wait perturbations improve BC stability over the fixed 3-cycle demo. |
| `small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune` best at 25k | 400 | 3.00 | 60.0 | 111.00 | Checkpoint-selected PPO fine-tune solves the 3-soup horizon in all 20 eval episodes. |
| `small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune` final 50k | 400 | 0.85 | 17.0 | 29.65 | Continuing training past the best checkpoint collapses the policy. |

Episode-level distribution:

| Run/checkpoint | Soup count distribution over 20 deterministic episodes |
| --- | --- |
| Jitter BC | 3 episodes with 1 soup, 4 with 2 soups, 13 with 3 soups |
| PPO 25k best | 20 episodes with 3 soups |
| PPO 50k final | 3 episodes with 0 soups, 17 with 1 soup |

Trace and demo:

| Artifact | Result |
| --- | --- |
| `outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_from_v3/traces/standard_start_h400_trace.json` | 2.00 soups, sparse reward 40.0, shaped reward 37.0. |
| `outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune/traces/standard_start_h400_trace.json` | 3.00 soups, sparse reward 60.0, shaped reward 51.0. |
| `outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune/demo/small_corridor_full_chain_3cycle_jitter3_bc_ppo_best_demo.gif` | Qualitative demo for the checkpoint-selected 3-soup policy. |

Router impact:

| Layout | Selected run | Mean soups | Mean sparse reward | Mean episode reward | Status |
| --- | --- | ---: | ---: | ---: | --- |
| `simple` | `curriculum_simple_random0` | 9.55 | 191.0 | 360.40 | ok |
| `random0` | `baseline_random0_long_seed52` | 8.85 | 177.0 | 339.45 | ok |
| `small_corridor` | `small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune` | 3.00 | 60.0 | 111.00 | ok |
| `random1` | `baseline_random1` | 5.80 | 116.0 | 225.10 | ok |
| `unident_s` | `baseline_unident_s` | 12.70 | 254.0 | 481.25 | ok |
| `simple_tomato` | - | - | - | - | skipped `NoRoute` |

The five-onion-layout average improves to 7.98 soups, and the minimum improves to 3.00 soups. This is the strongest current coverage result. It remains a specialist/router result, not a unified zero-shot policy, and it depends on selecting the best PPO checkpoint rather than using the final checkpoint.

Conclusion: perturbed demonstrations plus checkpoint selection solve the immediate `small_corridor` stability problem within the 3-soup, 400-step evaluation horizon. The important lesson is that PPO fine-tuning is non-monotonic here: the best 25k checkpoint is excellent, while the 50k final checkpoint is bad. Future training scripts should preserve periodic best checkpoints and report both best and final results.

## Current Findings

1. The local machine can run 200k-step PPO experiments in about 70-90 seconds per run, so it is enough for short experiments and debugging.
2. The baseline is valid on `simple`: about 7.5 soups per 400-step episode.
3. Sparse-only learning fails at this budget, confirming the need for shaped rewards.
4. The tested distance shaping does not beat the default shaping.
5. Single-partner training overfits partner style. A baseline ego fails with `baseline_seed12` even though `baseline_seed12` itself is a strong self-play run.
6. Partner-diversity training helps a little against one alternate partner, but not enough to be considered a robust solution.
7. Cross-layout transfer is essentially zero for single-layout policies.
8. The first multi-layout curriculum is not yet a solution: it produces high shaped reward during training but almost no sparse reward in evaluation.
9. Conservative `simple + random0` fine-tuning improves `simple`, but it still does not unlock `random0`.
10. `random0` is learnable as a single-layout specialist, reaching 0.85 soups after 300k steps.
11. The layout-router baseline composes the best current `simple` specialist and the current `random0` specialist, reaching 9.55 soups on `simple` and 0.85 on `random0`.
12. Increasing `random0` specialist training to 800k steps improves `random0` from 0.85 to 6.30 soups.
13. The updated router reaches 9.55 soups on `simple` and 6.30 on `random0`, so specialist routing is now a strong practical baseline.
14. Cross-layout capability is currently best treated as a composition problem over specialists, not as a simple zero-shot or naive mixed-training problem.
15. `small_corridor` is the clearest remaining failure: default training, inactive-distance shaping, and three structured shaping variants all stay at 0.00 soups.
16. `random1` is learnable as a 300k specialist, reaching 5.80 soups.
17. `unident_s` is the strongest current hard-layout specialist, reaching 12.70 soups.
18. The expanded onion router reaches 8.59 supported-layout average soups and 5.80 supported-layout minimum soups over four routed layouts, while explicitly skipping `small_corridor` and tomato layouts.
19. Initial trace diagnosis shows why the first `small_corridor` runs failed: default training never uses ego `interact`, and inactive-distance shaping causes many interactions but no successful pickup or placement.
20. A second 800k `random0` seed (`baseline_random0_long_seed52`) improves the `random0` specialist from 6.30 to 8.85 soups.
21. Replacing the `random0` route with `baseline_random0_long_seed52` improves the routed supported-layout average from 8.59 to 9.23 soups.
22. Hard-layout partner robustness is uneven: `unident_s` is robust across seeds, `random1` collapses under held-out partners, and `random0` shows asymmetric compatibility.
23. Self-play score alone is not enough evidence for general coordination; cross-play matrices are needed for every claimed robust specialist.
24. The upstream Overcooked-AI distance shaping parameters are inactive in this checkout because the distance-reward call is commented out, so layout-specific progress shaping has to be implemented in the wrapper if we want it to affect training.
25. Structured `small_corridor` shaping can move the agents from "no useful subgoal" to "pot progress and soup pickup," but the delivery leg to the left-side serving station still fails.
26. The next `small_corridor` optimization should focus on delivery-stage warm starts, scripted subtask trajectories, or behavior-cloning pretraining instead of more generic reward coefficient tuning.
27. Delivery warm-start PPO from the v3 checkpoint still gets 0.00 soups after 100k steps, even though shaped/progress reward rises from 34.07 to 73.76.
28. Scripted delivery demos solve the isolated final delivery subtask with 100/100 success and provide the first clean data source for behavior cloning.
29. Delivery behavior cloning reaches 1.00 soups on the isolated `small_corridor_delivery` start state.
30. Delivery-only BC does not solve standard `small_corridor`; from the standard start the BC policy chooses `stay` for the whole 400-step trace.
31. Delivery-only BC does not solve the full task; it must be broadened to the whole cooking chain or used through an explicit subtask/hierarchical controller.
32. Full-chain scripted demos solve one standard-start `small_corridor` soup with 100/100 success and mean 122 steps.
33. Full-chain BC reaches 1.00 standard-start soup on `small_corridor`, where all PPO-only specialists still get 0.00.
34. PPO fine-tuning from full-chain BC does not improve the policy at 50k steps; the best checkpoint remains the initial BC policy, while the final checkpoint drops to 0.90 soups.
35. Adding the full-chain BC specialist to the router gives nonzero coverage on all five onion layouts, with 7.58 average soups and 1.00 minimum soup.
36. The new `small_corridor` result should be framed as scripted imitation / specialist rescue, not as evidence that PPO alone solved the layout.
37. Multi-cycle full-chain demos solve three repeated `small_corridor` soup cycles with 100/100 scripted success and mean 376 steps.
38. Multi-cycle BC improves standard-start `small_corridor` from 1.00 to 1.90 mean soups and can complete 3 soups in a deterministic trace.
39. PPO fine-tuning from the 3-cycle BC does not outperform BC-only at 50k steps; best remains initialization, while final reaches 1.55 soups.
40. Replacing the one-cycle `small_corridor` BC route with the 3-cycle BC route improves the five-onion-layout router from 7.58 average / 1.00 min to 7.76 average / 1.90 min.
41. Wait-perturbed 3-cycle demos improve `small_corridor` BC from 1.90 to 2.50 mean soups.
42. PPO fine-tuning from the perturbed BC reaches 3.00 mean soups at the 25k best checkpoint, but the 50k final checkpoint collapses to 0.85 soups.
43. Replacing the `small_corridor` route with the checkpoint-selected perturbed BC+PPO specialist improves the five-onion-layout router to 7.98 average soups and 3.00 minimum soups.

## Artifacts

- Baseline metrics: `outputs/runs/baseline_simple/metrics/`
- Reward ablations: `outputs/runs/no_shaping_simple/`, `outputs/runs/distance_shaping_simple/`
- Partner runs: `outputs/runs/baseline_seed11/`, `outputs/runs/baseline_seed12/`
- Partner-diversity run: `outputs/runs/partner_diversity_simple/`
- Partner matrix CSVs:
  - `outputs/runs/baseline_simple/metrics/partner_matrix_baseline.csv`
  - `outputs/runs/partner_diversity_simple/metrics/partner_matrix.csv`
- Zero-shot layout CSVs:
  - `outputs/runs/baseline_simple/metrics/zero_shot_layouts.csv`
  - `outputs/runs/partner_diversity_simple/metrics/zero_shot_layouts.csv`
  - `outputs/runs/multi_layout_curriculum/metrics/zero_shot_layouts.csv`
  - `outputs/runs/curriculum_simple_random0/metrics/zero_shot_layouts.csv`
  - `outputs/runs/baseline_random0/metrics/zero_shot_layouts.csv`
- Multi-layout curriculum run:
  - `outputs/runs/multi_layout_curriculum/metrics/train_summary.json`
  - `outputs/runs/multi_layout_curriculum/metrics/eval_metrics.json`
- Staged curriculum run:
  - `outputs/runs/curriculum_simple_random0/metrics/curriculum_eval.csv`
  - `outputs/runs/curriculum_simple_random0/metrics/train_summary.json`
- Random0 specialist run:
  - `outputs/runs/baseline_random0/metrics/eval_metrics.json`
  - `outputs/runs/baseline_random0/metrics/zero_shot_layouts.csv`
- Long Random0 specialist run:
  - `outputs/runs/baseline_random0_long/metrics/train_summary.json`
  - `outputs/runs/baseline_random0_long/metrics/eval_metrics.json`
  - `outputs/runs/baseline_random0_long/metrics/zero_shot_layouts.csv`
- Hard-layout partner-robustness runs:
  - `outputs/runs/baseline_random0_long_seed52/metrics/eval_metrics.json`
  - `outputs/runs/baseline_random1_seed71/metrics/eval_metrics.json`
  - `outputs/runs/baseline_unident_s_seed81/metrics/eval_metrics.json`
  - `outputs/runs/baseline_random0_long/metrics/partner_matrix_hard_random0.csv`
  - `outputs/runs/baseline_random0_long_seed52/metrics/partner_matrix_hard_random0.csv`
  - `outputs/runs/baseline_random1/metrics/partner_matrix_hard_random1.csv`
  - `outputs/runs/baseline_random1_seed71/metrics/partner_matrix_hard_random1.csv`
  - `outputs/runs/baseline_unident_s/metrics/partner_matrix_hard_unident_s.csv`
  - `outputs/runs/baseline_unident_s_seed81/metrics/partner_matrix_hard_unident_s.csv`
- Structured `small_corridor` runs:
  - `outputs/runs/small_corridor_structured_shaping_v1/metrics/eval_metrics.json`
  - `outputs/runs/small_corridor_structured_shaping_v1/metrics/eval_stochastic.json`
  - `outputs/runs/small_corridor_structured_shaping_v1/traces/small_corridor_structured_trace.json`
  - `outputs/runs/small_corridor_structured_shaping_v1/traces/small_corridor_structured_stochastic_trace.json`
  - `outputs/runs/small_corridor_structured_shaping_v2/metrics/eval_metrics.json`
  - `outputs/runs/small_corridor_structured_shaping_v2/metrics/eval_stochastic.json`
  - `outputs/runs/small_corridor_structured_shaping_v2/traces/small_corridor_structured_trace.json`
  - `outputs/runs/small_corridor_structured_shaping_v2/traces/small_corridor_structured_stochastic_trace_v2breakdown.json`
  - `outputs/runs/small_corridor_structured_shaping_v3/metrics/eval_metrics.json`
  - `outputs/runs/small_corridor_structured_shaping_v3/metrics/eval_stochastic.json`
  - `outputs/runs/small_corridor_structured_shaping_v3/traces/small_corridor_structured_trace.json`
  - `outputs/runs/small_corridor_structured_shaping_v3/traces/small_corridor_structured_stochastic_trace.json`
- Delivery warm-start and scripted demo artifacts:
  - `outputs/runs/small_corridor_delivery_warmstart_from_v3/metrics/curriculum_eval.csv`
  - `outputs/runs/small_corridor_delivery_warmstart_from_v3/metrics/train_summary.json`
  - `outputs/demos/small_corridor_delivery_scripted.json`
- Delivery BC artifacts:
  - `outputs/runs/small_corridor_delivery_bc_from_v3/metrics/bc_summary.json`
  - `outputs/runs/small_corridor_delivery_bc_from_v3/metrics/eval_delivery_warmstart.json`
  - `outputs/runs/small_corridor_delivery_bc_from_v3/metrics/eval_standard_start_h400.json`
  - `outputs/runs/small_corridor_delivery_bc_from_v3/traces/delivery_warmstart_trace.json`
  - `outputs/runs/small_corridor_delivery_bc_from_v3/traces/standard_start_h400_trace.json`
  - `outputs/runs/small_corridor_delivery_bc_ppo_finetune/metrics/curriculum_eval.csv`
  - `outputs/runs/small_corridor_delivery_bc_ppo_finetune/metrics/eval_standard_start_h400.json`
  - `outputs/runs/small_corridor_delivery_bc_ppo_finetune/traces/standard_start_h400_trace.json`
- Full-chain `small_corridor` BC artifacts:
  - `outputs/demos/small_corridor_full_chain_scripted.json`
  - `outputs/runs/small_corridor_full_chain_bc_from_v3/metrics/bc_summary.json`
  - `outputs/runs/small_corridor_full_chain_bc_from_v3/metrics/eval_standard_start_h400.json`
  - `outputs/runs/small_corridor_full_chain_bc_from_v3/traces/standard_start_h400_trace.json`
  - `outputs/runs/small_corridor_full_chain_bc_from_v3/demo/small_corridor_full_chain_bc_demo.gif`
  - `outputs/runs/small_corridor_full_chain_bc_ppo_finetune/metrics/curriculum_eval.csv`
  - `outputs/runs/small_corridor_full_chain_bc_ppo_finetune/metrics/train_summary.json`
  - `outputs/runs/small_corridor_full_chain_bc_ppo_finetune/metrics/eval_standard_start_h400.json`
  - `outputs/runs/small_corridor_full_chain_bc_ppo_finetune/traces/standard_start_h400_trace.json`
- Multi-cycle full-chain `small_corridor` BC artifacts:
  - `outputs/demos/small_corridor_full_chain_3cycle_scripted.json`
  - `outputs/runs/small_corridor_full_chain_3cycle_bc_from_v3/metrics/bc_summary.json`
  - `outputs/runs/small_corridor_full_chain_3cycle_bc_from_v3/metrics/eval_standard_start_h400.json`
  - `outputs/runs/small_corridor_full_chain_3cycle_bc_from_v3/traces/standard_start_h400_trace.json`
  - `outputs/runs/small_corridor_full_chain_3cycle_bc_from_v3/demo/small_corridor_full_chain_3cycle_bc_demo.gif`
  - `outputs/runs/small_corridor_full_chain_3cycle_bc_ppo_finetune/metrics/curriculum_eval.csv`
  - `outputs/runs/small_corridor_full_chain_3cycle_bc_ppo_finetune/metrics/train_summary.json`
- Perturbed multi-cycle full-chain `small_corridor` artifacts:
  - `outputs/demos/small_corridor_full_chain_3cycle_jitter3_scripted.json`
  - `outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_from_v3/metrics/bc_summary.json`
  - `outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_from_v3/metrics/eval_standard_start_h400.json`
  - `outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_from_v3/traces/standard_start_h400_trace.json`
  - `outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_from_v3/demo/small_corridor_full_chain_3cycle_jitter3_bc_demo.gif`
  - `outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune/metrics/train_summary.json`
  - `outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune/metrics/curriculum_eval.csv`
  - `outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune/metrics/eval_standard_start_h400.json`
  - `outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune/traces/standard_start_h400_trace.json`
  - `outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune/demo/small_corridor_full_chain_3cycle_jitter3_bc_ppo_best_demo.gif`
- Phase 2 specialist runs:
  - `outputs/runs/baseline_small_corridor/metrics/eval_metrics.json`
  - `outputs/runs/baseline_small_corridor/metrics/zero_shot_layouts.csv`
  - `outputs/runs/small_corridor_shaping_v1/metrics/eval_metrics.json`
  - `outputs/runs/small_corridor_shaping_v1/metrics/zero_shot_layouts.csv`
  - `outputs/runs/baseline_random1/metrics/eval_metrics.json`
  - `outputs/runs/baseline_random1/metrics/zero_shot_layouts.csv`
  - `outputs/runs/baseline_unident_s/metrics/eval_metrics.json`
  - `outputs/runs/baseline_unident_s/metrics/zero_shot_layouts.csv`
- Layout-router run:
  - `outputs/runs/router_simple_random0/router_config.resolved.json`
  - `outputs/runs/router_simple_random0/metrics/router_eval.csv`
  - `outputs/runs/router_simple_random0/metrics/router_eval.json`
- Updated layout-router run:
  - `outputs/runs/router_simple_random0_long/router_config.resolved.json`
  - `outputs/runs/router_simple_random0_long/metrics/router_eval.csv`
  - `outputs/runs/router_simple_random0_long/metrics/router_eval.json`
- Expanded onion-router run:
  - `outputs/runs/router_onion_layouts/router_config.resolved.json`
  - `outputs/runs/router_onion_layouts/metrics/router_eval.csv`
  - `outputs/runs/router_onion_layouts/metrics/router_eval.json`
- Improved onion-router run:
  - `outputs/runs/router_onion_layouts_seed52_random0/router_config.resolved.json`
  - `outputs/runs/router_onion_layouts_seed52_random0/metrics/router_eval.csv`
  - `outputs/runs/router_onion_layouts_seed52_random0/metrics/router_eval.json`
- Onion-router run with full-chain `small_corridor` BC:
  - `outputs/runs/router_onion_layouts_with_small_corridor_bc/router_config.resolved.json`
  - `outputs/runs/router_onion_layouts_with_small_corridor_bc/metrics/router_eval.csv`
  - `outputs/runs/router_onion_layouts_with_small_corridor_bc/metrics/router_eval.json`
- Onion-router run with 3-cycle full-chain `small_corridor` BC:
  - `outputs/runs/router_onion_layouts_with_small_corridor_3cycle_bc/router_config.resolved.json`
  - `outputs/runs/router_onion_layouts_with_small_corridor_3cycle_bc/metrics/router_eval.csv`
  - `outputs/runs/router_onion_layouts_with_small_corridor_3cycle_bc/metrics/router_eval.json`
- Onion-router run with checkpoint-selected perturbed `small_corridor` BC+PPO:
  - `outputs/runs/router_onion_layouts_with_small_corridor_jitter3_bc_ppo/router_config.resolved.json`
  - `outputs/runs/router_onion_layouts_with_small_corridor_jitter3_bc_ppo/metrics/router_eval.csv`
  - `outputs/runs/router_onion_layouts_with_small_corridor_jitter3_bc_ppo/metrics/router_eval.json`
- Demos:
  - `outputs/runs/baseline_simple/demo/demo.gif`
  - `outputs/runs/no_shaping_simple/demo/demo.gif`
  - `outputs/runs/distance_shaping_simple/demo/demo.gif`
  - `outputs/runs/partner_diversity_simple/demo/demo.gif`
  - `outputs/runs/multi_layout_curriculum/demo/random0_demo.gif`
  - `outputs/runs/multi_layout_curriculum/demo/simple_demo.gif`
  - `outputs/runs/curriculum_simple_random0/demo/simple_best_demo.gif`
  - `outputs/runs/curriculum_simple_random0/demo/random0_best_demo.gif`
  - `outputs/runs/baseline_random0/demo/random0_demo.gif`
  - `outputs/runs/baseline_random0/demo/simple_transfer_demo.gif`
  - `outputs/runs/baseline_random0_long/demo/random0_long_demo.gif`
  - `outputs/runs/small_corridor_full_chain_bc_from_v3/demo/small_corridor_full_chain_bc_demo.gif`
  - `outputs/runs/small_corridor_full_chain_3cycle_bc_from_v3/demo/small_corridor_full_chain_3cycle_bc_demo.gif`
  - `outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune/demo/small_corridor_full_chain_3cycle_jitter3_bc_ppo_best_demo.gif`
  - `outputs/runs/baseline_random1/demo/random1_demo.gif`
  - `outputs/runs/baseline_unident_s/demo/unident_s_demo.gif`
- Episode traces:
  - `outputs/runs/baseline_small_corridor/traces/small_corridor_trace.json`
  - `outputs/runs/small_corridor_shaping_v1/traces/small_corridor_shaping_trace.json`
  - `outputs/runs/baseline_random1/traces/random1_trace.json`

## Step 23: `random1` Partner-Aware Training

This step tests whether training an ego policy against multiple fixed `random1`
partners can repair the partner mismatch collapse found in Step 11.

New configs:

- `configs/partner_diversity_random1.json`
- `configs/baseline_random1_seed72.json`

Commands:

```bash
bash scripts/train_diverse.sh configs/partner_diversity_random1.json
bash scripts/train.sh configs/baseline_random1_seed72.json

bash scripts/evaluate_matrix.sh outputs/runs/baseline_random1 \
  --partner-run-dir outputs/runs/baseline_random1 \
  --partner-run-dir outputs/runs/baseline_random1_seed71 \
  --partner-run-dir outputs/runs/baseline_random1_seed72 \
  --layout random1 \
  --output-name partner_matrix_hard_random1_three_partners

bash scripts/evaluate_matrix.sh outputs/runs/partner_diversity_random1 \
  --partner-run-dir outputs/runs/baseline_random1 \
  --partner-run-dir outputs/runs/baseline_random1_seed71 \
  --partner-run-dir outputs/runs/baseline_random1_seed72 \
  --layout random1 \
  --output-name partner_matrix_hard_random1_three_partners
```

Three-partner `random1` compatibility matrix, reported as mean soups delivered:

| Ego run | `baseline_random1` partner | `baseline_random1_seed71` partner | `baseline_random1_seed72` partner | Average | Minimum |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_random1` | 5.80 | 0.25 | 0.65 | 2.23 | 0.25 |
| `baseline_random1_seed71` | 0.00 | 5.20 | 0.35 | 1.85 | 0.00 |
| `baseline_random1_seed72` | 0.75 | 0.00 | 1.15 | 0.63 | 0.00 |
| `partner_diversity_random1` | 2.25 | 4.55 | 0.45 | 2.42 | 0.45 |

The partner-diversity ego was trained against `baseline_random1` and
`baseline_random1_seed71` as fixed partners. Within that training partner pool,
it improves the minimum from 0.25 soups to 2.25 soups and the average to 3.40
soups. However, it still performs poorly with the held-out seed72 partner
(0.45 soups), so this is not enough to claim true partner robustness.

Conclusion: multi-partner training is directionally useful for the partners it
sees, but a two-partner pool does not generalize reliably to a new `random1`
partner. The current router should keep `baseline_random1` for the `random1`
route because its self-play route score remains 5.80 soups, while
`partner_diversity_random1` is best treated as a robustness diagnostic.

Artifacts:

- `outputs/runs/partner_diversity_random1/metrics/train_summary.json`
- `outputs/runs/partner_diversity_random1/metrics/partner_matrix_hard_random1_three_partners.csv`
- `outputs/runs/baseline_random1_seed72/metrics/train_summary.json`
- `outputs/runs/baseline_random1_seed72/metrics/partner_matrix_hard_random1_three_partners.csv`
- `outputs/runs/baseline_random1/metrics/partner_matrix_hard_random1_three_partners.csv`
- `outputs/runs/baseline_random1_seed71/metrics/partner_matrix_hard_random1_three_partners.csv`

## Step 24: Submission Readiness Audit

This step audits the final packaging requirements from `组队课题.pdf` and maps
the current repository artifacts to those requirements.

Assignment requirements extracted from the PDF:

- Final archive name should be `学号+姓名`.
- Research report should be submitted as PDF and should include the research
  plan, training process and parameters, core code, experimental analysis,
  interesting observations, figures, and references.
- Complete code and trained models should be submitted.
- Demo recording should show how to start the algorithm and the final result.
- Grading split is 60% report, 40% code plus demo.
- For non-graduating students, the prompt deadline is 2026-07-01 18:00; for
  graduating students, it is 2026-06-19 18:00.

New submission-facing docs:

- `docs/submission_manifest.md`: maps report/code/models/metrics/demo assets to
  the assignment requirements and notes the remaining manual items.
- `report/demo_script.md`: gives a 5-8 minute recording outline, commands to
  show, narration points, and GIF demo order.

Current package-ready artifacts:

- Report PDF: `report/final_report.pdf` (8 pages).
- Support slides: `report/slides.pdf` (14 pages).
- Main code/config/scripts: `src/`, `configs/`, `scripts/`.
- Selected trained runs: `baseline_simple`, `no_shaping_simple`,
  `multi_layout_curriculum`, `baseline_random0_long_seed52`,
  `baseline_random1`, `baseline_unident_s`,
  `small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune`,
  `router_onion_layouts_with_small_corridor_jitter3_bc_ppo`, and
  `partner_diversity_random1`.
- Demo GIFs listed in `docs/submission_manifest.md`.

Remaining manual items before upload:

1. Add real course/team/name/student-id metadata if required by the final
   template.
2. Record the screen-demo video; GIF demos are ready but are not the same as the
   prompt's requested complete recording.
3. Rename the final archive to `学号+姓名`.
4. Decide whether to include the large `external/PantheonRL` submodule offline
   or rely on `git submodule update --init --recursive`.

## Step 25: Submission Packager

This step turns the submission manifest into an executable packaging helper.

New script:

- `scripts/package_submission.py`

Default behavior:

- Builds a compact `dist/<name>.zip` archive.
- Includes source code, configs, scripts, report files, docs, selected trained
  runs, metrics, and GIF demos.
- Excludes the 867 MB `external/PantheonRL` submodule by default while keeping
  `.gitmodules` and setup instructions for reproducibility.
- Supports `--include-external` for a fully offline package.
- Supports `--demo-video path/to/demo.mp4` to include a screen recording.
- Supports `--dry-run` for validation without writing an archive.

Verified commands:

```bash
python -m py_compile scripts/package_submission.py
python scripts/package_submission.py --name submission_dry_run --dry-run
python scripts/package_submission.py --name submission_dry_run --output-dir tmp/package_test --force
```

The test archive was written to `tmp/package_test/submission_dry_run.zip`
with size 6.3 MB. A zip-content check confirmed that it contains:

- `report/final_report.pdf`
- `report/slides.pdf`
- `docs/submission_manifest.md`
- `report/demo_script.md`
- `src/overcooked_project/train.py`
- `scripts/package_submission.py`
- `outputs/runs/baseline_simple/models/ego.zip`
- `outputs/runs/baseline_simple/demo/demo.gif`
- `outputs/runs/router_onion_layouts_with_small_corridor_jitter3_bc_ppo/metrics/router_eval.csv`

Conclusion: the final archive can now be generated reproducibly once the real
`学号+姓名` archive stem and optional demo video path are known.

## Step 26: Submission Metadata Helper

This step removes the last report-editing guesswork around real course/team
metadata without committing fake personal information.

New files:

- `scripts/apply_submission_metadata.py`
- `report/submission_metadata.example.json`

Behavior:

- The real `report/submission_metadata.json` is gitignored, so personal
  information does not get committed accidentally.
- The script validates that the JSON no longer contains placeholder text such as
  `填写`.
- It inserts or replaces an invisible-marker metadata block in
  `report/final_report.md` and `report/slides.md`.
- `scripts/build_report_exports.py` now skips HTML comment marker lines so the
  metadata replacement markers do not appear in exported HTML/PDF.
- `--export` can regenerate report exports after applying real metadata.

Example flow:

```bash
cp report/submission_metadata.example.json report/submission_metadata.json
# Edit report/submission_metadata.json with real course/team/name/student-id data.
python scripts/apply_submission_metadata.py --metadata report/submission_metadata.json --export
```

Verification:

```bash
python -m py_compile scripts/apply_submission_metadata.py scripts/build_report_exports.py
python scripts/apply_submission_metadata.py --metadata tmp/submission_metadata_test.json --dry-run
```

Conclusion: final identity metadata is now ready to apply as soon as the real
submission information is available, without adding placeholders to the current
report.

## Step 27: Submission Preflight Helper

This step adds a read-only preflight script so the final submission state can be
checked before packaging.

New script:

- `scripts/check_submission_ready.py`

Checks:

- required report PDFs and source files,
- submission manifest and demo script,
- source/config/script directories,
- selected package manifest entries from `scripts/package_submission.py`,
- optional real `report/submission_metadata.json`,
- optional demo recording path,
- final archive name,
- git worktree and branch status.

Verified commands:

```bash
python -m py_compile scripts/check_submission_ready.py scripts/package_submission.py scripts/apply_submission_metadata.py
python scripts/check_submission_ready.py --name 学号+姓名
```

Current expected result without real personal metadata or a demo recording:

- 0 failures.
- Warnings for placeholder archive name, missing real metadata JSON, missing
  demo video path, dirty worktree while this change is uncommitted, and branch
  ahead of remote.

Conclusion: final packaging now has a read-only audit step. Once real
`学号+姓名`, optional `report/submission_metadata.json`, and optional demo video
are available, rerun the preflight and then build the archive.

## Step 28: Demo Video Draft

This step reduces the demo-recording blocker by generating a reproducible
GIF-based MP4 draft from the existing experiment demos and report narrative.
It is a submission scaffold and review artifact, not a replacement for a real
screen recording if the teacher strictly requires live terminal interaction.

New script:

- `scripts/build_demo_video.py`

Inputs:

- `outputs/runs/baseline_simple/demo/demo.gif`
- `outputs/runs/no_shaping_simple/demo/demo.gif`
- `outputs/runs/multi_layout_curriculum/demo/simple_demo.gif`
- `outputs/runs/baseline_random0_long/demo/random0_long_demo.gif`
- `outputs/runs/baseline_random1/demo/random1_demo.gif`
- `outputs/runs/baseline_unident_s/demo/unident_s_demo.gif`
- `outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune/demo/small_corridor_full_chain_3cycle_jitter3_bc_ppo_best_demo.gif`

Verified commands:

```bash
python -m py_compile scripts/build_demo_video.py scripts/check_submission_ready.py scripts/package_submission.py scripts/apply_submission_metadata.py
python scripts/build_demo_video.py --gif-seconds 2 --output tmp/demo_video_test.mp4 --force
python scripts/build_demo_video.py --force
ffprobe -v error -show_entries format=duration,size -of default=nw=1 report/demo_video_draft.mp4
ffmpeg -v error -y -ss 20 -i report/demo_video_draft.mp4 -frames:v 1 tmp/demo_video_frame.png
python scripts/check_submission_ready.py --name 学号+姓名
```

Generated artifact:

- `report/demo_video_draft.mp4`
- Duration: `437.162760` seconds.
- Size: `2411213` bytes.

Preflight result after generation:

- 0 failures.
- Warnings remain for placeholder archive name, missing real metadata JSON, use
  of the generated demo draft instead of an explicit real recording path, dirty
  worktree while this change is uncommitted, and branch ahead of remote.

Conclusion: the repository now has a reproducible demo-video draft path. The
remaining course-facing choice is whether this GIF-based MP4 is acceptable, or
whether to record a live screen-demo using `report/demo_script.md`.

## Step 29: Larger `random1` Partner Population

This step follows the project plan's remaining partner-robustness item: expand
the `random1` partner population beyond the first two-partner diversity run and
test whether this fixes held-out partner collapse.

New configs:

- `configs/baseline_random1_seed73.json`
- `configs/partner_diversity_random1_three_partners.json`

Commands:

```bash
bash scripts/train.sh configs/baseline_random1_seed73.json
bash scripts/train_diverse.sh configs/partner_diversity_random1_three_partners.json

bash scripts/evaluate_matrix.sh outputs/runs/baseline_random1 \
  --partner-run-dir outputs/runs/baseline_random1 \
  --partner-run-dir outputs/runs/baseline_random1_seed71 \
  --partner-run-dir outputs/runs/baseline_random1_seed72 \
  --partner-run-dir outputs/runs/baseline_random1_seed73 \
  --layout random1 \
  --output-name partner_matrix_hard_random1_four_partners

bash scripts/evaluate_matrix.sh outputs/runs/partner_diversity_random1 \
  --partner-run-dir outputs/runs/baseline_random1 \
  --partner-run-dir outputs/runs/baseline_random1_seed71 \
  --partner-run-dir outputs/runs/baseline_random1_seed72 \
  --partner-run-dir outputs/runs/baseline_random1_seed73 \
  --layout random1 \
  --output-name partner_matrix_hard_random1_four_partners

bash scripts/evaluate_matrix.sh outputs/runs/partner_diversity_random1_three_partners \
  --partner-run-dir outputs/runs/baseline_random1 \
  --partner-run-dir outputs/runs/baseline_random1_seed71 \
  --partner-run-dir outputs/runs/baseline_random1_seed72 \
  --partner-run-dir outputs/runs/baseline_random1_seed73 \
  --layout random1 \
  --output-name partner_matrix_hard_random1_four_partners
```

Training results:

| Run | Seed | Timesteps | Train seconds | Deterministic self-play soups |
| --- | ---: | ---: | ---: | ---: |
| `baseline_random1_seed73` | 73 | 300000 | 117.01 | 0.90 |
| `partner_diversity_random1_three_partners` | 73 | 300000 | 106.36 | not self-play; trained against three fixed partners |

Four-partner compatibility matrix, reported as mean soups delivered:

| Ego run | `baseline_random1` partner | `baseline_random1_seed71` partner | `baseline_random1_seed72` partner | `baseline_random1_seed73` partner | Average | Minimum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_random1` | 5.80 | 0.25 | 0.65 | 1.10 | 1.95 | 0.25 |
| `partner_diversity_random1` | 2.25 | 4.55 | 0.45 | 1.10 | 2.09 | 0.45 |
| `partner_diversity_random1_three_partners` | 4.90 | 1.40 | 0.65 | 1.00 | 1.99 | 0.65 |
| `baseline_random1_seed73` | 1.65 | 0.00 | 0.55 | 0.90 | 0.78 | 0.00 |

Interpretation:

- The three-partner run improves the minimum over its three seen partners from
  0.45 to 0.65 soups, so population training is not useless.
- It does not improve the held-out seed73 result over the two-partner run:
  1.00 soups versus 1.10 soups.
- Its four-partner average is 1.99, slightly below the two-partner run's 2.09.
- Therefore, simply adding one more fixed partner is not enough to solve
  `random1` held-out robustness. A more systematic population method, partner
  conditioning, or online co-play/self-play population training would be needed
  for a stronger robustness claim.

Artifacts:

- `outputs/runs/baseline_random1_seed73/metrics/train_summary.json`
- `outputs/runs/baseline_random1_seed73/metrics/eval_metrics.json`
- `outputs/runs/baseline_random1_seed73/metrics/partner_matrix_hard_random1_four_partners.csv`
- `outputs/runs/partner_diversity_random1_three_partners/metrics/train_summary.json`
- `outputs/runs/partner_diversity_random1_three_partners/metrics/partner_matrix_hard_random1_four_partners.csv`
- `outputs/runs/partner_diversity_random1/metrics/partner_matrix_hard_random1_four_partners.csv`
- `outputs/runs/baseline_random1/metrics/partner_matrix_hard_random1_four_partners.csv`

## Step 30: Role-Balanced `small_corridor` BC Diagnostic

This step follows the remaining `small_corridor` extension item in
`docs/project_plan.md`: test whether the perturbed 3-cycle BC specialist is
limited by fixed player-role data. The code now supports `--role-balanced` in
`scripts/train_delivery_bc.sh`, which trains each policy on the combined p0 and
p1 demonstration observations/actions instead of training ego only on p0 and
alt only on p1.

New config:

- `configs/small_corridor_full_chain_3cycle_jitter3_role_balanced_bc_from_v3.json`

Commands:

```bash
bash scripts/train_delivery_bc.sh \
  --config configs/small_corridor_full_chain_3cycle_jitter3_role_balanced_bc_from_v3.json \
  --demo-path outputs/demos/small_corridor_full_chain_3cycle_jitter3_scripted.json \
  --run-name small_corridor_full_chain_3cycle_jitter3_role_balanced_bc_from_v3 \
  --epochs 60 \
  --batch-size 256 \
  --learning-rate 0.001 \
  --role-balanced

bash scripts/evaluate.sh outputs/runs/small_corridor_full_chain_3cycle_jitter3_role_balanced_bc_from_v3 \
  --episodes 20 \
  --standard-start \
  --horizon 400 \
  --output-name eval_standard_start_h400
```

BC dataset and validation:

| Run | Source demo | Role-balanced | Training steps per model | Ego val acc | Alt val acc |
| --- | --- | --- | ---: | ---: | ---: |
| `small_corridor_full_chain_3cycle_jitter3_role_balanced_bc_from_v3` | `outputs/demos/small_corridor_full_chain_3cycle_jitter3_scripted.json` | yes | 77018 | 0.9830 | 0.9817 |

Standard-start h400 evaluation:

| Run/checkpoint | Mean soups | Mean sparse reward | Mean episode reward | Soup distribution over 20 deterministic episodes |
| --- | ---: | ---: | ---: | --- |
| Fixed-role jitter BC | 2.50 | 50.0 | 93.25 | 3 episodes with 1 soup, 4 with 2 soups, 13 with 3 soups |
| Role-balanced jitter BC | 2.40 | 48.0 | 90.15 | 3 episodes with 1 soup, 6 with 2 soups, 11 with 3 soups |
| Checkpoint-selected BC+PPO best | 3.00 | 60.0 | 111.00 | 20 episodes with 3 soups |

Interpretation:

- The role-balanced model has high validation accuracy, but its environment
  performance is slightly worse than the fixed-role jitter BC baseline.
- This means the remaining `small_corridor` instability is not solved by simply
  sharing both role datasets across both policies.
- Keep this as a negative diagnostic. The best route remains the
  checkpoint-selected perturbed BC+PPO specialist.
- A stronger next attempt should be a more structured subtask router or staged
  controller, not another simple role-balanced BC run.

Artifacts:

- `outputs/runs/small_corridor_full_chain_3cycle_jitter3_role_balanced_bc_from_v3/metrics/bc_summary.json`
- `outputs/runs/small_corridor_full_chain_3cycle_jitter3_role_balanced_bc_from_v3/metrics/eval_standard_start_h400.json`
- `outputs/runs/small_corridor_full_chain_3cycle_jitter3_role_balanced_bc_from_v3/metrics/eval_standard_start_h400.csv`

## Next Experiments

The next project direction should move beyond naive multi-layout mixing:

1. Final submission packaging: add real identity metadata if required, record
   the demo video or use the generated draft if acceptable, and run
   `scripts/package_submission.py` with the real `学号+姓名` archive stem.
2. Stronger partner-diversity: the 3-partner fixed-pool run is completed but
   still does not solve held-out robustness; future work should use a larger or
   more structured population method rather than just adding one fixed partner.
3. If time remains, test a more structured `small_corridor` subtask router as
   an extension beyond the solved 3-soup specialist; simple role-balanced BC is
   already tested and did not improve over fixed-role jitter BC.
4. Tomato support decision: either avoid tomato maps in this environment stack
   or patch/replace the featurizer before using tomato layouts.
