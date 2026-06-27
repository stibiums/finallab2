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
| `router_simple_random0_long` | - | 0 | - | 9.55 / 6.30 | 191.0 / 126.0 | 360.40 / 244.45 | - |

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
- Layout-router run:
  - `outputs/runs/router_simple_random0/router_config.resolved.json`
  - `outputs/runs/router_simple_random0/metrics/router_eval.csv`
  - `outputs/runs/router_simple_random0/metrics/router_eval.json`
- Updated layout-router run:
  - `outputs/runs/router_simple_random0_long/router_config.resolved.json`
  - `outputs/runs/router_simple_random0_long/metrics/router_eval.csv`
  - `outputs/runs/router_simple_random0_long/metrics/router_eval.json`
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

## Next Experiments

The next project direction should move beyond naive multi-layout mixing:

1. Add specialists for `small_corridor`, `random1`, and `unident_s`, then expand the router coverage table.
2. Add periodic checkpoint selection for long specialist runs, because `baseline_random0_long` improved strongly but training reward was not monotonic.
3. Try reverse curriculum: initialize from `baseline_random0`, then introduce `simple` with a small sampling weight, to see whether the easier layout can be added without destroying `random0`.
4. Add layout-conditioning or policy selection before claiming one unified policy generalizes.
5. Stronger partner-diversity: include held-out partner seeds and evaluate against them, not only against training partners.
6. Reward redesign: add or tune shaping around pot progress, dish pickup, and delivery so high shaped reward correlates with soups delivered.
7. Tomato support decision: either avoid tomato maps in this environment stack or patch/replace the featurizer before using tomato layouts.
