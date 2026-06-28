# Overcooked MARL Project Plan

Last updated: 2026-06-28

Project path: `/Volumes/share/pku/26_spring/多智能体/finallab2`

## Current Position

The project already has a reproducible PPO baseline, a set of ablations, cross-partner evaluation, cross-layout evaluation, staged curriculum, a `random0` specialist diagnostic, and a layout-router evaluator.

Key results so far:

| Capability | Current best result | Meaning |
| --- | ---: | --- |
| `simple` specialist | 9.55 soups | The easy map is basically solved well enough for the report. |
| `random0` specialist | 8.85 soups | A second 800k seed makes `random0` stronger and useful. |
| `random1` specialist | 5.80 soups | The default 300k specialist is usable. |
| `unident_s` specialist | 12.70 soups | This is the strongest current hard-layout specialist. |
| `small_corridor` specialist | 3.00 soups standard start via perturbed 3-cycle BC + 25k PPO checkpoint selection | PPO from scratch remains at 0.00 from standard start; perturbed scripted imitation plus checkpoint selection stabilizes the repeated loop. |
| naive multi-layout PPO | 0.00 soups on most maps | Simple layout mixing is not enough. |
| staged `simple + random0` fine-tuning | 9.55 on `simple`, 0.00 on `random0` | Fine-tuning from the easy-map expert does not unlock `random0`. |
| PPO-only onion router | 9.23 average soups, 5.80 min soups over four PPO-routed layouts | Strongest pure-PPO specialist composition, but it skips `small_corridor`. |
| onion router with `small_corridor` BC+PPO | 7.98 average soups, 3.00 min soups over five onion layouts | Broadest current coverage; the minimum is now `small_corridor` at 3 soups. |
| hard-layout partner robustness | mixed | `unident_s` is robust, `random1` is brittle, and `random0` is asymmetric. |
| tomato layouts | blocked by `KeyError: 'tomato'` | Treat tomato support as an environment issue, not a policy result. |

The main bottleneck is no longer whether the baseline can run or whether specialists are useful. The bottleneck is now turning the specialist/router result into a clean report and strengthening partner robustness for the successful PPO specialists.

## Project Goal

The target is a course-ready Overcooked MARL project with:

1. A solid PPO baseline on `simple`.
2. Clear evidence that sparse reward and naive layout generalization fail.
3. A stronger improvement story based on specialists, router composition, and possibly layout-conditioned policy selection.
4. Reproducible scripts, configs, metrics, demos, and experiment records.
5. A report narrative that explains both successes and failures honestly.

The minimum acceptable final result is not "solve every layout." A good project can be built around showing why naive generalization fails and then improving coverage with specialist routing.

## Strategy

Do not spend the next stage trying to make one unified PPO policy generalize from scratch. The experiments already suggest that this is low-yield at the current budget.

Use this order instead:

1. Strengthen weak single-layout specialists.
2. Expand router coverage with those specialists.
3. Compare router composition against naive multi-layout and staged curriculum.
4. Only after specialist coverage is meaningful, try a unified or layout-conditioned policy as a stretch.

This gives the report a clean arc:

`baseline works on simple -> sparse and zero-shot fail -> naive multi-layout fails -> specialists can learn hard layouts -> router composes specialists -> unified policy remains future/stretch work`.

## Phase 1: Strengthen `random0`

Status: completed.

Why this phase matters:

`simple` is already strong. The current router average is dragged down almost entirely by `random0`. Improving `random0` from 0.85 soups to 3-5 soups would immediately make the project result much more convincing.

Planned experiments:

| Experiment | Change | Purpose | Success criterion |
| --- | --- | --- | --- |
| `baseline_random0_long` | Train `random0` for 800k-1M steps | Test whether more budget solves the weak specialist | `random0` mean soups >= 3.0 |
| `baseline_random0_seed51/52` | Train extra 300k-step seeds | Check whether 0.85 is seed luck or stable | Best seed clearly beats 0.85 |
| `random0_shaping_v1` | Tune shaping around pot progress, dish pickup, soup pickup, delivery | Use only if longer training still stalls | Higher sparse reward, not just higher shaped reward |

Completed result:

| Run | Timesteps | Mean soups | Mean sparse reward | Mean episode reward | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `baseline_random0_long` | 800000 | 6.30 | 126.0 | 244.45 | Success; use as the new `random0` router specialist. |
| `baseline_random0_long_seed52` | 800000 | 8.85 | 177.0 | 339.45 | Stronger success; use as the preferred `random0` router specialist. |

Recommended first command after adding the config:

```bash
bash scripts/train.sh configs/baseline_random0_long.json
```

Evaluation commands:

```bash
bash scripts/evaluate_matrix.sh outputs/runs/baseline_random0_long \
  --partner-run-dir outputs/runs/baseline_random0_long \
  --layout random0 --layout simple --layout small_corridor --layout random1 --layout unident_s --layout simple_tomato \
  --output-name zero_shot_layouts

bash scripts/evaluate_router.sh configs/router_simple_random0.json \
  --route random0=outputs/runs/baseline_random0_long \
  --output-dir outputs/runs/router_simple_random0_long \
  --output-name router_eval
```

Decision rule:

- If `baseline_random0_long` reaches at least 3 soups, use it as the new `random0` route and move to Phase 2. This condition is satisfied.
- If it improves but stays below 3 soups, run two more seeds before changing reward shaping.
- If it stays near 0-1 soups, stop adding budget and move to reward redesign.

## Phase 2: Expand Specialist Coverage

Status: completed for multi-soup BC coverage; `small_corridor` remains unresolved as a PPO/robust multi-soup policy.

Target layouts:

- `small_corridor`
- `random1`
- `unident_s`

Planned experiments:

| Experiment | Layout | Initial budget | Success criterion |
| --- | --- | ---: | --- |
| `baseline_small_corridor` | `small_corridor` | 300k steps | mean soups > 1.0 |
| `baseline_random1` | `random1` | 300k steps | mean soups > 1.0 |
| `baseline_unident_s` | `unident_s` | 300k steps | mean soups > 1.0 |

Completed results:

| Run | Layout | Timesteps | Mean soups | Mean sparse reward | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `baseline_small_corridor` | `small_corridor` | 300000 | 0.00 | 0.0 | Failed; do not route. |
| `small_corridor_shaping_v1` | `small_corridor` | 300000 | 0.00 | 0.0 | Failed; upstream distance reward path is inactive in this checkout. |
| `small_corridor_structured_shaping_v1` | `small_corridor` | 300000 | 0.00 | 0.0 | Failed; shaped reward rises but no pot progress. |
| `small_corridor_structured_shaping_v2` | `small_corridor` | 300000 | 0.00 | 0.0 | Failed; stochastic policy farms onion/progress reward without delivery. |
| `small_corridor_structured_shaping_v3` | `small_corridor` | 300000 | 0.00 | 0.0 | Partial progress; reaches soup pickup but still fails serving. |
| `small_corridor_delivery_warmstart_from_v3` | `small_corridor` | 100000 | 0.00 | 0.0 | Failed as PPO, but confirms progress reward alone does not teach final serving. |
| `small_corridor_delivery_bc_from_v3` | `small_corridor_delivery` | BC 40 epochs | 1.00 | 20.0 | Solves the isolated delivery warm-start, but gets 0.00 from the standard start. |
| `small_corridor_delivery_bc_ppo_finetune` | `small_corridor_delivery` | 50000 | 1.00 | 20.0 | PPO fine-tuning preserves delivery BC, but still gets 0.00 from the standard start. |
| `small_corridor_full_chain_bc_from_v3` | `small_corridor` | BC 60 epochs | 1.00 | 20.0 | Full-chain scripted BC solves the first standard-start soup. |
| `small_corridor_full_chain_bc_ppo_finetune` | `small_corridor` | 50000 | 1.00 best / 0.90 final | 20.0 best / 18.0 final | PPO fine-tuning does not improve the BC policy; best checkpoint remains the initial BC. |
| `small_corridor_full_chain_3cycle_bc_from_v3` | `small_corridor` | BC 60 epochs | 1.90 | 38.0 | Multi-cycle scripted BC improves the repeated standard-start loop and reaches 3 soups in some episodes. |
| `small_corridor_full_chain_3cycle_bc_ppo_finetune` | `small_corridor` | 50000 | 1.90 best / 1.55 final | 38.0 best / 31.0 final | PPO fine-tuning still does not beat BC-only; best checkpoint remains the initial 3-cycle BC. |
| `small_corridor_full_chain_3cycle_jitter3_bc_from_v3` | `small_corridor` | BC 60 epochs | 2.50 | 50.0 | Wait-perturbed demos make the BC loop substantially more stable. |
| `small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune` | `small_corridor` | 50000 | 3.00 best at 25k / 0.85 final | 60.0 best / 17.0 final | Checkpoint selection is essential; the 25k checkpoint solves the 3-soup horizon, while 50k over-training collapses. |
| `baseline_random1` | `random1` | 300000 | 5.80 | 116.0 | Success; add to router. |
| `baseline_unident_s` | `unident_s` | 300000 | 12.70 | 254.0 | Success; add to router. |

Router update for successful specialists:

```bash
bash scripts/evaluate_router.sh configs/router_simple_random0.json \
  --route simple=outputs/runs/curriculum_simple_random0 \
  --route random0=outputs/runs/baseline_random0_long \
  --route random1=outputs/runs/baseline_random1 \
  --route unident_s=outputs/runs/baseline_unident_s \
  --output-dir outputs/runs/router_onion_layouts \
  --output-name router_eval
```

Expanded router result:

| Layout | Selected run | Mean soups | Status |
| --- | --- | ---: | --- |
| `simple` | `curriculum_simple_random0` | 9.55 | ok |
| `random0` | `baseline_random0_long` | 6.30 | ok |
| `small_corridor` | - | - | skipped `NoRoute` |
| `random1` | `baseline_random1` | 5.80 | ok |
| `unident_s` | `baseline_unident_s` | 12.70 | ok |

Supported-layout average: 8.59 soups.
Supported-layout minimum: 5.80 soups.

Updated router with `baseline_random0_long_seed52`:

| Layout | Selected run | Mean soups | Status |
| --- | --- | ---: | --- |
| `simple` | `curriculum_simple_random0` | 9.55 | ok |
| `random0` | `baseline_random0_long_seed52` | 8.85 | ok |
| `small_corridor` | - | - | skipped `NoRoute` |
| `random1` | `baseline_random1` | 5.80 | ok |
| `unident_s` | `baseline_unident_s` | 12.70 | ok |

Updated supported-layout average: 9.23 soups.
Updated supported-layout minimum: 5.80 soups.

Router with full-chain `small_corridor` BC:

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

| Layout | Selected run | Mean soups | Status |
| --- | --- | ---: | --- |
| `simple` | `curriculum_simple_random0` | 9.55 | ok |
| `random0` | `baseline_random0_long_seed52` | 8.85 | ok |
| `small_corridor` | `small_corridor_full_chain_bc_from_v3` | 1.00 | ok |
| `random1` | `baseline_random1` | 5.80 | ok |
| `unident_s` | `baseline_unident_s` | 12.70 | ok |

Five-onion-layout average: 7.58 soups.
Five-onion-layout minimum: 1.00 soups.

Router with 3-cycle full-chain `small_corridor` BC:

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

| Layout | Selected run | Mean soups | Status |
| --- | --- | ---: | --- |
| `simple` | `curriculum_simple_random0` | 9.55 | ok |
| `random0` | `baseline_random0_long_seed52` | 8.85 | ok |
| `small_corridor` | `small_corridor_full_chain_3cycle_bc_from_v3` | 1.90 | ok |
| `random1` | `baseline_random1` | 5.80 | ok |
| `unident_s` | `baseline_unident_s` | 12.70 | ok |

Current five-onion-layout average: 7.76 soups.
Current five-onion-layout minimum: 1.90 soups.

Router with checkpoint-selected perturbed 3-cycle `small_corridor` BC+PPO:

```bash
bash scripts/evaluate_router.sh configs/router_simple_random0.json \
  --route simple=outputs/runs/curriculum_simple_random0 \
  --route random0=outputs/runs/baseline_random0_long_seed52 \
  --route random1=outputs/runs/baseline_random1 \
  --route unident_s=outputs/runs/baseline_unident_s \
  --route small_corridor=outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune \
  --output-dir outputs/runs/router_onion_layouts_with_small_corridor_jitter3_bc_ppo \
  --output-name router_eval
```

| Layout | Selected run | Mean soups | Status |
| --- | --- | ---: | --- |
| `simple` | `curriculum_simple_random0` | 9.55 | ok |
| `random0` | `baseline_random0_long_seed52` | 8.85 | ok |
| `small_corridor` | `small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune` | 3.00 | ok |
| `random1` | `baseline_random1` | 5.80 | ok |
| `unident_s` | `baseline_unident_s` | 12.70 | ok |

Best five-onion-layout average: 7.98 soups.
Best five-onion-layout minimum: 3.00 soups.

Phase success criterion:

- Minimum: `simple` plus at least two harder layouts have nonzero sparse reward. Satisfied.
- Good: four supported onion-style layouts have nonzero sparse reward. Satisfied for PPO specialists; five onion layouts have nonzero sparse reward if the scripted-BC `small_corridor` specialist is allowed.
- Strong: router supported-layout average >= 4 soups and supported-layout minimum >= 1 soup. Satisfied over all five onion layouts by the router with `small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune`.

## Phase 3: Robustness And Partner Generalization

Status: initial hard-layout matrices completed; stronger partner-aware training remains future work.

Why this phase matters:

The current experiments show that policies can overfit partner style. A final report should not only show self-play success; it should also show what happens when partners differ.

Planned experiments:

| Experiment | Purpose | Success criterion |
| --- | --- | --- |
| Additional `simple` partner seeds | Build held-out partners | At least two held-out partners for evaluation |
| Additional hard-layout partner seeds | Test whether specialists are brittle | Specialist does not collapse to zero with every alternate partner |
| Router with mixed partner choices | Evaluate composition under partner mismatch | Identify which layouts are robust vs brittle |

Evaluation command pattern:

```bash
bash scripts/evaluate_matrix.sh outputs/runs/<ego_run> \
  --partner-run-dir outputs/runs/<partner_a> \
  --partner-run-dir outputs/runs/<partner_b> \
  --layout <layout> \
  --output-name partner_matrix
```

Phase success criterion:

- Report at least one partner-generalization matrix.
- Explain whether the improvement is layout-specific, partner-specific, or both.

Completed hard-layout matrix summary:

| Layout | Self-play result | Held-out partner result | Interpretation |
| --- | ---: | ---: | --- |
| `random0` | 6.30 to 8.85 soups | 1.65 to 7.70 soups | Asymmetric. The seed-52 ego is robust to the old partner, but the old ego is brittle with the seed-52 partner. |
| `random1` | 5.20 to 5.80 soups | 0.00 to 0.25 soups | Strong partner brittleness despite good self-play. |
| `unident_s` | 12.60 to 12.70 soups | 12.60 to 12.65 soups | Robust across two independent seeds. |

Decision: partner robustness must be reported per layout. `unident_s` can be described as robust across the tested seeds, but `random1` should be described as a self-play specialist rather than a generally compatible teammate.

## Phase 4: Unified Or Layout-Conditioned Policy

Status: stretch phase.

Do this only after the specialist/router baseline is strong enough to serve as a comparison.

Candidate approaches:

| Approach | Description | Risk |
| --- | --- | --- |
| Layout-conditioned observation | Append a layout one-hot feature to observations | Requires code changes and careful observation-space handling |
| Policy selection router | Keep separate specialists and route by known layout | Simple and already working, but not a single neural policy |
| Distillation from specialists | Train one policy on specialist rollouts | More engineering, but a better unified-policy story |
| Reverse curriculum | Start from `random0`, then add `simple` | Might preserve hard-layout behavior better than starting from `simple` |

Success criterion:

- A unified or conditioned method should beat naive multi-layout PPO.
- It does not need to beat the router. The router is a strong specialist-composition baseline.

Decision rule:

- If the unified method approaches router performance, make it the main improvement.
- If it fails, report it as a meaningful negative result and keep router composition as the practical solution.

## Phase 5: Report And Demo Package

Status: report-facing materials started in `docs/report_materials.md`, a prose report draft exists in `docs/report_draft.md`, report SVG assets are available in `docs/assets/`, and final report/slides drafts plus HTML/PDF exports exist under `report/`.

Required report components:

1. Problem setup: Overcooked as cooperative MARL.
2. Environment and algorithm: PantheonRL wrapper, PPO, reward shaping, horizon, metrics.
3. Baseline: `simple` PPO result.
4. Ablations: sparse-only failure and distance-shaping non-improvement.
5. Generalization failures: partner mismatch and zero-shot layout transfer.
6. Naive multi-layout failure: high shaped reward does not imply soup delivery.
7. Specialist/router improvement: coverage and supported-layout average/min.
8. Scripted/BC rescue for `small_corridor`: supervised full-chain demos solve repeated soup cycles where PPO exploration fails.
9. Remaining limitations: `small_corridor` is still brittle and scripted-BC dependent, tomato environment issue, partner brittleness.
10. Demo GIFs and metric tables.

Recommended figures/tables:

| Artifact | Source |
| --- | --- |
| Run summary table | `docs/experiment_log.md` |
| Report-ready summary | `docs/report_materials.md` |
| Prose report draft | `docs/report_draft.md` |
| Report SVG figures | `docs/assets/router_comparison.svg`, `docs/assets/small_corridor_progression.svg`, `docs/assets/partner_robustness.svg` |
| Final report draft | `report/final_report.md` |
| Slides draft | `report/slides.md` |
| Final report exports | `report/final_report.html`, `report/final_report.pdf`, `report/slides.html`, `report/slides.pdf` |
| Reward shaping ablation table | `outputs/runs/*/metrics/eval_metrics.json` |
| Cross-layout matrix | `outputs/runs/*/metrics/zero_shot_layouts.csv` |
| Router coverage table | `outputs/runs/router_*/metrics/router_eval.csv` |
| Demo GIFs | `outputs/runs/*/demo/*.gif` |
| Training curves | TensorBoard logs under each run directory |

## Execution Checklist

For every new experiment:

1. Add or update a config under `configs/`.
2. Run through `scripts/*.sh` so conda and streaming behavior are consistent.
3. Save metrics under `outputs/runs/<run_name>/metrics/`.
4. Record the command, result, conclusion, and artifact paths in `docs/experiment_log.md`.
5. If the result changes project direction, update this plan.
6. Record experiment summaries in the Notion project page.
7. Commit code/config/docs changes with git.

## Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| `small_corridor` depends on checkpoint selection | Final PPO checkpoint can collapse even after a strong 25k checkpoint | Always report best and final checkpoints; save checkpoint-selection curves and avoid judging only final models |
| Shaped reward rises without sparse reward | Misleading training curves | Always report soups delivered and sparse reward |
| Tomato layouts keep failing | Coverage gap | Exclude tomato from main claims or patch featurizer as separate engineering work |
| Partner overfitting | Self-play results look stronger than they are | Include partner matrix and held-out seeds |
| Too many experiments dilute the report | Hard to finish cleanly | Prioritize Phase 1 and Phase 2; keep Phase 4 as stretch |

## Immediate Next Work

The next concrete work item is:

1. Do final submission polish on `report/final_report.pdf` and `report/slides.pdf`: add course/team metadata if required by the teacher, and adjust wording to the final submission template if one is provided.
2. Keep the three router baselines explicit in the final text: PPO-only four-layout router (`9.23` average, `5.80` min), five-layout router with 3-cycle BC `small_corridor` (`7.76` average, `1.90` min), and five-layout router with checkpoint-selected perturbed BC+PPO (`7.98` average, `3.00` min).
3. Consider partner-aware training for `random1`, because held-out partner evaluation collapses despite strong self-play.
4. If time remains, test role-balanced `small_corridor` demos or a subtask router as an extension beyond the now-successful 3-soup specialist.

After each new attempt, decide whether to:

- keep it as a router route,
- use it only as a negative result,
- extend its budget,
- try another seed,
- or redesign the training setup for that layout.
