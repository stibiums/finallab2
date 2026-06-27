# Overcooked MARL Project Plan

Last updated: 2026-06-27

Project path: `/Volumes/share/pku/26_spring/多智能体/finallab2`

## Current Position

The project already has a reproducible PPO baseline, a set of ablations, cross-partner evaluation, cross-layout evaluation, staged curriculum, a `random0` specialist diagnostic, and a layout-router evaluator.

Key results so far:

| Capability | Current best result | Meaning |
| --- | ---: | --- |
| `simple` specialist | 9.55 soups | The easy map is basically solved well enough for the report. |
| `random0` specialist | 0.85 soups | `random0` is learnable, but still weak. |
| naive multi-layout PPO | 0.00 soups on most maps | Simple layout mixing is not enough. |
| staged `simple + random0` fine-tuning | 9.55 on `simple`, 0.00 on `random0` | Fine-tuning from the easy-map expert does not unlock `random0`. |
| `simple + random0` router | 9.55 on `simple`, 0.85 on `random0` | Specialist composition is currently the most promising route. |
| tomato layouts | blocked by `KeyError: 'tomato'` | Treat tomato support as an environment issue, not a policy result. |

The main bottleneck is no longer whether the baseline can run. The bottleneck is policy coverage across layouts, especially making `random0` and the remaining onion layouts reliably deliver soups.

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

Status: next immediate phase.

Why this phase matters:

`simple` is already strong. The current router average is dragged down almost entirely by `random0`. Improving `random0` from 0.85 soups to 3-5 soups would immediately make the project result much more convincing.

Planned experiments:

| Experiment | Change | Purpose | Success criterion |
| --- | --- | --- | --- |
| `baseline_random0_long` | Train `random0` for 800k-1M steps | Test whether more budget solves the weak specialist | `random0` mean soups >= 3.0 |
| `baseline_random0_seed51/52` | Train extra 300k-step seeds | Check whether 0.85 is seed luck or stable | Best seed clearly beats 0.85 |
| `random0_shaping_v1` | Tune shaping around pot progress, dish pickup, soup pickup, delivery | Use only if longer training still stalls | Higher sparse reward, not just higher shaped reward |

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

- If `baseline_random0_long` reaches at least 3 soups, use it as the new `random0` route and move to Phase 2.
- If it improves but stays below 3 soups, run two more seeds before changing reward shaping.
- If it stays near 0-1 soups, stop adding budget and move to reward redesign.

## Phase 2: Expand Specialist Coverage

Status: after `random0` improves or after a decision that current `random0` is good enough for the report.

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

Router update after each successful specialist:

```bash
bash scripts/evaluate_router.sh configs/router_simple_random0.json \
  --route simple=outputs/runs/curriculum_simple_random0 \
  --route random0=outputs/runs/baseline_random0_long \
  --route small_corridor=outputs/runs/baseline_small_corridor \
  --route random1=outputs/runs/baseline_random1 \
  --route unident_s=outputs/runs/baseline_unident_s \
  --output-dir outputs/runs/router_onion_layouts \
  --output-name router_eval
```

Phase success criterion:

- Minimum: `simple` plus at least two harder layouts have nonzero sparse reward.
- Good: four onion layouts have nonzero sparse reward.
- Strong: router supported-layout average >= 4 soups and supported-layout minimum >= 1 soup.

## Phase 3: Robustness And Partner Generalization

Status: after router coverage is better than the current two-layout version.

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

Status: continuous, but final assembly should happen after Phase 2 or Phase 3.

Required report components:

1. Problem setup: Overcooked as cooperative MARL.
2. Environment and algorithm: PantheonRL wrapper, PPO, reward shaping, horizon, metrics.
3. Baseline: `simple` PPO result.
4. Ablations: sparse-only failure and distance-shaping non-improvement.
5. Generalization failures: partner mismatch and zero-shot layout transfer.
6. Naive multi-layout failure: high shaped reward does not imply soup delivery.
7. Specialist/router improvement: coverage and supported-layout average/min.
8. Remaining limitations: weak hard-layout specialists, tomato environment issue, partner brittleness.
9. Demo GIFs and metric tables.

Recommended figures/tables:

| Artifact | Source |
| --- | --- |
| Run summary table | `docs/experiment_log.md` |
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
| `random0` remains weak | Router improvement stays small | Try seeds, then reward shaping, then report as limitation if needed |
| Shaped reward rises without sparse reward | Misleading training curves | Always report soups delivered and sparse reward |
| Tomato layouts keep failing | Coverage gap | Exclude tomato from main claims or patch featurizer as separate engineering work |
| Partner overfitting | Self-play results look stronger than they are | Include partner matrix and held-out seeds |
| Too many experiments dilute the report | Hard to finish cleanly | Prioritize Phase 1 and Phase 2; keep Phase 4 as stretch |

## Immediate Next Work

The next concrete work item is:

1. Add `configs/baseline_random0_long.json`.
2. Train `baseline_random0_long` for 800k-1M steps.
3. Evaluate it on `random0` and the existing layout matrix.
4. Re-run router with the new `random0` route.
5. Update `docs/experiment_log.md` and the Notion experiment page with the result.

After that result, decide whether to:

- keep adding budget/seeds to `random0`,
- redesign shaping for `random0`,
- or move on to the remaining layout specialists.
