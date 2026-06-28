# Overcooked MARL Report Materials

Last updated: 2026-06-28

Project path: `/Volumes/share/pku/26_spring/多智能体/finallab2`

This document is the report-facing summary of the experiment log. Use
`docs/experiment_log.md` as the full audit trail and this file as the compact
source for report tables, demo choices, and narrative structure.

## Core Story

1. PPO with default shaping learns the easy `simple` layout, while sparse-only
   PPO fails at the same budget.
2. The learned `simple` policy does not transfer zero-shot to harder layouts.
3. Naive multi-layout training is also not enough: high shaped reward does not
   imply sparse soup delivery.
4. Single-layout specialists can solve several harder onion layouts.
5. A layout router over specialists is the strongest practical method in this
   project.
6. `small_corridor` is the key hard case: PPO from scratch and reward shaping
   fail, but scripted full-chain BC plus checkpoint-selected PPO reaches a
   stable 3-soup policy.
7. Partner compatibility remains a limitation, especially on `random1`.

## Main Result Table

| Method / run | Layout set | Mean soups | Key interpretation |
| --- | --- | ---: | --- |
| `baseline_simple` | `simple` | 7.50 | Default shaping gives a valid PPO baseline. |
| `no_shaping_simple` | `simple` | 0.00 | Sparse-only learning fails at the same budget. |
| `distance_shaping_simple` | `simple` | 7.50 | The tested distance shaping does not improve over default shaping. |
| `baseline_simple` zero-shot | harder layouts | 0.00 on `small_corridor`, `random0`, `random1`; 0.00 sparse on `unident_s` | Single-layout PPO does not generalize across layouts. |
| `multi_layout_curriculum` | mixed layouts | 0.00 on most maps | Naive multi-layout PPO is not a solution. |
| PPO-only onion router | four routed onion layouts | 9.23 average, 5.80 minimum | Strong pure-RL specialist composition, but skips `small_corridor`. |
| Router with 3-cycle `small_corridor` BC | five onion layouts | 7.76 average, 1.90 minimum | Adds nonzero `small_corridor` coverage, but BC is still brittle. |
| Router with perturbed BC+PPO `small_corridor` | five onion layouts | 7.98 average, 3.00 minimum | Current strongest result; uses checkpoint-selected `small_corridor` specialist. |

## Router Comparisons

| Router | `simple` | `random0` | `small_corridor` | `random1` | `unident_s` | Average | Minimum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PPO-only onion router | 9.55 | 8.85 | skipped | 5.80 | 12.70 | 9.23 | 5.80 |
| With 3-cycle BC `small_corridor` | 9.55 | 8.85 | 1.90 | 5.80 | 12.70 | 7.76 | 1.90 |
| With perturbed BC+PPO `small_corridor` | 9.55 | 8.85 | 3.00 | 5.80 | 12.70 | 7.98 | 3.00 |

Evidence:

- `outputs/runs/router_onion_layouts_seed52_random0/metrics/router_eval.csv`
- `outputs/runs/router_onion_layouts_with_small_corridor_3cycle_bc/metrics/router_eval.csv`
- `outputs/runs/router_onion_layouts_with_small_corridor_jitter3_bc_ppo/metrics/router_eval.csv`

Report figure:

![Router comparison](assets/router_comparison.svg)

## Small Corridor Progression

| Run | Mean soups | Mean sparse reward | Meaning |
| --- | ---: | ---: | --- |
| `baseline_small_corridor` | 0.00 | 0.0 | PPO from scratch fails. |
| `small_corridor_structured_shaping_v3` | 0.00 | 0.0 | Shaping reaches soup pickup but not delivery. |
| `small_corridor_delivery_bc_from_v3` standard start | 0.00 | 0.0 | Delivery-only BC is too narrow. |
| `small_corridor_full_chain_bc_from_v3` | 1.00 | 20.0 | One full-chain scripted BC solves the first soup. |
| `small_corridor_full_chain_3cycle_bc_from_v3` | 1.90 | 38.0 | Multi-cycle BC learns repeated loop partially. |
| `small_corridor_full_chain_3cycle_jitter3_bc_from_v3` | 2.50 | 50.0 | Wait-perturbed demos improve robustness. |
| `small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune` best at 25k | 3.00 | 60.0 | Best checkpoint solves the 3-soup evaluation horizon. |
| `small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune` final 50k | 0.85 | 17.0 | Final checkpoint collapses; checkpoint selection matters. |

Evidence:

- `outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_from_v3/metrics/eval_standard_start_h400.json`
- `outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune/metrics/train_summary.json`
- `outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune/metrics/curriculum_eval.csv`

Report figure:

![Small corridor progression](assets/small_corridor_progression.svg)

## Partner Robustness Table

| Layout / ego | Self-play soups | Held-out partner soups | Interpretation |
| --- | ---: | ---: | --- |
| `simple` baseline ego | 7.50 | 5.60 with seed11, 0.00 with seed12 | Partner overfitting is already visible on the easy map. |
| `random0` long specialists | 6.30 to 8.85 | 1.65 to 7.70 | Asymmetric compatibility. |
| `random1` specialists | 5.20 to 5.80 | 0.00 to 1.10 across held-out partners | Strong self-play, brittle cross-play. |
| `partner_diversity_random1` | 2.25 to 4.55 with two training partners | 0.45 with held-out seed72, 1.10 with held-out seed73 | Improves in-pool compatibility but does not solve held-out robustness. |
| `partner_diversity_random1_three_partners` | 0.65 to 4.90 with three training partners | 1.00 with held-out seed73 | Raises the seen-partner minimum but does not beat the two-partner run on four-partner average. |
| `unident_s` specialists | 12.60 to 12.70 | 12.60 to 12.65 | Robust across tested seeds. |

Evidence:

- `outputs/runs/baseline_simple/metrics/partner_matrix_baseline.csv`
- `outputs/runs/baseline_random0_long/metrics/partner_matrix_hard_random0.csv`
- `outputs/runs/baseline_random0_long_seed52/metrics/partner_matrix_hard_random0.csv`
- `outputs/runs/baseline_random1/metrics/partner_matrix_hard_random1.csv`
- `outputs/runs/baseline_random1_seed71/metrics/partner_matrix_hard_random1.csv`
- `outputs/runs/baseline_random1/metrics/partner_matrix_hard_random1_three_partners.csv`
- `outputs/runs/baseline_random1_seed71/metrics/partner_matrix_hard_random1_three_partners.csv`
- `outputs/runs/baseline_random1_seed72/metrics/partner_matrix_hard_random1_three_partners.csv`
- `outputs/runs/partner_diversity_random1/metrics/partner_matrix_hard_random1_three_partners.csv`
- `outputs/runs/baseline_random1/metrics/partner_matrix_hard_random1_four_partners.csv`
- `outputs/runs/partner_diversity_random1/metrics/partner_matrix_hard_random1_four_partners.csv`
- `outputs/runs/partner_diversity_random1_three_partners/metrics/partner_matrix_hard_random1_four_partners.csv`
- `outputs/runs/baseline_unident_s/metrics/partner_matrix_hard_unident_s.csv`
- `outputs/runs/baseline_unident_s_seed81/metrics/partner_matrix_hard_unident_s.csv`

Report figure:

![Partner robustness](assets/partner_robustness.svg)

## Demo Package Manifest

Recommended demos for presentation:

| Demo | Path | Use in report/demo |
| --- | --- | --- |
| Default PPO baseline | `outputs/runs/baseline_simple/demo/demo.gif` | Show the working baseline on `simple`. |
| Sparse reward failure | `outputs/runs/no_shaping_simple/demo/demo.gif` | Show that sparse-only PPO fails. |
| Naive multi-layout failure | `outputs/runs/multi_layout_curriculum/demo/simple_demo.gif` | Show that naive curriculum can lose even the easy layout. |
| `random0` specialist | `outputs/runs/baseline_random0_long/demo/random0_long_demo.gif` | Show hard-layout specialist success. |
| `random1` specialist | `outputs/runs/baseline_random1/demo/random1_demo.gif` | Show another successful specialist. |
| `unident_s` specialist | `outputs/runs/baseline_unident_s/demo/unident_s_demo.gif` | Show the strongest specialist. |
| One-soup `small_corridor` BC | `outputs/runs/small_corridor_full_chain_bc_from_v3/demo/small_corridor_full_chain_bc_demo.gif` | Show the first BC rescue step. |
| 3-cycle `small_corridor` BC | `outputs/runs/small_corridor_full_chain_3cycle_bc_from_v3/demo/small_corridor_full_chain_3cycle_bc_demo.gif` | Show repeated scripted-loop imitation. |
| Best `small_corridor` specialist | `outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune/demo/small_corridor_full_chain_3cycle_jitter3_bc_ppo_best_demo.gif` | Use as the final `small_corridor` success demo. |

## Suggested Report Outline

1. Task and environment: Overcooked cooperative MARL, two agents, soup delivery,
   sparse reward as `20 * soups_delivered`.
2. PPO baseline: setup, default shaping, `simple` result.
3. Failure analysis: sparse reward, zero-shot transfer, naive multi-layout
   curriculum.
4. Specialist training: `random0`, `random1`, `unident_s`.
5. Router composition: PPO-only router and five-layout router comparisons.
6. `small_corridor` case study: trace diagnosis, delivery BC, full-chain BC,
   multi-cycle BC, perturbed BC+PPO.
7. Robustness: partner matrices and checkpoint-selection risk.
8. Limitations and future work: tomato maps, unified policy, partner-aware
   training, role-balanced `small_corridor`.

## Final Claims To Keep Precise

- Do say: "The strongest current method is specialist routing with checkpoint
  selection."
- Do say: "`small_corridor` is solved for the 3-soup, 400-step evaluation
  horizon by perturbed BC+PPO at the 25k checkpoint."
- Do not say: "A single PPO policy generalizes across layouts."
- Do not say: "PPO from scratch solved `small_corridor`."
- Do not report final checkpoints alone for BC+PPO experiments; report both
  best and final because they diverge sharply.
