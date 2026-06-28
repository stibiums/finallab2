# Demo Recording Script

Last updated: 2026-06-28

This is a suggested script for the final demo recording required by
`组队课题.pdf`. The goal is to show both how to start the project and what the
final policies achieve.

A draft video has been generated from the GIF demos. It can be rebuilt with:

```bash
python scripts/build_demo_video.py --force
```

This creates `report/demo_video_draft.mp4`. It is useful as a demo artifact or
recording scaffold, but a real screen recording is still preferable if the
teacher strictly requires live terminal interaction.

## Recording Outline

Target length: 5-8 minutes.

1. Show the repository root and explain the project objective.
2. Show environment setup and reproducibility commands.
3. Show the strongest router evaluation command.
4. Open the final report and slides PDFs.
5. Play the key GIF demos in the order below.
6. End with the main conclusion and limitations.

## Commands To Show

From the project root:

```bash
pwd
git status --short --branch
```

Environment setup command:

```bash
git submodule update --init --recursive
bash scripts/create_env.sh
```

Quick verification command:

```bash
bash scripts/smoke_test.sh
```

Strongest router evaluation command:

```bash
ENV_PREFIX=/Volumes/data/conda_envs/overcooked-marl bash scripts/evaluate_router.sh configs/router_simple_random0.json \
  --route random0=outputs/runs/baseline_random0_long_seed52 \
  --route small_corridor=outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune \
  --route random1=outputs/runs/baseline_random1 \
  --route unident_s=outputs/runs/baseline_unident_s \
  --output-dir outputs/runs/router_onion_layouts_with_small_corridor_jitter3_bc_ppo \
  --output-name router_eval
```

Report regeneration command:

```bash
python scripts/build_report_assets.py
python scripts/build_report_exports.py
```

## Suggested Narration

Opening:

> This project studies two-agent cooperation in Overcooked. We start from PPO
> baselines, show sparse reward and naive generalization failures, then build a
> specialist-router solution with behavior-cloning support for the hardest
> layout.

Baseline:

> On the easy `simple` layout, PPO with default shaping reaches 7.50 soups,
> while sparse-only training stays at 0.00 soups. This establishes why reward
> shaping is necessary.

Generalization failure:

> The `simple` specialist does not zero-shot transfer to harder layouts, and
> naive multi-layout PPO also fails. That is why the project moves to
> single-layout specialists and a router.

Router result:

> The PPO-only router covers four onion layouts with 9.23 average soups and
> 5.80 minimum soups. After adding the checkpoint-selected perturbed
> `small_corridor` BC+PPO specialist, the five-layout router reaches 7.98
> average soups and 3.00 minimum soups.

Small corridor:

> `small_corridor` is the main hard case. PPO from scratch stays at 0.00 soups.
> Full-chain demonstrations, wait perturbations, and checkpoint-selected PPO
> fine-tuning raise it to 3.00 soups, but the final 50k checkpoint collapses to
> 0.85 soups. This is why checkpoint selection is reported explicitly.

Partner robustness:

> Self-play success is not the same as partner robustness. `random1` works well
> in self-play but collapses with held-out partners. A two-partner
> partner-diversity run improves seen partners but still only reaches 0.45 soups
> with held-out seed72.

Closing:

> The final claim is not that one neural policy generalizes to every map. The
> supported claim is that specialist composition plus explicit diagnostics gives
> a stronger and more honest Overcooked MARL system under the course budget.

## GIF Demo Order

1. `outputs/runs/baseline_simple/demo/demo.gif`
2. `outputs/runs/no_shaping_simple/demo/demo.gif`
3. `outputs/runs/multi_layout_curriculum/demo/simple_demo.gif`
4. `outputs/runs/baseline_random0_long/demo/random0_long_demo.gif`
5. `outputs/runs/baseline_random1/demo/random1_demo.gif`
6. `outputs/runs/baseline_unident_s/demo/unident_s_demo.gif`
7. `outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune/demo/small_corridor_full_chain_3cycle_jitter3_bc_ppo_best_demo.gif`

## Files To Open During Recording

- `report/final_report.pdf`
- `report/slides.pdf`
- `docs/submission_manifest.md`
- `outputs/runs/router_onion_layouts_with_small_corridor_jitter3_bc_ppo/metrics/router_eval.csv`

## Known Manual Step

The course prompt asks for a complete screen recording. This file is the
recording script and checklist; the actual video should still be recorded before
final upload unless the teacher accepts the generated GIF demos as the demo
artifact.
