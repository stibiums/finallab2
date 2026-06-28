# Overcooked Final Submission Manifest

Last updated: 2026-06-28

This file maps the current repository artifacts to the final group-project
requirements in `组队课题.pdf`.

## Assignment Requirements

The prompt asks for a compressed package named `学号+姓名`.

Required package contents:

1. Research report, PDF format. Chinese or English is allowed. There is no hard
   page limit, but the prompt recommends at least 3 pages. The report should
   include research plan, training process and parameters, core code,
   experimental analysis, interesting observations, figures when useful, and
   references.
2. Complete code and trained models. The code should be complete,
   reproducible, and explain the implementation. Trained policy-network models
   should be saved and submitted.
3. Demo recording. The recording should show how to start the algorithm and
   the final results.

The grading split is:

| Component | Weight |
| --- | ---: |
| Report | 60% |
| Code and demo | 40% |

For non-graduating students, the prompt lists the deadline as
2026-07-01 18:00. For graduating students, it lists 2026-06-19 18:00.

## Current Submission Artifacts

| Requirement | Current artifact | Status |
| --- | --- | --- |
| Research report PDF | `report/final_report.pdf` | Ready, 8 pages |
| Presentation PDF | `report/slides.pdf` | Ready, optional support material |
| Report source | `report/final_report.md`, `report/final_report.html` | Ready |
| Slides source | `report/slides.md`, `report/slides.html` | Ready |
| Full experiment log | `docs/experiment_log.md` | Ready |
| Report materials | `docs/report_materials.md`, `docs/report_draft.md`, `docs/assets/` | Ready |
| Main code | `src/overcooked_project/` | Ready |
| Reproducible scripts | `scripts/` | Ready |
| Configs | `configs/` | Ready |
| Python dependencies | `requirements.txt`, `scripts/create_env.sh` | Ready |
| External environment code | `external/PantheonRL` submodule | Present locally; very large, see packaging note |
| Trained models | selected `outputs/runs/*/models/` paths below | Ready locally |
| Metrics | selected `outputs/runs/*/metrics/` paths below | Ready locally |
| Demo visuals | selected `outputs/runs/*/demo/*.gif` paths below | Ready locally |
| Demo recording | `report/demo_script.md` gives the recording plan | Manual screen recording still needed if the teacher strictly requires video |
| Demo video draft | `scripts/build_demo_video.py` -> `report/demo_video_draft.mp4` | Generated and verified locally from GIF demos |
| Archive helper | `scripts/package_submission.py` | Ready; dry-run and test zip verified |
| Identity metadata helper | `scripts/apply_submission_metadata.py`, `report/submission_metadata.example.json` | Ready; use only with real course/member data |

## Recommended Runs To Include

These runs cover the final report narrative without packaging every exploratory
failure.

| Purpose | Run directory |
| --- | --- |
| PPO baseline and default shaping | `outputs/runs/baseline_simple` |
| Sparse reward failure | `outputs/runs/no_shaping_simple` |
| Naive multi-layout failure | `outputs/runs/multi_layout_curriculum` |
| Strong `random0` specialist | `outputs/runs/baseline_random0_long_seed52` |
| `random1` specialist | `outputs/runs/baseline_random1` |
| `unident_s` specialist | `outputs/runs/baseline_unident_s` |
| Best `small_corridor` specialist | `outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune` |
| Best five-layout router metrics | `outputs/runs/router_onion_layouts_with_small_corridor_jitter3_bc_ppo` |
| Partner robustness diagnostic | `outputs/runs/partner_diversity_random1` |

## Demo Assets

| Demo | Path |
| --- | --- |
| Default PPO baseline | `outputs/runs/baseline_simple/demo/demo.gif` |
| Sparse reward failure | `outputs/runs/no_shaping_simple/demo/demo.gif` |
| Naive multi-layout failure | `outputs/runs/multi_layout_curriculum/demo/simple_demo.gif` |
| `random0` specialist | `outputs/runs/baseline_random0_long/demo/random0_long_demo.gif` |
| `random1` specialist | `outputs/runs/baseline_random1/demo/random1_demo.gif` |
| `unident_s` specialist | `outputs/runs/baseline_unident_s/demo/unident_s_demo.gif` |
| Best `small_corridor` specialist | `outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune/demo/small_corridor_full_chain_3cycle_jitter3_bc_ppo_best_demo.gif` |

## Reproduction Commands

Create the environment:

```bash
git submodule update --init --recursive
bash scripts/create_env.sh
```

Run a smoke test:

```bash
bash scripts/smoke_test.sh
```

Build a demo video draft from the GIF demos:

```bash
python scripts/build_demo_video.py --force
```

Regenerate report figures and exports:

```bash
python scripts/build_report_assets.py
python scripts/build_report_exports.py
```

Apply real course/team/name/student-id metadata before final export:

```bash
cp report/submission_metadata.example.json report/submission_metadata.json
# Edit report/submission_metadata.json with real information first.
python scripts/apply_submission_metadata.py --metadata report/submission_metadata.json --export
```

Dry-run the final package checklist:

```bash
python scripts/package_submission.py --name 学号+姓名 --dry-run
```

Run a full preflight check. Without real metadata or a demo video, this should
pass the required artifact checks and warn about the remaining manual items:

```bash
python scripts/check_submission_ready.py --name 学号+姓名
```

Create the compact final archive after replacing `学号+姓名` with the real
archive name:

```bash
python scripts/package_submission.py --name 学号+姓名
```

Add a recorded demo video if available:

```bash
python scripts/package_submission.py --name 学号+姓名 --demo-video path/to/demo.mp4 --force
```

Evaluate the strongest router:

```bash
ENV_PREFIX=/Volumes/data/conda_envs/overcooked-marl bash scripts/evaluate_router.sh configs/router_simple_random0.json \
  --route random0=outputs/runs/baseline_random0_long_seed52 \
  --route small_corridor=outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune \
  --route random1=outputs/runs/baseline_random1 \
  --route unident_s=outputs/runs/baseline_unident_s \
  --output-dir outputs/runs/router_onion_layouts_with_small_corridor_jitter3_bc_ppo \
  --output-name router_eval
```

## Packaging Notes

The local `external/PantheonRL` submodule is about 867 MB. For a compact code
submission, `scripts/package_submission.py` includes `.gitmodules` and setup
instructions but excludes `external/PantheonRL` by default, so the reviewer can
initialize it with `git submodule update --init --recursive`. If the course
requires a fully offline package, pass `--include-external` and expect the
archive to be much larger.

The repository branch is currently ahead of `origin/main`; do not assume remote
state is up to date unless the branch is pushed.

Packaging verification performed on 2026-06-28:

```bash
python -m py_compile scripts/apply_submission_metadata.py scripts/package_submission.py
python -m py_compile scripts/check_submission_ready.py
python -m py_compile scripts/build_demo_video.py
python scripts/build_demo_video.py --gif-seconds 2 --output tmp/demo_video_test.mp4 --force
python scripts/build_demo_video.py --force
ffprobe -v error -show_entries format=duration,size -of default=nw=1 report/demo_video_draft.mp4
python scripts/check_submission_ready.py --name 学号+姓名
python scripts/apply_submission_metadata.py --metadata tmp/submission_metadata_test.json --dry-run
python -m py_compile scripts/package_submission.py
python scripts/package_submission.py --name submission_dry_run --dry-run
python scripts/package_submission.py --name submission_dry_run --output-dir tmp/package_test --force
```

The compact test archive wrote successfully to
`tmp/package_test/submission_dry_run.zip` and contained the required report,
code, model, demo, and router-metric paths checked after creation.
After the demo-video draft was generated, the test archive was rebuilt and
confirmed to include `submission_dry_run/report/demo_video_draft.mp4`.

The generated demo-video draft wrote successfully to
`report/demo_video_draft.mp4`; `ffprobe` reported duration `437.162760`
seconds and size `2411213` bytes. A frame extraction check also confirmed that
the MP4 contains rendered Overcooked demo footage.

Current expected preflight status before real identity/video are provided:
required artifacts pass, while the script warns about the placeholder archive
name, missing real `report/submission_metadata.json`, use of the generated demo
draft instead of an explicit real recording path, and local git state if there
are uncommitted changes or unpushed commits.

## Manual Items Before Final Upload

1. Rename the final archive to the required `学号+姓名` format.
2. Add course/team/name/student-id metadata with
   `scripts/apply_submission_metadata.py` if the teacher requires it or provides
   a template.
3. Record the demo video using `report/demo_script.md`, or confirm that GIF
   demos plus slides or the generated `report/demo_video_draft.mp4` are
   acceptable.
4. Decide whether to include the large `external/PantheonRL` submodule in the
   archive or rely on `git submodule update --init --recursive`.
