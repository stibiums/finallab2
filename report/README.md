# Report Deliverables

本目录放课程报告和展示材料的最终入口。

## Files

- `final_report.md`: 正式报告 Markdown。
- `slides.md`: Slidev/Markdown 风格展示稿，可直接作为答辩 slides 的文字和图表骨架。
- `final_report.html` / `final_report.pdf`: 已导出的报告预览与 PDF。
- `slides.html` / `slides.pdf`: 已导出的展示稿预览与 PDF。
- `demo_script.md`: 最终录屏展示脚本和 demo 顺序。

## Source Materials

- `../docs/report_draft.md`: 报告正文草稿。
- `../docs/report_materials.md`: 报告表格、demo manifest 和结论摘要。
- `../docs/submission_manifest.md`: 对照课程 PDF 的最终提交清单。
- `../docs/assets/`: 已生成的 SVG 图表。
- `../docs/experiment_log.md`: 完整实验审计记录。

## Submission Status

- Report PDF: `final_report.pdf`, ready.
- Slides PDF: `slides.pdf`, ready as support material.
- Code/config/scripts: tracked in `src/`, `configs/`, and `scripts/`.
- Models/metrics/GIF demos: stored under selected `outputs/runs/` directories,
  listed in `../docs/submission_manifest.md`.
- Optional metadata helper: fill `submission_metadata.json` from
  `submission_metadata.example.json`, then run
  `python scripts/apply_submission_metadata.py --metadata report/submission_metadata.json --export`.
- Manual remaining item: record the screen-demo video and rename the final
  archive to the required `学号+姓名` format.

## Build Submission Archive

From the project root, first dry-run the package manifest:

```bash
python scripts/package_submission.py --name 学号+姓名 --dry-run
```

Then replace `学号+姓名` with the real archive stem and build:

```bash
python scripts/package_submission.py --name 学号+姓名
```

## Regenerate Figures

Run from the project root:

```bash
python scripts/build_report_assets.py
```

The generated SVG assets are tracked under `docs/assets/`.

## Regenerate Report Exports

Run from the project root:

```bash
python scripts/build_report_exports.py
```

If Chrome/Chromium is available, this generates both HTML and PDF. Use
`--no-pdf` to generate HTML only.
