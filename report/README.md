# Report Deliverables

本目录放课程报告和展示材料的最终入口。

## Files

- `final_report.md`: 正式报告 Markdown 初版，可继续转成 PDF 或课程提交格式。
- `slides.md`: Slidev/Markdown 风格展示稿草案，可直接作为答辩 slides 的文字和图表骨架。
- `final_report.html` / `final_report.pdf`: 已导出的报告预览与 PDF。
- `slides.html` / `slides.pdf`: 已导出的展示稿预览与 PDF。

## Source Materials

- `../docs/report_draft.md`: 报告正文草稿。
- `../docs/report_materials.md`: 报告表格、demo manifest 和结论摘要。
- `../docs/assets/`: 已生成的 SVG 图表。
- `../docs/experiment_log.md`: 完整实验审计记录。

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
