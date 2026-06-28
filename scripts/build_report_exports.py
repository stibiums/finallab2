"""Export report Markdown files to polished HTML and optional PDF.

This script intentionally uses only the Python standard library. The project
machine does not currently have pandoc or a Markdown package installed, so this
small converter covers the Markdown subset used by `report/final_report.md` and
`report/slides.md`.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"


REPORT_CSS = """
:root {
  color-scheme: light;
  --fg: #172033;
  --muted: #536071;
  --line: #d8dee8;
  --soft: #f5f7fb;
  --accent: #2563eb;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: #eef2f7;
  color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans CJK SC",
    "Microsoft YaHei", Arial, sans-serif;
  line-height: 1.64;
}
main {
  width: min(980px, calc(100% - 40px));
  margin: 32px auto;
  padding: 56px 64px;
  background: #fff;
  border: 1px solid var(--line);
}
h1, h2, h3 {
  line-height: 1.25;
  margin: 1.65em 0 .65em;
}
h1 { margin-top: 0; font-size: 30px; text-align: center; }
h2 {
  border-bottom: 1px solid var(--line);
  padding-bottom: 8px;
  font-size: 23px;
}
h3 { font-size: 18px; }
p, ul, ol, table, pre { margin: 0 0 1em; }
ul, ol { padding-left: 1.5em; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
code {
  padding: 2px 5px;
  border-radius: 4px;
  background: var(--soft);
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: .92em;
}
pre {
  padding: 14px 16px;
  overflow-x: auto;
  background: #111827;
  color: #f9fafb;
  border-radius: 6px;
}
pre code { padding: 0; background: transparent; color: inherit; }
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}
th, td {
  padding: 9px 10px;
  border: 1px solid var(--line);
  vertical-align: top;
}
th { background: var(--soft); font-weight: 700; }
blockquote {
  margin: 1em 0;
  padding: 12px 16px;
  border-left: 4px solid var(--accent);
  background: var(--soft);
  color: var(--muted);
}
img {
  display: block;
  max-width: 100%;
  margin: 18px auto 26px;
  border: 1px solid var(--line);
}
.meta {
  color: var(--muted);
  text-align: center;
  margin-top: -10px;
}
@page { margin: 18mm 16mm; }
@media print {
  body { background: #fff; }
  main { width: 100%; margin: 0; padding: 0; border: 0; }
  h2 { break-after: avoid; }
  table, img, pre { break-inside: avoid; }
  a { color: inherit; }
}
"""


SLIDES_CSS = """
:root {
  color-scheme: light;
  --fg: #111827;
  --muted: #4b5563;
  --line: #d1d5db;
  --soft: #f8fafc;
  --accent: #2563eb;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: #111827;
  color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans CJK SC",
    "Microsoft YaHei", Arial, sans-serif;
}
.deck {
  width: 100%;
  max-width: 1180px;
  margin: 0 auto;
  padding: 24px;
}
.slide {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
  min-height: 560px;
  margin: 0 auto 28px;
  padding: 56px 72px;
  background: #fff;
  border: 1px solid var(--line);
  overflow: hidden;
}
.slide.cover {
  display: flex;
  flex-direction: column;
  justify-content: center;
  text-align: center;
}
h1 { margin: 0 0 28px; font-size: 44px; line-height: 1.15; }
h2 { margin: 0 0 24px; font-size: 34px; line-height: 1.2; }
p, li { font-size: 23px; line-height: 1.45; }
ul, ol { padding-left: 1.35em; }
li { margin: 10px 0; }
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 18px;
}
th, td {
  padding: 9px 10px;
  border: 1px solid var(--line);
  vertical-align: top;
}
th { background: var(--soft); }
code {
  padding: 2px 5px;
  border-radius: 4px;
  background: var(--soft);
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: .9em;
}
blockquote {
  margin: 28px 0;
  padding: 18px 24px;
  border-left: 5px solid var(--accent);
  background: var(--soft);
  color: var(--muted);
  font-size: 28px;
}
img {
  display: block;
  max-width: 100%;
  max-height: 390px;
  margin: 0 auto;
}
.slide-number {
  position: absolute;
  right: 28px;
  bottom: 18px;
  color: #9ca3af;
  font-size: 13px;
}
@page { size: 16in 9in; margin: 0; }
@media print {
  body { background: #fff; }
  .deck { max-width: none; margin: 0; padding: 0; }
  .slide {
    width: 16in;
    height: 9in;
    min-height: 0;
    margin: 0;
    border: 0;
    break-after: page;
    page-break-after: always;
  }
}
"""


def inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)",
        lambda m: f'<img src="{html.escape(m.group(2), quote=True)}" alt="{html.escape(m.group(1), quote=True)}">',
        escaped,
    )
    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{m.group(1)}</a>',
        escaped,
    )
    return escaped


def is_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def flush_paragraph(parts: list[str], paragraph: list[str]) -> None:
    if paragraph:
        parts.append(f"<p>{inline(' '.join(paragraph))}</p>")
        paragraph.clear()


def flush_list(parts: list[str], list_tag: str | None, list_items: list[str]) -> None:
    if list_tag and list_items:
        parts.append(f"<{list_tag}>")
        for item in list_items:
            parts.append(f"<li>{inline(item)}</li>")
        parts.append(f"</{list_tag}>")
    list_items.clear()


def markdown_to_html(markdown: str) -> str:
    lines = markdown.splitlines()
    parts: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []
    list_tag: str | None = None
    in_code = False
    code_lines: list[str] = []
    idx = 0

    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                parts.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
                code_lines.clear()
                in_code = False
            else:
                flush_paragraph(parts, paragraph)
                flush_list(parts, list_tag, list_items)
                list_tag = None
                in_code = True
            idx += 1
            continue

        if in_code:
            code_lines.append(line)
            idx += 1
            continue

        if not stripped:
            flush_paragraph(parts, paragraph)
            flush_list(parts, list_tag, list_items)
            list_tag = None
            idx += 1
            continue

        if stripped.startswith("|") and idx + 1 < len(lines) and is_table_separator(lines[idx + 1]):
            flush_paragraph(parts, paragraph)
            flush_list(parts, list_tag, list_items)
            list_tag = None
            headers = split_table_row(stripped)
            idx += 2
            rows: list[list[str]] = []
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                rows.append(split_table_row(lines[idx]))
                idx += 1
            parts.append("<table>")
            parts.append("<thead><tr>" + "".join(f"<th>{inline(cell)}</th>" for cell in headers) + "</tr></thead>")
            parts.append("<tbody>")
            for row in rows:
                parts.append("<tr>" + "".join(f"<td>{inline(cell)}</td>" for cell in row) + "</tr>")
            parts.append("</tbody></table>")
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            flush_paragraph(parts, paragraph)
            flush_list(parts, list_tag, list_items)
            list_tag = None
            level = len(heading.group(1))
            parts.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            idx += 1
            continue

        if stripped.startswith("> "):
            flush_paragraph(parts, paragraph)
            flush_list(parts, list_tag, list_items)
            list_tag = None
            parts.append(f"<blockquote>{inline(stripped[2:])}</blockquote>")
            idx += 1
            continue

        unordered = re.match(r"^[-*]\s+(.+)$", stripped)
        ordered = re.match(r"^\d+\.\s+(.+)$", stripped)
        if unordered or ordered:
            flush_paragraph(parts, paragraph)
            new_tag = "ul" if unordered else "ol"
            if list_tag and list_tag != new_tag:
                flush_list(parts, list_tag, list_items)
            list_tag = new_tag
            list_items.append((unordered or ordered).group(1))
            idx += 1
            continue

        flush_list(parts, list_tag, list_items)
        list_tag = None
        paragraph.append(stripped)
        idx += 1

    flush_paragraph(parts, paragraph)
    flush_list(parts, list_tag, list_items)
    if in_code:
        parts.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
    return "\n".join(parts)


def strip_frontmatter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[end + 5 :].lstrip()
    return text


def wrap_report(markdown: str) -> str:
    body = markdown_to_html(markdown)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>基于 Specialist Router 的 Overcooked 双智能体协作策略研究</title>
  <style>{REPORT_CSS}</style>
</head>
<body>
  <main>
{body}
  </main>
</body>
</html>
"""


def split_slides(markdown: str) -> list[str]:
    clean = strip_frontmatter(markdown)
    slides: list[list[str]] = [[]]
    for line in clean.splitlines():
        if line.strip() == "---":
            slides.append([])
        else:
            slides[-1].append(line)
    return ["\n".join(slide).strip() for slide in slides if "\n".join(slide).strip()]


def wrap_slides(markdown: str) -> str:
    slides = split_slides(markdown)
    sections: list[str] = []
    for idx, slide in enumerate(slides, start=1):
        cover = " cover" if idx == 1 else ""
        sections.append(
            f'<section class="slide{cover}">\n'
            f"{markdown_to_html(slide)}\n"
            f'<div class="slide-number">{idx} / {len(slides)}</div>\n'
            "</section>"
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Overcooked MARL Specialist Router Slides</title>
  <style>{SLIDES_CSS}</style>
</head>
<body>
  <main class="deck">
{chr(10).join(sections)}
  </main>
</body>
</html>
"""


def chrome_path() -> str | None:
    candidates = [
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def export_pdf(html_path: Path, pdf_path: Path, chrome: str) -> None:
    subprocess.run(
        [
            chrome,
            "--headless",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--print-to-pdf-no-header",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_path.resolve().as_uri(),
        ],
        check=True,
        cwd=ROOT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-pdf", action="store_true", help="Only generate HTML exports.")
    args = parser.parse_args()

    report_md = (REPORT_DIR / "final_report.md").read_text(encoding="utf-8")
    slides_md = (REPORT_DIR / "slides.md").read_text(encoding="utf-8")

    report_html = REPORT_DIR / "final_report.html"
    slides_html = REPORT_DIR / "slides.html"
    report_html.write_text(wrap_report(report_md), encoding="utf-8")
    slides_html.write_text(wrap_slides(slides_md), encoding="utf-8")
    print(f"wrote {report_html.relative_to(ROOT)}")
    print(f"wrote {slides_html.relative_to(ROOT)}")

    if args.no_pdf:
        return

    chrome = chrome_path()
    if not chrome:
        print("Chrome/Chromium not found; skipped PDF export.")
        return

    export_pdf(report_html, REPORT_DIR / "final_report.pdf", chrome)
    print("wrote report/final_report.pdf")
    export_pdf(slides_html, REPORT_DIR / "slides.pdf", chrome)
    print("wrote report/slides.pdf")


if __name__ == "__main__":
    main()
