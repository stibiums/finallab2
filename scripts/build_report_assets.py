"""Build report-ready SVG figures from existing experiment outputs.

The figures are intentionally dependency-free so they can be regenerated in any
Python environment without matplotlib.
"""

from __future__ import annotations

import csv
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"

ROUTER_RUNS = [
    (
        "PPO-only router",
        ROOT / "outputs/runs/router_onion_layouts_seed52_random0/metrics/router_eval.csv",
        "#3b82f6",
    ),
    (
        "3-cycle BC router",
        ROOT / "outputs/runs/router_onion_layouts_with_small_corridor_3cycle_bc/metrics/router_eval.csv",
        "#10b981",
    ),
    (
        "Perturbed BC+PPO router",
        ROOT
        / "outputs/runs/router_onion_layouts_with_small_corridor_jitter3_bc_ppo/metrics/router_eval.csv",
        "#f59e0b",
    ),
]

LAYOUTS = ["simple", "random0", "small_corridor", "random1", "unident_s"]

SMALL_CORRIDOR_PROGRESS = [
    ("PPO from scratch", 0.00),
    ("Structured shaping v3", 0.00),
    ("Full-chain BC", 1.00),
    ("3-cycle BC", 1.90),
    ("Wait-jittered BC", 2.50),
    ("BC+PPO best 25k", 3.00),
    ("BC+PPO final 50k", 0.85),
]

PARTNER_ROBUSTNESS = [
    ("simple", 7.50, 0.00),
    ("random0", 8.85, 1.65),
    ("random1", 5.80, 0.00),
    ("unident_s", 12.70, 12.60),
]


def read_router_scores(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {
            row["layout"]: float(row["mean_soups_delivered"])
            for row in rows
            if row.get("status") == "ok" and row.get("mean_soups_delivered")
        }


def write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def text(x: float, y: float, value: str, size: int = 13, anchor: str = "start", weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="#111827">{escape(value)}</text>'
    )


def score_label(value: float) -> str:
    return f"{value:.2f}"


def title_block(title: str, subtitle: str, width: int) -> list[str]:
    return [
        text(width / 2, 34, title, size=22, anchor="middle", weight="700"),
        text(width / 2, 58, subtitle, size=13, anchor="middle"),
    ]


def svg_open(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def svg_close() -> str:
    return "</svg>\n"


def build_router_comparison() -> None:
    scores = [(name, read_router_scores(path), color) for name, path, color in ROUTER_RUNS]
    width, height = 980, 520
    left, right, top, bottom = 90, 30, 95, 100
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_value = 13.0
    group_w = chart_w / len(LAYOUTS)
    bar_w = min(24, group_w / 5)

    parts = svg_open(width, height)
    parts += title_block(
        "Router comparison across onion layouts",
        "Mean soups delivered over 20 deterministic evaluation episodes",
        width,
    )

    for tick in range(0, 14, 2):
        y = top + chart_h - (tick / max_value) * chart_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(text(left - 12, y + 4, str(tick), size=12, anchor="end"))

    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#111827"/>')
    parts.append(f'<line x1="{left}" y1="{top + chart_h}" x2="{width - right}" y2="{top + chart_h}" stroke="#111827"/>')

    for group_idx, layout in enumerate(LAYOUTS):
        center = left + group_idx * group_w + group_w / 2
        parts.append(text(center, top + chart_h + 32, layout, size=12, anchor="middle"))
        for series_idx, (name, run_scores, color) in enumerate(scores):
            x = center - (len(scores) * bar_w) / 2 + series_idx * bar_w + 2
            value = run_scores.get(layout)
            if value is None:
                parts.append(
                    f'<rect x="{x:.1f}" y="{top + chart_h - 10:.1f}" width="{bar_w - 4:.1f}" '
                    f'height="10" fill="#d1d5db"/>'
                )
                parts.append(text(x + bar_w / 2 - 2, top + chart_h - 16, "skip", size=10, anchor="middle"))
                continue
            bar_h = (value / max_value) * chart_h
            y = top + chart_h - bar_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 4:.1f}" height="{bar_h:.1f}" '
                f'rx="2" fill="{color}"/>'
            )
            if layout == "small_corridor":
                parts.append(text(x + bar_w / 2 - 2, y - 6, score_label(value), size=10, anchor="middle"))

    legend_y = height - 38
    legend_x = 190
    for idx, (name, _run_scores, color) in enumerate(scores):
        x = legend_x + idx * 230
        parts.append(f'<rect x="{x}" y="{legend_y - 12}" width="14" height="14" fill="{color}" rx="2"/>')
        parts.append(text(x + 22, legend_y, name, size=12))

    parts.append(text(width - 30, height - 12, "Source: outputs/runs/router_*/metrics/router_eval.csv", size=10, anchor="end"))
    parts.append(svg_close())
    write(ASSET_DIR / "router_comparison.svg", "\n".join(parts))


def build_small_corridor_progression() -> None:
    width, height = 920, 500
    left, right, top, row_h = 240, 55, 90, 48
    max_value = 3.0
    chart_w = width - left - right

    parts = svg_open(width, height)
    parts += title_block(
        "Small corridor progression",
        "From PPO failure to checkpoint-selected 3-soup specialist",
        width,
    )

    for tick in range(0, 4):
        x = left + (tick / max_value) * chart_w
        parts.append(f'<line x1="{x:.1f}" y1="{top - 12}" x2="{x:.1f}" y2="{height - 72}" stroke="#e5e7eb"/>')
        parts.append(text(x, height - 48, str(tick), size=12, anchor="middle"))

    for idx, (label, value) in enumerate(SMALL_CORRIDOR_PROGRESS):
        y = top + idx * row_h
        bar_w = (value / max_value) * chart_w
        color = "#f59e0b" if "best" in label else "#3b82f6"
        if value == 0:
            color = "#9ca3af"
        parts.append(text(left - 16, y + 20, label, size=13, anchor="end"))
        parts.append(f'<rect x="{left}" y="{y}" width="{max(bar_w, 2):.1f}" height="26" fill="{color}" rx="3"/>')
        parts.append(text(left + max(bar_w, 2) + 8, y + 18, f"{value:.2f}", size=12))

    parts.append(text(left + chart_w / 2, height - 22, "Mean soups delivered", size=12, anchor="middle"))
    parts.append(text(width - 30, height - 12, "Source: docs/report_materials.md and run metrics", size=10, anchor="end"))
    parts.append(svg_close())
    write(ASSET_DIR / "small_corridor_progression.svg", "\n".join(parts))


def build_partner_robustness() -> None:
    width, height = 860, 430
    left, right, top, bottom = 100, 40, 92, 90
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_value = 13.0
    group_w = chart_w / len(PARTNER_ROBUSTNESS)
    bar_w = 32

    parts = svg_open(width, height)
    parts += title_block(
        "Partner robustness stress test",
        "Self-play score versus worst held-out partner score",
        width,
    )

    for tick in range(0, 14, 2):
        y = top + chart_h - (tick / max_value) * chart_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(text(left - 12, y + 4, str(tick), size=12, anchor="end"))

    for idx, (layout, self_play, held_out) in enumerate(PARTNER_ROBUSTNESS):
        center = left + idx * group_w + group_w / 2
        parts.append(text(center, top + chart_h + 32, layout, size=12, anchor="middle"))
        for offset, value, color, label_y_offset in [
            (-bar_w / 2, self_play, "#3b82f6", -6),
            (bar_w / 2, held_out, "#ef4444", -6),
        ]:
            bar_h = (value / max_value) * chart_h
            x = center + offset - bar_w / 2
            y = top + chart_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="3"/>')
            parts.append(text(x + bar_w / 2, y + label_y_offset, score_label(value), size=10, anchor="middle"))

    legend_y = height - 36
    parts.append(f'<rect x="280" y="{legend_y - 12}" width="14" height="14" fill="#3b82f6" rx="2"/>')
    parts.append(text(302, legend_y, "Self-play", size=12))
    parts.append(f'<rect x="410" y="{legend_y - 12}" width="14" height="14" fill="#ef4444" rx="2"/>')
    parts.append(text(432, legend_y, "Worst held-out partner", size=12))
    parts.append(text(width - 30, height - 12, "Source: outputs/runs/*/metrics/partner_matrix*.csv", size=10, anchor="end"))
    parts.append(svg_close())
    write(ASSET_DIR / "partner_robustness.svg", "\n".join(parts))


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    build_router_comparison()
    build_small_corridor_progression()
    build_partner_robustness()


if __name__ == "__main__":
    main()
