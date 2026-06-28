#!/usr/bin/env python3
"""Build a demo-video draft from the report narrative and recorded GIF demos."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = ROOT / "tmp" / "demo_video"

WIDTH = 1280
HEIGHT = 720
FPS = 12

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]

DEMO_SEQUENCE = [
    {
        "title": "Default PPO baseline",
        "subtitle": "simple layout, default shaping",
        "bullets": [
            "Mean soups: 7.50",
            "Takeaway: reward shaping makes the easy map learnable.",
        ],
        "gif": "outputs/runs/baseline_simple/demo/demo.gif",
    },
    {
        "title": "Sparse reward failure",
        "subtitle": "simple layout, no shaping",
        "bullets": [
            "Mean soups: 0.00",
            "Takeaway: sparse-only PPO does not discover the delivery chain at this budget.",
        ],
        "gif": "outputs/runs/no_shaping_simple/demo/demo.gif",
    },
    {
        "title": "Naive multi-layout failure",
        "subtitle": "mixed-layout PPO curriculum",
        "bullets": [
            "Mean soups: near zero on most fixed layouts",
            "Takeaway: simple layout mixing is not enough.",
        ],
        "gif": "outputs/runs/multi_layout_curriculum/demo/simple_demo.gif",
    },
    {
        "title": "random0 specialist",
        "subtitle": "single-layout specialist route",
        "bullets": [
            "Best route uses baseline_random0_long_seed52: 8.85 soups",
            "Demo GIF uses the random0 specialist behavior.",
        ],
        "gif": "outputs/runs/baseline_random0_long/demo/random0_long_demo.gif",
    },
    {
        "title": "random1 specialist",
        "subtitle": "single-layout specialist route",
        "bullets": [
            "Mean soups: 5.80",
            "Caveat: self-play success does not imply partner robustness.",
        ],
        "gif": "outputs/runs/baseline_random1/demo/random1_demo.gif",
    },
    {
        "title": "unident_s specialist",
        "subtitle": "strongest PPO hard-layout specialist",
        "bullets": [
            "Mean soups: 12.70",
            "This layout is robust across the tested partner seeds.",
        ],
        "gif": "outputs/runs/baseline_unident_s/demo/unident_s_demo.gif",
    },
    {
        "title": "small_corridor rescue",
        "subtitle": "perturbed BC + checkpoint-selected PPO",
        "bullets": [
            "Best checkpoint: 3.00 soups",
            "Final 50k checkpoint collapses, so checkpoint selection is required.",
        ],
        "gif": "outputs/runs/small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune/demo/small_corridor_full_chain_3cycle_jitter3_bc_ppo_best_demo.gif",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="report/demo_video_draft.mp4",
        help="Output MP4 path. Default: report/demo_video_draft.mp4",
    )
    parser.add_argument(
        "--gif-seconds",
        type=float,
        default=0.0,
        help="Trim each GIF segment to this many seconds. Default 0 keeps full GIF duration.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep tmp/demo_video segments.")
    return parser.parse_args()


def ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg not found")
    return path


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, selected_font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=selected_font)
    return bbox[2] - bbox[0]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, selected_font: ImageFont.ImageFont, max_width: int) -> list[str]:
    if text_width(draw, text, selected_font) <= max_width:
        return [text]
    words = text.split()
    if len(words) <= 1:
        lines: list[str] = []
        current = ""
        for char in text:
            if current and text_width(draw, current + char, selected_font) > max_width:
                lines.append(current)
                current = char
            else:
                current += char
        if current:
            lines.append(current)
        return lines

    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and text_width(draw, candidate, selected_font) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def draw_card(path: Path, title: str, subtitle: str, bullets: list[str], footer: str = "") -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f8fafc")
    draw = ImageDraw.Draw(image)
    title_font = font(52)
    subtitle_font = font(30)
    body_font = font(28)
    small_font = font(20)

    draw.rectangle((0, 0, WIDTH, 108), fill="#172033")
    draw.text((72, 30), title, fill="white", font=title_font)
    draw.text((74, 138), subtitle, fill="#2563eb", font=subtitle_font)

    y = 220
    for bullet in bullets:
        wrapped = wrap_text(draw, bullet, body_font, WIDTH - 190)
        draw.text((88, y), "-", fill="#172033", font=body_font)
        for line in wrapped:
            draw.text((124, y), line, fill="#172033", font=body_font)
            y += 42
        y += 18

    if footer:
        draw.text((74, HEIGHT - 64), footer, fill="#536071", font=small_font)

    image.save(path)


def run(cmd: list[str]) -> None:
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stderr.splitlines()[-40:])
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{tail}")


def card_segment(card: Path, segment: Path, duration: float) -> None:
    run(
        [
            ffmpeg(),
            "-y",
            "-loop",
            "1",
            "-t",
            str(duration),
            "-i",
            str(card),
            "-vf",
            f"fps={FPS},format=yuv420p",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            str(segment),
        ]
    )


def gif_segment(gif: Path, segment: Path, seconds: float) -> None:
    cmd = [
        ffmpeg(),
        "-y",
        "-i",
        str(gif),
    ]
    if seconds > 0:
        cmd.extend(["-t", str(seconds)])
    cmd.extend(
        [
            "-vf",
            f"fps={FPS},scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=white,format=yuv420p",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            str(segment),
        ]
    )
    run(cmd)


def concat_segments(segments: list[Path], output: Path) -> None:
    concat_file = TMP_DIR / "segments.txt"
    concat_file.write_text(
        "\n".join(f"file '{segment.resolve()}'" for segment in segments) + "\n",
        encoding="utf-8",
    )
    run(
        [
            ffmpeg(),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(output),
        ]
    )


def validate_gifs() -> None:
    missing = [item["gif"] for item in DEMO_SEQUENCE if not (ROOT / item["gif"]).exists()]
    if missing:
        raise FileNotFoundError("Missing GIF demos: " + ", ".join(missing))


def main() -> int:
    args = parse_args()
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = ROOT / output
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} exists; use --force to overwrite")

    validate_gifs()
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    segments: list[Path] = []

    intro = TMP_DIR / "card_000_intro.png"
    draw_card(
        intro,
        "Overcooked MARL Demo",
        "PPO baseline -> failure analysis -> specialists -> router",
        [
            "Goal: train two agents to cooperate in Overcooked layouts.",
            "Main metric: soups delivered = sparse reward / 20.",
            "Final result: five onion layouts, 7.98 average soups, 3.00 minimum soups.",
        ],
        footer="Generated from report/demo_script.md and outputs/runs/*/demo/*.gif",
    )
    segment = TMP_DIR / "seg_000_intro.mp4"
    card_segment(intro, segment, 5.0)
    segments.append(segment)

    command_card = TMP_DIR / "card_001_commands.png"
    draw_card(
        command_card,
        "How to run",
        "Reproducible entry points",
        [
            "Setup: git submodule update --init --recursive && bash scripts/create_env.sh",
            "Smoke test: bash scripts/smoke_test.sh",
            "Strongest router: bash scripts/evaluate_router.sh configs/router_simple_random0.json ...",
            "Package: python scripts/package_submission.py --name 学号+姓名",
        ],
    )
    segment = TMP_DIR / "seg_001_commands.mp4"
    card_segment(command_card, segment, 5.0)
    segments.append(segment)

    for idx, item in enumerate(DEMO_SEQUENCE, start=2):
        card = TMP_DIR / f"card_{idx:03d}.png"
        draw_card(card, item["title"], item["subtitle"], item["bullets"])
        segment = TMP_DIR / f"seg_{idx:03d}_card.mp4"
        card_segment(card, segment, 4.0)
        segments.append(segment)

        gif = ROOT / item["gif"]
        segment = TMP_DIR / f"seg_{idx:03d}_gif.mp4"
        gif_segment(gif, segment, args.gif_seconds)
        segments.append(segment)

    conclusion = TMP_DIR / "card_999_conclusion.png"
    draw_card(
        conclusion,
        "Conclusion",
        "What the final system claims",
        [
            "A single PPO policy did not generalize across layouts.",
            "Specialist composition plus explicit diagnostics is the strongest practical method here.",
            "Partner robustness and checkpoint selection must be reported, not hidden.",
        ],
    )
    segment = TMP_DIR / "seg_999_conclusion.mp4"
    card_segment(conclusion, segment, 6.0)
    segments.append(segment)

    concat_segments(segments, output)
    print(f"Wrote {output.relative_to(ROOT)}")

    if not args.keep_temp:
        shutil.rmtree(TMP_DIR)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
