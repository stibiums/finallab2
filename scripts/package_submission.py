#!/usr/bin/env python3
"""Build the final course submission archive.

The compact default archive includes report files, source code, configs,
selected trained runs, metrics, and GIF demos. The large PantheonRL submodule is
left out by default and can be included explicitly with --include-external.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CORE_FILES = [
    "README.md",
    "requirements.txt",
    ".gitignore",
    ".gitmodules",
]

CORE_DIRS = [
    "configs",
    "docs",
    "report",
    "scripts",
    "src",
]

SELECTED_RUNS = [
    "baseline_simple",
    "no_shaping_simple",
    "multi_layout_curriculum",
    "baseline_random0_long",
    "baseline_random0_long_seed52",
    "baseline_random1",
    "baseline_random1_seed73",
    "baseline_unident_s",
    "small_corridor_full_chain_3cycle_jitter3_role_balanced_bc_from_v3",
    "small_corridor_full_chain_3cycle_jitter3_bc_ppo_finetune",
    "router_onion_layouts_with_small_corridor_jitter3_bc_ppo",
    "partner_diversity_random1",
    "partner_diversity_random1_three_partners",
]

IGNORE_NAMES = {
    ".DS_Store",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "tensorboard",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        required=True,
        help="Archive stem, normally the required 学号+姓名 string.",
    )
    parser.add_argument(
        "--output-dir",
        default="dist",
        help="Directory for the generated zip archive. Default: dist",
    )
    parser.add_argument(
        "--demo-video",
        help="Optional screen-recording file to include under demo_recording/.",
    )
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Include external/PantheonRL. This makes the archive much larger.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print what would be packaged without writing a zip.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing archive with the same name.",
    )
    return parser.parse_args()


def validate_name(name: str) -> None:
    if not name.strip():
        raise ValueError("--name must not be empty")
    if "/" in name or "\\" in name:
        raise ValueError("--name must be an archive stem, not a path")


def ignore_filter(_dir: str, names: list[str]) -> set[str]:
    return {name for name in names if name in IGNORE_NAMES or name.endswith(".pyc")}


def require_path(rel_path: str) -> Path:
    path = ROOT / rel_path
    if not path.exists():
        raise FileNotFoundError(rel_path)
    return path


def package_items(include_external: bool, demo_video: str | None) -> list[tuple[Path, Path]]:
    items: list[tuple[Path, Path]] = []

    for rel_path in CORE_FILES:
        src = require_path(rel_path)
        items.append((src, Path(rel_path)))

    for rel_path in CORE_DIRS:
        src = require_path(rel_path)
        items.append((src, Path(rel_path)))

    for run_name in SELECTED_RUNS:
        src = require_path(f"outputs/runs/{run_name}")
        items.append((src, Path("outputs") / "runs" / run_name))

    if include_external:
        src = require_path("external/PantheonRL")
        items.append((src, Path("external") / "PantheonRL"))

    if demo_video:
        src = Path(demo_video).expanduser()
        if not src.is_absolute():
            src = ROOT / src
        if not src.exists():
            raise FileNotFoundError(str(src))
        items.append((src, Path("demo_recording") / src.name))

    return items


def copy_item(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, ignore=ignore_filter)
    else:
        shutil.copy2(src, dst)


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def main() -> int:
    args = parse_args()
    validate_name(args.name)

    output_dir = (ROOT / args.output_dir).resolve()
    archive_path = output_dir / f"{args.name}.zip"
    stage_root = ROOT / "tmp" / "submission_package" / args.name
    items = package_items(args.include_external, args.demo_video)

    total_size = sum(path_size(src) for src, _ in items)
    print(f"Archive name: {archive_path}")
    print(f"Items: {len(items)}")
    print(f"Estimated source size: {format_size(total_size)}")

    for src, dst in items:
        print(f"  {src.relative_to(ROOT)} -> {dst}")

    if args.dry_run:
        print("Dry run complete; no archive written.")
        return 0

    if archive_path.exists() and not args.force:
        raise FileExistsError(f"{archive_path} already exists; use --force to overwrite")

    if stage_root.exists():
        shutil.rmtree(stage_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    for src, rel_dst in items:
        copy_item(src, stage_root / rel_dst)

    if archive_path.exists():
        archive_path.unlink()
    shutil.make_archive(str(archive_path.with_suffix("")), "zip", stage_root.parent, stage_root.name)
    print(f"Wrote {archive_path} ({format_size(archive_path.stat().st_size)})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
