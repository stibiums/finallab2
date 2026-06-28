#!/usr/bin/env python3
"""Preflight-check the final Overcooked submission package.

This script does not edit files. It checks required artifacts, optional identity
metadata, demo recording status, package contents, and the Git worktree so the
remaining manual submission work is explicit.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import apply_submission_metadata  # noqa: E402
import package_submission  # noqa: E402


PLACEHOLDER_ARCHIVE_NAMES = {"", "学号+姓名", "submission_dry_run", "test"}
DEMO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}
DEFAULT_DEMO_VIDEO = ROOT / "report" / "demo_video_draft.mp4"


class Reporter:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def pass_(self, message: str) -> None:
        print(f"PASS {message}")

    def warn(self, message: str) -> None:
        self.warnings += 1
        print(f"WARN {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        print(f"FAIL {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        default="",
        help="Final archive stem, normally the required 学号+姓名 string.",
    )
    parser.add_argument(
        "--metadata",
        default="report/submission_metadata.json",
        help="Optional real metadata JSON. Default: report/submission_metadata.json",
    )
    parser.add_argument(
        "--demo-video",
        help="Optional screen recording to verify before final packaging.",
    )
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Check packaging with external/PantheonRL included.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures.",
    )
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def check_path(reporter: Reporter, path: Path, label: str) -> None:
    if path.exists():
        reporter.pass_(f"{label}: {rel(path)}")
    else:
        reporter.fail(f"missing {label}: {rel(path)}")


def check_required_artifacts(reporter: Reporter) -> None:
    required = [
        ("report PDF", ROOT / "report" / "final_report.pdf"),
        ("slides PDF", ROOT / "report" / "slides.pdf"),
        ("submission manifest", ROOT / "docs" / "submission_manifest.md"),
        ("demo script", ROOT / "report" / "demo_script.md"),
        ("package helper", ROOT / "scripts" / "package_submission.py"),
        ("metadata helper", ROOT / "scripts" / "apply_submission_metadata.py"),
        ("report source", ROOT / "report" / "final_report.md"),
        ("slides source", ROOT / "report" / "slides.md"),
        ("source package", ROOT / "src" / "overcooked_project"),
        ("configs", ROOT / "configs"),
        ("scripts", ROOT / "scripts"),
    ]
    for label, path in required:
        check_path(reporter, path, label)


def check_package_manifest(reporter: Reporter, include_external: bool, demo_video: str | None) -> None:
    try:
        items = package_submission.package_items(include_external, demo_video)
    except Exception as exc:
        reporter.fail(f"package manifest validation failed: {exc}")
        return

    total_size = sum(package_submission.path_size(src) for src, _ in items)
    reporter.pass_(
        f"package manifest: {len(items)} items, estimated source size "
        f"{package_submission.format_size(total_size)}"
    )

    required_inside = {
        Path("report/final_report.pdf"),
        Path("report/slides.pdf"),
        Path("docs/submission_manifest.md"),
        Path("report/demo_script.md"),
        Path("src"),
        Path("configs"),
        Path("scripts"),
        Path("outputs/runs/baseline_simple"),
        Path("outputs/runs/router_onion_layouts_with_small_corridor_jitter3_bc_ppo"),
    }
    destinations = {dst for _, dst in items}
    for dst in sorted(required_inside):
        if package_covers(dst, destinations):
            reporter.pass_(f"package includes {dst}")
        else:
            reporter.fail(f"package missing {dst}")


def package_covers(required: Path, destinations: set[Path]) -> bool:
    return any(required == destination or required.is_relative_to(destination) for destination in destinations)


def check_archive_name(reporter: Reporter, name: str) -> None:
    stripped = name.strip()
    if stripped in PLACEHOLDER_ARCHIVE_NAMES:
        reporter.warn("final archive name is not provided yet; expected 学号+姓名")
        return
    try:
        package_submission.validate_name(stripped)
    except Exception as exc:
        reporter.fail(f"invalid archive name: {exc}")
        return
    reporter.pass_(f"archive name looks usable: {stripped}.zip")


def check_metadata(reporter: Reporter, metadata: str) -> None:
    path = Path(metadata).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        reporter.warn(
            f"real metadata file not found: {rel(path)}; "
            "copy report/submission_metadata.example.json if identity metadata is required"
        )
        return
    try:
        apply_submission_metadata.read_metadata(path)
    except Exception as exc:
        reporter.fail(f"metadata validation failed: {exc}")
        return
    reporter.pass_(f"metadata validates: {rel(path)}")


def check_demo_video(reporter: Reporter, demo_video: str | None) -> None:
    if not demo_video:
        if DEFAULT_DEMO_VIDEO.exists():
            reporter.warn(
                "demo video path not provided; using generated report/demo_video_draft.mp4, "
                "but replace it with a real screen recording if strictly required"
            )
            reporter.pass_(f"demo video draft exists: {rel(DEFAULT_DEMO_VIDEO)}")
            return
        reporter.warn("demo video path not provided; GIF demos exist but prompt asks for screen recording")
        return
    path = Path(demo_video).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        reporter.fail(f"demo video not found: {rel(path)}")
        return
    if path.suffix.lower() not in DEMO_SUFFIXES:
        reporter.warn(f"demo video has unusual suffix: {path.suffix}")
    reporter.pass_(f"demo video exists: {rel(path)}")


def check_git_status(reporter: Reporter) -> None:
    try:
        result = subprocess.run(
            ["git", "status", "--short", "--branch"],
            check=True,
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
    except Exception as exc:
        reporter.warn(f"could not inspect git status: {exc}")
        return

    lines = result.stdout.splitlines()
    branch_line = lines[0] if lines else ""
    dirty_lines = lines[1:]
    if dirty_lines:
        reporter.warn("git worktree has uncommitted changes")
        for line in dirty_lines:
            print(f"     {line}")
    else:
        reporter.pass_("git worktree is clean")
    if "ahead" in branch_line:
        reporter.warn(f"branch is ahead of remote: {branch_line}")
    else:
        reporter.pass_(f"branch status: {branch_line}")


def main() -> int:
    args = parse_args()
    reporter = Reporter()

    check_required_artifacts(reporter)
    check_archive_name(reporter, args.name)
    check_metadata(reporter, args.metadata)
    check_demo_video(reporter, args.demo_video)
    check_package_manifest(reporter, args.include_external, args.demo_video)
    check_git_status(reporter)

    print()
    print(f"Summary: {reporter.failures} failure(s), {reporter.warnings} warning(s)")
    if reporter.failures:
        return 1
    if args.strict and reporter.warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
