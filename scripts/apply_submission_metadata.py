#!/usr/bin/env python3
"""Apply real course/team/member metadata to the report and slides.

The script is intentionally separate from the report source so the repository
does not need fake names or student IDs. Fill a JSON file based on
report/submission_metadata.example.json, then run this script and regenerate the
exports.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
START = "<!-- submission-metadata:start -->"
END = "<!-- submission-metadata:end -->"
PLACEHOLDER_TOKENS = ("填写", "TODO", "TBD", "<", ">")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        default="report/submission_metadata.json",
        help="JSON metadata file. Default: report/submission_metadata.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated metadata blocks without editing files.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Regenerate report HTML/PDF exports after applying metadata.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="With --export, regenerate HTML only.",
    )
    return parser.parse_args()


def read_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("metadata must be a JSON object")
    validate_metadata(data)
    return data


def value_is_placeholder(value: str) -> bool:
    return any(token in value for token in PLACEHOLDER_TOKENS)


def require_text(data: dict[str, Any], key: str) -> str:
    value = str(data.get(key, "")).strip()
    if not value:
        raise ValueError(f"metadata.{key} must not be empty")
    if value_is_placeholder(value):
        raise ValueError(f"metadata.{key} still looks like a placeholder: {value}")
    return value


def validate_metadata(data: dict[str, Any]) -> None:
    require_text(data, "course")
    require_text(data, "date")
    team = str(data.get("team", "")).strip()
    if team and value_is_placeholder(team):
        raise ValueError(f"metadata.team still looks like a placeholder: {team}")

    members = data.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("metadata.members must be a non-empty list")
    for idx, member in enumerate(members, start=1):
        if not isinstance(member, dict):
            raise ValueError(f"metadata.members[{idx}] must be an object")
        for key in ("name", "student_id"):
            value = str(member.get(key, "")).strip()
            if not value:
                raise ValueError(f"metadata.members[{idx}].{key} must not be empty")
            if value_is_placeholder(value):
                raise ValueError(
                    f"metadata.members[{idx}].{key} still looks like a placeholder: {value}"
                )


def member_label(member: dict[str, Any]) -> str:
    label = f"{member['name']}（{member['student_id']}）"
    role = str(member.get("role", "")).strip()
    if role:
        label += f"，{role}"
    return label


def metadata_lines(data: dict[str, Any]) -> list[str]:
    lines = [
        f"课程：{data['course']}",
    ]
    team = str(data.get("team", "")).strip()
    if team:
        lines.append(f"队伍：{team}")
    lines.append("成员：" + "；".join(member_label(member) for member in data["members"]))
    lines.append(f"日期：{data['date']}")
    return lines


def report_block(data: dict[str, Any]) -> str:
    lines = [START, "提交信息："]
    lines.extend(f"- {line}" for line in metadata_lines(data))
    lines.append(END)
    return "\n".join(lines)


def slides_block(data: dict[str, Any]) -> str:
    lines = [START]
    lines.extend(metadata_lines(data))
    lines.append(END)
    return "\n".join(lines)


def replace_between_markers(text: str, block: str) -> str | None:
    start = text.find(START)
    end = text.find(END)
    if start == -1 and end == -1:
        return None
    if start == -1 or end == -1 or end < start:
        raise ValueError("Malformed submission metadata marker block")
    end += len(END)
    return text[:start].rstrip() + "\n\n" + block + "\n\n" + text[end:].lstrip()


def insert_after_line(text: str, predicate, block: str) -> str:
    existing = replace_between_markers(text, block)
    if existing is not None:
        return existing

    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if predicate(line):
            return "\n".join(lines[: idx + 1] + ["", block, ""] + lines[idx + 1 :]) + "\n"
    raise ValueError("Could not find insertion point")


def apply_metadata(data: dict[str, Any], dry_run: bool) -> None:
    report_path = ROOT / "report" / "final_report.md"
    slides_path = ROOT / "report" / "slides.md"

    report_text = report_path.read_text(encoding="utf-8")
    slides_text = slides_path.read_text(encoding="utf-8")

    new_report = insert_after_line(
        report_text,
        lambda line: line.startswith("更新时间："),
        report_block(data),
    )
    new_slides = insert_after_line(
        slides_text,
        lambda line: line == "PPO baseline → failure analysis → specialists → layout router",
        slides_block(data),
    )

    if dry_run:
        print("Report metadata block:")
        print(report_block(data))
        print()
        print("Slides metadata block:")
        print(slides_block(data))
        return

    report_path.write_text(new_report, encoding="utf-8")
    slides_path.write_text(new_slides, encoding="utf-8")
    print(f"updated {report_path.relative_to(ROOT)}")
    print(f"updated {slides_path.relative_to(ROOT)}")


def export_reports(no_pdf: bool) -> None:
    cmd = [sys.executable, "scripts/build_report_exports.py"]
    if no_pdf:
        cmd.append("--no-pdf")
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> int:
    args = parse_args()
    metadata_path = Path(args.metadata).expanduser()
    if not metadata_path.is_absolute():
        metadata_path = ROOT / metadata_path

    data = read_metadata(metadata_path)
    apply_metadata(data, args.dry_run)
    if args.export and not args.dry_run:
        export_reports(args.no_pdf)
    elif args.export and args.dry_run:
        print("Dry run requested; skipped export.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
