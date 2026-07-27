#!/usr/bin/env python3
"""Build a manifest CSV for preprocessed LEMON EEGLAB files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PARTICIPANT_RE = re.compile(r"(sub-\d+)", re.IGNORECASE)


def infer_participant(path: Path) -> str:
    match = PARTICIPANT_RE.search(path.name)
    if match:
        return match.group(1)
    stem = path.stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def infer_condition(path: Path) -> str:
    name = path.stem.lower()
    tokens = name.replace("-", "_")
    if "_ec" in tokens or "eyesclosed" in tokens or "closed" in tokens:
        return "eyes_closed"
    if "_eo" in tokens or "eyesopen" in tokens or "open" in tokens:
        return "eyes_open"
    return "unknown"


def collect_set_files(input_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        files = sorted(p for p in input_dir.rglob("*.set") if p.is_file())
    else:
        files = sorted(p for p in input_dir.glob("*.set") if p.is_file())
    return files


def write_manifest(files: list[Path], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["participant", "condition", "file_path"])
        for path in files:
            writer.writerow(
                [infer_participant(path), infer_condition(path), str(path.resolve())]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a LEMON manifest CSV from preprocessed EEGLAB .set files."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing preprocessed LEMON .set files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only scan the top level of --input-dir.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    files = collect_set_files(input_dir, recursive=not args.non_recursive)
    if not files:
        raise SystemExit(f"No .set files found under: {input_dir}")

    write_manifest(files, args.output.expanduser().resolve())
    print(f"Wrote {len(files)} rows to {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
