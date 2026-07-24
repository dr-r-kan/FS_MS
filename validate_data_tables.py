"""Validate supplementary CSV readability, dictionary coverage and deidentification."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


PATTERNS = {
    "raw_participant_code": re.compile(r"\bP_[0-9a-f]{10}\b", re.IGNORECASE),
    "windows_user_path": re.compile(r"\bC:\\Users\\", re.IGNORECASE),
    "study_drive_path": re.compile(r"\bG:\\", re.IGNORECASE),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dictionary_path = args.data_dir / "data_dictionary.csv"
    dictionary = pd.read_csv(dictionary_path) if dictionary_path.exists() else pd.DataFrame(columns=["file"])
    documented = set(dictionary.get("file", pd.Series(dtype=str)).astype(str))
    tables, issues = [], []
    for path in sorted(args.data_dir.glob("*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            issues.append(f"Unreadable CSV {path.name}: {exc}")
            continue
        duplicate_columns = frame.columns[frame.columns.duplicated()].tolist()
        if duplicate_columns:
            issues.append(f"Duplicate columns in {path.name}: {duplicate_columns}")
        hits = {name: 0 for name in PATTERNS}
        for column in frame.select_dtypes(include=["object", "string"]).columns:
            values = frame[column].dropna().astype(str)
            for name, pattern in PATTERNS.items():
                hits[name] += int(values.str.contains(pattern, regex=True).sum())
        for name, count in hits.items():
            if count:
                issues.append(f"Potential identifier/path leak in {path.name}: {name}={count}")
        tables.append({"file": path.name, "rows": int(len(frame)), "columns": int(len(frame.columns)),
                       "documented": path.name in documented, "duplicate_columns": duplicate_columns,
                       "identifier_or_path_hits": hits})
    undocumented = sorted(row["file"] for row in tables if not row["documented"] and row["file"] != "data_dictionary.csv")
    if undocumented:
        issues.append("Undocumented CSV files: " + ", ".join(undocumented))
    result = {"tables": tables, "all_csv_readable": not any(x.startswith("Unreadable") for x in issues),
              "all_tables_documented": not undocumented, "deidentification_scan_passed": not any("leak" in x for x in issues),
              "issues": issues, "validation_passed": not issues}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
