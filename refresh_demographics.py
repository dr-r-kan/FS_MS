"""Refresh demographic outputs without rerunning statistical analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analyse_focused_fs_ms import demographics, markdown_table


COHORT_HEADING = "## Cohort and event characteristics"
NEXT_HEADING = "## Descriptive comparisons"


def replace_cohort_section(report: str, demographic_table: pd.DataFrame, event_table: pd.DataFrame) -> str:
    start = report.index(COHORT_HEADING)
    end = report.index(NEXT_HEADING, start)
    replacement = "\n\n".join([
        COHORT_HEADING,
        markdown_table(demographic_table, list(demographic_table.columns)),
        markdown_table(event_table, list(event_table.columns)),
        "",
    ])
    return report[:start] + replacement + report[end:]


def self_check() -> None:
    demo = pd.DataFrame([{"group": "FS", "female": 2, "male": 1}])
    events = pd.DataFrame([{"group": "FS", "records": 3}])
    report = f"before\n{COHORT_HEADING}\nold\n{NEXT_HEADING}\nafter"
    refreshed = replace_cohort_section(report, demo, events)
    assert "old" not in refreshed and refreshed.count(NEXT_HEADING) == 1
    assert "| FS | 2 | 1 |" in refreshed and "| FS | 3 |" in refreshed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path(r"G:\FS_microstates_study\vg_microstates\vg_microstate_manifest_mat_paper_20260723.csv"))
    parser.add_argument("--demographics-csv", type=Path, default=Path("analysis_database.csv"))
    parser.add_argument("--analysis-dir", type=Path,
                        default=Path(r"G:\FS_microstates_study\vg_microstates\focused_analysis_final"))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    self_check()
    if args.self_check:
        print("self-check passed")
        return

    report_path = args.analysis_dir / "RESULTS.md"
    for path in (args.manifest, args.demographics_csv, report_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    data_dir = args.analysis_dir / "data"
    manifest = pd.read_csv(args.manifest)
    demographic_table, event_table = demographics(manifest, args.demographics_csv, data_dir)
    report = report_path.read_text(encoding="utf-8")
    report_path.write_text(replace_cohort_section(report, demographic_table, event_table), encoding="utf-8")

    participants = pd.read_csv(data_dir / "participant_demographics.csv")
    unknown = int(participants.gender.eq("unknown").sum())
    print(f"Refreshed demographics for {len(participants)} participants ({unknown} unknown).")
    print("Statistical outputs were not rerun or modified.")


if __name__ == "__main__":
    main()
