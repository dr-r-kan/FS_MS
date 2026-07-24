"""Read and validate the fitted-state labels selected at the global K=6 fit."""

from __future__ import annotations

import csv
from pathlib import Path


def validate_label_order(labels) -> tuple[str, ...]:
    order = tuple(str(label).strip().upper() for label in labels)
    if len(order) != 6 or len(set(order)) != 6 or not set(order) <= set("ABCDEFG") or not set("ACE") <= set(order):
        raise ValueError("Global labels must be six unique A-G labels including A, C and E")
    return order


def load_global_labels(results_dir: Path) -> tuple[str, ...]:
    manual = results_dir / "global" / "global_manual_labels.csv"
    if manual.exists():
        with manual.open(newline="", encoding="utf-8-sig") as handle:
            rows = sorted(csv.DictReader(handle), key=lambda row: int(row["state_index"]))
        return validate_label_order(row["label"] for row in rows)

    covariance = results_dir / "hierarchical_state_covariance_metrics.csv"
    with covariance.open(newline="", encoding="utf-8-sig") as handle:
        rows = sorted(
            (row for row in csv.DictReader(handle) if row["level"] == "global"),
            key=lambda row: int(float(row["state_index"])),
        )
    return validate_label_order(row["template_label"] for row in rows)


def format_label_order(labels) -> str:
    return ", ".join(f"State {index}={label}" for index, label in enumerate(validate_label_order(labels), 1))


if __name__ == "__main__":
    assert validate_label_order("FEGBAC") == ("F", "E", "G", "B", "A", "C")
    try:
        validate_label_order("FFGBAC")
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate-label check failed")
    print("self-check passed")
