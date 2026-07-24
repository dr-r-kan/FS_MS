"""Combine disjoint window-refit chunks with production integrity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from microstate_labels import load_global_labels

SPM_MS_COMMIT = "0dbde1146f6777412dce19edfa4962cd45157fca"
SPM_COMMIT = "04db1be762077630772ee660a470a0fd63d4ff43"


def path_key(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace("/", "\\", regex=False).str.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frames = [pd.read_csv(path, low_memory=False) for path in args.inputs]
    data = pd.concat(frames, ignore_index=True)
    manifest = pd.read_csv(args.manifest)
    label_order = load_global_labels(args.out.parent)
    expected = set(path_key(manifest.loc[manifest["condition"].astype(str).str.lower().isin({"preict", "preictal"}), "file_path"]))
    observed = set(path_key(data["file_path"]))
    key = ["file_path", "window_index_from_ictus", "template_label"]

    errors: list[str] = []
    if observed != expected:
        errors.append(f"event coverage differs: missing={len(expected-observed)}, extra={len(observed-expected)}")
    if data.duplicated(key).any():
        errors.append(f"duplicate event/window/state rows={int(data.duplicated(key).sum())}")
    if set(data["template_label"].dropna().astype(str)) != set(label_order):
        errors.append(f"labels differ from the saved manual mapping: {label_order}")
    if not data["K"].eq(6).all():
        errors.append("K is not uniformly six")
    if not data["prior_mode"].eq("spm_mix_prior_component_means").all() or not data["spm_prior_weight"].eq(1).all():
        errors.append("prior mode or weight differs from the locked mean prior")
    if not data["spm_ms_commit"].eq(SPM_MS_COMMIT).all() or not data["spm_commit"].eq(SPM_COMMIT).all():
        errors.append("SPM_MS/SPM provenance differs from the pinned commits")
    if (pd.to_numeric(data["minutes_to_ictus"], errors="coerce") >= 0).any():
        errors.append("a window includes or follows seizure onset")
    good = data["fit_status"].eq("ok")
    amp = [column for column in data.columns if column.startswith("amp_")]
    norms = np.sqrt(np.square(data.loc[good, amp].apply(pd.to_numeric, errors="coerce")).sum(axis=1))
    if len(norms) and not np.allclose(norms, 1.0, atol=1e-8):
        errors.append("a valid fitted map is not unit norm")
    if (pd.to_numeric(data.loc[good, "window_peak_count"], errors="coerce") < 30).any():
        errors.append("a reported valid map comes from a window with fewer than 30 real peaks")
    if (pd.to_numeric(data.loc[good, "state_peak_count"], errors="coerce") < 5).any():
        errors.append("a reported valid state map has fewer than five assigned peaks")
    assigned = data[data["window_state_index"].notna()]
    assignment_counts = assigned.groupby(["file_path", "window_index_from_ictus"])["window_state_index"].nunique(dropna=True)
    if len(assignment_counts) and not assignment_counts.eq(6).all():
        errors.append("a fitted window lacks a one-to-one six-state assignment")

    audit = {
        "inputs": [str(path) for path in args.inputs],
        "rows": int(len(data)),
        "events": int(len(observed)),
        "expected_events": int(len(expected)),
        "windows": int(data[["file_path", "window_index_from_ictus"]].drop_duplicates().shape[0]),
        "fit_status": {str(k): int(v) for k, v in data["fit_status"].value_counts(dropna=False).to_dict().items()},
        "labels_by_state": dict(enumerate(label_order, 1)),
        "errors": errors,
        "validation_passed": not errors,
    }
    if errors:
        raise SystemExit(json.dumps(audit, indent=2))

    data["_label_order"] = pd.Categorical(data["template_label"], label_order, ordered=True)
    data = data.sort_values(["participant", "file_path", "window_index_from_ictus", "_label_order"]).drop(columns="_label_order")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False)
    args.out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
