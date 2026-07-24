"""Validate the fresh paper hierarchy and emit a compact, identifier-free audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from microstate_labels import load_global_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_dir
    manifest = pd.read_csv(args.manifest)
    checks: dict[str, object] = {}
    issues: list[str] = []
    manual_labels = root / "global" / "global_manual_labels.csv"
    if not manual_labels.exists():
        issues.append("Saved manual global-label audit is missing.")
    try:
        label_order = load_global_labels(root)
        checks["manual_global_labels"] = list(label_order)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        label_order = ()
        issues.append(f"Global labels are invalid: {exc}")

    paths = manifest["file_path"].astype(str).str.replace("/", "\\", regex=False).str.lower()
    checks["manifest_rows"] = int(len(manifest))
    checks["manifest_participants"] = int(manifest["participant"].nunique())
    checks["manifest_unique_paths"] = int(paths.nunique())
    checks["manifest_conditions"] = {str(k): int(v) for k, v in manifest["condition"].value_counts().to_dict().items()}
    if not paths.is_unique:
        issues.append("Paper manifest contains duplicate file paths.")

    required = [
        "hierarchical_fit_summary.csv",
        "hierarchical_state_covariance_metrics.csv",
        "participant_condition_record_backfit_summary.csv",
        "global_record_backfit_summary.csv",
        "participant_condition_preictal_time_metrics.csv",
        "global_preictal_time_metrics.csv",
        "participant_condition_transition_metrics.csv",
        "global_transition_metrics.csv",
        "hierarchical_microstate_results.mat",
    ]
    missing = [name for name in required if not (root / name).exists()]
    checks["required_outputs_present"] = len(required) - len(missing)
    checks["required_outputs_expected"] = len(required)
    if missing:
        issues.append("Missing required outputs: " + ", ".join(missing))

    hierarchy_path = root / "hierarchical_fit_summary.csv"
    if hierarchy_path.exists():
        hierarchy = pd.read_csv(hierarchy_path)
        checks["hierarchy_rows"] = int(len(hierarchy))
        checks["hierarchy_level_counts"] = {str(k): int(v) for k, v in hierarchy["level"].value_counts().to_dict().items()}
        checks["selected_k_values"] = sorted(int(x) for x in hierarchy["K_estimated"].dropna().unique())
        checks["participant_fits_complete_strong"] = int((hierarchy.loc[hierarchy["level"].eq("participant"), "template_strong_matches"] == hierarchy.loc[hierarchy["level"].eq("participant"), "K_estimated"]).sum())
        checks["participant_fits_with_weak_mapping"] = int((hierarchy.loc[hierarchy["level"].eq("participant"), "template_strong_matches"] < hierarchy.loc[hierarchy["level"].eq("participant"), "K_estimated"]).sum())
        if checks["selected_k_values"] != [6]:
            issues.append(f"Hierarchy is not fixed at K=6: {checks['selected_k_values']}")
        expected_levels = {
            "global": 1,
            "group": int(manifest["group"].nunique()),
            "condition": int(manifest["condition"].nunique()),
            "participant": int(manifest["participant"].nunique()),
            "participant_condition": int(manifest[["participant", "condition"]].drop_duplicates().shape[0]),
        }
        if checks["hierarchy_level_counts"] != expected_levels:
            issues.append(f"Hierarchy level counts differ from manifest-derived counts: {expected_levels}")
        if not hierarchy["K_estimated"].eq(6).all():
            issues.append("At least one hierarchy node is not K=6.")
        if not hierarchy["real_observation_count"].eq(hierarchy["n_maps"]).all():
            issues.append("At least one fit includes observations other than its real GFP-peak maps.")
        child = hierarchy["level"].ne("global")
        checks["child_prior_modes"] = {str(k): int(v) for k, v in hierarchy.loc[child, "prior_mode"].value_counts(dropna=False).to_dict().items()}
        checks["child_prior_weights"] = sorted(float(x) for x in hierarchy.loc[child, "spm_prior_weight"].dropna().unique())
        checks["child_beta_0_values"] = {str(k): int(v) for k, v in hierarchy.loc[child, "prior_beta_0"].value_counts(dropna=False).to_dict().items()}
        if not hierarchy.loc[child, "prior_mode"].eq("spm_mix_prior_component_means").all():
            issues.append("A child fit did not use component-mean spm_mix_prior.")
        if not hierarchy.loc[child, "spm_prior_weight"].eq(1).all():
            issues.append("A child fit did not use spm_prior_weight=1.")
        if not hierarchy.loc[child, "prior_beta_0"].eq("1;1;1;1;1;1").all():
            issues.append("A child fit did not record beta_0=1 for all six states.")
        pc = hierarchy["level"].eq("participant_condition")
        if not hierarchy.loc[pc, "prior_source"].eq("participant_k6").all():
            issues.append("A participant-condition fit did not use its participant K=6 parent.")
        if hierarchy["spm_ms_commit"].nunique() != 1 or hierarchy["spm_ms_commit"].iat[0] != "0dbde1146f6777412dce19edfa4962cd45157fca":
            issues.append("SPM_MS provenance does not match the pinned upstream commit.")
        if hierarchy["spm_commit"].nunique() != 1 or hierarchy["spm_commit"].iat[0] != "04db1be762077630772ee660a470a0fd63d4ff43":
            issues.append("SPM provenance does not match the pinned installed SPM25 commit.")
        forbidden = {"spm_prior_pseudocount", "prior_pseudocount"} & set(hierarchy.columns)
        if forbidden:
            issues.append("Obsolete pseudocount columns remain: " + ", ".join(sorted(forbidden)))

    covariance_path = root / "hierarchical_state_covariance_metrics.csv"
    if covariance_path.exists():
        covariance = pd.read_csv(covariance_path)
        global_states = covariance.loc[covariance["level"].eq("global"), ["state_index", "template_label"]].sort_values("state_index")
        checks["global_template_labels"] = global_states["template_label"].astype(str).tolist()
        if tuple(checks["global_template_labels"]) != label_order:
            issues.append("Global state labels differ from the saved manual mapping.")
        ace = covariance.loc[
            covariance["level"].eq("participant") & covariance["template_label"].isin(["A", "C", "E"]),
            ["participant", "template_label", "template_match_abs_correlation"],
        ].pivot(index="participant", columns="template_label", values="template_match_abs_correlation")
        checks["participant_ace_reference_pass_count"] = int(ace.ge(0.5).all(axis=1).sum())
        checks["participant_ace_reference_total"] = int(len(ace))
        checks["participant_ace_below_0_5_by_state"] = {str(k): int(v) for k, v in ace.lt(0.5).sum().to_dict().items()}

    for prefix in ("participant_condition", "global"):
        summary_path = root / f"{prefix}_record_backfit_summary.csv"
        if summary_path.exists():
            frame = pd.read_csv(summary_path)
            checks[f"{prefix}_record_summary_rows"] = int(len(frame))
            checks[f"{prefix}_record_files"] = int(frame["file_path"].astype(str).str.lower().nunique())
        time_path = root / f"{prefix}_preictal_time_metrics.csv"
        if time_path.exists():
            frame = pd.read_csv(time_path, usecols=lambda c: c in {
                "file_path", "backfit_method", "template_label", "bin_duration_s", "bin_from_ictus", "minutes_to_ictus",
                "coverage", "mean_gfp", "occurrence_count", "mean_dwell_s", "mean_topographic_gmd"
            })
            checks[f"{prefix}_preictal_rows"] = int(len(frame))
            checks[f"{prefix}_preictal_files"] = int(frame["file_path"].astype(str).str.lower().nunique())
            checks[f"{prefix}_bin_durations_s"] = sorted(float(x) for x in frame["bin_duration_s"].dropna().unique())
            checks[f"{prefix}_nearest_bin_index_from_ictus"] = float(pd.to_numeric(frame["bin_from_ictus"], errors="coerce").min())
            checks[f"{prefix}_farthest_bin_index_from_ictus"] = float(pd.to_numeric(frame["bin_from_ictus"], errors="coerce").max())
            checks[f"{prefix}_earliest_minutes_to_ictus"] = float(pd.to_numeric(frame["minutes_to_ictus"], errors="coerce").min())
            expected = {"occurrence_count", "mean_dwell_s", "mean_topographic_gmd"}
            absent = sorted(expected - set(frame.columns))
            if absent:
                issues.append(f"{prefix} rolling output lacks: {', '.join(absent)}")
            durations = checks[f"{prefix}_bin_durations_s"]
            if not durations or min(durations) <= 0 or max(durations) > 10 or 10.0 not in durations:
                issues.append(f"{prefix} rolling bin durations fall outside the valid (0, 10] s range or lack full 10-s bins.")
            if checks[f"{prefix}_farthest_bin_index_from_ictus"] < 120 or checks[f"{prefix}_earliest_minutes_to_ictus"] > -19.9:
                issues.append(f"{prefix} rolling output does not reach the planned 20-minute lookback.")

    checks["validation_passed"] = not issues
    checks["issues"] = issues
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
