"""Create a compact, identifier-free statistical audit for report interpretation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def clean(record: dict) -> dict:
    out = {}
    for key, value in record.items():
        if isinstance(value, (np.integer,)):
            value = int(value)
        elif isinstance(value, (np.floating,)):
            value = None if not np.isfinite(value) else float(value)
        out[key] = value
    return out


def top_record(frame: pd.DataFrame, p_column: str, columns: list[str]) -> dict | None:
    if frame.empty or p_column not in frame:
        return None
    valid = frame.dropna(subset=[p_column]).sort_values(p_column)
    return clean(valid.iloc[0][[c for c in columns if c in valid]].to_dict()) if len(valid) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    read = lambda name: pd.read_csv(args.data_dir / name) if (args.data_dir / name).exists() else pd.DataFrame()
    scalar = read("rolling_scalar_gee_terms.csv")
    cardiac = read("rolling_cardiac_phase_gee_terms.csv")
    tanova = read("tanova_results.csv")
    bhatt = read("bhattacharyya_tanova_results.csv")
    transition = read("transition_group_condition_edge_model_terms.csv")
    primary = scalar[(scalar.get("analysis_role", pd.Series(index=scalar.index, dtype=str)).eq("co_primary")) &
                     scalar.get("scheme", pd.Series(index=scalar.index, dtype=str)).eq("20min_60s_step10s") &
                     scalar.get("backfit_method", pd.Series(index=scalar.index, dtype=str)).eq("hard") &
                     scalar.get("stratum", pd.Series(index=scalar.index, dtype=str)).astype(str).str.contains("functional", na=False)]
    sensitivity = scalar[scalar.get("analysis_role", pd.Series(index=scalar.index, dtype=str)).ne("co_primary")]
    cardiac_primary = cardiac[(cardiac.get("systolic_window_start_ms", pd.Series(index=cardiac.index, dtype=float)).eq(180)) &
                              (cardiac.get("systolic_window_end_ms", pd.Series(index=cardiac.index, dtype=float)).eq(500)) &
                              cardiac.get("stratum", pd.Series(index=cardiac.index, dtype=str)).astype(str).str.contains("functional", na=False) &
                              cardiac.get("term", pd.Series(index=cardiac.index, dtype=str)).astype(str).str.contains(":", na=False)]
    result = {
        "scalar_status_counts": {str(k): int(v) for k, v in scalar.get("status", pd.Series(dtype=str)).fillna("missing").value_counts().to_dict().items()},
        "scalar_significant_fdr_all": int(scalar.get("p_fdr", pd.Series(dtype=float)).lt(.05).sum()),
        "scalar_significant_fdr_primary_functional": int(primary.get("p_fdr", pd.Series(dtype=float)).lt(.05).sum()),
        "best_primary_functional_scalar": top_record(primary, "p_fdr", ["source", "template_label", "analysis_role", "stratum", "metric", "term", "estimate", "ci_low", "ci_high", "p_value", "p_fdr", "n_participants", "status"]),
        "best_sensitivity_or_exploratory_scalar": top_record(sensitivity, "p_fdr", ["source", "template_label", "analysis_role", "backfit_method", "stratum", "scheme", "metric", "term", "estimate", "ci_low", "ci_high", "p_value", "p_fdr", "n_participants", "status"]),
        "cardiac_status_counts": {str(k): int(v) for k, v in cardiac.get("status", pd.Series(dtype=str)).fillna("missing").value_counts().to_dict().items()},
        "cardiac_significant_holm_primary_functional": int(cardiac_primary.get("p_holm", pd.Series(dtype=float)).lt(.05).sum()),
        "best_primary_functional_cardiac_interaction": top_record(cardiac_primary, "p_holm", ["template_label", "stratum", "term", "rate_ratio", "ci_low", "ci_high", "p_value", "p_holm", "n_participants", "status"]),
        "tanova_significant_max_state": int(tanova.get("p_max_state", pd.Series(dtype=float)).lt(.05).sum()),
        "tanova_significant_records": [clean(record) for record in tanova.loc[
            tanova.get("p_max_state", pd.Series(index=tanova.index, dtype=float)).lt(.05),
            [c for c in ["effect", "state", "statistic", "p_uncorrected", "p_max_state", "n_units"] if c in tanova],
        ].sort_values("p_max_state").to_dict("records")],
        "best_tanova": top_record(tanova, "p_max_state", ["effect", "state", "statistic", "p_uncorrected", "p_max_state", "n_units"]),
        "bhattacharyya_significant_bayesian_fdr": int(bhatt.get("bayesian_fdr_significant", pd.Series(dtype=float)).eq(1).sum()),
        "bhattacharyya_significant_C_or_E": int((bhatt.get("bayesian_fdr_significant", pd.Series(dtype=float)).eq(1) & bhatt.get("template_label", pd.Series(index=bhatt.index, dtype=str)).isin(["C", "E"])).sum()),
        "bhattacharyya_significant_C_or_E_by_effect": {
            str(key): int(value) for key, value in bhatt.loc[
                bhatt.get("bayesian_fdr_significant", pd.Series(index=bhatt.index, dtype=float)).eq(1)
                & bhatt.get("template_label", pd.Series(index=bhatt.index, dtype=str)).isin(["C", "E"])
            ].groupby(["effect", "template_label"]).size().to_dict().items()
        },
        "bhattacharyya_significant_C_or_E_records": [clean(record) for record in bhatt.loc[
            bhatt.get("bayesian_fdr_significant", pd.Series(index=bhatt.index, dtype=float)).eq(1)
            & bhatt.get("template_label", pd.Series(index=bhatt.index, dtype=str)).isin(["C", "E"]),
            [c for c in ["effect", "template_label", "channel_label", "bhattacharyya_distance", "permutation_p", "posterior_prob_above_null95", "posterior_null_probability", "n_participants"] if c in bhatt],
        ].sort_values("posterior_null_probability").to_dict("records")],
        "best_bhattacharyya_C_or_E": top_record(bhatt[bhatt.get("template_label", pd.Series(index=bhatt.index, dtype=str)).isin(["C", "E"])], "posterior_null_probability", ["effect", "template_label", "channel_label", "bhattacharyya_distance", "permutation_p", "posterior_prob_above_null95", "posterior_null_probability", "bayesian_fdr_significant", "n_participants"]),
        "transition_significant_fdr": int(transition.get("p_fdr", pd.Series(dtype=float)).lt(.05).sum()),
        "best_transition_term": top_record(transition, "p_fdr", ["edge", "term", "estimate", "p_value", "p_fdr", "status"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
