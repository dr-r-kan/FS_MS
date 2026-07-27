#!/usr/bin/env python3
"""Small LEMON proof-of-concept: TAS/condition regressions plus EC-vs-EO TANOVA."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from _analysis import _analyse_lemon_state_e_psychometry as tanova


ROOT = Path(__file__).resolve().parent
HIER_ROOT = ROOT / "outputs" / "hierarchical_microstates"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-metrics", type=Path, default=HIER_ROOT / "participant_condition_state_backfit_metrics.csv")
    parser.add_argument("--record-summary", type=Path, default=HIER_ROOT / "participant_condition_record_backfit_summary.csv")
    parser.add_argument("--hierarchical-root", type=Path, default=HIER_ROOT)
    parser.add_argument(
        "--tas-csv",
        type=Path,
        default=Path.home() / "EEG" / "LEMON" / "Behavioural_Data_MPILMBB_LEMON" / "Emotion_and_Personality_Test_Battery_LEMON" / "TAS.csv",
    )
    parser.add_argument("--name-match-csv", type=Path, default=Path.home() / "EEG" / "LEMON" / "name_match.csv")
    parser.add_argument("--output-dir", type=Path, default=HIER_ROOT / "lemon_poc_analysis")
    parser.add_argument("--tanova-permutations", type=int, default=5000)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def load_tas(tas_csv: Path, name_match_csv: Path) -> pd.DataFrame:
    tas = pd.read_csv(tas_csv)
    tas = tas.rename(columns={tas.columns[0]: "INDI_ID"})
    tas["TAS_OverallScore"] = pd.to_numeric(tas["TAS_OverallScore"], errors="coerce")

    names = pd.read_csv(name_match_csv)
    names = names.rename(columns={"Initial_ID": "participant"})
    return names[["participant", "INDI_ID"]].merge(
        tas[["INDI_ID", "TAS_OverallScore"]],
        on="INDI_ID",
        how="inner",
    )


def hard_ec_eo_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[
        df["condition"].isin(["EC", "EO"])
        & df["backfit_method"].eq("hard")
        & pd.to_numeric(df["backfit_available"], errors="coerce").eq(1)
    ].copy()


def ols_rows(df: pd.DataFrame, y_col: str, metric: str, state_label: str = "") -> list[dict[str, object]]:
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(float)
    tas_score = pd.to_numeric(df["TAS_OverallScore"], errors="coerce").to_numpy(float)
    condition_eo = df["condition"].eq("EO").astype(float).to_numpy()
    ok = np.isfinite(y) & np.isfinite(tas_score) & np.isfinite(condition_eo)
    y = y[ok]
    X = np.column_stack([np.ones(ok.sum()), tas_score[ok], condition_eo[ok]])
    if y.size <= X.shape[1] or np.linalg.matrix_rank(X) < X.shape[1]:
        return [skip_row(metric, state_label, f"Not enough non-degenerate rows for {y_col}.")]

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ beta
    dof = int(y.size - X.shape[1])
    sigma2 = float(residual @ residual / dof)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t_stat = beta / se
    p_values = 2 * stats.t.sf(np.abs(t_stat), dof)
    ss_total = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - (residual @ residual) / ss_total) if ss_total > 0 else np.nan

    rows = []
    for term, index in [("TAS_OverallScore", 1), ("condition_EO_vs_EC", 2)]:
        rows.append(
            {
                "metric": metric,
                "state_label": state_label,
                "term": term,
                "n": int(y.size),
                "beta": float(beta[index]),
                "se": float(se[index]),
                "t": float(t_stat[index]),
                "p": float(p_values[index]),
                "model_r2": r2,
                "status": "ok",
                "note": "OLS: metric ~ TAS_OverallScore + condition_EO_vs_EC",
            }
        )
    return rows


def skip_row(metric: str, state_label: str, note: str) -> dict[str, object]:
    return {
        "metric": metric,
        "state_label": state_label,
        "term": "",
        "n": 0,
        "beta": np.nan,
        "se": np.nan,
        "t": np.nan,
        "p": np.nan,
        "model_r2": np.nan,
        "status": "skipped",
        "note": note,
    }


def regression_table(state_metrics: Path, record_summary: Path, tas_scores: pd.DataFrame) -> pd.DataFrame:
    states = hard_ec_eo_rows(state_metrics).merge(tas_scores, on="participant", how="inner")
    records = hard_ec_eo_rows(record_summary).merge(tas_scores, on="participant", how="inner")

    rows: list[dict[str, object]] = []
    for label in sorted(states["template_label"].dropna().unique()):
        state_rows = states[states["template_label"].eq(label)]
        rows.extend(ols_rows(state_rows, "occupancy", "coverage", str(label)))
        rows.extend(ols_rows(state_rows, "mean_gfp", "gfp", str(label)))
    rows.extend(ols_rows(records, "record_shannon_entropy_bits", "entropy", "record"))
    return pd.DataFrame(rows)


def tanova_table(hierarchical_root: Path, permutations: int) -> pd.DataFrame:
    tanova.TANOVA_PERMUTATIONS = int(permutations)
    return pd.DataFrame(tanova.hierarchical_condition_tanova_rows(hierarchical_root), columns=tanova.TANOVA_FIELDS)


def main() -> None:
    args = parse_args()
    if args.self_test:
        tiny = pd.DataFrame(
            {
                "condition": ["EC", "EO", "EC", "EO"],
                "TAS_OverallScore": [40, 40, 60, 60],
                "metric": [1.0, 2.0, 2.0, 3.0],
            }
        )
        assert math.isclose(ols_rows(tiny, "metric", "demo")[0]["beta"], 0.05, rel_tol=1e-9)
        print("self-test passed")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tas_scores = load_tas(args.tas_csv, args.name_match_csv)

    regressions = regression_table(args.state_metrics, args.record_summary, tas_scores)
    regressions.to_csv(args.output_dir / "tas_condition_regressions.csv", index=False)

    tanova_results = tanova_table(args.hierarchical_root, args.tanova_permutations)
    tanova_results.to_csv(args.output_dir / "ec_eo_tanova_by_microstate.csv", index=False)

    (args.output_dir / "README.txt").write_text(
        "LEMON proof-of-concept analysis\n"
        "tas_condition_regressions.csv: OLS rows for coverage and GFP per microstate, plus record entropy. "
        "Predictors are TAS_OverallScore and condition_EO_vs_EC.\n"
        "ec_eo_tanova_by_microstate.csv: paired EC-vs-EO TANOVA by microstate/template label.\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
