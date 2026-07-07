from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["mean_gfp", "coverage", "dwell_time_s", "occurrence_rate_hz"]


def norm_path(value: str) -> str:
    try:
        return str(Path(str(value)).resolve()).lower()
    except OSError:
        return str(value).replace("/", "\\").lower()


def default_manifest(root: Path) -> Path:
    for name in ("vg_microstate_manifest_mat.csv", "vg_microstate_manifest_set.csv", "vg_microstate_manifest_fif.csv"):
        path = root / name
        if path.exists():
            return path
    return root / "vg_microstate_manifest_mat.csv"


def add_manifest_metadata(df: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    manifest = manifest.copy()
    manifest["_path"] = manifest["file_path"].map(norm_path)
    df = df.copy()
    df["_path"] = df["file_path"].map(norm_path)
    extra = [c for c in ["file_path", "hypo_hyper", "seizure_type", "event_classification", "event_number", "segment", "source_row"] if c in manifest.columns]
    return df.merge(manifest[["_path", *extra]].drop_duplicates("_path"), on="_path", how="left", suffixes=("", "_manifest"))


def prepared_state_metrics(results_dir: Path, manifest_csv: Path) -> pd.DataFrame:
    states = pd.read_csv(results_dir / "participant_condition_state_backfit_metrics.csv")
    manifest = pd.read_csv(manifest_csv)
    df = add_manifest_metadata(states, manifest)
    for col in ("occupancy", "duration_s", "occurrence_count", "mean_gfp", "occurrence_rate_hz"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["coverage"] = df["occupancy"]
    df["dwell_time_s"] = np.where(df["occurrence_count"] > 0, df["coverage"] * df["duration_s"] / df["occurrence_count"], np.nan)
    return df


def participant_level(df: pd.DataFrame, method: str) -> pd.DataFrame:
    keep = df["backfit_method"].eq(method) & pd.to_numeric(df["backfit_available"], errors="coerce").eq(1)
    cols = ["participant", "group", "condition", "hypo_hyper", "template_label", "state_index", "backfit_method"]
    return df.loc[keep].groupby(cols, dropna=False, as_index=False)[METRICS].mean()


def write_summary(unit: pd.DataFrame, out_dir: Path) -> None:
    long = unit.melt(
        id_vars=["participant", "group", "condition", "hypo_hyper", "template_label", "state_index", "backfit_method"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    )
    summary = (
        long.dropna(subset=["value"])
        .groupby(["metric", "backfit_method", "template_label", "state_index", "group", "condition"], dropna=False)["value"]
        .agg(n="count", mean="mean", sd="std")
        .reset_index()
    )
    summary["sem"] = summary["sd"] / np.sqrt(summary["n"])
    summary.to_csv(out_dir / "state_metric_summary.csv", index=False)


def dummy(values: pd.Series, name: str) -> tuple[np.ndarray, list[str]]:
    values = values.astype(str)
    levels = sorted(v for v in values.dropna().unique() if v and v.lower() != "nan")
    if len(levels) <= 1:
        return np.empty((len(values), 0)), []
    cols = [(values == level).to_numpy(float) for level in levels[1:]]
    return np.column_stack(cols), [f"{name}={level}" for level in levels[1:]]


def design(sub: pd.DataFrame, group: bool, condition: bool, interaction: bool, semiology: bool = False) -> tuple[np.ndarray, list[str]]:
    parts = [np.ones((len(sub), 1))]
    names = ["intercept"]
    g, gn = dummy(sub["group"], "group") if group else (np.empty((len(sub), 0)), [])
    c, cn = dummy(sub["condition"], "condition") if condition else (np.empty((len(sub), 0)), [])
    h, hn = dummy(sub["hypo_hyper"], "hypo_hyper") if semiology else (np.empty((len(sub), 0)), [])
    for mat, labels in ((g, gn), (c, cn), (h, hn)):
        if mat.size:
            parts.append(mat)
            names.extend(labels)
    if interaction and g.size and c.size:
        inter = np.column_stack([g[:, gi] * c[:, ci] for gi in range(g.shape[1]) for ci in range(c.shape[1])])
        parts.append(inter)
        names.extend([f"{a}:{b}" for a in gn for b in cn])
    return np.column_stack(parts), names


def ols_fit(y: np.ndarray, x: np.ndarray) -> dict:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    rank = np.linalg.matrix_rank(x)
    return {"rss": float(resid @ resid), "df": int(len(y) - rank), "rank": int(rank)}


def nested_f(reduced: dict, full: dict) -> tuple[float, float, int, int]:
    df_num = reduced["df"] - full["df"]
    df_den = full["df"]
    if df_num <= 0 or df_den <= 0 or full["rss"] <= 0:
        return np.nan, np.nan, df_num, df_den
    f_value = ((reduced["rss"] - full["rss"]) / df_num) / (full["rss"] / df_den)
    try:
        from scipy import stats

        p_value = float(stats.f.sf(f_value, df_num, df_den))
    except Exception:
        p_value = np.nan
    return float(f_value), p_value, df_num, df_den


def test_row(sub: pd.DataFrame, term: str, reduced_args: dict, full_args: dict) -> dict:
    y = sub["value"].to_numpy(float)
    xr, _ = design(sub, **reduced_args)
    xf, names = design(sub, **full_args)
    f_value, p_value, df_num, df_den = nested_f(ols_fit(y, xr), ols_fit(y, xf))
    return {"term": term, "df_num": df_num, "df_den": df_den, "f": f_value, "p_value": p_value, "model_terms": ";".join(names), "note": ""}


def ols_factor_tests(unit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    long = unit.melt(
        id_vars=["participant", "group", "condition", "template_label", "state_index"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value", "group", "condition"])
    for (metric, label, state), sub in long.groupby(["metric", "template_label", "state_index"], dropna=False):
        prefix = {"metric": metric, "template_label": label, "state_index": state}
        if sub["group"].nunique() < 2 or sub["condition"].nunique() < 2 or len(sub) < 8:
            rows.append({**prefix, "term": "", "note": "insufficient group/condition levels"})
            continue
        try:
            tests = [
                test_row(sub, "group", {"group": False, "condition": True, "interaction": False}, {"group": True, "condition": True, "interaction": False}),
                test_row(sub, "condition", {"group": True, "condition": False, "interaction": False}, {"group": True, "condition": True, "interaction": False}),
                test_row(sub, "group:condition", {"group": True, "condition": True, "interaction": False}, {"group": True, "condition": True, "interaction": True}),
            ]
            rows.extend({**prefix, **row} for row in tests)
        except Exception as exc:
            rows.append({**prefix, "term": "", "note": str(exc)})
    return pd.DataFrame(rows)


def semiology_model_comparison(unit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    long = unit.melt(
        id_vars=["participant", "group", "condition", "hypo_hyper", "template_label", "state_index"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    )
    long = long[long["hypo_hyper"].isin(["hypo", "hyper"])].dropna(subset=["value"])
    for (metric, label, state), sub in long.groupby(["metric", "template_label", "state_index"], dropna=False):
        if sub["hypo_hyper"].nunique() < 2 or len(sub) < 8:
            rows.append({"metric": metric, "template_label": label, "state_index": state, "note": "insufficient hypo/hyper rows"})
            continue
        try:
            base_args = {"group": sub["group"].nunique() > 1, "condition": sub["condition"].nunique() > 1, "interaction": sub["group"].nunique() > 1 and sub["condition"].nunique() > 1}
            xb, base_names = design(sub, **base_args)
            xp, plus_names = design(sub, **base_args, semiology=True)
            y = sub["value"].to_numpy(float)
            base_fit = ols_fit(y, xb)
            plus_fit = ols_fit(y, xp)
            f_value, p_value, df_num, df_den = nested_f(base_fit, plus_fit)
            rows.append(
                {
                    "metric": metric,
                    "template_label": label,
                    "state_index": state,
                    "n": len(sub),
                    "base_terms": ";".join(base_names),
                    "semiology_terms": ";".join(plus_names),
                    "df_num": df_num,
                    "df_den": df_den,
                    "f": f_value,
                    "p_value": p_value,
                    "rss_base": base_fit["rss"],
                    "rss_semiology": plus_fit["rss"],
                    "note": "",
                }
            )
        except Exception as exc:
            rows.append({"metric": metric, "template_label": label, "state_index": state, "note": str(exc)})
    return pd.DataFrame(rows)


def analyse(results_dir: Path, manifest_csv: Path, out_dir: Path, method: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = prepared_state_metrics(results_dir, manifest_csv)
    unit = participant_level(state, method)
    unit.to_csv(out_dir / "participant_level_state_metrics.csv", index=False)
    write_summary(unit, out_dir)
    ols_factor_tests(unit).to_csv(out_dir / "metric_group_condition_tests.csv", index=False)
    semiology_model_comparison(unit).to_csv(out_dir / "metric_semiology_model_comparison.csv", index=False)
    # ponytail: OLS screen; use mixed models/permutation if this becomes confirmatory repeated-measures inference.


def self_check() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        results = root / "hierarchical"
        results.mkdir()
        manifest = root / "vg_microstate_manifest_mat.csv"
        files = [str(root / f"f{i}.mat") for i in range(8)]
        pd.DataFrame(
            {
                "file_path": files,
                "hypo_hyper": ["hypo", "hyper"] * 4,
                "segment": ["preict", "postict"] * 4,
            }
        ).to_csv(manifest, index=False)
        pd.DataFrame(
            {
                "participant": [f"P{i//2}" for i in range(8)],
                "condition": ["preict", "postict"] * 4,
                "group": ["FS", "ES"] * 4,
                "file_path": files,
                "backfit_method": ["hard"] * 8,
                "backfit_available": [1] * 8,
                "state_index": [1] * 8,
                "template_label": ["A"] * 8,
                "occupancy": np.linspace(0.2, 0.6, 8),
                "duration_s": [10] * 8,
                "occurrence_count": [2] * 8,
                "mean_gfp": np.linspace(1, 2, 8),
                "occurrence_rate_hz": np.linspace(0.1, 0.2, 8),
            }
        ).to_csv(results / "participant_condition_state_backfit_metrics.csv", index=False)
        analyse(results, manifest, root / "analysis", "hard")
        assert (root / "analysis" / "state_metric_summary.csv").exists()
        assert (root / "analysis" / "metric_semiology_model_comparison.csv").exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise VG microstate scalar metrics and screen hypo/hyper model improvement.")
    parser.add_argument("--pipeline-dir", type=Path, default=default_manifest(Path.cwd()).parent)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--method", default="hard")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        print("self-check passed")
        return
    pipeline_dir = args.pipeline_dir
    results_dir = args.results_dir or pipeline_dir / "hierarchical"
    manifest = args.manifest or default_manifest(pipeline_dir)
    out_dir = args.out_dir or pipeline_dir / "scalar_analysis"
    analyse(results_dir, manifest, out_dir, args.method)
    print(f"Scalar analysis: {out_dir}")


if __name__ == "__main__":
    main()
