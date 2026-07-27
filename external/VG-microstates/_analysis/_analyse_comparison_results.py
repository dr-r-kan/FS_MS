#!/usr/bin/env python3
"""Python analysis and plotting for simulation comparison results.

"""

from __future__ import annotations

import argparse
import io as text_io
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import io, stats
from scipy.interpolate import griddata

import h5py

try:
    import mne
except Exception:  # pragma: no cover - optional outside the repo venv
    mne = None


OUTCOMES = [
    "K_correct",
    "f1_score",
    "sensitivity",
    "precision",
    "mean_recovery_matched",
    "K_sq_error",
]
OUTCOME_LABELS = {
    "K_correct": "K_true selection accuracy",
    "f1_score": "F1 score",
    "sensitivity": "Sensitivity",
    "precision": "Precision",
    "mean_recovery_matched": "Mean matched correlation",
    "K_abs_error": "Absolute K error",
    "K_sq_error": "Squared K error",
    "runtime_s": "Runtime (s)",
}
OUTCOMES_FULL = OUTCOMES[:4] + ["runtime_s"] + OUTCOMES[4:]
COVARIANCE_SPECS = {
    "selected_spm_cov_trace_mean": "Mean covariance trace",
    "selected_spm_cov_trace_median": "Median covariance trace",
    "selected_spm_cov_logdet_mean": "Mean covariance logdet",
}

OVERLAP_METHOD_CRITERIA = [
    ("koenig kmeans", "silhouette"),
    ("spm vb", "free energy elbow"),
]
BEST_MODEL_CRITERIA = [
    ("koenig kmeans", "silhouette", "K-Means / silhouette"),
    ("spm vb", "free energy elbow", "VB-GMM / free-energy elbow"),
]


@dataclass
class TemplateTopographies:
    labels: list[str]
    channel_labels: list[str]
    maps: np.ndarray
    xy: np.ndarray
    info: object | None = None


@dataclass
class BackfitDiagnostic:
    labels: list[str]
    counts: np.ndarray
    accuracy: float = np.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze simulation comparison_results.csv with Python plots and confusion summaries."
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=Path("outputs/simulations/results"),
        help="Directory containing comparison_results.csv.",
    )
    parser.add_argument("--results-csv", type=Path, help="Explicit comparison_results.csv path.")
    parser.add_argument("--output-dir", type=Path, help="Plot/statistics output directory.")
    parser.add_argument("--template-file", type=Path, default=Path("MetaMaps_2023_06.set"))
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--n-folds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-confusion-pair-plots",
        type=int,
        default=0,
        help="Limit method-pair confusion plots. 0 means all.",
    )
    parser.add_argument("--skip-confusion-outputs", action="store_true", help="Skip HDF5 backfit confusion outputs.")
    parser.add_argument("--skip-mixed-models", action="store_true", help="Skip optional statsmodels MixedLM fits.")
    parser.add_argument(
        "--skip-microstate-map-plot",
        action="store_true",
        help="Skip the Python-rendered microstate recovery topography plot.",
    )
    parser.add_argument(
        "--skip-confusion-pair-plots",
        action="store_true",
        help="Skip Python-rendered template topographies around pairwise confusion plots.",
    )
    parser.add_argument("--self-test", action="store_true", help="Run lightweight internal checks.")
    return parser.parse_args()


def analyze_comparison_results(
    results_dir: str | Path = Path("outputs/simulations/results"),
    *,
    results_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    template_file: str | Path = Path("MetaMaps_2023_06.set"),
    n_boot: int = 200,
    n_folds: int = 2,
    seed: int = 0,
    max_confusion_pair_plots: int = 0,
    run_mixed_models: bool = True,
    run_confusion_outputs: bool = True,
    run_microstate_map_plot: bool = True,
    run_confusion_pair_plots: bool = True,
) -> dict[str, Path | pd.DataFrame]:
    results_dir = Path(results_dir)
    csv_file = Path(results_csv) if results_csv else results_dir / "comparison_results.csv"
    if not csv_file.is_file():
        raise FileNotFoundError(f"Results file not found: {csv_file}")

    results_dir = csv_file.parent
    plots_dir = Path(output_dir) if output_dir else results_dir.parent / "analysis_plots_python"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    table = prepare_results_table(pd.read_csv(csv_file))
    stats_file = plots_dir / "bootstrap_statistics.txt"

    template = load_template_topographies(template_file) if (run_microstate_map_plot or run_confusion_pair_plots) else None

    with stats_file.open("w", encoding="utf-8") as handle:
        write_header(handle, csv_file, table, n_boot, n_folds)
        design = summarize_design(table, handle)

        method_table = subset_to_shared_criteria(table, design["shared_criteria"])
        criterion_table = subset_to_full_support_methods(table, design["full_support_methods"])

        method_summary = analyze_factor(handle, method_table, "method", OUTCOMES_FULL, n_boot, rng)
        criterion_summary = analyze_factor(handle, criterion_table, "criterion_clean", OUTCOMES_FULL, n_boot, rng)
        snr_summary = analyze_snr(handle, table, OUTCOMES_FULL, n_boot, rng)
        overlap_summary = analyze_overlap(handle, table, OUTCOMES_FULL, n_boot, rng)
        interaction = analyze_interaction(handle, table, OUTCOMES_FULL)
        cv = analyze_cross_validation(handle, table, OUTCOMES_FULL, n_folds, rng)
        if run_mixed_models:
            analyze_mixed_models(handle, table, OUTCOMES_FULL)
        else:
            handle.write("Mixed models skipped by request.\n\n")
        vb_covariance = analyze_vb_covariance(handle, table, n_boot, rng)

    make_standard_plots(
        table,
        method_table,
        criterion_table,
        method_summary,
        criterion_summary,
        snr_summary,
        overlap_summary,
        interaction,
        cv,
        vb_covariance,
        plots_dir,
        n_boot,
        rng,
    )
    if run_microstate_map_plot:
        create_simulated_microstate_method_comparison(table, plots_dir, template, seed=seed)
    if run_confusion_outputs:
        make_confusion_outputs(
            table,
            results_dir,
            plots_dir,
            template,
            max_pair_plots=max_confusion_pair_plots,
            run_pair_plots=run_confusion_pair_plots,
        )

    return {"table": table, "plots_dir": plots_dir, "stats_file": stats_file}


def prepare_results_table(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    if "montage_type" not in table:
        table["montage_type"] = "full"
    if "n_leads" not in table:
        table["n_leads"] = 71
    if "method" not in table or "criterion" not in table:
        raise ValueError("comparison_results.csv must contain method and criterion columns")

    table["method"] = table["method"].map(canonicalize_method)
    table = table[table["method"].isin(["koenig kmeans", "spm vb"])].copy()
    if table.empty:
        raise ValueError("No supported method rows found")

    table["criterion_clean"] = table["criterion"].map(canonicalize_criterion)
    numeric_candidates = set(OUTCOMES_FULL) | {
        "K_true",
        "K_estimated",
        "K_gap",
        "SNR_dB",
        "n_leads",
        *COVARIANCE_SPECS,
    }
    for col in sorted(numeric_candidates & set(table.columns)):
        table[col] = pd.to_numeric(table[col], errors="coerce")

    if {"K_estimated", "K_true"}.issubset(table.columns):
        table["K_error"] = table["K_estimated"] - table["K_true"]
        if "K_gap" not in table:
            table["K_gap"] = table["K_true"] - table["K_estimated"]
    else:
        table["K_error"] = np.nan
        table["K_gap"] = np.nan
    table["K_abs_error"] = table["K_error"].abs()
    table["K_sq_error"] = table["K_error"].pow(2)

    if "mean_recovery_matched" not in table:
        recovery_cols = [c for c in table.columns if c.startswith("recovery_")]
        if recovery_cols:
            table["mean_recovery_matched"] = table[recovery_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        else:
            table["mean_recovery_matched"] = np.nan
    else:
        table["mean_recovery_matched"] = pd.to_numeric(table["mean_recovery_matched"], errors="coerce")

    if "subject" not in table:
        if "eeg_idx" in table:
            table["subject"] = "eeg_" + table["eeg_idx"].astype(str)
        elif "fit_id" in table:
            table["subject"] = "fit_" + table["fit_id"].astype(str)
        else:
            table["subject"] = [f"row_{i}" for i in range(len(table))]
    table["method_criterion_combo"] = table["method"] + " | " + table["criterion_clean"]
    return table.reset_index(drop=True)


def canonicalize_method(value: object) -> str:
    text = clean_token(value)
    if "spm vb" in text:
        return "spm vb"
    if "kmeans koenig" in text or "koenig kmeans" in text:
        return "koenig kmeans"
    if "standard kmeans" in text or "kmeans standard" in text:
        return "standard kmeans"
    return text


def canonicalize_criterion(value: object) -> str:
    text = clean_token(value)
    if text == "elbow":
        return "free energy elbow"
    if text in {"gfp", "global field power", "global explained variance"}:
        return "gev"
    if "elbow sil" in text or "free energy elbow sil" in text:
        return "elbow sil combined"
    aliases = {
        "free energy elbow only": "free energy elbow",
        "silhouette only": "silhouette",
        "covariance raw": "covariance",
        "covariance min": "covariance",
        "calinski harabasz": "calinski harabasz score",
        "ch": "calinski harabasz score",
    }
    return aliases.get(text, re.sub(r"(\b\w+\b)(\s+\1)+", r"\1", text))


def clean_token(value: object) -> str:
    text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
    text = text.lower().strip().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text)


def display_method(value: object) -> str:
    return {"spm vb": "VB-GMM", "koenig kmeans": "K-Means"}.get(canonicalize_method(value), str(value))


def selected_best_model_rows(table: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for method, criterion, label in BEST_MODEL_CRITERIA:
        sub = table[(table["method"] == method) & (table["criterion_clean"] == criterion)].copy()
        if not sub.empty:
            sub["best_model_label"] = label
            frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else table.iloc[0:0].copy()


def write_header(handle, csv_file: Path, table: pd.DataFrame, n_boot: int, n_folds: int) -> None:
    handle.write("BOOTSTRAPPED STATISTICAL ANALYSIS (Python)\n")
    handle.write(f"Results file: {csv_file}\n")
    handle.write(f"Observations: {len(table)}\n")
    handle.write(f"Bootstrap iterations: {n_boot}\n")
    handle.write(f"CV folds: {n_folds}\n\n")


def ordered_unique(values: Iterable[object]) -> list:
    return [v for v in pd.Series(values).dropna().drop_duplicates().tolist()]


def summarize_design(table: pd.DataFrame, handle) -> dict[str, list[str]]:
    methods = ordered_unique(table["method"])
    criteria = ordered_unique(table["criterion_clean"])
    observed = pd.crosstab(table["method"], table["criterion_clean"]).reindex(index=methods, columns=criteria, fill_value=0)
    shared_criteria = [c for c in criteria if (observed[c] > 0).all()]
    full_support_methods = [m for m in methods if (observed.loc[m] > 0).all()]
    handle.write("Observed method-criterion design:\n")
    for method in methods:
        seen = [c for c in criteria if observed.loc[method, c] > 0]
        handle.write(f"  {method}: {', '.join(seen) if seen else '<none>'}\n")
    handle.write(f"Shared criteria across all methods: {', '.join(shared_criteria) or '<none>'}\n")
    handle.write(f"Methods with full criterion support: {', '.join(full_support_methods) or '<none>'}\n\n")
    return {"shared_criteria": shared_criteria, "full_support_methods": full_support_methods}


def subset_to_shared_criteria(table: pd.DataFrame, shared_criteria: list[str]) -> pd.DataFrame:
    return table[table["criterion_clean"].isin(shared_criteria)].copy() if shared_criteria else table.copy()


def subset_to_full_support_methods(table: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    return table[table["method"].isin(methods)].copy() if methods else table.copy()


def analyze_factor(handle, table: pd.DataFrame, factor: str, outcomes: list[str], n_boot: int, rng) -> pd.DataFrame:
    rows = []
    levels = ordered_unique(table[factor]) if factor in table else []
    handle.write(f"Factor: {factor}\nLevels: {', '.join(map(str, levels))}\n\n")
    for outcome in outcomes:
        if outcome not in table:
            continue
        handle.write(f"--- {outcome} ---\n")
        groups = []
        for level in levels:
            values = finite_values(table.loc[table[factor] == level, outcome])
            mean, lo, hi = mean_ci(values, n_boot, rng)
            rows.append({"factor": factor, "level": level, "outcome": outcome, "mean": mean, "ci_low": lo, "ci_high": hi, "n": len(values)})
            groups.append(values)
            handle.write(f"  {level}: {mean:.4f} [{lo:.4f}, {hi:.4f}] n={len(values)}\n")
        valid_groups = [g for g in groups if len(g) > 0]
        if len(valid_groups) > 1:
            try:
                h_stat, p_value = stats.kruskal(*valid_groups)
                handle.write(f"Kruskal-Wallis: H={h_stat:.4f} p={p_value:.4g}\n")
            except Exception as exc:
                handle.write(f"Kruskal-Wallis failed: {exc}\n")
        handle.write("\n")
    return pd.DataFrame(rows)


def analyze_snr(handle, table: pd.DataFrame, outcomes: list[str], n_boot: int, rng) -> pd.DataFrame:
    rows = []
    if "SNR_dB" not in table:
        return pd.DataFrame(rows)
    snrs = finite_unique(table["SNR_dB"])
    handle.write(f"SNR levels: {snrs}\n")
    for outcome in outcomes:
        if outcome not in table:
            continue
        handle.write(f"--- {outcome} by SNR ---\n")
        for snr in snrs:
            values = finite_values(table.loc[table["SNR_dB"] == snr, outcome])
            mean, lo, hi = mean_ci(values, n_boot, rng)
            rows.append({"SNR_dB": snr, "outcome": outcome, "mean": mean, "ci_low": lo, "ci_high": hi, "n": len(values)})
            handle.write(f"  SNR {snr:+g} dB: {mean:.4f} [{lo:.4f}, {hi:.4f}] n={len(values)}\n")
        handle.write("\n")
    return pd.DataFrame(rows)


def analyze_overlap(handle, table: pd.DataFrame, outcomes: list[str], n_boot: int, rng) -> pd.DataFrame:
    rows = []
    if "overlap_prob" not in table:
        return pd.DataFrame(rows)
    handle.write("Overlap effects: K-Means/silhouette vs VB-GMM/free-energy elbow\n")
    for method, criterion in OVERLAP_METHOD_CRITERIA:
        group = table[(table["method"] == method) & (table["criterion_clean"] == criterion)].copy()
        if group.empty:
            handle.write(f"  {display_method(method)} / {criterion}: no rows\n")
            continue
        levels = finite_unique(group["overlap_prob"])
        handle.write(f"  {display_method(method)} / {criterion}\n")
        for outcome in outcomes:
            if outcome not in group:
                continue
            means = []
            handle.write(f"    {outcome}\n")
            for level in levels:
                values = finite_values(group.loc[group["overlap_prob"] == level, outcome])
                mean, lo, hi = mean_ci(values, n_boot, rng)
                means.append((level, mean))
                rows.append(
                    {
                        "method": method,
                        "criterion_clean": criterion,
                        "overlap_prob": level,
                        "outcome": outcome,
                        "mean": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n": len(values),
                    }
                )
                handle.write(f"      overlap {level:g}: {mean:.4f} [{lo:.4f}, {hi:.4f}] n={len(values)}\n")
            if len(means) >= 2:
                handle.write(f"      delta {means[-1][0]:g}-{means[0][0]:g}: {means[-1][1] - means[0][1]:+.4f}\n")
        handle.write("\n")
    return pd.DataFrame(rows)


def analyze_interaction(handle, table: pd.DataFrame, outcomes: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    methods = ordered_unique(table["method"])
    criteria = ordered_unique(table["criterion_clean"])
    handle.write("Interaction method x criterion\n")
    for outcome in outcomes:
        if outcome not in table:
            continue
        grid = pd.DataFrame(index=methods, columns=criteria, dtype=float)
        for method in methods:
            for criterion in criteria:
                mask = (table["method"] == method) & (table["criterion_clean"] == criterion)
                vals = finite_values(table.loc[mask, outcome])
                grid.loc[method, criterion] = float(np.mean(vals)) if len(vals) else np.nan
        out[outcome] = grid
        handle.write(f"Interaction table for {outcome}:\n{grid.to_string()}\n\n")
    return out


def analyze_cross_validation(handle, table: pd.DataFrame, outcomes: list[str], n_folds: int, rng) -> pd.DataFrame:
    n_folds = max(2, int(n_folds))
    folds = np.resize(np.arange(n_folds), len(table))
    rng.shuffle(folds)
    rows = []
    handle.write(f"Cross-validation: {n_folds} folds\n")
    for outcome in outcomes:
        if outcome not in table:
            continue
        errors = []
        for fold in range(n_folds):
            test = finite_values(table.loc[folds == fold, outcome])
            train = finite_values(table.loc[folds != fold, outcome])
            if len(test) and len(train):
                err = abs(float(np.mean(test)) - float(np.mean(train)))
                errors.append(err)
                rows.append({"outcome": outcome, "fold": fold + 1, "error": err})
                handle.write(f"{outcome} fold {fold + 1}: train={np.mean(train):.4f} test={np.mean(test):.4f} err={err:.4f}\n")
        if errors:
            handle.write(f"{outcome} CV error mean={np.mean(errors):.4f} std={np.std(errors, ddof=1) if len(errors) > 1 else 0:.4f}\n\n")
    return pd.DataFrame(rows)


def analyze_mixed_models(handle, table: pd.DataFrame, outcomes: list[str]) -> None:
    try:
        import statsmodels.formula.api as smf
    except Exception as exc:
        handle.write(f"Mixed models skipped: statsmodels unavailable ({exc})\n\n")
        return
    handle.write("Mixed models (statsmodels MixedLM)\n")
    for outcome in outcomes:
        if outcome not in table or "SNR_dB" not in table:
            continue
        cols = [outcome, "method_criterion_combo", "SNR_dB", "subject"]
        sub = table[cols].dropna().copy()
        if sub["subject"].nunique() < 2 or sub[outcome].nunique() < 2:
            handle.write(f"{outcome}: skipped, not enough variation\n")
            continue
        try:
            formula = f"{outcome} ~ C(method_criterion_combo) + SNR_dB"
            fit = smf.mixedlm(formula, sub, groups=sub["subject"]).fit(reml=True, method="lbfgs", disp=False)
            handle.write(f"\n{outcome}\n{fit.summary().as_text()}\n")
        except Exception as exc:
            handle.write(f"{outcome}: MixedLM failed: {exc}\n")
    handle.write("\n")


def analyze_vb_covariance(handle, table: pd.DataFrame, n_boot: int, rng) -> dict[str, dict]:
    cov_cols = [c for c in COVARIANCE_SPECS if c in table]
    vb = table[table["method"].str.contains("vb", na=False)].copy()
    out = {"covariates": [{"name": c, "label": COVARIANCE_SPECS[c]} for c in cov_cols]}
    if not cov_cols or vb.empty:
        handle.write("VB covariance relationships skipped: no covariance columns or VB rows.\n\n")
        return out

    handle.write("VB covariance relationships\n")
    for cov_col in cov_cols:
        cov_vals = pd.to_numeric(vb[cov_col], errors="coerce").to_numpy(float)
        metric = {"label": COVARIANCE_SPECS[cov_col], "outcomes": {}}
        handle.write(f"Covariance summary: {COVARIANCE_SPECS[cov_col]} ({cov_col})\n")
        for outcome in ["f1_score", "sensitivity", "precision"]:
            if outcome not in vb:
                continue
            x = pd.to_numeric(vb[outcome], errors="coerce").to_numpy(float)
            mask = np.isfinite(x) & np.isfinite(cov_vals)
            result = correlation_summary(x[mask], cov_vals[mask], n_boot, rng)
            metric["outcomes"][outcome] = result
            handle.write(
                f"  {outcome}: n={result['n']} Spearman rho={result['spearman_rho']:.4f} "
                f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}] p={result['spearman_p']:.4g}\n"
            )
        if "K_correct" in vb:
            k = pd.to_numeric(vb["K_correct"], errors="coerce").to_numpy(float)
            metric["k_correct_groups"] = covariance_kcorrect_groups(cov_vals, k, n_boot, rng)
            group = metric["k_correct_groups"]
            handle.write(
                f"  K correct mean={group['mean_correct']:.4f}; "
                f"K incorrect mean={group['mean_incorrect']:.4f}; "
                f"diff={group['mean_difference']:.4f}\n"
            )
        handle.write("\n")
        out[cov_col] = metric
    return out


def finite_values(values: Iterable[object]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    return arr[np.isfinite(arr)]


def finite_unique(values: Iterable[object]) -> list[float]:
    return [float(x) for x in sorted(set(finite_values(values)))]


def mean_ci(values: np.ndarray, n_boot: int, rng) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    if len(values) == 1 or n_boot <= 0:
        return mean, mean, mean
    sample_idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[sample_idx].mean(axis=1)
    return mean, float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))


def correlation_summary(x: np.ndarray, y: np.ndarray, n_boot: int, rng) -> dict[str, float]:
    result = {
        "n": int(len(x)),
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "slope": np.nan,
        "intercept": np.nan,
    }
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return result
    result["spearman_rho"], result["spearman_p"] = stats.spearmanr(x, y)
    result["pearson_r"], result["pearson_p"] = stats.pearsonr(x, y)
    result["slope"], result["intercept"] = np.polyfit(x, y, 1)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        xb, yb = x[idx], y[idx]
        if len(np.unique(xb)) > 1 and len(np.unique(yb)) > 1:
            boot.append(stats.spearmanr(xb, yb).statistic)
    if boot:
        result["ci_low"], result["ci_high"] = np.nanpercentile(boot, [2.5, 97.5])
    return result


def covariance_kcorrect_groups(cov_vals: np.ndarray, k_correct: np.ndarray, n_boot: int, rng) -> dict[str, float]:
    correct = cov_vals[np.isfinite(cov_vals) & (k_correct >= 0.5)]
    incorrect = cov_vals[np.isfinite(cov_vals) & (k_correct < 0.5)]
    mc, lc, hc = mean_ci(correct, n_boot, rng)
    mi, li, hi = mean_ci(incorrect, n_boot, rng)
    diff = np.nan
    diff_ci = (np.nan, np.nan)
    if len(correct) and len(incorrect) and n_boot > 0:
        boot = []
        for _ in range(n_boot):
            boot.append(np.mean(rng.choice(correct, len(correct), replace=True)) - np.mean(rng.choice(incorrect, len(incorrect), replace=True)))
        diff = float(mc - mi)
        diff_ci = tuple(np.nanpercentile(boot, [2.5, 97.5]))
    return {
        "n_correct": int(len(correct)),
        "n_incorrect": int(len(incorrect)),
        "mean_correct": mc,
        "ci_correct_low": lc,
        "ci_correct_high": hc,
        "mean_incorrect": mi,
        "ci_incorrect_low": li,
        "ci_incorrect_high": hi,
        "mean_difference": diff,
        "ci_difference_low": diff_ci[0],
        "ci_difference_high": diff_ci[1],
    }


def make_standard_plots(
    table: pd.DataFrame,
    method_table: pd.DataFrame,
    criterion_table: pd.DataFrame,
    method_summary: pd.DataFrame,
    criterion_summary: pd.DataFrame,
    snr_summary: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    interaction: dict[str, pd.DataFrame],
    cv: pd.DataFrame,
    vb_covariance: dict[str, dict],
    plots_dir: Path,
    n_boot: int,
    rng,
) -> None:
    create_boxplot_comparison(table, plots_dir)
    create_factor_plot(
        method_summary,
        "method",
        "Method effects with 95% CI (shared criterion: silhouette only)",
        plots_dir / "method_effects_with_ci.png",
    )
    create_factor_plot(criterion_summary, "criterion_clean", "Criterion effects (full-support methods only)", plots_dir / "criterion_effects_with_ci.png")
    create_snr_plot(table, plots_dir, n_boot, rng)
    create_overlap_plot(table, plots_dir)
    create_interaction_plot(interaction, plots_dir)
    create_cross_validation_plot(cv, plots_dir)
    create_runtime_snr_plot(table, plots_dir)
    create_avg_k_error_plot(table, plots_dir)
    create_k_estimated_vs_true_heatmap(table, plots_dir)
    create_abs_k_error_plot(table, plots_dir, n_boot, rng)
    create_method_criterion_boxplots(table, plots_dir)
    create_vb_covariance_plots(table, vb_covariance, plots_dir)


def choose_grid(n: int) -> tuple[int, int]:
    if n <= 3:
        return 1, max(1, n)
    if n == 4:
        return 2, 2
    if n <= 6:
        return 2, 3
    return math.ceil(n / 3), 3


def label_for_outcome(outcome: str) -> str:
    return OUTCOME_LABELS.get(outcome, outcome.replace("_", " "))


def create_boxplot_comparison(table: pd.DataFrame, plots_dir: Path) -> None:
    nrows, ncols = choose_grid(len(OUTCOMES))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 9), squeeze=False)
    pairs = table[["method", "criterion_clean"]].drop_duplicates()
    for ax, outcome in zip(axes.ravel(), OUTCOMES):
        data, labels = [], []
        for _, pair in pairs.iterrows():
            vals = finite_values(table.loc[(table["method"] == pair.method) & (table["criterion_clean"] == pair.criterion_clean), outcome])
            if len(vals):
                data.append(vals)
                labels.append(f"{display_method(pair.method)}\n{pair.criterion_clean}")
        if data:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
            ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.set_title(label_for_outcome(outcome))
        ax.grid(alpha=0.25)
    hide_unused_axes(axes, len(OUTCOMES))
    fig.suptitle("Method x criterion comparison")
    fig.tight_layout()
    fig.savefig(plots_dir / "boxplot_comparison.png", dpi=220)
    plt.close(fig)


def create_factor_plot(summary: pd.DataFrame, factor: str, title: str, output_file: Path) -> None:
    if summary.empty:
        return
    levels = ordered_unique(summary["level"])
    nrows, ncols = choose_grid(len(OUTCOMES))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8), squeeze=False)
    for ax, outcome in zip(axes.ravel(), OUTCOMES):
        sub = summary[summary["outcome"] == outcome].set_index("level").reindex(levels)
        x = np.arange(len(levels))
        means = sub["mean"].to_numpy(float)
        lo = clean_errorbar_delta(means - sub["ci_low"].to_numpy(float))
        hi = clean_errorbar_delta(sub["ci_high"].to_numpy(float) - means)
        ax.bar(x, means, color="#4c78a8", edgecolor="black")
        ax.errorbar(x, means, yerr=[lo, hi], fmt="k.", capsize=5)
        ax.set_xticks(x, [display_method(v) if factor == "method" else v for v in levels], rotation=35, ha="right")
        ax.set_title(label_for_outcome(outcome))
        ax.grid(axis="y", alpha=0.25)
    hide_unused_axes(axes, len(OUTCOMES))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_file, dpi=220)
    plt.close(fig)


def create_snr_plot(table: pd.DataFrame, plots_dir: Path, n_boot: int, rng) -> None:
    if "SNR_dB" not in table:
        return
    methods = ordered_unique(table["method"])
    snrs = finite_unique(table["SNR_dB"])
    nrows, ncols = choose_grid(len(OUTCOMES))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8), squeeze=False)
    for ax, outcome in zip(axes.ravel(), OUTCOMES):
        for method in methods:
            means, lows, highs = [], [], []
            for snr in snrs:
                values = finite_values(table.loc[(table["method"] == method) & (table["SNR_dB"] == snr), outcome])
                mean, lo, hi = mean_ci(values, n_boot, rng)
                means.append(mean)
                lows.append(lo)
                highs.append(hi)
            ax.plot(snrs, means, marker="o", linewidth=2, label=display_method(method))
            ax.fill_between(snrs, lows, highs, alpha=0.14)
        ax.set_title(label_for_outcome(outcome))
        ax.set_xlabel("SNR (dB)")
        ax.grid(alpha=0.25)
    hide_unused_axes(axes, len(OUTCOMES))
    axes.ravel()[0].legend()
    fig.suptitle("SNR effects by method with bootstrap 95% confidence intervals")
    fig.tight_layout()
    fig.savefig(plots_dir / "snr_effects.png", dpi=220)
    plt.close(fig)


def create_overlap_plot(table: pd.DataFrame, plots_dir: Path) -> None:
    data = selected_best_model_rows(table)
    if data.empty or "overlap_prob" not in data:
        return
    outcomes = [x for x in ["K_correct", "f1_score", "sensitivity", "precision", "K_sq_error", "K_abs_error"] if x in data]
    nrows, ncols = choose_grid(len(outcomes))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8), squeeze=False)
    for ax, outcome in zip(axes.ravel(), outcomes):
        groups, labels = [], []
        for _, _, label in BEST_MODEL_CRITERIA:
            sub = data[data["best_model_label"] == label]
            for overlap in finite_unique(sub["overlap_prob"]):
                vals = finite_values(sub.loc[sub["overlap_prob"] == overlap, outcome])
                if len(vals):
                    groups.append(vals)
                    labels.append(f"{label}\noverlap={overlap:g}")
        if groups:
            ax.boxplot(groups, tick_labels=labels, showfliers=False)
            ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        ax.set_title(label_for_outcome(outcome))
        ax.grid(alpha=0.25)
        if outcome in {"K_correct", "f1_score", "sensitivity", "precision"}:
            ax.set_ylim(0, 1)
    hide_unused_axes(axes, len(outcomes))
    fig.suptitle("Overlap effects: K-Means/silhouette vs VB-GMM/free-energy elbow")
    fig.tight_layout()
    fig.savefig(plots_dir / "overlap_effects_selected_methods.png", dpi=220)
    plt.close(fig)


def create_interaction_plot(interaction: dict[str, pd.DataFrame], plots_dir: Path) -> None:
    if not interaction:
        return
    nrows, ncols = choose_grid(len(OUTCOMES))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 9), squeeze=False)
    for ax, outcome in zip(axes.ravel(), OUTCOMES):
        grid = interaction.get(outcome)
        if grid is None:
            continue
        x = np.arange(len(grid.columns))
        for method, row in grid.iterrows():
            ax.plot(x, row.to_numpy(float), marker="o", linewidth=2, label=display_method(method))
        ax.set_xticks(x, grid.columns, rotation=35, ha="right")
        ax.set_title(label_for_outcome(outcome))
        ax.grid(alpha=0.25)
    hide_unused_axes(axes, len(OUTCOMES))
    axes.ravel()[0].legend()
    fig.suptitle("Method x criterion interaction")
    fig.tight_layout()
    fig.savefig(plots_dir / "interaction_plot.png", dpi=220)
    plt.close(fig)


def create_cross_validation_plot(cv: pd.DataFrame, plots_dir: Path) -> None:
    if cv.empty:
        return
    nrows, ncols = choose_grid(len(OUTCOMES))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8), squeeze=False)
    for ax, outcome in zip(axes.ravel(), OUTCOMES):
        sub = cv[cv["outcome"] == outcome]
        if sub.empty:
            continue
        ax.bar(sub["fold"], sub["error"], color="#59a14f", edgecolor="black")
        ax.axhline(sub["error"].mean(), color="#c44e52", linestyle="--")
        ax.set_title(label_for_outcome(outcome))
        ax.set_xlabel("Fold")
        ax.set_ylabel("CV error")
        ax.grid(axis="y", alpha=0.25)
    hide_unused_axes(axes, len(OUTCOMES))
    fig.suptitle("Cross-validation results")
    fig.tight_layout()
    fig.savefig(plots_dir / "cross_validation.png", dpi=220)
    plt.close(fig)


def create_runtime_snr_plot(table: pd.DataFrame, plots_dir: Path) -> None:
    if "runtime_s" not in table or "SNR_dB" not in table:
        return
    snrs = finite_unique(table["SNR_dB"])
    fig, ax = plt.subplots(figsize=(12, 7))
    for (method, criterion), sub in table.groupby(["method", "criterion_clean"], sort=False):
        means = [np.nanmean(finite_values(sub.loc[sub["SNR_dB"] == snr, "runtime_s"])) for snr in snrs]
        if np.isfinite(means).any():
            ax.plot(snrs, means, marker="o", linewidth=1.8, label=f"{display_method(method)} - {criterion}")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Runtime vs SNR by method and criterion")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "runtime_snr_effect.png", dpi=220)
    plt.close(fig)


def create_avg_k_error_plot(table: pd.DataFrame, plots_dir: Path) -> None:
    data = selected_best_model_rows(table)
    if data.empty:
        return
    rows = [label for _, _, label in BEST_MODEL_CRITERIA if label in set(data["best_model_label"])]
    if "K_true" in data:
        cols = sorted(finite_unique(data["K_true"]))
        data["_heatmap_col"] = pd.to_numeric(data["K_true"], errors="coerce")
        col_labels = [f"K={int(c)}" if float(c).is_integer() else f"K={c:g}" for c in cols]
    else:
        cols = ["overall"]
        data["_heatmap_col"] = "overall"
        col_labels = ["overall"]
    grid = data.pivot_table(values="K_error", index="best_model_label", columns="_heatmap_col", aggfunc="mean").reindex(index=rows, columns=cols)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    matrix = grid.to_numpy(float)
    vmax = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), col_labels, rotation=0)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title("Average signed K error for selected best models")
    fig.colorbar(im, ax=ax)
    for i, row_label in enumerate(rows):
        for j, col in enumerate(cols):
            val = grid.loc[row_label, col]
            if pd.notna(val):
                n = int(((data["best_model_label"] == row_label) & (data["_heatmap_col"] == col)).sum())
                ax.text(j, i, f"{val:.2f}\n(n={n})", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "avg_k_error_heatmap.png", dpi=220)
    plt.close(fig)


def create_k_estimated_vs_true_heatmap(table: pd.DataFrame, plots_dir: Path) -> None:
    if not {"K_true", "K_estimated", "method"}.issubset(table.columns):
        return
    data = selected_best_model_rows(table).dropna(subset=["K_true", "K_estimated"]).copy()
    if data.empty:
        return
    data["K_true"] = data["K_true"].astype(int)
    data["K_estimated"] = data["K_estimated"].astype(int)
    labels = [label for _, _, label in BEST_MODEL_CRITERIA if label in set(data["best_model_label"])]
    true_vals = sorted(data["K_true"].unique())
    estimated_vals = sorted(data["K_estimated"].unique())
    fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 5), squeeze=False)
    im = None
    for ax, label in zip(axes.ravel(), labels):
        sub = data[data["best_model_label"] == label]
        counts = pd.crosstab(sub["K_true"], sub["K_estimated"]).reindex(index=true_vals, columns=estimated_vals, fill_value=0)
        rates = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
        im = ax.imshow(rates.to_numpy(float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(estimated_vals)), estimated_vals)
        ax.set_yticks(range(len(true_vals)), true_vals)
        ax.set_xlabel("Estimated K")
        ax.set_ylabel("True K")
        ax.set_title(label)
        for r in range(len(true_vals)):
            for c in range(len(estimated_vals)):
                value = rates.iloc[r, c]
                count = counts.iloc[r, c]
                if pd.notna(value) and count:
                    ax.text(c, r, f"{value:.2f}\n(n={count})", ha="center", va="center", fontsize=8, color="white" if value >= 0.55 else "black")
    if im is not None:
        cbar_ax = fig.add_axes([0.91, 0.16, 0.018, 0.64])
        fig.colorbar(im, cax=cbar_ax, label="Row-normalized rate")
    fig.suptitle("Estimated K vs true K for selected best models")
    fig.subplots_adjust(left=0.08, right=0.88, top=0.84, bottom=0.14, wspace=0.35)
    fig.savefig(plots_dir / "k_estimated_vs_true_heatmap.png", dpi=220)
    plt.close(fig)


def create_abs_k_error_plot(table: pd.DataFrame, plots_dir: Path, n_boot: int, rng) -> None:
    rows = []
    for method in ordered_unique(table["method"]):
        vals = finite_values(table.loc[table["method"] == method, "K_abs_error"])
        mean, lo, hi = mean_ci(vals, n_boot, rng)
        rows.append((method, len(vals), mean, lo, hi))
    x = np.arange(len(rows))
    means = np.array([r[2] for r in rows])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, means, color="#76a5c9", edgecolor="black")
    ax.errorbar(
        x,
        means,
        yerr=[clean_errorbar_delta(means - np.array([r[3] for r in rows])), clean_errorbar_delta(np.array([r[4] for r in rows]) - means)],
        fmt="k.",
        capsize=8,
    )
    ax.set_xticks(x, [display_method(r[0]) for r in rows], rotation=25, ha="right")
    ax.set_ylabel("Mean absolute K error")
    ax.set_title("Absolute K error by method")
    ax.grid(axis="y", alpha=0.25)
    for idx, row in enumerate(rows):
        ax.text(idx, means[idx], f"n={row[1]}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / "abs_k_error_by_method.png", dpi=220)
    plt.close(fig)


def create_method_criterion_boxplots(table: pd.DataFrame, plots_dir: Path) -> None:
    for outcome, filename in [("K_abs_error", "method_criterion_abs_k_error_boxplots.png"), ("f1_score", "method_criterion_f1_score_boxplots.png")]:
        methods = ordered_unique(table["method"])
        fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 5), squeeze=False)
        for ax, method in zip(axes.ravel(), methods):
            sub = table[table["method"] == method]
            data = [finite_values(g[outcome]) for _, g in sub.groupby("criterion_clean", sort=False)]
            labels = ordered_unique(sub["criterion_clean"])
            data_labels = [(d, l) for d, l in zip(data, labels) if len(d)]
            if data_labels:
                ax.boxplot([x[0] for x in data_labels], tick_labels=[x[1] for x in data_labels], showfliers=False)
                ax.tick_params(axis="x", rotation=35)
            ax.set_title(display_method(method))
            ax.set_ylabel(label_for_outcome(outcome))
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"{label_for_outcome(outcome)} by method and criterion")
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=220)
        plt.close(fig)


def create_vb_covariance_plots(table: pd.DataFrame, vb_covariance: dict[str, dict], plots_dir: Path) -> None:
    if not vb_covariance.get("covariates"):
        return
    vb = table[table["method"].str.contains("vb", na=False)].copy()
    for spec in vb_covariance["covariates"]:
        cov_col = spec["name"]
        if cov_col not in vb:
            continue
        nrows, ncols = choose_grid(3)
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4), squeeze=False)
        y = pd.to_numeric(vb[cov_col], errors="coerce").to_numpy(float)
        for ax, outcome in zip(axes.ravel(), ["f1_score", "sensitivity", "precision"]):
            x = pd.to_numeric(vb[outcome], errors="coerce").to_numpy(float)
            mask = np.isfinite(x) & np.isfinite(y)
            if "SNR_dB" in vb:
                sc = ax.scatter(x[mask], y[mask], c=vb.loc[mask, "SNR_dB"], cmap="viridis", s=28, edgecolors="black", linewidths=0.2)
                fig.colorbar(sc, ax=ax, label="SNR (dB)")
            else:
                ax.scatter(x[mask], y[mask], s=28)
            stats_i = vb_covariance.get(cov_col, {}).get("outcomes", {}).get(outcome)
            if stats_i and np.isfinite(stats_i["slope"]):
                xs = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
                ax.plot(xs, stats_i["slope"] * xs + stats_i["intercept"], color="#c44e52", linewidth=2)
                ax.set_title(f"{label_for_outcome(outcome)}\nrho={stats_i['spearman_rho']:.2f}, p={stats_i['spearman_p']:.3g}")
            else:
                ax.set_title(label_for_outcome(outcome))
            ax.set_xlabel(label_for_outcome(outcome))
            ax.set_ylabel(spec["label"])
            ax.grid(alpha=0.25)
        hide_unused_axes(axes, 3)
        fig.suptitle(f"VB covariance relationships: {spec['label']}")
        fig.tight_layout()
        fig.savefig(plots_dir / f"vb_covariance_relationships_{cov_col}.png", dpi=220)
        plt.close(fig)

    create_vb_covariance_kcorrect_barplots(vb_covariance, plots_dir)


def create_vb_covariance_kcorrect_barplots(vb_covariance: dict[str, dict], plots_dir: Path) -> None:
    specs = vb_covariance.get("covariates") or []
    if not specs:
        return
    fig, axes = plt.subplots(1, len(specs), figsize=(5 * len(specs), 4), squeeze=False)
    for ax, spec in zip(axes.ravel(), specs):
        group = vb_covariance.get(spec["name"], {}).get("k_correct_groups")
        if not group:
            continue
        means = [group["mean_correct"], group["mean_incorrect"]]
        lo = clean_errorbar_delta(np.array([means[0] - group["ci_correct_low"], means[1] - group["ci_incorrect_low"]]))
        hi = clean_errorbar_delta(np.array([group["ci_correct_high"] - means[0], group["ci_incorrect_high"] - means[1]]))
        ax.bar([0, 1], means, color=["#4c78a8", "#e45756"], edgecolor="black")
        ax.errorbar([0, 1], means, yerr=[lo, hi], fmt="k.", capsize=8)
        ax.set_xticks([0, 1], ["K correct", "K incorrect"], rotation=20)
        ax.set_title(spec["label"])
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("VB covariance by K selection accuracy")
    fig.tight_layout()
    fig.savefig(plots_dir / "vb_covariance_by_k_correct.png", dpi=220)
    plt.close(fig)


def create_simulated_microstate_method_comparison(
    table: pd.DataFrame,
    plots_dir: Path,
    template: TemplateTopographies | None,
    *,
    seed: int = 42,
    montage_type: str = "full",
    kmeans_criterion: str = "silhouette",
    spm_criterion: str = "elbow sil combined",
) -> Path | None:
    if template is None or "json_file" not in table:
        return None
    keys = [c for c in ["rep", "K_true", "SNR_dB", "overlap_prob", "montage_type", "n_leads"] if c in table]
    if not keys:
        return None

    subset = table.copy()
    if "montage_type" in subset and (subset["montage_type"] == montage_type).any():
        subset = subset[subset["montage_type"] == montage_type].copy()

    spm = subset[(subset["method"] == "spm vb") & (subset["criterion_clean"] == spm_criterion)]
    if spm.empty:
        try:
            spm_criterion = select_best_spm_criterion(subset)
            spm = subset[(subset["method"] == "spm vb") & (subset["criterion_clean"] == spm_criterion)]
        except Exception:
            return None
    km = subset[(subset["method"] == "koenig kmeans") & (subset["criterion_clean"] == kmeans_criterion)]
    if spm.empty or km.empty:
        return None

    spm_keep = spm[keys + ["json_file", "K_estimated"]].rename(columns={"json_file": "spm_json_file", "K_estimated": "K_estimated_spm"})
    km_keep = km[keys + ["json_file", "K_estimated"]].rename(columns={"json_file": "kmeans_json_file", "K_estimated": "K_estimated_kmeans"})
    joined = spm_keep.merge(km_keep, on=keys)
    joined = joined[joined["spm_json_file"].map(valid_path).notna() & joined["kmeans_json_file"].map(valid_path).notna()]
    if joined.empty:
        return None

    selection = joined.sample(n=1, random_state=seed).iloc[0]
    spm_data = json.loads(valid_path(selection["spm_json_file"]).read_text(encoding="utf-8"))
    km_data = json.loads(valid_path(selection["kmeans_json_file"]).read_text(encoding="utf-8"))

    true_maps, true_channels = extract_json_state_maps(spm_data, "true_microstates")
    spm_maps, spm_channels = extract_json_state_maps(spm_data, "estimated_microstates")
    km_maps, km_channels = extract_json_state_maps(km_data, "estimated_microstates")

    true_labels, true_corrs = labels_and_correlations_from_json(spm_data, "true", true_maps.shape[0])
    spm_labels, spm_corrs = labels_and_correlations_from_json(spm_data, "estimated", spm_maps.shape[0])
    km_labels, km_corrs = labels_and_correlations_from_json(km_data, "estimated", km_maps.shape[0])

    rows = [
        ("True", project_maps_to_template_channels(true_maps, true_channels, template.channel_labels), true_labels, true_corrs),
        ("K-Means", project_maps_to_template_channels(km_maps, km_channels, template.channel_labels), km_labels, km_corrs),
        ("VB-GMM", project_maps_to_template_channels(spm_maps, spm_channels, template.channel_labels), spm_labels, spm_corrs),
    ]
    n_cols = max((maps.shape[0] for _, maps, _, _ in rows), default=0)
    if n_cols == 0:
        return None
    finite_maps = [m[np.isfinite(m)] for _, maps, _, _ in rows for m in maps if np.isfinite(m).any()]
    clim = float(np.nanmax(np.abs(np.concatenate(finite_maps)))) if finite_maps else 1.0
    if not np.isfinite(clim) or clim <= 0:
        clim = 1.0

    fig, axes = plt.subplots(3, n_cols, figsize=(max(13, 2.4 * n_cols), 7.5), squeeze=False)
    for row_idx, (row_label, maps, labels, corrs) in enumerate(rows):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            if col_idx < maps.shape[0]:
                vals = maps[col_idx]
                plot_topomap(ax, vals, template.info if template.info is not None else template.xy, clim=clim)
                label = labels[col_idx] if col_idx < len(labels) else f"state {col_idx + 1}"
                ax.set_title(label, fontsize=10, fontweight="bold")
                if col_idx < len(corrs) and np.isfinite(corrs[col_idx]):
                    ax.text(0.5, -0.08, f"r={corrs[col_idx]:.2f}", transform=ax.transAxes, ha="center", fontsize=8)
            else:
                ax.axis("off")

    title_bits = {
        "rep": selection.get("rep", ""),
        "K": selection.get("K_true", ""),
        "SNR": selection.get("SNR_dB", ""),
        "overlap": selection.get("overlap_prob", ""),
        "montage": selection.get("montage_type", ""),
    }
    fig.suptitle(
        "Simulated microstate method comparison\n"
        f"rep={title_bits['rep']} | K_true={title_bits['K']} | SNR={title_bits['SNR']} dB | "
        f"overlap={title_bits['overlap']} | montage={title_bits['montage']} | "
        f"K-Means={kmeans_criterion} | VB-GMM={spm_criterion}",
        fontsize=12,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.82, bottom=0.08, wspace=0.35, hspace=0.55)
    for row_idx, (row_label, _, _, _) in enumerate(rows):
        pos = axes[row_idx, 0].get_position()
        fig.text(0.06, (pos.y0 + pos.y1) / 2, row_label, rotation=90, ha="center", va="center", fontsize=11, fontweight="bold")
    output_file = plots_dir / f"simulated_microstate_method_comparison_seed{seed}.png"
    fig.savefig(output_file, dpi=220)
    plt.close(fig)
    return output_file


def extract_json_state_maps(data: dict, field_name: str) -> tuple[np.ndarray, list[str]]:
    states = data.get(field_name)
    if not states:
        raise ValueError(f"JSON payload is missing {field_name}")
    state_names = sorted(states, key=lambda name: int(re.search(r"(\d+)$", name).group(1)) if re.search(r"(\d+)$", name) else name)
    channel_info = data.get("channel_info", {})
    labels = [str(x) for x in channel_info.get("labels", data.get("metadata", {}).get("channel_labels", []))]
    sanitized = [str(x) for x in channel_info.get("labels_sanitized", [])]
    if not sanitized:
        sanitized = [sanitize_json_channel_label(x) for x in labels]
    maps = np.full((len(state_names), len(labels)), np.nan)
    for row, state_name in enumerate(state_names):
        state = states[state_name]
        for col, (label, clean_label) in enumerate(zip(labels, sanitized)):
            if label in state:
                maps[row, col] = state[label]
            elif clean_label in state:
                maps[row, col] = state[clean_label]
    return maps, labels


def labels_and_correlations_from_json(data: dict, kind: str, n_states: int) -> tuple[list[str], list[float]]:
    meta = data.get("metadata", {})
    if kind == "true":
        labels = meta.get("true_template_labels") or data.get("backfit", {}).get("true_state_template_labels")
        correlations = [np.nan] * n_states
    else:
        labels = meta.get("template_alignment", {}).get("labels") or data.get("backfit", {}).get("estimated_state_template_labels")
        correlations = meta.get("template_alignment", {}).get("correlations", [])
    labels = [str(x) for x in labels] if labels else []
    correlations = [float(x) if x is not None else np.nan for x in correlations]
    labels.extend(f"state {i + 1}" for i in range(len(labels), n_states))
    correlations.extend(np.nan for _ in range(len(correlations), n_states))
    return labels[:n_states], correlations[:n_states]


def project_maps_to_template_channels(maps: np.ndarray, source_channels: list[str], template_channels: list[str]) -> np.ndarray:
    source_index = {canonical_channel_label(label): idx for idx, label in enumerate(source_channels)}
    out = np.full((maps.shape[0], len(template_channels)), np.nan)
    for col, label in enumerate(template_channels):
        src = source_index.get(canonical_channel_label(label))
        if src is not None and src < maps.shape[1]:
            out[:, col] = maps[:, src]
    return normalize_map_rows(out)


def canonical_channel_label(label: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(label).lower().replace("eeg", "").strip())


def sanitize_json_channel_label(label: object) -> str:
    text = re.sub(r"[-/\\\s.,()\[\]{}]", "_", str(label))
    text = re.sub(r"^_+|_+$", "", text)
    if not text or not re.match(r"[A-Za-z]", text[0]):
        text = "Ch" + text
    return text


def clean_errorbar_delta(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.nan_to_num(np.maximum(arr, 0), nan=0.0, posinf=0.0, neginf=0.0)


def hide_unused_axes(axes: np.ndarray, used: int) -> None:
    for ax in axes.ravel()[used:]:
        ax.axis("off")


def make_confusion_outputs(
    table: pd.DataFrame,
    results_dir: Path,
    plots_dir: Path,
    template: TemplateTopographies | None,
    *,
    max_pair_plots: int = 0,
    run_pair_plots: bool = True,
) -> None:
    if h5py is None or "backfit_diagnostic_file" not in table:
        return
    create_backfit_confusion_summary(table, plots_dir)
    create_confusion_report_csvs(table, plots_dir / "backfit_confusions")
    if run_pair_plots:
        create_backfit_confusion_comparison_plots(table, plots_dir, template, max_pair_plots=max_pair_plots)


def valid_path(value: object) -> Path | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    path = Path(str(value))
    return path if path.is_file() else None


def compute_k_gap(table: pd.DataFrame) -> pd.Series:
    if "K_gap" in table:
        return pd.to_numeric(table["K_gap"], errors="coerce")
    return pd.to_numeric(table["K_true"], errors="coerce") - pd.to_numeric(table["K_estimated"], errors="coerce")


def select_best_spm_criterion(table: pd.DataFrame, metric: str = "K_correct") -> str:
    spm = table[table["method"] == "spm vb"].copy()
    if spm.empty:
        raise ValueError("No VB-GMM rows available")
    score = spm.groupby("criterion_clean", sort=False)[metric].mean().sort_values(ascending=False)
    return str(score.index[0])


def create_backfit_confusion_summary(table: pd.DataFrame, plots_dir: Path) -> None:
    paths = table["backfit_diagnostic_file"].map(valid_path)
    valid = paths.notna()
    if not valid.any():
        return
    gap = compute_k_gap(table)
    kmeans_criterion = "silhouette"
    spm_comparison_criterion = "silhouette"
    try:
        spm_best_criterion = select_best_spm_criterion(table)
    except Exception:
        spm_best_criterion = spm_comparison_criterion

    mask_kacc_km = (table["method"] == "koenig kmeans") & (table["criterion_clean"] == kmeans_criterion)
    mask_kacc_spm = (table["method"] == "spm vb") & (table["criterion_clean"] == spm_comparison_criterion)
    mask_diag_km = mask_kacc_km & valid & gap.isin([0, 1, 2, 3])
    mask_diag_spm = (table["method"] == "spm vb") & (table["criterion_clean"] == spm_best_criterion) & valid & gap.isin([0, 1, 2, 3])
    if not mask_kacc_km.any() or not mask_kacc_spm.any() or not mask_diag_km.any() or not mask_diag_spm.any():
        return

    diag_km = collect_backfit_diagnostics(table.loc[mask_diag_km, "backfit_diagnostic_file"])
    diag_spm = collect_backfit_diagnostics(table.loc[mask_diag_spm, "backfit_diagnostic_file"])
    if not diag_km or not diag_spm:
        return

    label_order = merge_label_order([], [x.labels for x in diag_km + diag_spm])
    conf_km = mean_confusion_matrix(diag_km, label_order)
    conf_spm = mean_confusion_matrix(diag_spm, label_order)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_method_bar(
        axes[0, 0],
        [table.loc[mask_kacc_km, "K_correct"].mean(), table.loc[mask_kacc_spm, "K_correct"].mean()],
        [int(mask_kacc_km.sum()), int(mask_kacc_spm.sum())],
        ["K-Means", "VB-GMM"],
        "K selection accuracy\n(silhouette for both methods)",
    )
    plot_method_bar(
        axes[0, 1],
        [np.nanmean([x.accuracy for x in diag_km]), np.nanmean([x.accuracy for x in diag_spm])],
        [len(diag_km), len(diag_spm)],
        ["K-Means", "VB-GMM"],
        f"Backfit accuracy\n(K-Means: {kmeans_criterion}; VB-GMM: {spm_best_criterion})",
    )
    plot_confusion_heatmap(axes[1, 0], conf_km, label_order, "Confusion grid: K-Means")
    plot_confusion_heatmap(axes[1, 1], conf_spm, label_order, "Confusion grid: VB-GMM")
    fig.suptitle("Simulated method comparison and backfit confusion summary", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "backfit_confusion_summary.png", dpi=220)
    plt.close(fig)


def collect_backfit_diagnostics(files: Iterable[object], mode: str = "legacy") -> list[BackfitDiagnostic]:
    diagnostics = []
    for file_value in files:
        path = valid_path(file_value)
        if path is None:
            continue
        try:
            labels, counts, accuracy = read_backfit_diagnostic(path, mode=mode)
        except Exception:
            continue
        if labels and counts.size:
            diagnostics.append(BackfitDiagnostic(labels, counts, accuracy))
    return diagnostics


def read_backfit_diagnostic(path: Path, mode: str = "legacy") -> tuple[list[str], np.ndarray, float]:
    if h5py is None:
        return [], np.empty((0, 0)), np.nan
    with h5py.File(path, "r") as handle:
        root = "BackfitDiagnostics"
        if root not in handle or h5_scalar(handle, f"{root}/ok") < 1:
            return [], np.empty((0, 0)), np.nan
        labels = h5_strings(handle, f"{root}/template_labels")
        paths = {
            "legacy": f"{root}/confusion_counts",
            "cluster": f"{root}/cluster/label_confusion_counts",
            "hard": f"{root}/hard/label_confusion_counts",
            "mixture": f"{root}/mixture/label_confusion_counts",
        }
        count_path = paths.get(mode, paths["legacy"])
        if count_path not in handle:
            return [], np.empty((0, 0)), np.nan
        if mode == "mixture" and f"{root}/mixture/available" in handle and h5_scalar(handle, f"{root}/mixture/available") < 1:
            return [], np.empty((0, 0)), np.nan
        counts = np.asarray(handle[count_path], dtype=float)
        if counts.ndim != 2:
            return [], np.empty((0, 0)), np.nan
        n = min(len(labels), counts.shape[0], counts.shape[1])
        accuracy = h5_scalar(handle, f"{root}/hard/label_top1_accuracy") if f"{root}/hard/label_top1_accuracy" in handle else np.nan
        return labels[:n], counts[:n, :n], accuracy


def h5_scalar(handle, path: str) -> float:
    arr = np.asarray(handle[path])
    return float(np.ravel(arr)[0])


def h5_strings(handle, path: str) -> list[str]:
    ds = handle[path]
    arr = ds[()]
    if arr.dtype == object:
        return [decode_h5_ref(handle, ref) for ref in arr.ravel()]
    return [decode_h5_chars(arr)]


def decode_h5_ref(handle, ref) -> str:
    return decode_h5_chars(np.asarray(handle[ref]))


def decode_h5_chars(arr: np.ndarray) -> str:
    flat = np.ravel(arr)
    if flat.dtype.kind in {"u", "i"}:
        return "".join(chr(int(x)) for x in flat if int(x) != 0)
    if flat.dtype.kind in {"S", "U"}:
        return "".join(str(x.decode() if isinstance(x, bytes) else x) for x in flat)
    return str(flat[0]) if len(flat) else ""


def merge_label_order(template_labels: list[str], label_groups: Iterable[list[str]]) -> list[str]:
    out = list(template_labels)
    for labels in label_groups:
        for label in labels:
            if label not in out:
                out.append(label)
    return out


def reorder_matrix(values: np.ndarray, labels: list[str], order: list[str], fill=np.nan) -> np.ndarray:
    out = np.full((len(order), len(order)), fill, dtype=float)
    label_index = {label: idx for idx, label in enumerate(labels)}
    for r, row_label in enumerate(order):
        if row_label not in label_index:
            continue
        for c, col_label in enumerate(order):
            if col_label in label_index:
                out[r, c] = values[label_index[row_label], label_index[col_label]]
    return out


def normalize_rows(values: np.ndarray) -> np.ndarray:
    out = values.astype(float).copy()
    for idx in range(out.shape[0]):
        row = out[idx]
        mask = np.isfinite(row)
        total = row[mask].sum()
        out[idx, mask] = row[mask] / total if total > 0 else np.nan
    return out


def mean_confusion_matrix(diagnostics: list[BackfitDiagnostic], order: list[str]) -> np.ndarray:
    mats = [normalize_rows(reorder_matrix(d.counts, d.labels, order)) for d in diagnostics]
    return np.nanmean(np.stack(mats, axis=2), axis=2) if mats else np.empty((0, 0))


def aggregate_confusions(files: Iterable[object], mode: str = "legacy") -> tuple[np.ndarray, list[str], int]:
    labels_total: list[str] = []
    counts_total = np.empty((0, 0))
    n_runs = 0
    for diag in collect_backfit_diagnostics(files, mode=mode):
        labels_total = merge_label_order(labels_total, [diag.labels])
        if counts_total.size == 0:
            counts_total = reorder_matrix(diag.counts, diag.labels, labels_total, fill=0)
        else:
            counts_total = reorder_matrix(counts_total, labels_total[: counts_total.shape[0]], labels_total, fill=0) + reorder_matrix(
                diag.counts, diag.labels, labels_total, fill=0
            )
        n_runs += 1
    return counts_total, labels_total, n_runs


def plot_method_bar(ax, values: list[float], counts: list[int], labels: list[str], title: str) -> None:
    ax.bar(range(len(values)), values, color=["#4c78a8", "#f58518"], edgecolor="black")
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(values)), labels, rotation=15)
    ax.set_ylabel("Mean accuracy")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(values):
        if np.isfinite(value):
            ax.text(i, value + 0.03, f"{value:.2f}\n(n={counts[i]})", ha="center", fontweight="bold")


def plot_confusion_heatmap(ax, values: np.ndarray, labels: list[str], title: str, counts: np.ndarray | None = None) -> None:
    if values.size == 0 or not np.isfinite(values).any():
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    im = ax.imshow(values, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Estimated template label")
    ax.set_ylabel("True template label")
    ax.set_title(title)
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            v = values[r, c]
            if np.isfinite(v):
                text = f"{v:.2f}" if counts is None or counts.size == 0 else f"{v:.2f}\n(n={counts[r, c]:.0f})"
                ax.text(c, r, text, ha="center", va="center", color="white" if v >= 0.55 else "black", fontsize=8, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Confusion rate")


def create_confusion_report_csvs(table: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("cluster_*"):
        stale.unlink()
    rows = []
    valid = table["backfit_diagnostic_file"].map(valid_path).notna()
    for mode in ["legacy", "hard", "mixture"]:
        for key, group in table[valid].groupby(["method", "criterion_clean", "K_true", "K_estimated"], sort=False):
            counts, labels, n_runs = aggregate_confusions(group["backfit_diagnostic_file"], mode=mode)
            if counts.size == 0:
                continue
            rownorm = normalize_rows(counts)
            stub = sanitize_stub("__".join(map(str, (mode, *key))))
            counts_csv = output_dir / f"{stub}_counts.csv"
            rownorm_csv = output_dir / f"{stub}_row_normalized.csv"
            pd.DataFrame(counts, index=labels, columns=labels).to_csv(counts_csv)
            pd.DataFrame(rownorm, index=labels, columns=labels).to_csv(rownorm_csv)
            fig, ax = plt.subplots(figsize=(7, 6))
            plot_confusion_heatmap(ax, rownorm, labels, f"{mode}: {key[0]} | {key[1]}")
            png = output_dir / f"{stub}_row_normalized.png"
            fig.tight_layout()
            fig.savefig(png, dpi=180)
            plt.close(fig)
            rows.append(
                {
                    "analysis": mode,
                    "method": key[0],
                    "criterion": key[1],
                    "K_true": key[2],
                    "K_estimated": key[3],
                    "n_runs": n_runs,
                    "counts_csv": counts_csv,
                    "row_normalized_csv": rownorm_csv,
                    "row_normalized_png": png,
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "backfit_confusion_manifest.csv", index=False)


def create_backfit_confusion_comparison_plots(
    table: pd.DataFrame,
    plots_dir: Path,
    template: TemplateTopographies | None,
    *,
    max_pair_plots: int = 0,
) -> None:
    valid = table["backfit_diagnostic_file"].map(valid_path).notna()
    underfit = pd.to_numeric(table["K_true"], errors="coerce") > pd.to_numeric(table["K_estimated"], errors="coerce")
    keep = valid & underfit & table["method"].isin(["koenig kmeans", "spm vb"])
    if not keep.any():
        return
    output_dir = plots_dir / "backfit_confusion_comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    made = 0
    for key, group in table[keep].groupby(["criterion_clean", "K_true", "K_estimated"], sort=False):
        km = group[group["method"] == "koenig kmeans"]
        vb = group[group["method"] == "spm vb"]
        counts_km, labels_km, n_km = aggregate_confusions(km["backfit_diagnostic_file"])
        counts_vb, labels_vb, n_vb = aggregate_confusions(vb["backfit_diagnostic_file"])
        if counts_km.size == 0 and counts_vb.size == 0:
            continue
        base_labels = template.labels if template else []
        order = merge_label_order(base_labels, [labels_km, labels_vb])
        km_counts = reorder_matrix(counts_km, labels_km, order, fill=0) if counts_km.size else np.full((len(order), len(order)), np.nan)
        vb_counts = reorder_matrix(counts_vb, labels_vb, order, fill=0) if counts_vb.size else np.full((len(order), len(order)), np.nan)
        km_norm = normalize_rows(km_counts) if np.isfinite(km_counts).any() else np.full_like(km_counts, np.nan)
        vb_norm = normalize_rows(vb_counts) if np.isfinite(vb_counts).any() else np.full_like(vb_counts, np.nan)
        criterion, k_true, k_est = key
        output_file = output_dir / f"confusion_compare__{sanitize_stub(criterion)}__Ktrue{int(k_true)}__Kest{int(k_est)}.png"
        create_method_confusion_pair_plot(km_norm, vb_norm, km_counts, vb_counts, order, template, output_file, criterion, k_true, k_est, n_km, n_vb)
        rows.append(
            {
                "criterion": criterion,
                "K_true": k_true,
                "K_estimated": k_est,
                "K_gap": k_true - k_est,
                "n_runs_kmeans": n_km,
                "n_runs_spm_vb": n_vb,
                "comparison_plot": output_file,
            }
        )
        made += 1
        if max_pair_plots and made >= max_pair_plots:
            break
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "backfit_confusion_comparison_manifest.csv", index=False)


def create_method_confusion_pair_plot(
    km_norm: np.ndarray,
    vb_norm: np.ndarray,
    km_counts: np.ndarray,
    vb_counts: np.ndarray,
    labels: list[str],
    template: TemplateTopographies | None,
    output_file: Path,
    criterion: str,
    k_true: float,
    k_est: float,
    n_km: int,
    n_vb: int,
) -> None:
    n = len(labels)
    fig = plt.figure(figsize=(17, 9))
    fig.suptitle(f"Microstate misrecognition | {criterion} | K_true={int(k_true)}, K_est={int(k_est)}", fontsize=15, fontweight="bold")
    x_row, w_row, gap = 0.03, 0.10, 0.02
    x_left, w_heat = x_row + w_row + gap, 0.30
    x_right = x_left + w_heat + 0.13
    y_heat, h_heat = 0.14, 0.56
    y_top, h_top = y_heat + h_heat + 0.035, 0.11

    fig.text(x_left + w_heat / 2, y_top + h_top + 0.02, f"K-Means (n={n_km})", ha="center", fontsize=12, fontweight="bold")
    fig.text(x_right + w_heat / 2, y_top + h_top + 0.02, f"VB-GMM (n={n_vb})", ha="center", fontsize=12, fontweight="bold")

    clim = template_clim(template)
    for i, label in enumerate(labels):
        y_ax = y_heat + (n - i - 1) * h_heat / n
        ax = fig.add_axes([x_row, y_ax, w_row, h_heat / n])
        render_template_topography(ax, label, template, clim)
        ax = fig.add_axes([x_left + i * w_heat / n, y_top, w_heat / n, h_top])
        render_template_topography(ax, label, template, clim)
        ax = fig.add_axes([x_right + i * w_heat / n, y_top, w_heat / n, h_top])
        render_template_topography(ax, label, template, clim)

    ax_left = fig.add_axes([x_left, y_heat, w_heat, h_heat])
    ax_right = fig.add_axes([x_right, y_heat, w_heat, h_heat])
    im = ax_left.imshow(km_norm, vmin=0, vmax=1, cmap="viridis")
    ax_right.imshow(vb_norm, vmin=0, vmax=1, cmap="viridis")
    for ax, norm, counts in [(ax_left, km_norm, km_counts), (ax_right, vb_norm, vb_counts)]:
        ax.set_xticks(range(n), [""] * n)
        ax.set_yticks(range(n), [""] * n)
        ax.set_xlabel("Estimated label")
        if not np.isfinite(norm).any():
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontweight="bold")
            continue
        for r in range(norm.shape[0]):
            for c in range(norm.shape[1]):
                v = norm[r, c]
                if np.isfinite(v):
                    ax.text(c, r, f"{v:.2f}\n(n={counts[r, c]:.0f})", ha="center", va="center", fontsize=7, color="white" if v >= 0.6 else "black")
    ax_left.set_ylabel("True label")
    cax = fig.add_axes([0.92, 0.24, 0.015, 0.40])
    fig.colorbar(im, cax=cax, label="Row-normalized confusion")
    fig.savefig(output_file, dpi=220)
    plt.close(fig)


def load_template_topographies(template_file: str | Path, k: int = 7) -> TemplateTopographies | None:
    path = Path(template_file)
    if not path.is_file():
        return None
    try:
        mat = io.loadmat(path, squeeze_me=True, struct_as_record=False)
        maps, labels = template_maps_from_mat(mat, k)
        chanlocs = np.atleast_1d(mat["chanlocs"])
        template_info = template_info_from_eeglab(path, chanlocs, maps.shape[1])
        if template_info is None:
            channel_labels, xy, keep, info = (*template_xy_from_chanlocs(chanlocs, maps.shape[1]), None)
        else:
            channel_labels, xy, keep, info = template_info
        maps = normalize_map_rows(maps[:, keep])
        order = np.argsort([x.lower() for x in labels])
        return TemplateTopographies([labels[i] for i in order], list(channel_labels), maps[order], xy, info)
    except Exception:
        return None


def template_maps_from_mat(mat: dict, k: int) -> tuple[np.ndarray, list[str]]:
    msinfo = mat.get("msinfo")
    if msinfo is not None and hasattr(msinfo, "MSMaps") and len(np.atleast_1d(msinfo.MSMaps)) >= k:
        rec = np.atleast_1d(msinfo.MSMaps)[k - 1]
        maps = np.asarray(rec.Maps, dtype=float)
        if maps.shape[0] != k and maps.shape[1] == k:
            maps = maps.T
        labels = [str(x) for x in np.atleast_1d(getattr(rec, "Labels", [chr(65 + i) for i in range(k)]))[:k]]
        return maps[:k], labels
    data = np.asarray(mat["data"], dtype=float)
    all_maps = data.T if data.shape[0] > data.shape[1] else data
    start = 1 + sum(range(4, k))
    labels_by_k = {
        4: ["B", "C", "A", "D"],
        5: ["D", "C", "E", "B", "A"],
        6: ["E", "C", "A", "G", "D", "B"],
        7: ["D", "A", "C", "F", "B", "G", "E"],
    }
    return all_maps[start - 1 : start - 1 + k], labels_by_k.get(k, [chr(65 + i) for i in range(k)])


def template_xy_from_chanlocs(chanlocs: np.ndarray, n_channels: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    labels, xy, keep = [], [], []
    for idx, ch in enumerate(chanlocs[:n_channels]):
        label = str(getattr(ch, "labels", f"Ch{idx + 1}"))
        x = scalar_or_nan(getattr(ch, "X", np.nan))
        y = scalar_or_nan(getattr(ch, "Y", np.nan))
        if np.isfinite(x) and np.isfinite(y):
            point = [-y, x]
        else:
            point = None
        radius = scalar_or_nan(getattr(ch, "radius", np.nan))
        theta = scalar_or_nan(getattr(ch, "theta", np.nan))
        if point is None and np.isfinite(radius) and 0 < radius <= 0.5 and np.isfinite(theta):
            angle = np.deg2rad(theta)
            point = [-radius * np.cos(angle), -radius * np.sin(angle)]
        if point is None:
            continue
        labels.append(label)
        xy.append(point)
        keep.append(idx)
    xy = np.asarray(xy, dtype=float)
    radius = np.sqrt((xy**2).sum(axis=1))
    scale = np.nanmax(radius)
    if scale > 0:
        xy = xy / scale
    return labels, xy, np.asarray(keep, dtype=int)


def template_info_from_eeglab(
    template_file: Path, chanlocs: np.ndarray, n_channels: int
) -> tuple[list[str], np.ndarray, np.ndarray, object] | None:
    if mne is None:
        return None
    try:
        raw = mne.io.read_raw_eeglab(str(template_file), preload=False, verbose="ERROR")
        wanted = [str(getattr(ch, "labels", f"Ch{idx + 1}")) for idx, ch in enumerate(chanlocs[:n_channels])]
        raw_picks_by_name = {name: idx for idx, name in enumerate(raw.ch_names)}
        picks, keep = [], []
        for idx, name in enumerate(wanted):
            pick = raw_picks_by_name.get(name)
            if pick is None:
                continue
            loc = np.asarray(raw.info["chs"][pick]["loc"][:3], dtype=float)
            if np.isfinite(loc).all() and np.linalg.norm(loc) > 0:
                picks.append(pick)
                keep.append(idx)
        if len(picks) < 4:
            return None
        info = mne.pick_info(raw.info.copy(), picks, copy=True)
        xy = topomap_xy_from_info(info)
        if xy is None:
            return None
        return list(info.ch_names), xy, np.asarray(keep, dtype=int), info
    except Exception:
        return None


def topomap_xy_from_info(info: object) -> np.ndarray | None:
    if mne is None:
        return None
    try:
        from mne.channels.layout import _find_topomap_coords

        return _find_topomap_coords(info, picks="all")
    except Exception:
        return None


def scalar_or_nan(value: object) -> float:
    try:
        arr = np.asarray(value, dtype=float)
        return float(arr.ravel()[0])
    except Exception:
        return np.nan


def normalize_map_rows(values: np.ndarray) -> np.ndarray:
    out = values.astype(float).copy()
    for idx, row in enumerate(out):
        mask = np.isfinite(row)
        if mask.sum() < 2:
            continue
        row = row[mask] - np.mean(row[mask])
        denom = np.linalg.norm(row)
        out[idx, mask] = row / denom if denom > np.finfo(float).eps else row
    return out


def template_clim(template: TemplateTopographies | None) -> float:
    if template is None or template.maps.size == 0:
        return 1.0
    clim = float(np.nanmax(np.abs(template.maps)))
    return clim if np.isfinite(clim) and clim > 0 else 1.0


def render_template_topography(ax, label: str, template: TemplateTopographies | None, clim: float) -> None:
    ax.axis("off")
    if template is None or label not in template.labels:
        ax.text(0.5, 0.5, label, ha="center", va="center", fontweight="bold")
        return
    idx = template.labels.index(label)
    plot_topomap(ax, template.maps[idx], template.info if template.info is not None else template.xy, clim=clim)
    ax.set_title(label, fontsize=8, fontweight="bold", pad=0)


def plot_topomap(ax, values: np.ndarray, pos: object, *, clim: float | None = None) -> None:
    values = np.asarray(values, dtype=float)
    info = pos if mne is not None and isinstance(pos, mne.Info) else None
    ax.set_aspect("equal")
    ax.axis("off")
    if clim is None:
        finite = values[np.isfinite(values)]
        clim = float(np.nanmax(np.abs(finite))) if len(finite) else 1.0
        if not np.isfinite(clim) or clim <= 0:
            clim = 1.0
    if info is not None and values.size == len(info.ch_names) and np.isfinite(values).all():
        try:
            mne.viz.plot_topomap(
                values,
                info,
                axes=ax,
                show=False,
                sensors=False,
                contours=6,
                outlines="head",
                cmap="jet",
                vlim=(-clim, clim),
            )
            ax.axis("off")
            return
        except Exception:
            ax.clear()
            ax.set_aspect("equal")
            ax.axis("off")
    xy = topomap_xy_from_info(info) if info is not None else np.asarray(pos, dtype=float)
    if xy is None or xy.ndim != 2 or xy.shape[0] != values.size:
        xy = np.full((values.size, 2), np.nan, dtype=float)
    mask = np.isfinite(values) & np.isfinite(xy).all(axis=1)
    points = xy[mask]
    vals = values[mask]
    if mne is not None and len(vals) >= 4:
        try:
            mne.viz.plot_topomap(
                vals,
                points,
                axes=ax,
                show=False,
                sensors=False,
                contours=6,
                outlines="head",
                cmap="jet",
                vlim=(-clim, clim),
            )
            ax.axis("off")
            return
        except Exception:
            ax.clear()
            ax.set_aspect("equal")
            ax.axis("off")
    if len(vals) < 4:
        ax.scatter(points[:, 0], points[:, 1], c=vals, cmap="jet", vmin=-clim if clim else None, vmax=clim)
        return
    grid = np.linspace(-1.05, 1.05, 80)
    xx, yy = np.meshgrid(grid, grid)
    zi = griddata(points, vals, (xx, yy), method="cubic")
    if not np.isfinite(zi).any():
        zi = griddata(points, vals, (xx, yy), method="linear")
    if not np.isfinite(zi).any():
        zi = griddata(points, vals, (xx, yy), method="nearest")
    zi[np.sqrt(xx**2 + yy**2) > 1.0] = np.nan
    ax.imshow(zi, origin="lower", extent=[-1.05, 1.05, -1.05, 1.05], cmap="jet", vmin=-clim, vmax=clim)
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="black", linewidth=0.8))
    ax.plot([0, -0.08, 0.08, 0], [1.0, 1.12, 1.12, 1.0], color="black", linewidth=0.8)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.12, 1.15)


def sanitize_stub(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).lower())
    return re.sub(r"^_+|_+$", "", text) or "unnamed"


def _self_test() -> None:
    assert canonicalize_method("spm_vb") == "spm vb"
    assert canonicalize_method("kmeans_koenig") == "koenig kmeans"
    assert canonicalize_criterion("elbow_sil_combined") == "elbow sil combined"
    arr = np.array([[1.0, 1.0], [1.0, 3.0]])
    normed = normalize_map_rows(arr)
    assert np.allclose(normed[0], [0.0, 0.0])
    assert np.isclose(np.linalg.norm(normed[1]), 1.0)
    labels = merge_label_order(["A"], [["B", "A"], ["C"]])
    assert labels == ["A", "B", "C"]
    mat = reorder_matrix(np.eye(2), ["B", "A"], ["A", "B"], fill=0)
    assert np.allclose(mat, np.eye(2)[::-1, ::-1])
    rows = []
    for overlap in [0.0, 0.5]:
        for method, criterion, f1 in [
            ("koenig kmeans", "silhouette", 0.8 - 0.1 * overlap),
            ("spm vb", "free energy elbow", 0.9 - 0.05 * overlap),
        ]:
            rows.append({"method": method, "criterion_clean": criterion, "overlap_prob": overlap, "f1_score": f1})
    out = analyze_overlap(text_io.StringIO(), pd.DataFrame(rows), ["f1_score"], 10, np.random.default_rng(0))
    assert set(out["method"]) == {"koenig kmeans", "spm vb"}


def main() -> None:
    args = parse_args()
    if args.self_test:
        _self_test()
        print("self-test passed")
        return
    result = analyze_comparison_results(
        args.results_dir,
        results_csv=args.results_csv,
        output_dir=args.output_dir,
        template_file=args.template_file,
        n_boot=args.n_boot,
        n_folds=args.n_folds,
        seed=args.seed,
        max_confusion_pair_plots=args.max_confusion_pair_plots,
        run_mixed_models=not args.skip_mixed_models,
        run_confusion_outputs=not args.skip_confusion_outputs,
        run_microstate_map_plot=not args.skip_microstate_map_plot,
        run_confusion_pair_plots=not args.skip_confusion_pair_plots,
    )
    print(f"Statistics: {result['stats_file']}")
    print(f"Plots: {result['plots_dir']}")


if __name__ == "__main__":
    main()
