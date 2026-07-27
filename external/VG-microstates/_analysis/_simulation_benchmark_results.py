#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import os
import re
import tempfile
from dataclasses import dataclass
from html import escape
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "outputs" / "simulations" / "results"
COMPARISON_CSV = RESULTS_DIR / "comparison_results.csv"
OUTPUT_HTML = RESULTS_DIR / "simulation_benchmark_results.html"

SPM_LABEL = "VB-GMM"
KM_LABEL = "K-Means"
SPM_CRITERION = "free energy elbow"
KM_CRITERION = "silhouette"

PAIR_COLUMNS = [
    "rep",
    "K_true",
    "SNR_dB",
    "overlap_prob",
    "overlap_strength",
    "overlap_ms_min",
    "overlap_ms_max",
    "true_template_labels",
    "true_template_indices",
    "montage_type",
    "n_leads",
]

BLOCK_COLUMNS = [
    "rep",
    "K_true",
    "SNR_dB",
    "overlap_prob",
    "overlap_strength",
    "overlap_ms_min",
    "overlap_ms_max",
    "true_template_labels",
    "true_template_indices",
]


@dataclass(frozen=True)
class Metric:
    column: str
    label: str
    direction: str
    kind: str = "score"


METRICS = [
    Metric("K_correct", "Exact K recovery", "higher", "rate"),
    Metric("K_abs_error", "Absolute K error", "lower", "count"),
    Metric("K_sq_error", "Squared K error", "lower", "count"),
    Metric("f1_score", "Recovered-state F1", "higher", "rate"),
    Metric("sensitivity", "Sensitivity", "higher", "rate"),
    Metric("precision", "Precision", "higher", "rate"),
    Metric("mean_recovery_padded", "Map recovery, penalized", "higher", "score"),
    Metric("mean_recovery_matched", "Map recovery, matched", "higher", "score"),
    Metric("cluster_identity_accuracy", "Template identity accuracy", "higher", "rate"),
    Metric("backfit_mix_label_top1_accuracy", "Backfit label accuracy", "higher", "rate"),
    Metric("backfit_mix_label_weight_mae", "Backfit label-weight MAE", "lower", "score"),
    Metric("backfit_mix_label_pair_accuracy_overlap", "Overlap pair accuracy", "higher", "rate"),
    Metric("backfit_mix_label_weight_mae_overlap", "Overlap label-weight MAE", "lower", "score"),
    Metric("runtime_s", "Runtime", "lower", "seconds"),
]

PLOT_METRICS = {"K_sq_error", "f1_score", "sensitivity", "precision", "mean_recovery_padded", "runtime_s"}


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        print("self-test passed")
        return

    rng = np.random.default_rng(args.seed)
    table = prepare_table(pd.read_csv(args.comparison_csv))
    pairs = make_pairs(table, spm_criterion=args.spm_criterion)
    metric_summary = summarize_metrics(pairs, available_metrics(pairs), args.n_boot, args.n_perm, rng)
    criterion_summary = summarize_spm_criteria(table)

    figures = [
        ("VB-GMM Criterion Audit by Montage", plot_criterion_audit(criterion_summary, args.spm_criterion)),
        ("Paired Metric Means by Montage", plot_metric_means(metric_summary)),
        ("Within-Montage Improvement With 95% CI", plot_improvement_forest(metric_summary)),
        ("K Selection Confusion by Montage", plot_k_confusion(pairs)),
        ("Within-Montage SNR Robustness", plot_robustness(pairs)),
        ("Within-Montage Robustness Heatmaps", plot_delta_heatmaps(pairs)),
    ]

    html = render_html(
        metric_summary=metric_summary,
        criterion_summary=criterion_summary,
        figures=figures,
        args=args,
        n_pairs=len(pairs),
        montage_counts=montage_counts(pairs),
    )
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html, encoding="utf-8")
    print(args.output_html)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the paired VB-GMM vs K-Means simulation benchmark report.")
    parser.add_argument("--comparison-csv", type=Path, default=COMPARISON_CSV)
    parser.add_argument("--output-html", type=Path, default=OUTPUT_HTML)
    parser.add_argument("--spm-criterion", default=SPM_CRITERION)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def prepare_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = {"method", "criterion", *PAIR_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"comparison CSV is missing columns: {', '.join(missing)}")

    df["method_clean"] = df["method"].map(canonical_method)
    df["criterion_clean"] = df["criterion"].map(clean_token)
    if {"K_estimated", "K_true"}.issubset(df.columns):
        k_error = pd.to_numeric(df["K_estimated"], errors="coerce") - pd.to_numeric(df["K_true"], errors="coerce")
        df["K_abs_error"] = k_error.abs()
        df["K_sq_error"] = k_error.pow(2)
    elif "K_error" in df.columns:
        k_error = pd.to_numeric(df["K_error"], errors="coerce")
        df["K_abs_error"] = k_error.abs()
        df["K_sq_error"] = k_error.pow(2)

    for metric in METRICS:
        if metric.column in df.columns:
            df[metric.column] = pd.to_numeric(df[metric.column], errors="coerce")
    return df


def canonical_method(value: object) -> str:
    text = clean_token(value)
    if "spm" in text:
        return "spm"
    if "kmeans" in text or "k means" in text:
        return "kmeans"
    return text


def clean_token(value: object) -> str:
    text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
    text = text.lower().strip().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text)


def make_pairs(df: pd.DataFrame, *, spm_criterion: str) -> pd.DataFrame:
    spm_key = clean_token(spm_criterion)
    km_key = clean_token(KM_CRITERION)
    spm = df[(df["method_clean"].eq("spm")) & (df["criterion_clean"].eq(spm_key))].copy()
    km = df[(df["method_clean"].eq("kmeans")) & (df["criterion_clean"].eq(km_key))].copy()
    if spm.empty or km.empty:
        raise ValueError(f"Could not find both VB-GMM/{spm_criterion} and K-Means/{KM_CRITERION} rows.")

    metric_cols = sorted({m.column for m in METRICS if m.column in df.columns} | {"K_estimated"})
    cols = PAIR_COLUMNS + metric_cols
    spm = spm[cols].set_index(PAIR_COLUMNS).sort_index()
    km = km[cols].set_index(PAIR_COLUMNS).sort_index()
    if spm.index.has_duplicates or km.index.has_duplicates:
        raise ValueError("Pairing columns are not unique; add a simulation id before benchmarking.")

    common = spm.index.intersection(km.index)
    if common.empty:
        raise ValueError("No paired simulations found.")

    pairs = spm.loc[common].join(km.loc[common], lsuffix="_spm", rsuffix="_km").reset_index()
    if len(pairs) != len(spm) or len(pairs) != len(km):
        raise ValueError(f"Only {len(pairs)} paired rows found from {len(spm)} VB-GMM and {len(km)} K-Means rows.")
    return pairs


def available_metrics(pairs: pd.DataFrame) -> list[Metric]:
    out = []
    for metric in METRICS:
        spm_col = f"{metric.column}_spm"
        km_col = f"{metric.column}_km"
        if spm_col in pairs and km_col in pairs and paired_values(pairs, metric)[0].size:
            out.append(metric)
    return out


def summarize_metrics(
    pairs: pd.DataFrame, metrics: list[Metric], n_boot: int, n_perm: int, rng: np.random.Generator
) -> pd.DataFrame:
    rows = []
    for montage, group in sorted_montage_groups(pairs):
        n_leads = int(pd.to_numeric(group["n_leads"], errors="coerce").dropna().iloc[0])
        montage_label = format_montage(montage, n_leads)
        for metric in metrics:
            spm, km = paired_values(group, metric)
            block_improvement = block_improvement_values(group, metric)
            seed_improvement = grouped_improvement_values(group, metric, ["rep"])
            lo, hi = bootstrap_ci(block_improvement, n_boot, rng)
            p_value, p_floor = sign_flip_p_value(block_improvement, n_perm, rng)
            wins = int(np.sum(block_improvement > 1e-12))
            ties = int(np.sum(np.abs(block_improvement) <= 1e-12))
            losses = int(np.sum(block_improvement < -1e-12))
            rows.append(
                {
                    "montage": montage,
                    "montage_label": montage_label,
                    "n_leads": n_leads,
                    "metric": metric.column,
                    "label": metric.label,
                    "direction": "higher is better" if metric.direction == "higher" else "lower is better",
                    "kind": metric.kind,
                    "n": len(spm),
                    "block_n": len(block_improvement),
                    "spm_mean": float(np.mean(spm)),
                    "km_mean": float(np.mean(km)),
                    "improvement": float(np.mean(block_improvement)),
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_value": p_value,
                    "p_floor": p_floor,
                    "effect_dz": effect_dz(block_improvement),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "win_rate": wins / len(block_improvement) if len(block_improvement) else np.nan,
                    "seed_n": len(seed_improvement),
                    "seed_wins": int(np.sum(seed_improvement > 1e-12)),
                    "seed_ties": int(np.sum(np.abs(seed_improvement) <= 1e-12)),
                    "seed_losses": int(np.sum(seed_improvement < -1e-12)),
                }
            )
    return add_holm_p(pd.DataFrame(rows))


def paired_values(pairs: pd.DataFrame, metric: Metric) -> tuple[np.ndarray, np.ndarray]:
    spm = pd.to_numeric(pairs[f"{metric.column}_spm"], errors="coerce")
    km = pd.to_numeric(pairs[f"{metric.column}_km"], errors="coerce")
    keep = spm.notna() & km.notna()
    return spm[keep].to_numpy(float), km[keep].to_numpy(float)


def block_improvement_values(pairs: pd.DataFrame, metric: Metric) -> np.ndarray:
    return grouped_improvement_values(pairs, metric, BLOCK_COLUMNS)


def grouped_improvement_values(pairs: pd.DataFrame, metric: Metric, group_columns: list[str]) -> np.ndarray:
    spm = pd.to_numeric(pairs[f"{metric.column}_spm"], errors="coerce")
    km = pd.to_numeric(pairs[f"{metric.column}_km"], errors="coerce")
    improvement = spm - km if metric.direction == "higher" else km - spm
    tmp = pairs[group_columns].assign(_improvement=improvement).dropna(subset=["_improvement"])
    return tmp.groupby(group_columns, dropna=False)["_improvement"].mean().to_numpy(float)


def count_blocks(pairs: pd.DataFrame) -> int:
    return pairs.groupby(BLOCK_COLUMNS, dropna=False).ngroups


def sorted_montage_groups(df: pd.DataFrame):
    groups = []
    for montage, group in df.groupby("montage_type", sort=False, dropna=False):
        n_leads = pd.to_numeric(group["n_leads"], errors="coerce").dropna()
        groups.append((int(n_leads.iloc[0]) if not n_leads.empty else -1, str(montage), group))
    for _, montage, group in sorted(groups, key=lambda item: (-item[0], item[1])):
        yield montage, group


def format_montage(montage: object, n_leads: int | float | None = None) -> str:
    label = str(montage)
    if n_leads is None or not np.isfinite(float(n_leads)):
        return label
    return f"{label} ({int(n_leads)} ch)"


def montage_counts(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for montage, group in sorted_montage_groups(pairs):
        n_leads = pd.to_numeric(group["n_leads"], errors="coerce").dropna()
        n_leads_value = int(n_leads.iloc[0]) if not n_leads.empty else np.nan
        rows.append(
            {
                "Montage": format_montage(montage, n_leads_value),
                "Pairs": len(group),
                "EEG blocks": count_blocks(group),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if n_boot <= 0:
        return np.nan, np.nan
    samples = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
    return tuple(np.percentile(samples, [2.5, 97.5]))


def sign_flip_p_value(values: np.ndarray, n_perm: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2 or n_perm <= 0:
        return np.nan, np.nan
    observed = values.mean()
    exceed = 0
    done = 0
    chunk = 5000
    while done < n_perm:
        size = min(chunk, n_perm - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, values.size))
        null = signs.dot(values) / values.size
        exceed += int(np.sum(null >= observed - 1e-12))
        done += size
    floor = 1.0 / (n_perm + 1)
    return (exceed + 1.0) / (n_perm + 1), floor


def add_holm_p(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pvals = pd.to_numeric(df["p_value"], errors="coerce").to_numpy(float)
    adjusted = np.full(len(df), np.nan)
    finite = np.flatnonzero(np.isfinite(pvals))
    order = finite[np.argsort(pvals[finite])]
    running = 0.0
    total = len(order)
    for rank, idx in enumerate(order):
        running = max(running, (total - rank) * pvals[idx])
        adjusted[idx] = min(running, 1.0)
    df["p_holm"] = adjusted
    return df


def effect_dz(improvement: np.ndarray) -> float:
    improvement = np.asarray(improvement, dtype=float)
    improvement = improvement[np.isfinite(improvement)]
    sd = improvement.std(ddof=1)
    return float(improvement.mean() / sd) if sd > 0 else np.nan


def summarize_spm_criteria(df: pd.DataFrame) -> pd.DataFrame:
    spm = df[df["method_clean"].eq("spm")].copy()
    if spm.empty:
        return pd.DataFrame()
    spm["K_signed_error"] = pd.to_numeric(spm["K_estimated"], errors="coerce") - pd.to_numeric(spm["K_true"], errors="coerce")
    spm["under_selected"] = spm["K_signed_error"] < 0
    spm["over_selected"] = spm["K_signed_error"] > 0
    spm["missed_states"] = (pd.to_numeric(spm["K_true"], errors="coerce") - pd.to_numeric(spm["n_matched"], errors="coerce")).clip(lower=0)
    spm["extra_states"] = (pd.to_numeric(spm["K_estimated"], errors="coerce") - pd.to_numeric(spm["n_matched"], errors="coerce")).clip(lower=0)
    rows = []
    for montage, montage_group in sorted_montage_groups(spm):
        n_leads = int(pd.to_numeric(montage_group["n_leads"], errors="coerce").dropna().iloc[0])
        for criterion, group in montage_group.groupby("criterion_clean", sort=False):
            rows.append(
                {
                    "montage": montage,
                    "montage_label": format_montage(montage, n_leads),
                    "criterion": criterion,
                    "label": title_token(criterion),
                    "selected": criterion == clean_token(SPM_CRITERION),
                    "n": len(group),
                    "K_accuracy": group["K_correct"].mean() if "K_correct" in group else np.nan,
                    "K_abs_error": group["K_abs_error"].mean() if "K_abs_error" in group else np.nan,
                    "K_sq_error": group["K_sq_error"].mean() if "K_sq_error" in group else np.nan,
                    "K_signed_error": group["K_signed_error"].mean(),
                    "under_selection_rate": group["under_selected"].mean(),
                    "over_selection_rate": group["over_selected"].mean(),
                    "missed_states": group["missed_states"].mean(),
                    "extra_states": group["extra_states"].mean(),
                    "sensitivity": group["sensitivity"].mean() if "sensitivity" in group else np.nan,
                    "precision": group["precision"].mean() if "precision" in group else np.nan,
                    "F1": group["f1_score"].mean() if "f1_score" in group else np.nan,
                    "backfit_label_accuracy": group["backfit_mix_label_top1_accuracy"].mean()
                    if "backfit_mix_label_top1_accuracy" in group
                    else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["montage_label", "F1", "K_sq_error"], ascending=[True, False, True])


def title_token(value: object) -> str:
    text = clean_token(value)
    return text.title().replace("Spm", "SPM").replace("F1", "F1")


def plot_criterion_audit(summary: pd.DataFrame, spm_criterion: str) -> str:
    if summary.empty:
        fig, _ = plt.subplots(figsize=(8, 4))
        return fig_to_img(fig)

    montages = summary["montage_label"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(montages), figsize=(5.2 * len(montages), 5.6), sharey=True, squeeze=False)
    for ax, montage in zip(axes.ravel(), montages):
        sub = summary[summary["montage_label"].eq(montage)].sort_values(["F1", "K_sq_error"], ascending=[False, True])
        labels = sub["label"].tolist()
        y = np.arange(len(sub))
        colors = np.where(sub["criterion"].eq(clean_token(spm_criterion)), "#155e75", "#8aa4ad")
        ax.barh(y, sub["F1"], color=colors)
        ax.set_title(montage)
        ax.set_xlabel("Recovered-state F1")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_yticks(y, labels)
        ax.grid(True, axis="x", alpha=0.25)
    fig.suptitle(f"VB-GMM criterion check by recoverable-state F1; highlighted = {title_token(spm_criterion)}", fontweight="bold")
    fig.tight_layout()
    return fig_to_img(fig)


def plot_metric_means(summary: pd.DataFrame) -> str:
    summary = summary[summary["metric"].isin(PLOT_METRICS)].copy()
    metric_rows = summary.drop_duplicates("metric").to_dict("records")
    ncols = 3
    nrows = int(np.ceil(len(metric_rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows), squeeze=False)
    for ax, metric_row in zip(axes.ravel(), metric_rows):
        sub = summary[summary["metric"].eq(metric_row["metric"])]
        x = np.arange(len(sub))
        width = 0.38
        ax.bar(x - width / 2, sub["km_mean"], color="#9ca3af", width=width, label=KM_LABEL)
        ax.bar(x + width / 2, sub["spm_mean"], color="#155e75", width=width, label=SPM_LABEL)
        ax.set_xticks(x, sub["montage_label"], rotation=25, ha="right")
        ax.set_title(f"{metric_row['label']} ({metric_row['direction']})")
        ax.grid(True, axis="y", alpha=0.25)
        if metric_row["kind"] == "rate":
            ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8)
    hide_unused_axes(axes, len(metric_rows))
    fig.suptitle("Paired means within each montage", fontweight="bold")
    fig.tight_layout()
    return fig_to_img(fig)


def plot_improvement_forest(summary: pd.DataFrame) -> str:
    summary = summary[summary["metric"].isin(PLOT_METRICS)].copy()
    metric_rows = summary.drop_duplicates("metric").to_dict("records")
    fig, axes = plt.subplots(len(metric_rows), 1, figsize=(10, max(6, 1.35 * len(metric_rows))), squeeze=False)
    for ax, metric_row in zip(axes.ravel(), metric_rows):
        sub = summary[summary["metric"].eq(metric_row["metric"])]
        y = np.arange(len(sub))
        ax.axvline(0, color="0.35", linewidth=1)
        ax.errorbar(
            sub["improvement"],
            y,
            xerr=[sub["improvement"] - sub["ci_low"], sub["ci_high"] - sub["improvement"]],
            fmt="o",
            color="#155e75",
            capsize=4,
        )
        pad = max(abs(sub["ci_low"]).max(), abs(sub["ci_high"]).max(), abs(sub["improvement"]).max(), 1e-6) * 1.2
        ax.set_xlim(-pad, pad)
        ax.set_yticks(y, sub["montage_label"])
        ax.set_title(f"{metric_row['label']}: + means {SPM_LABEL} better", loc="left", fontsize=10)
        ax.grid(True, axis="x", alpha=0.18)
        for row_y, row in zip(y, sub.to_dict("records")):
            ax.text(
                0.99,
                row_y,
                f"p={format_p(row['p_value'], row['p_floor'])}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=8,
            )
    fig.suptitle("Within-montage improvement with block-bootstrap 95% confidence intervals", fontweight="bold")
    fig.tight_layout()
    return fig_to_img(fig)


def plot_k_confusion(pairs: pd.DataFrame) -> str:
    montage_groups = list(sorted_montage_groups(pairs))
    fig, axes = plt.subplots(len(montage_groups), 2, figsize=(12, 4.2 * len(montage_groups)), sharey=True, squeeze=False)
    for row, (montage, group) in enumerate(montage_groups):
        n_leads = pd.to_numeric(group["n_leads"], errors="coerce").dropna().iloc[0]
        for col, (suffix, title) in enumerate([("km", KM_LABEL), ("spm", SPM_LABEL)]):
            ax = axes[row, col]
            true = pd.to_numeric(group["K_true"], errors="coerce")
            est = pd.to_numeric(group[f"K_estimated_{suffix}"], errors="coerce")
            tab = pd.crosstab(true, est, normalize="index").sort_index(axis=0).sort_index(axis=1)
            data = tab.to_numpy(float)
            img = ax.imshow(data, cmap="Blues", vmin=0, vmax=max(0.01, np.nanmax(data)), aspect="auto")
            ax.set_title(f"{format_montage(montage, n_leads)} | {title}")
            ax.set_xlabel("Estimated K")
            ax.set_xticks(np.arange(len(tab.columns)), [int(x) for x in tab.columns])
            ax.set_yticks(np.arange(len(tab.index)), [int(x) for x in tab.index])
            ax.set_ylabel("True K")
            for r in range(data.shape[0]):
                for c in range(data.shape[1]):
                    if data[r, c] >= 0.05:
                        ax.text(c, r, f"{100 * data[r, c]:.0f}%", ha="center", va="center", fontsize=8)
            fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04, label="Row percent")
    fig.suptitle("K-selection confusion matrices by montage", fontweight="bold")
    fig.tight_layout()
    return fig_to_img(fig)


def plot_robustness(pairs: pd.DataFrame) -> str:
    montage_groups = list(sorted_montage_groups(pairs))
    metrics = [
        Metric("f1_score", "Recovered-state F1", "higher", "rate"),
        Metric("K_sq_error", "Squared K error", "lower", "count"),
    ]
    fig, axes = plt.subplots(len(metrics), len(montage_groups), figsize=(15, 7), squeeze=False)
    for r, metric in enumerate(metrics):
        for c, (montage, group) in enumerate(montage_groups):
            n_leads = pd.to_numeric(group["n_leads"], errors="coerce").dropna().iloc[0]
            plot_by_stratum(axes[r, c], group, "SNR_dB", "SNR (dB)", metric)
            axes[r, c].set_title(f"{format_montage(montage, n_leads)} | {metric.label}")
    fig.suptitle("SNR robustness within each montage", fontweight="bold")
    fig.tight_layout()
    return fig_to_img(fig)


def plot_by_stratum(ax: plt.Axes, pairs: pd.DataFrame, column: str, x_label: str, metric: Metric) -> None:
    rows = []
    for value, group in pairs.groupby(column, sort=True, dropna=False):
        rows.append(
            {
                "x": value,
                KM_LABEL: pd.to_numeric(group[f"{metric.column}_km"], errors="coerce").mean(),
                SPM_LABEL: pd.to_numeric(group[f"{metric.column}_spm"], errors="coerce").mean(),
            }
        )
    data = pd.DataFrame(rows)
    x = np.arange(len(data))
    if pd.api.types.is_numeric_dtype(data["x"]):
        ax.plot(data["x"], data[KM_LABEL], marker="o", label=KM_LABEL, color="#6b7280")
        ax.plot(data["x"], data[SPM_LABEL], marker="o", label=SPM_LABEL, color="#155e75")
    else:
        width = 0.38
        ax.bar(x - width / 2, data[KM_LABEL], width=width, label=KM_LABEL, color="#9ca3af")
        ax.bar(x + width / 2, data[SPM_LABEL], width=width, label=SPM_LABEL, color="#155e75")
        ax.set_xticks(x, data["x"].astype(str), rotation=25, ha="right")
    ax.set_title(metric.label)
    ax.set_xlabel(x_label)
    ax.grid(True, alpha=0.22)
    if metric.kind == "rate":
        ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)


def plot_delta_heatmaps(pairs: pd.DataFrame) -> str:
    montage_groups = list(sorted_montage_groups(pairs))
    metrics = [
        Metric("f1_score", "F1 improvement", "higher", "rate"),
        Metric("K_sq_error", "Squared K-error reduction", "lower", "count"),
    ]
    fig, axes = plt.subplots(len(metrics), len(montage_groups), figsize=(15, 7.5), sharey="row", squeeze=False)
    for r, metric in enumerate(metrics):
        for c, (montage, group) in enumerate(montage_groups):
            n_leads = pd.to_numeric(group["n_leads"], errors="coerce").dropna().iloc[0]
            heatmap_delta(axes[r, c], group, metric)
            axes[r, c].set_title(f"{format_montage(montage, n_leads)} | {metric.label}")
    fig.suptitle(f"Within-montage cells; positive values favor {SPM_LABEL}", fontweight="bold")
    fig.tight_layout()
    return fig_to_img(fig)


def heatmap_delta(ax: plt.Axes, pairs: pd.DataFrame, metric: Metric) -> None:
    spm = pd.to_numeric(pairs[f"{metric.column}_spm"], errors="coerce")
    km = pd.to_numeric(pairs[f"{metric.column}_km"], errors="coerce")
    improvement = spm - km if metric.direction == "higher" else km - spm
    tmp = pairs.assign(_improvement=improvement)
    heat = tmp.pivot_table(index="K_true", columns="SNR_dB", values="_improvement", aggfunc="mean").sort_index()
    data = heat.to_numpy(float)
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1
    img = ax.imshow(data, cmap="PiYG", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(metric.label)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("True K")
    ax.set_xticks(np.arange(len(heat.columns)), [str(x) for x in heat.columns])
    ax.set_yticks(np.arange(len(heat.index)), [int(x) for x in heat.index])
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            if np.isfinite(data[r, c]):
                ax.text(c, r, f"{data[r, c]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)


def render_html(
    *,
    metric_summary: pd.DataFrame,
    criterion_summary: pd.DataFrame,
    figures: list[tuple[str, str]],
    args: argparse.Namespace,
    n_pairs: int,
    montage_counts: pd.DataFrame,
) -> str:
    cards = summary_cards(metric_summary, n_pairs, montage_counts)
    metric_table = metric_summary_table(metric_summary)
    criterion_table = criterion_summary_table(criterion_summary)
    assumptions = assumption_table(n_pairs, montage_counts, args)
    figure_html = "\n".join(
        f'<section><h2>{escape(title)}</h2><img src="{src}" alt="{escape(title)}"></section>' for title, src in figures
    )
    criterion_note = spm_criterion_note(criterion_summary, args.spm_criterion)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>VB-GMM simulation benchmark</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    h1 {{ margin-bottom: 4px; }}
    h2 {{ margin-top: 34px; }}
    .meta {{ color: #52616b; margin-top: 0; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 24px 0; }}
    .card {{ border: 1px solid #d9e2ec; padding: 14px; border-radius: 6px; }}
    .card b {{ display: block; font-size: 22px; margin-bottom: 4px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child, td:nth-child(2) {{ text-align: left; }}
    th {{ background: #f3f6f8; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #d9e2ec; }}
    li {{ margin: 5px 0; }}
  </style>
</head>
<body>
  <h1>VB-GMM simulation benchmark</h1>
  <p class="meta">Source file: {escape(str(args.comparison_csv))}. Paired comparison: {SPM_LABEL} using {escape(title_token(args.spm_criterion))} vs {KM_LABEL} using silhouette, reported separately by montage.</p>
  <div class="cards">{cards}</div>
  <h2>Benchmarking Covered</h2>
  <ul>
    <li>Paired tests within each montage over the same simulated runs, so each delta is within-run.</li>
    <li>Primary criterion choice is based on recoverable-state F1: sensitivity penalises missed true states and precision penalises extra estimated states.</li>
    <li>Squared K error, (estimated K - true K)<sup>2</sup>, is the nonlinear K-count cost used for K-error reporting and criterion tie-breaks.</li>
    <li>Literal exact-K recovery is retained as a diagnostic because true generated K can exceed the recoverable structure in noisy or overlapping EEG.</li>
    <li>K selection, map recovery, template-label identity, backfit accuracy, overlap backfit, and runtime context.</li>
    <li>Robustness by SNR and true K is shown within montage only.</li>
    <li>No inferential claim pools full and limited montages.</li>
    <li>Within-montage block-bootstrap 95% confidence intervals plus one-sided sign-flip permutation tests where positive improvement favors {SPM_LABEL}.</li>
  </ul>
  <h2>Assumption Checks</h2>
  {assumptions}
  <p>{escape(criterion_note)}</p>
  <h2>Paired Statistical Summary</h2>
  {metric_table}
  <h2>VB-GMM Criterion Summary</h2>
  {criterion_table}
  {figure_html}
</body>
</html>
"""


def assumption_table(n_pairs: int, counts: pd.DataFrame, args: argparse.Namespace) -> str:
    counts_text = "; ".join(
        f"{row['Montage']}: {int(row['EEG blocks']):,} blocks" for row in counts.to_dict("records")
    )
    rows = [
        {
            "Check": "Pairing",
            "Result": f"{n_pairs:,} matched method pairs compare the same simulated data.",
        },
        {
            "Check": "Independent unit for inference",
            "Result": f"Generated EEG block within montage. Counts: {counts_text}.",
        },
        {
            "Check": "Montage scope",
            "Result": "Full and limited montages are not averaged or tested as one population.",
        },
        {
            "Check": "Parametric assumptions",
            "Result": "No t-tests or normal-error models are used for the main p-values.",
        },
        {
            "Check": "Distribution shape",
            "Result": "Discrete, tied, and skewed deltas are handled by sign-flip permutation on block means.",
        },
        {
            "Check": "Uncertainty",
            "Result": f"95% confidence intervals are nonparametric bootstrap intervals over EEG blocks ({args.n_boot:,} resamples).",
        },
        {
            "Check": "Multiple metrics",
            "Result": "Holm-adjusted permutation p-values are reported across the metric table.",
        },
        {
            "Check": "Replicate-seed sensitivity",
            "Result": "The metric table also shows win/tie/loss after collapsing all conditions to the 10 replicate seeds.",
        },
        {
            "Check": "Permutation resolution",
            "Result": f"{args.n_perm:,} random sign flips; p-values below the Monte Carlo floor are displayed as a threshold.",
        },
    ]
    return pd.DataFrame(rows).to_html(index=False, escape=True)


def summary_cards(summary: pd.DataFrame, n_pairs: int, counts: pd.DataFrame) -> str:
    rows = [("Paired rows", f"{n_pairs:,}"), ("Montages reported separately", f"{len(counts):,}")]
    for row in counts.to_dict("records"):
        rows.append((f"{row['Montage']} EEG blocks", f"{int(row['EEG blocks']):,}"))
    return "".join(f"<div class=\"card\"><b>{escape(value)}</b>{escape(label)}</div>" for label, value in rows)


def metric_summary_table(summary: pd.DataFrame) -> str:
    rows = []
    for row in summary.to_dict("records"):
        rows.append(
            {
                "Montage": row["montage_label"],
                "Metric": row["label"],
                "Direction": row["direction"],
                "Rows": f"{int(row['n']):,}",
                "Blocks": f"{int(row['block_n']):,}",
                KM_LABEL: format_value(row["km_mean"], row["kind"]),
                SPM_LABEL: format_value(row["spm_mean"], row["kind"]),
                "Improvement": format_signed(row["improvement"], row["kind"]),
                "95% CI": f"{format_signed(row['ci_low'], row['kind'])} to {format_signed(row['ci_high'], row['kind'])}",
                "Permutation p": format_p(row["p_value"], row["p_floor"]),
                "Holm p": format_p(row["p_holm"]),
                "dz": f"{row['effect_dz']:.2f}" if np.isfinite(row["effect_dz"]) else "",
                "Block W/T/L": f"{row['wins']}/{row['ties']}/{row['losses']}",
                "Seed W/T/L": f"{row['seed_wins']}/{row['seed_ties']}/{row['seed_losses']}",
            }
        )
    return pd.DataFrame(rows).to_html(index=False, escape=True)


def criterion_summary_table(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "<p>No VB-GMM criterion rows found.</p>"
    rows = summary.copy()
    rows["Criterion"] = rows["label"] + np.where(rows["selected"], " (selected)", "")
    rows["Montage"] = rows["montage_label"]
    rows["Recovered-state F1"] = rows["F1"].map(lambda x: format_value(x, "rate"))
    rows["Sensitivity"] = rows["sensitivity"].map(lambda x: format_value(x, "rate"))
    rows["Precision"] = rows["precision"].map(lambda x: format_value(x, "rate"))
    rows["Exact K recovery"] = rows["K_accuracy"].map(lambda x: format_value(x, "rate"))
    rows["Under-selected"] = rows["under_selection_rate"].map(lambda x: format_value(x, "rate"))
    rows["Over-selected"] = rows["over_selection_rate"].map(lambda x: format_value(x, "rate"))
    rows["Squared K error"] = rows["K_sq_error"].map(lambda x: format_value(x, "count"))
    rows["Absolute K error"] = rows["K_abs_error"].map(lambda x: format_value(x, "count"))
    rows["Backfit label accuracy"] = rows["backfit_label_accuracy"].map(lambda x: format_value(x, "rate"))
    rows["N"] = rows["n"].map(lambda x: f"{int(x):,}")
    return rows[
        [
            "Montage",
            "Criterion",
            "N",
            "Recovered-state F1",
            "Sensitivity",
            "Precision",
            "Exact K recovery",
            "Under-selected",
            "Over-selected",
            "Squared K error",
            "Absolute K error",
            "Backfit label accuracy",
        ]
    ].to_html(index=False, escape=True)


def spm_criterion_note(summary: pd.DataFrame, spm_criterion: str) -> str:
    if summary.empty:
        return ""
    notes = []
    label = title_token(spm_criterion)
    for montage, group in summary.groupby("montage_label", sort=False):
        selected = group[group["criterion"].eq(clean_token(spm_criterion))]
        if selected.empty:
            notes.append(f"{montage}: {label} was not found.")
            continue
        best_f1 = group["F1"].max()
        selected_f1 = float(selected["F1"].iloc[0])
        if np.isclose(selected_f1, best_f1):
            tied = group[np.isclose(group["F1"], best_f1)]
            best_cost = tied["K_sq_error"].min()
            selected_cost = float(selected["K_sq_error"].iloc[0])
            if np.isclose(selected_cost, best_cost):
                notes.append(
                    f"{montage}: {label} is tied for the best recoverable-state F1 and squared K-error cost."
                )
            else:
                notes.append(
                    f"{montage}: {label} is tied for the best recoverable-state F1; squared K-error cost is {selected_cost:.3f}."
                )
        else:
            rank = int((group["F1"] > selected_f1).sum() + 1)
            notes.append(f"{montage}: {label} ranks {rank} by recoverable-state F1.")
    return " ".join(notes)


def format_value(value: float, kind: str) -> str:
    if not np.isfinite(value):
        return ""
    if kind == "rate":
        return f"{100 * value:.1f}%"
    if kind == "seconds":
        return f"{value:.1f}s"
    return f"{value:.3f}" if abs(value) < 10 else f"{value:.2f}"


def format_signed(value: float, kind: str) -> str:
    if not np.isfinite(value):
        return ""
    sign = "+" if value >= 0 else ""
    if kind == "rate":
        return f"{sign}{100 * value:.1f} pp"
    if kind == "seconds":
        return f"{sign}{value:.1f}s"
    return f"{sign}{value:.3f}" if abs(value) < 10 else f"{sign}{value:.2f}"


def format_p(value: float, floor: float | None = None) -> str:
    if not np.isfinite(value):
        return ""
    if floor is not None and np.isfinite(floor) and value <= floor + 1e-15:
        return f"<{floor:.1e}"
    if value == 0:
        return "<1e-300"
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def fig_to_img(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def hide_unused_axes(axes: np.ndarray, used: int) -> None:
    for ax in axes.ravel()[used:]:
        ax.axis("off")


def self_test() -> None:
    rows = []
    for rep in [1, 2]:
        for method, criterion, k_est, f1 in [
            ("spm_vb", SPM_CRITERION, 4, 0.9),
            ("kmeans_koenig", "silhouette", 3, 0.7),
        ]:
            rows.append(
                {
                    "rep": rep,
                    "K_true": 4,
                    "SNR_dB": 0,
                    "overlap_prob": 0,
                    "overlap_strength": 0.5,
                    "overlap_ms_min": 10,
                    "overlap_ms_max": 40,
                    "true_template_labels": "A|B|C|D",
                    "true_template_indices": "[1 2 3 4]",
                    "montage_type": "full",
                    "n_leads": 71,
                    "method": method,
                    "criterion": criterion,
                    "K_estimated": k_est,
                    "K_correct": int(k_est == 4),
                    "f1_score": f1,
                }
            )
    table = prepare_table(pd.DataFrame(rows))
    pairs = make_pairs(table, spm_criterion=SPM_CRITERION)
    assert len(pairs) == 2
    summary = summarize_metrics(pairs, available_metrics(pairs), 20, 20, np.random.default_rng(0))
    assert summary.loc[summary["metric"].eq("f1_score"), "improvement"].iloc[0] > 0
    html = metric_summary_table(summary)
    assert "Koenig" not in html


if __name__ == "__main__":
    main()
