#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import os
import tempfile
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
K_CANDIDATE_CSV = RESULTS_DIR / "k_candidate_metrics.csv"
COMPARISON_CSV = RESULTS_DIR / "comparison_results.csv"
OUTPUT_HTML = RESULTS_DIR / "simulation_descriptive_results.html"

SCORE_COLUMNS = [
    "score_silhouette",
    "score_free_energy",
    "score_log_likelihood",
    "score_bic",
    "score_icl",
    "score_free_energy_elbow",
    "score_gev",
    "score_calinski_harabasz",
    "score_covariance",
    "score_covariance_elbow",
    "score_elbow_sil_combined",
]

OVERLAP_METHOD_CRITERIA = [
    ("kmeans_koenig", "silhouette"),
    ("spm_vb", "free_energy_elbow"),
]
OVERLAP_METRICS = [
    ("K_estimated", "Mean selected K", "count"),
    ("signed_K_error", "Signed K error", "count"),
    ("K_sq_error", "Squared K error", "count"),
    ("K_correct", "Exact K", "rate"),
    ("f1_score", "Recovered-state F1", "rate"),
    ("sensitivity", "Sensitivity", "rate"),
    ("precision", "Precision", "rate"),
]


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        print("self-test passed")
        return

    k_candidates = pd.read_csv(args.k_candidate_csv)
    comparisons = pd.read_csv(args.comparison_csv)

    k_candidates = add_signed_candidate_error(k_candidates)
    comparisons = add_signed_result_error(comparisons)
    criterion_summary = criterion_selection_summary(comparisons)
    overlap_summary = overlap_effect_summary(comparisons)

    figures = [
        ("VB-GMM Criterion Scores vs K", plot_scores(k_candidates, "K_candidate", "Attempted K")),
        (
            "VB-GMM Criterion Scores vs Attempted - Actual K",
            plot_scores(k_candidates, "candidate_signed_error", "Attempted K - actual K"),
        ),
        ("Average Signed K Error", plot_signed_k_error_heatmap(comparisons)),
        (
            "F1 Score by Method and Criterion",
            plot_group_boxplot(comparisons, "f1_score", higher_is_better=True, best_label_side="left"),
        ),
        (
            "Squared K Error by Method and Criterion",
            plot_group_boxplot(comparisons, "K_sq_error", higher_is_better=False, best_label_side="right"),
        ),
        ("SNR vs F1 Score", plot_snr_f1(comparisons)),
        ("True K vs F1 Score", plot_true_k_f1(comparisons)),
        ("Overlap Effects for Selected Methods", plot_overlap_effects(comparisons)),
    ]

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(render_html(figures, criterion_summary, overlap_summary, args), encoding="utf-8")
    print(args.output_html)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build descriptive simulation plots as a standalone HTML report.")
    parser.add_argument("--k-candidate-csv", type=Path, default=K_CANDIDATE_CSV)
    parser.add_argument("--comparison-csv", type=Path, default=COMPARISON_CSV)
    parser.add_argument("--output-html", type=Path, default=OUTPUT_HTML)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def add_signed_candidate_error(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["candidate_signed_error"] = pd.to_numeric(df["K_candidate"], errors="coerce") - pd.to_numeric(df["K_true"], errors="coerce")
    return df


def add_signed_result_error(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["signed_K_error"] = pd.to_numeric(df["K_estimated"], errors="coerce") - pd.to_numeric(df["K_true"], errors="coerce")
    df["K_abs_error"] = df["signed_K_error"].abs()
    df["K_sq_error"] = df["signed_K_error"].pow(2)
    df["under_selected"] = df["signed_K_error"] < 0
    df["over_selected"] = df["signed_K_error"] > 0
    df["missed_states"] = (pd.to_numeric(df["K_true"], errors="coerce") - pd.to_numeric(df["n_matched"], errors="coerce")).clip(lower=0)
    df["extra_states"] = (pd.to_numeric(df["K_estimated"], errors="coerce") - pd.to_numeric(df["n_matched"], errors="coerce")).clip(lower=0)
    df["method_label"] = df["method"].map(label)
    df["criterion_label"] = df["criterion"].map(label)
    df["comparison"] = df["method_label"] + " | " + df["criterion_label"]
    return df


def criterion_selection_summary(df: pd.DataFrame) -> str:
    spm = df[df["method"].astype(str).str.lower().eq("spm_vb")].copy()
    if spm.empty:
        return "<p>No VB-GMM rows found.</p>"
    rows = (
        spm.groupby("criterion_label", sort=False)
        .agg(
            rows=("fit_id", "size"),
            recovered_f1=("f1_score", "mean"),
            sensitivity=("sensitivity", "mean"),
            precision=("precision", "mean"),
            exact_k=("K_correct", "mean"),
            under_selected=("under_selected", "mean"),
            over_selected=("over_selected", "mean"),
            sq_k_error=("K_sq_error", "mean"),
            abs_k_error=("K_abs_error", "mean"),
            missed_states=("missed_states", "mean"),
            extra_states=("extra_states", "mean"),
            map_recovery=("mean_recovery_padded", "mean"),
        )
        .reset_index()
        .sort_values(["recovered_f1", "sq_k_error"], ascending=[False, True])
    )
    for col in ["recovered_f1", "sensitivity", "precision", "exact_k", "under_selected", "over_selected"]:
        rows[col] = rows[col].map(lambda x: f"{100 * x:.1f}%")
    for col in ["sq_k_error", "abs_k_error", "missed_states", "extra_states", "map_recovery"]:
        rows[col] = rows[col].map(lambda x: f"{x:.3f}")
    rows["rows"] = rows["rows"].map(lambda x: f"{int(x):,}")
    rows = rows.rename(
        columns={
            "criterion_label": "Criterion",
            "rows": "Rows",
            "recovered_f1": "Recovered-state F1",
            "sensitivity": "Sensitivity",
            "precision": "Precision",
            "exact_k": "Exact K",
            "under_selected": "Under-selected",
            "over_selected": "Over-selected",
            "sq_k_error": "Squared K error",
            "abs_k_error": "Abs K error",
            "missed_states": "Missed states",
            "extra_states": "Extra states",
            "map_recovery": "Penalised map recovery",
        }
    )
    return rows.to_html(index=False, escape=True)


def overlap_effect_summary(df: pd.DataFrame) -> str:
    rows = []
    for method, criterion in OVERLAP_METHOD_CRITERIA:
        sub = selected_method_rows(df, method, criterion)
        if sub.empty:
            continue
        summaries = []
        for overlap, group in sub.groupby("_overlap_prob", sort=True):
            summaries.append(overlap_metric_row(group, method, criterion, overlap, delta=False))
        rows.extend(format_overlap_row(row) for row in summaries)
        if len(summaries) >= 2:
            base, high = summaries[0], summaries[-1]
            delta = {
                "Method": high["Method"],
                "Overlap": f"{format_overlap(high['Overlap'])} - {format_overlap(base['Overlap'])}",
                "Rows": "",
                "_delta": True,
            }
            for _, col_label, _ in OVERLAP_METRICS:
                delta[col_label] = high[col_label] - base[col_label]
            rows.append(format_overlap_row(delta))
    if not rows:
        return "<p>No selected overlap rows found.</p>"
    return pd.DataFrame(rows).to_html(index=False, escape=True)


def selected_method_rows(df: pd.DataFrame, method: str, criterion: str) -> pd.DataFrame:
    sub = df[df["method"].eq(method) & df["criterion"].eq(criterion)].copy()
    if sub.empty or "overlap_prob" not in sub:
        return sub.iloc[0:0]
    sub["_overlap_prob"] = pd.to_numeric(sub["overlap_prob"], errors="coerce")
    return sub.dropna(subset=["_overlap_prob"])


def selected_best_model_rows(df: pd.DataFrame, *, require_overlap: bool = False) -> pd.DataFrame:
    frames = []
    for method, criterion in OVERLAP_METHOD_CRITERIA:
        sub = selected_method_rows(df, method, criterion) if require_overlap else df[df["method"].eq(method) & df["criterion"].eq(criterion)].copy()
        if not sub.empty:
            sub["comparison"] = f"{label(method)} | {label(criterion)}"
            frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else df.iloc[0:0].copy()


def overlap_metric_row(group: pd.DataFrame, method: str, criterion: str, overlap: float, *, delta: bool) -> dict[str, float | str | bool]:
    row: dict[str, float | str | bool] = {
        "Method": f"{label(method)} | {label(criterion)}",
        "Overlap": float(overlap),
        "Rows": len(group),
        "_delta": delta,
    }
    for col, col_label, _ in OVERLAP_METRICS:
        row[col_label] = pd.to_numeric(group[col], errors="coerce").mean() if col in group else np.nan
    return row


def format_overlap_row(row: dict[str, float | str | bool]) -> dict[str, str]:
    is_delta = bool(row.get("_delta"))
    out = {
        "Method": str(row["Method"]),
        "Overlap": str(row["Overlap"]) if is_delta else format_overlap(float(row["Overlap"])),
        "Rows": "" if is_delta else f"{int(row['Rows']):,}",
    }
    for _, col_label, kind in OVERLAP_METRICS:
        out[col_label] = format_overlap_value(float(row[col_label]), kind, is_delta)
    return out


def format_overlap(value: float) -> str:
    return f"{value:g}"


def format_overlap_value(value: float, kind: str, is_delta: bool) -> str:
    if not np.isfinite(value):
        return ""
    sign = "+" if is_delta and value >= 0 else ""
    if kind == "rate":
        suffix = " pp" if is_delta else "%"
        return f"{sign}{100 * value:.1f}{suffix}"
    return f"{sign}{value:.3f}"


def label(value: object) -> str:
    text = str(value).lower().strip()
    if text in {"bic", "score_bic"}:
        return "BIC"
    if text in {"icl", "score_icl"}:
        return "ICL"
    if text in {"ll", "log_likelihood", "score_log_likelihood"}:
        return "LL"
    if "spm" in text and "vb" in text:
        return "VB-GMM"
    if "kmeans" in text or "k-means" in text:
        return "K-Means"
    return str(value).replace("_", " ").replace("-", " ").title().replace("Spm", "SPM").replace("Vb", "VB")


def plot_scores(df: pd.DataFrame, x_col: str, x_label: str) -> str:
    spm = df[df["method"].astype(str).str.lower().eq("spm_vb")].copy()
    score_cols = [c for c in SCORE_COLUMNS if c in spm]
    ncols = 3
    nrows = int(np.ceil(len(score_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), squeeze=False)

    for ax, col in zip(axes.ravel(), score_cols):
        series = grouped_mean(spm, x_col, col)
        if not series.empty:
            ax.plot(series.index, series.values, marker="o", linewidth=1.8, markersize=4)
        if x_col == "candidate_signed_error":
            ax.axvline(0, color="0.75", linewidth=1)
        ax.set_title(label(col.replace("score_", "")))
        ax.set_xlabel(x_label)
        ax.set_ylabel("Mean score")
        ax.grid(True, alpha=0.25)

    for ax in axes.ravel()[len(score_cols) :]:
        ax.axis("off")
    fig.suptitle("VB-GMM candidate criterion scores", y=1.01)
    fig.tight_layout()
    return fig_to_img(fig)


def grouped_mean(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    sub = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    return sub.groupby(x_col)[y_col].mean().sort_index()


def plot_signed_k_error_heatmap(df: pd.DataFrame) -> str:
    best = selected_best_model_rows(df)
    if best.empty:
        best = df.iloc[0:0].copy()
    if "K_true" in best:
        best["_heatmap_col"] = pd.to_numeric(best["K_true"], errors="coerce")
        cols = sorted(best["_heatmap_col"].dropna().unique())
        col_labels = [f"K={int(c)}" if float(c).is_integer() else f"K={c:g}" for c in cols]
    else:
        best["_heatmap_col"] = "overall"
        cols = ["overall"]
        col_labels = ["overall"]
    rows = [f"{label(method)} | {label(criterion)}" for method, criterion in OVERLAP_METHOD_CRITERIA]
    heat = best.pivot_table(index="comparison", columns="_heatmap_col", values="signed_K_error", aggfunc="mean").reindex(index=rows, columns=cols)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    data = heat.to_numpy(float)
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1
    img = ax.imshow(data, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_title("Mean signed K error for selected best models")
    ax.set_xticks(np.arange(len(cols)), col_labels)
    ax.set_yticks(np.arange(len(heat.index)), heat.index)
    annotate_heatmap(ax, data)
    fig.colorbar(img, ax=ax, label="Mean signed K error")
    fig.tight_layout()
    return fig_to_img(fig)


def annotate_heatmap(ax: plt.Axes, data: np.ndarray) -> None:
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8)


def plot_group_boxplot(df: pd.DataFrame, metric: str, *, higher_is_better: bool, best_label_side: str) -> str:
    values_by_group = []
    summary = df.groupby("comparison", sort=False)[metric].mean().dropna()
    summary = summary.sort_values(ascending=not higher_is_better)
    groups = summary.index.tolist()

    for group in groups:
        values = pd.to_numeric(df.loc[df["comparison"].eq(group), metric], errors="coerce").dropna().to_numpy()
        values_by_group.append(values)

    best = groups[0] if groups else ""
    fig, ax = plt.subplots(figsize=(11, max(5, 0.42 * len(groups))))
    ax.boxplot(values_by_group, vert=False, tick_labels=groups, patch_artist=True, showfliers=False)
    ax.set_title(f"{label(metric)} by method and criterion")
    ax.set_xlabel(label(metric))
    ax.grid(True, axis="x", alpha=0.25)

    if groups:
        y = groups.index(best) + 1
        ax.scatter([summary.loc[best]], [y], color="#b00020", zorder=3)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if groups:
        fig.text(
            0.01 if best_label_side == "left" else 0.99,
            0.985,
            f"best: {best}",
            ha=best_label_side,
            va="top",
            color="#b00020",
            fontsize=9,
        )
    return fig_to_img(fig)


def plot_snr_f1(df: pd.DataFrame) -> str:
    return plot_f1_line(df, "SNR_dB", "SNR (dB)", "F1 score by SNR (95% CI)")


def plot_true_k_f1(df: pd.DataFrame) -> str:
    return plot_f1_line(df, "K_true", "True K", "F1 score by true K (95% CI)")


def plot_f1_line(df: pd.DataFrame, x_col: str, x_label: str, title: str) -> str:
    best_spm = selected_spm_criterion(df)
    lines = [
        ("spm_vb", best_spm, f"VB-GMM | {label(best_spm)}"),
        ("kmeans_koenig", "silhouette", "K-Means | Silhouette"),
    ]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for method, criterion, line_label in lines:
        mask = df["method"].eq(method) & df["criterion"].eq(criterion)
        series = grouped_mean_ci(df.loc[mask], x_col, "f1_score")
        ax.plot(series.index, series["mean"], marker="o", linewidth=2.2, label=line_label)
        ax.fill_between(series.index, series["low"], series["high"], alpha=0.16)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean F1 score")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig_to_img(fig)


def selected_spm_criterion(df: pd.DataFrame) -> str:
    spm = df[df["method"].astype(str).str.lower().eq("spm_vb")]
    if spm["criterion"].astype(str).eq("free_energy_elbow").any():
        return "free_energy_elbow"
    return spm.groupby("criterion", sort=False)["f1_score"].mean().sort_values(ascending=False).index[0]


def plot_overlap_effects(df: pd.DataFrame) -> str:
    data = selected_best_model_rows(df, require_overlap=True)
    metrics = [
        ("f1_score", "Recovered-state F1", "rate"),
        ("K_correct", "Exact K recovery", "rate"),
        ("K_sq_error", "Squared K error", "count"),
        ("signed_K_error", "Signed K error", "count"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), squeeze=False)
    for ax, (metric, title, kind) in zip(axes.ravel(), metrics):
        groups, labels = [], []
        for method, criterion in OVERLAP_METHOD_CRITERIA:
            comparison = f"{label(method)} | {label(criterion)}"
            sub = data[data["comparison"].eq(comparison)]
            if sub.empty or metric not in sub:
                continue
            for overlap in sorted(sub["_overlap_prob"].dropna().unique()):
                vals = pd.to_numeric(sub.loc[sub["_overlap_prob"].eq(overlap), metric], errors="coerce").dropna().to_numpy()
                if len(vals):
                    groups.append(vals)
                    labels.append(f"{comparison}\noverlap={format_overlap(float(overlap))}")
        if groups:
            ax.boxplot(groups, tick_labels=labels, showfliers=False)
            ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        if kind == "rate":
            ax.set_ylim(0, 1)
    fig.suptitle("Overlap effect: selected best models", y=1.01)
    fig.tight_layout()
    return fig_to_img(fig)


def grouped_mean_ci(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    sub = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    stats = sub.groupby(x_col)[y_col].agg(["mean", "std", "count"]).sort_index()
    ci = 1.96 * stats["std"].fillna(0) / np.sqrt(stats["count"])
    stats["low"] = (stats["mean"] - ci).clip(0, 1)
    stats["high"] = (stats["mean"] + ci).clip(0, 1)
    return stats


def fig_to_img(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_html(figures: list[tuple[str, str]], criterion_summary: str, overlap_summary: str, args: argparse.Namespace) -> str:
    sections = "\n".join(
        f"<section><h2>{escape(title)}</h2><img src=\"{src}\" alt=\"{escape(title)}\"></section>" for title, src in figures
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Simulation descriptive results</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    h1 {{ margin-bottom: 4px; }}
    .meta {{ color: #52616b; margin-top: 0; }}
    section {{ margin: 34px 0; }}
    h2 {{ margin-bottom: 12px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #d9e2ec; }}
  </style>
</head>
<body>
  <h1>Simulation descriptive results</h1>
  <p class="meta">Source files: {escape(str(args.k_candidate_csv))} and {escape(str(args.comparison_csv))}</p>
  <h2>VB-GMM Criterion Selection</h2>
  <p>Primary criterion choice uses recoverable-state F1, not literal exact K. Sensitivity penalises missed true states; precision penalises extra estimated states, so overestimating K is not rewarded. Squared K error, (estimated K - true K)<sup>2</sup>, is reported as the nonlinear K-count cost so large count misses are penalised more strongly than midpoint behaviour.</p>
  {criterion_summary}
  <h2>Overlap Effect: K-Means vs VB-GMM Free-Energy Elbow</h2>
  <p>Rows labelled 0.5 - 0 show the change when boundary-overlap probability increases from 0 to 0.5.</p>
  {overlap_summary}
  {sections}
</body>
</html>
"""


def self_test() -> None:
    rows = []
    for overlap, km_f1, spm_f1 in [(0.0, 0.8, 0.9), (0.5, 0.7, 0.85)]:
        for method, criterion, f1, k_est in [
            ("kmeans_koenig", "silhouette", km_f1, 4),
            ("spm_vb", "free_energy_elbow", spm_f1, 5),
        ]:
            rows.append(
                {
                    "fit_id": len(rows) + 1,
                    "method": method,
                    "criterion": criterion,
                    "K_true": 4,
                    "K_estimated": k_est,
                    "n_matched": 4,
                    "K_correct": int(k_est == 4),
                    "f1_score": f1,
                    "sensitivity": f1,
                    "precision": f1,
                    "overlap_prob": overlap,
                }
            )
    df = add_signed_result_error(pd.DataFrame(rows))
    html = overlap_effect_summary(df)
    assert "0.5 - 0" in html
    assert "VB-GMM | Free Energy Elbow" in html
    assert plot_overlap_effects(df).startswith("data:image/png;base64,")


if __name__ == "__main__":
    main()
