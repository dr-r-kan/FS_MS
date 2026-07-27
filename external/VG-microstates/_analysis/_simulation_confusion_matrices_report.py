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
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "outputs" / "simulations" / "results"
COMPARISON_CSV = RESULTS_DIR / "comparison_results.csv"
OUTPUT_HTML = RESULTS_DIR / "simulation_confusion_matrices_report.html"


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        print("self-test passed")
        return

    df = prepare_table(pd.read_csv(args.comparison_csv))
    true_levels = sorted(df["K_true"].dropna().astype(int).unique())
    est_levels = sorted(df["K_estimated"].dropna().astype(int).unique())
    figures = [
        (group_title(method, criterion), plot_group_confusions(group, true_levels, est_levels))
        for (method, criterion), group in method_criterion_groups(df)
    ]

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(render_html(args, summary_table(df), figures), encoding="utf-8")
    print(args.output_html)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build K-selection confusion matrices for every simulation criterion.")
    parser.add_argument("--comparison-csv", type=Path, default=COMPARISON_CSV)
    parser.add_argument("--output-html", type=Path, default=OUTPUT_HTML)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def prepare_table(df: pd.DataFrame) -> pd.DataFrame:
    required = {"method", "criterion", "K_true", "K_estimated", "montage_type", "n_leads"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"comparison CSV is missing columns: {', '.join(missing)}")

    df = df.copy()
    df["K_true"] = pd.to_numeric(df["K_true"], errors="coerce")
    df["K_estimated"] = pd.to_numeric(df["K_estimated"], errors="coerce")
    df = df.dropna(subset=["K_true", "K_estimated"])
    df["K_error"] = df["K_estimated"] - df["K_true"]
    df["K_correct"] = df["K_error"].eq(0)
    df["method_label"] = df["method"].map(label)
    df["criterion_label"] = df["criterion"].map(label)
    return df


def method_criterion_groups(df: pd.DataFrame):
    groups = list(df.groupby(["method", "criterion"], sort=False))
    return sorted(groups, key=lambda item: (method_order(item[0][0]), label(item[0][1])))


def method_order(method: object) -> int:
    text = str(method).lower()
    if "kmeans" in text or "k-means" in text:
        return 0
    if "spm" in text:
        return 1
    return 2


def label(value: object) -> str:
    text = str(value).lower().strip()
    known = {
        "bic": "BIC",
        "icl": "ICL",
        "gev": "GEV",
        "log_likelihood": "Log likelihood",
        "calinski_harabasz_score": "Calinski-Harabasz",
        "spm_vb": "VB-GMM",
        "kmeans_koenig": "K-Means",
    }
    return known.get(text, str(value).replace("_", " ").replace("-", " ").title())


def group_title(method: object, criterion: object) -> str:
    return f"{label(method)} | {label(criterion)}"


def sorted_montage_groups(df: pd.DataFrame):
    groups = list(df.groupby(["montage_type", "n_leads"], sort=False))
    return sorted(groups, key=lambda item: (-int(item[0][1]), str(item[0][0])))


def confusion_counts(df: pd.DataFrame, true_levels: list[int], est_levels: list[int]) -> pd.DataFrame:
    counts = pd.crosstab(df["K_true"].astype(int), df["K_estimated"].astype(int))
    return counts.reindex(index=true_levels, columns=est_levels, fill_value=0)


def row_rates(counts: pd.DataFrame) -> pd.DataFrame:
    return counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


def plot_group_confusions(df: pd.DataFrame, true_levels: list[int], est_levels: list[int]) -> str:
    panels = [("All montages", df)] + [
        (f"{montage} ({int(n_leads)} ch)", group) for (montage, n_leads), group in sorted_montage_groups(df)
    ]
    ncols = 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows), squeeze=False, constrained_layout=True)
    used_axes = []

    for ax, (title, group) in zip(axes.ravel(), panels):
        used_axes.append(ax)
        counts = confusion_counts(group, true_levels, est_levels)
        rates = row_rates(counts)
        img = ax.imshow(rates.to_numpy(float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"{title}; n={len(group):,}")
        ax.set_xlabel("Estimated K")
        ax.set_ylabel("True K")
        ax.set_xticks(np.arange(len(est_levels)), est_levels)
        ax.set_yticks(np.arange(len(true_levels)), true_levels)
        annotate(ax, rates.to_numpy(float))
        outline_exact_k(ax, true_levels, est_levels)

    for ax in axes.ravel()[len(panels) :]:
        ax.axis("off")

    fig.colorbar(img, ax=used_axes, fraction=0.025, pad=0.02, label="Row percent")
    fig.suptitle(group_title(df["method"].iloc[0], df["criterion"].iloc[0]), fontweight="bold")
    return fig_to_img(fig)


def annotate(ax: plt.Axes, rates: np.ndarray) -> None:
    for row in range(rates.shape[0]):
        for col in range(rates.shape[1]):
            value = rates[row, col]
            if value >= 0.03:
                ax.text(col, row, f"{100 * value:.0f}%", ha="center", va="center", fontsize=8)


def outline_exact_k(ax: plt.Axes, true_levels: list[int], est_levels: list[int]) -> None:
    for row, k_true in enumerate(true_levels):
        if k_true in est_levels:
            col = est_levels.index(k_true)
            ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="#111827", linewidth=1.2))


def summary_table(df: pd.DataFrame) -> str:
    rows = (
        df.groupby(["method_label", "criterion_label"], sort=False)
        .agg(
            rows=("K_true", "size"),
            exact_k=("K_correct", "mean"),
            mean_signed_error=("K_error", "mean"),
            mean_abs_error=("K_error", lambda x: np.abs(x).mean()),
        )
        .reset_index()
        .rename(
            columns={
                "method_label": "Method",
                "criterion_label": "Criterion",
                "rows": "Rows",
                "exact_k": "Exact K",
                "mean_signed_error": "Mean signed K error",
                "mean_abs_error": "Mean abs K error",
            }
        )
    )
    rows = rows.sort_values(["Method", "Criterion"])
    rows["Rows"] = rows["Rows"].map(lambda x: f"{int(x):,}")
    rows["Exact K"] = rows["Exact K"].map(lambda x: f"{100 * x:.1f}%")
    for col in ["Mean signed K error", "Mean abs K error"]:
        rows[col] = rows[col].map(lambda x: f"{x:.2f}")
    return rows.to_html(index=False, escape=True)


def render_html(args: argparse.Namespace, summary: str, figures: list[tuple[str, str]]) -> str:
    sections = "\n".join(
        f"<section><h2>{escape(title)}</h2><img src=\"{src}\" alt=\"{escape(title)}\"></section>" for title, src in figures
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Simulation K-selection confusion matrices</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    h1 {{ margin-bottom: 4px; }}
    .meta {{ color: #52616b; margin-top: 0; }}
    section {{ margin: 34px 0; }}
    h2 {{ margin-bottom: 12px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #d9e2ec; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 18px 0 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ background: #f3f6f8; }}
  </style>
</head>
<body>
  <h1>Simulation K-selection confusion matrices</h1>
  <p class="meta">Source file: {escape(str(args.comparison_csv))}. Rows are normalized within true K; the outlined cells are exact K recovery.</p>
  {summary}
  {sections}
</body>
</html>
"""


def fig_to_img(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def self_test() -> None:
    df = prepare_table(
        pd.DataFrame(
            [
                {"method": "spm_vb", "criterion": "icl", "K_true": 4, "K_estimated": 4, "montage_type": "full", "n_leads": 71},
                {"method": "spm_vb", "criterion": "icl", "K_true": 4, "K_estimated": 5, "montage_type": "full", "n_leads": 71},
                {"method": "kmeans_koenig", "criterion": "silhouette", "K_true": 5, "K_estimated": 5, "montage_type": "full", "n_leads": 71},
            ]
        )
    )
    counts = confusion_counts(df[df["criterion"].eq("icl")], [4, 5], [4, 5])
    rates = row_rates(counts)
    assert counts.loc[4, 4] == 1
    assert rates.loc[4, 4] == 0.5
    html = render_html(argparse.Namespace(comparison_csv=Path("x.csv")), summary_table(df), [("x", "data:image/png;base64,abc")])
    assert "Simulation K-selection confusion matrices" in html


if __name__ == "__main__":
    main()
