"""Rebuild secondary topography figures from saved numerical result tables."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd


def global_templates(data_dir: Path, figure_dir: Path) -> None:
    templates = pd.read_csv(data_dir / "global_microstate_templates.csv")
    positions = pd.read_csv(data_dir / "morphology_tfce_channel_results.csv").drop_duplicates("channel_label").copy()
    positions.index = positions["channel_label"].str.lower()
    channels = [column for column in templates.columns if column != "state"]
    if not all(channel.lower() in positions.index for channel in channels):
        raise ValueError("Template channels do not match morphology channel geometry")
    xy = positions.loc[[channel.lower() for channel in channels], ["x", "y"]].to_numpy(float)
    xy *= .94 / np.sqrt(np.sum(xy**2, axis=1)).max()
    values = templates[channels].to_numpy(float)
    limit = float(np.nanmax(np.abs(values))) or 1.
    triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1])
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.8))
    contour = None
    for ax, (_, row), state_values in zip(axes.ravel(), templates.iterrows(), values):
        contour = ax.tricontourf(
            triangulation, state_values, levels=np.linspace(-limit, limit, 21), cmap="RdBu_r", extend="both"
        )
        ax.scatter(xy[:, 0], xy[:, 1], s=5, c="black", alpha=.55)
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="black", linewidth=1))
        ax.plot([-.09, 0, .09], [.99, 1.10, .99], color="black", linewidth=1)
        ax.set(xlim=(-1.12, 1.12), ylim=(-1.08, 1.14), aspect="equal")
        ax.axis("off")
        ax.set_title(str(row["state"]), fontsize=10, fontweight="bold", color="black")
    fig.suptitle("Global microstate templates (K=6)", fontsize=14, fontweight="bold", color="black")
    fig.colorbar(
        contour, ax=axes.ravel().tolist(), orientation="horizontal", fraction=.045, pad=.05,
        label="Template amplitude (a.u.)"
    )
    fig.subplots_adjust(left=.03, right=.97, bottom=.12, top=.88, hspace=.18, wspace=.06)
    fig.savefig(figure_dir / "global_microstate_templates.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def tanova_heatmap(data_dir: Path, figure_dir: Path) -> None:
    tanova = pd.read_csv(data_dir / "tanova_results.csv")
    effects = tanova["effect"].drop_duplicates().tolist()
    states = sorted(tanova["state"].dropna().astype(int).unique())
    matrix = tanova.pivot(index="state", columns="effect", values="p_max_state").reindex(index=states, columns=effects)
    scores = -np.log10(matrix.clip(lower=np.finfo(float).tiny).to_numpy(float))
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    image = ax.imshow(scores, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(states)), [f"State {state}" for state in states])
    ax.set_xticks(np.arange(len(effects)), [textwrap.fill(effect, 24) for effect in effects], rotation=38, ha="right")
    ax.set_title("Microstate TANOVA: max-state-corrected p-values", fontsize=13, fontweight="bold", color="black")
    for row, state in enumerate(states):
        for column, effect in enumerate(effects):
            value = matrix.loc[state, effect]
            if pd.notna(value) and value < .05:
                ax.text(column, row, "*", ha="center", va="center", color="white", fontsize=14, fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, pad=.02)
    colorbar.set_label("−log10(max-state p)")
    ax.tick_params(colors="black", labelsize=8)
    fig.subplots_adjust(left=.08, right=.94, bottom=.41, top=.88)
    fig.savefig(figure_dir / "tanova_pmax_heatmap.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    data_dir, figure_dir = args.package / "data", args.package / "figures"
    global_templates(data_dir, figure_dir)
    tanova_heatmap(data_dir, figure_dir)


if __name__ == "__main__":
    main()
