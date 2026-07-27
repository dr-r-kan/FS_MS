#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _analyse_comparison_results import load_template_topographies, plot_topomap  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a simulation and microstate-recovery workflow figure.")
    parser.add_argument("--comparison-csv", type=Path, default=ROOT / "outputs/simulations/results/comparison_results.csv")
    parser.add_argument("--template-file", type=Path, default=ROOT / "MetaMaps_2023_06.set")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/simulations/plots")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = make_pipeline_figure(args.comparison_csv, args.template_file, args.output_dir, seed=args.seed)
    print(out)


def make_pipeline_figure(comparison_csv: Path, template_file: Path, output_dir: Path, *, seed: int = 0) -> Path:
    table = pd.read_csv(comparison_csv)
    row = pick_example_row(table)
    payload = json.loads(Path(row["json_file"]).read_text(encoding="utf-8"))
    template = load_template_topographies(template_file, k=7)
    if template is None:
        raise FileNotFoundError(f"Could not load template topographies from {template_file}")

    rng = np.random.default_rng(seed)
    true_labels = true_template_labels(payload, template.labels)
    sfreq = float(payload.get("metadata", {}).get("sfreq", 250))
    snr = float(payload.get("metadata", {}).get("SNR_dB", row.get("SNR_dB", 5)))
    eeg, channels, time_s, state_index = reconstructed_eeg_snippet(payload, template, true_labels, sfreq, snr, rng)

    n_maps = len(true_labels)
    fig = plt.figure(figsize=(15.5, 9.2), constrained_layout=False)
    grid = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.25, 1.0], hspace=0.42)

    map_grid = grid[0].subgridspec(1, n_maps, wspace=0.25)
    clim = template_clim_for_labels(template, true_labels)
    for idx, label in enumerate(true_labels):
        ax = fig.add_subplot(map_grid[0, idx])
        if label in template.labels:
            plot_topomap(
                ax,
                template.maps[template.labels.index(label)],
                template.info if template.info is not None else template.xy,
                clim=clim,
            )
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, label, ha="center", va="center", fontweight="bold")
        ax.set_title(f"Template {label}", fontsize=11, fontweight="bold")

    ax_eeg = fig.add_subplot(grid[1])
    plot_eeg_snippet(ax_eeg, eeg, channels, time_s, state_index, true_labels, template.labels)

    ax_flow = fig.add_subplot(grid[2])
    plot_flow(ax_flow)

    meta = payload.get("metadata", {})
    fig.suptitle(
        "How the simulated EEG microstate benchmark works\n"
        f"example: K={meta.get('K_true', row.get('K_true', ''))}, SNR={meta.get('SNR_dB', row.get('SNR_dB', ''))} dB, "
        f"montage={meta.get('montage_type', row.get('montage_type', ''))}",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "simulation_microstate_recovery_workflow.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def pick_example_row(table: pd.DataFrame) -> pd.Series:
    table = table.copy()
    table = table[table.get("json_file", pd.Series(dtype=object)).map(valid_path).notna()]
    if table.empty:
        raise FileNotFoundError("No rows with valid json_file entries were found.")
    method = table["method"].astype(str).str.lower()
    criterion = table["criterion"].astype(str).str.lower()
    montage = table.get("montage_type", pd.Series("", index=table.index)).astype(str).str.lower()
    for mask in [
        method.eq("spm_vb") & criterion.eq("free_energy_elbow") & montage.eq("full"),
        method.eq("spm_vb") & criterion.eq("free_energy_elbow"),
        method.eq("spm_vb"),
    ]:
        if mask.any():
            return table[mask].iloc[0]
    return table.iloc[0]


def valid_path(value: object) -> Path | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    path = Path(str(value))
    return path if path.is_file() else None


def true_template_labels(payload: dict, fallback: list[str]) -> list[str]:
    meta = payload.get("metadata", {})
    labels = meta.get("true_template_labels") or meta.get("backfit", {}).get("true_state_template_labels")
    labels = [str(x) for x in labels] if labels else fallback[:4]
    return [x for x in labels if x in fallback] or fallback[:4]


def reconstructed_eeg_snippet(
    payload: dict,
    template,
    labels: list[str],
    sfreq: float,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    backfit = payload.get("metadata", {}).get("backfit", {})
    sequence = (
        backfit.get("hard", {}).get("true_top_label_index")
        or backfit.get("mixture", {}).get("true_top_label_index")
        or []
    )
    label_to_index = {label: idx for idx, label in enumerate(template.labels)}
    valid_template_idx = [label_to_index[label] for label in labels if label in label_to_index]
    if sequence:
        seq = np.asarray(sequence, dtype=int) - 1
        seq = seq[np.isin(seq, valid_template_idx)]
    else:
        seq = np.repeat(valid_template_idx, max(1, int(sfreq * 0.12)))
    if seq.size == 0:
        seq = np.repeat(valid_template_idx, max(1, int(sfreq * 0.12)))

    n_samples = min(seq.size, max(200, int(round(sfreq * 2.5))))
    seq = seq[:n_samples]
    channel_names = pick_channels(template.channel_labels)
    channel_idx = [template.channel_labels.index(ch) for ch in channel_names]
    signal = 8.0 * template.maps[seq][:, channel_idx].T
    signal_sd = np.nanstd(signal) or 1.0
    noise_sd = signal_sd / max(10 ** (snr_db / 20), 1e-3)
    eeg = signal + rng.normal(0, noise_sd, size=signal.shape)
    time_s = np.arange(n_samples) / sfreq
    return eeg, channel_names, time_s, seq


def pick_channels(channel_labels: list[str]) -> list[str]:
    preferred = ["Fp1", "Fz", "Cz", "Pz", "Oz", "C3", "C4"]
    by_key = {canonical_channel(label): label for label in channel_labels}
    chosen = [by_key[canonical_channel(ch)] for ch in preferred if canonical_channel(ch) in by_key]
    if len(chosen) >= 5:
        return chosen[:5]
    return channel_labels[: min(5, len(channel_labels))]


def canonical_channel(label: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(label).lower())


def plot_eeg_snippet(
    ax,
    eeg: np.ndarray,
    channels: list[str],
    time_s: np.ndarray,
    state_index: np.ndarray,
    labels: list[str],
    all_template_labels: list[str],
) -> None:
    scale = np.nanstd(eeg) or 1.0
    offsets = np.arange(eeg.shape[0])[::-1] * scale * 4.0
    for row, (trace, channel) in enumerate(zip(eeg, channels)):
        ax.plot(time_s, trace + offsets[row], linewidth=1.1, color="#1f2937")
        ax.text(time_s[0] - 0.035, offsets[row], channel, ha="right", va="center", fontsize=9)
    label_lookup = {idx: label for idx, label in enumerate(all_template_labels)}
    changes = np.r_[0, np.flatnonzero(np.diff(state_index) != 0) + 1, len(state_index)]
    colors = plt.cm.tab10(np.linspace(0, 1, max(3, len(labels))))
    for start, stop in zip(changes[:-1], changes[1:]):
        idx = int(state_index[start])
        label = label_lookup.get(idx, labels[0])
        color = colors[labels.index(label) % len(colors)] if label in labels else "0.7"
        ax.axvspan(time_s[start], time_s[stop - 1], ymin=0.02, ymax=0.16, color=color, alpha=0.28, linewidth=0)
    ax.set_xlim(time_s[0], time_s[-1])
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title("Short simulated EEG snippet: template maps are activated over time and mixed with noise", fontweight="bold")
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["left", "right", "top"]].set_visible(False)


def plot_flow(ax) -> None:
    ax.axis("off")
    steps = [
        ("1. Choose templates", "Canonical maps define\ntrue scalp fields"),
        ("2. Generate state sequence", "Dwell times and overlaps\ncreate dynamics"),
        ("3. Simulate EEG", "Active maps plus\nsensor noise"),
        ("4. Extract GFP peaks", "Fit high-information\nscalp maps"),
        ("5. Recover microstates", "K-Means and VB-GMM\nestimate maps"),
        ("6. Score recovery", "Align, backfit,\ncompute recovery"),
    ]
    positions = [(0.06, 0.66), (0.39, 0.66), (0.72, 0.66), (0.72, 0.24), (0.39, 0.24), (0.06, 0.24)]
    width = 0.22
    height = 0.26
    centers = []
    for (title, body), (xi, y) in zip(steps, positions):
        box = FancyBboxPatch(
            (xi, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=1.0,
            edgecolor="#0f172a",
            facecolor="#f8fafc",
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        centers.append((xi + width / 2, y))
        ax.text(xi + width / 2, y + 0.06, title, transform=ax.transAxes, ha="center", va="center", fontsize=9.8, fontweight="bold")
        ax.text(xi + width / 2, y - 0.055, body, transform=ax.transAxes, ha="center", va="center", fontsize=8.5, linespacing=1.25)
    for start, stop in [(0, 1), (1, 2)]:
        add_arrow(ax, (centers[start][0] + width / 2 + 0.012, centers[start][1]), (centers[stop][0] - width / 2 - 0.012, centers[stop][1]))
    add_arrow(ax, (centers[2][0], centers[2][1] - height / 2 - 0.015), (centers[3][0], centers[3][1] + height / 2 + 0.015))
    for start, stop in [(3, 4), (4, 5)]:
        add_arrow(ax, (centers[start][0] - width / 2 - 0.012, centers[start][1]), (centers[stop][0] + width / 2 + 0.012, centers[stop][1]))
    ax.set_title("Workflow", fontweight="bold", pad=6)


def add_arrow(ax, start: tuple[float, float], stop: tuple[float, float], *, connectionstyle: str = "arc3,rad=0") -> None:
    arrow = FancyArrowPatch(
        start,
        stop,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.1,
        color="#334155",
        transform=ax.transAxes,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)


def template_clim_for_labels(template, labels: list[str]) -> float:
    rows = [template.maps[template.labels.index(label)] for label in labels if label in template.labels]
    vals = np.concatenate([row[np.isfinite(row)] for row in rows]) if rows else template.maps[np.isfinite(template.maps)]
    clim = float(np.nanmax(np.abs(vals))) if vals.size else 1.0
    return clim if np.isfinite(clim) and clim > 0 else 1.0


if __name__ == "__main__":
    main()
