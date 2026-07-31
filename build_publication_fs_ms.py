"""Build the non-duplicated, publication-facing FS_MS analysis package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


FS_COLOR = "#0072B2"
ES_COLOR = "#D55E00"
ACCENT = "#009E73"
PURPLE = "#7A5195"
GREY = "#6B7280"
STATES = ("F", "E", "G", "B", "A", "C")
TIME_BINS = ("-20 to -10", "-10 to -5", "-5 to -2", "-2 to -1", "-1 to 0")
FIGURES = (
    "figure_01_retained_context",
    "figure_02_autonomic_and_recovery",
    "figure_03_transition_trajectories",
    "figure_04_planned_contrasts",
    "figure_05_source_and_route",
    "figure_06_autonomic_factors",
    "figure_07_dynamic_models",
    "figure_08_pseudo_onset_balance",
    "figure_09_fs_kappa_js_cpi_3d",
    "figure_10_es_kappa_js_cpi_3d",
)

RETAINED_TABLES = (
    "demographic_summary.csv", "event_characteristics.csv", "hr_bayesian_cells.csv",
    "coverage_bayesian_cells.csv", "autonomic_level_participant_cells.csv",
    "autonomic_level_bayesian_cells.csv", "autonomic_level_bayesian_contrasts.csv",
    "c_origin_js_recovery.csv", "c_origin_js_recovery_paired_events.csv",
    "state_reference_qc.csv",
)
PRIMARY_TABLES = (
    "planned_early_late_contrasts.csv", "group_specific_spline_trajectory.csv",
    "hierarchical_source_specificity.csv", "posterior_source_metric_cells.csv",
    "posterior_destination_changes.csv", "interictal_autonomic_pca_loadings.csv",
    "lagged_autonomic_factor_tests.csv", "dynamic_loop_coefficients.csv",
    "dynamic_loop_model_comparison.csv", "pseudo_onset_match_balance.csv",
    "matched_pseudo_difference_in_differences.csv", "fullrank_restart_diagnostics.csv",
    "fullrank_loss.csv",
    "posterior_predictive_checks.csv", "posterior_predictive_discrepancy.csv",
    "simulation_recovery.csv", "confirmatory_fit_diagnostics.json",
    "simulation_recovery_summary.json", "pseudo_onset_covariate_availability.json", "summary.json",
)


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.5, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .8, "grid.linewidth": .55, "grid.alpha": .18,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_figure(fig, figure_dir: Path, stem: str) -> None:
    fig.savefig(figure_dir / f"{stem}.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel(ax, label: str) -> None:
    ax.text(-.12, 1.05, label, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="bottom", ha="left")


def errorbar(ax, frame: pd.DataFrame, x, *, color, label=None, marker="o", zorder=3) -> None:
    middle = frame["median"].to_numpy(float)
    ax.errorbar(x, middle,
                yerr=[middle - frame.ci95_low.to_numpy(float),
                      frame.ci95_high.to_numpy(float) - middle],
                color=color, marker=marker, ms=4.8, lw=1.25, capsize=2.2,
                label=label, zorder=zorder)


def plot_retained_context(data: Path, figures: Path) -> None:
    hr = pd.read_csv(data / "hr_bayesian_cells.csv")
    coverage = pd.read_csv(data / "coverage_bayesian_cells.csv")
    fig = plt.figure(figsize=(7.4, 6.2), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(.9, 1.25))
    ax_hr = fig.add_subplot(grid[0, :])
    periods = [p for p in ("interictal", "preictal", "ictal") if p in set(hr.period)]
    styles = {("FS", "hyper"): (FS_COLOR, "-", "o"), ("FS", "hypo"): (FS_COLOR, "--", "^"),
              ("ES", "hyper"): (ES_COLOR, "-", "o"), ("ES", "hypo"): (ES_COLOR, "--", "^")}
    x = np.arange(len(periods))
    for (group, semiology), (color, linestyle, marker) in styles.items():
        part = (hr[hr.group.eq(group) & hr.semiology.eq(semiology)]
                .set_index("period").reindex(periods).reset_index())
        middle = part["median"].to_numpy(float)
        ax_hr.plot(x, middle, color=color, linestyle=linestyle, marker=marker, ms=5,
                   label=f"{group} {semiology}")
        ax_hr.fill_between(x, part.ci95_low, part.ci95_high, color=color, alpha=.08)
    ax_hr.set(xticks=x, xticklabels=[p.title() for p in periods], ylabel="Heart rate (beats/min)",
              title="Cardiac rate across clinical periods")
    ax_hr.grid(axis="y"); ax_hr.legend(frameon=False, ncol=4, loc="upper left")
    panel(ax_hr, "A")

    offsets = {("FS", "hyper"): -.24, ("FS", "hypo"): -.08,
               ("ES", "hyper"): .08, ("ES", "hypo"): .24}
    for letter, ax, phase in zip(("B", "C"), (fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])),
                                 ("preictal", "ictal")):
        phase_mask = coverage.contrast.str.endswith(phase)
        for (group, semiology), offset in offsets.items():
            labels = [f"{group} {semiology} {phase}"]
            part = (coverage[phase_mask & coverage.contrast.isin(labels)]
                    .set_index("microstate").reindex(STATES).reset_index())
            color = FS_COLOR if group == "FS" else ES_COLOR
            marker = "o" if semiology == "hyper" else "^"
            middle = part["median"].to_numpy(float)
            ax.errorbar(np.arange(len(STATES)) + offset, middle,
                        yerr=[middle - part.ci95_low.to_numpy(float),
                              part.ci95_high.to_numpy(float) - middle],
                        fmt=marker, color=color, ms=4.2, lw=1, capsize=1.8,
                        label=f"{group} {semiology}" if phase == "preictal" else None)
        ax.set(xticks=np.arange(len(STATES)), xticklabels=STATES, xlabel="Microstate",
               ylabel="Coverage", title=f"{phase.title()} coverage", ylim=(0, .34))
        ax.grid(axis="y"); panel(ax, letter)
    fig.axes[1].legend(frameon=False, ncol=2, loc="upper center")
    save_figure(fig, figures, FIGURES[0])


def plot_autonomic_recovery(data: Path, figures: Path) -> None:
    participant = pd.read_csv(data / "autonomic_level_participant_cells.csv")
    contrasts = pd.read_csv(data / "autonomic_level_bayesian_contrasts.csv")
    recovery_events = pd.read_csv(data / "c_origin_js_recovery_paired_events.csv")
    recovery = pd.read_csv(data / "c_origin_js_recovery.csv")
    groups = (("FS", "hyper"), ("FS", "hypo"), ("ES", "hyper"), ("ES", "hypo"))
    labels = [f"{g}\n{s}" for g, s in groups]
    rng = np.random.default_rng(30072026)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.25), layout="constrained")
    for letter, ax, index in zip(("A", "B"), axes[:2], ("CSI", "CPI")):
        for position, (group, semiology) in enumerate(groups):
            event_type = "functional" if group == "FS" else "epileptic"
            subset = participant[participant.event_type.eq(event_type) & participant.semiology.eq(semiology)]
            paired = subset.pivot_table(index="participant", columns="period", values=index.lower(), aggfunc="mean")
            paired = paired.dropna(subset=["interictal", "preictal"])
            values = np.log(paired.preictal / paired.interictal).to_numpy(float)
            color = FS_COLOR if group == "FS" else ES_COLOR
            ax.scatter(position + rng.uniform(-.15, .15, len(values)), values, s=8, alpha=.18,
                       color=color, linewidths=0)
            name = f"{group} {semiology}: preictal minus interictal"
            estimate = contrasts[contrasts["index"].eq(index) & contrasts.contrast.eq(name)].iloc[0]
            ax.errorbar(position, estimate["median"],
                        yerr=[[estimate["median"] - estimate.ci95_low],
                              [estimate.ci95_high - estimate["median"]]],
                        fmt="D", ms=5, color=color, markeredgecolor="white", markeredgewidth=.7,
                        capsize=2.5, zorder=4)
        ax.axhline(0, color=GREY, lw=.8); ax.grid(axis="y")
        ax.set(xticks=np.arange(4), xticklabels=labels, ylabel=f"Log preictal/interictal {index}",
               title=f"Participant-level {index} change")
        panel(ax, letter)

    ax = axes[2]
    for position, semiology in enumerate(("hyper", "hypo")):
        values = recovery_events.loc[recovery_events.semiology.eq(semiology), "log_post_pre_js"].to_numpy(float)
        color = PURPLE if semiology == "hyper" else ACCENT
        ax.scatter(position + rng.uniform(-.16, .16, len(values)), values, s=8, alpha=.18,
                   color=color, linewidths=0)
        name = f"{semiology.title()}motor postictal minus preictal"
        estimate = recovery[recovery.contrast.eq(name)].iloc[0]
        ax.errorbar(position, estimate["median"],
                    yerr=[[estimate["median"] - estimate.ci95_low],
                          [estimate.ci95_high - estimate["median"]]],
                    fmt="D", ms=5, color=color, markeredgecolor="white", markeredgewidth=.7,
                    capsize=2.5, zorder=4)
    ax.axhline(0, color=GREY, lw=.8); ax.grid(axis="y")
    ax.set(xticks=(0, 1), xticklabels=("Hyper", "Hypo"), ylabel="Log postictal/preictal J",
           title="Postictal transition recovery")
    panel(ax, "C")
    save_figure(fig, figures, FIGURES[1])


def plot_trajectories(data: Path, figures: Path) -> None:
    frame = pd.read_csv(data / "group_specific_spline_trajectory.csv")
    labels = {"J": "J: divergence from baseline", "K_H": "Kᴴ: route-independent concentration",
              "D_route": "D route: concentration-matched distance", "kappa": "κ: baseline-oriented concentration"}
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.6), layout="constrained", sharex=True)
    for letter, ax, outcome in zip("ABCD", axes.flat, ("J", "K_H", "D_route", "kappa")):
        for group, color in (("FS", FS_COLOR), ("ES", ES_COLOR)):
            part = frame[frame.group.eq(group) & frame.outcome.eq(outcome)].sort_values("minutes_to_onset")
            ax.fill_between(part.minutes_to_onset, part.ci95_low, part.ci95_high, color=color, alpha=.12)
            ax.plot(part.minutes_to_onset, part["median"], color=color, lw=1.8, label=group)
        ax.axvspan(-20, -10, color=GREY, alpha=.055); ax.axvspan(-1, 0, color=GREY, alpha=.09)
        ax.set(title=labels[outcome], xlabel="Minutes to onset", ylabel="Posterior value", xlim=(-20, 0))
        ax.grid(); panel(ax, letter)
    axes[0, 0].legend(frameon=False, ncol=2)
    save_figure(fig, figures, FIGURES[2])


def plot_planned_contrasts(data: Path, figures: Path) -> None:
    frame = pd.read_csv(data / "planned_early_late_contrasts.csv")
    contrasts = (
        ("FS(-20,-10) minus ES(-20,-10)", "Early FS − ES κ", "< 0"),
        ("FS(-1,0) minus FS(-20,-10)", "Late − early FS κ", "> 0"),
        ("FS(-1,0) minus FS(-20,-10)", "Late − early FS J", "≈ 0"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.65), layout="constrained")
    colors = (PURPLE, ACCENT)
    for letter, ax, (contrast, title, direction), outcome in zip("ABC", axes, contrasts, ("kappa", "kappa", "J")):
        part = frame[frame.contrast.eq(contrast) & frame.outcome.eq(outcome)]
        for y, (analysis, color) in enumerate(zip(("all eligible events", "complete temporal coverage"), colors)):
            row = part[part.analysis.eq(analysis)].iloc[0]
            ax.errorbar(row["median"], y,
                        xerr=[[row["median"] - row.ci95_low], [row.ci95_high - row["median"]]],
                        fmt="o", color=color, capsize=3, ms=5)
        if outcome == "J": ax.axvspan(-.02, .02, color=ACCENT, alpha=.09, label="equivalence region")
        ax.axvline(0, color=GREY, lw=.8)
        ax.set(yticks=(0, 1), yticklabels=("All events", "Complete coverage"),
               xlabel=f"Contrast ({outcome})", title=f"{title}\nplanned {direction}")
        ax.grid(axis="x"); panel(ax, letter)
    save_figure(fig, figures, FIGURES[3])


def plot_source_route(data: Path, figures: Path) -> None:
    specificity = pd.read_csv(data / "hierarchical_source_specificity.csv")
    destinations = pd.read_csv(data / "posterior_destination_changes.csv")
    fig = plt.figure(figsize=(7.5, 3.35), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=(1.15, 1, 1))
    ax = fig.add_subplot(grid[0, 0])
    source = specificity[specificity.source_state.isin(STATES)].set_index("source_state").reindex(STATES).reset_index()
    y = np.arange(len(source)); middle = source["median"].to_numpy(float)
    colors = [ES_COLOR if state == "C" else FS_COLOR for state in source.source_state]
    for yi, row, color in zip(y, source.itertuples(), colors):
        ax.errorbar(row.median, yi, xerr=[[row.median - row.ci95_low], [row.ci95_high - row.median]],
                    fmt="o", color=color, capsize=2.3, ms=5)
    ax.axvline(0, color=GREY, lw=.8); ax.set(yticks=y, yticklabels=source.source_state,
        xlabel="Final-minute FS − ES in J", title="Hierarchical source effects")
    ax.invert_yaxis(); ax.grid(axis="x"); panel(ax, "A")
    c_row = specificity[specificity.source_state.eq("C-specificity")].iloc[0]
    ax.text(.02, -.18, f"P(C > mean other sources) = {c_row.probability_gt_zero:.3f}",
            transform=ax.transAxes, ha="left", va="top")

    matrices = []
    for time_bin in ("-20 to -10", "-1 to 0"):
        part = destinations[destinations.time_bin.eq(time_bin) & destinations.group.eq("FS minus ES")]
        matrix = 100 * part.pivot(index="source_state", columns="destination", values="median").reindex(index=STATES, columns=STATES)
        matrices.append(matrix)
    maximum = max(np.nanmax(np.abs(matrix.to_numpy(float))) for matrix in matrices)
    image = None
    for letter, slot, matrix, title in zip(("B", "C"), (grid[0, 1], grid[0, 2]), matrices,
                                            ("Early routing difference", "Final-minute routing difference")):
        axis = fig.add_subplot(slot)
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-maximum, vmax=maximum, aspect="equal")
        axis.set(xticks=np.arange(6), xticklabels=STATES, yticks=np.arange(6), yticklabels=STATES,
                 xlabel="Destination", ylabel="Source", title=title)
        for row in range(6):
            for column in range(6):
                value = matrix.iloc[row, column]
                if np.isfinite(value): axis.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=6.3)
        panel(axis, letter)
    colorbar = fig.colorbar(image, ax=fig.axes[1:3], shrink=.72, pad=.02)
    colorbar.set_label("FS − ES destination change (percentage points)")
    save_figure(fig, figures, FIGURES[4])


def plot_autonomic_factors(data: Path, figures: Path) -> None:
    loadings = pd.read_csv(data / "interictal_autonomic_pca_loadings.csv")
    lagged = pd.read_csv(data / "lagged_autonomic_factor_tests.csv")
    factor_order = ("A_mobilisation", "A_variability", "A_directional")
    inputs = list(dict.fromkeys(loadings.input))
    matrix = loadings.pivot(index="factor", columns="input", values="loading").reindex(index=factor_order, columns=inputs)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.15), layout="constrained", gridspec_kw={"width_ratios": (1.8, 1)})
    image = axes[0].imshow(matrix, cmap="PuOr_r", vmin=-.65, vmax=.65, aspect="auto")
    short_names = {"mean_hr_bpm_z": "HR", "mean_rr_s_z": "RR", "csi_z": "CSI", "cpi_z": "CPI",
                   "rmssd_s_z": "RMSSD", "rr_first_difference_s_z": "ΔRR",
                   "rr_acceleration_s_z": "Δ²RR", "rr_surprise_z": "RR surprise"}
    short = [short_names.get(name, name) for name in inputs]
    axes[0].set(xticks=np.arange(len(inputs)), xticklabels=short, yticks=np.arange(3),
                yticklabels=("Mobilisation", "Variability", "Directional"), title="Interictal PCA loadings")
    axes[0].tick_params(axis="x", rotation=48)
    for row in range(3):
        for column in range(len(inputs)):
            axes[0].text(column, row, f"{matrix.iloc[row, column]:.2f}", ha="center", va="center", fontsize=6.2)
    fig.colorbar(image, ax=axes[0], shrink=.75, pad=.02, label="Loading")
    panel(axes[0], "A")

    scaled = lagged.copy(); scaled[["estimate", "ci95_low", "ci95_high"]] *= 1e6
    y = np.arange(len(scaled)); middle = scaled.estimate.to_numpy(float)
    axes[1].errorbar(middle, y, xerr=[middle - scaled.ci95_low, scaled.ci95_high - middle],
                     fmt="o", color=FS_COLOR, capsize=3, ms=5)
    axes[1].axvline(0, color=GREY, lw=.8); axes[1].set(yticks=y,
        yticklabels=("Mobilisation → J", "Variability → κ"), xlabel="Lag coefficient × 10⁶",
        title="Prespecified lagged tests")
    axes[1].grid(axis="x"); panel(axes[1], "B")
    save_figure(fig, figures, FIGURES[5])


def plot_dynamic_models(data: Path, figures: Path) -> None:
    frame = pd.read_csv(data / "dynamic_loop_model_comparison.csv")
    overall = frame[frame.outcome.eq("all") & ~frame.model.eq("shared_state_only")].sort_values("delta_elpd")
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.15), layout="constrained", gridspec_kw={"width_ratios": (1, 1.3)})
    y = np.arange(len(overall)); x = overall.delta_elpd.to_numpy(float)
    axes[0].errorbar(x, y, xerr=overall.se_delta_elpd, fmt="o", color=FS_COLOR, capsize=3, ms=5)
    axes[0].axvline(0, color=GREY, lw=.8); axes[0].set(yticks=y,
        yticklabels=[name.replace("_", " ") for name in overall.model], xlabel="Δ leave-one-event-out ELPD",
        title="Competitive dynamic models")
    axes[0].grid(axis="x"); panel(axes[0], "A")

    outcomes = ("A_mobilisation", "O", "R", "K")
    models = ("autoregression_only", "autonomic_to_EEG", "EEG_to_autonomic", "bidirectional_loop", "shared_state_only")
    pivot = frame[frame.outcome.isin(outcomes)].pivot(index="model", columns="outcome", values="rmse").reindex(index=models, columns=outcomes)
    relative = 100 * (pivot / pivot.min(axis=0) - 1)
    image = axes[1].imshow(relative, cmap="YlOrRd", vmin=0, vmax=min(75, np.nanmax(relative)), aspect="auto")
    axes[1].set(xticks=np.arange(4), xticklabels=("Autonomic", "C expression", "Route", "Concentration"),
        yticks=np.arange(5), yticklabels=[name.replace("_", " ") for name in models],
        title="Prediction error above outcome-specific best")
    axes[1].tick_params(axis="x", rotation=30)
    for row in range(5):
        for column in range(4):
            value = relative.iloc[row, column]
            axes[1].text(column, row, f"{value:.1f}%", ha="center", va="center", fontsize=6.5)
    fig.colorbar(image, ax=axes[1], shrink=.75, pad=.02, label="RMSE above best (%)")
    panel(axes[1], "B")
    save_figure(fig, figures, FIGURES[6])


def plot_balance(data: Path, figures: Path) -> None:
    did = pd.read_csv(data / "matched_pseudo_difference_in_differences.csv")
    order = ("Global", *STATES)
    effect = (did[did.outcome.eq("J") & did.contrast.eq("matched difference-in-differences")]
              .set_index("source_state").reindex(order).dropna().reset_index())
    balance = pd.read_csv(data / "pseudo_onset_match_balance.csv")
    main = balance[~balance.matching_variable.eq("record_fraction")].sort_values("standardised_mean_difference")
    position = balance[balance.matching_variable.eq("record_fraction")].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.7), layout="constrained",
                             gridspec_kw={"width_ratios": (1.15, 1.55)})

    y = np.arange(len(effect)); middle = effect["median"].to_numpy(float)
    colors = [PURPLE if state == "Global" else (ES_COLOR if state == "C" else FS_COLOR)
              for state in effect.source_state]
    for yi, row, color in zip(y, effect.itertuples(), colors):
        axes[0].errorbar(row.median, yi,
                         xerr=[[row.median - row.ci95_low], [row.ci95_high - row.median]],
                         fmt="o", color=color, capsize=2.5, ms=5)
    axes[0].axvline(0, color=GREY, lw=.8)
    axes[0].set(yticks=y, yticklabels=effect.source_state,
                xlabel="[FS genuine − pseudo] −\n[ES genuine − pseudo] in J",
                title="Matched onset-specific contrast")
    axes[0].invert_yaxis(); axes[0].grid(axis="x"); panel(axes[0], "A")

    labels = [name.replace("_", " ").replace("n ", "count: ") for name in main.matching_variable]
    y = np.arange(len(main)); values = main.standardised_mean_difference.to_numpy(float)
    colors = np.where(np.abs(values) <= .1, ACCENT, np.where(np.abs(values) <= .25, PURPLE, ES_COLOR))
    axes[1].hlines(y, 0, values, color=colors, lw=1.3)
    axes[1].scatter(values, y, color=colors, s=28, zorder=3)
    axes[1].axvline(0, color=GREY, lw=.8); axes[1].axvspan(-.1, .1, color=ACCENT, alpha=.08)
    axes[1].set(yticks=y, yticklabels=labels,
                xlabel="Standardised mean difference (true − pseudo)",
                title="Within-participant matching balance")
    axes[1].text(.99, -.16,
                 f"Record-position SMD = {position.standardised_mean_difference:.2f} (design restriction)",
                 transform=axes[1].transAxes, ha="right", va="top", color=GREY, fontsize=7)
    axes[1].set_xlim(min(-.35, values.min() - .03), max(.15, values.max() + .03))
    axes[1].grid(axis="x"); panel(axes[1], "B")
    save_figure(fig, figures, FIGURES[7])


def make_3d_frame(data: Path) -> pd.DataFrame:
    metrics = pd.read_csv(data / "posterior_global_transition_window_means.csv.gz")
    factors = pd.read_csv(data / "autonomic_factor_windows.csv.gz")
    keys = ["participant", "file_path", "window_end_to_ictus_s"]
    factors = factors[factors.condition.eq("preict")][keys + ["cpi_z", "semiology"]]
    merged = metrics[metrics.condition.eq("preict")].merge(factors, on=keys, how="inner")
    event = (merged.groupby(["participant", "event_key", "event_type", "time_bin", "semiology"], observed=True)
             [["J", "kappa", "cpi_z", "window_end_to_ictus_s"]].mean().reset_index())
    event = event[np.isfinite(event[["J", "kappa", "cpi_z"]]).all(axis=1) & event.J.gt(0) & event.kappa.gt(0)].copy()
    event["log10_J"] = np.log10(event.J); event["log10_kappa"] = np.log10(event.kappa)
    event["cpi_display"] = np.arcsinh(event.cpi_z / 2)
    event["minutes_before_onset"] = event.window_end_to_ictus_s / 60
    return event


def plot_3d(event: pd.DataFrame, figures: Path, group: str, stem: str, limits=None) -> tuple:
    event_type = "functional" if group == "FS" else "epileptic"
    use = event[event.event_type.eq(event_type)].copy()
    rng = np.random.default_rng(91 if group == "FS" else 92)
    xyz = use[["log10_J", "log10_kappa", "cpi_display"]].to_numpy(float)
    if limits is None:
        all_xyz = event[["log10_J", "log10_kappa", "cpi_display"]].to_numpy(float)
        limits = tuple((np.nanpercentile(all_xyz[:, column], .25), np.nanpercentile(all_xyz[:, column], 99.75))
                       for column in range(3))
    spans = np.array([high - low for low, high in limits])
    xyz += rng.normal(0, spans * np.array([.004, .004, .007]), size=xyz.shape)
    figure = plt.figure(figsize=(6.3, 5.35))
    figure.subplots_adjust(left=.02, right=.84, bottom=.07, top=.91)
    ax = figure.add_subplot(111, projection="3d")
    norm = Normalize(0, 20); cmap = plt.get_cmap("viridis")
    for semiology, marker in (("hyper", "o"), ("hypo", "^")):
        keep = use.semiology.eq(semiology).to_numpy()
        ax.scatter(xyz[keep, 0], xyz[keep, 1], xyz[keep, 2],
                   c=use.loc[keep, "minutes_before_onset"], cmap=cmap, norm=norm,
                   marker=marker, s=13, alpha=.28, depthshade=False, linewidths=0,
                   label=semiology.title())
    centroids = (use[use.time_bin.isin(TIME_BINS)].groupby("time_bin", observed=True)
                 [["log10_J", "log10_kappa", "cpi_display", "minutes_before_onset"]].median().reindex(TIME_BINS).dropna())
    ax.plot(centroids.log10_J, centroids.log10_kappa, centroids.cpi_display,
            color="black", lw=1.6, marker="D", ms=5, markerfacecolor="white", markeredgewidth=1,
            label="Time-bin median")
    ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2],
           xlabel="J divergence (log scale)", ylabel="κ (log scale)", zlabel="CPI (within-person z)",
           title=f"{group}: transition divergence, concentration and cardiac performance\n(n={len(use)} event-bin means)")
    j_ticks = np.array([.002, .005, .01, .02, .05, .1]); j_ticks = j_ticks[(np.log10(j_ticks) >= limits[0][0]) & (np.log10(j_ticks) <= limits[0][1])]
    k_ticks = np.array([.02, .05, .1, .25, .5, 1, 1.5]); k_ticks = k_ticks[(np.log10(k_ticks) >= limits[1][0]) & (np.log10(k_ticks) <= limits[1][1])]
    cpi_ticks = np.array([-10, -5, 0, 5, 10, 20]); cpi_display = np.arcsinh(cpi_ticks / 2); keep = (cpi_display >= limits[2][0]) & (cpi_display <= limits[2][1])
    ax.set_xticks(np.log10(j_ticks), [f"{value:g}" for value in j_ticks])
    ax.set_yticks(np.log10(k_ticks), [f"{value:g}" for value in k_ticks])
    ax.set_zticks(cpi_display[keep], [f"{value:g}" for value in cpi_ticks[keep]])
    ax.view_init(elev=23, azim=-57); ax.legend(frameon=False, loc="upper left")
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap); scalar.set_array([])
    colorbar = figure.colorbar(scalar, ax=ax, shrink=.62, pad=.08); colorbar.set_label("Minutes before onset")
    save_figure(figure, figures, stem)
    return limits


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    table = frame.loc[:, columns].copy()
    for column in table.select_dtypes(include=["number"]): table[column] = table[column].map(lambda value: f"{value:.4g}" if pd.notna(value) else "")
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in table.itertuples(index=False, name=None)]
    return "\n".join((header, divider, *rows))


def analysis_decisions() -> pd.DataFrame:
    return pd.DataFrame([
        ("Cohort/event characteristics", "retain", "Not addressed by the transition hierarchy", "Retained descriptive tables"),
        ("Heart rate", "retain", "Clinical context, not a transition estimand", "Retained Bayesian cells"),
        ("Microstate coverage", "retain", "Occupancy is distinct from transition grammar", "Retained Bayesian cells"),
        ("Autonomic level", "retain", "Describes physiology rather than lagged coupling", "Retained CSI/CPI level model"),
        ("Postictal recovery", "retain secondary", "Not included in the preictal hierarchy", "Retained paired recovery model"),
        ("Legacy C-origin JS/kappa models", "replace", "Superseded by joint count likelihood and posterior-derived metrics", "Hierarchical multinomial model"),
        ("Raw-count/fixed-exit sensitivities", "replace", "Count uncertainty now enters the multinomial likelihood directly", "Hierarchical multinomial model; N=30 sensitivity retained in audit"),
        ("Pairwise C-specificity tests", "replace", "Multiplicity and shrinkage handled hierarchically", "P(C-source J effect > mean other-source effects)"),
        ("Legacy route/destination tests", "replace", "Routes and destinations are derived coherently from one posterior", "J, K_H, D_route and destination changes"),
        ("Legacy autonomic kitchen-sink models", "replace", "Correlated inputs reduced interictally", "Three orthogonal autonomic PCs"),
        ("Legacy predictive-flow forest", "replace", "Redundant pairwise lag models", "Five-model joint dynamic comparison"),
        ("Legacy pseudo-onset comparison", "replace", "Unpaired genuine and pseudo contrasts do not test onset specificity", "Matched FS/ES genuine-pseudo difference-in-differences"),
    ], columns=["analysis", "decision", "reason", "publication_replacement"])


def build(args) -> dict:
    focused_data = args.focused_dir / "data"
    optimised_data = args.optimised_dir / "data"
    required = [*(focused_data / name for name in RETAINED_TABLES),
                *(optimised_data / name for name in PRIMARY_TABLES),
                optimised_data / "posterior_global_transition_window_means.csv.gz",
                optimised_data / "autonomic_factor_windows.csv.gz"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise FileNotFoundError("Missing publication inputs: " + ", ".join(missing))
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Publication output is not empty: {args.out_dir}")
    figures = args.out_dir / "figures"; tables = args.out_dir / "tables"; provenance = args.out_dir / "provenance"
    for directory in (figures, tables, provenance): directory.mkdir(parents=True, exist_ok=True)
    configure_style()

    for name in RETAINED_TABLES: shutil.copy2(focused_data / name, tables / f"retained_{name}")
    for name in PRIMARY_TABLES: shutil.copy2(optimised_data / name, tables / name)
    for source, destination in ((args.focused_dir / "AUDIT.txt", provenance / "focused_AUDIT.txt"),
                                (args.optimised_dir / "AUDIT.json", provenance / "optimised_AUDIT.json")):
        if source.is_file(): shutil.copy2(source, destination)
    decisions = analysis_decisions(); decisions.to_csv(args.out_dir / "ANALYSIS_DECISIONS.csv", index=False)

    plot_retained_context(focused_data, figures)
    plot_autonomic_recovery(focused_data, figures)
    plot_trajectories(optimised_data, figures)
    plot_planned_contrasts(optimised_data, figures)
    plot_source_route(optimised_data, figures)
    plot_autonomic_factors(optimised_data, figures)
    plot_dynamic_models(optimised_data, figures)
    plot_balance(optimised_data, figures)
    three_d = make_3d_frame(optimised_data); three_d.to_csv(tables / "kappa_js_cpi_event_bins.csv", index=False)
    limits = plot_3d(three_d, figures, "FS", FIGURES[8])
    plot_3d(three_d, figures, "ES", FIGURES[9], limits)

    planned = pd.read_csv(optimised_data / "planned_early_late_contrasts.csv")
    specificity = pd.read_csv(optimised_data / "hierarchical_source_specificity.csv")
    did = pd.read_csv(optimised_data / "matched_pseudo_difference_in_differences.csv")
    fit_diagnostics = json.loads((optimised_data / "confirmatory_fit_diagnostics.json").read_text(encoding="utf-8"))
    recovery_diagnostics = json.loads((optimised_data / "simulation_recovery_summary.json").read_text(encoding="utf-8"))
    ppc = pd.read_csv(optimised_data / "posterior_predictive_checks.csv")
    ppc_coverage = float(ppc.observed_in_predictive_interval.astype(str).str.lower().eq("true").mean())
    lagged = pd.read_csv(optimised_data / "lagged_autonomic_factor_tests.csv")
    loop = pd.read_csv(optimised_data / "dynamic_loop_model_comparison.csv")
    recovery = pd.read_csv(focused_data / "c_origin_js_recovery.csv")
    early = planned[(planned.analysis.eq("all eligible events")) & planned.contrast.str.startswith("FS(-20")].iloc[0]
    consolidate = planned[(planned.analysis.eq("complete temporal coverage")) & planned.outcome.eq("kappa") & planned.contrast.str.startswith("FS(-1")].iloc[0]
    stable = planned[(planned.analysis.eq("complete temporal coverage")) & planned.outcome.eq("J")].iloc[0]
    c_specific = specificity[specificity.source_state.eq("C-specificity")].iloc[0]
    onset_did = did[(did.source_state.eq("Global")) & did.outcome.eq("J") &
                    did.contrast.eq("matched difference-in-differences")].iloc[0]
    best = loop[loop.outcome.eq("all")].sort_values("elpd", ascending=False).iloc[0]
    overall_recovery = recovery[recovery.contrast.str.startswith("Overall")].iloc[0]
    consolidation_text = ("supported" if consolidate.probability_gt_zero >= .975 else "not established")
    equivalence_text = ("supported" if stable.probability_in_rope >= .95 else "not established")
    validation_text = ("met" if fit_diagnostics["confirmatory_thresholds_met"] else "did not meet")
    report = [
        "# Publication analysis: retained physiology plus optimised transition model", "",
        "This report is deliberately non-duplicative. The hierarchical multinomial model replaces the legacy transition, route, specificity, autonomic-association and pairwise flow analyses. Cohort context, heart rate, coverage, autonomic levels and postictal recovery remain because they answer distinct questions.", "",
        "## Analysis decisions", "", markdown_table(decisions, list(decisions.columns)), "",
        "## Primary transition findings", "",
        f"- Early FS baseline-oriented concentration was lower than ES: median contrast {early['median']:.4g} (95% CrI {early.ci95_low:.4g} to {early.ci95_high:.4g}); P(<0)={early.probability_lt_zero:.3f}.",
        f"- Complete-coverage later FS consolidation was {consolidation_text}: median late-minus-early κ {consolidate['median']:.4g} (95% CrI {consolidate.ci95_low:.4g} to {consolidate.ci95_high:.4g}); P(>0)={consolidate.probability_gt_zero:.3f}.",
        f"- Early-to-final FS J equivalence was {equivalence_text}: posterior probability within the prespecified ±0.02 region={stable.probability_in_rope:.3f}.",
        f"- Matched onset-specific J difference-in-differences: median {onset_did['median']:.4g} (95% CrI {onset_did.ci95_low:.4g} to {onset_did.ci95_high:.4g}); P(>0)={onset_did.probability_gt_zero:.3f}.",
        f"- C specificity was not established: P(C-source J effect > mean other-source effects)={c_specific.probability_gt_zero:.3f}. The final-minute effect is distributed across sources.", "",
        "![Transition trajectories](figures/figure_03_transition_trajectories.png)", "",
        "![Planned contrasts](figures/figure_04_planned_contrasts.png)", "",
        "![Source and route](figures/figure_05_source_and_route.png)", "",
        "## Autonomic and dynamic findings", "",
        f"Neither prespecified latent-factor lag was conventionally significant (p={lagged.p_value.iloc[0]:.3f} and p={lagged.p_value.iloc[1]:.3f}). The highest leave-one-event-out ELPD was for {best.model.replace('_', ' ')}; differences among the competitive autoregressive/directional models were smaller than their uncertainty.",
        f"Postictal recovery remains a distinct secondary result: median log post/pre J {overall_recovery['median']:.3f} (95% CrI {overall_recovery.ci95_low:.3f} to {overall_recovery.ci95_high:.3f}); P(post<pre)={overall_recovery.probability_lt_zero:.3f}.", "",
        "![Autonomic factors](figures/figure_06_autonomic_factors.png)", "",
        "![Dynamic models](figures/figure_07_dynamic_models.png)", "",
        "## Onset specificity and model validation", "",
        "The primary onset-specific estimand is [FS genuine − FS matched pseudo] − [ES genuine − ES matched pseudo], with matching performed within participant and paired before aggregation.",
        ("This contrast supports onset specificity beyond the FS-associated phenotype."
         if onset_did.probability_gt_zero >= .975 or onset_did.probability_lt_zero >= .975 else
         "This contrast does not establish onset specificity; the evidence supports an FS-associated transition phenotype, not an onset-specific biomarker."),
        f"The destination-labelled orthonormal sum-to-zero masked-softmax model used {fit_diagnostics['restarts']} independent full-rank ADVI restarts × {fit_diagnostics['draws_per_restart']} posterior draws. The confirmatory thresholds {validation_text}: restart max R-hat={fit_diagnostics['max_r_hat']:.3f}; minimum bulk ESS={fit_diagnostics['min_ess_bulk']:.0f}; maximum tail relative ELBO change={fit_diagnostics['max_tail_relative_elbo_change']:.4f}. Posterior-predictive cell coverage={ppc_coverage:.3f}; simulation-recovery coefficient correlation={recovery_diagnostics['coefficient_correlation']:.3f}.", "",
        "![Matched onset specificity and balance](figures/figure_08_pseudo_onset_balance.png)", "",
        "## Qualitative κ–J–CPI geometry", "",
        "The matched 3D views use identical axes for FS and ES. Points are event-by-time-bin means with small deterministic display jitter; colour is minutes before onset, and the black-diamond path joins time-bin medians. Log transforms preserve the strongly skewed positive J and κ axes; CPI tick labels remain in within-person z units. These plots are descriptive, not additional hypothesis tests.", "",
        "![FS 3D](figures/figure_09_fs_kappa_js_cpi_3d.png)", "",
        "![ES 3D](figures/figure_10_es_kappa_js_cpi_3d.png)", "",
        "## Retained context and matching", "",
        "![Retained context](figures/figure_01_retained_context.png)", "",
        "![Autonomic and recovery](figures/figure_02_autonomic_and_recovery.png)", "",
        "Primary transition outcomes are J, Kᴴ and D route. Baseline-oriented κ is secondary. The pseudo-onset audit explicitly records unavailable movement/EMG, EOG, muscle power, bad-channel, sleep/wake, medication, respiration, CFA, segmentation and VR-overlap covariates.",
    ]
    (args.out_dir / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    output_files = [path for path in args.out_dir.rglob("*") if path.is_file()]
    audit = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "retain non-overlapping clinical context; replace legacy transition and flow analyses",
        "focused_source": str(args.focused_dir), "optimised_source": str(args.optimised_dir),
        "legacy_equalised_sensitivity_transitions": 30,
        "primary_model_uses_raw_transition_totals": True,
        "parameterisation": "destination-labelled orthonormal sum-to-zero masked softmax",
        "inference": "independent-restart full-rank ADVI", "primary_outcomes": ["J", "K_H", "D_route"],
        "secondary_outcome": "kappa", "figures_png": len(list(figures.glob("*.png"))),
        "figures_pdf": len(list(figures.glob("*.pdf"))), "three_d_display_jitter_seeded": True,
        "three_d_transforms": {"J": "log10", "kappa": "log10", "CPI_z": "asinh(CPI_z/2) with raw-unit ticks"},
        "code_sha256": sha256(Path(__file__)),
        "software": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "matplotlib")},
        "input_sha256": {str(path): sha256(path) for path in required},
        "output_sha256": {str(path.relative_to(args.out_dir)): sha256(path) for path in output_files},
    }
    (args.out_dir / "PUBLICATION_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    summary = {"retained_analyses": int(decisions.decision.str.startswith("retain").sum()),
               "replaced_analyses": int(decisions.decision.eq("replace").sum()),
               "figures": len(FIGURES), "figure_formats": ["PNG 400 dpi", "PDF vector"],
               "three_d_fs_event_bins": int(three_d.event_type.eq("functional").sum()),
               "three_d_es_event_bins": int(three_d.event_type.eq("epileptic").sum()),
               "c_specificity_probability": float(c_specific.probability_gt_zero),
               "matched_j_did_median": float(onset_did["median"]),
               "matched_j_did_ci95": [float(onset_did.ci95_low), float(onset_did.ci95_high)],
               "matched_j_did_probability_gt_zero": float(onset_did.probability_gt_zero),
               "fullrank_restarts": int(fit_diagnostics["restarts"]),
               "fullrank_draws_per_restart": int(fit_diagnostics["draws_per_restart"]),
               "fullrank_max_r_hat": float(fit_diagnostics["max_r_hat"]),
               "ppc_cell_coverage": ppc_coverage,
               "simulation_recovery_correlation": float(recovery_diagnostics["coefficient_correlation"]),
               "best_dynamic_model": str(best.model)}
    (args.out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def self_check() -> None:
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"event_type": np.repeat(("functional", "epileptic"), 20),
                          "semiology": np.tile(np.repeat(("hyper", "hypo"), 10), 2),
                          "time_bin": np.tile(TIME_BINS, 8), "J": np.exp(rng.normal(-4.5, .3, 40)),
                          "kappa": np.exp(rng.normal(-.4, .2, 40)), "cpi_z": rng.normal(size=40),
                          "window_end_to_ictus_s": np.tile((900, 450, 210, 90, 30), 8)})
    frame["log10_J"] = np.log10(frame.J); frame["log10_kappa"] = np.log10(frame.kappa)
    frame["cpi_display"] = np.arcsinh(frame.cpi_z / 2); frame["minutes_before_onset"] = frame.window_end_to_ictus_s / 60
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory); configure_style(); limits = plot_3d(frame, target, "FS", "check", None)
        assert (target / "check.png").stat().st_size > 0 and (target / "check.pdf").stat().st_size > 0
        assert all(high > low for low, high in limits)


def build_with_overwrite(args) -> dict:
    target = args.out_dir.resolve()
    if not args.overwrite or not target.exists() or not any(target.iterdir()):
        return build(args)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-building-", dir=target.parent))
    backup = target.with_name(f".{target.name}-previous")
    if backup.exists():
        shutil.rmtree(temporary)
        raise FileExistsError(f"Refusing overwrite because backup path exists: {backup}")
    temporary_args = argparse.Namespace(**vars(args)); temporary_args.out_dir = temporary
    try:
        summary = build(temporary_args)
        target.rename(backup)
        try:
            temporary.rename(target)
        except Exception:
            backup.rename(target)
            raise
        shutil.rmtree(backup)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focused-dir", type=Path)
    parser.add_argument("--optimised-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check: self_check(); print("publication build self-check passed"); return
    if args.focused_dir is None or args.optimised_dir is None or args.out_dir is None:
        parser.error("--focused-dir, --optimised-dir and --out-dir are required")
    print(json.dumps(build_with_overwrite(args), indent=2))


if __name__ == "__main__":
    main()
