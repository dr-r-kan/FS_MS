"""Regenerate focused FS/ES paper figures from saved analysis outputs only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyse_focused_fs_ms import make_lags, within_record_z


ALL_STATES = ("F", "E", "G", "B", "A", "C")
FUNCTIONAL = ("FS functional", "DD functional")
EPILEPTIC = ("ES epileptic", "DD epileptic")
GROUP_COLORS = {"FS hyper": "#D55E00", "FS hypo": "#0072B2",
                "ES hyper": "#E69F00", "ES hypo": "#56B4E9"}
REQUIRED_INPUTS = (
    "analysis_schema.json", "hr_bayesian_cells.csv", "coverage_bayesian_cells.csv",
    "autonomic_level_participant_cells.csv", "autonomic_level_bayesian_cells.csv",
    "autonomic_level_bayesian_contrasts.csv", "c_origin_primary_bayesian.csv",
    "c_origin_preictal_windows.csv",
    "c_origin_dd_paired_values.csv", "c_origin_subsidiary_bayesian.csv",
    "c_origin_js_recovery.csv", "c_origin_js_recovery_paired_events.csv",
    "fs_autonomic_c_origin_correlations.csv", "bayesian_predictive_flow.csv",
    "preictal_kappa_js_participant.csv", "preictal_kappa_js_lines.csv",
    "autonomic_peak_microstate_common_windows.csv.gz",
    "posterior_fs_c_origin_c_log_kappa_autonomic_association.npz",
    "posterior_fs_ii_versus_last_minute_flow_autonomic_to_c_coverage.npz",
    "raw_count_c_origin_bayesian.csv", "raw_count_c_origin_event_means.csv",
    "c_reference_threshold_sensitivity_bayesian.csv", "c_origin_recovery_matched_bayesian.csv",
    "c_origin_route_destination_bayesian.csv", "c_origin_destination_participant_means.csv",
    "all_state_raw_count_specificity_bayesian.csv", "preictal_physiology_trajectory_bayesian.csv",
    "matched_interictal_pseudo_onsets.csv",
)
SUPPORTED_SCHEMAS = {"focused-c-reference-050-v4", "focused-c-reference-050-v5", "focused-c-reference-050-v6"}
FIGURE_NAMES = (
    "dense_rr_hr.png", "microstate_coverage.png", "robust_autonomic_interictal_preictal.png",
    "c_origin_primary.png", "c_origin_subsidiary_raw.png", "c_origin_js_recovery.png",
    "fs_autonomic_c_origin_correlations.png", "bayesian_predictive_flow.png",
    "preictal_kappa_vs_js.png", "c_origin_final_minute_boxplots.png",
    "c_origin_js_recovery_boxplots.png", "fs_cpi_kappa_regression.png",
    "fs_csi_to_c_regression.png",
    "raw_count_c_origin.png", "c_origin_route_destinations.png", "all_state_specificity.png",
    "preictal_physiology_trajectory.png", "c_origin_js_kappa_state_space.png",
    "matched_pseudo_onsets.png", "measured_loop_diagram.png",
)


def save(fig, output: Path) -> None:
    fig.tight_layout(); fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plot_cells(summary: pd.DataFrame, output: Path, title: str, ylabel: str) -> None:
    periods = [period for period in ("interictal", "preictal", "ictal") if period in set(summary.period)]
    x = np.arange(len(periods)); fig, ax = plt.subplots(figsize=(9, 5.5))
    for group in ("FS", "ES"):
        for semiology in ("hyper", "hypo"):
            label = f"{group} {semiology}"
            part = (summary[summary.group.eq(group) & summary.semiology.eq(semiology)]
                    .set_index("period").reindex(periods))
            y = part["median"].to_numpy(float); low = part.ci95_low.to_numpy(float); high = part.ci95_high.to_numpy(float)
            ax.errorbar(x, y, yerr=[y - low, high - y], marker="o", capsize=3, lw=1.5,
                        color=GROUP_COLORS[label], label=label)
    ax.set(xticks=x, xticklabels=[period.title() for period in periods], ylabel=ylabel, title=title)
    ax.legend(frameon=False, ncol=2); ax.grid(axis="y", alpha=.2); save(fig, output)


def plot_coverage(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, phase in zip(axes, ("preictal", "ictal")):
        exact_phase = summary.contrast.str.rsplit(n=1).str[-1].eq(phase)
        for state in ALL_STATES:
            part = summary[summary.microstate.eq(state) & exact_phase]; x = np.arange(len(part))
            ax.errorbar(x, part["median"], yerr=[part["median"] - part.ci95_low, part.ci95_high - part["median"]],
                        marker="o", capsize=2, label=state)
        labels = summary[exact_phase].contrast.drop_duplicates().str.replace(f" {phase}$", "", regex=True)
        ax.set(xticks=np.arange(len(labels)), xticklabels=labels, title=phase.title(), ylabel="Coverage", ylim=(0, .45))
        ax.tick_params(axis="x", rotation=30)
    axes[0].legend(frameon=False, ncol=3)
    fig.suptitle("Microstate coverage by seizure type, semiology and phase (posterior median, 95% CrI)")
    save(fig, output)


def plot_autonomic(participant: pd.DataFrame, cells: pd.DataFrame,
                    contrasts: pd.DataFrame, output: Path) -> None:
    groups = (("FS", "hyper"), ("FS", "hypo"), ("ES", "hyper"), ("ES", "hypo"))
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), gridspec_kw={"width_ratios": [1.35, 1]})
    rng = np.random.default_rng(29)
    for row, index in enumerate(("CSI", "CPI")):
        ax = axes[row, 0]
        for position, (group, semiology) in enumerate(groups):
            event_type = "functional" if group == "FS" else "epileptic"
            part = participant[participant.event_type.eq(event_type) & participant.semiology.eq(semiology)]
            paired = (part.pivot_table(index="participant", columns="period", values=index.lower(), aggfunc="mean")
                      .dropna(subset=["interictal", "preictal"]))
            color = GROUP_COLORS[f"{group} {semiology}"]; left, right = position - .14, position + .14
            jitter = rng.uniform(-.035, .035, len(paired))
            for offset, values in zip(jitter, paired[["interictal", "preictal"]].to_numpy(float)):
                ax.plot([left + offset, right + offset], values, color=color, alpha=.09, lw=.7)
            ax.scatter(left + jitter, paired.interictal, s=8, color=color, alpha=.14)
            ax.scatter(right + jitter, paired.preictal, s=8, color=color, alpha=.14)
            for period, x in (("interictal", left), ("preictal", right)):
                estimate = cells[cells["index"].eq(index) & cells.group.eq(group) &
                                 cells.semiology.eq(semiology) & cells.period.eq(period)].iloc[0]
                ax.errorbar(x, estimate["median"],
                            yerr=[[estimate["median"] - estimate.ci95_low], [estimate.ci95_high - estimate["median"]]],
                            fmt="o", color=color, markerfacecolor="white" if period == "interictal" else color,
                            markeredgewidth=1.5, ms=7, capsize=3, zorder=4)
        ax.set(xticks=np.arange(len(groups)), xticklabels=[f"{group} {semiology.title()}" for group, semiology in groups],
               ylabel=f"{index} (index units)",
               title="Participant-matched levels: II (open), final preictal minute (filled)")
        ax.set_yscale("log"); ax.grid(axis="y", alpha=.18)

        part = contrasts[contrasts["index"].eq(index) & ~contrasts.contrast.str.contains(":")]
        y = np.arange(len(part)); estimate = part["median"].to_numpy(float)
        axes[row, 1].errorbar(estimate, y, xerr=[estimate - part.ci95_low, part.ci95_high - estimate], fmt="o", capsize=3)
        axes[row, 1].axvline(0, color=".5", lw=1)
        labels = [f"{item.contrast}\nP(>0)={item.probability_gt_zero:.3f}" for item in part.itertuples()]
        axes[row, 1].set(yticks=y, yticklabels=labels, xlabel=f"Log {index} ratio",
                         title="Bayesian factorial contrasts, 95% CrI")
        axes[row, 1].invert_yaxis(); axes[row, 1].grid(axis="x", alpha=.18)
    fig.suptitle("Robust beat-centred 15-second autonomic levels: interictal versus final preictal minute\n"
                 "Thin lines are matched participant cells; large points are DD-adjusted posterior cell means")
    save(fig, output)


def plot_forest(summary: pd.DataFrame, output: Path, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10, max(4, .6 * len(summary)))); y = np.arange(len(summary)); x = summary["median"].to_numpy(float)
    labels = [f"{getattr(row, 'outcome', '')} {row.contrast}\nP(>0)={row.probability_gt_zero:.3f}" for row in summary.itertuples()]
    ax.errorbar(x, y, xerr=[x - summary.ci95_low, summary.ci95_high - x], fmt="o", capsize=3)
    ax.axvline(0, color=".5", lw=1); ax.set(yticks=y, yticklabels=labels, xlabel=xlabel, title=title)
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.2); save(fig, output)


def box_with_points(ax, arrays: list[np.ndarray], labels: list[str], colors: list[str]) -> None:
    boxes = ax.boxplot(arrays, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(.25)
    rng = np.random.default_rng(27)
    for position, (values, color) in enumerate(zip(arrays, colors), 1):
        ax.scatter(position + rng.uniform(-.12, .12, len(values)), values, color=color, alpha=.45, s=18)
    ax.grid(axis="y", alpha=.2)


def plot_subsidiary_transition(windows: pd.DataFrame, pairs: pd.DataFrame,
                               summary: pd.DataFrame, output: Path) -> None:
    frame = windows[windows.window_end_to_ictus_s.eq(0) & windows.semiology.notna()].copy()
    frame["is_functional"] = frame.stratum.isin(FUNCTIONAL).astype(int)
    frame["log_js"] = np.log(frame.js_from_II.where(frame.js_from_II.gt(0)))
    frame = frame[~frame.stratum.astype(str).str.startswith("DD")]
    frame["group"] = np.where(frame.is_functional.eq(1), "FS", "ES")
    lookup = {(row.outcome, row.contrast): row for row in summary.itertuples()}
    fig, axes = plt.subplots(2, 2, figsize=(13, 10)); colors = ["#7A5195", "#009E73"]
    for column, (outcome, eligible, ylabel) in enumerate((
            ("JS", "log_js", "C-origin JS divergence from interictal"),
            ("kappa", "log_kappa", "C-origin kappa"))):
        participant = (frame[np.isfinite(frame[eligible])]
                       .groupby(["participant", "group"], observed=True)[eligible].mean().reset_index())
        arrays = [np.exp(participant.loc[participant.group.eq(group), eligible].to_numpy(float)) for group in ("ES", "FS")]
        box_with_points(axes[0, column], arrays, ["ES", "FS"], colors)
        result = lookup[(outcome, "FS minus ES excluding DD")]
        axes[0, column].set(ylabel=ylabel,
                            title=(f"Non-DD participant geometric means\nFS - ES log effect={result.median:.3f} "
                                   f"[{result.ci95_low:.3f}, {result.ci95_high:.3f}]; "
                                   f"P(>0)={result.probability_gt_zero:.3f}"))

        use = pairs.dropna(subset=[f"{eligible}_epileptic", f"{eligible}_functional", f"{eligible}_fs_minus_es"])
        for row in use.itertuples(index=False):
            axes[1, column].plot([1, 2], np.exp([getattr(row, f"{eligible}_epileptic"),
                                                getattr(row, f"{eligible}_functional")]),
                                 color=".65", lw=.7, alpha=.45, zorder=1)
        arrays = [np.exp(use[f"{eligible}_epileptic"].to_numpy(float)),
                  np.exp(use[f"{eligible}_functional"].to_numpy(float))]
        box_with_points(axes[1, column], arrays, ["ES", "FS"], colors)
        result = lookup[(outcome, "FS minus ES within DD (paired)")]
        axes[1, column].set(ylabel=ylabel,
                            title=(f"Within-DD paired geometric means (n={len(use)})\n"
                                   f"Adjusted FS - ES log effect={result.median:.3f} "
                                   f"[{result.ci95_low:.3f}, {result.ci95_high:.3f}]; "
                                   f"P(>0)={result.probability_gt_zero:.3f}"))
        axes[0, column].set_yscale("log"); axes[1, column].set_yscale("log")
    fig.suptitle("Subsidiary final-minute C-origin comparisons (participant geometric means)"); save(fig, output)


def plot_c_origin_boxplots(windows: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    frame = windows[windows.window_end_to_ictus_s.eq(0)].copy()
    frame["group"] = np.select([frame.stratum.isin(FUNCTIONAL), frame.stratum.isin(EPILEPTIC)], ["FS", "ES"], default="")
    frame["log_js"] = np.log(frame.js_from_II.where(frame.js_from_II.gt(0)))
    event = (frame[frame.group.ne("")].groupby(["participant", "file_path", "group", "semiology"], observed=True)
             [["log_js", "log_kappa"]].mean().reset_index())
    overall = event.groupby(["participant", "group"], observed=True)[["log_js", "log_kappa"]].mean().reset_index()
    semio = event[event.group.eq("FS")].groupby(["participant", "semiology"], observed=True)[["log_js", "log_kappa"]].mean().reset_index()
    lookup = {(row.outcome, row.contrast): row for row in summary.itertuples()}
    panels = [
        (overall, "group", ("ES", "FS"), "log_js", "C-origin JS", ("JS", "FS minus ES"), "FS minus ES"),
        (overall, "group", ("ES", "FS"), "log_kappa", "C-origin \u03ba", ("kappa", "FS minus ES"), "FS minus ES"),
        (semio, "semiology", ("hyper", "hypo"), "log_js", "C-origin JS", ("JS", "Hypomotor minus hypermotor within FS"), "Hypomotor minus hypermotor (FS)"),
        (semio, "semiology", ("hyper", "hypo"), "log_kappa", "C-origin \u03ba", ("kappa", "Hypomotor minus hypermotor within FS"), "Hypomotor minus hypermotor (FS)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (data, grouping, levels, outcome, ylabel, key, title) in zip(axes.flat, panels):
        arrays = [np.exp(data.loc[data[grouping].eq(level), outcome].dropna().to_numpy(float)) for level in levels]
        colors = ["#7A5195", "#009E73"] if grouping == "group" else [GROUP_COLORS["FS hyper"], GROUP_COLORS["FS hypo"]]
        box_with_points(ax, arrays, [level.upper() if grouping == "group" else level.title() for level in levels], colors)
        result = lookup[key]
        ax.set_yscale("log")
        ax.set(ylabel=ylabel, title=(f"{title}\nadjusted log effect={result.median:.3f} "
                                    f"[{result.ci95_low:.3f}, {result.ci95_high:.3f}]; "
                                    f"P(>0)={result.probability_gt_zero:.3f}"))
    fig.suptitle("Final-minute C-origin transition metrics (participant geometric means)"); save(fig, output)


def plot_js_recovery_boxplots(paired: pd.DataFrame, recovery: pd.DataFrame, output: Path) -> None:
    paired = paired[np.isfinite(paired[["pre_js", "post_js", "log_post_pre_js"]]).all(axis=1) &
                    paired.pre_js.gt(0) & paired.post_js.gt(0)].copy()
    paired["log_pre_js"] = np.log(paired.pre_js); paired["log_post_js"] = np.log(paired.post_js)
    overall = paired.groupby("participant", observed=True)[["log_pre_js", "log_post_js"]].mean().reset_index()
    semio = (paired.groupby(["participant", "semiology"], observed=True)[["log_pre_js", "log_post_js"]]
             .mean().reset_index())
    for frame in (overall, semio):
        frame["pre_js"] = np.exp(frame.log_pre_js); frame["post_js"] = np.exp(frame.log_post_js)
    result = {row.contrast: row for row in recovery.itertuples()}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    box_with_points(axes[0], [overall.pre_js.to_numpy(), overall.post_js.to_numpy()],
                    ["PreI", "PostI"], ["#CC79A7", "#009E73"])
    for row in overall.itertuples():
        axes[0].plot([1, 2], [row.pre_js, row.post_js], color=".5", alpha=.12, lw=.7)
    estimate = result["Overall postictal minus preictal"]
    axes[0].set(ylabel="C-origin JS divergence from interictal baseline",
                title=(f"All FS\nadjusted log(PostI/PreI)={estimate.median:.3f} "
                       f"[{estimate.ci95_low:.3f}, {estimate.ci95_high:.3f}]; "
                       f"P(<0)={estimate.probability_lt_zero:.3f}"))
    levels = (("hyper", "pre_js"), ("hyper", "post_js"), ("hypo", "pre_js"), ("hypo", "post_js"))
    arrays = [semio.loc[semio.semiology.eq(group), phase].to_numpy(float) for group, phase in levels]
    box_with_points(axes[1], arrays, ["Hyper\nPreI", "Hyper\nPostI", "Hypo\nPreI", "Hypo\nPostI"],
                    [GROUP_COLORS["FS hyper"], "#E69F00", GROUP_COLORS["FS hypo"], "#56B4E9"])
    for group, positions in (("hyper", (1, 2)), ("hypo", (3, 4))):
        for row in semio[semio.semiology.eq(group)].itertuples():
            axes[1].plot(positions, [row.pre_js, row.post_js], color=".5", alpha=.12, lw=.7)
    hyper = result["Hypermotor postictal minus preictal"]; hypo = result["Hypomotor postictal minus preictal"]
    axes[1].set(title=("FS by semiology: adjusted log(PostI/PreI), 95% CrI\n"
                       f"hyper={hyper.median:.3f} [{hyper.ci95_low:.3f}, {hyper.ci95_high:.3f}]; "
                       f"hypo={hypo.median:.3f} [{hypo.ci95_low:.3f}, {hypo.ci95_high:.3f}]"))
    axes[0].set_yscale("log"); axes[1].set_yscale("log")
    fig.suptitle("Postictal correction of C-origin transition divergence (participant geometric means)"); save(fig, output)


def fixed_posterior(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    with np.load(path) as saved:
        names = saved["parameter_names"].astype(str).tolist()
        draws = saved["draws"][..., :len(names)].reshape(-1, len(names))
    return draws, {name: index for index, name in enumerate(names)}


def regression_panel(ax, frame: pd.DataFrame, x_name: str, y_name: str, draws: np.ndarray,
                     intercept: np.ndarray, slope: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    use = frame[[x_name, y_name]].replace([np.inf, -np.inf], np.nan).dropna()
    low, high = use[x_name].quantile([.01, .99]); use = use[use[x_name].between(low, high)]
    ax.scatter(use[x_name], use[y_name], color="#4C78A8", alpha=.07, s=9)
    grid = np.linspace(low, high, 120); prediction = intercept[:, None] + slope[:, None] * grid
    lo, middle, hi = np.quantile(prediction, [.025, .5, .975], axis=0)
    beta_low, beta, beta_high = np.quantile(slope, [.025, .5, .975])
    ax.plot(grid, middle, color="#D55E00", lw=2); ax.fill_between(grid, lo, hi, color="#D55E00", alpha=.2)
    ax.set(xlabel=xlabel, ylabel=ylabel,
           title=f"{title}\n\u03b2={beta:.3f} [{beta_low:.3f}, {beta_high:.3f}]"); ax.grid(alpha=.2)


def plot_cpi_kappa_regression(common: pd.DataFrame, functional_participants: set[str],
                              data_dir: Path, output: Path) -> None:
    frame = common[common.participant.astype(str).isin(functional_participants) & common.condition.isin(("ii", "preict"))]
    draws, names = fixed_posterior(data_dir / "posterior_fs_c_origin_c_log_kappa_autonomic_association.npz")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    regression_panel(axes[0], frame[frame.condition.eq("ii")], "cpi_z", "c_log_kappa", draws,
                     draws[:, names["Intercept"]], draws[:, names["cpi_z"]],
                     "Interictal", "CPI (participant II SD)", "C-origin log \u03ba")
    regression_panel(axes[1], frame[frame.condition.eq("preict")], "cpi_z", "c_log_kappa", draws,
                     draws[:, names["Intercept"]] + draws[:, names["is_preict"]],
                     draws[:, names["cpi_z"]] + draws[:, names["cpi_z_preict"]],
                     "Preictal", "CPI (participant II SD)", "C-origin log \u03ba")
    fig.suptitle("Within-FS CPI association with C-origin transition precision\n"
                 "Adjusted posterior line and 95% credible band; central 98% of CPI shown\n"
                 "CSI, transition count and DD held at model reference values")
    save(fig, output)


def plot_csi_to_c_regression(common: pd.DataFrame, functional_participants: set[str],
                             data_dir: Path, output: Path) -> None:
    variables = ["c_expression", "csi_z", "cpi_z"]
    lagged = make_lags(within_record_z(common[common.condition.isin(("ii", "preict"))], variables),
                       [f"{name}_wz" for name in variables])
    frame = lagged[lagged.participant.astype(str).isin(functional_participants)].copy()
    frame["is_last_minute"] = (frame.condition.eq("preict") & frame.window_end_to_ictus_s.eq(0)).astype(int)
    frame = frame[frame.condition.eq("ii") | frame.is_last_minute.eq(1)]
    draws, names = fixed_posterior(data_dir / "posterior_fs_ii_versus_last_minute_flow_autonomic_to_c_coverage.npz")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    regression_panel(axes[0], frame[frame.condition.eq("ii")], "csi_z_wz_lag", "c_expression_wz", draws,
                     draws[:, names["Intercept"]], draws[:, names["csi_z_wz_lag"]],
                     "Interictal", "Prior-window CSI (within-record SD)", "Next-window C expression (within-record SD)")
    regression_panel(axes[1], frame[frame.is_last_minute.eq(1)], "csi_z_wz_lag", "c_expression_wz", draws,
                     draws[:, names["Intercept"]] + draws[:, names["is_last_minute"]],
                     draws[:, names["csi_z_wz_lag"]] + draws[:, names["last_x_csi_z_wz_lag"]],
                     "Final preictal minute", "Prior-window CSI (within-record SD)",
                     "Next-window C expression (within-record SD)")
    fig.suptitle("Within-FS CSI prediction of subsequent microstate-C expression\n"
                 "Adjusted posterior line and 95% credible band; central 98% of CSI shown\n"
                 "Own-lag and CPI-lag held at zero")
    save(fig, output)


def plot_preictal_kappa_js(participant: pd.DataFrame, lines: pd.DataFrame, output: Path) -> None:
    lookup = lines.set_index(["group", "semiology"])
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, color in GROUP_COLORS.items():
        group, semiology = label.split(); part = participant[participant.group.eq(group) & participant.semiology.eq(semiology)]
        if len(part) < 3 or (group, semiology) not in lookup.index:
            continue
        estimate = lookup.loc[(group, semiology)]; grid = np.linspace(part.log_kappa.min(), part.log_kappa.max(), 100)
        ax.scatter(part.kappa, part.js_from_II, color=color, alpha=.45, s=28)
        ax.plot(np.exp(grid), estimate.intercept + estimate.slope_per_log_kappa * grid,
                color=color, lw=2, label=f"{label} (n={int(estimate.n_participants)})")
    ax.set_xscale("log")
    ax.set(xlabel="C-origin \u03ba (log scale)", ylabel="C-origin JS divergence from interictal baseline",
           title="Preictal C-origin transition precision versus divergence")
    ax.grid(alpha=.2); ax.legend(frameon=False); save(fig, output)


def plot_raw_count(primary: pd.DataFrame, sensitivity: pd.DataFrame,
                   recovery: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    panels = ((primary[primary.contrast.eq("FS minus ES")], "Raw C-exit counts",
               lambda row: f"{row.estimand}\n{row.outcome}"),
              (sensitivity, "C-label threshold sensitivity",
               lambda row: f"|r| ≥ {row.threshold:.2f}"),
              (recovery, "Equal-information recovery",
               lambda row: f"{row.estimand}\n{row.outcome}"))
    for ax, (part, title, label) in zip(axes, panels):
        part = part.reset_index(drop=True); y = np.arange(len(part)); estimate = part["median"].to_numpy(float)
        ax.errorbar(estimate, y, xerr=[estimate - part.ci95_low, part.ci95_high - estimate], fmt="o", capsize=3)
        ax.axvline(0, color=".5", lw=1)
        ax.set(yticks=y, yticklabels=[label(row) for row in part.itertuples()], xlabel="Posterior log effect", title=title)
        ax.invert_yaxis(); ax.grid(axis="x", alpha=.2)
    fig.suptitle("C-origin transition result after raw-count uncertainty, fixed-exit and matched-duration checks")
    save(fig, output)


def plot_route_destinations(summary: pd.DataFrame, participant: pd.DataFrame, output: Path) -> None:
    destinations = [state for state in ALL_STATES if state != "C"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    x = np.arange(len(destinations)); width = .34
    for offset, (functional, label, color) in zip((-width / 2, width / 2),
                                                  ((0, "ES", "#7A5195"), (1, "FS", "#009E73"))):
        means, errors = [], []
        for index, state in enumerate(destinations):
            column = f"delta_p_{state}" if f"delta_p_{state}" in participant else f"delta_p_{index}"
            values = participant.loc[participant.is_functional.eq(functional), column].dropna()
            means.append(values.mean()); errors.append(values.sem())
        axes[0].bar(x + offset, means, width, yerr=errors, capsize=2, color=color, alpha=.75, label=label)
    axes[0].axhline(0, color=".5", lw=1); axes[0].set(xticks=x, xticklabels=destinations,
        xlabel="Destination", ylabel="Preictal minus II probability", title="Observed participant means ± SE")
    axes[0].legend(frameon=False)

    for ax, outcome, title in ((axes[1], "delta_probability", "FS−ES destination shift"),
                               (axes[2], "js_contribution", "FS−ES JS contribution")):
        part = summary[summary.outcome.eq(outcome)].set_index("destination").reindex(destinations)
        estimate = part["median"].to_numpy(float); y = np.arange(len(part))
        ax.errorbar(estimate, y, xerr=[estimate - part.ci95_low, part.ci95_high - estimate], fmt="o", capsize=3)
        ax.axvline(0, color=".5", lw=1); ax.set(yticks=y, yticklabels=destinations,
            xlabel="Posterior FS minus ES effect", title=title); ax.invert_yaxis(); ax.grid(axis="x", alpha=.2)
    fig.suptitle("Which exits from C drive the transition-regime difference?")
    save(fig, output)


def plot_all_state_specificity(summary: pd.DataFrame, output: Path) -> None:
    order = [*ALL_STATES, "global", "C-specificity", "C-minus-global"]
    part = summary[summary.outcome.eq("JS")].set_index("source_state").reindex(order).dropna(subset=["median"])
    fig, ax = plt.subplots(figsize=(11, 7)); y = np.arange(len(part)); estimate = part["median"].to_numpy(float)
    colors = ["#D55E00" if state.startswith("C") else ("black" if state == "global" else "#4C78A8") for state in part.index]
    for yi, value, low, high, color in zip(y, estimate, part.ci95_low, part.ci95_high, colors):
        ax.errorbar(value, yi, xerr=[[value - low], [high - value]], fmt="o", color=color, capsize=3)
    labels = [state if state in ALL_STATES or state == "global" else part.loc[state, "contrast"] for state in part.index]
    ax.axvline(0, color=".5", lw=1); ax.set(yticks=y, yticklabels=labels,
        xlabel="Posterior FS minus ES difference in JS", title="Direct test of C specificity with equal row weights")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.2); save(fig, output)


def plot_physiology_trajectory(summary: pd.DataFrame, output: Path) -> None:
    panels = (("js", "C-origin JS"), ("log_kappa", "C-origin κ"), ("c_expression", "C coverage"),
              ("csi_z", "CSI (II SD)"), ("cpi_z", "CPI (II SD)"), ("mean_hr_bpm", "Heart rate (bpm)"))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    for ax, (outcome, title) in zip(axes.flat, panels):
        for group, color in (("FS", "#009E73"), ("ES", "#7A5195")):
            part = summary[summary.outcome.eq(outcome) & summary.contrast.eq(group)].sort_values("minutes_to_ictus")
            x = part.minutes_to_ictus.to_numpy(float); middle = part["median"].to_numpy(float)
            low, high = part.ci95_low.to_numpy(float), part.ci95_high.to_numpy(float)
            if outcome == "log_kappa":
                middle, low, high = np.exp(middle), np.exp(low), np.exp(high)
            ax.plot(x, middle, color=color, label=group); ax.fill_between(x, low, high, color=color, alpha=.16)
        ax.axvline(0, color=".5", lw=1); ax.set(title=title, xlabel="Minutes to onset"); ax.grid(alpha=.18)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Aligned preictal cortical and cardiac physiology (participant-level means, 95% intervals)")
    save(fig, output)


def plot_js_kappa_state_space(events: pd.DataFrame, output: Path) -> None:
    part = events[events.estimand.eq("fixed 60-second duration")]
    part = part.groupby(["participant", "is_functional"], observed=True)[["js", "kappa"]].mean().reset_index()
    x0, y0 = part.js.median(), part.kappa.median()
    fig, ax = plt.subplots(figsize=(9, 7))
    for functional, label, color in ((0, "ES", "#7A5195"), (1, "FS", "#009E73")):
        use = part[part.is_functional.eq(functional)]
        ax.scatter(use.js, use.kappa, s=24, alpha=.5, color=color, label=label)
    ax.axvline(x0, color=".6", lw=1); ax.axhline(y0, color=".6", lw=1)
    ax.text(.98, .96, "Distinct organised regime", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.text(.98, .04, "Divergent diffuse regime", transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    ax.text(.02, .96, "Baseline-like organised", transform=ax.transAxes, ha="left", va="top", fontsize=9)
    ax.text(.02, .04, "Baseline-like diffuse", transform=ax.transAxes, ha="left", va="bottom", fontsize=9)
    ax.set(xlabel="C-origin JS divergence from II", ylabel="C-origin κ",
           title="C-origin transition regime: divergence and concentration are distinct axes")
    ax.set_yscale("log"); ax.legend(frameon=False); ax.grid(alpha=.15); save(fig, output)


def plot_pseudo_onsets(matches: pd.DataFrame, output: Path) -> None:
    event = matches.groupby(["participant", "event_key", "is_functional"], observed=True)[["true_js", "pseudo_js"]].mean().reset_index()
    part = event.groupby(["participant", "is_functional"], observed=True)[["true_js", "pseudo_js"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, functional, label, color in zip(axes, (0, 1), ("ES", "FS"), ("#7A5195", "#009E73")):
        use = part[part.is_functional.eq(functional)]
        for row in use.itertuples():
            ax.plot([1, 2], [row.pseudo_js, row.true_js], color=color, alpha=.18, lw=.8)
        box_with_points(ax, [use.pseudo_js.to_numpy(), use.true_js.to_numpy()], ["Matched II\npseudo", "Genuine\npreictal"], [color, color])
        ax.set(title=f"{label} (n={len(use)})", ylabel="C-origin JS divergence from participant II")
    fig.suptitle("Genuine final preictal windows versus within-participant matched interictal pseudo-onsets")
    save(fig, output)


def plot_loop_diagram(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6)); ax.axis("off")
    boxes = {"A": ((.08, .39), "Autonomic observations\nCSI / CPI / RR / HR"),
             "O": ((.40, .39), "C expression\nOᶜ = C GFP peaks / all peaks"),
             "R": ((.72, .39), "C-origin grammar\nJSᶜ, κᶜ, p(C→j)"),
             "S": ((.40, .78), "Shared preictal state Sₜ\nproposed, not identified")}
    for (x, y), text_value in boxes.values():
        ax.text(x + .1, y + .08, text_value, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=.6", fc="white", ec=".25", lw=1.4))
    def arrow(start, end, dashed=False):
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.8,
                    color="#D55E00" if not dashed else ".45", linestyle="--" if dashed else "-"))
    arrow((.28, .47), (.40, .47)); arrow((.40, .43), (.28, .43))
    arrow((.60, .47), (.72, .47)); arrow((.72, .43), (.60, .43))
    arrow((.28, .52), (.72, .52)); arrow((.72, .38), (.28, .38))
    for target in ((.18, .57), (.50, .57), (.82, .57)):
        arrow((.50, .78), target, True)
    ax.text(.50, .14, "Solid: lagged effects tested in the present pipeline     Dashed: shared-cause model still required",
            ha="center", fontsize=10)
    ax.set_title("Measured coverage loop and transition-grammar loop", fontsize=15)
    save(fig, output)


def plot_all(analysis_dir: Path) -> list[Path]:
    data_dir = analysis_dir / "data"; figure_dir = analysis_dir / "figures"
    missing = [name for name in REQUIRED_INPUTS if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing saved plot inputs: " + ", ".join(missing))
    schema = json.loads((data_dir / "analysis_schema.json").read_text(encoding="utf-8")).get("schema")
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f"Unsupported analysis schema for plotting: {schema!r}")
    figure_dir.mkdir(parents=True, exist_ok=True)

    hr = pd.read_csv(data_dir / "hr_bayesian_cells.csv")
    coverage = pd.read_csv(data_dir / "coverage_bayesian_cells.csv")
    autonomic = pd.read_csv(data_dir / "autonomic_level_participant_cells.csv")
    autonomic_cells = pd.read_csv(data_dir / "autonomic_level_bayesian_cells.csv")
    autonomic_contrasts = pd.read_csv(data_dir / "autonomic_level_bayesian_contrasts.csv")
    primary = pd.read_csv(data_dir / "c_origin_primary_bayesian.csv")
    windows = pd.read_csv(data_dir / "c_origin_preictal_windows.csv")
    dd_pairs = pd.read_csv(data_dir / "c_origin_dd_paired_values.csv")
    subsidiary = pd.read_csv(data_dir / "c_origin_subsidiary_bayesian.csv")
    recovery = pd.read_csv(data_dir / "c_origin_js_recovery.csv")
    recovery_pairs = pd.read_csv(data_dir / "c_origin_js_recovery_paired_events.csv")
    associations = pd.read_csv(data_dir / "fs_autonomic_c_origin_correlations.csv")
    flow = pd.read_csv(data_dir / "bayesian_predictive_flow.csv")
    kappa_js = pd.read_csv(data_dir / "preictal_kappa_js_participant.csv")
    kappa_js_lines = pd.read_csv(data_dir / "preictal_kappa_js_lines.csv")
    common = pd.read_csv(data_dir / "autonomic_peak_microstate_common_windows.csv.gz")
    raw_count = pd.read_csv(data_dir / "raw_count_c_origin_bayesian.csv")
    raw_events = pd.read_csv(data_dir / "raw_count_c_origin_event_means.csv")
    reference_sensitivity = pd.read_csv(data_dir / "c_reference_threshold_sensitivity_bayesian.csv")
    matched_recovery = pd.read_csv(data_dir / "c_origin_recovery_matched_bayesian.csv")
    route = pd.read_csv(data_dir / "c_origin_route_destination_bayesian.csv")
    destinations = pd.read_csv(data_dir / "c_origin_destination_participant_means.csv")
    specificity = pd.read_csv(data_dir / "all_state_raw_count_specificity_bayesian.csv")
    physiology_trajectory = pd.read_csv(data_dir / "preictal_physiology_trajectory_bayesian.csv")
    pseudo_matches = pd.read_csv(data_dir / "matched_interictal_pseudo_onsets.csv")
    if "is_functional_participant" not in common:
        raise ValueError("Saved common-window data lack is_functional_participant")
    functional_participants = set(common.loc[common.is_functional_participant.eq(1), "participant"].astype(str))

    plot_cells(hr, figure_dir / "dense_rr_hr.png", "Dense RR-derived heart rate", "Mean HR (bpm)")
    plot_coverage(coverage, figure_dir / "microstate_coverage.png")
    plot_autonomic(autonomic, autonomic_cells, autonomic_contrasts,
                   figure_dir / "robust_autonomic_interictal_preictal.png")
    plot_forest(primary, figure_dir / "c_origin_primary.png",
                "C-origin transition divergence and precision", "Posterior log effect")
    plot_subsidiary_transition(windows, dd_pairs, subsidiary, figure_dir / "c_origin_subsidiary_raw.png")
    plot_forest(recovery, figure_dir / "c_origin_js_recovery.png",
                "C-origin JS: postictal correction", "Log postictal/preictal JS")
    plot_forest(associations[~associations.contrast.str.contains("minus")],
                figure_dir / "fs_autonomic_c_origin_correlations.png",
                "Within-FS autonomic associations with C-origin transitions", "Posterior regression coefficient")
    plot_forest(flow, figure_dir / "bayesian_predictive_flow.png",
                "Bayesian one-window predictive information flow", "Standardised lag coefficient")
    plot_preictal_kappa_js(kappa_js, kappa_js_lines, figure_dir / "preictal_kappa_vs_js.png")
    plot_c_origin_boxplots(windows, primary, figure_dir / "c_origin_final_minute_boxplots.png")
    plot_js_recovery_boxplots(recovery_pairs, recovery, figure_dir / "c_origin_js_recovery_boxplots.png")
    plot_cpi_kappa_regression(common, functional_participants, data_dir, figure_dir / "fs_cpi_kappa_regression.png")
    plot_csi_to_c_regression(common, functional_participants, data_dir, figure_dir / "fs_csi_to_c_regression.png")
    plot_raw_count(raw_count, reference_sensitivity, matched_recovery, figure_dir / "raw_count_c_origin.png")
    plot_route_destinations(route, destinations, figure_dir / "c_origin_route_destinations.png")
    plot_all_state_specificity(specificity, figure_dir / "all_state_specificity.png")
    plot_physiology_trajectory(physiology_trajectory, figure_dir / "preictal_physiology_trajectory.png")
    plot_js_kappa_state_space(raw_events, figure_dir / "c_origin_js_kappa_state_space.png")
    plot_pseudo_onsets(pseudo_matches, figure_dir / "matched_pseudo_onsets.png")
    plot_loop_diagram(figure_dir / "measured_loop_diagram.png")

    outputs = [figure_dir / name for name in FIGURE_NAMES]
    missing_outputs = [path.name for path in outputs if not path.is_file() or path.stat().st_size == 0]
    if missing_outputs:
        raise RuntimeError("Plotting completed without outputs: " + ", ".join(missing_outputs))
    audit = {"completed_utc": datetime.now(timezone.utc).isoformat(), "analysis_schema": schema,
             "renderer_sha256": sha256(Path(__file__)), "input_files": list(REQUIRED_INPUTS),
             "figure_sha256": {path.name: sha256(path) for path in outputs}}
    (analysis_dir / "PLOT_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return outputs


def self_check() -> None:
    rows = []
    for group in ("FS", "ES"):
        for semiology in ("hyper", "hypo"):
            for number, period in enumerate(("interictal", "preictal", "ictal")):
                rows.append({"group": group, "semiology": semiology, "period": period,
                             "median": 70 + number, "ci95_low": 69 + number, "ci95_high": 71 + number})
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "check.png"
        plot_cells(pd.DataFrame(rows), output, "check", "value")
        assert output.is_file() and output.stat().st_size > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(); print("plot self-check passed"); return
    if args.analysis_dir is None:
        parser.error("--analysis-dir is required")
    outputs = plot_all(args.analysis_dir)
    print(f"Regenerated {len(outputs)} figures in {args.analysis_dir / 'figures'}")


if __name__ == "__main__":
    main()
