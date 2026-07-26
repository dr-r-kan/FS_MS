"""Requested HR and statewise transition comparisons from completed pipeline outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import build_design_matrices
from scipy.stats import norm
from scipy.stats import t as t_dist, ttest_1samp

import analyse_ace_cardioneural as ace
from analyse_preictal_publication import attach_strata, piecewise_columns


FUNCTIONAL = ("FS functional", "DD functional")
EPILEPTIC = ("ES epileptic", "DD epileptic")
ALL_STRATA = FUNCTIONAL + EPILEPTIC
COLORS = {"hyper": "#D55E00", "hypo": "#0072B2", "functional": "#009E73", "epileptic": "#7A5195"}


def mean_ci(values) -> tuple[int, float, float, float]:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if len(x) < 2:
        return len(x), float(np.mean(x)) if len(x) else np.nan, np.nan, np.nan
    mean, se = float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))
    half = float(t_dist.ppf(.975, len(x) - 1) * se)
    return len(x), mean, mean - half, mean + half


def paired_test(frame: pd.DataFrame, index: list[str], columns: str, value: str,
                left: str, right: str, label: str) -> tuple[dict, pd.DataFrame]:
    wide = frame.pivot_table(index=index, columns=columns, values=value, aggfunc="mean").dropna(subset=[left, right])
    diff = wide[right] - wide[left]
    n, estimate, low, high = mean_ci(diff)
    test = ttest_1samp(diff, 0) if n >= 2 else None
    row = {"analysis": label, "left": left, "right": right, "estimate_right_minus_left": estimate,
           "ci95_low": low, "ci95_high": high, "n_pairs": n,
           "t": float(test.statistic) if test else np.nan, "p": float(test.pvalue) if test else np.nan}
    return row, wide.reset_index()


def add_design(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["event_type"] = np.where(out["stratum"].isin(FUNCTIONAL), "functional", "epileptic")
    out["study_cohort"] = np.where(out["stratum"].str.startswith("DD"), "DD", "single-diagnosis")
    out["is_functional"] = out["event_type"].eq("functional").astype(int)
    out["is_hypo"] = out["semiology"].eq("hypo").astype(int)
    out["is_dd"] = out["study_cohort"].eq("DD").astype(int)
    return out


def model_contrasts(frame: pd.DataFrame, outcome: str, label: str) -> list[dict]:
    fit, _ = ace.fit_gee(frame, f"{outcome} ~ is_functional + is_hypo + is_functional:is_hypo + is_dd", label)
    specs = (
        ("Epileptic: hypomotor minus hypermotor", {"is_hypo": 1}),
        ("Functional: hypomotor minus hypermotor", {"is_hypo": 1, "is_functional:is_hypo": 1}),
        ("Hypermotor: functional minus epileptic", {"is_functional": 1}),
        ("Hypomotor: functional minus epileptic", {"is_functional": 1, "is_functional:is_hypo": 1}),
    )
    return [ace.linear_contrast(fit, weights, name) for name, weights in specs]


def final_minute_slopes(manifest: pd.DataFrame, cardiac: pd.DataFrame, ecg_qc_dir: Path) -> pd.DataFrame:
    valid = set(cardiac.loc[cardiac["window_end_to_ictus_s"].eq(0) & cardiac["hr_window_valid"], "file_path"])
    rows = []
    events = manifest[manifest["condition"].astype(str).str.lower().eq("preict") & manifest["stratum"].isin(ALL_STRATA)]
    for event in events.itertuples(index=False):
        if event.file_path not in valid:
            continue
        peaks, recovered = ace.read_rpeaks_with_recovery_flags(ace.rpeak_file(ecg_qc_dir, event.source_raw_fif))
        start, stop = float(event.crop_start_seconds), float(event.crop_end_seconds)
        onset = stop - start
        keep = (peaks >= start - 2) & (peaks <= stop + 1)
        peaks, recovered = peaks[keep] - start, recovered[keep]
        rr = np.diff(peaks)
        mids = (peaks[:-1] + peaks[1:]) / 2
        flags = recovered[:-1] | recovered[1:]
        keep = (mids >= onset - 60) & (mids <= onset) & (rr > 0)
        hr, time = 60 / rr[keep], (mids[keep] - onset) / 60
        if len(hr) < 20 or np.ptp(time) < .5 or np.mean(flags[keep]) > .10 or np.any((hr < 35) | (hr > 140)):
            continue
        rows.append({"participant": event.participant, "file_path": event.file_path, "stratum": event.stratum,
                     "semiology": str(event.hypo_hyper).lower(), "hr_gradient_bpm_per_min": np.polyfit(time, hr, 1)[0],
                     "rr_intervals": len(hr), "time_span_min": np.ptp(time)})
    return add_design(pd.DataFrame(rows))


def forest(ax, frame: pd.DataFrame, group: str, title: str, xlabel: str, p_col: str = "p_bh") -> None:
    states = sorted(frame["template_label"].unique())
    offsets = {name: value for name, value in zip(frame[group].unique(), np.linspace(-.16, .16, frame[group].nunique()))}
    for name, part in frame.groupby(group, observed=True):
        part = part.set_index("template_label").reindex(states)
        y = np.arange(len(states)) + offsets[name]
        x = part["estimate"].to_numpy(float)
        ax.errorbar(x, y, xerr=[x - part["ci95_low"], part["ci95_high"] - x], fmt="o", capsize=3,
                    label=str(name).capitalize(), color=COLORS.get(str(name), None))
        for xi, yi, p in zip(x, y, part[p_col]):
            ax.annotate(f"p={p:.3g}", (xi, yi), xytext=(5, 0), textcoords="offset points", va="center", fontsize=7)
    ax.axvline(0, color=".45", lw=1)
    ax.set(yticks=np.arange(len(states)), yticklabels=states, xlabel=xlabel, title=title)
    ax.legend(frameon=False, fontsize=8)


def plot_hr(trajectory: pd.DataFrame, final5: pd.DataFrame, stats: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for semio in ("hyper", "hypo"):
        part = trajectory[trajectory["semiology"].eq(semio)]
        rows = []
        for minute, values in part.groupby("minutes_to_ictus")["delta_hr_bpm"]:
            n, mean, low, high = mean_ci(values); rows.append((minute, n, mean, low, high))
        summary = pd.DataFrame(rows, columns=["minute", "n", "mean", "low", "high"]).sort_values("minute")
        axes[0].plot(summary.minute, summary["mean"], label=semio.capitalize(), color=COLORS[semio], lw=2)
        axes[0].fill_between(summary.minute, summary.low, summary.high, color=COLORS[semio], alpha=.18)
    axes[0].axhline(0, color=".45", lw=1); axes[0].axvline(-5, color=".65", lw=1, ls="--")
    axes[0].set(xlabel="Minutes to seizure onset", ylabel="HR change from −20 to −5 min baseline (bpm)",
                title="Preictal functional-seizure HR")
    axes[0].legend(frameon=False)
    rng = np.random.default_rng(7)
    for x, semio in enumerate(("hyper", "hypo")):
        values = final5.loc[final5.semiology.eq(semio), "mean_delta_hr_last5"]
        axes[1].scatter(x + rng.uniform(-.09, .09, len(values)), values, s=10, alpha=.22, color=COLORS[semio])
        n, mean, low, high = mean_ci(values)
        axes[1].errorbar(x, mean, yerr=[[mean-low], [high-mean]], fmt="o", ms=7, capsize=4, color=COLORS[semio])
    result = stats.iloc[0]
    axes[1].set(xticks=[0, 1], xticklabels=["Hyper", "Hypo"], ylabel="Mean HR change in final 5 min (bpm)",
                title=f"Hypo − hyper = {result.estimate:.2f} bpm\n95% CI {result.ci95_low:.2f} to {result.ci95_high:.2f}; Holm p={result.p_holm:.3g}")
    fig.tight_layout(); fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_gradient(participant: pd.DataFrame, stat: pd.Series, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5)); rng = np.random.default_rng(9)
    for x, semio in enumerate(("hyper", "hypo")):
        values = participant.loc[participant.semiology.eq(semio), "hr_gradient_bpm_per_min"]
        ax.scatter(x + rng.uniform(-.1, .1, len(values)), values, s=12, alpha=.25, color=COLORS[semio])
        n, mean, low, high = mean_ci(values)
        ax.errorbar(x, mean, yerr=[[mean-low], [high-mean]], fmt="o", ms=7, capsize=4, color=COLORS[semio])
    ax.axhline(0, color=".45", lw=1)
    ax.set(xticks=[0, 1], xticklabels=["Hyper", "Hypo"], ylabel="Within-final-minute HR slope (bpm/min)",
           title=f"Last-minute HR gradient\nHypo − hyper = {stat.estimate:.2f}; 95% CI {stat.ci95_low:.2f} to {stat.ci95_high:.2f}; Holm p={stat.p_holm:.3g}")
    fig.tight_layout(); fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_last30_means(descriptive: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    labels = sorted(descriptive.template_label.unique()); x = np.arange(len(labels)); groups = list(descriptive.group.unique())
    offsets = np.linspace(-.24, .24, len(groups))
    for ax, outcome, ylabel in zip(axes, ("log_kappa", "js_from_II"), ("log κ", "JS divergence from II")):
        for offset, group in zip(offsets, groups):
            part = descriptive[(descriptive.outcome.eq(outcome)) & descriptive.group.eq(group)].set_index("template_label").reindex(labels)
            mean = part["mean"].to_numpy(float)
            ax.errorbar(x + offset, mean, yerr=[mean-part.ci95_low, part.ci95_high-mean], fmt="o", capsize=2, label=group)
        ax.set(ylabel=ylabel); ax.grid(axis="y", alpha=.2)
    axes[0].set_title("Final 30 seconds: statewise transition precision and divergence")
    axes[1].set(xticks=x, xticklabels=labels, xlabel="Microstate origin")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_last30_contrasts(stats: pd.DataFrame, output: Path) -> None:
    contrasts = list(stats.contrast.unique())
    fig, axes = plt.subplots(len(contrasts), 2, figsize=(12, 3 * len(contrasts)), squeeze=False)
    for row, contrast in enumerate(contrasts):
        for col, (outcome, xlabel) in enumerate((("log_kappa", "Difference in log κ"), ("js_from_II", "Difference in JS from II"))):
            part = stats[(stats.contrast.eq(contrast)) & stats.outcome.eq(outcome)].sort_values("template_label")
            y = np.arange(len(part)); x = part.estimate.to_numpy(float)
            axes[row, col].errorbar(x, y, xerr=[x-part.ci95_low, part.ci95_high-x], fmt="o", capsize=3)
            axes[row, col].axvline(0, color=".45", lw=1)
            axes[row, col].set(yticks=y, yticklabels=[f"{s}  pBH={p:.3g}" for s, p in zip(part.template_label, part.p_bh)],
                               xlabel=xlabel, title=contrast)
    fig.tight_layout(); fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_prepost(stats: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (outcome, xlabel) in zip(axes, (("log_kappa", "Postictal − preictal log κ"),
                                           ("js_from_II", "Postictal − preictal JS from II"))):
        forest(ax, stats[stats.outcome.eq(outcome)], "semiology", "Functional seizures: paired pre/post change", xlabel)
    fig.tight_layout(); fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_dd(stats: pd.DataFrame, hr_pairs: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if not hr_pairs.empty:
        for row in hr_pairs.itertuples(index=False):
            axes[0].plot([0, 1], [row.epileptic, row.functional], color=".75", lw=.7, alpha=.6)
        axes[0].scatter(np.zeros(len(hr_pairs)), hr_pairs.epileptic, s=12, alpha=.5, color=COLORS["epileptic"])
        axes[0].scatter(np.ones(len(hr_pairs)), hr_pairs.functional, s=12, alpha=.5, color=COLORS["functional"])
    hr = stats[stats.outcome.eq("mean_delta_hr_last5")].iloc[0]
    axes[0].set(xticks=[0, 1], xticklabels=["ES", "FS"], ylabel="Final-5-min HR change (bpm)",
                title=f"DD paired HR (n={int(hr.n_pairs)})\nFS − ES={hr.estimate:.2f}; pBH={hr.p_bh:.3g}")
    for ax, (outcome, xlabel) in zip(axes[1:], (("log_kappa", "FS − ES log κ"), ("js_from_II", "FS − ES JS from II"))):
        part = stats[stats.outcome.eq(outcome)].sort_values("template_label")
        y = np.arange(len(part)); x = part.estimate.to_numpy(float)
        ax.errorbar(x, y, xerr=[x-part.ci95_low, part.ci95_high-x], fmt="o", capsize=3)
        ax.axvline(0, color=".45", lw=1)
        ax.set(yticks=y, yticklabels=[f"{s}  pBH={p:.3g}" for s, p in zip(part.template_label, part.p_bh)], xlabel=xlabel,
               title="DD paired statewise comparison")
    fig.tight_layout(); fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig)


def vector_contrast(fit, vector: np.ndarray, name: str) -> dict:
    estimate = float(vector @ fit.params.to_numpy(float))
    se = float(np.sqrt(vector @ fit.cov_params().to_numpy(float) @ vector))
    z = estimate / se if se > 0 else np.nan
    return {"contrast": name, "estimate": estimate, "se": se, "z": z, "p": float(2 * norm.sf(abs(z))),
            "ci95_low": estimate - 1.96 * se, "ci95_high": estimate + 1.96 * se}


def dense_hr_analysis(cardiac: pd.DataFrame, figure_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    participant = (cardiac[cardiac.stratum.isin(FUNCTIONAL) & cardiac.semiology.notna()]
                   .groupby(["participant", "semiology", "study_cohort", "minutes_to_ictus"], observed=True)
                   .delta_hr_bpm.mean().dropna().reset_index())
    participant["is_hypo"] = participant.semiology.eq("hypo").astype(int)
    participant["is_dd"] = participant.study_cohort.eq("DD").astype(int)
    formula = ("delta_hr_bpm ~ bs(minutes_to_ictus, df=5, degree=3, include_intercept=False) * is_hypo + is_dd")
    fit = ace.smf.gee(formula, "participant", participant, family=ace.sm.families.Gaussian(),
                      cov_struct=ace.sm.cov_struct.Independence()).fit(maxiter=200)
    interaction = [i for i, name in enumerate(fit.params.index) if ":is_hypo" in name]
    restriction = np.zeros((len(interaction), len(fit.params)))
    for row, column in enumerate(interaction): restriction[row, column] = 1
    omnibus = fit.wald_test(restriction, scalar=True)
    rows = [{"contrast": "nonlinear trajectory by semiology", "estimate": np.nan, "se": np.nan,
             "z": float(omnibus.statistic), "p": float(omnibus.pvalue), "ci95_low": np.nan, "ci95_high": np.nan}]
    design_info = fit.model.data.design_info
    def design(time, hypo):
        new = pd.DataFrame({"minutes_to_ictus": [time], "is_hypo": [hypo], "is_dd": [0]})
        return np.asarray(build_design_matrices([design_info], new)[0])[0]
    hyper_change = design(0, 0) - design(-5, 0)
    hypo_change = design(0, 1) - design(-5, 1)
    rows.extend((vector_contrast(fit, hyper_change, "hypermotor change −5 min to onset"),
                 vector_contrast(fit, hypo_change, "hypomotor change −5 min to onset"),
                 vector_contrast(fit, hypo_change - hyper_change, "hypomotor minus hypermotor five-minute change")))
    stats = pd.DataFrame(rows); stats["p_holm"] = ace.holm(stats.p)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for semio in ("hyper", "hypo"):
        part = participant[participant.semiology.eq(semio)]
        summary = []
        for minute, values in part.groupby("minutes_to_ictus").delta_hr_bpm:
            n, mean, low, high = mean_ci(values); summary.append((minute, mean, low, high))
        summary = pd.DataFrame(summary, columns=["time", "mean", "low", "high"]).sort_values("time")
        ax.plot(summary.time, summary["mean"], color=COLORS[semio], label=semio.capitalize(), lw=1.8)
        ax.fill_between(summary.time, summary.low, summary.high, color=COLORS[semio], alpha=.14)
    ax.axhline(0, color=".5", lw=1); ax.axvline(-5, color=".6", ls="--", lw=1)
    ax.set(xlabel="Minutes to seizure onset", ylabel="HR change from −20 to −5 min baseline (bpm)",
           title=f"Dense RR-derived HR trajectory (60-s windows, 10-s steps)\nSpline × semiology Holm p={stats.iloc[0].p_holm:.3g}")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(figure_dir / "dense_hr_nonlinear_trajectory.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    return participant, stats


def allstate_comparisons(all_windows: pd.DataFrame, all_phases: pd.DataFrame,
                         hr_final5: pd.DataFrame, data_dir: Path, figure_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    windows = add_design(all_windows[all_windows.semiology.notna()].copy())
    last30 = (windows[windows.window_end_to_ictus_s.between(0, 30, inclusive="left")]
              .groupby(["participant", "stratum", "semiology", "event_type", "study_cohort"], observed=True)
              [["log_kappa_all", "js_all_from_II"]].mean().reset_index())
    last30 = add_design(last30)
    rows = []
    for outcome in ("log_kappa_all", "js_all_from_II"):
        for result in model_contrasts(last30, outcome, f"all-state final 30 seconds {outcome}"):
            rows.append({"outcome": outcome, **result})
    last_stats = pd.DataFrame(rows); last_stats["p_bh"] = ace.bh(last_stats.p)
    last30.to_csv(data_dir / "allstate_last30s_participant.csv", index=False)
    last_stats.to_csv(data_dir / "allstate_last30s_contrasts.csv", index=False)

    phase = all_phases[all_phases.stratum.isin(FUNCTIONAL) & all_phases.semiology.notna()].copy()
    prepost_rows = []
    for semio, frame in phase.groupby("semiology", observed=True):
        for outcome in ("log_kappa_all", "js_all_from_II"):
            result, _ = paired_test(frame, ["participant"], "phase", outcome,
                                    "late_preictal_5min", "postictal_full_record", f"all-state {semio} pre/post {outcome}")
            prepost_rows.append({"semiology": semio, "outcome": outcome,
                                 "estimate": result.pop("estimate_right_minus_left"), **result})
    prepost = pd.DataFrame(prepost_rows); prepost["p_bh"] = ace.bh(prepost.p)
    prepost.to_csv(data_dir / "allstate_functional_prepost_stats.csv", index=False)

    dd_rows = []
    dd = last30[last30.stratum.isin(("DD functional", "DD epileptic"))]
    for outcome in ("log_kappa_all", "js_all_from_II"):
        result, _ = paired_test(dd, ["participant"], "event_type", outcome, "epileptic", "functional",
                                f"DD all-state FS minus ES {outcome}")
        dd_rows.append({"outcome": outcome, "estimate": result.pop("estimate_right_minus_left"), **result})
    dd_stats = pd.DataFrame(dd_rows); dd_stats["p_bh"] = ace.bh(dd_stats.p)
    dd_stats.to_csv(data_dir / "allstate_dd_paired_stats.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    group = last30.assign(group=last30.event_type.str.title() + " " + last30.semiology.str.title())
    for ax, outcome, ylabel in zip(axes[0], ("log_kappa_all", "js_all_from_II"), ("All-state log κ", "All-state JS from II")):
        names = list(group.group.unique())
        for x, name in enumerate(names):
            n, mean, low, high = mean_ci(group.loc[group.group.eq(name), outcome])
            ax.errorbar(x, mean, yerr=[[mean-low], [high-mean]], fmt="o", capsize=4)
        ax.set(xticks=np.arange(len(names)), xticklabels=names, ylabel=ylabel); ax.tick_params(axis="x", rotation=30)
    for ax, outcome, xlabel in zip(axes[1], ("log_kappa_all", "js_all_from_II"),
                                   ("Postictal − late-preictal log κ", "Postictal − late-preictal JS from II")):
        for x, semio in enumerate(("hyper", "hypo")):
            row = prepost[(prepost.semiology.eq(semio)) & prepost.outcome.eq(outcome)].iloc[0]
            ax.errorbar(row.estimate, x, xerr=[[row.estimate-row.ci95_low], [row.ci95_high-row.estimate]],
                        fmt="o", capsize=3, color=COLORS[semio])
            ax.annotate(f"BH p={row.p_bh:.3g}", (row.estimate, x), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8)
        ax.axvline(0, color=".5", lw=1); ax.set(yticks=[0, 1], yticklabels=["Hyper", "Hypo"],
                                                 xlabel=xlabel, title="All-state recovery")
    fig.tight_layout(); fig.savefig(figure_dir / "allstate_kappa_js_comparisons.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    return last_stats, prepost, dd_stats


def cardioneural_allstate(dense_hr: pd.DataFrame, all_windows: pd.DataFrame,
                          data_dir: Path, figure_dir: Path) -> pd.DataFrame:
    neural = all_windows[["participant", "file_path", "stratum", "semiology", "window_end_to_ictus_s",
                          "minutes_to_ictus", "log_kappa_all", "js_all_from_II"]]
    hr = dense_hr[dense_hr.hr_window_valid][["participant", "file_path", "window_end_to_ictus_s", "delta_hr_bpm"]]
    merged = neural.merge(hr, on=["participant", "file_path", "window_end_to_ictus_s"], how="inner", validate="one_to_one")
    merged = add_design(merged[merged.stratum.isin(FUNCTIONAL) & merged.semiology.notna()].copy())
    merged["hr_within_event"] = merged.delta_hr_bpm - merged.groupby(["participant", "file_path"]).delta_hr_bpm.transform("mean")
    merged.to_csv(data_dir / "allstate_exact_cardioneural_windows.csv", index=False)
    group = ["participant", "stratum", "semiology", "study_cohort", "window_end_to_ictus_s", "minutes_to_ictus"]
    participant = merged.groupby(group, observed=True)[["log_kappa_all", "js_all_from_II", "hr_within_event"]].mean().reset_index()
    participant = piecewise_columns(add_design(participant))
    rows = []
    for outcome in ("log_kappa_all", "js_all_from_II"):
        fit, _ = ace.fit_gee(participant, f"{outcome} ~ early_elapsed_min + late_elapsed_min + hr_within_event + "
                            "hr_within_event:is_hypo + is_hypo + is_dd", f"all-state cardioneural {outcome}")
        for name, weights in (("hypermotor HR association", {"hr_within_event": 1}),
                              ("hypomotor HR association", {"hr_within_event": 1, "hr_within_event:is_hypo": 1}),
                              ("hypomotor minus hypermotor HR association", {"hr_within_event:is_hypo": 1})):
            rows.append({"outcome": outcome, **ace.linear_contrast(fit, weights, name)})
    stats = pd.DataFrame(rows); stats["p_bh"] = ace.bh(stats.p)
    stats.to_csv(data_dir / "allstate_cardioneural_gee_contrasts.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, outcome, ylabel in zip(axes, ("log_kappa_all", "js_all_from_II"), ("All-state log κ", "All-state JS from II")):
        for semio in ("hyper", "hypo"):
            part = participant[participant.semiology.eq(semio)]
            ax.scatter(part.hr_within_event, part[outcome], s=7, alpha=.05, color=COLORS[semio], label=semio.capitalize())
            bins = pd.qcut(part.hr_within_event, 12, duplicates="drop")
            means = part.groupby(bins, observed=True)[["hr_within_event", outcome]].mean()
            ax.plot(means.hr_within_event, means[outcome], color=COLORS[semio], lw=2)
        ax.set(xlabel="Within-event HR deviation (bpm)", ylabel=ylabel); ax.legend(frameon=False)
    fig.suptitle("Concurrent HR and all-state transition dynamics")
    fig.tight_layout(); fig.savefig(figure_dir / "allstate_cardioneural_associations.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    return stats


def c_systole_diastole(metrics_path: Path, manifest: pd.DataFrame, data_dir: Path, figure_dir: Path) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    metrics = metrics[(metrics.template_label.eq("C")) & metrics.stratum.isin(FUNCTIONAL)].copy()
    meta = manifest[["file_key", "hypo_hyper"]].copy()
    metrics["file_key"] = metrics.file_path.astype(str).str.replace("/", "\\", regex=False).str.lower()
    metrics = metrics.merge(meta, on="file_key", how="left", validate="many_to_one")
    metrics["semiology"] = ace.normalise_semiology(metrics.hypo_hyper)
    metrics = metrics[metrics.semiology.notna()]
    rows = []
    for (start, end), frame in metrics.groupby(["systolic_window_start_ms", "systolic_window_end_ms"], observed=True):
        use = (frame[frame.minutes_to_ictus.between(-5, 0)]
               .groupby(["participant", "phase", "semiology", "stratum", "window_end_to_ictus_s"], observed=True)
               [["count", "exposure_s"]].mean().reset_index())
        use = use[use.exposure_s.gt(0)].copy(); use["is_systole"] = use.phase.eq("systole").astype(int)
        use["is_hypo"] = use.semiology.eq("hypo").astype(int); use["is_dd"] = use.stratum.eq("DD functional").astype(int)
        fit = ace.smf.gee("count ~ is_systole * is_hypo + is_dd", "participant", use,
                          family=ace.sm.families.Poisson(), cov_struct=ace.sm.cov_struct.Independence(),
                          offset=np.log(use.exposure_s)).fit(maxiter=200)
        for name, weights in (("hypermotor C systole/diastole", {"is_systole": 1}),
                              ("hypomotor C systole/diastole", {"is_systole": 1, "is_systole:is_hypo": 1}),
                              ("hypomotor minus hypermotor log ratio", {"is_systole:is_hypo": 1})):
            result = ace.linear_contrast(fit, weights, name)
            rows.append({"systolic_window_start_ms": start, "systolic_window_end_ms": end, **result,
                         "rate_ratio": np.exp(result["estimate"]), "ratio_ci95_low": np.exp(result["ci95_low"]),
                         "ratio_ci95_high": np.exp(result["ci95_high"])})
    stats = pd.DataFrame(rows); stats["p_bh"] = ace.bh(stats.p)
    stats.to_csv(data_dir / "c_systole_diastole_rate_ratio_stats.csv", index=False)
    primary = metrics[(metrics.systolic_window_start_ms.eq(180)) & metrics.systolic_window_end_ms.eq(500)]
    wide = primary.pivot_table(index=["participant", "file_path", "semiology", "window_end_to_ictus_s", "minutes_to_ictus"],
                               columns="phase", values=["count", "exposure_s"], aggfunc="sum").dropna()
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]; wide = wide.reset_index()
    wide["log_c_sd_ratio"] = np.log((wide.count_systole + .5) / wide.exposure_s_systole) - np.log((wide.count_diastole + .5) / wide.exposure_s_diastole)
    participant = wide.groupby(["participant", "semiology", "minutes_to_ictus"], observed=True).log_c_sd_ratio.mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.7))
    for semio in ("hyper", "hypo"):
        summary = []
        for minute, values in participant[participant.semiology.eq(semio)].groupby("minutes_to_ictus").log_c_sd_ratio:
            n, mean, low, high = mean_ci(values); summary.append((minute, np.exp(mean), np.exp(low), np.exp(high)))
        summary = pd.DataFrame(summary, columns=["time", "ratio", "low", "high"]).sort_values("time")
        ax.plot(summary.time, summary.ratio, color=COLORS[semio], label=semio.capitalize())
        ax.fill_between(summary.time, summary.low, summary.high, color=COLORS[semio], alpha=.15)
    ax.axhline(1, color=".5", lw=1); ax.set(xlabel="Minutes to onset", ylabel="C occurrence-rate ratio: systole / diastole",
                                             title="Microstate C cardiac-phase expression (180–500 ms systole)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(figure_dir / "c_systole_diastole_ratio.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    return stats


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    use = frame[columns].copy()
    for column in use.select_dtypes(include="number"):
        use[column] = use[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    return "| " + " | ".join(columns) + " |\n|" + "|".join(["---"] * len(columns)) + "|\n" + "\n".join(
        "| " + " | ".join(map(str, row)) + " |" for row in use.itertuples(index=False, name=None))


def run(args) -> dict:
    data_dir, figure_dir = args.out_dir / "data", args.out_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True); figure_dir.mkdir(parents=True, exist_ok=True)
    manifest = attach_strata(pd.read_csv(args.manifest))
    manifest["semiology"] = ace.normalise_semiology(manifest["hypo_hyper"])
    cardiac = ace.cardiac_windows(manifest, args.ecg_qc_dir, args.original_ecg_qc_dir, ALL_STRATA)
    cardiac = add_design(cardiac)
    cardiac.to_csv(data_dir / "cardiac_all_strata_windows.csv", index=False)

    functional_hr = cardiac[cardiac.stratum.isin(FUNCTIONAL) & cardiac.semiology.notna()]
    event5 = (functional_hr[functional_hr.minutes_to_ictus.between(-5, 0) & functional_hr.delta_hr_bpm.notna()]
              .groupby(["participant", "file_path", "stratum", "semiology", "study_cohort"], observed=True)
              .delta_hr_bpm.mean().rename("mean_delta_hr_last5").reset_index())
    participant5 = event5.groupby(["participant", "semiology", "study_cohort"], observed=True).mean_delta_hr_last5.mean().reset_index()
    participant5["is_hypo"] = participant5.semiology.eq("hypo").astype(int); participant5["is_dd"] = participant5.study_cohort.eq("DD").astype(int)
    fit, _ = ace.fit_gee(participant5, "mean_delta_hr_last5 ~ is_hypo + is_dd", "final-five-minute HR by semiology")
    hr_mean = ace.linear_contrast(fit, {"is_hypo": 1}, "Functional HR: hypomotor minus hypermotor")
    hr_mean.update({"analysis": "preictal_hr_semiology", "outcome": "mean_delta_hr_last5"})
    trajectory = (functional_hr.groupby(["participant", "semiology", "minutes_to_ictus"], observed=True)
                  .delta_hr_bpm.mean().reset_index())

    gradients = final_minute_slopes(manifest, cardiac, args.ecg_qc_dir)
    gradient_participant = (gradients[gradients.stratum.isin(FUNCTIONAL) & gradients.semiology.notna()]
                            .groupby(["participant", "semiology", "study_cohort"], observed=True)
                            .hr_gradient_bpm_per_min.mean().reset_index())
    gradient_participant["is_hypo"] = gradient_participant.semiology.eq("hypo").astype(int)
    gradient_participant["is_dd"] = gradient_participant.study_cohort.eq("DD").astype(int)
    fit, _ = ace.fit_gee(gradient_participant, "hr_gradient_bpm_per_min ~ is_hypo + is_dd", "last-minute HR gradient")
    hr_gradient = ace.linear_contrast(fit, {"is_hypo": 1}, "HR gradient: hypomotor minus hypermotor")
    hr_gradient.update({"analysis": "hr_gradient_semiology", "outcome": "hr_gradient_bpm_per_min"})
    hr_stats = pd.DataFrame([hr_mean, hr_gradient]); hr_stats["p_holm"] = ace.holm(hr_stats.p)
    hr_stats.to_csv(data_dir / "hr_semiology_stats.csv", index=False)
    participant5.to_csv(data_dir / "hr_final5_participant.csv", index=False)
    gradients.to_csv(data_dir / "hr_final_minute_event_gradients.csv", index=False)
    gradient_participant.to_csv(data_dir / "hr_final_minute_participant_gradients.csv", index=False)
    plot_hr(trajectory, participant5, hr_stats.iloc[[0]], figure_dir / "hr_preictal_semiology.png")
    plot_gradient(gradient_participant, hr_stats.iloc[1], figure_dir / "hr_final_minute_gradient.png")

    pre = pd.read_csv(args.statewise_dir / "statewise_preictal_transition_participant.csv")
    last30 = pre[pre.window_end_to_ictus_s.between(0, 30, inclusive="left") & pre.semiology.notna()].copy()
    last30 = (last30.groupby(["participant", "stratum", "semiology", "template_label"], observed=True)
              [["log_kappa", "js_from_II"]].mean().reset_index())
    last30 = add_design(last30); last30.to_csv(data_dir / "statewise_last30s_participant.csv", index=False)
    contrast_rows = []
    for state in sorted(last30.template_label.unique()):
        frame = last30[last30.template_label.eq(state)]
        for outcome in ("log_kappa", "js_from_II"):
            for result in model_contrasts(frame, outcome, f"state {state} final 30 seconds {outcome}"):
                contrast_rows.append({"template_label": state, "outcome": outcome, **result})
    last30_stats = pd.DataFrame(contrast_rows); last30_stats["p_bh"] = ace.bh(last30_stats.p)
    last30_stats.to_csv(data_dir / "statewise_last30s_contrasts.csv", index=False)
    descriptive_rows = []
    last30["group"] = last30["event_type"].str.title() + " " + last30["semiology"].str.title()
    for key, frame in last30.groupby(["template_label", "group"], observed=True):
        for outcome in ("log_kappa", "js_from_II"):
            n, mean, low, high = mean_ci(frame[outcome]); descriptive_rows.append(
                {"template_label": key[0], "group": key[1], "outcome": outcome, "n": n, "mean": mean,
                 "ci95_low": low, "ci95_high": high})
    last30_desc = pd.DataFrame(descriptive_rows); last30_desc.to_csv(data_dir / "statewise_last30s_descriptives.csv", index=False)
    plot_last30_means(last30_desc, figure_dir / "statewise_last30s_group_means.png")
    plot_last30_contrasts(last30_stats, figure_dir / "statewise_last30s_contrasts.png")

    recovery = pd.read_csv(args.statewise_dir / "statewise_recovery_participant.csv")
    recovery = recovery[recovery.stratum.isin(FUNCTIONAL) & recovery.semiology.notna()]
    prepost_rows, prepost_pairs = [], []
    for (state, semio), frame in recovery.groupby(["template_label", "semiology"], observed=True):
        for outcome in ("log_kappa", "js_from_II"):
            row, pairs = paired_test(frame, ["participant"], "phase", outcome, "late_preictal_5min", "postictal_full_record",
                                     f"{state} {semio} postictal minus preictal {outcome}")
            prepost_rows.append({"template_label": state, "semiology": semio, "outcome": outcome,
                                 "estimate": row.pop("estimate_right_minus_left"), **row})
            pairs["template_label"], pairs["semiology"], pairs["outcome"] = state, semio, outcome
            prepost_pairs.append(pairs)
    prepost_stats = pd.DataFrame(prepost_rows); prepost_stats["p_bh"] = ace.bh(prepost_stats.p)
    prepost_stats.to_csv(data_dir / "functional_prepost_statewise_stats.csv", index=False)
    pd.concat(prepost_pairs, ignore_index=True).to_csv(data_dir / "functional_prepost_statewise_pairs.csv", index=False)
    plot_prepost(prepost_stats, figure_dir / "functional_prepost_statewise.png")

    dd_hr_event = (cardiac[cardiac.stratum.isin(("DD functional", "DD epileptic")) & cardiac.minutes_to_ictus.between(-5, 0) & cardiac.delta_hr_bpm.notna()]
                   .groupby(["participant", "file_path", "event_type"], observed=True).delta_hr_bpm.mean().reset_index())
    dd_hr = dd_hr_event.groupby(["participant", "event_type"], observed=True).delta_hr_bpm.mean().rename("mean_delta_hr_last5").reset_index()
    hr_pair_row, hr_pairs = paired_test(dd_hr, ["participant"], "event_type", "mean_delta_hr_last5", "epileptic", "functional", "DD paired HR FS minus ES")
    dd_rows = [{"outcome": "mean_delta_hr_last5", "template_label": "HR", "estimate": hr_pair_row.pop("estimate_right_minus_left"), **hr_pair_row}]
    for state, frame in last30[last30.stratum.isin(("DD functional", "DD epileptic"))].groupby("template_label", observed=True):
        for outcome in ("log_kappa", "js_from_II"):
            row, _ = paired_test(frame, ["participant"], "event_type", outcome, "epileptic", "functional",
                                 f"DD paired {state} FS minus ES {outcome}")
            dd_rows.append({"outcome": outcome, "template_label": state, "estimate": row.pop("estimate_right_minus_left"), **row})
    dd_stats = pd.DataFrame(dd_rows); dd_stats["p_bh"] = ace.bh(dd_stats.p)
    dd_stats.to_csv(data_dir / "dd_paired_fs_es_stats.csv", index=False)
    hr_pairs.to_csv(data_dir / "dd_paired_hr_values.csv", index=False)
    plot_dd(dd_stats, hr_pairs, figure_dir / "dd_paired_fs_es.png")

    dense_cardiac = ace.cardiac_windows(manifest, args.ecg_qc_dir, args.original_ecg_qc_dir,
                                        ALL_STRATA, window_length_s=60, step_s=10)
    dense_cardiac = add_design(dense_cardiac)
    dense_cardiac.to_csv(data_dir / "cardiac_dense_60s_10s_windows.csv", index=False)
    dense_hr_participant, dense_hr_stats = dense_hr_analysis(dense_cardiac, figure_dir)
    dense_hr_participant.to_csv(data_dir / "cardiac_dense_participant.csv", index=False)
    dense_hr_stats.to_csv(data_dir / "cardiac_dense_spline_stats.csv", index=False)

    all_windows = pd.read_csv(args.statewise_dir / "allstate_preictal_transition_windows.csv")
    all_phases = pd.read_csv(args.statewise_dir / "allstate_recovery_event_phases.csv")
    all_last, all_prepost, all_dd = allstate_comparisons(
        all_windows, all_phases, participant5, data_dir, figure_dir)
    all_cardioneural = cardioneural_allstate(dense_cardiac, all_windows, data_dir, figure_dir)
    c_ratio = c_systole_diastole(args.cardioneural_data / "ace_heartbeat_phase_rolling_metrics.csv",
                                 manifest, data_dir, figure_dir)

    summary = {"labels": sorted(last30.template_label.unique()), "hr_participants": int(participant5.participant.nunique()),
               "hr_gradient_participants": int(gradient_participant.participant.nunique()),
               "last30_participants": int(last30.participant.nunique()), "prepost_participants": int(recovery.participant.nunique()),
               "dd_hr_pairs": int(hr_pair_row["n_pairs"]), "dd_state_pairs_min": int(dd_stats.loc[dd_stats.outcome.ne("mean_delta_hr_last5"), "n_pairs"].min()),
               "allstate_windows": int(len(all_windows)), "dense_hr_windows": int(len(dense_cardiac)),
               "allstate_cardioneural_participants": int(pd.read_csv(data_dir / "allstate_exact_cardioneural_windows.csv").participant.nunique())}
    (data_dir / "requested_comparisons_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = ["# Requested cardiac and transition comparisons", "",
              "All analyses use the confirmed manual state labels. HR is corrected to each event's mean from −20 to −5 minutes. "
              "Statewise κ is reported on the log scale; JS is divergence from the participant's interictal transition operator.", "",
              "## HR by functional-seizure semiology", "",
              markdown_table(hr_stats, ["analysis", "estimate", "ci95_low", "ci95_high", "p", "p_holm"]), "",
              "![Preictal HR](figures/hr_preictal_semiology.png)", "",
              "![Last-minute HR gradient](figures/hr_final_minute_gradient.png)", "",
              "## Final 30 seconds: functional versus epileptic seizures matched by semiology", "",
              markdown_table(last30_stats, ["template_label", "outcome", "contrast", "estimate", "ci95_low", "ci95_high", "p_bh"]), "",
              "![Final-30-second means](figures/statewise_last30s_group_means.png)", "",
              "![Final-30-second contrasts](figures/statewise_last30s_contrasts.png)", "",
              "## Functional seizures: late preictal versus postictal", "",
              "The preictal phase is the final five minutes; the postictal phase is the available whole postictal record, so this is a phase comparison rather than a duration-matched contrast.", "",
              markdown_table(prepost_stats, ["template_label", "semiology", "outcome", "estimate", "ci95_low", "ci95_high", "n_pairs", "p_bh"]), "",
              "![Pre/post statewise changes](figures/functional_prepost_statewise.png)", "",
              "## Within-DD paired FS versus ES", "",
              markdown_table(dd_stats, ["template_label", "outcome", "estimate", "ci95_low", "ci95_high", "n_pairs", "p_bh"]), "",
              "![DD paired comparisons](figures/dd_paired_fs_es.png)", "",
              "## Primary pooled six-state κ and JS", "",
              "These estimates use a single κ fitted jointly to every off-diagonal transition from all six origin states. JS is transition-count weighted across all six origin distributions.", "",
              markdown_table(all_last, ["outcome", "contrast", "estimate", "ci95_low", "ci95_high", "p_bh"]), "",
              markdown_table(all_prepost, ["semiology", "outcome", "estimate", "ci95_low", "ci95_high", "n_pairs", "p_bh"]), "",
              markdown_table(all_dd, ["outcome", "estimate", "ci95_low", "ci95_high", "n_pairs", "p_bh"]), "",
              "![All-state comparisons](figures/allstate_kappa_js_comparisons.png)", "",
              "## Sensitive cardiac and cardioneural analyses", "",
              markdown_table(dense_hr_stats, ["contrast", "estimate", "ci95_low", "ci95_high", "p_holm"]), "",
              markdown_table(all_cardioneural, ["outcome", "contrast", "estimate", "ci95_low", "ci95_high", "p_bh"]), "",
              "![Dense cardiac trajectory](figures/dense_hr_nonlinear_trajectory.png)", "",
              "![All-state cardioneural associations](figures/allstate_cardioneural_associations.png)", "",
              "## Microstate C systole/diastole ratio", "",
              markdown_table(c_ratio, ["systolic_window_start_ms", "systolic_window_end_ms", "contrast", "rate_ratio", "ratio_ci95_low", "ratio_ci95_high", "p_bh"]), "",
              "![C systole/diastole ratio](figures/c_systole_diastole_ratio.png)", "",
              "## Statistical notes", "",
              "- Participant is the inferential unit. Repeated semiology/event-type rows use participant-clustered Gaussian GEE with DD-cohort adjustment.",
              "- Pre/post and within-DD comparisons are paired participant differences with t-based 95% confidence intervals.",
              "- HR hypotheses are Holm-adjusted together. Statewise contrast families are BH-adjusted.",
              "- Last-minute HR gradient is an OLS slope across valid RR-derived instantaneous HR observations within the final 60 seconds."]
    (args.out_dir / "requested_comparisons_findings.md").write_text("\n".join(report), encoding="utf-8")
    return summary


def self_check() -> None:
    frame = pd.DataFrame({"id": [1, 1, 2, 2], "phase": ["a", "b", "a", "b"], "value": [1, 3, 2, 5]})
    result, pairs = paired_test(frame, ["id"], "phase", "value", "a", "b", "check")
    assert result["n_pairs"] == 2 and np.isclose(result["estimate_right_minus_left"], 2.5) and len(pairs) == 2
    assert mean_ci([1, 2, 3])[0] == 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--statewise-dir", type=Path)
    parser.add_argument("--cardioneural-data", type=Path)
    parser.add_argument("--ecg-qc-dir", type=Path)
    parser.add_argument("--original-ecg-qc-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(); print("self-check passed"); return
    if not all((args.manifest, args.statewise_dir, args.cardioneural_data, args.ecg_qc_dir, args.original_ecg_qc_dir, args.out_dir)):
        parser.error("all path arguments are required")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
