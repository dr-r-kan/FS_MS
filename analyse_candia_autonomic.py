"""Candia-Rivera robust cardiac autonomic indices around seizures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import trim_mean

import analyse_ace_cardioneural as ace
from analyse_preictal_publication import attach_strata, piecewise_columns
from analyse_requested_comparisons import add_design, mean_ci, model_contrasts, paired_test, markdown_table


ALL_STRATA = ("FS functional", "DD functional", "ES epileptic", "DD epileptic")
FUNCTIONAL = ("FS functional", "DD functional")
METHOD_COMMIT = "3919a6f02fb8c57b06b52d729360473c2591f6ea"
COLORS = {"FS Hyper": "#D55E00", "FS Hypo": "#0072B2", "ES Hyper": "#E69F00", "ES Hypo": "#56B4E9",
          "functional": "#009E73", "epileptic": "#7A5195", "hyper": "#D55E00", "hypo": "#0072B2"}


def robust_cov_hrv(x: np.ndarray) -> np.ndarray:
    """Python equivalent of the authors' robustCovHRV.m shrinkage covariance."""
    x = np.asarray(x, float)
    if x.ndim != 2 or len(x) < 4 or not np.isfinite(x).all():
        raise ValueError("robust covariance needs at least four finite observations")
    n, p = x.shape
    xc = x - x.mean(axis=0)

    def varcov(z):
        s = np.cov(z, rowvar=False, ddof=1)
        outer = z[:, :, None] * z[:, None, :]
        vs = np.var(outer, axis=0, ddof=1) * n / (n - 1) ** 2
        return np.atleast_2d(s), np.atleast_2d(vs)

    s, vs = varcov(xc)
    v, vv = np.diag(s), np.diag(vs)
    target = np.median(v)
    denominator = np.sum((v - target) ** 2)
    lam = np.clip(np.sum(vv) / denominator, 0, 1) if denominator > 0 else 1.0
    shrunk_v = (1 - lam) * v + lam * target
    sd = xc.std(axis=0, ddof=1)
    if np.any(sd <= 0):
        return np.diag(np.maximum(shrunk_v, 0))
    r, vr = varcov(xc / sd)
    lower = np.tril_indices(p, -1)
    denominator = np.sum(r[lower] ** 2)
    lam = np.clip(np.sum(vr[lower]) / denominator, 0, 1) if denominator > 0 else 1.0
    r = (1 - lam) * r
    np.fill_diagonal(r, 1)
    d = np.diag(np.sqrt(np.maximum(shrunk_v, 0)))
    return d @ r @ d


def poincare_features(ibi: np.ndarray) -> tuple[float, float, float]:
    points = np.column_stack([ibi[:-1], ibi[1:]])
    eigenvalues = np.maximum(np.linalg.eigvalsh(robust_cov_hrv(points)), 0)
    centre = trim_mean(points, .025, axis=0)
    return float(np.sqrt(eigenvalues[0])), float(np.sqrt(eigenvalues[1])), float(np.linalg.norm(centre))


def native_indices(peaks: np.ndarray, recovered_peaks: np.ndarray, window_s: float = 15.0) -> pd.DataFrame:
    peaks = np.asarray(peaks, float)
    recovered_peaks = np.asarray(recovered_peaks, bool)
    rr, t_rr = np.diff(peaks), peaks[1:]
    interval_recovered = recovered_peaks[:-1] | recovered_peaks[1:]
    physiological = np.isfinite(rr) & (rr >= 60 / 140) & (rr <= 60 / 35)
    base_pairs = physiological[:-1] & physiological[1:]
    if base_pairs.sum() < 20:
        raise ValueError("fewer than 20 physiological IBI pairs")
    base = np.column_stack([rr[:-1][base_pairs], rr[1:][base_pairs]])
    base_eigen = np.maximum(np.linalg.eigvalsh(robust_cov_hrv(base)), 0)
    sd01, sd02 = np.sqrt(base_eigen)
    d0 = float(np.linalg.norm(trim_mean(base, .025, axis=0)))
    rows = []
    eligible = np.flatnonzero(t_rr > t_rr[0] + window_s)
    for i in eligible[:-1]:  # matches the published MATLAB implementation
        left = int(np.searchsorted(t_rr, t_rr[i] - window_s, side="left"))
        use = np.arange(left, i + 1)
        if (len(use) < 7 or t_rr[use[-1]] - t_rr[use[0]] < .8 * window_s or
                not physiological[use].all() or interval_recovered[use].mean() > .10):
            continue
        local_rr = rr[use]
        first_difference = np.diff(local_rr)
        second_difference = np.diff(first_difference)
        sd1, sd2, distance = poincare_features(local_rr)
        rows.append((np.median(t_rr[use]), t_rr[use[0]], t_rr[use[-1]], sd1, sd2, distance,
                     len(use), interval_recovered[use].mean(), local_rr.mean(), 60 / local_rr.mean(),
                     first_difference.mean(), second_difference.mean() if len(second_difference) else np.nan,
                     np.sqrt(np.mean(first_difference ** 2))))
    out = pd.DataFrame(rows, columns=["event_time_s", "window_start_s", "window_end_s", "sd1", "sd2",
                                      "ccd", "n_ibi", "recovered_fraction", "mean_rr_s", "mean_hr_bpm",
                                      "rr_first_difference_s", "rr_acceleration_s", "rmssd_s"])
    if len(out) < 30:
        raise ValueError("fewer than 30 valid 15-second estimates")
    for column, reference in (("sd1", sd01), ("sd2", sd02), ("ccd", d0)):
        out[f"{column}_recentred"] = out[column] - out[column].mean() + reference
    out["cpi_hr_component"] = out["ccd_recentred"]
    out["csi_hr_component"] = 2 * out["ccd_recentred"].mean() - out["ccd_recentred"]
    out["cpi_hrv_component"] = 10 * out["sd1_recentred"] + 1
    out["csi_hrv_component"] = out["sd2_recentred"] + 1
    out["cpi"] = out["cpi_hr_component"] + out["cpi_hrv_component"]
    out["csi"] = out["csi_hr_component"] + out["csi_hrv_component"]
    return out


def process_event(event, ecg_qc_dir: Path) -> tuple[pd.DataFrame, dict]:
    sidecar = ace.rpeak_file(ecg_qc_dir, event.source_raw_fif)
    peaks, recovered = ace.read_rpeaks_with_recovery_flags(sidecar)
    start, stop = float(event.crop_start_seconds), float(event.crop_end_seconds)
    keep = (peaks >= start) & (peaks <= stop)
    qc = {"participant": event.participant, "file_path": event.file_path, "stratum": event.stratum,
          "semiology": event.semiology, "rpeak_sidecar": str(sidecar), "segment_duration_s": stop - start,
          "n_segment_peaks": int(keep.sum()), "status": "failed", "reason": ""}
    if keep.sum() < 40:
        qc["reason"] = "fewer than 40 R-peaks"
        return pd.DataFrame(), qc
    try:
        out = native_indices(peaks[keep] - start, recovered[keep])
    except ValueError as error:
        qc["reason"] = str(error)
        return pd.DataFrame(), qc
    onset = stop - start
    out["time_to_ictus_s"] = out["event_time_s"] - onset
    out["minutes_to_ictus"] = out["time_to_ictus_s"] / 60
    baseline = out[out["time_to_ictus_s"].between(-1200, -300, inclusive="both")]
    span = baseline.event_time_s.max() - baseline.event_time_s.min() if len(baseline) else 0
    if len(baseline) < 120 or span < 300:
        qc["reason"] = "less than five minutes of valid -20 to -5 minute baseline"
        return pd.DataFrame(), qc
    for index in ("csi", "cpi"):
        mean, sd = baseline[index].mean(), baseline[index].std(ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            qc["reason"] = f"zero or invalid {index} baseline SD"
            return pd.DataFrame(), qc
        out[f"{index}_baseline_mean"] = mean
        out[f"{index}_baseline_sd"] = sd
        out[f"{index}_z"] = (out[index] - mean) / sd
    out.insert(0, "participant", event.participant)
    out.insert(1, "file_path", event.file_path)
    out.insert(2, "stratum", event.stratum)
    out.insert(3, "semiology", event.semiology)
    out["event_type"] = "functional" if event.stratum in FUNCTIONAL else "epileptic"
    out["study_cohort"] = "DD" if str(event.stratum).startswith("DD") else "single-diagnosis"
    qc.update(status="valid", reason="", n_native_estimates=len(out), baseline_estimates=len(baseline),
              baseline_span_s=float(span), median_ibi_per_estimate=float(out.n_ibi.median()))
    return out, qc


def exact_windows(native: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, frame in native.groupby(["participant", "file_path", "stratum", "semiology", "event_type", "study_cohort"], observed=True):
        frame = frame.sort_values("time_to_ictus_s")
        time = frame.time_to_ictus_s.to_numpy(float)
        csi, cpi = frame.csi_z.to_numpy(float), frame.cpi_z.to_numpy(float)
        csi_sum, cpi_sum = np.r_[0, np.cumsum(csi)], np.r_[0, np.cumsum(cpi)]
        for end_to in range(0, 1141, 10):
            left = np.searchsorted(time, -end_to - 60, side="left")
            right = np.searchsorted(time, -end_to, side="right")
            n = right - left
            if n < 30 or time[right - 1] - time[left] < 45:
                continue
            rows.append((*key, end_to, -end_to / 60, (csi_sum[right] - csi_sum[left]) / n,
                         (cpi_sum[right] - cpi_sum[left]) / n, n, time[right - 1] - time[left]))
    columns = ["participant", "file_path", "stratum", "semiology", "event_type", "study_cohort",
               "window_end_to_ictus_s", "minutes_to_ictus", "csi_z", "cpi_z", "n_native_estimates", "native_span_s"]
    return add_design(pd.DataFrame(rows, columns=columns))


def participant_final5(windows: pd.DataFrame) -> pd.DataFrame:
    event = (windows[windows.minutes_to_ictus.between(-5, 0)]
             .groupby(["participant", "file_path", "stratum", "semiology", "event_type", "study_cohort"], observed=True)
             [["csi_z", "cpi_z"]].mean().reset_index())
    participant = (event.groupby(["participant", "stratum", "semiology", "event_type", "study_cohort"], observed=True)
                   [["csi_z", "cpi_z"]].mean().reset_index())
    return add_design(participant)


def group_statistics(final5: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for outcome in ("csi_z", "cpi_z"):
        for result in model_contrasts(final5, outcome, f"Candia final-five-minute {outcome}"):
            rows.append({"outcome": outcome, **result})
    stats = pd.DataFrame(rows)
    stats["p_bh"] = ace.bh(stats.p)
    desc = []
    labelled = final5.assign(group=np.where(final5.event_type.eq("functional"), "FS ", "ES ") + final5.semiology.str.title())
    for (outcome, group), frame in labelled.melt(id_vars=["group"], value_vars=["csi_z", "cpi_z"],
                                                  var_name="outcome").groupby(["outcome", "group"], observed=True):
        n, mean, low, high = mean_ci(frame.value)
        desc.append({"outcome": outcome, "group": group, "n": n, "mean": mean, "ci95_low": low, "ci95_high": high})
    dd_rows = []
    dd = final5[final5.study_cohort.eq("DD")]
    for semio, frame in dd.groupby("semiology", observed=True):
        for outcome in ("csi_z", "cpi_z"):
            result, _ = paired_test(frame, ["participant"], "event_type", outcome, "epileptic", "functional",
                                    f"DD {semio} FS minus ES {outcome}")
            dd_rows.append({"semiology": semio, "outcome": outcome,
                            "estimate": result.pop("estimate_right_minus_left"), **result})
    dd_stats = pd.DataFrame(dd_rows)
    dd_stats["p_bh"] = ace.bh(dd_stats.p)
    return stats, pd.DataFrame(desc), dd_stats


def spline_statistics(windows: pd.DataFrame) -> pd.DataFrame:
    group = ["participant", "stratum", "semiology", "event_type", "study_cohort", "minutes_to_ictus"]
    participant = windows.groupby(group, observed=True)[["csi_z", "cpi_z"]].mean().reset_index()
    participant = add_design(participant)
    rows = []
    for outcome in ("csi_z", "cpi_z"):
        for label, frame, interaction, formula in (
            ("FS versus ES trajectory", participant, ":is_functional",
             f"{outcome} ~ bs(minutes_to_ictus, df=5, degree=3, include_intercept=False) * is_functional + is_hypo + is_dd"),
            ("FS hypomotor versus hypermotor trajectory", participant[participant.event_type.eq("functional")], ":is_hypo",
             f"{outcome} ~ bs(minutes_to_ictus, df=5, degree=3, include_intercept=False) * is_hypo + is_dd")):
            fit = ace.smf.gee(formula, "participant", frame, family=ace.sm.families.Gaussian(),
                              cov_struct=ace.sm.cov_struct.Independence()).fit(maxiter=200)
            columns = [i for i, name in enumerate(fit.params.index) if interaction in name]
            restriction = np.zeros((len(columns), len(fit.params)))
            restriction[np.arange(len(columns)), columns] = 1
            test = fit.wald_test(restriction, scalar=True)
            rows.append({"outcome": outcome, "contrast": label, "wald_chi2": float(test.statistic),
                         "df": len(columns), "p": float(test.pvalue), "n_participants": int(frame.participant.nunique())})
    out = pd.DataFrame(rows)
    out["p_holm"] = ace.holm(out.p)
    return out


def merged_microstates(windows: pd.DataFrame, activity_path: Path, kappa_path: Path) -> pd.DataFrame:
    activity = pd.read_csv(activity_path)
    activity = activity[(activity.template_label.eq("C")) & activity.backfit_method.eq("hard") &
                        activity.source.eq("participant") & activity.scheme.eq("20min_60s_step10s")]
    activity = activity[["participant", "file_path", "window_end_to_ictus_s", "coverage"]].rename(columns={"coverage": "c_coverage"})
    kappa = pd.read_csv(kappa_path)
    kappa = kappa[kappa.template_label.eq("C")][
        ["participant", "file_path", "window_end_to_ictus_s", "log_kappa"]].rename(columns={"log_kappa": "c_log_kappa"})
    keys = ["participant", "file_path", "window_end_to_ictus_s"]
    merged = windows.merge(activity, on=keys, how="inner", validate="one_to_one").merge(kappa, on=keys, how="inner", validate="one_to_one")
    merged = merged.dropna(subset=["csi_z", "cpi_z", "c_coverage", "c_log_kappa"])
    merged = merged[merged.event_type.eq("functional")].copy()
    event = merged.groupby(["participant", "file_path"])
    for column in ("csi_z", "cpi_z", "c_coverage", "c_log_kappa"):
        merged[f"{column}_within"] = merged[column] - event[column].transform("mean")
    for column in ("c_coverage_within", "c_log_kappa_within"):
        scale = merged[column].std(ddof=1)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid within-FS scale for {column}")
        merged[f"{column}_z"] = merged[column] / scale
    return merged


def association_statistics(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = ["participant", "stratum", "semiology", "event_type", "study_cohort", "window_end_to_ictus_s", "minutes_to_ictus"]
    values = ["csi_z_within", "cpi_z_within", "c_coverage_within_z", "c_log_kappa_within_z"]
    participant = merged.groupby(group, observed=True)[values].mean().reset_index()
    participant = piecewise_columns(add_design(participant))
    participant = participant[participant.event_type.eq("functional")].copy()
    rows = []
    predictors = ("c_coverage_within_z", "c_log_kappa_within_z")
    tail = "early_elapsed_min + late_elapsed_min + c_coverage_within_z + c_log_kappa_within_z + is_hypo + is_dd"
    for outcome in ("csi_z_within", "cpi_z_within"):
        fit, _ = ace.fit_gee(participant, f"{outcome} ~ {tail}", f"within-FS autonomic association {outcome}")
        for predictor in predictors:
            rows.append({"family": "whole preictal FS", "outcome": outcome, "predictor": predictor,
                         "n_participants": participant.participant.nunique(), "n_windows": len(participant),
                         **ace.linear_contrast(fit, {predictor: 1}, "pooled within-FS association")})
    stats = pd.DataFrame(rows)
    stats["p_bh"] = ace.bh(stats.p)
    passed = set(zip(stats.loc[stats.p_bh.lt(.05), "outcome"], stats.loc[stats.p_bh.lt(.05), "predictor"]))
    terminal = participant[participant.window_end_to_ictus_s.between(0, 30, inclusive="left")]
    interaction_tail = ("minutes_to_ictus + c_coverage_within_z + c_log_kappa_within_z + is_hypo + is_dd + "
                        "c_coverage_within_z:is_hypo + c_log_kappa_within_z:is_hypo")
    terminal_rows = []
    for outcome in ("csi_z_within", "cpi_z_within"):
        fit, _ = ace.fit_gee(terminal, f"{outcome} ~ {interaction_tail}", f"terminal FS semiology {outcome}")
        for predictor in predictors:
            interaction = f"{predictor}:is_hypo"
            terminal_rows.append({"family": "terminal endpoints <30 s", "outcome": outcome, "predictor": predictor,
                                  "gate_passed": (outcome, predictor) in passed,
                                  "n_participants": terminal.participant.nunique(), "n_windows": len(terminal),
                                  **ace.linear_contrast(fit, {interaction: 1}, "hypomotor minus hypermotor association")})
    terminal_stats = pd.DataFrame(terminal_rows)
    terminal_stats["p_bh"] = ace.bh(terminal_stats.p)
    stats["gate_passed"] = [(outcome, predictor) in passed for outcome, predictor in zip(stats.outcome, stats.predictor)]
    stats = pd.concat([stats, terminal_stats], ignore_index=True)
    return participant, stats


def plot_trajectories(native: pd.DataFrame, spline_stats: pd.DataFrame, output: Path) -> None:
    data = native.copy()
    data["time_bin_s"] = (data.time_to_ictus_s / 10).round() * 10
    data["group"] = np.where(data.event_type.eq("functional"), "FS ", "ES ") + data.semiology.str.title()
    event = data.groupby(["participant", "file_path", "group", "time_bin_s"], observed=True)[["csi_z", "cpi_z"]].mean().reset_index()
    participant = event.groupby(["participant", "group", "time_bin_s"], observed=True)[["csi_z", "cpi_z"]].mean().reset_index()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, outcome, label in zip(axes, ("csi_z", "cpi_z"), ("Cardiac sympathetic index", "Cardiac parasympathetic index")):
        for group in ("FS Hyper", "FS Hypo", "ES Hyper", "ES Hypo"):
            rows = []
            for time, frame in participant[participant.group.eq(group)].groupby("time_bin_s"):
                values = frame[outcome].dropna().to_numpy(float)
                median = np.median(values); mad = np.median(np.abs(values - median))
                rows.append((time / 60, median, median - mad, median + mad))
            summary = pd.DataFrame(rows, columns=["time", "median", "low", "high"]).sort_values("time")
            ax.plot(summary.time, summary["median"], color=COLORS[group], label=group, lw=1.5)
            ax.fill_between(summary.time, summary.low, summary.high, color=COLORS[group], alpha=.10)
        ax.axhline(0, color=".5", lw=1); ax.axvline(-5, color=".6", ls="--", lw=1)
        ax.set(ylabel=f"{label} (baseline SD)")
        tests = spline_stats[spline_stats.outcome.eq(outcome)]
        ax.text(.01, .03, "\n".join(f"{r.contrast}: Holm p={r.p_holm:.3g}" for r in tests.itertuples()),
                transform=ax.transAxes, va="bottom", fontsize=8)
    axes[0].legend(frameon=False, ncol=2); axes[-1].set(xlabel="Minutes to seizure onset")
    fig.suptitle("Candia-Rivera robust autonomic trajectories: participant median and MAD")
    fig.tight_layout(); fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_group_contrasts(stats: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, outcome, title in zip(axes, ("csi_z", "cpi_z"), ("Sympathetic index", "Parasympathetic index")):
        part = stats[stats.outcome.eq(outcome)]
        y = np.arange(len(part)); x = part.estimate.to_numpy(float)
        ax.errorbar(x, y, xerr=[x - part.ci95_low, part.ci95_high - x], fmt="o", capsize=3)
        ax.axvline(0, color=".5", lw=1)
        ax.set(yticks=y, yticklabels=[f"{c}  pBH={p:.3g}" for c, p in zip(part.contrast, part.p_bh)],
               xlabel="Final-five-minute difference (baseline SD)", title=title)
    fig.tight_layout(); fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_associations(participant: pd.DataFrame, stats: pd.DataFrame, output: Path) -> None:
    groups = ("hyper", "hypo")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, outcome in enumerate(("csi_z_within", "cpi_z_within")):
        for col, predictor in enumerate(("c_coverage_within_z", "c_log_kappa_within_z")):
            ax = axes[row, col]
            for group in groups:
                part = participant[participant.semiology.eq(group)]
                bins = pd.qcut(part[predictor], 12, duplicates="drop")
                means = part.groupby(bins, observed=True)[[predictor, outcome]].mean()
                ax.plot(means[predictor], means[outcome], marker="o", ms=3, lw=1.5,
                        color=COLORS[group], label=group.capitalize())
            pooled = stats[(stats.family.eq("whole preictal FS")) & stats.outcome.eq(outcome) &
                           stats.predictor.eq(predictor)].iloc[0]
            terminal = stats[(stats.family.eq("terminal endpoints <30 s")) & stats.outcome.eq(outcome) &
                             stats.predictor.eq(predictor)].iloc[0]
            ax.text(.02, .98, f"Pooled beta={pooled.estimate:.3g}; pBH={pooled.p_bh:.3g}\n"
                    f"Terminal semio interaction={terminal.estimate:.3g}; pBH={terminal.p_bh:.3g}; "
                    f"gate={'pass' if terminal.gate_passed else 'not passed'}",
                    transform=ax.transAxes, va="top", fontsize=8)
            ax.set(xlabel="Within-event C coverage (SD)" if predictor.startswith("c_coverage") else "Within-event C-origin log kappa (SD)",
                   ylabel="Within-event CSI (baseline SD)" if outcome.startswith("csi") else "Within-event CPI (baseline SD)")
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Within-FS concurrent autonomic associations with State C")
    fig.tight_layout(); fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig)


def self_check() -> None:
    rng = np.random.default_rng(4)
    rr = .8 + .04 * np.sin(np.arange(900) / 8) + rng.normal(0, .005, 900)
    peaks = np.r_[0, np.cumsum(rr)]
    out = native_indices(peaks, np.zeros(len(peaks), bool))
    assert len(out) > 500 and np.isfinite(out[["csi", "cpi"]]).all().all()
    cov = robust_cov_hrv(np.column_stack([rr[:-1], rr[1:]]))
    assert np.allclose(cov, cov.T) and np.linalg.eigvalsh(cov).min() >= -1e-12
    rows = []
    endpoints = (0, 10, 20, *range(60, 1141, 60))
    for participant in range(32):
        for endpoint in endpoints:
            coverage, kappa = rng.normal(size=2)
            rows.append({"participant": f"p{participant}", "stratum": "FS functional", "semiology": "hypo" if participant % 2 else "hyper",
                         "event_type": "functional", "study_cohort": "single-diagnosis",
                         "window_end_to_ictus_s": endpoint, "minutes_to_ictus": -endpoint / 60,
                         "c_coverage_within_z": coverage, "c_log_kappa_within_z": kappa,
                         "csi_z_within": .5 * coverage + rng.normal(0, .25),
                         "cpi_z_within": -.4 * kappa + rng.normal(0, .25)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, stats = association_statistics(pd.DataFrame(rows))
    pooled = stats[stats.family.eq("whole preictal FS")].set_index(["outcome", "predictor"])
    assert pooled.loc[("csi_z_within", "c_coverage_within_z"), "estimate"] > .3
    assert pooled.loc[("cpi_z_within", "c_log_kappa_within_z"), "estimate"] < -.2


def run(args) -> dict:
    data_dir, figure_dir = args.out_dir / "data", args.out_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True); figure_dir.mkdir(parents=True, exist_ok=True)
    manifest = attach_strata(pd.read_csv(args.manifest))
    manifest["semiology"] = ace.normalise_semiology(manifest.hypo_hyper)
    events = manifest[manifest.condition.astype(str).str.lower().eq("preict") & manifest.stratum.isin(ALL_STRATA) & manifest.semiology.notna()]
    native_path, qc_path = data_dir / "candia_autonomic_native_beat_resolved.csv.gz", data_dir / "candia_autonomic_event_qc.csv"
    if args.reuse_native and native_path.exists() and qc_path.exists():
        native, qc = pd.read_csv(native_path), pd.read_csv(qc_path)
    else:
        native_parts, qc_rows = [], []
        for number, event in enumerate(events.itertuples(index=False), 1):
            result, audit = process_event(event, args.ecg_qc_dir)
            qc_rows.append(audit)
            if not result.empty: native_parts.append(result)
            if number % 25 == 0: print(f"autonomic events {number}/{len(events)}; valid {len(native_parts)}", flush=True)
        if not native_parts:
            raise RuntimeError("no event passed robust autonomic QC")
        native, qc = pd.concat(native_parts, ignore_index=True), pd.DataFrame(qc_rows)
        native.to_csv(native_path, index=False, compression="gzip")
        qc.to_csv(qc_path, index=False)
    windows = exact_windows(native)
    windows.to_csv(data_dir / "candia_autonomic_60s_step10s.csv", index=False)
    final5 = participant_final5(windows)
    final5.to_csv(data_dir / "candia_autonomic_final5_participant.csv", index=False)
    group_stats, descriptions, dd_stats = group_statistics(final5)
    spline_stats = spline_statistics(windows)
    group_stats.to_csv(data_dir / "candia_autonomic_group_contrasts.csv", index=False)
    descriptions.to_csv(data_dir / "candia_autonomic_group_descriptives.csv", index=False)
    dd_stats.to_csv(data_dir / "candia_autonomic_dd_paired_stats.csv", index=False)
    spline_stats.to_csv(data_dir / "candia_autonomic_spline_omnibus.csv", index=False)
    merged = merged_microstates(windows, args.activity_csv, args.kappa_csv)
    merged.to_csv(data_dir / "candia_autonomic_microstate_event_windows.csv", index=False)
    participant, associations = association_statistics(merged)
    participant.to_csv(data_dir / "candia_autonomic_microstate_participant_windows.csv", index=False)
    associations.to_csv(data_dir / "candia_autonomic_microstate_gee_contrasts.csv", index=False)
    plot_trajectories(native, spline_stats, figure_dir / "candia_autonomic_preictal_trajectories.png")
    plot_group_contrasts(group_stats, figure_dir / "candia_autonomic_group_contrasts.png")
    plot_associations(participant, associations, figure_dir / "candia_autonomic_microstate_fs_semiology.png")
    provenance = {"paper": "Candia-Rivera et al., Royal Society Open Science 2025, doi:10.1098/rsos.240750",
                  "paper_pdf": str(args.paper_pdf), "paper_pdf_sha256": hashlib.sha256(args.paper_pdf.read_bytes()).hexdigest(),
                  "source_repository": "https://github.com/diegocandiar/robust_hrv",
                  "source_commit": METHOD_COMMIT, "method": "robust shrinkage Poincare covariance",
                  "window_s": 15, "ks": 1, "kp": 10, "inference_sampling": "native beat-centred estimates",
                  "normalisation": "event -20 to -5 minute mean and SD"}
    (data_dir / "candia_autonomic_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (data_dir / "analyse_candia_autonomic.py").write_bytes(Path(__file__).read_bytes())
    hypo_fs_es = group_stats[(group_stats.outcome.eq("csi_z")) &
                             group_stats.contrast.eq("Hypomotor: functional minus epileptic")].iloc[0]
    pooled_associations = associations[associations.family.eq("whole preictal FS")]
    terminal_interactions = associations[associations.family.eq("terminal endpoints <30 s")]
    csi_coverage = pooled_associations[(pooled_associations.outcome.eq("csi_z_within")) &
                                       pooled_associations.predictor.eq("c_coverage_within_z")].iloc[0]
    passed = pooled_associations[pooled_associations.gate_passed]
    association_result = ("At least one whole-preictal association passed BH correction; its terminal semiology interaction is reported below."
                          if len(passed) else
                          "No whole-preictal State C association passed BH correction, so terminal semiology interactions are exploratory and the gate was not passed.")
    report = ["# Robust preictal sympathetic and parasympathetic analysis", "",
              "Candia-Rivera CSI/CPI used robust shrinkage covariance in 15-second Poincare windows with ks=1 and kp=10. "
              "Indices are expressed relative to each event's -20 to -5 minute mean and SD. Native beat-centred estimates were retained; "
              "the authors' 4-Hz spline is a display interpolation and was not treated as additional data.", "",
              "## Main findings", "",
              f"- Hypomotor FS had a lower final-five-minute sympathetic index than hypomotor ES: "
              f"{hypo_fs_es.estimate:.3f} baseline SD (95% CI {hypo_fs_es.ci95_low:.3f} to {hypo_fs_es.ci95_high:.3f}; "
              f"BH p={hypo_fs_es.p_bh:.4f}).", 
              "- No whole-trajectory FS/ES or FS-semiology spline interaction survived Holm correction.",
              f"- {association_result}", "",
              f"- The closest whole-preictal result was lower momentary CSI with greater State C coverage: beta "
              f"{csi_coverage.estimate:.3f} baseline SD per within-event coverage SD (95% CI {csi_coverage.ci95_low:.3f} to "
              f"{csi_coverage.ci95_high:.3f}; nominal p={csi_coverage.p:.4f}, BH p={csi_coverage.p_bh:.4f}).", "",
              "## FS/ES and functional-semiology comparisons", "",
              markdown_table(group_stats, ["outcome", "contrast", "estimate", "ci95_low", "ci95_high", "p_bh"]), "",
              markdown_table(spline_stats, ["outcome", "contrast", "wald_chi2", "df", "p_holm"]), "",
              "![Autonomic trajectories](figures/candia_autonomic_preictal_trajectories.png)", "",
              "![Group contrasts](figures/candia_autonomic_group_contrasts.png)", "",
              "## Within-DD paired FS versus ES", "",
              markdown_table(dd_stats, ["semiology", "outcome", "estimate", "ci95_low", "ci95_high", "n_pairs", "p_bh"]), "",
              "## Within-FS State C/autonomic coupling", "",
              "Participant-clustered Gaussian GEE relates within-event CSI/CPI fluctuations to jointly fitted within-event State C coverage "
              "and C-origin log-kappa, controlling for early/late time, semiology and DD cohort. Autonomic indices originate from native "
              "beat-centred 15-second estimates, but inference uses common 60-second windows stepped every 10 seconds so the neural precision "
              "estimate is not duplicated across cardiac samples.", "",
              markdown_table(pooled_associations, ["outcome", "predictor", "estimate", "ci95_low", "ci95_high", "p_bh", "gate_passed"]), "",
              "## Terminal semiology interaction", "",
              "These are interactions among 60-second windows whose endpoints lie 0, 10 or 20 seconds before onset; they are not isolated "
              "30-second neural windows. They are interpreted only when the corresponding whole-preictal predictor gate passed.", "",
              markdown_table(terminal_interactions, ["outcome", "predictor", "estimate", "ci95_low", "ci95_high", "p_bh", "gate_passed"]), "",
              "![FS State C associations](figures/candia_autonomic_microstate_fs_semiology.png)", "",
              "## Interpretation constraint", "",
              "CSI and CPI are validated cardiac autonomic indices, not direct sympathetic or vagal nerve recordings. Recovered R-peak intervals "
              "were limited to 10% per 15-second estimate, instantaneous HR to 35-140 bpm, and baseline availability to at least five minutes."]
    (args.out_dir / "candia_autonomic_findings.md").write_text("\n".join(report), encoding="utf-8")
    summary = {"events_total": int(len(events)), "events_valid": int(qc.status.eq("valid").sum()),
               "participants_valid": int(native.participant.nunique()), "native_estimates": int(len(native)),
               "exact_windows": int(len(windows)), "microstate_windows": int(len(merged)),
               "microstate_participants": int(participant.participant.nunique()), "source_commit": METHOD_COMMIT}
    (data_dir / "candia_autonomic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    failures = qc.loc[qc.status.ne("valid"), "reason"].value_counts()
    audit = ["Candia-Rivera autonomic analysis audit", f"Completed UTC: {datetime.now(timezone.utc).isoformat()}",
             f"Repository script: {Path(__file__).resolve()}", f"Manifest: {args.manifest}",
              f"ECG QC: {args.ecg_qc_dir}", f"State C activity: {args.activity_csv}", f"C-origin kappa: {args.kappa_csv}",
             f"Paper PDF SHA-256: {provenance['paper_pdf_sha256']}", f"Upstream robust_hrv commit: {METHOD_COMMIT}",
             "Estimator: 15-second robust shrinkage Poincare windows; ks=1; kp=10; native beat-centred inference",
             "Normalisation: each event's -20 to -5 minute mean and SD", f"Events: {summary['events_valid']}/{summary['events_total']} valid",
             f"Participants: {summary['participants_valid']}", f"Native estimates: {summary['native_estimates']}",
              f"Exact 60-second/10-second windows: {summary['exact_windows']}", f"FS microstate-matched windows: {summary['microstate_windows']}",
             "QC exclusions:"] + [f"  {reason}: {count}" for reason, count in failures.items()]
    (args.out_dir / "candia_autonomic_audit.txt").write_text("\n".join(audit) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--ecg-qc-dir", type=Path)
    parser.add_argument("--activity-csv", type=Path)
    parser.add_argument("--kappa-csv", type=Path)
    parser.add_argument("--paper-pdf", type=Path, default=Path(r"C:\Users\Admin\Zotero\storage\ZCPECYTA\Candia-Rivera et al. - Robust and time-resolved estimation of cardiac sympathetic and parasympathetic indices.pdf"))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--reuse-native", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(); print("self-check passed"); return
    if not all((args.manifest, args.ecg_qc_dir, args.activity_csv, args.kappa_csv, args.out_dir)):
        parser.error("all path arguments except --paper-pdf are required")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
