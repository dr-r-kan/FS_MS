"""Reciprocal-lag autonomic/microstate analysis with genuine interictal sensitivity."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import analyse_ace_cardioneural as ace
from analyse_candia_autonomic import native_indices
from analyse_preictal_publication import attach_strata
from analyse_requested_comparisons import markdown_table
from microstate_labels import load_global_labels


LABELS = ("F", "E", "G", "B", "A", "C")
NEURAL = {
    "c_coverage": "C coverage",
    "c_log_kappa": "C-origin log kappa",
    "c_js_from_ii": "C-origin JS from II",
}
AUTONOMIC = {"csi": "CSI", "cpi": "CPI"}
COLORS = {"ES": "#7A5195", "FS": "#009E73", "Hyper": "#D55E00", "Hypo": "#0072B2",
          "II": "#666666", "Preictal": "#CC79A7"}


def normalise_path(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace("/", "\\", regex=False).str.lower()


def nonoverlap_preictal(autonomic_csv: Path, activity_csv: Path, statewise_csv: Path) -> pd.DataFrame:
    autonomic = pd.read_csv(autonomic_csv)
    autonomic = autonomic[autonomic.window_end_to_ictus_s.mod(60).eq(0)].copy()
    activity = pd.read_csv(activity_csv)
    activity = activity[(activity.template_label.eq("C")) & activity.backfit_method.eq("hard") &
                        activity.source.eq("participant") & activity.scheme.eq("20min_60s_step10s")]
    activity = activity[["participant", "file_path", "window_end_to_ictus_s", "coverage"]].rename(
        columns={"coverage": "c_coverage"})
    statewise = pd.read_csv(statewise_csv)
    statewise = statewise[statewise.template_label.eq("C")][
        ["participant", "file_path", "window_end_to_ictus_s", "log_kappa", "js_from_II"]].rename(
        columns={"log_kappa": "c_log_kappa", "js_from_II": "c_js_from_ii"})
    keys = ["participant", "file_path", "window_end_to_ictus_s"]
    out = autonomic.merge(activity, on=keys, validate="one_to_one").merge(statewise, on=keys, validate="one_to_one")
    out["chronological_s"] = -out.window_end_to_ictus_s
    out["is_functional"] = out.event_type.eq("functional").astype(int)
    out["is_hypo"] = out.semiology.eq("hypo").astype(int)
    out["is_dd"] = out.study_cohort.eq("DD").astype(int)
    return out


def within_record_z(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    grouped = out.groupby(["participant", "file_path"], observed=True)
    for column in columns:
        mean = grouped[column].transform("mean")
        sd = grouped[column].transform("std")
        out[f"{column}_z"] = (out[column] - mean) / sd.where(sd.gt(0))
    return out


def lagged_rows(frame: pd.DataFrame, columns: list[str], reverse: bool = False) -> pd.DataFrame:
    order = not reverse
    parts = []
    for _, record in frame.groupby(["participant", "file_path"], observed=True, sort=False):
        record = record.sort_values("chronological_s", ascending=order).copy()
        if len(record) < 8:
            continue
        gap = record.chronological_s.diff().abs()
        for column in columns:
            record[f"{column}_lag"] = record[column].shift(1)
        record["time_z"] = np.linspace(-1, 1, len(record))
        record = record[(gap.isna() | np.isclose(gap, 60))]
        parts.append(record.iloc[1:])
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def fit_one(frame: pd.DataFrame, outcome: str, predictor: str, own: str, other: str,
            indicator: str, covariates: list[str], analysis: str):
    columns = [outcome, predictor, own, other, indicator, "participant", "time_z", *covariates]
    use = frame.dropna(subset=columns).copy()
    if use.participant.nunique() < 8 or use[indicator].nunique() != 2:
        raise ValueError(f"{analysis}: insufficient participants/groups")
    rhs = [own, predictor, other, "time_z", indicator, *covariates, f"{predictor}:{indicator}"]
    fit = smf.gee(f"{outcome} ~ {' + '.join(rhs)}", groups="participant", data=use,
                  family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Independence()).fit(maxiter=100)
    return fit, use


def crosslag_models(lagged: pd.DataFrame, comparison: str) -> pd.DataFrame:
    if comparison == "FS versus ES":
        frame, indicator, group0, group1, covariates = lagged.copy(), "is_functional", "ES", "FS", ["is_hypo", "is_dd"]
    elif comparison == "FS hypomotor versus hypermotor":
        frame = lagged[lagged.event_type.eq("functional")].copy()
        indicator, group0, group1, covariates = "is_hypo", "Hyper", "Hypo", ["is_dd"]
    else:
        raise ValueError(comparison)
    rows = []
    for neural in NEURAL:
        nz = f"{neural}_z"
        for autonomic in AUTONOMIC:
            az, other = f"{autonomic}_z", f"{'cpi' if autonomic == 'csi' else 'csi'}_z"
            specifications = (
                ("autonomic to neural", nz, f"{az}_lag", f"{nz}_lag", f"{other}_lag"),
                ("neural to autonomic", az, f"{nz}_lag", f"{az}_lag", f"{other}_lag"),
            )
            for direction, outcome, predictor, own, other_lag in specifications:
                label = f"{comparison}; {AUTONOMIC[autonomic]}; {NEURAL[neural]}; {direction}"
                try:
                    fit, use = fit_one(frame, outcome, predictor, own, other_lag, indicator, covariates, label)
                    interaction = f"{predictor}:{indicator}"
                    for group, weights in ((group0, {predictor: 1}),
                                           (group1, {predictor: 1, interaction: 1}),
                                           (f"{group1} minus {group0}", {interaction: 1})):
                        result = ace.linear_contrast(fit, weights, group)
                        rows.append({"comparison": comparison, "direction": direction, "autonomic": autonomic,
                                     "neural": neural, "group": group, "n_rows": len(use),
                                     "n_participants": use.participant.nunique(), **result})
                except (ValueError, KeyError, np.linalg.LinAlgError) as error:
                    rows.append({"comparison": comparison, "direction": direction, "autonomic": autonomic,
                                 "neural": neural, "group": f"{group1} minus {group0}", "error": str(error)})
    out = pd.DataFrame(rows)
    if "p" not in out:
        out["p"] = np.nan
    out["p_bh"] = np.nan
    difference = out.group.eq(f"{group1} minus {group0}")
    out.loc[difference, "p_bh"] = ace.bh(out.loc[difference, "p"])
    return out


def read_ii_native(manifest: pd.DataFrame, ecg_qc_dir: Path, original_ecg_qc_dir: Path,
                   native_path: Path, qc_path: Path, reuse: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if reuse and native_path.exists() and qc_path.exists():
        return pd.read_csv(native_path), pd.read_csv(qc_path)
    rows, audits = [], []
    records = manifest[manifest.condition.astype(str).str.lower().eq("ii")]
    for number, event in enumerate(records.itertuples(index=False), 1):
        sidecar = ace.rpeak_file(ecg_qc_dir, event.source_raw_fif)
        if not sidecar.exists():
            sidecar = ace.rpeak_file(original_ecg_qc_dir, event.source_raw_fif)
        peaks, recovered = ace.read_rpeaks_with_recovery_flags(sidecar)
        start, stop = float(event.crop_start_seconds), float(event.crop_end_seconds)
        keep = (peaks >= start) & (peaks <= stop)
        audit = {"participant": event.participant, "file_path": event.file_path, "rpeak_sidecar": str(sidecar),
                 "status": "failed", "reason": "", "n_segment_peaks": int(keep.sum())}
        if keep.sum() < 40:
            audit["reason"] = "fewer than 40 R-peaks"
        else:
            try:
                native = native_indices(peaks[keep] - start, recovered[keep])
                native.insert(0, "participant", event.participant)
                native.insert(1, "file_path", event.file_path)
                rows.append(native)
                audit.update(status="valid", n_native_estimates=len(native))
            except ValueError as error:
                audit["reason"] = str(error)
        audits.append(audit)
        if number % 40 == 0:
            print(f"interictal autonomic records {number}/{len(records)}; valid {len(rows)}", flush=True)
    if not rows:
        raise RuntimeError("No interictal ECG record passed autonomic QC")
    native, qc = pd.concat(rows, ignore_index=True), pd.DataFrame(audits)
    native.to_csv(native_path, index=False, compression="gzip")
    qc.to_csv(qc_path, index=False)
    return native, qc


def transition_matrix(row) -> np.ndarray:
    return np.array([[getattr(row, f"t_{i}_{j}") for j in range(1, 7)] for i in range(1, 7)], float)


def peak_neural_metrics(path: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    table = pd.read_csv(path)
    table["file_key"] = normalise_path(table.file_path)
    metadata = manifest[["file_key", "stratum", "hypo_hyper"]]
    table = table.drop(columns=[name for name in ("stratum", "hypo_hyper") if name in table]).merge(
        metadata, on="file_key", how="left", validate="many_to_one")
    baseline = {}
    transition_columns = [f"t_{i}_{j}" for i in range(1, 7) for j in range(1, 7)]
    for participant, frame in table[table.condition.eq("ii")].groupby("participant", observed=True):
        counts = frame[transition_columns].sum().to_numpy(float).reshape(6, 6)
        probs = counts + .5
        np.fill_diagonal(probs, 0)
        probs /= probs.sum(axis=1, keepdims=True)
        baseline[str(participant)] = (probs, counts.sum(axis=1))
    rows = []
    for row in table.itertuples(index=False):
        reference = baseline.get(str(row.participant))
        if reference is None or row.n_gfp_peaks < 30:
            continue
        b0, ii_n = reference
        counts = transition_matrix(row)
        enough = counts[5].sum() >= 5 and ii_n[5] >= 5
        logk, _ = ace.fit_kappa(counts, b0, (5,), 5) if enough else (np.nan, 0)
        rows.append({"participant": row.participant, "file_path": row.file_path, "condition": row.condition,
                     "window_index": row.window_index, "window_start_s": row.window_start_s,
                     "window_end_s": row.window_end_s, "window_end_to_ictus_s": row.window_end_to_ictus_s,
                     "stratum": row.stratum, "hypo_hyper": row.hypo_hyper,
                     "c_peak_expression": row.n_C / row.n_gfp_peaks, "c_peak_log_kappa": logk,
                     "c_peak_js_from_ii": ace.js_divergence(counts, b0, (5,)) if enough else np.nan,
                     "n_gfp_peaks": row.n_gfp_peaks, "c_outgoing_transitions": counts[5].sum()})
    return pd.DataFrame(rows)


def window_native(native: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    lookup = {key: value.sort_values("event_time_s") for key, value in native.groupby(["participant", "file_path"], observed=True)}
    rows = []
    for row in windows.itertuples(index=False):
        frame = lookup.get((row.participant, row.file_path))
        if frame is None:
            continue
        part = frame[frame.event_time_s.between(row.window_start_s, row.window_end_s, inclusive="left")]
        span = part.event_time_s.max() - part.event_time_s.min() if len(part) else 0
        if len(part) < 30 or span < 45:
            continue
        rows.append({"participant": row.participant, "file_path": row.file_path, "window_index": row.window_index,
                     "csi": part.csi.mean(), "cpi": part.cpi.mean(), "n_native": len(part), "native_span_s": span})
    return pd.DataFrame(rows)


def baseline_sensitivity(peak_windows: Path, manifest: pd.DataFrame, preict_native_path: Path,
                         ecg_qc_dir: Path, original_ecg_qc_dir: Path, data_dir: Path,
                         reuse: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ii_native, ii_qc = read_ii_native(manifest, ecg_qc_dir, original_ecg_qc_dir,
                                      data_dir / "directional_interictal_native.csv.gz",
                                      data_dir / "directional_interictal_autonomic_qc.csv", reuse)
    preict_native = pd.read_csv(preict_native_path)
    native = pd.concat([preict_native[["participant", "file_path", "event_time_s", "csi", "cpi"]],
                        ii_native[["participant", "file_path", "event_time_s", "csi", "cpi"]]], ignore_index=True)
    neural = peak_neural_metrics(peak_windows, manifest)
    autonomic = window_native(native, neural)
    merged = neural.merge(autonomic, on=["participant", "file_path", "window_index"], validate="one_to_one")
    merged["event_type"] = np.select([merged.stratum.isin(["FS functional", "DD functional"]),
                                      merged.stratum.isin(["ES epileptic", "DD epileptic"])],
                                     ["functional", "epileptic"], default="interictal")
    merged["semiology"] = ace.normalise_semiology(merged.hypo_hyper)
    dd_participants = set(manifest.loc[manifest.group.astype(str).str.upper().eq("DD"), "participant"])
    merged["is_dd"] = merged.participant.isin(dd_participants).astype(int)
    merged["chronological_s"] = merged.window_start_s
    columns = ["csi", "cpi", "c_peak_expression", "c_peak_log_kappa", "c_peak_js_from_ii"]
    merged = within_record_z(merged, columns)
    lagged = lagged_rows(merged, [f"{column}_z" for column in columns])
    rows = []
    for event_type, label in (("functional", "FS preictal versus II"), ("epileptic", "ES preictal versus II")):
        participants = set(lagged.loc[lagged.event_type.eq(event_type), "participant"])
        frame = lagged[lagged.participant.isin(participants) & lagged.event_type.isin([event_type, "interictal"])].copy()
        frame["is_preictal"] = frame.event_type.eq(event_type).astype(int)
        for neural_name in ("c_peak_expression", "c_peak_log_kappa", "c_peak_js_from_ii"):
            nz = f"{neural_name}_z"
            for autonomic_name in AUTONOMIC:
                az = f"{autonomic_name}_z"; other = f"{'cpi' if autonomic_name == 'csi' else 'csi'}_z"
                for direction, outcome, predictor, own, other_lag in (
                    ("autonomic to neural", nz, f"{az}_lag", f"{nz}_lag", f"{other}_lag"),
                    ("neural to autonomic", az, f"{nz}_lag", f"{az}_lag", f"{other}_lag")):
                    try:
                        fit, use = fit_one(frame, outcome, predictor, own, other_lag, "is_preictal", ["is_dd"], label)
                        interaction = f"{predictor}:is_preictal"
                        for group, weights in (("II", {predictor: 1}), ("Preictal", {predictor: 1, interaction: 1}),
                                               ("Preictal minus II", {interaction: 1})):
                            result = ace.linear_contrast(fit, weights, group)
                            rows.append({"comparison": label, "direction": direction, "autonomic": autonomic_name,
                                         "neural": neural_name, "group": group, "n_rows": len(use),
                                         "n_participants": use.participant.nunique(), **result})
                    except (ValueError, KeyError, np.linalg.LinAlgError) as error:
                        rows.append({"comparison": label, "direction": direction, "autonomic": autonomic_name,
                                     "neural": neural_name, "group": "Preictal minus II", "error": str(error)})
    stats = pd.DataFrame(rows)
    if "p" not in stats:
        stats["p"] = np.nan
    stats["p_bh"] = np.nan
    for comparison, index in stats[stats.group.eq("Preictal minus II")].groupby("comparison").groups.items():
        stats.loc[index, "p_bh"] = ace.bh(stats.loc[index, "p"])
    return merged, stats, ii_qc


def forest_plot(stats: pd.DataFrame, output: Path, differences: bool = False) -> None:
    comparisons = ["FS versus ES", "FS hypomotor versus hypermotor"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True)
    for ax, comparison in zip(axes, comparisons):
        part = stats[stats.comparison.eq(comparison)].copy()
        if differences:
            suffix = "FS minus ES" if comparison == "FS versus ES" else "Hypo minus Hyper"
            part = part[part.group.eq(suffix)]
            labels = [f"{AUTONOMIC[r.autonomic]} {'→' if r.direction == 'autonomic to neural' else '←'} {NEURAL[r.neural]}\nBH p={r.p_bh:.3g}"
                      for r in part.itertuples()]
            y = np.arange(len(part)); x = part.estimate.to_numpy(float)
            ax.errorbar(x, y, xerr=[x - part.ci95_low, part.ci95_high - x], fmt="o", capsize=3, color="#333333")
            ax.set(yticks=y, yticklabels=labels, title=comparison, xlabel="Difference in standardised lag coefficient")
        else:
            groups = ("ES", "FS") if comparison == "FS versus ES" else ("Hyper", "Hypo")
            base = part[part.group.eq(groups[0])]
            labels = [f"{AUTONOMIC[r.autonomic]} {'→' if r.direction == 'autonomic to neural' else '←'} {NEURAL[r.neural]}"
                      for r in base.itertuples()]
            y = np.arange(len(base))
            for offset, group in zip((-.13, .13), groups):
                values = part[part.group.eq(group)].set_index(["direction", "autonomic", "neural"]).loc[
                    list(zip(base.direction, base.autonomic, base.neural))].reset_index()
                x = values.estimate.to_numpy(float)
                ax.errorbar(x, y + offset, xerr=[x - values.ci95_low, values.ci95_high - x], fmt="o", capsize=2,
                            color=COLORS[group], label=group)
            diff_group = f"{groups[1]} minus {groups[0]}"
            diff = part[part.group.eq(diff_group)].set_index(["direction", "autonomic", "neural"])
            labels = [f"{label}\npBH={diff.loc[(r.direction, r.autonomic, r.neural), 'p_bh']:.3g}"
                      for label, r in zip(labels, base.itertuples())]
            ax.set(yticks=y, yticklabels=labels, title=comparison, xlabel="Standardised lag coefficient")
            ax.legend(frameon=False)
        ax.axvline(0, color=".5", lw=1); ax.grid(axis="x", alpha=.2); ax.invert_yaxis()
    fig.suptitle("Reciprocal one-window (60 s) cardioneural predictive coupling\n95% robust confidence intervals; non-overlapping windows")
    fig.tight_layout(); fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig)


def baseline_plot(stats: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True)
    for ax, comparison in zip(axes, ("FS preictal versus II", "ES preictal versus II")):
        part = stats[(stats.comparison.eq(comparison)) & stats.group.eq("Preictal minus II")]
        y = np.arange(len(part)); x = part.estimate.to_numpy(float)
        peak_labels = {"c_peak_expression": "C GFP-peak expression", "c_peak_log_kappa": "C-origin log kappa",
                       "c_peak_js_from_ii": "C-origin JS from II"}
        labels = [f"{AUTONOMIC[r.autonomic]} {'→' if r.direction == 'autonomic to neural' else '←'} "
                  f"{peak_labels[r.neural]}\nBH p={r.p_bh:.3g}"
                  for r in part.itertuples()]
        ax.errorbar(x, y, xerr=[x - part.ci95_low, part.ci95_high - x], fmt="o", capsize=3, color=COLORS["Preictal"])
        ax.axvline(0, color=".5", lw=1); ax.set(yticks=y, yticklabels=labels, title=comparison,
                                               xlabel="Preictal minus genuine-II lag coefficient")
        ax.grid(axis="x", alpha=.2); ax.invert_yaxis()
    fig.suptitle("Estimator-matched GFP-peak backfit sensitivity against genuine interictal recordings\n95% robust confidence intervals")
    fig.tight_layout(); fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig)


def self_check() -> None:
    rng = np.random.default_rng(28)
    rows = []
    for participant in range(80):
        functional = participant >= 40
        for event in range(2):
            a = rng.normal(); m = rng.normal()
            for time in range(20):
                previous_a, previous_m = a, m
                a = .45 * previous_a + rng.normal(scale=.7)
                m = .45 * previous_m + (.05 + .45 * functional) * previous_a + rng.normal(scale=.7)
                rows.append({"participant": f"p{participant}", "file_path": f"p{participant}_e{event}",
                             "chronological_s": time * 60, "event_type": "functional" if functional else "epileptic",
                             "is_functional": int(functional), "is_hypo": participant % 2, "is_dd": 0,
                             "semiology": "hypo" if participant % 2 else "hyper", "csi_z": a, "cpi_z": rng.normal(),
                             "c_coverage_z": m, "c_log_kappa_z": rng.normal(), "c_js_from_ii_z": rng.normal()})
    lagged = lagged_rows(pd.DataFrame(rows), ["csi_z", "cpi_z", "c_coverage_z", "c_log_kappa_z", "c_js_from_ii_z"])
    stats = crosslag_models(lagged, "FS versus ES")
    recovered = stats[(stats.direction.eq("autonomic to neural")) & stats.autonomic.eq("csi") &
                      stats.neural.eq("c_coverage") & stats.group.eq("FS minus ES")].iloc[0]
    assert recovered.estimate > .2 and recovered.p < .05
    counts = np.ones((6, 6)); np.fill_diagonal(counts, 0)
    b0 = counts / counts.sum(axis=1, keepdims=True)
    assert np.isfinite(ace.fit_kappa(counts, b0, (5,), 5)[0])


def run(args) -> dict:
    data_dir, figure_dir = args.out_dir / "data", args.out_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True); figure_dir.mkdir(parents=True, exist_ok=True)
    fitted_order = tuple(load_global_labels(args.results_dir))
    if len(fitted_order) != 6 or set(fitted_order) != set(LABELS):
        raise ValueError("Directional analysis requires a one-to-one manual F/E/G/B/A/C label mapping")
    manifest = attach_strata(pd.read_csv(args.manifest))
    primary = nonoverlap_preictal(args.autonomic_dir / "data" / "candia_autonomic_60s_step10s.csv",
                                  args.activity_csv, args.statewise_csv)
    primary = primary.rename(columns={"csi_z": "csi_baseline_z", "cpi_z": "cpi_baseline_z"})
    primary = within_record_z(primary, ["csi_baseline_z", "cpi_baseline_z", *NEURAL])
    primary = primary.rename(columns={"csi_baseline_z_z": "csi_z", "cpi_baseline_z_z": "cpi_z"})
    lagged = lagged_rows(primary, ["csi_z", "cpi_z", *[f"{name}_z" for name in NEURAL]])
    forward = pd.concat([crosslag_models(lagged, comparison) for comparison in
                         ("FS versus ES", "FS hypomotor versus hypermotor")], ignore_index=True)
    reversed_lagged = lagged_rows(primary, ["csi_z", "cpi_z", *[f"{name}_z" for name in NEURAL]], reverse=True)
    reversed_stats = pd.concat([crosslag_models(reversed_lagged, comparison) for comparison in
                                ("FS versus ES", "FS hypomotor versus hypermotor")], ignore_index=True)
    primary.to_csv(data_dir / "directional_primary_nonoverlap_windows.csv.gz", index=False, compression="gzip")
    lagged.to_csv(data_dir / "directional_primary_lagged_windows.csv.gz", index=False, compression="gzip")
    forward.to_csv(data_dir / "directional_primary_gee_contrasts.csv", index=False)
    reversed_stats.to_csv(data_dir / "directional_time_reversal_sensitivity.csv", index=False)
    peak_merged, baseline_stats, ii_qc = baseline_sensitivity(
        args.peak_windows_csv, manifest, args.autonomic_dir / "data" / "candia_autonomic_native_beat_resolved.csv.gz",
        args.ecg_qc_dir, args.original_ecg_qc_dir, data_dir, args.reuse_interictal)
    peak_merged.to_csv(data_dir / "directional_peak_backfit_common_windows.csv.gz", index=False, compression="gzip")
    baseline_stats.to_csv(data_dir / "directional_interictal_baseline_gee_contrasts.csv", index=False)
    forest_plot(forward, figure_dir / "directional_group_coefficients.png")
    forest_plot(forward, figure_dir / "directional_group_differences.png", differences=True)
    baseline_plot(baseline_stats, figure_dir / "directional_interictal_sensitivity.png")
    differences = forward[forward.group.str.contains("minus", na=False)]
    significant = differences[differences.p_bh.lt(.05)]
    baseline_differences = baseline_stats[baseline_stats.group.eq("Preictal minus II")]
    baseline_significant = baseline_differences[baseline_differences.p_bh.lt(.05)]
    report = ["# Directional cardioneural predictive-coupling analysis", "",
              "## Estimand and safeguards", "",
              "The primary analysis uses non-overlapping 60-second windows. Reciprocal lag models ask whether the preceding autonomic "
              "window predicts the current State C measure after its own preceding value, and vice versa. CSI and CPI enter mutually "
              "adjusted models, with participant-clustered sandwich-robust GEE (independence working correlation), linear preictal time, cohort, and the relevant group terms. "
              "All measures are standardised within event, so coefficients describe momentary coupling rather than between-person level differences.", "",
              "The genuine-interictal sensitivity uses the manually aligned participant templates to backfit cached GFP-peak maps in both "
              "interictal and preictal records. It performs no microstate refitting. Its C-expression estimand is the fraction of GFP peaks "
              "assigned to C, so it is deliberately kept separate from the primary full-sample C coverage analysis.", "",
              "These are tests of temporal precedence or predictive coupling, not causal autonomic-to-brain or brain-to-autonomic flow.", "",
              "## FS versus ES and FS hypomotor versus hypermotor", "",
              f"{len(significant)} of {len(differences)} corrected primary group-difference tests passed BH q<0.05.", "",
              markdown_table(forward[forward.group.str.contains("minus", na=False)],
                             ["comparison", "direction", "autonomic", "neural", "estimate", "ci95_low", "ci95_high", "p", "p_bh", "n_participants"]), "",
              "![Group-specific coupling estimates](figures/directional_group_coefficients.png)", "",
              "![Group differences](figures/directional_group_differences.png)", "",
              "## Genuine interictal baseline sensitivity", "",
              f"{len(baseline_significant)} of {len(baseline_differences)} corrected preictal-versus-II coupling tests passed BH q<0.05.", "",
              markdown_table(baseline_differences,
                             ["comparison", "direction", "autonomic", "neural", "estimate", "ci95_low", "ci95_high", "p", "p_bh", "n_participants"]), "",
              "![Interictal sensitivity](figures/directional_interictal_sensitivity.png)", "",
              "## Time-reversal control", "",
              "The complete reversed-time model table is supplied as `data/directional_time_reversal_sensitivity.csv`. A forward association "
              "that is equally strong after reversing within-event time is not direction-specific and should not be interpreted as flow."]
    (args.out_dir / "directional_cardioneural_findings.md").write_text("\n".join(report), encoding="utf-8")
    summary = {"completed_utc": datetime.now(timezone.utc).isoformat(), "primary_windows": len(primary),
               "primary_lagged_windows": len(lagged), "primary_participants": int(lagged.participant.nunique()),
               "primary_group_tests": len(differences), "primary_bh_significant": len(significant),
               "interictal_records_valid": int(ii_qc.status.eq("valid").sum()), "peak_common_windows": len(peak_merged),
               "baseline_group_tests": len(baseline_differences), "baseline_bh_significant": len(baseline_significant)}
    (data_dir / "directional_cardioneural_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    audit = ["Directional cardioneural analysis audit", f"Completed UTC: {summary['completed_utc']}",
             f"Results hierarchy: {args.results_dir}", f"Peak-window export: {args.peak_windows_csv}",
             "Primary estimator: full-sample C coverage plus C-origin kappa/JS; non-overlapping 60-second windows",
             "Baseline sensitivity: cached GFP peaks backfit to manually aligned participant templates; no fitting/refitting",
             "Model: reciprocal one-window lag GEE with independence working correlation and participant-clustered sandwich SE; own-lag, other-autonomic-lag, linear time and group covariates",
             "Inference unit: participant cluster; two-sided robust 95% CI; BH correction within comparison family",
             "Interpretation: predictive temporal precedence, not causality", json.dumps(summary, indent=2)]
    (args.out_dir / "directional_cardioneural_audit.txt").write_text("\n".join(audit) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--ecg-qc-dir", type=Path)
    parser.add_argument("--original-ecg-qc-dir", type=Path)
    parser.add_argument("--autonomic-dir", type=Path)
    parser.add_argument("--activity-csv", type=Path)
    parser.add_argument("--statewise-csv", type=Path)
    parser.add_argument("--peak-windows-csv", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--reuse-interictal", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self_check()
        print("self-check passed")
        return
    required = (args.manifest, args.results_dir, args.ecg_qc_dir, args.original_ecg_qc_dir, args.autonomic_dir,
                args.activity_csv, args.statewise_csv, args.peak_windows_csv, args.out_dir)
    if not all(required):
        parser.error("all path arguments are required")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
