"""A/C/E activity, E-gating, transition-precision, cardiac and cardioneural analysis.

The inferential unit is the participant. Events are averaged equally within a
participant after event-level QC. Morphology is handled separately by
``analyse_preictal_morphology_rft.py`` and uses only window-refitted maps.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf

from analyse_preictal_publication import (
    attach_strata,
    build_cardiac,
    build_rolling,
    fit_cardiac_models,
    piecewise_columns,
    rolling_outlier_audit,
    validate_event_window_outliers,
)
from microstate_labels import load_global_labels


LABELS = ("F", "E", "G", "B", "A", "C")
FOCAL = ("A", "C", "E")
FUNCTIONAL = ("FS functional", "DD functional")
FS = 256.0


def normalise_semiology(values: pd.Series) -> pd.Series:
    out = values.astype(str).str.strip().str.lower()
    return out.where(out.isin(["hypo", "hyper"]))


def add_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["cohort"] = np.where(out["stratum"].eq("FS functional"), "FS", "DD")
    out["semiology"] = normalise_semiology(out["hypo_hyper"])
    return out


def model_terms(result, analysis: str) -> pd.DataFrame:
    ci = result.conf_int()
    return pd.DataFrame({
        "analysis": analysis,
        "term": result.params.index,
        "estimate": result.params.to_numpy(float),
        "se": result.bse.to_numpy(float),
        "z": result.tvalues.to_numpy(float),
        "p_two_sided": result.pvalues.to_numpy(float),
        "ci95_low": ci.iloc[:, 0].to_numpy(float),
        "ci95_high": ci.iloc[:, 1].to_numpy(float),
        "n_rows": int(result.nobs),
        "n_participants": int(pd.Series(result.model.groups).nunique()),
    })


def fit_gee(frame: pd.DataFrame, formula: str, analysis: str):
    required = {formula.split("~", 1)[0].strip(), "participant"}
    use = frame.dropna(subset=list(required)).copy()
    if use["participant"].nunique() < 8:
        raise ValueError(f"{analysis}: fewer than eight participants")
    result = smf.gee(
        formula,
        groups="participant",
        data=use,
        family=sm.families.Gaussian(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit(maxiter=200)
    return result, model_terms(result, analysis)


def linear_contrast(result, weights: dict[str, float], name: str, alternative: str = "two-sided") -> dict:
    vector = np.zeros(len(result.params), dtype=float)
    for term, weight in weights.items():
        if term not in result.params.index:
            raise KeyError(f"{name}: missing model term {term!r}")
        vector[result.params.index.get_loc(term)] = weight
    estimate = float(vector @ result.params.to_numpy(float))
    se = float(np.sqrt(vector @ result.cov_params().to_numpy(float) @ vector))
    z = estimate / se if se > 0 else np.nan
    if alternative == "greater":
        p = float(norm.sf(z))
    elif alternative == "less":
        p = float(norm.cdf(z))
    else:
        p = float(2 * norm.sf(abs(z)))
    return {"contrast": name, "estimate": estimate, "se": se, "z": z,
            "alternative": alternative, "p": p,
            "ci95_low": estimate - 1.96 * se, "ci95_high": estimate + 1.96 * se}


def holm(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors="coerce").to_numpy(float)
    out = np.full(p.shape, np.nan)
    good = np.flatnonzero(np.isfinite(p))
    if not len(good):
        return pd.Series(out, index=pvalues.index)
    order = good[np.argsort(p[good])]
    adjusted = np.maximum.accumulate([(len(order) - i) * p[idx] for i, idx in enumerate(order)])
    out[order] = np.minimum(adjusted, 1.0)
    return pd.Series(out, index=pvalues.index)


def bh(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors="coerce").to_numpy(float)
    out = np.full(p.shape, np.nan)
    good = np.flatnonzero(np.isfinite(p))
    if not len(good):
        return pd.Series(out, index=pvalues.index)
    order = good[np.argsort(p[good])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.minimum(ranked, 1.0)
    return pd.Series(out, index=pvalues.index)


def activity_analysis(results_dir: Path, manifest: pd.DataFrame, data_dir: Path, figure_dir: Path):
    time_csv = results_dir / "participant_condition_preictal_time_metrics.csv"
    participant, event = build_rolling(time_csv, "participant", set(FOCAL), manifest,
                                       include_primary=True, include_final=False)
    validate_event_window_outliers(event)
    event.to_csv(data_dir / "ace_activity_rolling_event_raw_cleaned.csv", index=False)
    participant.to_csv(data_dir / "ace_activity_rolling_participant.csv", index=False)
    rolling_outlier_audit(event).to_csv(data_dir / "ace_activity_outlier_audit.csv", index=False)

    hard = event[event["backfit_method"].eq("hard")].copy()
    keys = ["participant", "file_path", "file_key", "stratum", "hypo_hyper", "seizure_type",
            "scheme", "window_s", "step_s", "window_end_to_ictus_s", "minutes_to_ictus"]
    cov = hard.pivot_table(index=keys, columns="template_label", values="coverage", aggfunc="first")
    corr = hard.pivot_table(index=keys, columns="template_label", values="template_match_abs_correlation", aggfunc="first")
    raw = hard.pivot_table(index=keys, columns="template_label", values="coverage_raw", aggfunc="first")
    common_index = cov.index.union(raw.index).union(corr.index)
    out = pd.DataFrame(index=common_index)
    for label in FOCAL:
        out[f"coverage_{label}"] = cov[label].reindex(common_index) if label in cov else np.nan
        out[f"coverage_raw_{label}"] = raw[label].reindex(common_index) if label in raw else np.nan
        out[f"reference_correlation_{label}"] = corr[label].reindex(common_index) if label in corr else np.nan
    out = out.reset_index()
    out["reference_quality_ok"] = out[[f"reference_correlation_{x}" for x in FOCAL]].ge(.50).all(axis=1)
    for label in FOCAL:
        out[f"n_{label}"] = out[f"coverage_{label}"] * out["window_s"] * FS
    out["z_C_over_A"] = np.log((out["n_C"] + .5) / (out["n_A"] + .5)) / np.sqrt(2)
    out["z_E_gate"] = np.sqrt(2 / 3) * np.log(
        (out["n_E"] + .5) / np.sqrt((out["n_A"] + .5) * (out["n_C"] + .5)))
    out.loc[~out["reference_quality_ok"], ["z_C_over_A", "z_E_gate"]] = np.nan
    out = add_cohort(out)
    out.to_csv(data_dir / "ace_processing_balances_event.csv", index=False)

    group = ["participant", "stratum", "cohort", "semiology", "scheme", "window_s", "step_s",
             "window_end_to_ictus_s", "minutes_to_ictus"]
    weighted = out.groupby(group, observed=True, dropna=False)[["z_C_over_A", "z_E_gate"]].mean().reset_index()
    weighted["n_events"] = out.groupby(group, observed=True, dropna=False).size().to_numpy()
    weighted = piecewise_columns(weighted)
    weighted.to_csv(data_dir / "ace_processing_balances_participant.csv", index=False)
    func = weighted[weighted["stratum"].isin(FUNCTIONAL) & weighted["semiology"].notna()].copy()
    base_formula = "z_C_over_A ~ early_elapsed_min + late_elapsed_min + C(semiology, Treatment('hyper')) + C(cohort, Treatment('FS')) + early_elapsed_min:C(semiology, Treatment('hyper')) + late_elapsed_min:C(semiology, Treatment('hyper'))"
    ca_fit, ca_terms = fit_gee(func, base_formula, "pooled functional C/A balance")
    ca_primary = linear_contrast(ca_fit, {"late_elapsed_min": 1}, "pooled functional late C/A slope > 0", "greater")
    ca_interaction = linear_contrast(ca_fit, {"late_elapsed_min:C(semiology, Treatment('hyper'))[T.hypo]": 1},
                                             "C/A late slope: hypomotor minus hypermotor")

    func["z_E_gate_within"] = func["z_E_gate"] - func.groupby("participant")["z_E_gate"].transform("mean")
    gate_formula = "z_C_over_A ~ early_elapsed_min + late_elapsed_min + z_E_gate_within + late_elapsed_min:z_E_gate_within + C(semiology, Treatment('hyper')) + C(cohort, Treatment('FS'))"
    gate_fit, gate_terms = fit_gee(func, gate_formula, "E gating moderation of C/A")
    gate_contrast = linear_contrast(gate_fit, {"late_elapsed_min:z_E_gate_within": 1}, "E-by-late-time moderation")

    secondary_terms = []
    secondary = participant[(participant["backfit_method"].eq("hard")) & participant["stratum"].isin(FUNCTIONAL)].copy()
    secondary["cohort"] = np.where(secondary["stratum"].eq("FS functional"), "FS", "DD")
    secondary = piecewise_columns(secondary)
    for state in FOCAL:
        state_data = secondary[secondary["template_label"].eq(state)].copy()
        for outcome in ("coverage", "occurrence_rate_hz", "mean_dwell_s", "mean_gfp", "mean_topographic_gmd"):
            try:
                fitted, terms = fit_gee(state_data, f"{outcome} ~ early_elapsed_min + late_elapsed_min + C(cohort, Treatment('FS'))",
                                        f"secondary {state} {outcome}")
                row = linear_contrast(fitted, {"late_elapsed_min": 1}, f"{state} {outcome} late slope")
                row.update({"state": state, "outcome": outcome})
                secondary_terms.append(row)
            except Exception as exc:
                secondary_terms.append({"state": state, "outcome": outcome, "contrast": "late slope", "error": str(exc), "p": np.nan})

    model_table = pd.concat([ca_terms, gate_terms], ignore_index=True)
    model_table.to_csv(data_dir / "ace_activity_gee_terms.csv", index=False)
    contrasts = pd.DataFrame([ca_primary, ca_interaction, gate_contrast])
    contrasts.to_csv(data_dir / "ace_activity_planned_contrasts.csv", index=False)
    secondary_table = pd.DataFrame(secondary_terms)
    secondary_table["p_bh"] = bh(secondary_table["p"])
    secondary_table.to_csv(data_dir / "ace_activity_secondary_bh.csv", index=False)
    plot_trajectories(func, ["z_C_over_A", "z_E_gate"], ["C/A processing balance", "E expression balance"],
                      figure_dir / "ace_processing_balance_trajectories.png")
    return out, weighted, ca_primary, contrasts, secondary_table


def parse_sequence(value) -> list[int]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [int(x) for x in str(value).split(",") if x.strip()]


def collapse(sequence: list[int]) -> list[int]:
    return [x for i, x in enumerate(sequence) if i == 0 or x != sequence[i - 1]]


def transition_counts(sequence: list[int]) -> np.ndarray:
    counts = np.zeros((6, 6), dtype=float)
    for a, b in zip(sequence[:-1], sequence[1:]):
        if 1 <= a <= 6 and 1 <= b <= 6 and a != b:
            counts[a - 1, b - 1] += 1
    return counts


def fit_kappa(counts: np.ndarray, b0: np.ndarray, origins: tuple[int, ...], min_transitions: int) -> tuple[float, int]:
    n = int(sum(counts[i].sum() for i in origins))
    if n < min_transitions:
        return np.nan, n
    def objective(log_kappa: float) -> float:
        kappa = math.exp(log_kappa)
        value = 0.0
        for i in origins:
            allowed = np.arange(6) != i
            base = np.clip(b0[i, allowed], 1e-12, 1)
            q = base ** kappa
            q /= q.sum()
            value -= float(counts[i, allowed] @ np.log(np.clip(q, 1e-12, 1)))
        return value
    fit = minimize_scalar(objective, bounds=(-4, 4), method="bounded")
    return (float(fit.x) if fit.success else np.nan), n


def js_divergence(counts: np.ndarray, b0: np.ndarray, origins: tuple[int, ...]) -> float:
    total, weighted = 0.0, 0.0
    for i in origins:
        allowed = np.arange(6) != i
        n = counts[i, allowed].sum()
        if n <= 0:
            continue
        p = (counts[i, allowed] + .5) / (counts[i, allowed].sum() + .5 * allowed.sum())
        q = b0[i, allowed]
        q = q / q.sum()
        m = (p + q) / 2
        js = .5 * np.sum(p * np.log(p / m)) + .5 * np.sum(q * np.log(q / m))
        weighted += n * js
        total += n
    return weighted / total if total else np.nan


def gating_metrics(sequence: list[int]) -> dict:
    a, e, c = LABELS.index("A") + 1, LABELS.index("E") + 1, LABELS.index("C") + 1
    aec = sum(sequence[i:i + 3] == [a, e, c] for i in range(max(0, len(sequence) - 2)))
    cea = sum(sequence[i:i + 3] == [c, e, a] for i in range(max(0, len(sequence) - 2)))
    direct = sum({x, y} == {a, c} for x, y in zip(sequence[:-1], sequence[1:]))
    via = aec + cea
    denom = direct + via
    return {"A_E_C": aec, "C_E_A": cea,
            "G_direction": math.log((aec + .5) / (cea + .5)),
            "E_routed_switches": via, "direct_AC_switches": direct,
            "E_routed_fraction": via / denom if denom else np.nan, "qualifying_switches": denom}


def interictal_b0(transition_csv: Path) -> dict[str, np.ndarray]:
    use = ["participant", "condition", "backfit_method", "state_from_label", "state_to_label", "transition_count"]
    table = pd.read_csv(transition_csv, usecols=use)
    table = table[table["condition"].astype(str).str.upper().eq("II") & table["backfit_method"].eq("hard")]
    output = {}
    for participant, frame in table.groupby("participant", observed=True):
        counts = np.zeros((6, 6), dtype=float)
        for row in frame.itertuples(index=False):
            if row.state_from_label in LABELS and row.state_to_label in LABELS:
                counts[LABELS.index(row.state_from_label), LABELS.index(row.state_to_label)] += float(row.transition_count)
        probs = counts + .5
        np.fill_diagonal(probs, 0)
        probs /= probs.sum(axis=1, keepdims=True)
        output[str(participant)] = probs
    return output


def transition_analysis(results_dir: Path, manifest: pd.DataFrame, data_dir: Path, figure_dir: Path):
    cols = ["participant", "file_path", "backfit_method", "template_label", "bin_from_ictus",
            "minutes_to_ictus", "collapsed_sequence"]
    table = pd.read_csv(results_dir / "participant_condition_preictal_time_metrics.csv", usecols=cols)
    table = table[table["backfit_method"].eq("hard") & table["template_label"].eq("A")].copy()
    meta = manifest[["file_key", "stratum", "hypo_hyper"]].copy()
    table["file_key"] = table["file_path"].astype(str).str.replace("/", "\\", regex=False).str.lower()
    table = table.merge(meta, on="file_key", how="left", validate="many_to_one")
    b0s = interictal_b0(results_dir / "participant_condition_transition_metrics.csv")
    focal_origins = tuple(LABELS.index(x) for x in FOCAL)
    e_origin = (LABELS.index("E"),)
    rows, period_rows = [], []
    for (participant, file_path), frame in table.groupby(["participant", "file_path"], observed=True, sort=False):
        frame = frame.sort_values("bin_from_ictus")
        if str(participant) not in b0s:
            continue
        b0 = b0s[str(participant)]
        info = frame.iloc[0]
        sequences = {int(r.bin_from_ictus): parse_sequence(r.collapsed_sequence) for r in frame.itertuples(index=False)}
        max_bin = max(sequences, default=0)
        for start in range(1, max_bin - 5 + 1):
            bins = list(range(start + 5, start - 1, -1))
            if not all(b in sequences for b in bins):
                continue
            seq = collapse([value for b in bins for value in sequences[b]])
            counts = transition_counts(seq)
            logk, n = fit_kappa(counts, b0, focal_origins, 30)
            logk_e, n_e = fit_kappa(counts, b0, e_origin, 20)
            rows.append({"participant": participant, "file_path": file_path, "stratum": info.stratum,
                         "hypo_hyper": info.hypo_hyper, "window_end_to_ictus_s": (start - 1) * 10,
                         "minutes_to_ictus": -(start - 1) / 6, "log_kappa_ACE": logk,
                         "kappa_ACE": math.exp(logk) if np.isfinite(logk) else np.nan,
                         "ACE_outgoing_transitions": n, "log_kappa_E": logk_e,
                         "kappa_E": math.exp(logk_e) if np.isfinite(logk_e) else np.nan,
                         "E_outgoing_transitions": n_e, "weighted_JS": js_divergence(counts, b0, focal_origins)})
        for period, lo, hi in (("early", 31, 120), ("late", 1, 30)):
            bins = [b for b in range(hi, lo - 1, -1) if b in sequences]
            seq = collapse([value for b in bins for value in sequences[b]])
            metrics = gating_metrics(seq)
            period_rows.append({"participant": participant, "file_path": file_path, "stratum": info.stratum,
                                "hypo_hyper": info.hypo_hyper, "period": period, **metrics})
    windows = add_cohort(pd.DataFrame(rows))
    periods = add_cohort(pd.DataFrame(period_rows))
    windows.to_csv(data_dir / "b_like_precision_event_windows.csv", index=False)
    periods.to_csv(data_dir / "e_gating_event_periods.csv", index=False)
    group = ["participant", "stratum", "cohort", "semiology", "window_end_to_ictus_s", "minutes_to_ictus"]
    participant = windows.groupby(group, observed=True, dropna=False)[["log_kappa_ACE", "log_kappa_E", "weighted_JS"]].mean().reset_index()
    participant = piecewise_columns(participant)
    participant.to_csv(data_dir / "b_like_precision_participant.csv", index=False)
    func = participant[participant["stratum"].isin(FUNCTIONAL) & participant["semiology"].notna()].copy()
    formula_tail = "early_elapsed_min + late_elapsed_min + C(semiology, Treatment('hyper')) + C(cohort, Treatment('FS')) + late_elapsed_min:C(semiology, Treatment('hyper'))"
    precision_fit, precision_terms = fit_gee(func, "log_kappa_ACE ~ " + formula_tail, "pooled B-like ACE transition precision")
    precision_primary = linear_contrast(precision_fit, {"late_elapsed_min": 1}, "pooled B-like precision late trend")
    e_fit, e_terms = fit_gee(func, "log_kappa_E ~ " + formula_tail, "E-origin transition precision")
    js_fit, js_terms = fit_gee(func, "weighted_JS ~ " + formula_tail, "transition JS divergence")
    pd.concat([precision_terms, e_terms, js_terms], ignore_index=True).to_csv(data_dir / "b_like_precision_gee_terms.csv", index=False)
    transition_contrasts = pd.DataFrame([
        precision_primary,
        linear_contrast(precision_fit, {"late_elapsed_min:C(semiology, Treatment('hyper'))[T.hypo]": 1}, "B-like precision late hypomotor interaction"),
        linear_contrast(e_fit, {"late_elapsed_min": 1}, "E-origin precision late trend"),
        linear_contrast(js_fit, {"late_elapsed_min": 1}, "weighted JS late trend"),
    ])
    transition_contrasts.to_csv(data_dir / "b_like_precision_planned_contrasts.csv", index=False)

    pgroup = ["participant", "stratum", "cohort", "semiology", "period"]
    pweighted = periods.groupby(pgroup, observed=True, dropna=False)[["G_direction", "E_routed_fraction"]].mean().reset_index()
    pweighted["late_period"] = pweighted["period"].eq("late").astype(int)
    pfunc = pweighted[pweighted["stratum"].isin(FUNCTIONAL) & pweighted["semiology"].notna()].copy()
    gating_terms, gating_contrasts = [], []
    for outcome in ("G_direction", "E_routed_fraction"):
        fitted, terms = fit_gee(pfunc, f"{outcome} ~ late_period + C(semiology, Treatment('hyper')) + C(cohort, Treatment('FS')) + late_period:C(semiology, Treatment('hyper'))", f"sequence gating {outcome}")
        gating_terms.append(terms)
        gating_contrasts.append(linear_contrast(fitted, {"late_period": 1}, f"late versus early {outcome}"))
    pd.concat(gating_terms, ignore_index=True).to_csv(data_dir / "e_gating_gee_terms.csv", index=False)
    pd.DataFrame(gating_contrasts).to_csv(data_dir / "e_gating_planned_contrasts.csv", index=False)
    plot_trajectories(func, ["log_kappa_ACE", "log_kappa_E", "weighted_JS"],
                      ["A/C/E log transition precision", "E-origin log precision", "Weighted JS divergence"],
                      figure_dir / "b_like_transition_precision.png")
    return windows, participant, precision_primary, transition_contrasts, pd.DataFrame(gating_contrasts)


def rpeak_file(ecg_qc_dir: Path, source_raw_fif: str) -> Path:
    base = Path(str(source_raw_fif)).name
    for suffix in ("_pp_raw_keepcfa.fif", "_pp_raw.fif", ".fif"):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    return ecg_qc_dir / f"{base}_rpeaks_corrected.csv"


def read_rpeaks(path: Path) -> np.ndarray:
    if not path.exists():
        return np.array([], dtype=float)
    data = pd.read_csv(path)
    return np.sort(pd.to_numeric(data.get("rpeak_time_s"), errors="coerce").dropna().to_numpy(float))


def read_rpeaks_with_recovery_flags(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.array([], dtype=float), np.array([], dtype=bool)
    data = pd.read_csv(path)
    times = pd.to_numeric(data.get("rpeak_time_s"), errors="coerce")
    keep = times.notna()
    data = data.loc[keep].copy()
    data["rpeak_time_s"] = times[keep].to_numpy(float)
    data = data.sort_values("rpeak_time_s")
    if "is_gap_recovered" in data:
        recovered = pd.to_numeric(data["is_gap_recovered"], errors="coerce").fillna(0).astype(bool).to_numpy()
    else:
        recovered = np.zeros(len(data), dtype=bool)
    return data["rpeak_time_s"].to_numpy(float), recovered


def recovered_peak_flags(final: np.ndarray, original: np.ndarray, tolerance_s: float = .05) -> np.ndarray:
    if not len(original):
        return np.ones(len(final), dtype=bool)
    return np.array([np.min(np.abs(original - peak)) > tolerance_s for peak in final], dtype=bool)


def baseline_correct_hr(windows: pd.DataFrame) -> pd.DataFrame:
    keys = ["participant", "file_path"]
    baseline = (windows[windows["minutes_to_ictus"].between(-20, -5, inclusive="both") & windows["mean_hr_bpm"].notna()]
                .groupby(keys, observed=True)["mean_hr_bpm"]
                .agg(baseline_hr_bpm="mean", baseline_hr_valid_windows="size").reset_index())
    out = windows.merge(baseline, on=keys, how="left", validate="many_to_one")
    out["baseline_hr_expected_windows"] = int(windows.loc[
        windows["minutes_to_ictus"].between(-20, -5, inclusive="both"), "window_end_to_ictus_s"].nunique())
    out["baseline_hr_valid_fraction"] = out["baseline_hr_valid_windows"] / out["baseline_hr_expected_windows"]
    out["delta_hr_bpm"] = out["mean_hr_bpm"] - out["baseline_hr_bpm"]
    return out


def cardiac_windows(manifest: pd.DataFrame, ecg_qc_dir: Path, original_ecg_qc_dir: Path,
                    strata: tuple[str, ...] = FUNCTIONAL, window_length_s: int = 60,
                    step_s: int = 60) -> pd.DataFrame:
    rows = []
    events = manifest[manifest["condition"].astype(str).str.lower().eq("preict") & manifest["stratum"].isin(strata)]
    for event in events.itertuples(index=False):
        final, recovered = read_rpeaks_with_recovery_flags(rpeak_file(ecg_qc_dir, event.source_raw_fif))
        start, stop = float(event.crop_start_seconds), float(event.crop_end_seconds)
        onset = stop - start
        keep = (final >= start - 6) & (final <= stop + 1)
        peaks = final[keep] - start
        flags = recovered[keep]
        if len(peaks) < 3:
            continue
        rr = np.diff(peaks)
        mids = (peaks[:-1] + peaks[1:]) / 2
        interval_recovered = flags[:-1] | flags[1:]
        for end_to in range(0, 1200 - window_length_s + 1, step_s):
            lo, hi = onset - end_to - window_length_s, onset - end_to
            overlap = np.maximum(0, np.minimum(peaks[1:], hi) - np.maximum(peaks[:-1], lo))
            coverage = overlap.sum() / window_length_s
            included = overlap > 0
            gaps = rr[included]
            hr = 60 / rr[included]
            recovered_fraction = interval_recovered[included].mean() if included.any() else np.nan
            valid = (coverage >= .80 and len(gaps) > 0 and np.max(gaps) <= 5 and
                     recovered_fraction <= .10 and np.all((hr >= 35) & (hr <= 140)))
            rows.append({"participant": event.participant, "file_path": event.file_path, "stratum": event.stratum,
                         "hypo_hyper": event.hypo_hyper, "source_raw_fif": event.source_raw_fif,
                         "rpeak_sidecar": str(rpeak_file(ecg_qc_dir, event.source_raw_fif)),
                         "window_end_to_ictus_s": end_to, "minutes_to_ictus": -end_to / 60,
                         "window_length_s": window_length_s, "step_s": step_s,
                         "mean_hr_bpm_raw": float(np.average(hr, weights=overlap[included])) if included.any() else np.nan,
                         "rr_coverage": coverage, "max_rr_gap_s": float(np.max(gaps)) if len(gaps) else np.nan,
                         "recovered_interval_fraction": recovered_fraction, "n_rr_intervals": int(included.sum()),
                         "hr_window_valid": bool(valid)})
    out = add_cohort(pd.DataFrame(rows))
    out["mean_hr_bpm"] = out["mean_hr_bpm_raw"].where(out["hr_window_valid"])
    return baseline_correct_hr(out)


def cardiac_analysis(manifest: pd.DataFrame, activity_event: pd.DataFrame, transition_event: pd.DataFrame,
                     ecg_qc_dir: Path, original_ecg_qc_dir: Path, data_dir: Path, figure_dir: Path):
    event = cardiac_windows(manifest, ecg_qc_dir, original_ecg_qc_dir)
    event.to_csv(data_dir / "cardiac_rolling_windows_qc.csv", index=False)
    qc = event.groupby(["stratum", "semiology"], observed=True, dropna=False).agg(
        windows=("hr_window_valid", "size"), valid_windows=("hr_window_valid", "sum"),
        events=("file_path", "nunique"), participants=("participant", "nunique"),
        median_rr_coverage=("rr_coverage", "median"), median_recovered_fraction=("recovered_interval_fraction", "median")).reset_index()
    qc.to_csv(data_dir / "cardiac_window_qc_audit.csv", index=False)
    group = ["participant", "stratum", "cohort", "semiology", "window_end_to_ictus_s", "minutes_to_ictus"]
    participant = event.groupby(group, observed=True, dropna=False)["delta_hr_bpm"].mean().reset_index()
    participant = piecewise_columns(participant)
    participant.to_csv(data_dir / "cardiac_participant_trajectories.csv", index=False)
    func = participant[participant["stratum"].isin(FUNCTIONAL) & participant["semiology"].notna()].copy()
    formula = "delta_hr_bpm ~ early_elapsed_min + late_elapsed_min + C(semiology, Treatment('hyper')) + C(cohort, Treatment('FS')) + early_elapsed_min:C(semiology, Treatment('hyper')) + late_elapsed_min:C(semiology, Treatment('hyper'))"
    fitted, terms = fit_gee(func, formula, "pooled functional delta HR")
    hyper = linear_contrast(fitted, {"late_elapsed_min": 1}, "hypermotor late HR slope")
    hypo = linear_contrast(fitted, {"late_elapsed_min": 1, "late_elapsed_min:C(semiology, Treatment('hyper'))[T.hypo]": 1}, "hypomotor late HR slope < 0", "less")
    interaction = linear_contrast(fitted, {"late_elapsed_min:C(semiology, Treatment('hyper'))[T.hypo]": 1}, "hypomotor minus hypermotor late HR slope")
    lower, upper = -1.0, 1.0
    hyper["tost_lower_p"] = float(norm.sf((hyper["estimate"] - lower) / hyper["se"]))
    hyper["tost_upper_p"] = float(norm.sf((upper - hyper["estimate"]) / hyper["se"]))
    hyper["tost_p"] = max(hyper["tost_lower_p"], hyper["tost_upper_p"])
    hyper["equivalent_within_plus_minus_1_bpm_per_min"] = hyper["tost_p"] < .05
    terms.to_csv(data_dir / "cardiac_gee_terms.csv", index=False)
    cardiac_contrasts = pd.DataFrame([hypo, interaction, hyper])
    cardiac_contrasts.to_csv(data_dir / "cardiac_planned_contrasts.csv", index=False)
    plot_trajectories(func, ["delta_hr_bpm"], ["Heart-rate change from mean -20 to -5 min baseline (bpm)"], figure_dir / "cardiac_trajectories.png")

    hr = event[event["hr_window_valid"]][["participant", "file_path", "stratum", "semiology", "cohort", "window_end_to_ictus_s", "minutes_to_ictus", "delta_hr_bpm"]]
    ace = activity_event[activity_event["window_end_to_ictus_s"].mod(60).eq(0)][["participant", "file_path", "window_end_to_ictus_s", "z_C_over_A", "z_E_gate"]]
    precision = transition_event[transition_event["window_end_to_ictus_s"].mod(60).eq(0)][["participant", "file_path", "window_end_to_ictus_s", "log_kappa_ACE", "log_kappa_E", "weighted_JS"]]
    merged = hr.merge(ace, on=["participant", "file_path", "window_end_to_ictus_s"], how="inner", validate="one_to_one")
    merged = merged.merge(precision, on=["participant", "file_path", "window_end_to_ictus_s"], how="left", validate="one_to_one")
    merged["hr_within_event"] = merged["delta_hr_bpm"] - merged.groupby(["participant", "file_path"])["delta_hr_bpm"].transform("mean")
    merged.to_csv(data_dir / "exact_cardioneural_windows.csv", index=False)
    mgroup = ["participant", "stratum", "semiology", "cohort", "window_end_to_ictus_s", "minutes_to_ictus"]
    mp = merged.groupby(mgroup, observed=True, dropna=False)[["z_C_over_A", "z_E_gate", "log_kappa_ACE", "log_kappa_E", "weighted_JS", "hr_within_event"]].mean().reset_index()
    mp = piecewise_columns(mp)
    cardioneural_terms, cardioneural_contrasts = [], []
    for outcome in ("z_C_over_A", "z_E_gate", "log_kappa_ACE", "log_kappa_E", "weighted_JS"):
        try:
            fit, terms2 = fit_gee(mp, f"{outcome} ~ early_elapsed_min + late_elapsed_min + hr_within_event + hr_within_event:C(semiology, Treatment('hyper')) + C(semiology, Treatment('hyper')) + C(cohort, Treatment('FS'))", f"cardioneural {outcome}")
            cardioneural_terms.append(terms2)
            row = linear_contrast(fit, {"hr_within_event": 1, "hr_within_event:C(semiology, Treatment('hyper'))[T.hypo]": 1}, f"hypomotor HR association with {outcome}")
            row["outcome"] = outcome
            cardioneural_contrasts.append(row)
        except Exception as exc:
            cardioneural_contrasts.append({"outcome": outcome, "contrast": "hypomotor HR association", "error": str(exc), "p": np.nan})
    if cardioneural_terms:
        pd.concat(cardioneural_terms, ignore_index=True).to_csv(data_dir / "cardioneural_gee_terms.csv", index=False)
    cc = pd.DataFrame(cardioneural_contrasts)
    cc["p_bh"] = bh(cc["p"])
    cc.to_csv(data_dir / "cardioneural_planned_contrasts_bh.csv", index=False)
    return event, participant, hypo, cardiac_contrasts, cc


def heartbeat_phase_analysis(results_dir: Path, manifest: pd.DataFrame, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    covariance = pd.read_csv(
        results_dir / "hierarchical_state_covariance_metrics.csv",
        usecols=["level", "participant", "template_label", "template_match_abs_correlation"],
    )
    quality = (covariance[covariance["level"].eq("participant") & covariance["template_label"].isin(FOCAL)]
               .groupby(["participant", "template_label"], observed=True)["template_match_abs_correlation"]
               .mean().reset_index())
    cardiac_csv = results_dir / "participant_ace_cardiac_phase_metrics.csv"
    metrics = pd.concat([
        build_cardiac(cardiac_csv, manifest, quality, 180, 500),
        build_cardiac(cardiac_csv, manifest, quality, 200, 400),
    ], ignore_index=True)
    models = fit_cardiac_models(metrics)
    metrics.to_csv(data_dir / "ace_heartbeat_phase_rolling_metrics.csv", index=False)
    models.to_csv(data_dir / "ace_heartbeat_phase_poisson_gee.csv", index=False)
    return metrics, models


def plot_trajectories(frame: pd.DataFrame, outcomes: list[str], titles: list[str], output: Path):
    fig, axes = plt.subplots(1, len(outcomes), figsize=(5.5 * len(outcomes), 4.5), squeeze=False)
    for ax, outcome, title in zip(axes[0], outcomes, titles):
        for semio, color in (("hypo", "#2c7fb8"), ("hyper", "#d95f0e")):
            part = frame[frame["semiology"].eq(semio)]
            summary = part.groupby("minutes_to_ictus", observed=True)[outcome].agg(["mean", "sem"]).reset_index().sort_values("minutes_to_ictus")
            if summary.empty:
                continue
            ax.plot(summary["minutes_to_ictus"], summary["mean"], color=color, label=semio.capitalize(), lw=2)
            ax.fill_between(summary["minutes_to_ictus"], summary["mean"] - summary["sem"], summary["mean"] + summary["sem"], color=color, alpha=.18)
        ax.axvline(-5, color="0.35", ls="--", lw=1)
        ax.axvline(0, color="black", lw=1)
        ax.set(title=title, xlabel="Minutes to seizure onset", ylabel=outcome)
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def self_check() -> None:
    seq = collapse([5, 5, 2, 6, 6, 5, 6])
    assert seq == [5, 2, 6, 5, 6]
    gate = gating_metrics(seq)
    assert gate["A_E_C"] == 1 and gate["direct_AC_switches"] == 2
    final = np.array([1., 2., 3.])
    assert recovered_peak_flags(final, np.array([1.01, 3.0])).tolist() == [False, True, False]
    counts = np.ones((6, 6)); np.fill_diagonal(counts, 0)
    b0 = counts / counts.sum(axis=1, keepdims=True)
    logk, n = fit_kappa(counts, b0, (4, 5, 1), 1)
    assert np.isfinite(logk) and n == 15
    p = pd.Series([.01, .04, .03])
    assert np.all(holm(p).between(0, 1)) and np.all(bh(p).between(0, 1))
    hr = pd.DataFrame({"participant": ["p"] * 4, "file_path": ["f"] * 4,
                       "minutes_to_ictus": [-19., -10., -5., 0.],
                       "window_end_to_ictus_s": [1140, 600, 300, 0],
                       "mean_hr_bpm": [60., 70., 80., 90.]})
    corrected = baseline_correct_hr(hr)
    assert corrected["baseline_hr_bpm"].eq(70).all() and corrected.loc[3, "delta_hr_bpm"] == 20


def main() -> None:
    global LABELS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--ecg-qc-dir", type=Path)
    parser.add_argument("--original-ecg-qc-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(); print("self-check passed"); return
    if not all((args.results_dir, args.manifest, args.ecg_qc_dir, args.original_ecg_qc_dir, args.out_dir)):
        parser.error("all path arguments are required")
    LABELS = load_global_labels(args.results_dir)
    data_dir, figure_dir = args.out_dir / "data", args.out_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True); figure_dir.mkdir(parents=True, exist_ok=True)
    manifest = attach_strata(pd.read_csv(args.manifest))
    if not manifest["file_key"].is_unique:
        raise ValueError("Manifest file paths are not unique")
    activity_event, activity_participant, ca_primary, activity_contrasts, secondary = activity_analysis(args.results_dir, manifest, data_dir, figure_dir)
    transition_event, transition_participant, precision_primary, transition_contrasts, gating = transition_analysis(args.results_dir, manifest, data_dir, figure_dir)
    cardiac_event, cardiac_participant, hr_primary, cardiac_contrasts, cardioneural = cardiac_analysis(
        manifest, activity_event, transition_event, args.ecg_qc_dir, args.original_ecg_qc_dir, data_dir, figure_dir)
    heartbeat_metrics, heartbeat_models = heartbeat_phase_analysis(args.results_dir, manifest, data_dir)
    primary = pd.DataFrame([ca_primary | {"family": "C/A late slope"},
                            hr_primary | {"family": "hypomotor HR late slope"},
                            precision_primary | {"family": "B-like precision trend"}])
    primary["p_holm"] = holm(primary["p"])
    primary.to_csv(data_dir / "primary_holm_corrected_tests.csv", index=False)
    summary = {"labels": dict(enumerate(LABELS, 1)), "primary_tests": primary.to_dict("records"),
               "n_activity_participants": int(activity_participant.participant.nunique()),
               "n_transition_participants": int(transition_participant.participant.nunique()),
               "n_cardiac_participants": int(cardiac_participant.participant.nunique()),
               "n_heartbeat_phase_participants": int(heartbeat_metrics.participant.nunique()),
               "ecg_qc_dir": str(args.ecg_qc_dir)}
    (data_dir / "ace_cardioneural_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    dictionary = pd.DataFrame([
        ("z_C_over_A", "ilr coordinate", "sqrt(1/2) log((n_C+0.5)/(n_A+0.5)); n is assigned samples in a 60-s window."),
        ("z_E_gate", "ilr coordinate", "sqrt(2/3) log((n_E+0.5)/sqrt((n_A+0.5)(n_C+0.5)))."),
        ("log_kappa_ACE", "log precision", "Log maximum-likelihood exponent scaling participant interictal A/C/E-origin transition probabilities."),
        ("weighted_JS", "nats", "Origin-transition-count weighted Jensen-Shannon divergence from the participant interictal operator."),
        ("delta_hr_bpm", "beats/min", "Window HR minus the event mean across valid -20 to -5 minute HR windows."),
        ("hr_window_valid", "boolean", "RR coverage >=80%, no gap >5 s, <=10% recovered intervals, and all instantaneous HR 35-140 bpm."),
        ("spm_prior_weight", "dimensionless", "VB component-mean prior weight; locked to 1 and never added to the observations."),
    ], columns=["variable", "unit", "definition"])
    dictionary.to_csv(data_dir / "ace_cardioneural_data_dictionary.csv", index=False)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
