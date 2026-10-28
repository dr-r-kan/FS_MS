"""Raw-count and specificity analyses requested after the focused FS/ES analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = ("F", "E", "G", "B", "A", "C")
FUNCTIONAL = ("FS functional", "DD functional")
EPILEPTIC = ("ES epileptic", "DD epileptic")
FIXED_EXITS = 30
TRAJECTORY_BINS = ((600, 1200, "-20 to -10"), (300, 600, "-10 to -5"),
                   (120, 300, "-5 to -2"), (60, 120, "-2 to -1"), (0, 60, "-1 to 0"))


def gamma_dirichlet(rng: np.random.Generator, alpha: np.ndarray) -> np.ndarray:
    value = rng.gamma(alpha, 1.0)
    return value / value.sum(axis=1, keepdims=True)


def thin_counts(rng: np.random.Generator, counts: np.ndarray, total: int, draws: int) -> np.ndarray:
    if counts.sum() < total:
        raise ValueError("cannot thin fewer transitions than requested")
    return rng.multivariate_hypergeometric(counts.astype(int), total, size=draws)


def softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - value.max(axis=1, keepdims=True)
    result = np.exp(shifted)
    return result / result.sum(axis=1, keepdims=True)


def transition_metrics(p: np.ndarray, q: np.ndarray) -> dict[str, np.ndarray]:
    midpoint = (p + q) / 2
    contribution = .5 * p * np.log(p / midpoint) + .5 * q * np.log(q / midpoint)
    log_q = np.log(q)
    target = np.sum(p * log_q, axis=1)
    low, high = np.full(len(p), math.exp(-4)), np.full(len(p), math.exp(4))
    for _ in range(24):
        kappa = (low + high) / 2
        expectation = np.sum(softmax(kappa[:, None] * log_q) * log_q, axis=1)
        low = np.where(expectation < target, kappa, low)
        high = np.where(expectation < target, high, kappa)
    log_p_centred = np.log(p) - np.log(p).mean(axis=1, keepdims=True)
    log_q_centred = log_q - log_q.mean(axis=1, keepdims=True)
    rho_p = np.linalg.norm(log_p_centred, axis=1)
    rho_q = np.linalg.norm(log_q_centred, axis=1)
    v_p = log_p_centred / np.maximum(rho_p[:, None], 1e-12)
    v_q = log_q_centred / np.maximum(rho_q[:, None], 1e-12)
    route = np.arccos(np.clip(np.sum(v_p * v_q, axis=1), -1, 1)) / np.pi
    shared_rho = ((rho_p + rho_q) / 2)[:, None]
    p_matched, q_matched = softmax(shared_rho * v_p), softmax(shared_rho * v_q)
    matched_midpoint = (p_matched + q_matched) / 2
    matched_js = (.5 * np.sum(p_matched * np.log(p_matched / matched_midpoint), axis=1) +
                  .5 * np.sum(q_matched * np.log(q_matched / matched_midpoint), axis=1))
    return {"js": contribution.sum(axis=1), "log_kappa": np.log((low + high) / 2),
            "delta_rho": rho_p - rho_q, "route_distance": route, "matched_js": matched_js,
            "delta_probability": p - q, "js_contribution": contribution}


def count_columns(source: int) -> tuple[list[str], list[int]]:
    destinations = [index for index in range(6) if index != source]
    return [f"t_{source + 1}_{index + 1}" for index in destinations], destinations


def baseline_counts(table: pd.DataFrame, source: int) -> dict[str, np.ndarray]:
    columns, _ = count_columns(source)
    return {str(participant): part[columns].sum().to_numpy(int)
            for participant, part in table[table.condition.eq("ii")].groupby("participant", observed=True)}


def draw_metrics(table: pd.DataFrame, source: int, draws: int, rng: np.random.Generator,
                 fixed_exits: int | None = None, baseline_table: pd.DataFrame | None = None
                 ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    columns, _ = count_columns(source)
    baselines = baseline_counts(table if baseline_table is None else baseline_table, source)
    eligible = table.participant.astype(str).isin(baselines)
    if fixed_exits is not None:
        eligible &= table[columns].sum(axis=1).ge(fixed_exits)
        eligible &= table.participant.astype(str).map(lambda value: baselines.get(value, np.zeros(1)).sum() >= fixed_exits)
    frame = table[eligible].reset_index(drop=True).copy()
    arrays: dict[str, list[np.ndarray]] = {name: [] for name in
        ("js", "log_kappa", "delta_rho", "route_distance", "matched_js", "delta_probability", "js_contribution")}
    q_cache: dict[str, np.ndarray] = {}
    for row in frame.itertuples(index=False):
        participant = str(row.participant)
        if participant not in q_cache:
            base = baselines[participant]
            alpha = thin_counts(rng, base, fixed_exits, draws) + .5 if fixed_exits else base[None, :] + .5
            q_cache[participant] = gamma_dirichlet(rng, np.broadcast_to(alpha, (draws, len(base))))
        counts = np.asarray([getattr(row, column) for column in columns], int)
        alpha = thin_counts(rng, counts, fixed_exits, draws) + .5 if fixed_exits else counts[None, :] + .5
        p = gamma_dirichlet(rng, np.broadcast_to(alpha, (draws, len(counts))))
        metrics = transition_metrics(p, q_cache[participant])
        for name, value in metrics.items():
            arrays[name].append(value)
    return frame, {name: np.stack(value) for name, value in arrays.items()}


def participant_design(frame: pd.DataFrame, values: np.ndarray, seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    meta = frame[["participant", "is_functional", "is_hypo", "is_dd"]].copy()
    codes, unique = pd.factorize(pd.MultiIndex.from_frame(meta), sort=False)
    rng = np.random.default_rng(seed)
    aggregated = []
    for index in range(len(unique)):
        event_values = values[codes == index]
        weights = rng.exponential(size=event_values.shape[:2])
        aggregated.append(np.sum(weights[:, :, None] * event_values, axis=0) / weights.sum(axis=0)[:, None])
    aggregated = np.stack(aggregated)
    rows = pd.DataFrame(unique.tolist(), columns=meta.columns)
    rows["is_hypo_centered"] = rows.is_hypo - rows.is_hypo.mean()
    rows["is_dd_centered"] = rows.is_dd - rows.is_dd.mean()
    x = np.column_stack([np.ones(len(rows)), rows.is_functional, rows.is_hypo_centered,
                         rows.is_functional * rows.is_hypo_centered, rows.is_dd_centered])
    return rows, x, aggregated


def bootstrap_coefficients(frame: pd.DataFrame, values: np.ndarray, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    if values.ndim == 2:
        values = values[:, :, None]
    rows, x, y = participant_design(frame, values, seed + 7919)
    participant_codes = pd.factorize(rows.participant, sort=True)[0]
    rng = np.random.default_rng(seed)
    result = np.empty((y.shape[1], x.shape[1], y.shape[2]))
    for draw in range(y.shape[1]):
        participant_weights = rng.exponential(size=participant_codes.max() + 1)
        weights = participant_weights[participant_codes]
        precision = x.T @ (weights[:, None] * x) + np.eye(x.shape[1]) * 1e-8
        result[draw] = np.linalg.solve(precision, x.T @ (weights[:, None] * y[:, draw]))
    return result, rows


def summarize(values: np.ndarray, analysis: str, contrast: str, *, n_rows: int,
              n_participants: int, **extra) -> dict:
    low, median, high = np.quantile(values, [.025, .5, .975])
    return {"analysis": analysis, "contrast": contrast, "median": median, "ci95_low": low,
            "ci95_high": high, "probability_gt_zero": float(np.mean(values > 0)),
            "probability_lt_zero": float(np.mean(values < 0)), "n_rows": n_rows,
            "n_participants": n_participants, **extra}


def final_preictal(table: pd.DataFrame) -> pd.DataFrame:
    return table[(table.condition.eq("preict")) & table.window_end_to_ictus_s.eq(0) &
                 table.event_type.isin(("functional", "epileptic")) & table.semiology.notna()].copy()


def reference_participants(qc: pd.DataFrame, threshold: float, state: str | None = None) -> set[str]:
    use = qc[qc.template_match_abs_correlation.ge(threshold)]
    if state is not None:
        return set(use.loc[use.template_label.eq(state), "participant"].astype(str))
    counts = use.groupby("participant", observed=True).template_label.nunique()
    return set(counts[counts.eq(6)].index.astype(str))


def raw_count_primary(table: pd.DataFrame, qc: pd.DataFrame, data_dir: Path,
                      draws: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    final = final_preictal(table)
    valid_c = reference_participants(qc, .50, "C")
    final = final[final.participant.astype(str).isin(valid_c)]
    rows, event_rows = [], []
    for fixed, label, offset in ((None, "fixed 60-second duration", 0),
                                 (FIXED_EXITS, f"fixed {FIXED_EXITS} C exits", 1)):
        frame, metrics = draw_metrics(final, 5, draws, np.random.default_rng(seed + offset), fixed, table)
        for outcome in ("js", "log_kappa"):
            values = np.log(metrics[outcome]) if outcome == "js" else metrics[outcome]
            coefficients, participants = bootstrap_coefficients(frame, values, seed + 100 + offset)
            for contrast, effect in (("FS minus ES", coefficients[:, 1, 0]),
                                     ("Hypomotor minus hypermotor within FS", coefficients[:, 2, 0] + coefficients[:, 3, 0])):
                rows.append(summarize(effect, "raw-count Dirichlet posterior with hierarchical event/participant Bayesian bootstrap",
                                      contrast, n_rows=len(frame), n_participants=participants.participant.nunique(),
                                      estimand=label, outcome="JS" if outcome == "js" else "kappa",
                                      scale="log ratio"))
        means = frame[["participant", "file_path", "event_key", "stratum", "semiology", "is_functional", "is_dd"]].copy()
        means["estimand"] = label
        for name in ("js", "log_kappa", "delta_rho", "route_distance", "matched_js"):
            means[name] = metrics[name].mean(axis=1)
        means["kappa"] = np.exp(means.log_kappa)
        event_rows.append(means)
    summary, events = pd.DataFrame(rows), pd.concat(event_rows, ignore_index=True)
    summary["ratio_median"] = np.exp(summary["median"])
    summary["ratio_ci95_low"] = np.exp(summary.ci95_low)
    summary["ratio_ci95_high"] = np.exp(summary.ci95_high)
    summary.to_csv(data_dir / "raw_count_c_origin_bayesian.csv", index=False)
    events.to_csv(data_dir / "raw_count_c_origin_event_means.csv", index=False)
    return summary, events


def route_and_destinations(table: pd.DataFrame, qc: pd.DataFrame, data_dir: Path,
                           draws: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = final_preictal(table)
    frame = frame[frame.participant.astype(str).isin(reference_participants(qc, .50, "C"))]
    frame, metrics = draw_metrics(frame, 5, draws, np.random.default_rng(seed), baseline_table=table)
    rows = []
    for outcome in ("delta_rho", "route_distance", "matched_js"):
        coefficients, participants = bootstrap_coefficients(frame, metrics[outcome], seed + 11)
        rows.append(summarize(coefficients[:, 1, 0], "route/concentration decomposition", "FS minus ES",
                              n_rows=len(frame), n_participants=participants.participant.nunique(), outcome=outcome))
    _, destinations = count_columns(5)
    for metric_name, output_name in (("delta_probability", "delta_probability"),
                                     ("js_contribution", "js_contribution")):
        coefficients, participants = bootstrap_coefficients(frame, metrics[metric_name], seed + 23)
        for column, destination in enumerate(destinations):
            rows.append(summarize(coefficients[:, 1, column], "joint C-origin destination decomposition",
                                  "FS minus ES", n_rows=len(frame),
                                  n_participants=participants.participant.nunique(), outcome=output_name,
                                  destination=LABELS[destination]))
    summary = pd.DataFrame(rows)
    summary.to_csv(data_dir / "c_origin_route_destination_bayesian.csv", index=False)
    participant_means = frame[["participant", "is_functional", "semiology"]].copy()
    for column, destination in enumerate(destinations):
        label = LABELS[destination]
        participant_means[f"delta_p_{label}"] = metrics["delta_probability"][:, :, column].mean(axis=1)
        participant_means[f"js_contribution_{label}"] = metrics["js_contribution"][:, :, column].mean(axis=1)
    participant_means = participant_means.groupby(["participant", "is_functional", "semiology"], observed=True).mean().reset_index()
    participant_means.to_csv(data_dir / "c_origin_destination_participant_means.csv", index=False)
    return summary, participant_means


def all_state_specificity(table: pd.DataFrame, qc: pd.DataFrame, data_dir: Path,
                          draws: int, seed: int) -> pd.DataFrame:
    final = final_preictal(table)
    final = final[final.participant.astype(str).isin(reference_participants(qc, .50))].reset_index(drop=True)
    js_values, rows = [], []
    js_effects, kappa_effects = [], []
    for source, label in enumerate(LABELS):
        frame, metrics = draw_metrics(final, source, draws, np.random.default_rng(seed + source), baseline_table=table)
        if source == 0:
            final = frame
        elif not frame[["participant", "file_path"]].equals(final[["participant", "file_path"]]):
            raise ValueError("all-state raw-count samples did not retain an identical event set")
        js_values.append(metrics["js"])
        for outcome, values, store in (("JS", metrics["js"], js_effects),
                                       ("kappa", metrics["log_kappa"], kappa_effects)):
            coefficients, participants = bootstrap_coefficients(frame, values, seed + 500)
            effect = coefficients[:, 1, 0]; store.append(effect)
            rows.append(summarize(effect, "all-source raw-count specificity", "FS minus ES",
                                  n_rows=len(frame), n_participants=participants.participant.nunique(),
                                  outcome=outcome, source_state=label))
    js_effects = np.stack(js_effects, axis=1); kappa_effects = np.stack(kappa_effects, axis=1)
    global_values = np.mean(np.stack(js_values, axis=2), axis=2)
    coefficients, participants = bootstrap_coefficients(final, global_values, seed + 500)
    global_effect = coefficients[:, 1, 0]
    c_index = LABELS.index("C")
    rows.extend([
        summarize(global_effect, "all-source raw-count specificity", "FS minus ES",
                  n_rows=len(final), n_participants=participants.participant.nunique(), outcome="JS", source_state="global"),
        summarize(js_effects[:, c_index] - np.delete(js_effects, c_index, axis=1).mean(axis=1),
                  "all-source raw-count specificity", "C effect minus mean other-source effect",
                  n_rows=len(final), n_participants=participants.participant.nunique(), outcome="JS", source_state="C-specificity"),
        summarize(js_effects[:, c_index] - global_effect, "all-source raw-count specificity",
                  "C effect minus global equal-row effect", n_rows=len(final),
                  n_participants=participants.participant.nunique(), outcome="JS", source_state="C-minus-global")])
    summary = pd.DataFrame(rows)
    summary.to_csv(data_dir / "all_state_raw_count_specificity_bayesian.csv", index=False)
    return summary


def reference_sensitivity(table: pd.DataFrame, qc: pd.DataFrame, data_dir: Path,
                          draws: int, seed: int) -> pd.DataFrame:
    final = final_preictal(table)
    frame, metrics = draw_metrics(final, 5, draws, np.random.default_rng(seed), baseline_table=table)
    rows = []
    for threshold in (.50, .60, .70):
        keep = frame.participant.astype(str).isin(reference_participants(qc, threshold, "C")).to_numpy()
        use = frame[keep].reset_index(drop=True); values = np.log(metrics["js"][keep])
        coefficients, participants = bootstrap_coefficients(use, values, seed + int(threshold * 100))
        rows.append(summarize(coefficients[:, 1, 0], "C-label reference-threshold sensitivity", "FS minus ES",
                              n_rows=len(use), n_participants=participants.participant.nunique(),
                              threshold=threshold, outcome="log JS"))
    summary = pd.DataFrame(rows)
    summary.to_csv(data_dir / "c_reference_threshold_sensitivity_bayesian.csv", index=False)
    return summary


def aggregate_trajectory(table: pd.DataFrame) -> pd.DataFrame:
    pre = table[table.condition.eq("preict") & table.event_type.isin(("functional", "epileptic")) & table.semiology.notna()].copy()
    pre["time_bin"] = ""
    for low, high, label in TRAJECTORY_BINS:
        pre.loc[pre.window_end_to_ictus_s.between(low, high, inclusive="left"), "time_bin"] = label
    pre = pre[pre.time_bin.ne("")]
    transition_columns = [f"t_{i}_{j}" for i in range(1, 7) for j in range(1, 7)]
    keys = ["participant", "file_path", "event_key", "stratum", "semiology", "event_type",
            "is_functional", "is_hypo", "is_dd", "time_bin"]
    return pre.groupby(keys, observed=True)[transition_columns + ["n_gfp_peaks", "n_C"]].sum().reset_index()


def trajectory_analysis(table: pd.DataFrame, qc: pd.DataFrame, common: pd.DataFrame, data_dir: Path,
                        draws: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    binned = aggregate_trajectory(table)
    binned = binned[binned.participant.astype(str).isin(reference_participants(qc, .50, "C"))]
    bin_rows, event_rows, effects = [], [], {}
    for order, (_, _, label) in enumerate(TRAJECTORY_BINS):
        use = binned[binned.time_bin.eq(label)].reset_index(drop=True)
        frame, metrics = draw_metrics(use, 5, draws, np.random.default_rng(seed + order), baseline_table=table)
        means = frame[["participant", "file_path", "event_key", "is_functional", "time_bin"]].copy()
        means["js"] = metrics["js"].mean(axis=1); means["log_kappa"] = metrics["log_kappa"].mean(axis=1)
        event_rows.append(means)
        for outcome in ("js", "log_kappa"):
            coefficients, participants = bootstrap_coefficients(frame, metrics[outcome], seed + 500 + order)
            effect = coefficients[:, 1, 0]; effects[(label, outcome)] = effect
            bin_rows.append(summarize(effect, "raw-count preictal trajectory", "FS minus ES",
                                      n_rows=len(frame), n_participants=participants.participant.nunique(),
                                      time_bin=label, time_order=order, outcome=outcome))
        for group, functional in (("ES", 0), ("FS", 1)):
            keep = frame.is_functional.eq(functional).to_numpy()
            part = frame.loc[keep, ["participant"]].copy()
            for outcome in ("js", "log_kappa"):
                values = metrics[outcome][keep]
                codes, participants = pd.factorize(part.participant, sort=True)
                values = np.stack([values[codes == index].mean(axis=0) for index in range(len(participants))])
                rng = np.random.default_rng(seed + 1000 + order * 10 + functional)
                weights = rng.exponential(size=values.shape)
                posterior = np.sum(weights * values, axis=0) / weights.sum(axis=0)
                bin_rows.append(summarize(posterior, "raw-count preictal trajectory", group,
                                          n_rows=int(keep.sum()), n_participants=len(participants),
                                          time_bin=label, time_order=order, outcome=outcome))
    early, final = TRAJECTORY_BINS[0][2], TRAJECTORY_BINS[-1][2]
    for outcome in ("js", "log_kappa"):
        value = effects[(final, outcome)] - effects[(early, outcome)]
        bin_rows.append(summarize(value, "raw-count preictal trajectory",
                                  "FS-ES difference: final bin minus early bin", n_rows=len(binned),
                                  n_participants=binned.participant.nunique(), time_bin=f"{final} minus {early}",
                                  time_order=len(TRAJECTORY_BINS), outcome=outcome))
    events = pd.concat(event_rows, ignore_index=True)
    summary = pd.DataFrame(bin_rows)

    minute = table[table.condition.eq("preict") & table.window_end_to_ictus_s.notna()].copy()
    minute = minute[minute.participant.astype(str).isin(reference_participants(qc, .50, "C"))]
    minute, minute_metrics = draw_metrics(minute, 5, min(200, draws), np.random.default_rng(seed + 2000),
                                          baseline_table=table)
    minute["js"] = minute_metrics["js"].mean(axis=1); minute["log_kappa"] = minute_metrics["log_kappa"].mean(axis=1)
    join = ["participant", "file_path", "window_end_to_ictus_s"]
    physiology = minute[join + ["is_functional", "js", "log_kappa"]].merge(
        common[common.condition.eq("preict")][join + [column for column in
            ("c_expression", "csi_z", "cpi_z", "mean_hr_bpm") if column in common]], on=join, how="inner")
    rows = []
    for (functional, endpoint), part in physiology.groupby(["is_functional", "window_end_to_ictus_s"], observed=True):
        group = "FS" if functional else "ES"
        for outcome in ("js", "log_kappa", "c_expression", "csi_z", "cpi_z", "mean_hr_bpm"):
            if outcome not in part:
                continue
            participant = part.groupby("participant", observed=True)[outcome].mean().dropna().to_numpy(float)
            if len(participant) < 4:
                continue
            rng = np.random.default_rng(seed + 3000 + int(endpoint) + int(functional))
            weights = rng.exponential(size=(draws, len(participant)))
            posterior = (weights @ participant) / weights.sum(axis=1)
            rows.append(summarize(posterior, "aligned physiology trajectory", group, n_rows=len(part),
                                  n_participants=len(participant), outcome=outcome,
                                  minutes_to_ictus=-float(endpoint) / 60))
    physiology_summary = pd.DataFrame(rows)
    summary.to_csv(data_dir / "c_origin_preictal_binned_trajectory_bayesian.csv", index=False)
    events.to_csv(data_dir / "c_origin_preictal_binned_event_means.csv", index=False)
    physiology_summary.to_csv(data_dir / "preictal_physiology_trajectory_bayesian.csv", index=False)
    return summary, physiology_summary


def paired_difference(frame: pd.DataFrame, values: np.ndarray, seed: int) -> tuple[np.ndarray, int]:
    codes, participants = pd.factorize(frame.participant, sort=True)
    participant_values = np.stack([values[codes == index].mean(axis=0) for index in range(len(participants))])
    rng = np.random.default_rng(seed)
    weights = rng.exponential(size=participant_values.shape)
    return np.sum(weights * participant_values, axis=0) / weights.sum(axis=0), len(participants)


def recovery_analysis(table: pd.DataFrame, qc: pd.DataFrame, data_dir: Path,
                      draws: int, seed: int) -> pd.DataFrame:
    valid = reference_participants(qc, .50, "C")
    pre = final_preictal(table); pre = pre[pre.event_type.eq("functional") & pre.participant.astype(str).isin(valid)]
    post = table[table.condition.eq("postict") & table.event_type.eq("functional") &
                 table.participant.astype(str).isin(valid)].copy()
    pairs = pre.merge(post, on=["participant", "event_key"], suffixes=("_pre", "_post"), validate="one_to_one")
    rows = []
    for fixed, label, offset in ((None, "matched 60-second windows", 0),
                                 (FIXED_EXITS, f"matched {FIXED_EXITS} C exits", 1)):
        # Restore the original peak-table columns for the paired rows.
        pre_use = pre[pre.event_key.isin(pairs.event_key)].sort_values(["participant", "event_key"]).reset_index(drop=True)
        post_use = post[post.event_key.isin(pairs.event_key)].sort_values(["participant", "event_key"]).reset_index(drop=True)
        if fixed:
            columns, _ = count_columns(5)
            keep = pre_use[columns].sum(axis=1).ge(fixed) & post_use[columns].sum(axis=1).ge(fixed)
            pre_use, post_use = pre_use[keep].reset_index(drop=True), post_use[keep].reset_index(drop=True)
        pre_draw_frame, pre_metrics = draw_metrics(pre_use, 5, draws, np.random.default_rng(seed + offset), fixed, table)
        post_draw_frame, post_metrics = draw_metrics(post_use, 5, draws, np.random.default_rng(seed + 10 + offset), fixed, table)
        if not pre_draw_frame[["participant", "event_key"]].equals(post_draw_frame[["participant", "event_key"]]):
            raise ValueError("matched recovery rows lost alignment")
        for outcome in ("js", "log_kappa"):
            difference = ((np.log(post_metrics[outcome]) - np.log(pre_metrics[outcome])) if outcome == "js"
                          else post_metrics[outcome] - pre_metrics[outcome])
            posterior, n_participants = paired_difference(pre_draw_frame, difference, seed + 100 + offset)
            rows.append(summarize(posterior, "equal-information postictal recovery", "Postictal minus preictal",
                                  n_rows=len(pre_draw_frame), n_participants=n_participants,
                                  estimand=label, outcome="JS" if outcome == "js" else "kappa", scale="log ratio"))
    summary = pd.DataFrame(rows)
    summary.to_csv(data_dir / "c_origin_recovery_matched_bayesian.csv", index=False)
    return summary


def pseudo_onset_analysis(table: pd.DataFrame, qc: pd.DataFrame, common: pd.DataFrame,
                          data_dir: Path, draws: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = reference_participants(qc, .50, "C")
    join = ["participant", "file_path", "window_start_s", "window_end_s"]
    quality = common[join + ["n_native"]].drop_duplicates(join)
    candidates = table[table.condition.eq("ii") & table.participant.astype(str).isin(valid)].merge(quality, on=join, how="inner")
    genuine = final_preictal(table)
    genuine = genuine[genuine.participant.astype(str).isin(valid)].merge(quality, on=join, how="inner")
    columns, _ = count_columns(5)
    for frame in (candidates, genuine):
        frame["c_exits"] = frame[columns].sum(axis=1)
        frame["c_expression"] = frame.n_C / frame.n_gfp_peaks.clip(lower=1)
    matches = []
    feature_names = ("window_end_s", "n_gfp_peaks", "c_exits", "c_expression", "n_native")
    for true_index, true in genuine.iterrows():
        pool = candidates[candidates.participant.astype(str).eq(str(true.participant))].copy()
        if pool.empty:
            continue
        scales = pool[list(feature_names)].std().replace(0, 1).fillna(1)
        target = pd.to_numeric(true[list(feature_names)], errors="coerce").to_numpy(float)
        score = np.sqrt((((pool[list(feature_names)].astype(float) - target) / scales.astype(float)) ** 2).sum(axis=1))
        for rank, pseudo_index in enumerate(score.nsmallest(min(3, len(score))).index, 1):
            matches.append({"true_index": true_index, "pseudo_index": pseudo_index, "match_rank": rank,
                            "match_distance": score.loc[pseudo_index]})
    match = pd.DataFrame(matches)
    if match.empty:
        raise ValueError("no autonomic-QC-matched interictal pseudo-onsets")
    true = genuine.loc[match.true_index].reset_index(drop=True)
    pseudo = candidates.loc[match.pseudo_index].reset_index(drop=True)
    true_frame, true_metrics = draw_metrics(true, 5, draws, np.random.default_rng(seed), baseline_table=table)
    pseudo_frame, pseudo_metrics = draw_metrics(pseudo, 5, draws, np.random.default_rng(seed + 1), baseline_table=table)
    if len(true_frame) != len(pseudo_frame):
        raise ValueError("pseudo-onset draw eligibility changed the matched pairs")
    difference = np.log(true_metrics["js"]) - np.log(pseudo_metrics["js"])
    coefficients, participants = bootstrap_coefficients(true_frame, difference, seed + 2)
    overall = coefficients[:, 0, 0] + coefficients[:, 1, 0] * true_frame.is_functional.mean()
    rows = [summarize(overall, "matched interictal pseudo-onset", "Genuine minus pseudo onset",
                      n_rows=len(true_frame), n_participants=participants.participant.nunique(), outcome="log JS"),
            summarize(coefficients[:, 1, 0], "matched interictal pseudo-onset",
                      "FS-ES interaction in genuine-minus-pseudo difference", n_rows=len(true_frame),
                      n_participants=participants.participant.nunique(), outcome="log JS")]
    output = match.copy()
    output["participant"] = true.participant.to_numpy(); output["true_file_path"] = true.file_path.to_numpy()
    output["pseudo_file_path"] = pseudo.file_path.to_numpy(); output["event_key"] = true.event_key.to_numpy()
    output["is_functional"] = true.is_functional.to_numpy()
    output["true_js"] = true_metrics["js"].mean(axis=1)
    output["pseudo_js"] = pseudo_metrics["js"].mean(axis=1)
    for name in feature_names:
        output[f"true_{name}"] = true[name].to_numpy(); output[f"pseudo_{name}"] = pseudo[name].to_numpy()
    summary = pd.DataFrame(rows)
    output.to_csv(data_dir / "matched_interictal_pseudo_onsets.csv", index=False)
    summary.to_csv(data_dir / "matched_interictal_pseudo_onset_bayesian.csv", index=False)
    return summary, output


def run(table: pd.DataFrame, reference_qc: pd.DataFrame, common: pd.DataFrame,
        data_dir: Path, draws: int, seed: int) -> dict:
    primary, _ = raw_count_primary(table, reference_qc, data_dir, draws, seed)
    route, _ = route_and_destinations(table, reference_qc, data_dir, draws, seed + 10000)
    specificity = all_state_specificity(table, reference_qc, data_dir, draws, seed + 20000)
    sensitivity = reference_sensitivity(table, reference_qc, data_dir, draws, seed + 30000)
    _, physiology = trajectory_analysis(table, reference_qc, common, data_dir, draws, seed + 40000)
    recovery = recovery_analysis(table, reference_qc, data_dir, draws, seed + 50000)
    pseudo, matches = pseudo_onset_analysis(table, reference_qc, common, data_dir, draws, seed + 60000)
    c_specific = specificity[specificity.source_state.eq("C-specificity")].iloc[0]
    result = {"posterior_draws": draws, "fixed_c_exits": FIXED_EXITS,
              "raw_count_c_js_probability_fs_gt_es": float(primary[(primary.estimand.eq("fixed 60-second duration")) &
                  primary.outcome.eq("JS") & primary.contrast.eq("FS minus ES")].probability_gt_zero.iloc[0]),
              "c_specificity_probability_gt_zero": float(c_specific.probability_gt_zero),
              "pseudo_matches": len(matches), "trajectory_points": len(physiology),
              "unavailable_requested_covariates": ["respiration", "movement/EMG/accelerometry", "CFA variants",
                                                     "alternate segmentation outputs", "VR participant overlap"]}
    (data_dir / "feedback_analysis_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def self_check() -> None:
    rng = np.random.default_rng(3)
    p = rng.dirichlet([5, 4, 3, 2, 1], size=20)
    q = rng.dirichlet([4, 4, 4, 4, 4], size=20)
    metrics = transition_metrics(p, q)
    assert set(metrics) == {"js", "log_kappa", "delta_rho", "route_distance", "matched_js",
                            "delta_probability", "js_contribution"}
    assert np.all(metrics["js"] >= 0) and np.allclose(metrics["js"], metrics["js_contribution"].sum(axis=1))
    assert np.all((metrics["route_distance"] >= 0) & (metrics["route_distance"] <= 1))
    counts = np.array([20, 15, 10, 5, 2])
    assert np.all(thin_counts(rng, counts, 30, 10).sum(axis=1) == 30)


if __name__ == "__main__":
    self_check()
    print("transition-priority self-check passed")
