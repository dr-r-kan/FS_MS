"""Fit the requested transition hierarchy from existing K=6 microstate backfits."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler
import statsmodels.formula.api as smf

import analyse_transition_priorities as legacy


STATES = ("F", "E", "G", "B", "A", "C")
TIME_BINS = ((600, 1200, "-20 to -10"), (300, 600, "-10 to -5"),
             (120, 300, "-5 to -2"), (60, 120, "-2 to -1"), (0, 60, "-1 to 0"))
EQUALISED_TRANSITIONS = 30
JS_ROPE = 0.02
AUTONOMIC_COLUMNS = ("mean_hr_bpm_z", "mean_rr_s_z", "csi_z", "cpi_z", "rmssd_s_z",
                      "rr_first_difference_s_z", "rr_acceleration_s_z", "rr_surprise_z")


def _configure_pymc_cache() -> None:
    cache = Path(__file__).with_name(".cache")
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("WIN_PD_OVERRIDE_LOCAL_APPDATA", str(cache))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache / "numba"))
    compiledir = (cache / "pytensor").resolve().as_posix()
    os.environ.setdefault("PYTENSOR_FLAGS", f"base_compiledir={compiledir},cxx=,mode=NUMBA")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarise(draws: np.ndarray, **labels) -> dict:
    draws = np.asarray(draws, float)
    low, median, high = np.quantile(draws, (.025, .5, .975))
    return {**labels, "mean": float(np.mean(draws)), "median": float(median),
            "ci95_low": float(low), "ci95_high": float(high),
            "probability_gt_zero": float(np.mean(draws > 0)),
            "probability_lt_zero": float(np.mean(draws < 0))}


def add_ecg_artifact(common: pd.DataFrame, native_csv: Path | None) -> pd.DataFrame:
    out = common.copy()
    out["ecg_artifact_fraction"] = np.nan
    if native_csv is None or not native_csv.exists():
        return out
    native = pd.read_csv(native_csv, usecols=["participant", "file_path", "event_time_s", "recovered_fraction"])
    common_groups = out.groupby(["participant", "file_path"], observed=True).groups
    for key, part in native.groupby(["participant", "file_path"], observed=True, sort=False):
        indices = common_groups.get(key)
        if indices is None:
            continue
        part = part.sort_values("event_time_s")
        times = part.event_time_s.to_numpy(float)
        values = part.recovered_fraction.fillna(0).to_numpy(float)
        prefix = np.r_[0.0, np.cumsum(values)]
        windows = out.loc[indices]
        left = np.searchsorted(times, windows.window_start_s.to_numpy(float), side="left")
        right = np.searchsorted(times, windows.window_end_s.to_numpy(float), side="left")
        count = right - left
        out.loc[indices, "ecg_artifact_fraction"] = np.divide(
            prefix[right] - prefix[left], count, out=np.full(len(count), np.nan), where=count > 0)
    return out


def enrich_table(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["event_type"] = out.seizure_type.astype(str).str.upper().map({"FS": "functional", "ES": "epileptic"})
    out["semiology"] = out.hypo_hyper.astype(str).str.lower().replace({"nan": np.nan, "": np.nan})
    out["is_functional"] = out.event_type.eq("functional").astype(int)
    out["is_hypo"] = out.semiology.eq("hypo").astype(int)
    out["is_dd"] = out.group.astype(str).str.upper().eq("DD").astype(int)
    out["stratum"] = np.where(out.is_dd.eq(1), "DD ", "") + out.event_type.fillna("")
    out["event_key"] = out.file_path.astype(str)
    return out


def assign_time_bin(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["time_bin"] = ""
    for low, high, label in TIME_BINS:
        out.loc[out.window_end_to_ictus_s.between(low, high, inclusive="left"), "time_bin"] = label
    return out[out.time_bin.ne("")].copy()


def match_pseudo_onsets(table: pd.DataFrame, common: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    join = ["participant", "file_path", "window_start_s", "window_end_s"]
    quality_columns = [column for column in ("n_native", "ecg_artifact_fraction") if column in common]
    quality = common[join + quality_columns].drop_duplicates(join)
    source = table.merge(quality, on=join, how="left")
    record_end = source.groupby("file_path", observed=True).window_end_s.transform("max")
    source["record_fraction"] = source.window_end_s / record_end.where(record_end.gt(0))
    source["has_autonomic"] = source[quality_columns].notna().all(axis=1) if quality_columns else False
    true = source[(source.condition.eq("preict")) & source.window_end_to_ictus_s.eq(0) &
                  source.event_type.isin(("functional", "epileptic"))].copy()
    pseudo = source[source.condition.eq("ii")].copy()
    base_features = ["record_fraction", "n_gfp_peaks", "n_state_changes", *(f"n_{state}" for state in STATES)]
    feature_names = [*base_features, *quality_columns]
    matches = []
    for participant, targets in true.groupby("participant", observed=True, sort=False):
        pool = pseudo[pseudo.participant.eq(participant) & pseudo.record_fraction.ge(.80)].copy()
        if pool.empty:
            continue
        for has_autonomic in (False, True):
            target_group = targets[targets.has_autonomic.eq(has_autonomic)]
            pool_group = pool[pool.has_autonomic.eq(has_autonomic)]
            if target_group.empty or pool_group.empty:
                continue
            scales = pool_group[feature_names].std().replace(0, 1).fillna(1)
            target_values = target_group[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            pool_values = pool_group[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            difference = (target_values[:, None, :] - pool_values[None, :, :]) / scales.to_numpy(float)
            weights = np.array([2.0 if name in quality_columns else 1.0 for name in feature_names])
            difference *= weights
            comparable = np.isfinite(difference).sum(axis=2)
            cost = np.sqrt(np.nansum(difference ** 2, axis=2) / np.maximum(comparable, 1))
            target_rows, pseudo_rows = linear_sum_assignment(cost)
            for target_row, pseudo_row in zip(target_rows, pseudo_rows):
                target_index = int(target_group.index[target_row]); pseudo_index = int(pool_group.index[pseudo_row])
                target = target_group.iloc[target_row]
                row = {"true_index": target_index, "pseudo_index": pseudo_index,
                       "match_distance": float(cost[target_row, pseudo_row]),
                       "n_match_features": int(comparable[target_row, pseudo_row])}
                for name in feature_names:
                    row[f"true_{name}"] = target[name]
                    row[f"pseudo_{name}"] = pseudo.loc[pseudo_index, name]
                matches.append(row)
    match = pd.DataFrame(matches)
    if match.empty:
        raise ValueError("No within-participant interictal pseudo-onsets could be matched")
    true_rows = source.loc[match.true_index].reset_index(drop=True)
    pseudo_rows = pseudo.loc[match.pseudo_index].reset_index(drop=True)
    pseudo_rows[["event_type", "semiology", "is_functional", "is_hypo", "is_dd", "stratum"]] = (
        true_rows[["event_type", "semiology", "is_functional", "is_hypo", "is_dd", "stratum"]].to_numpy())
    pseudo_rows["event_key"] = true_rows.event_key.to_numpy()
    pseudo_rows["matched_pseudo_file_path"] = pseudo_rows.file_path
    pseudo_rows["time_bin"] = "pseudo-onset"
    pseudo_rows["true_onset"] = 0
    match["participant"] = true_rows.participant.to_numpy()
    match["event_key"] = true_rows.event_key.to_numpy()
    match["event_type"] = true_rows.event_type.to_numpy()
    match["true_file_path"] = true_rows.file_path.to_numpy()
    match["pseudo_file_path"] = pseudo_rows.file_path.to_numpy()
    return pseudo_rows, match


def prepare_model_windows(table: pd.DataFrame, common: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    true = assign_time_bin(table[(table.condition.eq("preict")) &
                                 table.event_type.isin(("functional", "epileptic"))])
    pseudo, matches = match_pseudo_onsets(table, common)
    true["true_onset"] = 1
    frame = pd.concat([true, pseudo], ignore_index=True)
    reference_bin = TIME_BINS[0][2]
    frame["group_fs"] = frame.is_functional.astype(float)
    frame["group_x_true"] = frame.group_fs * frame.true_onset
    covariates = ["intercept", "group_fs", "true_onset", "group_x_true"]
    frame["intercept"] = 1.0
    for _, _, label in TIME_BINS[1:]:
        name = "time_" + label.replace(" ", "_")
        interaction = "fs_x_" + name
        frame[name] = frame.time_bin.eq(label).astype(float)
        frame[interaction] = frame.group_fs * frame[name]
        covariates.extend((name, interaction))
    frame["hypomotor"] = frame.is_hypo.astype(float)
    frame["dd_cohort"] = frame.is_dd.astype(float)
    covariates.extend(("hypomotor", "dd_cohort"))
    frame["reference_time_bin"] = reference_bin
    frame["window_id"] = np.arange(len(frame))
    return frame, matches, covariates


def reshape_counts(frame: pd.DataFrame, covariates: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows, counts = [], []
    for source in range(len(STATES)):
        destinations = [index for index in range(len(STATES)) if index != source]
        columns = [f"t_{source + 1}_{destination + 1}" for destination in destinations]
        values = frame[columns].to_numpy(int)
        keep = values.sum(axis=1) > 0
        part = frame.loc[keep, ["window_id", "participant", "event_key", *covariates]].copy()
        part["source"] = source
        part["total_transitions"] = values[keep].sum(axis=1)
        rows.append(part)
        counts.append(values[keep])
    long = pd.concat(rows, ignore_index=True)
    participants = pd.Index(frame.participant.unique(), name="participant")
    events = pd.Index(frame.event_key.unique(), name="event")
    long["participant_index"] = participants.get_indexer(long.participant)
    long["event_index"] = events.get_indexer(long.event_key)
    return long, np.vstack(counts), frame[covariates].to_numpy(float)


def fit_hierarchical_multinomial(frame: pd.DataFrame, covariates: list[str], out_dir: Path,
                                 draws: int, advi_iterations: int, seed: int):
    _configure_pymc_cache()
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt

    long, counts, _ = reshape_counts(frame, covariates)
    x = long[covariates].to_numpy(float)
    source = long.source.to_numpy(int)
    participant = long.participant_index.to_numpy(int)
    event = long.event_index.to_numpy(int)
    totals = counts.sum(axis=1).astype(int)
    coords = {"source": STATES, "covariate": covariates, "alr": np.arange(4),
              "participant": frame.participant.unique(), "event": frame.event_key.unique(),
              "observation": np.arange(len(long)), "destination_position": np.arange(5)}
    with pm.Model(coords=coords) as model:
        beta_mean = pm.Normal("beta_mean", 0, 1, dims=("covariate", "alr"))
        beta_sd = pm.HalfNormal("beta_sd", .5, dims=("covariate", "alr"))
        beta_offset = pm.Normal("beta_offset", 0, 1, dims=("source", "covariate", "alr"))
        beta = pm.Deterministic("beta", beta_mean[None, :, :] + beta_sd[None, :, :] * beta_offset,
                                dims=("source", "covariate", "alr"))
        participant_sd = pm.HalfNormal("participant_sd", .5, dims=("source", "alr"))
        participant_offset = pm.Normal("participant_offset", 0, 1,
                                       dims=("participant", "source", "alr"))
        event_sd = pm.HalfNormal("event_sd", .35, dims=("source", "alr"))
        event_offset = pm.Normal("event_offset", 0, 1, dims=("event", "source", "alr"))
        eta = (pt.sum(beta[source] * x[:, :, None], axis=1) +
               participant_offset[participant, source] * participant_sd[source] +
               event_offset[event, source] * event_sd[source])
        probability = pm.math.softmax(pt.concatenate((eta, pt.zeros((len(long), 1))), axis=1), axis=1)
        pm.Multinomial("transition_counts", n=totals, p=probability, observed=counts,
                       dims=("observation", "destination_position"))
        approximation = pm.fit(advi_iterations, method="advi", random_seed=seed, progressbar=True,
                               callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])
        posterior = approximation.sample(draws, random_seed=seed, return_inferencedata=True)
    posterior.posterior.attrs.update({"likelihood": "Multinomial raw transition counts",
                                      "inference": "mean-field ADVI", "advi_iterations": advi_iterations,
                                      "transition_count_role": "multinomial denominator only"})
    az.to_netcdf(posterior, out_dir / "hierarchical_multinomial_posterior.nc")
    pd.DataFrame({"iteration": np.arange(1, len(approximation.hist) + 1),
                  "negative_elbo": approximation.hist}).to_csv(out_dir / "advi_loss.csv", index=False)
    long.to_csv(out_dir / "hierarchical_multinomial_observations.csv", index=False)
    return posterior, long


def posterior_array(posterior, name: str) -> np.ndarray:
    value = posterior.posterior[name]
    dimensions = [dimension for dimension in value.dims if dimension not in ("chain", "draw")]
    return value.stack(sample=("chain", "draw")).transpose("sample", *dimensions).to_numpy()


def softmax_last_reference(eta: np.ndarray) -> np.ndarray:
    eta = np.concatenate((eta, np.zeros((*eta.shape[:-1], 1))), axis=-1)
    eta -= eta.max(axis=-1, keepdims=True)
    probability = np.exp(eta)
    return probability / probability.sum(axis=-1, keepdims=True)


def transition_metrics(probability: np.ndarray, baseline: np.ndarray) -> dict[str, np.ndarray]:
    eps = np.finfo(float).tiny
    p, q = np.clip(probability, eps, 1), np.clip(baseline, eps, 1)
    midpoint = (p + q) / 2
    js = .5 * np.sum(p * np.log(p / midpoint), axis=-1) + .5 * np.sum(q * np.log(q / midpoint), axis=-1)
    entropy_concentration = 1 + np.sum(p * np.log(p), axis=-1) / math.log(p.shape[-1])
    log_p = np.log(p); log_q = np.log(q)
    centred_p = log_p - log_p.mean(axis=-1, keepdims=True)
    centred_q = log_q - log_q.mean(axis=-1, keepdims=True)
    norm_p = np.linalg.norm(centred_p, axis=-1)
    norm_q = np.linalg.norm(centred_q, axis=-1)
    unit_p = centred_p / np.maximum(norm_p[..., None], 1e-12)
    unit_q = centred_q / np.maximum(norm_q[..., None], 1e-12)
    concentration = ((norm_p + norm_q) / 2)[..., None]
    matched_p = softmax_last_reference((concentration * unit_p)[..., :-1] -
                                       (concentration * unit_p)[..., -1, None])
    matched_q = softmax_last_reference((concentration * unit_q)[..., :-1] -
                                       (concentration * unit_q)[..., -1, None])
    matched_mid = (matched_p + matched_q) / 2
    route = (.5 * np.sum(matched_p * np.log(matched_p / matched_mid), axis=-1) +
             .5 * np.sum(matched_q * np.log(matched_q / matched_mid), axis=-1))
    target = np.sum(p * log_q, axis=-1)
    low, high = np.full(target.shape, math.exp(-6)), np.full(target.shape, math.exp(6))
    for _ in range(28):
        kappa = (low + high) / 2
        weighted = kappa[..., None] * log_q
        weighted -= weighted.max(axis=-1, keepdims=True)
        distribution = np.exp(weighted); distribution /= distribution.sum(axis=-1, keepdims=True)
        expectation = np.sum(distribution * log_q, axis=-1)
        low = np.where(expectation < target, kappa, low)
        high = np.where(expectation < target, high, kappa)
    return {"J": js, "K_H": entropy_concentration, "D_route": route, "kappa": (low + high) / 2,
            "delta_probability": p - q}


def equal_event_participant_mean(values: np.ndarray, meta: pd.DataFrame, keep: np.ndarray) -> np.ndarray:
    selected = meta.loc[keep, ["participant", "event_key"]].copy()
    selected["column"] = np.flatnonzero(keep)
    participant_draws = []
    for _, participant in selected.groupby("participant", observed=True, sort=False):
        event_draws = [values[:, event.column.to_numpy(int)].mean(axis=1)
                       for _, event in participant.groupby("event_key", observed=True, sort=False)]
        participant_draws.append(np.mean(event_draws, axis=0))
    if not participant_draws:
        raise ValueError("No observations in requested posterior cell")
    return np.mean(participant_draws, axis=0)


def derive_posterior_metrics(posterior, long: pd.DataFrame, frame: pd.DataFrame,
                             covariates: list[str], baseline_table: pd.DataFrame,
                             out_dir: Path, seed: int) -> tuple[pd.DataFrame, dict, dict]:
    beta = posterior_array(posterior, "beta")
    participant_effect = (posterior_array(posterior, "participant_offset") *
                          posterior_array(posterior, "participant_sd")[:, None, :, :])
    event_effect = (posterior_array(posterior, "event_offset") *
                    posterior_array(posterior, "event_sd")[:, None, :, :])
    n_draws = beta.shape[0]
    rng = np.random.default_rng(seed)
    baseline_counts = {}
    for source in range(len(STATES)):
        destinations = [index for index in range(len(STATES)) if index != source]
        columns = [f"t_{source + 1}_{destination + 1}" for destination in destinations]
        baseline_counts[source] = {str(participant): part[columns].sum().to_numpy(float)
                                   for participant, part in baseline_table[baseline_table.condition.eq("ii")]
                                   .groupby("participant", observed=True)}
    metric_names = ("J", "K_H", "D_route", "kappa")
    global_sum = {name: np.zeros((n_draws, len(frame))) for name in metric_names}
    global_n = np.zeros(len(frame), int)
    rows, destination_rows, cells = [], [], {}
    for source in range(len(STATES)):
        use = long.source.eq(source).to_numpy()
        meta = long.loc[use].reset_index(drop=True)
        x = meta[covariates].to_numpy(float)
        eta = (np.einsum("sck,oc->sok", beta[:, source], x) +
               participant_effect[:, meta.participant_index.to_numpy(int), source] +
               event_effect[:, meta.event_index.to_numpy(int), source])
        probability = softmax_last_reference(eta)
        baseline_by_participant = {}
        for participant in meta.participant.astype(str).unique():
            count = baseline_counts[source].get(participant)
            if count is None:
                raise ValueError(f"No interictal baseline for {participant}, source {STATES[source]}")
            baseline_by_participant[participant] = rng.dirichlet(count + .5, size=n_draws)
        baseline = np.stack([baseline_by_participant[str(participant)] for participant in meta.participant], axis=1)
        metrics = transition_metrics(probability, baseline)
        window_ids = meta.window_id.to_numpy(int)
        for name in metric_names:
            global_sum[name][:, window_ids] += metrics[name]
        global_n[window_ids] += 1
        destinations = [index for index in range(len(STATES)) if index != source]
        for column, destination in enumerate(destinations):
            change = metrics["delta_probability"][:, :, column]
            for time_bin in ["pseudo-onset", *(label for _, _, label in TIME_BINS)]:
                group_draws = {}
                for group, functional in (("ES", 0), ("FS", 1)):
                    keep = meta.window_id.map(frame.set_index("window_id").time_bin).eq(time_bin).to_numpy()
                    keep &= meta.window_id.map(frame.set_index("window_id").is_functional).eq(functional).to_numpy()
                    if keep.any():
                        draw = equal_event_participant_mean(change, meta, keep)
                        group_draws[group] = draw
                        destination_rows.append(summarise(draw, source_state=STATES[source],
                                                          destination=STATES[destination], time_bin=time_bin,
                                                          group=group, outcome="destination_probability_change"))
                if set(group_draws) == {"ES", "FS"}:
                    destination_rows.append(summarise(group_draws["FS"] - group_draws["ES"],
                                                      source_state=STATES[source],
                                                      destination=STATES[destination], time_bin=time_bin,
                                                      group="FS minus ES",
                                                      outcome="destination_probability_change"))
        window_meta = frame.set_index("window_id").loc[window_ids].reset_index()
        for observation, item in window_meta.iterrows():
            row = {key: item[key] for key in ("window_id", "participant", "event_key", "file_path", "condition",
                                               "time_bin", "window_start_s", "window_end_s",
                                               "window_end_to_ictus_s", "event_type", "semiology",
                                               "is_functional", "is_dd")}
            row.update({"source_state": STATES[source], "total_transitions": int(meta.total_transitions.iloc[observation])})
            for name in metric_names:
                draw = metrics[name][:, observation]
                low, median, high = np.quantile(draw, (.025, .5, .975))
                row.update({f"{name}_mean": float(draw.mean()), f"{name}_median": float(median),
                            f"{name}_ci95_low": float(low), f"{name}_ci95_high": float(high)})
            rows.append(row)
        for time_bin in ["pseudo-onset", *(label for _, _, label in TIME_BINS)]:
            for group, functional in (("ES", 0), ("FS", 1)):
                keep = window_meta.time_bin.eq(time_bin).to_numpy() & window_meta.is_functional.eq(functional).to_numpy()
                if not keep.any():
                    continue
                for name in metric_names:
                    draw = equal_event_participant_mean(metrics[name], meta, keep)
                    cells[(name, STATES[source], group, time_bin, False)] = draw
    global_metrics = {name: values / np.maximum(global_n, 1)[None, :] for name, values in global_sum.items()}
    output = pd.DataFrame(rows)
    output.to_csv(out_dir / "posterior_transition_window_metrics.csv.gz", index=False, compression="gzip")
    pd.DataFrame(destination_rows).to_csv(out_dir / "posterior_destination_changes.csv", index=False)
    pd.DataFrame([summarise(draw, outcome=name, source_state=source, group=group, time_bin=time_bin)
                  for (name, source, group, time_bin, _), draw in cells.items()]) \
        .to_csv(out_dir / "posterior_source_metric_cells.csv", index=False)
    global_means = frame[["window_id", "participant", "event_key", "file_path", "condition", "time_bin",
                          "window_start_s", "window_end_s", "window_end_to_ictus_s", "event_type",
                          "is_functional", "is_dd"]].copy()
    for name in metric_names:
        global_means[name] = global_metrics[name].mean(axis=0)
    global_means.to_csv(out_dir / "posterior_global_transition_window_means.csv.gz", index=False, compression="gzip")
    return global_means, global_metrics, cells


def planned_contrasts(frame: pd.DataFrame, global_metrics: dict[str, np.ndarray], cells: dict,
                      out_dir: Path) -> pd.DataFrame:
    early, late = TIME_BINS[0][2], TIME_BINS[-1][2]
    rows = []
    complete_events = set(frame[frame.true_onset.eq(1)].groupby("event_key", observed=True).time_bin.nunique()
                          .loc[lambda value: value.eq(len(TIME_BINS))].index)
    for complete in (False, True):
        suffix = "complete temporal coverage" if complete else "all eligible events"
        metric_cells = {}
        for name, values in global_metrics.items():
            for group, functional in (("ES", 0), ("FS", 1)):
                for time_bin in (early, late):
                    keep = frame.time_bin.eq(time_bin).to_numpy() & frame.is_functional.eq(functional).to_numpy()
                    if complete:
                        keep &= frame.event_key.isin(complete_events).to_numpy()
                    metric_cells[(name, group, time_bin)] = equal_event_participant_mean(values, frame, keep)
        draws = metric_cells[("kappa", "FS", early)] - metric_cells[("kappa", "ES", early)]
        rows.append(summarise(draws, analysis=suffix, outcome="kappa",
                              contrast="FS(-20,-10) minus ES(-20,-10)", planned_direction="<0"))
        draws = metric_cells[("kappa", "FS", late)] - metric_cells[("kappa", "FS", early)]
        rows.append(summarise(draws, analysis=suffix, outcome="kappa",
                              contrast="FS(-1,0) minus FS(-20,-10)", planned_direction=">0"))
        draws = metric_cells[("J", "FS", late)] - metric_cells[("J", "FS", early)]
        row = summarise(draws, analysis=suffix, outcome="J",
                        contrast="FS(-1,0) minus FS(-20,-10)", planned_direction=f"within +/-{JS_ROPE}")
        row["rope_low"], row["rope_high"] = -JS_ROPE, JS_ROPE
        row["probability_in_rope"] = float(np.mean(np.abs(draws) <= JS_ROPE))
        rows.append(row)
    result = pd.DataFrame(rows)
    result["complete_events"] = len(complete_events)
    result.to_csv(out_dir / "planned_early_late_contrasts.csv", index=False)
    return result


def spline_trajectory(frame: pd.DataFrame, global_metrics: dict[str, np.ndarray], out_dir: Path,
                      seed: int) -> pd.DataFrame:
    true = frame.true_onset.eq(1).to_numpy()
    meta = frame.loc[true].reset_index(drop=True)
    time = (-meta.window_end_to_ictus_s.to_numpy(float) / 60)[:, None]
    spline = SplineTransformer(n_knots=6, degree=3, include_bias=True).fit(time)
    design = spline.transform(time)
    grid = np.linspace(-20, 0, 81)[:, None]
    grid_design = spline.transform(grid)
    rng = np.random.default_rng(seed)
    rows = []
    for group, functional in (("ES", 0), ("FS", 1)):
        keep = meta.is_functional.eq(functional).to_numpy()
        x = design[keep]
        participant_codes = pd.factorize(meta.loc[keep, "participant"], sort=True)[0]
        for name in ("J", "kappa", "K_H", "D_route"):
            values = global_metrics[name][:, true][:, keep]
            predictions = np.empty((values.shape[0], len(grid)))
            for draw in range(values.shape[0]):
                participant_weight = rng.exponential(size=participant_codes.max() + 1)
                weight = participant_weight[participant_codes]
                precision = x.T @ (weight[:, None] * x) + np.eye(x.shape[1]) * 1e-3
                coefficient = np.linalg.solve(precision, x.T @ (weight * values[draw]))
                predictions[draw] = grid_design @ coefficient
            low, median, high = np.quantile(predictions, (.025, .5, .975), axis=0)
            rows.extend({"group": group, "outcome": name, "minutes_to_onset": float(grid[index, 0]),
                         "median": float(median[index]), "ci95_low": float(low[index]),
                         "ci95_high": float(high[index])} for index in range(len(grid)))
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "group_specific_spline_trajectory.csv", index=False)
    return result


def source_specificity(cells: dict, out_dir: Path) -> pd.DataFrame:
    late = TIME_BINS[-1][2]
    effects = {state: cells[("J", state, "FS", late, False)] - cells[("J", state, "ES", late, False)]
               for state in STATES}
    rows = [summarise(draw, source_state=state, outcome="J", estimand="FS minus ES in final minute")
            for state, draw in effects.items()]
    difference = effects["C"] - np.mean([effects[state] for state in STATES if state != "C"], axis=0)
    rows.append(summarise(difference, source_state="C-specificity", outcome="J",
                          estimand="beta_C minus mean beta_other"))
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "hierarchical_source_specificity.csv", index=False)
    return result


def autonomic_factors(common: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    complete = common.dropna(subset=list(AUTONOMIC_COLUMNS)).copy()
    interictal = complete.condition.eq("ii")
    if interictal.sum() < 30:
        raise ValueError("Fewer than 30 complete interictal windows for autonomic PCA")
    scaler = StandardScaler().fit(complete.loc[interictal, list(AUTONOMIC_COLUMNS)])
    pca = PCA(n_components=3).fit(scaler.transform(complete.loc[interictal, list(AUTONOMIC_COLUMNS)]))
    scores = pca.transform(scaler.transform(complete.loc[:, list(AUTONOMIC_COLUMNS)]))
    anchors = np.zeros((3, len(AUTONOMIC_COLUMNS)))
    anchors[0, AUTONOMIC_COLUMNS.index("mean_hr_bpm_z")] = 1
    anchors[0, AUTONOMIC_COLUMNS.index("mean_rr_s_z")] = -1
    anchors[1, AUTONOMIC_COLUMNS.index("rmssd_s_z")] = 1
    anchors[1, AUTONOMIC_COLUMNS.index("cpi_z")] = 1
    anchors[2, AUTONOMIC_COLUMNS.index("rr_first_difference_s_z")] = 1
    anchors[2, AUTONOMIC_COLUMNS.index("rr_acceleration_s_z")] = 1
    anchors[2, AUTONOMIC_COLUMNS.index("rr_surprise_z")] = 1
    similarity = anchors @ pca.components_.T
    anchor_index, component_index = linear_sum_assignment(-np.abs(similarity))
    names = ("A_mobilisation", "A_variability", "A_directional")
    ordered_scores = np.empty_like(scores)
    loading_rows = []
    for anchor, component in zip(anchor_index, component_index):
        sign = 1 if similarity[anchor, component] >= 0 else -1
        ordered_scores[:, anchor] = scores[:, component] * sign
        for column, loading in zip(AUTONOMIC_COLUMNS, pca.components_[component] * sign):
            loading_rows.append({"factor": names[anchor], "input": column, "loading": loading,
                                 "explained_variance_ratio": pca.explained_variance_ratio_[component]})
    for index, name in enumerate(names):
        complete[name] = ordered_scores[:, index]
    pd.DataFrame(loading_rows).to_csv(out_dir / "interictal_autonomic_pca_loadings.csv", index=False)
    complete.to_csv(out_dir / "autonomic_factor_windows.csv.gz", index=False, compression="gzip")
    return complete


def make_dynamic_frame(factors: pd.DataFrame, global_means: pd.DataFrame) -> pd.DataFrame:
    keys = ["participant", "file_path", "window_start_s", "window_end_s"]
    transition = global_means[keys + ["J", "K_H", "D_route", "kappa"]]
    frame = factors.merge(transition, on=keys, how="inner")
    frame = frame[frame.condition.eq("preict")].copy()
    frame["O"] = frame.c_expression
    frame["R"] = frame.D_route
    frame["K"] = frame.K_H
    rows = []
    columns = ["A_mobilisation", "A_variability", "A_directional", "O", "R", "K", "J", "kappa"]
    for _, event in frame.groupby(["participant", "event_key"], observed=True, sort=False):
        event = event.sort_values("window_start_s").copy()
        gap = event.window_start_s.diff()
        for column in columns:
            event[f"{column}_lag"] = event[column].shift(1)
        rows.append(event[gap.eq(60)])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def lagged_factor_tests(dynamic: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    specifications = (("J", "A_mobilisation", "mobilisation(t) -> J(t+1)"),
                      ("kappa", "A_variability", "variability(t) -> kappa(t+1)"))
    rows = []
    for outcome, factor, label in specifications:
        columns = [outcome, f"{outcome}_lag", f"{factor}_lag", "participant", "event_key",
                   "window_end_to_ictus_s"]
        use = dynamic.dropna(subset=columns).copy()
        use["time_scaled"] = use.window_end_to_ictus_s / 1200
        formula = f"{outcome} ~ {outcome}_lag + {factor}_lag + time_scaled + C(event_key)"
        try:
            fit = smf.ols(formula, use).fit(cov_type="cluster", cov_kwds={"groups": use.participant})
            coefficient = fit.params[f"{factor}_lag"]
            standard_error = fit.bse[f"{factor}_lag"]
            rows.append({"analysis": label, "outcome": outcome, "predictor": factor,
                         "estimate": coefficient, "standard_error": standard_error,
                         "ci95_low": coefficient - 1.96 * standard_error,
                         "ci95_high": coefficient + 1.96 * standard_error,
                         "p_value": fit.pvalues[f"{factor}_lag"], "n_windows": len(use),
                         "n_events": use.event_key.nunique(), "n_participants": use.participant.nunique(),
                         "model": "event fixed effects; participant-clustered standard errors"})
        except (np.linalg.LinAlgError, ValueError) as error:
            rows.append({"analysis": label, "outcome": outcome, "predictor": factor,
                         "error": str(error), "n_windows": len(use), "n_events": use.event_key.nunique(),
                         "n_participants": use.participant.nunique()})
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "lagged_autonomic_factor_tests.csv", index=False)
    return result


def dynamic_loop_comparison(dynamic: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    variables = ("A_mobilisation", "O", "R", "K")
    needed = [*variables, *(f"{name}_lag" for name in variables), "event_key"]
    frame = dynamic.dropna(subset=needed).copy()
    groups = frame.event_key.astype(str).to_numpy()
    unique_events = pd.unique(groups)
    event_rows, coefficient_rows = [], []
    models = ("autoregression_only", "shared_state_only", "autonomic_to_EEG",
              "EEG_to_autonomic", "bidirectional_loop")

    def predictors(model: str, outcome_index: int, state: np.ndarray, latent: np.ndarray) -> np.ndarray:
        own = state[:, [outcome_index]]
        if model == "autoregression_only":
            return own
        if model == "shared_state_only":
            return latent
        if model == "autonomic_to_EEG":
            return own if outcome_index == 0 else np.column_stack((own, state[:, 0]))
        if model == "EEG_to_autonomic":
            return np.column_stack((own, state[:, 1:])) if outcome_index == 0 else own
        return state

    x_raw = frame[[f"{name}_lag" for name in variables]].to_numpy(float)
    y_raw = frame[list(variables)].to_numpy(float)
    for held_event in unique_events:
        test = groups == held_event
        train = ~test
        x_scaler, y_scaler = StandardScaler().fit(x_raw[train]), StandardScaler().fit(y_raw[train])
        x_train, x_test = x_scaler.transform(x_raw[train]), x_scaler.transform(x_raw[test])
        y_train, y_test = y_scaler.transform(y_raw[train]), y_scaler.transform(y_raw[test])
        latent_model = PCA(n_components=1).fit(x_train)
        latent_train, latent_test = latent_model.transform(x_train), latent_model.transform(x_test)
        for model in models:
            for outcome_index, outcome in enumerate(variables):
                train_design = predictors(model, outcome_index, x_train, latent_train)
                test_design = predictors(model, outcome_index, x_test, latent_test)
                fit = Ridge(alpha=1).fit(train_design, y_train[:, outcome_index])
                residual = y_train[:, outcome_index] - fit.predict(train_design)
                variance = max(float(np.mean(residual ** 2)), 1e-9)
                prediction = fit.predict(test_design)
                error = y_test[:, outcome_index] - prediction
                event_rows.append({"held_out_event": held_event, "model": model, "outcome": outcome,
                                   "n_windows": int(test.sum()), "squared_error": float(np.sum(error ** 2)),
                                   "log_predictive_density": float(np.sum(-.5 * (np.log(2 * np.pi * variance) +
                                                                                  error ** 2 / variance)))})
    event_scores = pd.DataFrame(event_rows)
    comparison = (event_scores.groupby(["model", "outcome"], observed=True)
                  .agg(n_events=("held_out_event", "nunique"), n_windows=("n_windows", "sum"),
                       squared_error=("squared_error", "sum"), elpd=("log_predictive_density", "sum"))
                  .reset_index())
    comparison["rmse"] = np.sqrt(comparison.squared_error / comparison.n_windows)
    overall = (event_scores.groupby("model", observed=True)
               .agg(n_events=("held_out_event", "nunique"), n_windows=("n_windows", "sum"),
                    squared_error=("squared_error", "sum"), elpd=("log_predictive_density", "sum")).reset_index())
    overall["outcome"] = "all"
    overall["rmse"] = np.sqrt(overall.squared_error / overall.n_windows)
    event_elpd = event_scores.groupby(["held_out_event", "model"], observed=True).log_predictive_density.sum().unstack()
    best_model = overall.sort_values("elpd", ascending=False).model.iloc[0]
    for model in overall.model:
        difference = event_elpd[model] - event_elpd[best_model]
        overall.loc[overall.model.eq(model), "delta_elpd"] = difference.sum()
        overall.loc[overall.model.eq(model), "se_delta_elpd"] = (
            math.sqrt(len(difference) * difference.var(ddof=1)) if model != best_model else 0.0)
    comparison = pd.concat([comparison, overall], ignore_index=True)

    x_scaler, y_scaler = StandardScaler().fit(x_raw), StandardScaler().fit(y_raw)
    x, y = x_scaler.transform(x_raw), y_scaler.transform(y_raw)
    latent_model = PCA(n_components=1).fit(x); latent = latent_model.transform(x)
    for model in models:
        for outcome_index, outcome in enumerate(variables):
            design = predictors(model, outcome_index, x, latent)
            fit = Ridge(alpha=1).fit(design, y[:, outcome_index])
            for index, coefficient in enumerate(np.atleast_1d(fit.coef_)):
                coefficient_rows.append({"model": model, "outcome": outcome,
                                         "predictor_index": index, "standardised_coefficient": coefficient})
    pd.DataFrame(coefficient_rows).to_csv(out_dir / "dynamic_loop_coefficients.csv", index=False)
    event_scores.to_csv(out_dir / "dynamic_loop_leave_one_event_out.csv", index=False)
    comparison.to_csv(out_dir / "dynamic_loop_model_comparison.csv", index=False)
    return comparison, event_scores


def availability_audit(common: pd.DataFrame, out_dir: Path) -> dict:
    requested = {"movement/EMG": (), "EOG": (), "high-frequency muscle power": (),
                 "ECG artefact magnitude": ("ecg_artifact_fraction",), "bad-channel burden": (),
                 "sleep/wake state": (), "medication timing": ()}
    available = {name: [column for column in columns if column in common and common[column].notna().any()]
                 for name, columns in requested.items()}
    result = {"available_for_matching": {name: columns for name, columns in available.items() if columns},
              "unavailable_in_current_exports": [name for name, columns in available.items() if not columns],
              "other_previously_unavailable": ["respiration", "CFA variants", "alternative segmentations",
                                                  "VR participant overlap"],
              "policy": "Unavailable variables were not imputed or represented as matched."}
    (out_dir / "pseudo_onset_covariate_availability.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def matching_balance(matches: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for true_column in matches.filter(regex=r"^true_").columns:
        name = true_column.removeprefix("true_")
        if name in ("index", "file_path"):
            continue
        pseudo_column = f"pseudo_{name}"
        if pseudo_column not in matches:
            continue
        true = pd.to_numeric(matches[true_column], errors="coerce")
        pseudo = pd.to_numeric(matches[pseudo_column], errors="coerce")
        keep = true.notna() & pseudo.notna()
        pooled_sd = math.sqrt((true[keep].var(ddof=1) + pseudo[keep].var(ddof=1)) / 2)
        if not keep.any():
            continue
        rows.append({"matching_variable": name, "n_pairs": int(keep.sum()),
                     "standardised_mean_difference": ((true[keep].mean() - pseudo[keep].mean()) / pooled_sd
                                                       if pooled_sd > 0 else 0.0),
                     "median_absolute_paired_difference": float((true[keep] - pseudo[keep]).abs().median())})
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "pseudo_onset_match_balance.csv", index=False)
    return result


def write_report(out_dir: Path, planned: pd.DataFrame, specificity: pd.DataFrame,
                 factor_tests: pd.DataFrame, loop: pd.DataFrame, audit: dict) -> None:
    def markdown(frame: pd.DataFrame, columns: list[str]) -> str:
        use = frame[[column for column in columns if column in frame]].copy()
        for column in use.select_dtypes(include="number"):
            use[column] = use[column].map(lambda value: f"{value:.4g}" if pd.notna(value) else "")
        values = [[str(value).replace("|", "\\|") for value in row] for row in use.to_numpy()]
        header = "| " + " | ".join(use.columns) + " |"
        separator = "| " + " | ".join("---" for _ in use.columns) + " |"
        return "\n".join((header, separator, *("| " + " | ".join(row) + " |" for row in values)))

    best = loop[loop.outcome.eq("all")].sort_values("elpd", ascending=False).iloc[0]
    c_specific = specificity[specificity.source_state.eq("C-specificity")].iloc[0]
    balance = pd.read_csv(out_dir / "pseudo_onset_match_balance.csv").set_index("matching_variable")
    position_difference = balance.loc["record_fraction", "median_absolute_paired_difference"]
    report = ["# Equalised and optimised transition analysis", "",
              f"Completed: {datetime.now(timezone.utc).isoformat()}", "",
              "## Model", "",
              "Raw transition counts were fitted with one hierarchical multinomial ALR model. Transition totals enter only as each Multinomial denominator. Source-specific coefficients share population priors; participant and event random effects are included. Mean-field ADVI supplies posterior draws, so intervals are approximate and should be checked with NUTS if a publication-critical contrast is borderline.", "",
              "## Planned early-destabilisation / late-consolidation contrasts", "",
              markdown(planned, ["analysis", "outcome", "contrast", "median", "ci95_low", "ci95_high",
                                 "probability_lt_zero", "probability_gt_zero", "probability_in_rope"]), "",
              "The JS equivalence region was pre-set here to +/-0.02 nats. The complete-coverage rows retain only events represented in all five planned preictal bins.", "",
              "## C specificity", "",
              markdown(specificity, ["source_state", "estimand", "median", "ci95_low", "ci95_high",
                                     "probability_gt_zero"]), "",
              f"P(beta_C > mean beta_other) = {c_specific.probability_gt_zero:.3f}. C remains the pre-specified source; specificity is not claimed unless this posterior probability is compelling.", "",
              "## Latent autonomic factors", "",
              "Three orthogonal PCs were defined only from interictal windows and labelled by prespecified cardiac anchors. Lagged tests use within-event fixed effects and participant-clustered standard errors.", "",
              markdown(factor_tests, ["analysis", "estimate", "ci95_low", "ci95_high", "p_value", "n_events"]), "",
              "## Joint dynamic loop comparison", "",
              markdown(loop[loop.outcome.eq("all")], ["model", "rmse", "elpd", "delta_elpd", "se_delta_elpd", "n_events"]), "",
              f"Highest leave-one-event-out ELPD: {best['model']}. Differences smaller than their SE are not distinguishable; this comparison is predictive, not causal.", "",
              "## Pseudo-onset limitations", "",
              f"Pseudo-onsets were optimally assigned within participant, required the same ECG-availability status, and were restricted to the final 20% of each interictal record (median absolute relative-position difference {position_difference:.3f}). ECG artefact magnitude was exported from recovered-beat fractions and included in matching. " +
              "The following were absent from the available exports and were not imputed: " +
              ", ".join(audit["unavailable_in_current_exports"] + audit["other_previously_unavailable"]) + ".", "",
              "## Primary outcomes", "",
              "J is Jensen-Shannon divergence from the participant's interictal transition baseline; K_H = 1 - H(p)/log(5) is route-independent concentration; D_route is Jensen-Shannon route distance after matching the centred-log-ratio concentration of p and baseline. Baseline-oriented kappa is secondary and answers whether the baseline preference is sharpened or flattened.", "",
              "## Interpretation of the planned claim", ""]
    all_events = planned[planned.analysis.eq("all eligible events")]
    early_supported = all_events.iloc[0].probability_lt_zero >= .975
    late_supported = all_events.iloc[1].probability_gt_zero >= .975
    stable_js = all_events.iloc[2].probability_in_rope >= .95
    if early_supported and late_supported and stable_js:
        report.append("The full early-destabilisation/later-consolidation hypothesis is supported.")
    elif early_supported and stable_js:
        report.append("Early lower concentration and stable early-to-final JS are supported, but later consolidation is not established; the full clinical claim should not be made.")
    else:
        report.append("The full early-destabilisation/later-consolidation hypothesis is not supported.")
    (out_dir.parent / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def run(args) -> dict:
    data_dir = args.out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    table = enrich_table(pd.read_csv(args.peak_windows_csv))
    reference_qc = pd.read_csv(args.reference_qc_csv)
    legacy.FIXED_EXITS = EQUALISED_TRANSITIONS
    legacy.raw_count_primary(table, reference_qc, data_dir, args.draws, args.seed + 1)
    valid_participants = legacy.reference_participants(reference_qc, .50)
    valid_participants &= set(table.loc[table.condition.eq("ii"), "participant"].astype(str))
    table = table[table.participant.astype(str).isin(valid_participants)].copy()
    common = add_ecg_artifact(pd.read_csv(args.common_windows_csv), args.native_windows_csv)
    common = common[common.participant.astype(str).isin(valid_participants)].copy()
    common.to_csv(data_dir / "common_windows_with_ecg_artifact.csv.gz", index=False, compression="gzip")
    frame, matches, covariates = prepare_model_windows(table, common)
    matches.to_csv(data_dir / "matched_pseudo_onsets.csv", index=False)
    balance = matching_balance(matches, data_dir)
    frame.to_csv(data_dir / "hierarchical_multinomial_windows.csv.gz", index=False, compression="gzip")
    availability = availability_audit(common, data_dir)

    if args.reuse_posterior:
        _configure_pymc_cache()
        import arviz as az
        posterior = az.from_netcdf(data_dir / "hierarchical_multinomial_posterior.nc")
        long = pd.read_csv(data_dir / "hierarchical_multinomial_observations.csv")
    else:
        posterior, long = fit_hierarchical_multinomial(frame, covariates, data_dir, args.draws,
                                                       args.advi_iterations, args.seed)
    global_means, global_metrics, cells = derive_posterior_metrics(
        posterior, long, frame, covariates, table, data_dir, args.seed + 2)
    planned = planned_contrasts(frame, global_metrics, cells, data_dir)
    spline_trajectory(frame, global_metrics, data_dir, args.seed + 3)
    specificity = source_specificity(cells, data_dir)
    factors = autonomic_factors(common, data_dir)
    dynamic = make_dynamic_frame(factors, global_means)
    dynamic.to_csv(data_dir / "joint_dynamic_windows.csv.gz", index=False, compression="gzip")
    factor_tests = lagged_factor_tests(dynamic, data_dir)
    loop, _ = dynamic_loop_comparison(dynamic, data_dir)
    write_report(data_dir, planned, specificity, factor_tests, loop, availability)

    summary = {"model_windows": len(frame), "multinomial_rows": len(long),
               "participants": int(frame.participant.nunique()), "events": int(frame.event_key.nunique()),
               "pseudo_onset_matches": len(matches),
               "posterior_draws": args.draws, "advi_iterations": args.advi_iterations,
               "equalised_transitions": EQUALISED_TRANSITIONS,
               "complete_coverage_events": int(planned.complete_events.iloc[0]),
               "p_c_specific_gt_mean_other": float(specificity.loc[
                   specificity.source_state.eq("C-specificity"), "probability_gt_zero"].iloc[0]),
               "best_dynamic_model_by_elpd": str(loop[loop.outcome.eq("all")].sort_values("elpd", ascending=False).model.iloc[0])}
    non_position_balance = balance[~balance.matching_variable.eq("record_fraction")]
    summary["pseudo_max_abs_smd_excluding_record_position"] = float(
        non_position_balance.standardised_mean_difference.abs().max())
    loss = pd.read_csv(data_dir / "advi_loss.csv").negative_elbo
    if len(loss) >= 2000:
        recent, previous = loss.iloc[-1000:].mean(), loss.iloc[-2000:-1000].mean()
        summary["advi_tail_relative_change"] = float(abs(recent - previous) / abs(recent))
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    provenance = {"completed_utc": datetime.now(timezone.utc).isoformat(),
                  "microstate_fits_reused": True, "microstate_refit_performed": False,
                  "inputs": {str(path): sha256(path) for path in (args.peak_windows_csv, args.common_windows_csv,
                                                                  args.reference_qc_csv, args.native_windows_csv)
                             if path is not None and path.exists()},
                  "code": {str(path): sha256(path) for path in
                           (Path(__file__), Path(legacy.__file__))},
                  "model": "hierarchical multinomial ALR; source-hierarchical coefficients; participant/event random effects",
                  "inference": "mean-field ADVI", "transition_count_as_predictor": False,
                  "all_source_reference_qc_threshold": 0.50,
                  "primary_outcomes": ["J", "K_H", "D_route"], "secondary_outcome": "kappa",
                  "covariates": covariates,
                  "software": {name: importlib.metadata.version(name) for name in
                               ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "pymc", "arviz")},
                  **summary}
    (args.out_dir / "AUDIT.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return summary


def self_check() -> None:
    rng = np.random.default_rng(4)
    p = rng.dirichlet(np.ones(5), size=(7, 9))
    q = rng.dirichlet(np.ones(5), size=(7, 9))
    metrics = transition_metrics(p, q)
    assert set(metrics) == {"J", "K_H", "D_route", "kappa", "delta_probability"}
    assert np.all(metrics["J"] >= 0) and np.all((metrics["K_H"] >= 0) & (metrics["K_H"] <= 1))
    assert np.all(metrics["D_route"] >= 0) and np.all(metrics["kappa"] > 0)
    assert np.allclose(metrics["delta_probability"].sum(axis=-1), 0)
    counts = np.array([20, 15, 10, 5, 2])
    assert np.all(legacy.thin_counts(rng, counts, EQUALISED_TRANSITIONS, 10).sum(axis=1) == EQUALISED_TRANSITIONS)


def main() -> None:
    source = Path(r"G:\FS_microstates_study\vg_microstates\focused_analysis_20260730_feedback_rawcounts\data")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peak-windows-csv", type=Path, default=source / "directional_peak_backfit_windows.csv")
    parser.add_argument("--common-windows-csv", type=Path, default=source / "autonomic_peak_microstate_common_windows.csv.gz")
    parser.add_argument("--native-windows-csv", type=Path, default=source / "autonomic_native_ii_standardised.csv.gz")
    parser.add_argument("--reference-qc-csv", type=Path, default=source / "state_reference_qc.csv")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(r"C:\Users\Admin\OneDrive - NHS\FS_MS\equalised and optimised"))
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--advi-iterations", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=30072026)
    parser.add_argument("--reuse-posterior", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(); print("optimised transition self-check passed"); return
    required = (args.peak_windows_csv, args.common_windows_csv, args.reference_qc_csv)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("missing inputs: " + ", ".join(missing))
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
