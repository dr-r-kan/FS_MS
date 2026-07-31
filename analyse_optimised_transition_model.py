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
DESTINATION_INDEX = np.array([[destination for destination in range(len(STATES)) if destination != source]
                              for source in range(len(STATES))], dtype=int)
EDGE_LABELS = tuple(f"{STATES[source]}->{STATES[destination]}"
                    for source in range(len(STATES)) for destination in DESTINATION_INDEX[source])


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


def reshape_counts(frame: pd.DataFrame, covariates: list[str], aggregate: bool = False) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows, counts = [], []
    for source in range(len(STATES)):
        destinations = [index for index in range(len(STATES)) if index != source]
        columns = [f"t_{source + 1}_{destination + 1}" for destination in destinations]
        values = frame[columns].to_numpy(int)
        keep = values.sum(axis=1) > 0
        part = frame.loc[keep, ["window_id", "participant", "event_key", *covariates]].copy()
        if aggregate:
            count_names = [f"count_{position}" for position in range(5)]
            part[count_names] = values[keep]
            grouping = ["participant", "event_key", *covariates]
            part = part.groupby(grouping, observed=True, sort=False, as_index=False)[count_names].sum()
            values_part = part[count_names].to_numpy(int)
            part = part.drop(columns=count_names)
            part["window_id"] = -1
        else:
            values_part = values[keep]
        part["source"] = source
        part["total_transitions"] = values_part.sum(axis=1)
        rows.append(part)
        counts.append(values_part)
    long = pd.concat(rows, ignore_index=True)
    participants = pd.Index(frame.participant.unique(), name="participant")
    events = pd.Index(frame.event_key.unique(), name="event")
    long["participant_index"] = participants.get_indexer(long.participant)
    long["event_index"] = events.get_indexer(long.event_key)
    return long, np.vstack(counts), frame[covariates].to_numpy(float)


def build_masked_softmax_model(long: pd.DataFrame, counts: np.ndarray, covariates: list[str],
                               participant_labels, event_labels):
    _configure_pymc_cache()
    import pymc as pm
    import pytensor.tensor as pt

    x = long[covariates].to_numpy(float)
    source = long.source.to_numpy(int)
    participant = long.participant_index.to_numpy(int)
    event = long.event_index.to_numpy(int)
    totals = counts.sum(axis=1).astype(int)
    destinations = DESTINATION_INDEX[source]
    edge_destination = DESTINATION_INDEX.reshape(-1)
    coords = {"source": STATES, "destination": STATES, "edge": EDGE_LABELS,
              "covariate": covariates, "participant": participant_labels, "event": event_labels,
              "observation": np.arange(len(long)), "destination_position": np.arange(5)}
    with pm.Model(coords=coords) as model:
        beta_mean = pm.Normal("beta_mean", 0, 1, dims=("covariate", "destination"))
        beta_sd = pm.HalfNormal("beta_sd", .5, dims=("covariate", "destination"))
        beta_offset = pm.Normal("beta_offset", 0, 1, dims=("edge", "covariate"))
        raw_edge = (beta_mean[:, edge_destination].dimshuffle(1, 0) +
                    beta_sd[:, edge_destination].dimshuffle(1, 0) * beta_offset)
        by_source = raw_edge.reshape((len(STATES), 5, len(covariates)))
        beta_edge = pm.Deterministic("beta_edge",
                                     (by_source - pt.mean(by_source, axis=1, keepdims=True)).reshape((-1, len(covariates))),
                                     dims=("edge", "covariate"))
        fixed = pt.sum(beta_edge.reshape((len(STATES), 5, len(covariates)))[source] * x[:, None, :], axis=2)

        participant_sd = pm.HalfNormal("participant_sd", .5, dims=("source", "destination"))
        participant_offset = pm.Normal("participant_offset", 0, 1, dims=("participant", "destination"))
        participant_full = participant_offset[participant] * participant_sd[source]
        participant_allowed = participant_full[pt.arange(len(long))[:, None], destinations]
        participant_allowed -= pt.mean(participant_allowed, axis=1, keepdims=True)

        event_sd = pm.HalfNormal("event_sd", .35, dims=("source", "destination"))
        event_offset = pm.Normal("event_offset", 0, 1, dims=("event", "destination"))
        event_full = event_offset[event] * event_sd[source]
        event_allowed = event_full[pt.arange(len(long))[:, None], destinations]
        event_allowed -= pt.mean(event_allowed, axis=1, keepdims=True)

        probability = pm.math.softmax(fixed + participant_allowed + event_allowed, axis=1)
        pm.Multinomial("transition_counts", n=totals, p=probability, observed=counts,
                       dims=("observation", "destination_position"))
    return model


def fit_hierarchical_multinomial(frame: pd.DataFrame, covariates: list[str], out_dir: Path,
                                 draws: int, tune: int, chains: int, cores: int,
                                 target_accept: float, seed: int):
    _configure_pymc_cache()
    import arviz as az
    import pymc as pm

    long, counts, _ = reshape_counts(frame, covariates, aggregate=True)
    participants = frame.participant.unique()
    events = frame.event_key.unique()
    model = build_masked_softmax_model(long, counts, covariates, participants, events)
    with model:
        posterior = pm.sample(draws=draws, tune=tune, chains=chains, cores=min(cores, chains),
                              target_accept=target_accept, random_seed=seed, progressbar=True,
                              compute_convergence_checks=False,
                              idata_kwargs={"log_likelihood": False}, return_inferencedata=True)
    posterior.posterior.attrs.update({"likelihood": "Multinomial raw transition counts",
                                      "parameterisation": "destination-labelled symmetric sum-to-zero masked softmax",
                                      "inference": "multi-chain NUTS", "tune": tune,
                                      "chains": chains, "target_accept": target_accept,
                                      "transition_count_role": "multinomial denominator only"})
    az.to_netcdf(posterior, out_dir / "hierarchical_multinomial_posterior.nc")
    long.to_csv(out_dir / "hierarchical_multinomial_observations.csv", index=False)
    pd.DataFrame({"edge": EDGE_LABELS,
                  "source_state": np.repeat(STATES, 5),
                  "destination": np.array(STATES)[DESTINATION_INDEX.reshape(-1)]}) \
        .to_csv(out_dir / "hierarchical_edge_map.csv", index=False)

    diagnostics = az.summary(posterior, var_names=["beta_edge", "beta_mean", "beta_sd",
                                                    "participant_offset", "participant_sd",
                                                    "event_offset", "event_sd"], kind="diagnostics")
    diagnostics.rename_axis("parameter").reset_index().to_csv(out_dir / "nuts_diagnostics.csv", index=False)
    sample_stats = posterior.sample_stats
    confirmatory = diagnostics[~diagnostics.index.str.startswith(("participant_offset", "event_offset"))]
    fit_diagnostics = {"chains": chains, "draws_per_chain": draws, "tune_per_chain": tune,
                       "target_accept": target_accept,
                       "divergences": int(sample_stats["diverging"].sum()),
                       "max_r_hat": float(confirmatory.r_hat.max()),
                       "min_ess_bulk": float(confirmatory.ess_bulk.min()),
                       "min_ess_tail": float(confirmatory.ess_tail.min()),
                       "all_parameter_max_r_hat": float(diagnostics.r_hat.max()),
                       "all_parameter_min_ess_bulk": float(diagnostics.ess_bulk.min())}
    if "tree_depth" in sample_stats:
        fit_diagnostics["max_tree_depth"] = int(sample_stats["tree_depth"].max())
    fit_diagnostics["confirmatory_thresholds_met"] = bool(
        fit_diagnostics["divergences"] == 0 and fit_diagnostics["max_r_hat"] <= 1.01 and
        fit_diagnostics["min_ess_bulk"] >= 400)
    (out_dir / "confirmatory_fit_diagnostics.json").write_text(
        json.dumps(fit_diagnostics, indent=2), encoding="utf-8")
    return posterior, long, counts, fit_diagnostics


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


def matched_pseudo_estimands(values: np.ndarray, meta: pd.DataFrame, source_state: str,
                             outcome: str) -> list[dict]:
    columns = np.arange(len(meta))
    genuine = meta.loc[meta.true_onset.eq(1) & meta.window_end_to_ictus_s.eq(0),
                       ["participant", "event_key", "is_functional"]].copy()
    genuine["genuine_column"] = columns[meta.true_onset.eq(1) & meta.window_end_to_ictus_s.eq(0)]
    pseudo = meta.loc[meta.true_onset.eq(0), ["participant", "event_key", "is_functional"]].copy()
    pseudo["pseudo_column"] = columns[meta.true_onset.eq(0)]
    pairs = genuine.merge(pseudo, on=["participant", "event_key", "is_functional"], how="inner",
                           validate="one_to_one")
    if pairs.empty:
        return []

    def participant_mean(part: pd.DataFrame, left: str, right: str | None = None) -> np.ndarray:
        participant_draws = []
        for _, cells in part.groupby("participant", observed=True, sort=False):
            draw = values[:, cells[left].to_numpy(int)]
            if right is not None:
                draw = draw - values[:, cells[right].to_numpy(int)]
            participant_draws.append(draw.mean(axis=1))
        return np.mean(participant_draws, axis=0)

    group_draws = {}
    rows = []
    for group, functional in (("ES", 0), ("FS", 1)):
        part = pairs[pairs.is_functional.eq(functional)]
        if part.empty:
            continue
        genuine_draw = participant_mean(part, "genuine_column")
        pseudo_draw = participant_mean(part, "pseudo_column")
        change = participant_mean(part, "genuine_column", "pseudo_column")
        group_draws[group] = {"genuine": genuine_draw, "pseudo": pseudo_draw, "change": change}
        row = summarise(change, source_state=source_state, outcome=outcome,
                        contrast=f"{group} genuine minus matched pseudo")
        row.update({"n_pairs": len(part), "n_participants": part.participant.nunique(), "matched": True})
        rows.append(row)
    if set(group_draws) == {"ES", "FS"}:
        contrasts = {
            "FS minus ES at genuine onset": group_draws["FS"]["genuine"] - group_draws["ES"]["genuine"],
            "FS minus ES at matched pseudo-onset": group_draws["FS"]["pseudo"] - group_draws["ES"]["pseudo"],
            "matched difference-in-differences": group_draws["FS"]["change"] - group_draws["ES"]["change"],
        }
        for label, draw in contrasts.items():
            row = summarise(draw, source_state=source_state, outcome=outcome, contrast=label)
            row.update({"n_pairs": len(pairs), "n_participants": pairs.participant.nunique(), "matched": True})
            rows.append(row)
    return rows


def derive_posterior_metrics(posterior, long: pd.DataFrame, frame: pd.DataFrame,
                             covariates: list[str], baseline_table: pd.DataFrame,
                             out_dir: Path, seed: int) -> tuple[pd.DataFrame, dict, dict, pd.DataFrame]:
    beta_edge = posterior_array(posterior, "beta_edge")
    participant_offset = posterior_array(posterior, "participant_offset")
    participant_sd = posterior_array(posterior, "participant_sd")
    event_offset = posterior_array(posterior, "event_offset")
    event_sd = posterior_array(posterior, "event_sd")
    n_draws = beta_edge.shape[0]
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
    rows, destination_rows, did_rows, cells = [], [], [], {}
    for source in range(len(STATES)):
        use = long.source.eq(source).to_numpy()
        meta = long.loc[use].reset_index(drop=True)
        x = meta[covariates].to_numpy(float)
        beta = beta_edge[:, source * 5:(source + 1) * 5, :]
        participant_full = (participant_offset[:, meta.participant_index.to_numpy(int), :] *
                            participant_sd[:, None, source, :])
        participant_allowed = participant_full[:, :, DESTINATION_INDEX[source]]
        participant_allowed -= participant_allowed.mean(axis=2, keepdims=True)
        event_full = (event_offset[:, meta.event_index.to_numpy(int), :] *
                      event_sd[:, None, source, :])
        event_allowed = event_full[:, :, DESTINATION_INDEX[source]]
        event_allowed -= event_allowed.mean(axis=2, keepdims=True)
        eta = np.einsum("skc,oc->sok", beta, x) + participant_allowed + event_allowed
        probability = softmax_last_reference(eta[..., :-1] - eta[..., -1, None])
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
        for name in metric_names:
            did_rows.extend(matched_pseudo_estimands(metrics[name], window_meta, STATES[source], name))
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
        did_rows.extend(matched_pseudo_estimands(global_metrics[name], frame.reset_index(drop=True), "Global", name))
    global_means.to_csv(out_dir / "posterior_global_transition_window_means.csv.gz", index=False, compression="gzip")
    did = pd.DataFrame(did_rows)
    did.to_csv(out_dir / "matched_pseudo_difference_in_differences.csv", index=False)
    return global_means, global_metrics, cells, did


def posterior_predictive_checks(posterior, long: pd.DataFrame, counts: np.ndarray,
                                frame: pd.DataFrame, covariates: list[str], out_dir: Path,
                                seed: int, max_draws: int = 200) -> dict:
    beta_edge = posterior_array(posterior, "beta_edge")
    participant_offset = posterior_array(posterior, "participant_offset")
    participant_sd = posterior_array(posterior, "participant_sd")
    event_offset = posterior_array(posterior, "event_offset")
    event_sd = posterior_array(posterior, "event_sd")
    selected = np.linspace(0, len(beta_edge) - 1, min(max_draws, len(beta_edge)), dtype=int)
    beta_edge, participant_offset, participant_sd, event_offset, event_sd = (
        value[selected] for value in (beta_edge, participant_offset, participant_sd,
                                      event_offset, event_sd))
    rng = np.random.default_rng(seed)
    predictive_rows, discrepancy_rows = [], []
    covered = []
    for source in range(len(STATES)):
        use = long.source.eq(source).to_numpy()
        meta = long.loc[use].reset_index(drop=True)
        window_meta = frame.set_index("window_id").loc[meta.window_id].reset_index()
        observed = counts[use]
        totals = observed.sum(axis=1).astype(int)
        x = meta[covariates].to_numpy(float)
        beta = beta_edge[:, source * 5:(source + 1) * 5, :]
        participant_full = (participant_offset[:, meta.participant_index.to_numpy(int), :] *
                            participant_sd[:, None, source, :])
        participant_allowed = participant_full[:, :, DESTINATION_INDEX[source]]
        participant_allowed -= participant_allowed.mean(axis=2, keepdims=True)
        event_full = event_offset[:, meta.event_index.to_numpy(int), :] * event_sd[:, None, source, :]
        event_allowed = event_full[:, :, DESTINATION_INDEX[source]]
        event_allowed -= event_allowed.mean(axis=2, keepdims=True)
        eta = np.einsum("skc,oc->sok", beta, x) + participant_allowed + event_allowed
        eta -= eta.max(axis=2, keepdims=True)
        probability = np.exp(eta); probability /= probability.sum(axis=2, keepdims=True)

        observed_discrepancy, replicated_discrepancy = [], []
        cell_masks = {}
        for group, functional in (("ES", 0), ("FS", 1)):
            for window_type, mask in (
                    ("matched pseudo-onset", window_meta.true_onset.eq(0).to_numpy()),
                    ("early genuine", window_meta.time_bin.eq(TIME_BINS[0][2]).to_numpy()),
                    ("final-minute genuine", window_meta.time_bin.eq(TIME_BINS[-1][2]).to_numpy())):
                keep = mask & window_meta.is_functional.eq(functional).to_numpy()
                if keep.any():
                    cell_masks[(group, window_type)] = keep
        cell_draws = {key: [] for key in cell_masks}
        for draw in range(len(selected)):
            expected = totals[:, None] * probability[draw]
            replicated = rng.multinomial(totals, probability[draw])
            observed_discrepancy.append(float(np.sum((observed - expected) ** 2 / (expected + .5))))
            replicated_discrepancy.append(float(np.sum((replicated - expected) ** 2 / (expected + .5))))
            for key, keep in cell_masks.items():
                aggregate = replicated[keep].sum(axis=0)
                cell_draws[key].append(aggregate / aggregate.sum())
        observed_discrepancy = np.asarray(observed_discrepancy)
        replicated_discrepancy = np.asarray(replicated_discrepancy)
        discrepancy_rows.append({"source_state": STATES[source], "draws": len(selected),
                                 "bayesian_p_value": float(np.mean(replicated_discrepancy >= observed_discrepancy)),
                                 "observed_discrepancy_mean": float(observed_discrepancy.mean()),
                                 "replicated_discrepancy_mean": float(replicated_discrepancy.mean())})
        for (group, window_type), draws in cell_draws.items():
            keep = cell_masks[(group, window_type)]
            observed_proportion = observed[keep].sum(axis=0) / observed[keep].sum()
            draws = np.asarray(draws)
            low, median, high = np.quantile(draws, (.025, .5, .975), axis=0)
            for position, destination in enumerate(DESTINATION_INDEX[source]):
                in_interval = bool(low[position] <= observed_proportion[position] <= high[position])
                covered.append(in_interval)
                predictive_rows.append({"source_state": STATES[source], "destination": STATES[destination],
                                        "group": group, "window_type": window_type,
                                        "observed_proportion": float(observed_proportion[position]),
                                        "predictive_median": float(median[position]),
                                        "predictive_ci95_low": float(low[position]),
                                        "predictive_ci95_high": float(high[position]),
                                        "observed_in_predictive_interval": in_interval,
                                        "ppc_draws": len(selected)})
    pd.DataFrame(predictive_rows).to_csv(out_dir / "posterior_predictive_checks.csv", index=False)
    pd.DataFrame(discrepancy_rows).to_csv(out_dir / "posterior_predictive_discrepancy.csv", index=False)
    summary = {"ppc_draws": len(selected), "cell_interval_coverage": float(np.mean(covered)),
               "min_source_bayesian_p_value": float(pd.DataFrame(discrepancy_rows).bayesian_p_value.min()),
               "max_source_bayesian_p_value": float(pd.DataFrame(discrepancy_rows).bayesian_p_value.max())}
    return summary


def simulation_recovery(out_dir: Path, draws: int, tune: int, chains: int, seed: int) -> dict:
    _configure_pymc_cache()
    import arviz as az
    import pymc as pm

    rng = np.random.default_rng(seed)
    covariates = ["intercept", "group_fs", "true_onset", "group_x_true"]
    participants = np.array([f"sim_p{index:02d}" for index in range(12)])
    events = np.array([f"sim_e{index:02d}" for index in range(24)])
    windows = []
    for event_index, event in enumerate(events):
        participant_index = event_index % len(participants)
        group = float(event_index % 2)
        for genuine in (0.0, 1.0):
            windows.append({"participant": participants[participant_index], "event_key": event,
                            "participant_index": participant_index, "event_index": event_index,
                            "intercept": 1.0, "group_fs": group, "true_onset": genuine,
                            "group_x_true": group * genuine})
    windows = pd.DataFrame(windows)
    true_beta = np.zeros((len(STATES), 5, len(covariates)))
    destination_signal = np.array([-.45, -.25, -.05, .15, .35, .55])
    for source in range(len(STATES)):
        true_beta[source, :, 0] = np.linspace(-.35, .35, 5)
        signal = destination_signal[DESTINATION_INDEX[source]]
        true_beta[source, :, 3] = signal - signal.mean()
    participant_effect = rng.normal(0, .12, (len(participants), len(STATES)))
    event_effect = rng.normal(0, .08, (len(events), len(STATES)))
    rows, counts = [], []
    for source in range(len(STATES)):
        x = windows[covariates].to_numpy(float)
        eta = np.einsum("kc,oc->ok", true_beta[source], x)
        p_effect = participant_effect[windows.participant_index.to_numpy(int)][:, DESTINATION_INDEX[source]]
        e_effect = event_effect[windows.event_index.to_numpy(int)][:, DESTINATION_INDEX[source]]
        eta += p_effect - p_effect.mean(axis=1, keepdims=True)
        eta += e_effect - e_effect.mean(axis=1, keepdims=True)
        eta -= eta.max(axis=1, keepdims=True)
        probability = np.exp(eta); probability /= probability.sum(axis=1, keepdims=True)
        part = windows.copy(); part["source"] = source; part["window_id"] = np.arange(len(windows))
        part["total_transitions"] = 30; rows.append(part)
        counts.append(rng.multinomial(30, probability))
    long = pd.concat(rows, ignore_index=True)
    counts = np.vstack(counts)
    model = build_masked_softmax_model(long, counts, covariates, participants, events)
    with model:
        posterior = pm.sample(draws=draws, tune=tune, chains=chains, cores=1, target_accept=.9,
                              random_seed=seed + 1, progressbar=True, compute_convergence_checks=False,
                              idata_kwargs={"log_likelihood": False}, return_inferencedata=True)
    recovered = posterior_array(posterior, "beta_edge").reshape((-1, len(STATES), 5, len(covariates)))[:, :, :, 3]
    low, median, high = np.quantile(recovered, (.025, .5, .975), axis=0)
    rows = []
    for source in range(len(STATES)):
        for position, destination in enumerate(DESTINATION_INDEX[source]):
            truth = true_beta[source, position, 3]
            rows.append({"source_state": STATES[source], "destination": STATES[destination],
                         "covariate": "group_x_true", "truth": truth,
                         "posterior_median": median[source, position],
                         "ci95_low": low[source, position], "ci95_high": high[source, position],
                         "covered": bool(low[source, position] <= truth <= high[source, position])})
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "simulation_recovery.csv", index=False)
    diagnostic = az.summary(posterior, var_names=["beta_edge"], kind="diagnostics")
    summary = {"simulation_chains": chains, "simulation_draws_per_chain": draws,
               "simulation_tune_per_chain": tune,
               "coefficient_correlation": float(np.corrcoef(result.truth, result.posterior_median)[0, 1]),
               "coefficient_rmse": float(np.sqrt(np.mean((result.truth - result.posterior_median) ** 2))),
               "interval_coverage": float(result.covered.mean()),
               "max_r_hat": float(diagnostic.r_hat.max()),
               "min_ess_bulk": float(diagnostic.ess_bulk.min()),
               "divergences": int(posterior.sample_stats["diverging"].sum())}
    (out_dir / "simulation_recovery_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


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
    rows = [summarise(draw, source_state=state, outcome="J", estimand="final-minute FS minus ES in J")
            for state, draw in effects.items()]
    difference = effects["C"] - np.mean([effects[state] for state in STATES if state != "C"], axis=0)
    rows.append(summarise(difference, source_state="C-specificity", outcome="J",
                          estimand="C-source J effect minus mean other-source J effect"))
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
                 did: pd.DataFrame, factor_tests: pd.DataFrame, loop: pd.DataFrame,
                 audit: dict, fit_diagnostics: dict, ppc: dict, recovery: dict) -> None:
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
    headline_did = did[(did.source_state.eq("Global")) & did.outcome.eq("J") &
                       did.contrast.eq("matched difference-in-differences")].iloc[0]
    balance = pd.read_csv(out_dir / "pseudo_onset_match_balance.csv").set_index("matching_variable")
    position_difference = balance.loc["record_fraction", "median_absolute_paired_difference"]
    report = ["# Equalised and optimised transition analysis", "",
              f"Completed: {datetime.now(timezone.utc).isoformat()}", "",
              "## Model", "",
              "Raw transition counts were fitted with a destination-labelled hierarchical multinomial masked-softmax model. For every source, the five allowed destination logits are constrained to sum to zero; the self-transition is structurally excluded. Population shrinkage is indexed by destination label rather than by a source-dependent coordinate position. Participant and event random destination effects are included, and transition totals enter only as each Multinomial denominator.", "",
              f"The confirmatory fit used {fit_diagnostics['chains']} NUTS chains with {fit_diagnostics['draws_per_chain']} retained draws per chain after {fit_diagnostics['tune_per_chain']} tuning iterations. Divergences={fit_diagnostics['divergences']}; max R-hat={fit_diagnostics['max_r_hat']:.3f}; minimum bulk ESS={fit_diagnostics['min_ess_bulk']:.0f}.", "",
              "## Matched pseudo-onset difference-in-differences", "",
              "The onset-specific estimand is [FS genuine - FS matched pseudo] - [ES genuine - ES matched pseudo]. Genuine and pseudo windows are paired within the same participant and event match before participant-equal aggregation.", "",
              markdown(did[(did.outcome.eq("J")) &
                           ((did.source_state.eq("Global")) |
                            did.contrast.eq("matched difference-in-differences"))],
                       ["source_state", "contrast", "median", "ci95_low", "ci95_high",
                        "probability_lt_zero", "probability_gt_zero", "n_pairs", "n_participants"]), "",
              f"Global J difference-in-differences: median {headline_did['median']:.4g} "
              f"(95% CrI {headline_did.ci95_low:.4g} to {headline_did.ci95_high:.4g}); "
              f"P(>0)={headline_did.probability_gt_zero:.3f}.", "",
              "## Planned early-destabilisation / late-consolidation contrasts", "",
              markdown(planned, ["analysis", "outcome", "contrast", "median", "ci95_low", "ci95_high",
                                 "probability_lt_zero", "probability_gt_zero", "probability_in_rope"]), "",
              "The JS equivalence region was pre-set here to +/-0.02 nats. The complete-coverage rows retain only events represented in all five planned preictal bins.", "",
              "## C specificity", "",
              markdown(specificity, ["source_state", "estimand", "median", "ci95_low", "ci95_high",
                                     "probability_gt_zero"]), "",
              f"P(C-source J effect > mean other-source J effect) = {c_specific.probability_gt_zero:.3f}. C remains the pre-specified source; specificity is not claimed unless this posterior probability is compelling.", "",
              "## Model validation", "",
              f"Posterior predictive 95% interval coverage across source-destination/group/window cells was {ppc['cell_interval_coverage']:.3f}; source-level Bayesian discrepancy p-values ranged from {ppc['min_source_bayesian_p_value']:.3f} to {ppc['max_source_bayesian_p_value']:.3f}.",
              f"Simulation recovery for the destination-labelled genuine-onset interaction gave coefficient correlation {recovery['coefficient_correlation']:.3f}, RMSE {recovery['coefficient_rmse']:.3f}, and 95% interval coverage {recovery['interval_coverage']:.3f}.", "",
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
    report.extend(["", "## Onset-specific interpretation", ""])
    if headline_did.probability_gt_zero >= .975 or headline_did.probability_lt_zero >= .975:
        report.append("The matched difference-in-differences supports an onset-specific transition effect beyond the broader FS-associated phenotype.")
    else:
        report.append("The matched difference-in-differences does not establish onset specificity. The result supports an FS-associated transition phenotype, not an onset-specific biomarker.")
    (out_dir.parent / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def run(args) -> dict:
    data_dir = args.out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    table = enrich_table(pd.read_csv(args.peak_windows_csv))
    reference_qc = pd.read_csv(args.reference_qc_csv)
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
        if "destination-labelled" not in posterior.posterior.attrs.get("parameterisation", ""):
            raise ValueError("The saved posterior uses the obsolete source-position ALR parameterisation")
        long = pd.read_csv(data_dir / "hierarchical_multinomial_observations.csv")
        fit_diagnostics = json.loads((data_dir / "confirmatory_fit_diagnostics.json").read_text(encoding="utf-8"))
    else:
        posterior, long, counts, fit_diagnostics = fit_hierarchical_multinomial(
            frame, covariates, data_dir, args.draws, args.tune, args.chains, args.cores,
            args.target_accept, args.seed)
    prediction_long, prediction_counts, _ = reshape_counts(frame, covariates)
    global_means, global_metrics, cells, did = derive_posterior_metrics(
        posterior, prediction_long, frame, covariates, table, data_dir, args.seed + 2)
    ppc = posterior_predictive_checks(posterior, prediction_long, prediction_counts, frame, covariates, data_dir,
                                      args.seed + 3, args.ppc_draws)
    recovery = simulation_recovery(data_dir, args.recovery_draws, args.recovery_tune,
                                   args.recovery_chains, args.seed + 4)
    planned = planned_contrasts(frame, global_metrics, cells, data_dir)
    spline_trajectory(frame, global_metrics, data_dir, args.seed + 5)
    specificity = source_specificity(cells, data_dir)
    factors = autonomic_factors(common, data_dir)
    dynamic = make_dynamic_frame(factors, global_means)
    dynamic.to_csv(data_dir / "joint_dynamic_windows.csv.gz", index=False, compression="gzip")
    factor_tests = lagged_factor_tests(dynamic, data_dir)
    loop, _ = dynamic_loop_comparison(dynamic, data_dir)
    write_report(data_dir, planned, specificity, did, factor_tests, loop, availability,
                 fit_diagnostics, ppc, recovery)

    summary = {"model_windows": len(frame), "multinomial_rows": len(long),
               "participants": int(frame.participant.nunique()), "events": int(frame.event_key.nunique()),
               "pseudo_onset_matches": len(matches),
               "posterior_draws_total": args.draws * args.chains, "chains": args.chains,
               "draws_per_chain": args.draws, "tune_per_chain": args.tune,
               "legacy_equalised_sensitivity_transitions": EQUALISED_TRANSITIONS,
               "legacy_equalised_sensitivity_run": False,
               "complete_coverage_events": int(planned.complete_events.iloc[0]),
               "p_c_specific_gt_mean_other": float(specificity.loc[
                   specificity.source_state.eq("C-specificity"), "probability_gt_zero"].iloc[0]),
               "best_dynamic_model_by_elpd": str(loop[loop.outcome.eq("all")].sort_values("elpd", ascending=False).model.iloc[0])}
    headline_did = did[(did.source_state.eq("Global")) & did.outcome.eq("J") &
                       did.contrast.eq("matched difference-in-differences")].iloc[0]
    summary.update({"matched_j_did_median": float(headline_did["median"]),
                    "matched_j_did_ci95_low": float(headline_did.ci95_low),
                    "matched_j_did_ci95_high": float(headline_did.ci95_high),
                    "matched_j_did_probability_gt_zero": float(headline_did.probability_gt_zero),
                    **{f"nuts_{key}": value for key, value in fit_diagnostics.items()},
                    **{f"ppc_{key}": value for key, value in ppc.items()},
                    **{f"recovery_{key}": value for key, value in recovery.items()}})
    non_position_balance = balance[~balance.matching_variable.eq("record_fraction")]
    summary["pseudo_max_abs_smd_excluding_record_position"] = float(
        non_position_balance.standardised_mean_difference.abs().max())
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    provenance = {"completed_utc": datetime.now(timezone.utc).isoformat(),
                  "microstate_fits_reused": True, "microstate_refit_performed": False,
                  "inputs": {str(path): sha256(path) for path in (args.peak_windows_csv, args.common_windows_csv,
                                                                  args.reference_qc_csv, args.native_windows_csv)
                             if path is not None and path.exists()},
                  "code": {str(path): sha256(path) for path in
                           (Path(__file__), Path(legacy.__file__))},
                  "model": "destination-labelled hierarchical multinomial masked softmax; symmetric sum-to-zero allowed-destination logits; participant/event random effects",
                  "inference": "multi-chain NUTS", "transition_count_as_predictor": False,
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
    assert DESTINATION_INDEX.shape == (6, 5) and len(set(EDGE_LABELS)) == 30
    assert all(source not in DESTINATION_INDEX[source] for source in range(6))
    probability = rng.dirichlet(np.ones(5), size=(3, 4))
    replicated = rng.multinomial(np.array([10, 20, 30, 40]), probability)
    assert replicated.shape == (3, 4, 5) and np.all(replicated.sum(axis=2) == np.array([10, 20, 30, 40]))
    meta = pd.DataFrame({"participant": ["e1", "e2", "f1", "f2"] * 2,
                         "event_key": ["e1", "e2", "f1", "f2"] * 2,
                         "is_functional": [0, 0, 1, 1] * 2,
                         "true_onset": [1] * 4 + [0] * 4,
                         "window_end_to_ictus_s": [0] * 4 + [np.nan] * 4})
    values = np.tile(np.array([1.5, 2.0, 4.0, 5.0, 1.0, 1.0, 2.0, 2.0]), (20, 1))
    did = matched_pseudo_estimands(values, meta, "Global", "J")
    headline = next(row for row in did if row["contrast"] == "matched difference-in-differences")
    assert np.isclose(headline["median"], 1.75)


def main() -> None:
    source = Path(r"G:\FS_microstates_study\vg_microstates\focused_analysis_20260730_feedback_rawcounts\data")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peak-windows-csv", type=Path, default=source / "directional_peak_backfit_windows.csv")
    parser.add_argument("--common-windows-csv", type=Path, default=source / "autonomic_peak_microstate_common_windows.csv.gz")
    parser.add_argument("--native-windows-csv", type=Path, default=source / "autonomic_native_ii_standardised.csv.gz")
    parser.add_argument("--reference-qc-csv", type=Path, default=source / "state_reference_qc.csv")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(r"C:\Users\Admin\OneDrive - NHS\FS_MS\equalised and optimised"))
    parser.add_argument("--draws", type=int, default=750)
    parser.add_argument("--tune", type=int, default=750)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--target-accept", type=float, default=.92)
    parser.add_argument("--ppc-draws", type=int, default=200)
    parser.add_argument("--recovery-draws", type=int, default=300)
    parser.add_argument("--recovery-tune", type=int, default=300)
    parser.add_argument("--recovery-chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=30072026)
    parser.add_argument("--reuse-posterior", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(); print("optimised transition self-check passed"); return
    if args.chains < 2 or args.draws < 250 or args.tune < 250:
        parser.error("confirmatory inference requires at least 2 chains, 250 draws and 250 tuning iterations")
    if args.recovery_chains < 2:
        parser.error("simulation recovery requires at least 2 chains")
    required = (args.peak_windows_csv, args.common_windows_csv, args.reference_qc_csv)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("missing inputs: " + ", ".join(missing))
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
