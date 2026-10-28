"""Global-first FS/ES transition, autonomic, manifold, and classification analysis.

The script reuses K=6 peak-backfit transition counts.  It does not fit or relabel
microstates.  Headline quantities pool all six source grammars; source-specific
results are explicitly exploratory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, minimize_scalar
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler
import statsmodels.formula.api as smf

try:
    from .analyse_action_conditioned_transition_tensor import (
        ActionTensorConfig,
        run_analysis as run_action_tensor_analysis,
        run_sensitivities as run_action_tensor_sensitivities,
    )
except ImportError:
    from analyse_action_conditioned_transition_tensor import (
        ActionTensorConfig,
        run_analysis as run_action_tensor_analysis,
        run_sensitivities as run_action_tensor_sensitivities,
    )


STATES = ("F", "E", "G", "B", "A", "C")
TIME_BINS = ("-20 to -10", "-10 to -5", "-5 to -2", "-2 to -1", "-1 to 0")
TIME_ORDER = ("II", *TIME_BINS)
MIN_TRANSITIONS = 30
FS_COLOR = "#0072B2"
ES_COLOR = "#D55E00"
GREY = "#667085"
GREEN = "#009E73"
PURPLE = "#7A5195"
COUNT_COLUMNS = tuple(f"t_{source}_{destination}" for source in range(1, 7)
                      for destination in range(1, 7))
MODEL_FEATURES = ("log_js", "log_kappa", "csi_z", "cpi_z")
FEATURE_LABELS = {"log_js": "log JS", "log_kappa": "log κ", "csi_z": "CSI", "cpi_z": "CPI"}


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
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_figure(fig, directory: Path, stem: str) -> None:
    fig.savefig(directory / f"{stem}.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel(axis, label: str) -> None:
    axis.text(-.12, 1.04, label, transform=axis.transAxes, fontsize=11,
              fontweight="bold", ha="left", va="bottom")


def normalise_path(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace("/", "\\", regex=False).str.lower()


def bh(pvalues: pd.Series) -> pd.Series:
    values = pd.to_numeric(pvalues, errors="coerce").to_numpy(float)
    output = np.full(len(values), np.nan)
    good = np.flatnonzero(np.isfinite(values))
    if not len(good):
        return pd.Series(output, index=pvalues.index)
    order = good[np.argsort(values[good])]
    adjusted = values[order] * len(order) / np.arange(1, len(order) + 1)
    output[order] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    return pd.Series(output, index=pvalues.index)


def enrich_windows(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["row_id"] = np.arange(len(output))
    output["event_type"] = output.seizure_type.astype(str).str.upper().map(
        {"FS": "functional", "ES": "epileptic"})
    output["semiology"] = output.hypo_hyper.astype(str).str.lower().replace({"nan": np.nan, "": np.nan})
    output["event_key"] = output.file_path.astype(str)
    output["file_key"] = normalise_path(output.file_path)
    output["time_bin"] = np.where(output.condition.eq("ii"), "II", "")
    endpoint = pd.to_numeric(output.window_end_to_ictus_s, errors="coerce")
    for low, high, label in ((600, 1200, TIME_BINS[0]), (300, 600, TIME_BINS[1]),
                             (120, 300, TIME_BINS[2]), (60, 120, TIME_BINS[3]),
                             (0, 60, TIME_BINS[4])):
        output.loc[output.condition.eq("preict") & endpoint.between(low, high, inclusive="left"), "time_bin"] = label
    return output


def probability(counts: np.ndarray) -> np.ndarray:
    value = counts.astype(float) + .5
    np.fill_diagonal(value, 0)
    denominator = value.sum(axis=1, keepdims=True)
    return np.divide(value, denominator, out=np.zeros_like(value), where=denominator > 0)


def js_divergence(counts: np.ndarray, baseline: np.ndarray, origins: tuple[int, ...]) -> float:
    total = weighted = 0.0
    for source in origins:
        allowed = np.arange(6) != source
        n = float(counts[source, allowed].sum())
        if n <= 0:
            continue
        p = (counts[source, allowed] + .5) / (n + 2.5)
        q = baseline[source, allowed] / baseline[source, allowed].sum()
        midpoint = (p + q) / 2
        value = .5 * np.sum(p * np.log(p / midpoint)) + .5 * np.sum(q * np.log(q / midpoint))
        weighted += n * value
        total += n
    return weighted / total if total else np.nan


def fit_log_kappa(counts: np.ndarray, baseline: np.ndarray, origins: tuple[int, ...], minimum: int) -> float:
    if sum(counts[source].sum() for source in origins) < minimum:
        return np.nan

    def objective(log_kappa: float) -> float:
        kappa = math.exp(log_kappa)
        value = 0.0
        for source in origins:
            allowed = np.arange(6) != source
            base = np.clip(baseline[source, allowed], 1e-12, 1)
            q = base ** kappa
            q /= q.sum()
            value -= float(counts[source, allowed] @ np.log(np.clip(q, 1e-12, 1)))
        return value

    fitted = minimize_scalar(objective, bounds=(-4, 4), method="bounded")
    return float(fitted.x) if fitted.success else np.nan


def transition_metrics(windows: pd.DataFrame, minimum: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cubes = windows.loc[:, COUNT_COLUMNS].to_numpy(float).reshape(-1, 6, 6)
    diagonal = np.arange(6)
    cubes[:, diagonal, diagonal] = 0
    totals = {str(participant): cubes[index].sum(axis=0)
              for participant, index in windows[windows.condition.eq("ii")].groupby("participant", observed=True).groups.items()}
    global_rows, state_rows = [], []
    metadata = ("row_id", "participant", "file_path", "file_key", "event_key", "condition", "event_type",
                "semiology", "time_bin", "window_index", "window_start_s", "window_end_s",
                "window_end_to_ictus_s", "group", "seizure_type")
    for position, row in enumerate(windows.itertuples(index=False)):
        participant = str(row.participant)
        pooled = totals.get(participant)
        if pooled is None:
            continue
        counts = cubes[position]
        reference_counts = pooled - counts if row.condition == "ii" else pooled
        if reference_counts.sum() < minimum or counts.sum() < minimum:
            continue
        baseline = probability(reference_counts)
        meta = {name: getattr(row, name) for name in metadata}
        log_kappa = fit_log_kappa(counts, baseline, tuple(range(6)), minimum)
        global_rows.append(meta | {
            "global_js": js_divergence(counts, baseline, tuple(range(6))),
            "log_kappa": log_kappa, "kappa": math.exp(log_kappa),
            "total_transitions": int(counts.sum()), "baseline_transitions": int(reference_counts.sum()),
        })
        for source, label in enumerate(STATES):
            outgoing = int(counts[source].sum())
            baseline_outgoing = int(reference_counts[source].sum())
            if outgoing < minimum or baseline_outgoing < minimum:
                continue
            source_log_kappa = fit_log_kappa(counts, baseline, (source,), minimum)
            state_rows.append(meta | {
                "source_state": label, "source_js": js_divergence(counts, baseline, (source,)),
                "source_log_kappa": source_log_kappa, "source_kappa": math.exp(source_log_kappa),
                "outgoing_transitions": outgoing, "baseline_outgoing_transitions": baseline_outgoing,
            })
    global_frame = pd.DataFrame(global_rows)
    state_frame = pd.DataFrame(state_rows)
    assert global_frame.row_id.is_unique and set(state_frame.source_state).issubset(STATES)
    return global_frame, state_frame


def merge_autonomic(metrics: pd.DataFrame, common: pd.DataFrame) -> pd.DataFrame:
    keys = ["participant", "file_key", "window_start_s", "window_end_s"]
    autonomic = common.copy()
    autonomic["file_key"] = normalise_path(autonomic.file_path)
    columns = keys + ["csi_z", "cpi_z", "n_native"]
    autonomic = autonomic[columns].drop_duplicates(keys)
    return metrics.merge(autonomic, on=keys, how="left", validate="many_to_one")


def participant_cells(global_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    preictal = global_frame[
        global_frame.condition.eq("preict") & global_frame.time_bin.isin(TIME_BINS) &
        global_frame.event_type.isin(("functional", "epileptic"))].copy()
    preictal["log_js"] = np.log(preictal.global_js.clip(lower=1e-8))
    preictal["cohort_group"] = preictal["group"]
    event = (preictal.groupby(["participant", "event_key", "cohort_group", "event_type", "semiology", "time_bin"],
                              observed=True, dropna=False)[[*MODEL_FEATURES, "global_js", "kappa"]]
             .mean().reset_index())
    preictal_cells = (event.groupby(["participant", "cohort_group", "event_type", "time_bin"], observed=True)
                      [[*MODEL_FEATURES, "global_js", "kappa"]].mean().reset_index())
    participant_types = preictal[["participant", "cohort_group", "event_type"]].drop_duplicates()
    ii = global_frame[global_frame.condition.eq("ii")].copy()
    ii["log_js"] = np.log(ii.global_js.clip(lower=1e-8))
    ii = (ii.groupby("participant", observed=True)[[*MODEL_FEATURES, "global_js", "kappa"]]
          .mean().reset_index().merge(participant_types, on="participant", how="inner"))
    ii["time_bin"] = "II"
    cells = pd.concat([ii[preictal_cells.columns], preictal_cells], ignore_index=True)
    cells["group"] = cells.event_type.map({"functional": "FS", "epileptic": "ES"})
    return event, cells


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    estimates = np.mean(values[rng.integers(0, len(values), size=(draws, len(values)))], axis=1)
    low, high = np.quantile(estimates, (.025, .975))
    return float(np.mean(values)), float(low), float(high)


def participant_cluster_difference(
    frame: pd.DataFrame,
    outcome: str,
    first: str,
    second: str,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, int, int, int]:
    """Cluster bootstrap that preserves participants contributing to both groups."""
    use = frame[frame.group.isin((first, second))].dropna(subset=[outcome]).copy()
    pivot = use.pivot_table(index="participant", columns="group", values=outcome, aggfunc="mean")
    for group in (first, second):
        if group not in pivot.columns:
            raise ValueError(f"Missing group {group} for {outcome}")
    first_values = pivot[first].to_numpy(float)
    second_values = pivot[second].to_numpy(float)
    first_mask = np.isfinite(first_values).astype(float)
    second_mask = np.isfinite(second_values).astype(float)
    first_filled = np.nan_to_num(first_values, nan=0.0)
    second_filled = np.nan_to_num(second_values, nan=0.0)
    observed = float(np.nanmean(first_values) - np.nanmean(second_values))
    n_participants = len(pivot)
    weights = rng.multinomial(
        n_participants,
        np.repeat(1 / n_participants, n_participants),
        size=draws,
    )
    first_denominator = weights @ first_mask
    second_denominator = weights @ second_mask
    valid = (first_denominator > 0) & (second_denominator > 0)
    samples = (
        (weights[valid] @ first_filled) / first_denominator[valid]
        - (weights[valid] @ second_filled) / second_denominator[valid]
    )
    low, high = np.quantile(samples, (0.025, 0.975))
    return (
        observed,
        float(low),
        float(high),
        float(np.mean(samples > 0)),
        n_participants,
        int(first_mask.sum()),
        int(second_mask.sum()),
    )


def global_summaries(cells: pd.DataFrame, draws: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows, contrasts = [], []
    for outcome in ("log_js", "log_kappa"):
        for group in ("FS", "ES"):
            for time_bin in TIME_ORDER:
                part = cells[cells.group.eq(group) & cells.time_bin.eq(time_bin)]
                estimate, low, high = bootstrap_interval(part[outcome].to_numpy(float), rng, draws)
                rows.append({
                    "outcome": outcome,
                    "group": group,
                    "time_bin": time_bin,
                    "estimate": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_participants": part.participant.nunique(),
                    "estimator": "participant-equal mean; participant bootstrap",
                })
        for time_bin in TIME_ORDER:
            part = cells[cells.time_bin.eq(time_bin)]
            estimate, low, high, probability, n_participants, n_fs, n_es = participant_cluster_difference(
                part, outcome, "FS", "ES", draws, rng
            )
            contrasts.append({
                "outcome": outcome,
                "contrast": "FS minus ES",
                "time_bin": time_bin,
                "estimate": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "probability_gt_zero": probability,
                "n_participants": n_participants,
                "n_fs": n_fs,
                "n_es": n_es,
                "estimator": "participant-cluster bootstrap preserving dual-diagnosis overlap",
            })
        for group in ("FS", "ES"):
            pivot = cells[cells.group.eq(group)].pivot(index="participant", columns="time_bin", values=outcome)
            for time_bin in TIME_BINS:
                pair = pivot[["II", time_bin]].dropna()
                difference = (pair[time_bin] - pair["II"]).to_numpy(float)
                estimate, low, high = bootstrap_interval(difference, rng, draws)
                if len(difference):
                    samples = np.mean(
                        difference[rng.integers(0, len(difference), size=(draws, len(difference)))],
                        axis=1,
                    )
                    probability = float(np.mean(samples > 0))
                else:
                    probability = np.nan
                contrasts.append({
                    "outcome": outcome,
                    "contrast": f"{group} {time_bin} minus II",
                    "time_bin": time_bin,
                    "estimate": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "probability_gt_zero": probability,
                    "n_participants": len(pair),
                    "n_fs": len(pair) if group == "FS" else 0,
                    "n_es": len(pair) if group == "ES" else 0,
                    "estimator": "paired participant bootstrap",
                })
    summary = pd.DataFrame(rows)
    for column in ("estimate", "ci95_low", "ci95_high"):
        summary[f"natural_{column}"] = np.exp(summary[column])
    contrast = pd.DataFrame(contrasts)
    contrast["ratio"] = np.exp(contrast.estimate)
    contrast["ratio_ci95_low"] = np.exp(contrast.ci95_low)
    contrast["ratio_ci95_high"] = np.exp(contrast.ci95_high)
    return summary, contrast


def subgroup_comparisons(event: pd.DataFrame, draws: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Participant-level DD FS−ES and FS hypo−hyper contrasts by preictal bin."""
    rng = np.random.default_rng(seed)
    outcomes = ("log_js", "log_kappa")
    rows, contrasts = [], []

    dd_source = event[event.cohort_group.eq("DD")]
    dd_participants = dd_source.groupby("participant", observed=True).event_type.nunique().loc[lambda x: x == 2].index
    dd = (dd_source[dd_source.participant.isin(dd_participants)]
          .groupby(["participant", "event_type", "time_bin"], observed=True)[list(outcomes)]
          .mean().reset_index())
    for outcome in outcomes:
        for time_bin in TIME_BINS:
            pivot = dd[dd.time_bin.eq(time_bin)].pivot(index="participant", columns="event_type", values=outcome)
            pair = pivot[["functional", "epileptic"]].dropna()
            difference = (pair.functional - pair.epileptic).to_numpy(float)
            estimate, low, high = bootstrap_interval(difference, rng, draws)
            contrasts.append({"comparison": "DD paired FS minus ES", "outcome": outcome, "time_bin": time_bin,
                              "estimate": estimate, "ci95_low": low, "ci95_high": high,
                              "probability_gt_zero": float(np.mean(
                                  np.mean(difference[rng.integers(0, len(difference), (draws, len(difference)))], axis=1) > 0)),
                              "n_participants": len(pair), "n_level_1": len(pair), "n_level_2": len(pair),
                              "n_paired": len(pair), "estimator": "paired participant mean; participant bootstrap"})
            for level in ("functional", "epileptic"):
                rows.extend({"comparison": "DD paired FS versus ES", "participant": participant,
                             "level": "FS" if level == "functional" else "ES", "time_bin": time_bin,
                             "outcome": outcome, "value": value}
                            for participant, value in pair[level].items())

    fs = (event[event.cohort_group.eq("FS") & event.event_type.eq("functional") &
                      event.semiology.isin(("hypo", "hyper"))]
          .groupby(["participant", "semiology", "time_bin"], observed=True)[list(outcomes)]
          .mean().reset_index())
    for outcome in outcomes:
        for time_bin in TIME_BINS:
            use = fs[fs.time_bin.eq(time_bin)].dropna(subset=[outcome])
            participants = use.participant.unique()
            hypo = use[use.semiology.eq("hypo")][outcome]
            hyper = use[use.semiology.eq("hyper")][outcome]
            samples = []
            for _ in range(draws):
                weights = pd.Series(rng.choice(participants, len(participants), replace=True)).value_counts()
                sample = use.merge(weights.rename("weight"), left_on="participant", right_index=True)
                means = sample.groupby("semiology", observed=True).apply(
                    lambda part: np.average(part[outcome], weights=part.weight), include_groups=False)
                if {"hypo", "hyper"}.issubset(means.index):
                    samples.append(means.hypo - means.hyper)
            samples = np.asarray(samples, float)
            low, high = np.quantile(samples, (.025, .975))
            paired = use.groupby("participant", observed=True).semiology.nunique().eq(2).sum()
            contrasts.append({"comparison": "FS hypo minus hyper", "outcome": outcome, "time_bin": time_bin,
                              "estimate": float(hypo.mean() - hyper.mean()), "ci95_low": low, "ci95_high": high,
                              "probability_gt_zero": float(np.mean(samples > 0)),
                              "n_participants": len(participants), "n_level_1": hypo.size, "n_level_2": hyper.size,
                              "n_paired": int(paired),
                              "estimator": "participant-semiology mean; participant-cluster bootstrap"})
            rows.extend({"comparison": "FS hypo versus hyper", "participant": row.participant,
                         "level": row.semiology, "time_bin": time_bin, "outcome": outcome,
                         "value": getattr(row, outcome)} for row in use.itertuples(index=False))

    contrast = pd.DataFrame(contrasts)
    contrast["ratio"] = np.exp(contrast.estimate)
    contrast["ratio_ci95_low"] = np.exp(contrast.ci95_low)
    contrast["ratio_ci95_high"] = np.exp(contrast.ci95_high)
    cells = pd.DataFrame(rows)
    assert len(contrast) == 2 * 2 * len(TIME_BINS) and not cells.empty
    return cells, contrast


def refresh_demographics(manifest_path: Path, database_path: Path, tables: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refresh anonymised cohort tables from the current manifest and clinical database."""
    manifest = pd.read_csv(manifest_path, low_memory=False)
    database = pd.read_csv(database_path, low_memory=False)
    participants = manifest[["participant", "patient_id", "group"]].drop_duplicates("participant").copy()
    demo = database[["patient_id", "age", "gender"]].drop_duplicates("patient_id").copy()
    participants["patient_id"] = participants.patient_id.astype(str)
    demo["patient_id"] = demo.patient_id.astype(str)
    participants = participants.merge(demo, on="patient_id", how="left", validate="many_to_one")
    participants["age"] = pd.to_numeric(participants.age, errors="coerce")
    participants["gender"] = (participants.gender.astype(str).str.strip().str.lower()
                              .replace({"nan": "unknown", "": "unknown", "f": "female", "m": "male"}))
    summary = []
    for group, part in participants.groupby("group", observed=True):
        age = part.age.dropna(); genders = part.gender.value_counts()
        summary.append({"group": group, "participants": len(part), "age_available": len(age),
                        "age_median": age.median(), "age_q1": age.quantile(.25), "age_q3": age.quantile(.75),
                        "female": genders.get("female", 0), "male": genders.get("male", 0),
                        "gender_unknown": genders.get("unknown", 0)})
    summary = pd.DataFrame(summary)
    event_source = manifest[manifest.condition.astype(str).str.lower().isin(("preict", "transictal"))].copy()
    event_source["semiology"] = (event_source.hypo_hyper.astype(str).str.strip().str.lower()
                                  .replace({"nan": np.nan, "": np.nan}))
    events = (event_source.groupby(["group", "condition", "seizure_type", "semiology"],
                                   observed=True, dropna=False)
              .agg(records=("file_path", "size"), participants=("participant", "nunique")).reset_index())
    participants.to_csv(tables / "participant_demographics.csv", index=False)
    summary.to_csv(tables / "demographic_summary.csv", index=False)
    events.to_csv(tables / "event_characteristics.csv", index=False)
    assert participants.participant.is_unique and len(participants) == int(summary.participants.sum())
    return summary, events


def match_pseudo_onsets(
    windows: pd.DataFrame,
    common: pd.DataFrame,
    minimum_record_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = merge_autonomic(windows, common)
    source["record_fraction"] = source.window_end_s / source.groupby("file_path", observed=True).window_end_s.transform("max")
    source["has_autonomic"] = source[["n_native", "csi_z", "cpi_z"]].notna().all(axis=1)
    true = source[source.condition.eq("preict") & source.window_end_to_ictus_s.eq(0) &
                  source.event_type.isin(("functional", "epileptic"))]
    pseudo = source[source.condition.eq("ii") & source.record_fraction.ge(minimum_record_fraction)]
    features = ["record_fraction", "n_gfp_peaks", "n_state_changes", *(f"n_{state}" for state in STATES),
                "n_native", "csi_z", "cpi_z"]
    matches = []
    for participant, targets in true.groupby("participant", observed=True, sort=False):
        pool = pseudo[pseudo.participant.eq(participant)]
        if pool.empty:
            continue
        for has_autonomic in (False, True):
            target_group = targets[targets.has_autonomic.eq(has_autonomic)]
            pool_group = pool[pool.has_autonomic.eq(has_autonomic)]
            if target_group.empty or pool_group.empty:
                continue
            scales = pool_group[features].apply(pd.to_numeric, errors="coerce").std().replace(0, 1).fillna(1)
            target_values = target_group[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            pool_values = pool_group[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            delta = (target_values[:, None, :] - pool_values[None, :, :]) / scales.to_numpy(float)
            weights = np.array([2.0 if feature in ("n_native", "csi_z", "cpi_z") else 1.0 for feature in features])
            delta *= weights
            comparable = np.isfinite(delta).sum(axis=2)
            cost = np.sqrt(np.nansum(delta ** 2, axis=2) / np.maximum(comparable, 1))
            target_positions, pool_positions = linear_sum_assignment(cost)
            for target_position, pool_position in zip(target_positions, pool_positions):
                target = target_group.iloc[target_position]
                candidate = pool_group.iloc[pool_position]
                row = {"participant": participant, "event_key": target.event_key,
                       "event_type": target.event_type, "semiology": target.semiology,
                       "true_row_id": int(target.row_id), "pseudo_row_id": int(candidate.row_id),
                       "true_file_path": target.file_path, "pseudo_file_path": candidate.file_path,
                       "match_distance": float(cost[target_position, pool_position]),
                       "n_match_features": int(comparable[target_position, pool_position])}
                for feature in features:
                    row[f"true_{feature}"] = target[feature]
                    row[f"pseudo_{feature}"] = candidate[feature]
                    row[f"difference_{feature}"] = target[feature] - candidate[feature]
                matches.append(row)
    output = pd.DataFrame(matches)
    if output.empty:
        raise ValueError("No participant-matched pseudo-onsets were available")
    balance = []
    for feature in features:
        true_values = pd.to_numeric(output[f"true_{feature}"], errors="coerce")
        pseudo_values = pd.to_numeric(output[f"pseudo_{feature}"], errors="coerce")
        pooled_sd = math.sqrt((true_values.var() + pseudo_values.var()) / 2)
        balance.append({"feature": feature, "standardised_mean_difference":
                        float((true_values.mean() - pseudo_values.mean()) / pooled_sd) if pooled_sd > 0 else 0,
                        "true_mean": true_values.mean(), "pseudo_mean": pseudo_values.mean(),
                        "n_pairs": int((true_values.notna() & pseudo_values.notna()).sum())})
    return output, pd.DataFrame(balance)


def weighted_least_squares_beta(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    root = np.sqrt(np.asarray(weights, float))
    coefficients, *_ = np.linalg.lstsq(x * root[:, None], y * root, rcond=None)
    return coefficients


def pseudo_difference_in_differences(
    matches: pd.DataFrame,
    windows: pd.DataFrame,
    minimum: int,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Common-reference genuine/pseudo comparison with participant-cluster inference.

    Both members of each pair are evaluated against exactly the same participant II
    transition grammar after removing the matched pseudo-window.  This eliminates the
    earlier reference asymmetry.  Covariate-adjusted estimates account for remaining
    matching imbalance without treating repeated events as independent participants.
    """
    cubes = windows.loc[:, COUNT_COLUMNS].to_numpy(float).reshape(-1, 6, 6)
    diagonal = np.arange(6)
    cubes[:, diagonal, diagonal] = 0
    row_to_position = dict(zip(windows.row_id.astype(int), np.arange(len(windows))))
    ii_totals = {
        str(participant): cubes[np.asarray(list(indices), dtype=int)].sum(axis=0)
        for participant, indices in windows[windows.condition.eq("ii")].groupby(
            windows.participant.astype(str), observed=True
        ).groups.items()
    }
    paired = matches.copy()
    metric_rows = []
    origins = tuple(range(6))
    for row in paired.itertuples(index=False):
        participant = str(row.participant)
        true_position = row_to_position.get(int(row.true_row_id))
        pseudo_position = row_to_position.get(int(row.pseudo_row_id))
        entry = {"true_row_id": int(row.true_row_id)}
        if true_position is None or pseudo_position is None or participant not in ii_totals:
            entry.update({
                "true_log_global_js": np.nan,
                "pseudo_log_global_js": np.nan,
                "difference_log_global_js": np.nan,
                "true_log_kappa": np.nan,
                "pseudo_log_kappa": np.nan,
                "difference_log_kappa": np.nan,
            })
            metric_rows.append(entry)
            continue
        reference_counts = np.maximum(ii_totals[participant] - cubes[pseudo_position], 0)
        if reference_counts.sum() < minimum:
            metric_rows.append(entry | {
                "true_log_global_js": np.nan,
                "pseudo_log_global_js": np.nan,
                "difference_log_global_js": np.nan,
                "true_log_kappa": np.nan,
                "pseudo_log_kappa": np.nan,
                "difference_log_kappa": np.nan,
            })
            continue
        baseline = probability(reference_counts)
        values = {}
        for label, position in (("true", true_position), ("pseudo", pseudo_position)):
            counts = cubes[position]
            if counts.sum() < minimum:
                values[f"{label}_log_global_js"] = np.nan
                values[f"{label}_log_kappa"] = np.nan
                continue
            js = js_divergence(counts, baseline, origins)
            values[f"{label}_log_global_js"] = math.log(max(js, 1e-12))
            values[f"{label}_log_kappa"] = fit_log_kappa(counts, baseline, origins, minimum)
        entry.update(values)
        for outcome in ("log_global_js", "log_kappa"):
            entry[f"difference_{outcome}"] = entry.get(f"true_{outcome}", np.nan) - entry.get(
                f"pseudo_{outcome}", np.nan
            )
        metric_rows.append(entry)
    paired = paired.merge(pd.DataFrame(metric_rows), on="true_row_id", how="left", validate="one_to_one")

    covariate_candidates = [
        "difference_n_native",
        "difference_n_gfp_peaks",
        "difference_n_state_changes",
        "difference_csi_z",
        "difference_cpi_z",
    ]
    covariates = [column for column in covariate_candidates if column in paired.columns]
    participant = (
        paired.groupby(["participant", "event_type"], observed=True)[
            ["difference_log_global_js", "difference_log_kappa", *covariates]
        ].mean().reset_index()
    )
    participant["is_fs"] = participant.event_type.eq("functional").astype(float)
    for covariate in covariates:
        values = pd.to_numeric(participant[covariate], errors="coerce")
        scale = values.std()
        participant[covariate] = (values - values.mean()) / scale if np.isfinite(scale) and scale > 0 else 0.0
        participant[covariate] = participant[covariate].fillna(0.0)

    rng = np.random.default_rng(seed)
    rows = []
    for outcome in ("log_global_js", "log_kappa"):
        column = f"difference_{outcome}"
        complete = participant.dropna(subset=[column]).copy()
        unique_participants = complete.participant.astype(str).unique()
        if not len(unique_participants):
            continue
        design = np.column_stack((
            np.ones(len(complete)),
            complete.is_fs.to_numpy(float),
            *(complete[covariate].to_numpy(float) for covariate in covariates),
        ))
        y = complete[column].to_numpy(float)
        participant_codes = pd.Categorical(complete.participant.astype(str), categories=unique_participants).codes
        unit_weights = np.ones(len(complete), dtype=float)
        adjusted_observed = float(weighted_least_squares_beta(design, y, unit_weights)[1])
        means = complete.groupby("event_type", observed=True)[column].mean()
        unadjusted_observed = float(means.get("functional", np.nan) - means.get("epileptic", np.nan))
        adjusted_samples = []
        unadjusted_samples = []
        for _ in range(draws):
            participant_weights = rng.multinomial(
                len(unique_participants),
                np.repeat(1 / len(unique_participants), len(unique_participants)),
            )
            weights = participant_weights[participant_codes].astype(float)
            if not np.any(weights):
                continue
            beta = weighted_least_squares_beta(design, y, weights)
            adjusted_samples.append(beta[1])
            fs_mask = complete.event_type.eq("functional").to_numpy()
            es_mask = complete.event_type.eq("epileptic").to_numpy()
            fs_denominator = weights[fs_mask].sum()
            es_denominator = weights[es_mask].sum()
            if fs_denominator > 0 and es_denominator > 0:
                unadjusted_samples.append(
                    np.average(y[fs_mask], weights=weights[fs_mask])
                    - np.average(y[es_mask], weights=weights[es_mask])
                )
        for estimator, estimate, samples in (
            ("unadjusted participant-cluster DiD", unadjusted_observed, np.asarray(unadjusted_samples)),
            ("covariate-adjusted participant-cluster DiD", adjusted_observed, np.asarray(adjusted_samples)),
        ):
            low, high = np.quantile(samples, (0.025, 0.975))
            rows.append({
                "outcome": "log JS" if outcome == "log_global_js" else "log kappa",
                "contrast": "[FS genuine - pseudo] - [ES genuine - pseudo]",
                "estimator": estimator,
                "estimate": estimate,
                "ci95_low": float(low),
                "ci95_high": float(high),
                "probability_gt_zero": float(np.mean(samples > 0)),
                "n_pairs": int(paired[column].notna().sum()),
                "n_participants": complete.participant.nunique(),
                "covariates": "; ".join(covariates) if estimator.startswith("covariate") else "none",
                "reference": "same participant II grammar with matched pseudo-window removed for both members",
            })
    return paired, pd.DataFrame(rows)


def state_exploration(state_frame: pd.DataFrame, common: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = merge_autonomic(state_frame, common)
    frame = frame[frame.condition.eq("preict") & frame.time_bin.isin(TIME_BINS) &
                  frame.event_type.isin(("functional", "epileptic"))].copy()
    frame["log_source_js"] = np.log(frame.source_js.clip(lower=1e-8))
    event = (frame.groupby(["participant", "event_key", "event_type", "semiology", "time_bin", "source_state"],
                           observed=True, dropna=False)
             [["log_source_js", "source_log_kappa", "csi_z", "cpi_z"]].mean().reset_index())
    event["is_hypo"] = event.semiology.eq("hypo").astype(float)
    rows = []
    for source in STATES:
        for outcome in ("log_source_js", "source_log_kappa"):
            use = event[event.source_state.eq(source)].dropna(subset=[outcome]).copy()
            sd = use[outcome].std()
            if len(use) < 40 or not np.isfinite(sd) or sd == 0:
                continue
            use["outcome_z"] = (use[outcome] - use[outcome].mean()) / sd
            group_model = smf.ols(
                "outcome_z ~ C(event_type, Treatment('epileptic')) + C(time_bin)", data=use).fit(
                    cov_type="cluster", cov_kwds={"groups": use.participant})
            group_name = "C(event_type, Treatment('epileptic'))[T.functional]"
            rows.append(linear_result(group_model, {group_name: 1}, "FS minus ES",
                                      source_state=source, outcome=outcome,
                                      n_rows=len(use), n_participants=use.participant.nunique()))
            for autonomic, label in (("csi_z", "CSI association"), ("cpi_z", "CPI association")):
                subset = use.dropna(subset=[autonomic]).copy()
                scale = subset[autonomic].std()
                if len(subset) < 40 or not np.isfinite(scale) or scale == 0:
                    continue
                subset["autonomic_z"] = (subset[autonomic] - subset[autonomic].mean()) / scale
                fitted = smf.ols(
                    "outcome_z ~ autonomic_z + C(event_type, Treatment('epileptic')) + C(time_bin)",
                    data=subset).fit(cov_type="cluster", cov_kwds={"groups": subset.participant})
                rows.append(linear_result(fitted, {"autonomic_z": 1}, label,
                                          source_state=source, outcome=outcome,
                                          n_rows=len(subset), n_participants=subset.participant.nunique()))
            subset = use[use.semiology.isin(("hyper", "hypo"))].copy()
            if len(subset) >= 40:
                formula = ("outcome_z ~ is_hypo * C(event_type, Treatment('epileptic')) + C(time_bin)")
                fitted = smf.ols(formula, data=subset).fit(
                    cov_type="cluster", cov_kwds={"groups": subset.participant})
                interaction = "is_hypo:C(event_type, Treatment('epileptic'))[T.functional]"
                rows.append(linear_result(fitted, {"is_hypo": 1, interaction: 1},
                                          "FS hypo minus hyper", source_state=source, outcome=outcome,
                                          n_rows=len(subset), n_participants=subset.participant.nunique()))
                rows.append(linear_result(fitted, {"is_hypo": 1},
                                          "ES hypo minus hyper", source_state=source, outcome=outcome,
                                          n_rows=len(subset), n_participants=subset.participant.nunique()))
    results = pd.DataFrame(rows)
    results["p_bh"] = bh(results.p_value)
    return event, results


def plot_global(cells: pd.DataFrame, summary: pd.DataFrame, contrasts: pd.DataFrame,
                figures: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.8), layout="constrained",
                             gridspec_kw={"height_ratios": (1.15, 1)})
    x = np.arange(len(TIME_ORDER))
    labels = ("II", "-20\nto -10", "-10\nto -5", "-5\nto -2", "-2\nto -1", "-1\nto 0")
    for letter, axis, outcome, ylabel in zip("AB", axes[0], ("log_js", "log_kappa"),
                                             ("Global JS", "Global κ")):
        raw_column = "global_js" if outcome == "log_js" else "kappa"
        for group, color, offset in (("FS", FS_COLOR, -.08), ("ES", ES_COLOR, .08)):
            for position, time_bin in enumerate(TIME_ORDER):
                values = cells[cells.group.eq(group) & cells.time_bin.eq(time_bin)][raw_column].dropna().to_numpy(float)
                axis.scatter(position + offset + rng.uniform(-.055, .055, len(values)), values,
                             s=8, alpha=.14, color=color, linewidths=0, rasterized=True)
            part = summary[summary.group.eq(group) & summary.outcome.eq(outcome)].set_index("time_bin").reindex(TIME_ORDER)
            middle = part.natural_estimate.to_numpy(float)
            axis.errorbar(x + offset, middle,
                          yerr=[middle - part.natural_ci95_low.to_numpy(float),
                                part.natural_ci95_high.to_numpy(float) - middle],
                          color=color, marker="o", ms=4.5, lw=1.4, capsize=2.2, label=group, zorder=4)
        axis.set(xticks=x, xticklabels=labels, yscale="log", ylabel=ylabel,
                 xlabel="Period / minutes to onset", title=f"Participant-level {ylabel} trajectory")
        axis.grid(axis="y"); panel(axis, letter)
    axes[0, 0].legend(frameon=False, ncol=2)

    for letter, axis, outcome, title in zip("CD", axes[1], ("log_js", "log_kappa"),
                                            ("FS / ES global JS", "FS / ES global κ")):
        part = contrasts[contrasts.contrast.eq("FS minus ES") & contrasts.outcome.eq(outcome)].set_index("time_bin").reindex(TIME_ORDER)
        y = np.arange(len(part)); middle = part.estimate.to_numpy(float)
        axis.errorbar(middle, y, xerr=[middle - part.ci95_low.to_numpy(float),
                                      part.ci95_high.to_numpy(float) - middle],
                      fmt="o", color=PURPLE if outcome == "log_js" else GREEN, capsize=2.4, ms=4.8)
        axis.axvline(0, color=GREY, lw=.8)
        axis.set(yticks=y, yticklabels=labels, xlabel="Log ratio (FS / ES)", title=title)
        axis.invert_yaxis(); axis.grid(axis="x"); panel(axis, letter)
    save_figure(fig, figures, "figure_01_global_transition_trajectories")


def plot_manifold(scores: pd.DataFrame, loadings: pd.DataFrame, audit: dict, figures: Path) -> None:
    fig = plt.figure(figsize=(7.6, 3.35), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=(1.15, 1.15, .8))
    limits = []
    for component in ("PC1", "PC2"):
        values = scores[component].to_numpy(float)
        low, high = np.quantile(values, (.05, .95)); pad = (high - low) * .08
        limits.append((low - pad, high + pad))
    colours = plt.cm.viridis(np.linspace(.08, .9, len(TIME_ORDER)))
    for letter, group, color, slot in zip("AB", ("FS", "ES"), (FS_COLOR, ES_COLOR), (grid[0, 0], grid[0, 1])):
        axis = fig.add_subplot(slot)
        part = scores[scores.group.eq(group)]
        centres = []
        for index, time_bin in enumerate(TIME_ORDER):
            values = part[part.time_bin.eq(time_bin)]
            axis.scatter(values.PC1, values.PC2, s=7, color=colours[index], alpha=.12,
                         linewidths=0, rasterized=True)
            centres.append((values.PC1.mean(), values.PC2.mean()))
        centres = np.asarray(centres)
        axis.plot(centres[:, 0], centres[:, 1], color=color, lw=2, zorder=4)
        for index, ((pc1, pc2), time_bin) in enumerate(zip(centres, TIME_ORDER)):
            axis.scatter(pc1, pc2, s=32, color=colours[index], edgecolor="white", linewidth=.7, zorder=5)
        axis.axhline(0, color=GREY, lw=.5, alpha=.5); axis.axvline(0, color=GREY, lw=.5, alpha=.5)
        axis.set(xlim=limits[0], ylim=limits[1], xlabel="PC1", ylabel="PC2",
                 title=f"{group}: II → preictal movement")
        axis.grid(); panel(axis, letter)
    axis = fig.add_subplot(grid[0, 2])
    scale = 2.5
    for row in loadings.itertuples(index=False):
        axis.arrow(0, 0, row.PC1 * scale, row.PC2 * scale, color=PURPLE,
                   width=.012, head_width=.11, length_includes_head=True, alpha=.85)
        axis.text(row.PC1 * scale * 1.12, row.PC2 * scale * 1.12, row.feature_label,
                  ha="center", va="center", fontsize=7)
    axis.axhline(0, color=GREY, lw=.6); axis.axvline(0, color=GREY, lw=.6)
    axis.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), xlabel="PC1 loading", ylabel="PC2 loading",
             title="Auditable feature loadings")
    axis.grid(); panel(axis, "C")
    explained = 100 * np.asarray(audit["explained_variance_ratio"])
    fig.suptitle(f"Robustly scaled 4-variable manifold: PC1 {explained[0]:.1f}% + PC2 {explained[1]:.1f}% variance",
                 fontsize=10.5)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
                      markeredgecolor="white", markersize=5) for color in colours]
    fig.legend(handles, TIME_ORDER, ncol=6, frameon=False, loc="lower center",
               bbox_to_anchor=(.45, -.06), fontsize=7, handletextpad=.25, columnspacing=.8)
    fig.text(.45, -.075, "Display axes show the central 90% of manifold scores",
             ha="center", va="top", fontsize=6.5, color=GREY)
    save_figure(fig, figures, "figure_02_js_kappa_csi_cpi_manifold")


def plot_logistic(models: pd.DataFrame, predictions: pd.DataFrame, coefficients: pd.DataFrame,
                  figures: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.7, 3.2), layout="constrained",
                             gridspec_kw={"width_ratios": (1, 1, 1.05)})
    for scope, color, marker in (("all preictal bins", FS_COLOR, "o"), ("final minute", ES_COLOR, "s")):
        part = models[models.scope.eq(scope)].sort_values("step")
        axes[0].errorbar(part.step, part.auc,
                         yerr=[part.auc - part.auc_ci95_low, part.auc_ci95_high - part.auc],
                         color=color, marker=marker, capsize=2.5, label=scope)
    axes[0].axhline(.5, color=GREY, lw=.8, linestyle="--")
    axes[0].set(xticks=range(1, 5), xticklabels=("JS", "+κ", "+CSI", "+CPI"), ylim=(.42, 1),
                xlabel="Planned insertion", ylabel="Participant-held-out AUC",
                title="Incremental delineation")
    axes[0].grid(axis="y"); axes[0].legend(frameon=False, fontsize=7); panel(axes[0], "A")

    final = coefficients[coefficients.scope.eq("all preictal bins") & coefficients.step.eq(4)].copy()
    y = np.arange(len(final)); middle = final.coefficient.to_numpy(float)
    axes[1].errorbar(middle, y, xerr=[middle - final.ci95_low, final.ci95_high - middle],
                     fmt="o", color=PURPLE, capsize=2.7, ms=5)
    axes[1].axvline(0, color=GREY, lw=.8)
    axes[1].set(yticks=y, yticklabels=final.feature_label, xlabel="Regularised log-odds coefficient",
                title="Full model (FS positive)")
    axes[1].invert_yaxis(); axes[1].grid(axis="x"); panel(axes[1], "B")

    prediction = predictions[predictions.scope.eq("all preictal bins") & predictions.step.eq(4)]
    fpr, tpr, _ = roc_curve(prediction.target_fs, prediction.probability_fs)
    roc_line, = axes[2].plot(fpr, tpr, color=FS_COLOR, lw=1.8)
    axes[2].plot((0, 1), (0, 1), color=GREY, lw=.8, linestyle="--")
    bins = pd.qcut(prediction.probability_fs, q=8, duplicates="drop")
    calibration = prediction.assign(bin=bins).groupby("bin", observed=True).agg(
        predicted=("probability_fs", "mean"), observed=("target_fs", "mean")).reset_index()
    calibration_points = axes[2].scatter(calibration.predicted, calibration.observed,
                                         color=ES_COLOR, s=20, zorder=3)
    axes[2].set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted FS probability / false-positive rate",
                ylabel="Observed FS fraction / true-positive rate", title="Held-out ROC and calibration")
    axes[2].grid(); panel(axes[2], "C")
    axes[2].legend((roc_line, calibration_points), ("Held-out ROC", "Calibration bins"),
                   frameon=False, fontsize=7, loc="lower right")
    save_figure(fig, figures, "figure_03_stepwise_logistic")


def plot_state_exploration(event: pd.DataFrame, associations: pd.DataFrame,
                           figures: Path, seed: int) -> None:
    columns = ("FS minus ES", "CSI association", "CPI association", "FS hypo minus hyper")
    order = [(state, outcome) for outcome in ("log_source_js", "source_log_kappa") for state in STATES]
    matrix = np.full((len(order), len(columns)), np.nan)
    significant = np.zeros_like(matrix, dtype=bool)
    for row_index, (state, outcome) in enumerate(order):
        for column_index, association in enumerate(columns):
            hit = associations[associations.source_state.eq(state) & associations.outcome.eq(outcome) &
                               associations.association.eq(association)]
            if len(hit):
                matrix[row_index, column_index] = hit.estimate.iloc[0]
                significant[row_index, column_index] = hit.p_bh.iloc[0] < .05
    maximum = max(.5, np.nanmax(np.abs(matrix)))
    fig = plt.figure(figsize=(7.7, 5), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.5, 1))
    axis = fig.add_subplot(grid[0, :])
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-maximum, vmax=maximum, aspect="auto")
    ylabels = [f"{state} · {'JS' if outcome == 'log_source_js' else 'κ'}" for state, outcome in order]
    axis.set(xticks=np.arange(len(columns)), xticklabels=("FS−ES", "CSI", "CPI", "FS hypo−hyper"),
             yticks=np.arange(len(order)), yticklabels=ylabels,
             title="Exploratory source-state effects (standardised; BH-FDR dot)")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                axis.text(column, row, f"{value:.2f}{'•' if significant[row, column] else ''}",
                          ha="center", va="center", fontsize=6.6,
                          color="white" if abs(value) > maximum * .62 else "black")
    fig.colorbar(image, ax=axis, shrink=.75, pad=.015, label="Standardised coefficient")
    panel(axis, "A")

    rng = np.random.default_rng(seed)
    final = event[event.time_bin.eq("-1 to 0")].copy()
    for letter, axis, outcome, ylabel in zip("BC", (fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])),
                                             ("log_source_js", "source_log_kappa"),
                                             ("Source-state log JS", "Source-state log κ")):
        for group, event_type, color, offset in (("FS", "functional", FS_COLOR, -.13),
                                                 ("ES", "epileptic", ES_COLOR, .13)):
            for position, state in enumerate(STATES):
                values = final[final.event_type.eq(event_type) & final.source_state.eq(state)][outcome].dropna().to_numpy(float)
                axis.scatter(position + offset + rng.uniform(-.07, .07, len(values)), values,
                             color=color, alpha=.12, s=7, linewidths=0, rasterized=True)
                if len(values):
                    axis.scatter(position + offset, np.mean(values), color=color, edgecolor="white",
                                 linewidth=.5, s=28, label=group if position == 0 else None, zorder=4)
        axis.set(xticks=np.arange(6), xticklabels=STATES, xlabel="Source microstate",
                 ylabel=ylabel, title="Final-minute event-bin distributions")
        axis.grid(axis="y"); panel(axis, letter)
    fig.axes[-2].legend(frameon=False, ncol=2)
    save_figure(fig, figures, "figure_04_exploratory_state_drilldown")


def plot_pseudo(
    paired: pd.DataFrame,
    did: pd.DataFrame,
    balance: pd.DataFrame,
    figures: Path,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(
        1, 3, figsize=(7.7, 3.1), layout="constrained",
        gridspec_kw={"width_ratios": (1, 0.95, 1.2)},
    )
    difference_column = (
        "difference_log_global_js"
        if "difference_log_global_js" in paired.columns
        else "difference_global_js"
    )
    for position, (event_type, group, color) in enumerate((
        ("functional", "FS", FS_COLOR),
        ("epileptic", "ES", ES_COLOR),
    )):
        values = paired[paired.event_type.eq(event_type)][difference_column].dropna().to_numpy(float)
        axes[0].scatter(
            position + rng.uniform(-0.14, 0.14, len(values)),
            values,
            color=color,
            alpha=0.16,
            s=8,
            linewidths=0,
        )
        axes[0].scatter(
            position,
            np.mean(values),
            marker="D",
            color=color,
            edgecolor="white",
            linewidth=0.7,
            s=38,
            zorder=4,
        )
    axes[0].axhline(0, color=GREY, lw=0.8)
    axes[0].set(
        xticks=(0, 1),
        xticklabels=("FS", "ES"),
        ylabel="Genuine − pseudo log JS",
        title="Matched event differences",
    )
    axes[0].grid(axis="y")
    panel(axes[0], "A")

    if "estimator" in did.columns:
        adjusted = did[did.estimator.astype(str).str.startswith("covariate-adjusted")].copy()
        unadjusted = did[did.estimator.astype(str).str.startswith("unadjusted")].copy()
    else:
        adjusted = did.copy()
        unadjusted = pd.DataFrame()
    y = np.arange(len(adjusted))
    middle = adjusted.estimate.to_numpy(float)
    axes[1].errorbar(
        middle,
        y,
        xerr=[middle - adjusted.ci95_low, adjusted.ci95_high - middle],
        fmt="o",
        color=PURPLE,
        capsize=3,
        ms=5,
        label="Adjusted",
    )
    if not unadjusted.empty:
        unadjusted = unadjusted.set_index("outcome").reindex(adjusted.outcome)
        axes[1].scatter(unadjusted.estimate, y, color=GREY, marker="x", s=26, label="Unadjusted")
    axes[1].axvline(0, color=GREY, lw=0.8)
    axes[1].set(
        yticks=y,
        yticklabels=adjusted.outcome,
        xlabel="Matched difference-in-differences",
        title="Onset-specific sensitivity",
    )
    axes[1].invert_yaxis()
    axes[1].grid(axis="x")
    axes[1].legend(frameon=False, fontsize=7)
    panel(axes[1], "B")

    display = balance.sort_values("standardised_mean_difference")
    y = np.arange(len(display))
    axes[2].barh(
        y,
        display.standardised_mean_difference,
        color=np.where(display.standardised_mean_difference.abs() <= 0.1, GREEN, ES_COLOR),
        alpha=0.78,
    )
    axes[2].axvline(0, color=GREY, lw=0.7)
    axes[2].axvline(-0.1, color=GREY, lw=0.6, linestyle="--")
    axes[2].axvline(0.1, color=GREY, lw=0.6, linestyle="--")
    axes[2].set(
        yticks=y,
        yticklabels=display.feature.str.replace("_", " "),
        xlabel="Standardised mean difference",
        title="True/pseudo match balance",
    )
    axes[2].grid(axis="x")
    panel(axes[2], "C")
    save_figure(fig, figures, "figure_05_matched_pseudo_sensitivity")


def plot_subgroups(cells: pd.DataFrame, contrasts: pd.DataFrame, figures: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 2, figsize=(7.7, 5.2), layout="constrained", sharex=True)
    specifications = (
        ("DD paired FS versus ES", "DD paired FS minus ES", ("FS", "ES"), (FS_COLOR, ES_COLOR)),
        ("FS hypo versus hyper", "FS hypo minus hyper", ("hypo", "hyper"), (PURPLE, GREEN)),
    )
    for row, (cell_name, contrast_name, levels, colors) in enumerate(specifications):
        for column, outcome in enumerate(("log_js", "log_kappa")):
            axis = axes[row, column]
            use = cells[cells.comparison.eq(cell_name) & cells.outcome.eq(outcome)]
            for offset, level, color in zip((-.10, .10), levels, colors):
                means = []
                for position, time_bin in enumerate(TIME_BINS):
                    values = use[use.level.eq(level) & use.time_bin.eq(time_bin)].value.dropna().to_numpy(float)
                    axis.scatter(position + offset + rng.uniform(-.045, .045, len(values)), values,
                                 color=color, alpha=.13, s=7, linewidths=0, rasterized=True)
                    means.append(np.mean(values) if len(values) else np.nan)
                axis.plot(np.arange(len(TIME_BINS)) + offset, means, color=color, marker="o", ms=4,
                          lw=1.7, markeredgecolor="white", markeredgewidth=.55, label=level.upper())
            final = contrasts[contrasts.comparison.eq(contrast_name) & contrasts.outcome.eq(outcome) &
                              contrasts.time_bin.eq(TIME_BINS[-1])].iloc[0]
            axis.text(.02, .98, f"Final minute: {final.estimate:.2f} [{final.ci95_low:.2f}, {final.ci95_high:.2f}]",
                      transform=axis.transAxes, ha="left", va="top", fontsize=7, color=GREY)
            axis.set(ylabel="log JS" if outcome == "log_js" else "log κ",
                     title=("Dual diagnosis: FS versus ES" if row == 0 else "Functional seizures: hypo versus hyper"))
            axis.grid(axis="y")
            panel(axis, chr(ord("A") + row * 2 + column))
            if row == 1:
                axis.set(xticks=np.arange(len(TIME_BINS)), xticklabels=TIME_BINS, xlabel="Minutes before onset")
                axis.tick_params(axis="x", rotation=25)
            if column == 0:
                axis.legend(frameon=False, ncol=2, loc="lower left")
    fig.suptitle("Within-cohort global transition trajectories", fontsize=11)
    fig.text(.5, -.02, "Points are participant-level cells; lines are participant-equal means. Brackets are 95% participant-bootstrap intervals.",
             ha="center", va="top", fontsize=7, color=GREY)
    save_figure(fig, figures, "figure_06_within_cohort_comparisons")


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    use = frame[columns].copy()
    for column in use.select_dtypes(include="number"):
        use[column] = use[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    return ("| " + " | ".join(columns) + " |\n|" + "|".join(["---"] * len(columns)) + "|\n" +
            "\n".join("| " + " | ".join(map(str, row)) + " |" for row in use.itertuples(index=False, name=None)))


def write_report(out_dir: Path, summary: pd.DataFrame, contrasts: pd.DataFrame,
                 subgroup_contrasts: pd.DataFrame, demographics: pd.DataFrame,
                 logistic: pd.DataFrame, did: pd.DataFrame, associations: pd.DataFrame,
                 manifold_audit: dict, counts: dict) -> None:
    final = contrasts[contrasts.contrast.eq("FS minus ES") & contrasts.time_bin.eq("-1 to 0")]
    full = logistic[logistic.scope.eq("all preictal bins") & logistic.step.eq(4)].iloc[0]
    initial = logistic[logistic.scope.eq("all preictal bins") & logistic.step.eq(1)].iloc[0]
    top = associations.sort_values("p_bh").head(8)
    did_report = (
        did[did.estimator.astype(str).str.startswith("covariate-adjusted")]
        if "estimator" in did.columns else did
    )
    js_did = did_report[did_report.outcome.eq("log JS")].iloc[0]
    kappa_did = did_report[did_report.outcome.eq("log kappa")].iloc[0]
    js_specific = js_did.ci95_low > 0 or js_did.ci95_high < 0
    kappa_specific = kappa_did.ci95_low > 0 or kappa_did.ci95_high < 0
    report = [
        "# Global-first transition analysis", "",
        "## What this analysis changes", "",
        "No microstate is privileged at the outset. The headline analysis uses the entire six-state transition grammar: transition-count-weighted Jensen–Shannon divergence from each participant's interictal baseline and one joint baseline-oriented κ. Source-state, autonomic and semiological results are exploratory and BH-FDR controlled.", "",
        f"The analysis retained windows with at least {MIN_TRANSITIONS} off-diagonal transitions. Interictal windows use a leave-that-window-out participant reference; preictal windows use the participant's pooled interictal reference. Microstate fits were reused, not refitted.", "",
        "## Updated cohort demographics", "",
        markdown_table(demographics, ["group", "participants", "age_available", "age_median", "age_q1", "age_q3", "female", "male", "gender_unknown"]), "",
        "## Headline global findings", "",
        markdown_table(final[["outcome", "estimate", "ci95_low", "ci95_high", "ratio", "ratio_ci95_low", "ratio_ci95_high"]],
                       ["outcome", "estimate", "ci95_low", "ci95_high", "ratio", "ratio_ci95_low", "ratio_ci95_high"]), "",
        "The global trajectories are participant-equal summaries; points show participant cells and intervals are participant bootstraps. Ratios above one indicate higher FS values.", "",
        "![Global trajectories](figures/figure_01_global_transition_trajectories.png)", "",
        "## Within-cohort comparisons", "",
        "The dual-diagnosis comparison is paired within participant (FS minus ES). The functional-seizure semiology comparison is hypo minus hyper after event averaging within participant and semiology; its participant-cluster bootstrap preserves the partial pairing of participants who contributed both semiologies.", "",
        markdown_table(subgroup_contrasts[subgroup_contrasts.time_bin.eq("-1 to 0")],
                       ["comparison", "outcome", "estimate", "ci95_low", "ci95_high", "probability_gt_zero",
                        "n_participants", "n_level_1", "n_level_2", "n_paired"]), "",
        "![Within-cohort comparisons](figures/figure_06_within_cohort_comparisons.png)", "",
        "## Four-variable 2D manifold", "",
        f"A robustly scaled PCA captured {100 * sum(manifold_audit['explained_variance_ratio']):.1f}% of the JS–κ–CSI–CPI variance in two dimensions. Inputs were winsorised at the 1st/99th percentiles for this manifold only, with exact limits exported. It is unsupervised: FS/ES labels did not determine the axes. Participant paths and group centroids show movement from II through the five preictal bins.", "",
        "![Manifold](figures/figure_02_js_kappa_csi_cpi_manifold.png)", "",
        "## Planned stepwise logistic insertion", "",
        f"With leave-one-participant-out predictions, AUC was {initial.auc:.3f} ({initial.auc_ci95_low:.3f}–{initial.auc_ci95_high:.3f}) for JS alone and {full.auc:.3f} ({full.auc_ci95_low:.3f}–{full.auc_ci95_high:.3f}) after planned insertion of κ, CSI and CPI. These are internal phenotype-delineation estimates, not a clinical diagnostic validation.", "",
        "![Stepwise logistic regression](figures/figure_03_stepwise_logistic.png)", "",
        "## Exploratory microstate drill-down", "",
        "The heatmap asks which source grammars carry the global group difference and which covary with CSI, CPI or semiology. Coefficients are standardised; a dot marks BH-FDR < .05 across the full exploratory family.", "",
        markdown_table(top, ["source_state", "outcome", "association", "estimate", "ci95_low", "ci95_high", "p_bh"]), "",
        "![State drill-down](figures/figure_04_exploratory_state_drilldown.png)", "",
        "## Onset-specific sensitivity", "",
        "Participant-matched pseudo-onsets are retained as a guard against overclaiming. The estimand is [FS genuine − FS pseudo] − [ES genuine − ES pseudo]. " +
        ("Both intervals exclude zero, which supports onset-specific separation in this matched sensitivity; external validation is still required."
         if js_specific and kappa_specific else
         "The log-JS interval excludes zero but the log-κ interval crosses zero: matched data support onset-related divergence, not an onset-specific version of the full transition phenotype."
         if js_specific else
         "Neither interval excludes zero; the safer interpretation is an FS-associated transition phenotype, not an onset-specific biomarker."), "",
        markdown_table(did_report, ["outcome", "estimate", "ci95_low", "ci95_high", "probability_gt_zero", "n_pairs", "n_participants"]), "",
        "![Matched pseudo-onsets](figures/figure_05_matched_pseudo_sensitivity.png)", "",
        "## Traceability", "",
        f"{counts['global_windows']:,} global windows, {counts['state_windows']:,} eligible state-window rows, {counts['events']:,} preictal events and {counts['participants']:,} participants contributed. Exact analysis frames, PCA loadings/scores, held-out predictions, model summaries, matching balance and hashes are in `tables/` and `PUBLICATION_AUDIT.json`.", "",
        "Movement/EMG, EOG, high-frequency muscle power, bad-channel burden, sleep/wake state, medication timing, respiration, CFA variants, alternative segmentations and VR overlap were unavailable and therefore could not be matched or adjusted.",
    ]
    (out_dir / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def analysis_decisions() -> pd.DataFrame:
    return pd.DataFrame([
        ("headline transition scale", "replace", "Global six-source JS and joint κ; no privileged C source"),
        ("source-state analysis", "retain as exploratory", "Drill down only after the global result; BH-FDR across all state associations"),
        ("manifold", "replace", "Transparent robust-scaled PCA in JS, κ, CSI and CPI; report loadings and scores"),
        ("3D JS-κ-CPI displays", "replace", "The four-variable 2D manifold includes CSI and is easier to compare and reproduce"),
        ("classification", "add", "Planned JS → κ → CSI → CPI insertion; participant-held-out predictions"),
        ("within-cohort comparisons", "add", "Paired DD FS−ES and participant-clustered FS hypo−hyper trajectories"),
        ("action-conditioned transition tensor", "add", "Four CPI/CSI action regions with soft interpolation, II-only tuning, kappa/lambda, held-out prediction, lagged and hard-quadrant sensitivities"),
        ("onset claim", "retain sensitivity", "Matched genuine-minus-pseudo difference-in-differences; no unmatched onset claim"),
        ("transition count", "retain", "Count defines multinomial information/eligibility only; minimum and equalised sensitivity target is 30"),
        ("clinical context", "retain", "Demographics, heart rate, coverage and autonomic-level tables are copied without re-estimation"),
        ("failed confirmatory hierarchy", "retire", "Not used: previous NUTS/full-rank attempts failed validation or were superseded"),
    ], columns=("analysis", "decision", "reason"))


def build(args, out_dir: Path) -> dict:
    figures = out_dir / "figures"; tables = out_dir / "tables"
    figures.mkdir(parents=True); tables.mkdir(parents=True)
    windows = enrich_windows(pd.read_csv(args.peak_windows_csv))
    if not {"ii", "preict"}.issubset(set(windows.condition)):
        raise ValueError("Peak-window input must contain both genuine interictal and preictal rows")
    common = pd.read_csv(args.common_windows_csv)
    global_frame, state_frame = transition_metrics(windows, args.minimum_transitions)
    global_frame = merge_autonomic(global_frame, common)
    event, cells = participant_cells(global_frame)
    global_summary, global_contrasts = global_summaries(cells, args.bootstrap_draws, args.seed)
    subgroup_cells, subgroup_contrasts = subgroup_comparisons(event, args.bootstrap_draws, args.seed + 7)
    manifold_scores, manifold_loadings, manifold_trajectory, manifold_audit = fit_manifold(
        cells, args.bootstrap_draws, args.seed + 1)
    logistic_models, logistic_predictions, logistic_coefficients = nested_logistic(
        event, args.bootstrap_draws, args.seed + 2)
    state_event, state_associations = state_exploration(state_frame, common)
    matches, match_balance = match_pseudo_onsets(windows, common, args.pseudo_record_fraction)
    paired, did = pseudo_difference_in_differences(
        matches, windows, args.minimum_transitions, args.bootstrap_draws, args.seed + 3
    )

    outputs = {
        "global_window_metrics.csv.gz": global_frame,
        "state_window_metrics.csv.gz": state_frame,
        "global_event_bin_analysis_frame.csv.gz": event,
        "global_participant_period_cells.csv": cells,
        "global_trajectory_summary.csv": global_summary,
        "global_contrasts.csv": global_contrasts,
        "subgroup_participant_cells.csv": subgroup_cells,
        "subgroup_contrasts.csv": subgroup_contrasts,
        "manifold_scores.csv": manifold_scores,
        "manifold_loadings.csv": manifold_loadings,
        "manifold_trajectory.csv": manifold_trajectory,
        "stepwise_logistic_models.csv": logistic_models,
        "stepwise_logistic_predictions.csv.gz": logistic_predictions,
        "stepwise_logistic_coefficients.csv": logistic_coefficients,
        "state_event_bin_analysis_frame.csv.gz": state_event,
        "state_exploratory_associations.csv": state_associations,
        "matched_pseudo_pairs.csv": paired,
        "matched_pseudo_balance.csv": match_balance,
        "matched_pseudo_difference_in_differences.csv": did,
    }
    for name, frame in outputs.items():
        frame.to_csv(tables / name, index=False, compression="gzip" if name.endswith(".gz") else None)
    (tables / "manifold_audit.json").write_text(json.dumps(manifold_audit, indent=2), encoding="utf-8")
    decisions = analysis_decisions(); decisions.to_csv(out_dir / "ANALYSIS_DECISIONS.csv", index=False)

    retained = ("participant_demographics.csv", "demographic_summary.csv", "event_characteristics.csv", "hr_bayesian_cells.csv",
                "coverage_bayesian_cells.csv", "autonomic_level_participant_cells.csv",
                "autonomic_level_bayesian_cells.csv", "autonomic_level_bayesian_contrasts.csv")
    for name in retained:
        source = args.focused_dir / "data" / name
        if source.exists():
            shutil.copy2(source, tables / name)
    if args.manifest is not None:
        demographics, _ = refresh_demographics(args.manifest, args.demographics_csv, tables)
    else:
        demographics = pd.read_csv(tables / "demographic_summary.csv")

    configure_style()
    plot_global(cells, global_summary, global_contrasts, figures, args.seed + 4)
    plot_manifold(manifold_scores, manifold_loadings, manifold_audit, figures)
    plot_logistic(logistic_models, logistic_predictions, logistic_coefficients, figures)
    plot_state_exploration(state_event, state_associations, figures, args.seed + 5)
    plot_pseudo(paired, did, match_balance, figures, args.seed + 6)
    plot_subgroups(subgroup_cells, subgroup_contrasts, figures, args.seed + 7)

    action_summary = {}
    if not args.skip_action_tensor:
        action_config = ActionTensorConfig(
            minimum_transitions=args.minimum_transitions,
            action_temperature=args.action_temperature,
            action_mode="soft",
            action_lag_windows=args.action_lag_windows,
            pool_prior_strength=100.0,
            action_prior_grid=(2.0, 5.0, 10.0, 20.0, 50.0),
            bootstrap_draws=args.bootstrap_draws,
            seed=args.seed + 100,
            pseudo_record_fraction=args.action_pseudo_record_fraction,
        )
        action_directory = out_dir / "action_tensor"
        action_summary = run_action_tensor_analysis(
            args.peak_windows_csv,
            args.common_windows_csv,
            action_directory,
            action_config,
            make_plots=True,
            run_pseudo=True,
        )
        if args.action_sensitivities:
            run_action_tensor_sensitivities(
                args.peak_windows_csv,
                args.common_windows_csv,
                action_directory / "sensitivities",
                action_config,
            )

    counts = {"global_windows": len(global_frame), "state_windows": len(state_frame),
              "events": event.event_key.nunique(), "participants": event.participant.nunique(),
              "interictal_windows": int(global_frame.condition.eq("ii").sum()),
              "preictal_windows": int(global_frame.condition.eq("preict").sum()),
              "autonomic_complete_event_bins": int(event[list(MODEL_FEATURES)].notna().all(axis=1).sum()),
              "matched_pseudo_pairs": len(paired),
              "dd_paired_participants_final_minute": int(subgroup_contrasts[
                  subgroup_contrasts.comparison.eq("DD paired FS minus ES") &
                  subgroup_contrasts.time_bin.eq("-1 to 0")].n_paired.max()),
              "fs_hypo_hyper_participants_final_minute": int(subgroup_contrasts[
                  subgroup_contrasts.comparison.eq("FS hypo minus hyper") &
                  subgroup_contrasts.time_bin.eq("-1 to 0")].n_participants.max())}
    write_report(out_dir, global_summary, global_contrasts, subgroup_contrasts, demographics, logistic_models, did,
                 state_associations, manifold_audit, counts)
    summary = counts | {
        "strategy": "global transition grammar first; source states exploratory; action-conditioned mechanistic extension",
        "action_tensor": action_summary,
        "minimum_transitions": args.minimum_transitions,
        "microstate_refitting_performed": False,
        "manifold_explained_variance_2d": float(sum(manifold_audit["explained_variance_ratio"])),
        "full_preictal_auc": float(logistic_models[
            logistic_models.scope.eq("all preictal bins") & logistic_models.step.eq(4)].auc.iloc[0]),
        "final_minute_auc": float(logistic_models[
            logistic_models.scope.eq("final minute") & logistic_models.step.eq(4)].auc.iloc[0]),
    }
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    completed = [path for path in out_dir.rglob("*") if path.is_file()]
    audit = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "code": str(Path(__file__).resolve()), "code_sha256": sha256(Path(__file__)),
        "inputs": {str(path): sha256(path) for path in
                   (args.peak_windows_csv, args.common_windows_csv, args.manifest, args.demographics_csv)
                   if path is not None},
        "estimands": {"global_js": "transition-count-weighted JS across six source grammars",
                      "global_kappa": "joint maximum-likelihood baseline-oriented concentration across six sources",
                      "interictal_reference": "participant pooled II, leave-current-window-out for II rows",
                      "dd_comparison": "paired participant mean FS minus ES within dual-event-type participants",
                      "fs_semiology_comparison": "participant-semiology mean hypo minus hyper with participant-cluster bootstrap",
                      "fs_es_global_contrast": "participant-cluster bootstrap preserving dual-diagnosis overlap",
                      "pseudo_reference": "common participant II grammar excluding the matched pseudo-window for both pair members",
                      "action_tensor": "P(next distinct microstate | current microstate, CPI/CSI action region)"},
        "minimum_transitions": args.minimum_transitions,
        "bootstrap_draws": args.bootstrap_draws,
        "random_seed": args.seed,
        "software": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "matplotlib")},
        "outputs": {str(path.relative_to(out_dir)): sha256(path) for path in completed},
        **summary,
    }
    (out_dir / "PUBLICATION_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return summary


def atomic_build(args) -> dict:
    target = args.out_dir.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-building-", dir=target.parent))
    backup = target.with_name(f".{target.name}-previous-{time.time_ns()}")
    try:
        summary = build(args, temporary)
        promote_directory(temporary, target, backup, args.overwrite)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise


def remove_with_retries(path: Path, attempts: int = 4) -> bool:
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return True
        except PermissionError:
            if attempt + 1 < attempts:
                time.sleep(.5 * 2 ** attempt)
    return False


def promote_directory(temporary: Path, target: Path, backup: Path, overwrite: bool) -> None:
    if not target.exists():
        temporary.rename(target)
        return
    if not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {target}")
    try:
        target.rename(backup)
    except PermissionError:
        shutil.copytree(target, backup)
        shutil.copytree(temporary, target, dirs_exist_ok=True)
        for source in (path for path in temporary.rglob("*") if path.is_file()):
            relative = source.relative_to(temporary)
            if sha256(source) != sha256(target / relative):
                raise OSError(f"OneDrive fallback copy verification failed: {relative}")
        if not remove_with_retries(temporary):
            print(f"Warning: OneDrive retained temporary folder {temporary}")
    else:
        try:
            temporary.rename(target)
        except Exception:
            backup.rename(target)
            raise
    if backup.exists() and not remove_with_retries(backup):
        print(f"Warning: OneDrive retained recoverable backup {backup}")


def build_review_package(args) -> None:
    if args.review_dir is None:
        return
    target = args.review_dir.resolve(); target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-building-", dir=target.parent))
    backup = target.with_name(f".{target.name}-previous-{time.time_ns()}")
    try:
        (temporary / "code").mkdir(); (temporary / "input").mkdir()
        shutil.copy2(Path(__file__), temporary / "code" / Path(__file__).name)
        action_code = Path(__file__).with_name("analyse_action_conditioned_transition_tensor.py")
        if action_code.exists():
            shutil.copy2(action_code, temporary / "code" / action_code.name)
        peak = pd.read_csv(args.peak_windows_csv)
        peak.to_csv(temporary / "input" / "directional_peak_backfit_windows_all.csv.gz",
                    index=False, compression="gzip")
        shutil.copy2(args.common_windows_csv, temporary / "input" / "autonomic_peak_microstate_common_windows.csv.gz")
        shutil.copytree(args.out_dir / "tables", temporary / "tables")
        shutil.copytree(args.out_dir / "figures", temporary / "figures")
        if (args.out_dir / "action_tensor").exists():
            shutil.copytree(args.out_dir / "action_tensor", temporary / "action_tensor")
        for name in ("RESULTS.md", "ANALYSIS_DECISIONS.csv", "SUMMARY.json", "PUBLICATION_AUDIT.json"):
            shutil.copy2(args.out_dir / name, temporary / name)
        readme = [
            "# Review package", "",
            "This is the complete small package needed to inspect the global-first FS/ES findings.", "",
            "- `RESULTS.md`: concise interpretation and figures.",
            "- `tables/`: exact analysis frames, contrasts, PCA scores/loadings, held-out predictions and exploratory models.",
            "- `input/`: reused K=6 peak-backfit counts and aligned autonomic windows; no microstate refit is needed.",
            "- `code/analyse_global_transition_exploration.py`: the integrated analysis entry point.",
            "- `code/analyse_action_conditioned_transition_tensor.py`: the action-conditioned B-matrix analogue.",
            "- `action_tensor/`: tensor outputs, validation, matched pseudo-onsets and specification sensitivities.", "",
            "Re-run from the repository environment by pointing `--peak-windows-csv` and `--common-windows-csv` at the two files under `input/`, and choosing a new `--out-dir`. The SHA-256 ledger below makes every file traceable.",
        ]
        (temporary / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
        files = [path for path in temporary.rglob("*") if path.is_file()]
        pd.DataFrame({"path": [str(path.relative_to(temporary)) for path in files],
                      "sha256": [sha256(path) for path in files],
                      "bytes": [path.stat().st_size for path in files]}).to_csv(
                          temporary / "SHA256SUMS.csv", index=False)
        promote_directory(temporary, target, backup, args.overwrite)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise


def self_check() -> None:
    counts = np.zeros((6, 6))
    for source in range(6):
        counts[source, np.arange(6) != source] = (200, 400, 600, 800, 1000)
    baseline = probability(counts)
    assert js_divergence(counts, baseline, tuple(range(6))) < 1e-12
    assert abs(fit_log_kappa(counts, baseline, tuple(range(6)), 30)) < .002
    p = pd.Series([.04, .001, np.nan, .2])
    adjusted = bh(p)
    assert adjusted.iloc[1] <= adjusted.iloc[0] and pd.isna(adjusted.iloc[2])
    subgroup_rows = []
    for time_bin in TIME_BINS:
        for participant, event_type, semiology, value in (
                ("d1", "functional", "hypo", 3.), ("d1", "epileptic", "hyper", 2.),
                ("d2", "functional", "hypo", 4.), ("d2", "epileptic", "hyper", 3.),
                ("f1", "functional", "hypo", 3.), ("f2", "functional", "hypo", 4.),
                ("f3", "functional", "hyper", 2.), ("f4", "functional", "hyper", 2.)):
            subgroup_rows.append({"participant": participant, "event_key": participant + time_bin,
                                  "cohort_group": "DD" if participant.startswith("d") else "FS",
                                  "event_type": event_type, "semiology": semiology, "time_bin": time_bin,
                                  "log_js": value, "log_kappa": value})
    _, subgroup = subgroup_comparisons(pd.DataFrame(subgroup_rows), 100, 7)
    assert np.allclose(subgroup[subgroup.comparison.eq("DD paired FS minus ES")].estimate, 1)
    assert np.allclose(subgroup[subgroup.comparison.eq("FS hypo minus hyper")].estimate, 1.5)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); target = root / "target"; temporary = root / "temporary"
        target.mkdir(); temporary.mkdir(); (target / "old.txt").write_text("old")
        (temporary / "new.txt").write_text("new")
        promote_directory(temporary, target, root / "backup", True)
        assert (target / "new.txt").read_text() == "new" and not (root / "backup").exists()
    print("self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peak-windows-csv", type=Path)
    parser.add_argument("--common-windows-csv", type=Path)
    parser.add_argument("--focused-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--demographics-csv", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--minimum-transitions", type=int, default=MIN_TRANSITIONS)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--pseudo-record-fraction", type=float, default=0.99)
    parser.add_argument("--skip-action-tensor", action="store_true")
    parser.add_argument("--action-sensitivities", action="store_true")
    parser.add_argument("--action-temperature", type=float, default=0.5)
    parser.add_argument("--action-lag-windows", type=int, choices=(0, 1), default=0)
    parser.add_argument("--action-pseudo-record-fraction", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=31072026)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--package-only", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(); return
    required = (args.peak_windows_csv, args.common_windows_csv, args.focused_dir, args.out_dir)
    if any(value is None for value in required):
        parser.error("--peak-windows-csv, --common-windows-csv, --focused-dir and --out-dir are required")
    if (args.manifest is None) != (args.demographics_csv is None):
        parser.error("--manifest and --demographics-csv must be supplied together")
    for path in (args.peak_windows_csv, args.common_windows_csv, args.focused_dir,
                 args.manifest, args.demographics_csv):
        if path is None:
            continue
        if not path.exists():
            parser.error(f"Input does not exist: {path}")
    if args.minimum_transitions != 30:
        raise ValueError("This publication pipeline fixes the prespecified transition threshold at 30")
    if args.package_only:
        if args.review_dir is None or not args.out_dir.exists():
            parser.error("--package-only requires an existing --out-dir and --review-dir")
        build_review_package(args); print(f"Review package written to {args.review_dir}"); return
    summary = atomic_build(args)
    build_review_package(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
