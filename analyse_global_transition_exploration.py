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
    event = (preictal.groupby(["participant", "event_key", "event_type", "semiology", "time_bin"],
                              observed=True, dropna=False)[[*MODEL_FEATURES, "global_js", "kappa"]]
             .mean().reset_index())
    preictal_cells = (event.groupby(["participant", "event_type", "time_bin"], observed=True)
                      [[*MODEL_FEATURES, "global_js", "kappa"]].mean().reset_index())
    participant_types = preictal[["participant", "event_type"]].drop_duplicates()
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


def global_summaries(cells: pd.DataFrame, draws: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows, contrasts = [], []
    for outcome in ("log_js", "log_kappa"):
        for group in ("FS", "ES"):
            for time_bin in TIME_ORDER:
                part = cells[cells.group.eq(group) & cells.time_bin.eq(time_bin)]
                estimate, low, high = bootstrap_interval(part[outcome].to_numpy(float), rng, draws)
                rows.append({"outcome": outcome, "group": group, "time_bin": time_bin,
                             "estimate": estimate, "ci95_low": low, "ci95_high": high,
                             "n_participants": part.participant.nunique()})
        for time_bin in TIME_ORDER:
            fs = cells[cells.group.eq("FS") & cells.time_bin.eq(time_bin)][outcome].dropna().to_numpy(float)
            es = cells[cells.group.eq("ES") & cells.time_bin.eq(time_bin)][outcome].dropna().to_numpy(float)
            if not len(fs) or not len(es):
                continue
            samples = (np.mean(fs[rng.integers(0, len(fs), size=(draws, len(fs)))], axis=1) -
                       np.mean(es[rng.integers(0, len(es), size=(draws, len(es)))], axis=1))
            low, high = np.quantile(samples, (.025, .975))
            contrasts.append({"outcome": outcome, "contrast": "FS minus ES", "time_bin": time_bin,
                              "estimate": float(np.mean(fs) - np.mean(es)), "ci95_low": low,
                              "ci95_high": high, "probability_gt_zero": float(np.mean(samples > 0)),
                              "n_fs": len(fs), "n_es": len(es)})
        for group in ("FS", "ES"):
            pivot = cells[cells.group.eq(group)].pivot(index="participant", columns="time_bin", values=outcome)
            for time_bin in TIME_BINS:
                pair = pivot[["II", time_bin]].dropna()
                difference = (pair[time_bin] - pair["II"]).to_numpy(float)
                estimate, low, high = bootstrap_interval(difference, rng, draws)
                contrasts.append({"outcome": outcome, "contrast": f"{group} {time_bin} minus II",
                                  "time_bin": time_bin, "estimate": estimate, "ci95_low": low,
                                  "ci95_high": high, "probability_gt_zero": np.nan,
                                  "n_fs": len(pair) if group == "FS" else 0,
                                  "n_es": len(pair) if group == "ES" else 0})
    summary = pd.DataFrame(rows)
    for column in ("estimate", "ci95_low", "ci95_high"):
        summary[f"natural_{column}"] = np.exp(summary[column])
    contrast = pd.DataFrame(contrasts)
    contrast["ratio"] = np.exp(contrast.estimate)
    contrast["ratio_ci95_low"] = np.exp(contrast.ci95_low)
    contrast["ratio_ci95_high"] = np.exp(contrast.ci95_high)
    return summary, contrast


def match_pseudo_onsets(windows: pd.DataFrame, common: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = merge_autonomic(windows, common)
    source["record_fraction"] = source.window_end_s / source.groupby("file_path", observed=True).window_end_s.transform("max")
    source["has_autonomic"] = source[["n_native", "csi_z", "cpi_z"]].notna().all(axis=1)
    true = source[source.condition.eq("preict") & source.window_end_to_ictus_s.eq(0) &
                  source.event_type.isin(("functional", "epileptic"))]
    pseudo = source[source.condition.eq("ii") & source.record_fraction.ge(.8)]
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


def pseudo_difference_in_differences(matches: pd.DataFrame, global_frame: pd.DataFrame,
                                     draws: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = global_frame.set_index("row_id")[["global_js", "log_kappa"]]
    paired = matches.copy()
    for column in ("global_js", "log_kappa"):
        true = paired.true_row_id.map(values[column])
        pseudo = paired.pseudo_row_id.map(values[column])
        if column == "global_js":
            true = np.log(true.clip(lower=1e-8)); pseudo = np.log(pseudo.clip(lower=1e-8))
        paired[f"true_{column}"] = true
        paired[f"pseudo_{column}"] = pseudo
        paired[f"difference_{column}"] = true - pseudo
    participant = (paired.groupby(["participant", "event_type"], observed=True)
                   [["difference_global_js", "difference_log_kappa"]].mean().reset_index())
    participants = participant.participant.unique()
    rng = np.random.default_rng(seed)
    rows = []
    for outcome in ("global_js", "log_kappa"):
        column = f"difference_{outcome}"
        complete = participant.dropna(subset=[column]).copy()
        fs = complete[complete.event_type.eq("functional")][column]
        es = complete[complete.event_type.eq("epileptic")][column]
        observed = fs.mean() - es.mean()
        samples = []
        participants = complete.participant.unique()
        for _ in range(draws):
            chosen = rng.choice(participants, len(participants), replace=True)
            weights = pd.Series(chosen).value_counts()
            use = complete.merge(weights.rename("weight"), left_on="participant", right_index=True, how="inner")
            means = use.groupby("event_type", observed=True).apply(
                lambda part: np.average(part[column], weights=part.weight), include_groups=False)
            if {"functional", "epileptic"}.issubset(means.index):
                samples.append(means["functional"] - means["epileptic"])
        samples = np.asarray(samples, float)
        low, high = np.quantile(samples, (.025, .975))
        rows.append({"outcome": "log JS" if outcome == "global_js" else "log kappa",
                     "contrast": "[FS genuine - pseudo] - [ES genuine - pseudo]",
                     "estimate": observed, "ci95_low": low, "ci95_high": high,
                     "probability_gt_zero": float(np.mean(samples > 0)),
                     "n_pairs": int(paired[column].notna().sum()), "n_participants": complete.participant.nunique()})
    return paired, pd.DataFrame(rows)


def fit_manifold(cells: pd.DataFrame, draws: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    use = cells.dropna(subset=list(MODEL_FEATURES)).copy()
    winsor_limits = {}
    for feature in MODEL_FEATURES:
        low, high = use[feature].quantile((.01, .99))
        winsor_limits[feature] = [float(low), float(high)]
        use[feature] = use[feature].clip(low, high)
    scaler = RobustScaler(quantile_range=(25, 75))
    scaled = scaler.fit_transform(use[list(MODEL_FEATURES)])
    pca = PCA(n_components=2, random_state=seed)
    scores = pca.fit_transform(scaled)
    components = pca.components_.copy()
    signs = np.array([1 if components[component, MODEL_FEATURES.index(anchor)] >= 0 else -1
                      for component, anchor in enumerate(("log_js", "cpi_z"))])
    scores *= signs
    components *= signs[:, None]
    use[["PC1", "PC2"]] = scores
    loadings = pd.DataFrame(components.T, index=MODEL_FEATURES, columns=("PC1", "PC2")).reset_index(names="feature")
    loadings["feature_label"] = loadings.feature.map(FEATURE_LABELS)
    trajectory_rows = []
    rng = np.random.default_rng(seed + 101)
    for group in ("FS", "ES"):
        for time_bin in TIME_ORDER:
            part = use[use.group.eq(group) & use.time_bin.eq(time_bin)]
            for component in ("PC1", "PC2"):
                estimate, low, high = bootstrap_interval(part[component].to_numpy(float), rng, draws)
                trajectory_rows.append({"group": group, "time_bin": time_bin, "component": component,
                                        "estimate": estimate, "ci95_low": low, "ci95_high": high,
                                        "n_participants": part.participant.nunique()})
    trajectory = pd.DataFrame(trajectory_rows)
    audit = {"features": list(MODEL_FEATURES), "scaler": "median/IQR",
             "winsorisation": "1st/99th percentiles before PCA only", "winsor_limits": winsor_limits,
             "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
             "scaler_center": dict(zip(MODEL_FEATURES, scaler.center_.tolist())),
             "scaler_scale": dict(zip(MODEL_FEATURES, scaler.scale_.tolist())),
             "fit_rows": len(use), "participants": use.participant.nunique()}
    return use, loadings, trajectory, audit


def participant_weights(frame: pd.DataFrame) -> np.ndarray:
    count = frame.groupby("participant", observed=True).participant.transform("size").to_numpy(float)
    labels = frame.event_type.eq("functional").to_numpy(int)
    class_count = np.bincount(labels, minlength=2)
    class_weight = np.array([len(labels) / (2 * max(value, 1)) for value in class_count])
    return class_weight[labels] / count


def logistic_scope(frame: pd.DataFrame, scope: str, draws: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    use = frame.dropna(subset=list(MODEL_FEATURES)).copy()
    use["target_fs"] = use.event_type.eq("functional").astype(int)
    if use.target_fs.nunique() != 2:
        raise ValueError(f"{scope} logistic data do not contain both FS and ES")
    groups = use.participant.astype(str).to_numpy()
    y = use.target_fs.to_numpy(int)
    logo = LeaveOneGroupOut()
    rows, prediction_rows, coefficient_rows = [], [], []
    rng = np.random.default_rng(seed)
    for step in range(1, len(MODEL_FEATURES) + 1):
        features = list(MODEL_FEATURES[:step])
        probabilities = np.full(len(use), np.nan)
        x = use[features].to_numpy(float)
        for train, test in logo.split(x, y, groups):
            scaler = RobustScaler(quantile_range=(25, 75)).fit(x[train])
            model = LogisticRegression(C=1, class_weight=None, max_iter=1000, solver="lbfgs")
            model.fit(scaler.transform(x[train]), y[train], sample_weight=participant_weights(use.iloc[train]))
            probabilities[test] = model.predict_proba(scaler.transform(x[test]))[:, 1]
        auc = roc_auc_score(y, probabilities)
        boot_auc = []
        unique = np.unique(groups)
        for _ in range(draws):
            chosen = rng.choice(unique, len(unique), replace=True)
            indices = np.concatenate([np.flatnonzero(groups == participant) for participant in chosen])
            if np.unique(y[indices]).size == 2:
                boot_auc.append(roc_auc_score(y[indices], probabilities[indices]))
        low, high = np.quantile(boot_auc, (.025, .975))
        rows.append({"scope": scope, "step": step, "features": " + ".join(FEATURE_LABELS[name] for name in features),
                     "auc": auc, "auc_ci95_low": low, "auc_ci95_high": high,
                     "log_loss": log_loss(y, probabilities), "brier_score": brier_score_loss(y, probabilities),
                     "n_rows": len(use), "n_events": use.event_key.nunique(),
                     "n_participants": use.participant.nunique()})
        prediction_rows.extend({"scope": scope, "step": step, "participant": participant,
                                "event_key": event, "event_type": event_type, "target_fs": target,
                                "probability_fs": probability}
                               for participant, event, event_type, target, probability in zip(
                                   use.participant, use.event_key, use.event_type, y, probabilities))

        scaler = RobustScaler(quantile_range=(25, 75)).fit(x)
        transformed = scaler.transform(x)
        fitted = LogisticRegression(C=1, class_weight=None, max_iter=1000, solver="lbfgs")
        fitted.fit(transformed, y, sample_weight=participant_weights(use))
        bootstrap_coefficients = np.empty((draws, step))
        unique = np.unique(groups)
        counts = use.groupby("participant", observed=True).participant.transform("size").to_numpy(float)
        class_count = np.bincount(y, minlength=2)
        class_weight = np.array([len(y) / (2 * max(value, 1)) for value in class_count])
        for draw in range(draws):
            sampled = pd.Series(rng.choice(unique, len(unique), replace=True)).value_counts()
            weight = use.participant.map(sampled).fillna(0).to_numpy(float) / counts * class_weight[y]
            replicate = LogisticRegression(C=1, class_weight=None, max_iter=1000, solver="lbfgs")
            replicate.fit(transformed, y, sample_weight=weight)
            bootstrap_coefficients[draw] = replicate.coef_[0]
        for position, feature in enumerate(features):
            low_coef, high_coef = np.quantile(bootstrap_coefficients[:, position], (.025, .975))
            coefficient_rows.append({"scope": scope, "step": step, "feature": feature,
                                     "feature_label": FEATURE_LABELS[feature],
                                     "coefficient": fitted.coef_[0, position],
                                     "ci95_low": low_coef, "ci95_high": high_coef,
                                     "odds_ratio": math.exp(fitted.coef_[0, position]),
                                     "odds_ratio_ci95_low": math.exp(low_coef),
                                     "odds_ratio_ci95_high": math.exp(high_coef)})
    return pd.DataFrame(rows), pd.DataFrame(prediction_rows), pd.DataFrame(coefficient_rows)


def nested_logistic(event: pd.DataFrame, draws: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_results = []
    for offset, (scope, frame) in enumerate((
            ("all preictal bins", event),
            ("final minute", event[event.time_bin.eq("-1 to 0")]))):
        all_results.append(logistic_scope(frame, scope, draws, seed + offset * 10007))
    return tuple(pd.concat([result[index] for result in all_results], ignore_index=True) for index in range(3))


def linear_result(fitted, vector: dict[str, float], label: str, **metadata) -> dict:
    weights = pd.Series(0.0, index=fitted.params.index)
    for name, value in vector.items():
        if name not in weights:
            raise KeyError(name)
        weights[name] = value
    estimate = float(weights @ fitted.params)
    variance = float(weights @ fitted.cov_params() @ weights)
    standard_error = math.sqrt(max(variance, 0))
    z = estimate / standard_error if standard_error > 0 else np.nan
    from scipy.stats import norm
    p_value = float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return metadata | {"association": label, "estimate": estimate, "standard_error": standard_error,
                       "ci95_low": estimate - 1.96 * standard_error,
                       "ci95_high": estimate + 1.96 * standard_error, "p_value": p_value}


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


def plot_pseudo(paired: pd.DataFrame, did: pd.DataFrame, balance: pd.DataFrame,
                figures: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 3, figsize=(7.7, 3.1), layout="constrained",
                             gridspec_kw={"width_ratios": (1, .9, 1.2)})
    for position, (event_type, group, color) in enumerate((
            ("functional", "FS", FS_COLOR), ("epileptic", "ES", ES_COLOR))):
        values = paired[paired.event_type.eq(event_type)].difference_global_js.dropna().to_numpy(float)
        axes[0].scatter(position + rng.uniform(-.14, .14, len(values)), values,
                        color=color, alpha=.16, s=8, linewidths=0)
        axes[0].scatter(position, np.mean(values), marker="D", color=color,
                        edgecolor="white", linewidth=.7, s=38, zorder=4)
    axes[0].axhline(0, color=GREY, lw=.8)
    axes[0].set(xticks=(0, 1), xticklabels=("FS", "ES"), ylabel="Genuine − pseudo log JS",
                title="Matched event differences")
    axes[0].grid(axis="y"); panel(axes[0], "A")

    y = np.arange(len(did)); middle = did.estimate.to_numpy(float)
    axes[1].errorbar(middle, y, xerr=[middle - did.ci95_low, did.ci95_high - middle],
                     fmt="o", color=PURPLE, capsize=3, ms=5)
    axes[1].axvline(0, color=GREY, lw=.8)
    axes[1].set(yticks=y, yticklabels=did.outcome, xlabel="Matched difference-in-differences",
                title="Onset-specific sensitivity")
    axes[1].invert_yaxis(); axes[1].grid(axis="x"); panel(axes[1], "B")

    display = balance[~balance.feature.eq("record_fraction")].sort_values("standardised_mean_difference")
    y = np.arange(len(display))
    axes[2].barh(y, display.standardised_mean_difference, color=np.where(
        display.standardised_mean_difference.abs() <= .1, GREEN, ES_COLOR), alpha=.78)
    axes[2].axvline(0, color=GREY, lw=.7); axes[2].axvline(-.1, color=GREY, lw=.6, linestyle="--")
    axes[2].axvline(.1, color=GREY, lw=.6, linestyle="--")
    axes[2].set(yticks=y, yticklabels=display.feature.str.replace("_", " "),
                xlabel="Standardised mean difference", title="True/pseudo match balance")
    axes[2].grid(axis="x"); panel(axes[2], "C")
    axes[2].text(.98, -.19, "II candidates restricted to final 20% of record",
                 transform=axes[2].transAxes, ha="right", va="top", fontsize=6.5, color=GREY)
    save_figure(fig, figures, "figure_05_matched_pseudo_sensitivity")


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    use = frame[columns].copy()
    for column in use.select_dtypes(include="number"):
        use[column] = use[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    return ("| " + " | ".join(columns) + " |\n|" + "|".join(["---"] * len(columns)) + "|\n" +
            "\n".join("| " + " | ".join(map(str, row)) + " |" for row in use.itertuples(index=False, name=None)))


def write_report(out_dir: Path, summary: pd.DataFrame, contrasts: pd.DataFrame,
                 logistic: pd.DataFrame, did: pd.DataFrame, associations: pd.DataFrame,
                 manifold_audit: dict, counts: dict) -> None:
    final = contrasts[contrasts.contrast.eq("FS minus ES") & contrasts.time_bin.eq("-1 to 0")]
    full = logistic[logistic.scope.eq("all preictal bins") & logistic.step.eq(4)].iloc[0]
    initial = logistic[logistic.scope.eq("all preictal bins") & logistic.step.eq(1)].iloc[0]
    top = associations.sort_values("p_bh").head(8)
    js_did = did[did.outcome.eq("log JS")].iloc[0]
    kappa_did = did[did.outcome.eq("log kappa")].iloc[0]
    js_specific = js_did.ci95_low > 0 or js_did.ci95_high < 0
    kappa_specific = kappa_did.ci95_low > 0 or kappa_did.ci95_high < 0
    report = [
        "# Global-first transition analysis", "",
        "## What this analysis changes", "",
        "No microstate is privileged at the outset. The headline analysis uses the entire six-state transition grammar: transition-count-weighted Jensen–Shannon divergence from each participant's interictal baseline and one joint baseline-oriented κ. Source-state, autonomic and semiological results are exploratory and BH-FDR controlled.", "",
        f"The analysis retained windows with at least {MIN_TRANSITIONS} off-diagonal transitions. Interictal windows use a leave-that-window-out participant reference; preictal windows use the participant's pooled interictal reference. Microstate fits were reused, not refitted.", "",
        "## Headline global findings", "",
        markdown_table(final[["outcome", "estimate", "ci95_low", "ci95_high", "ratio", "ratio_ci95_low", "ratio_ci95_high"]],
                       ["outcome", "estimate", "ci95_low", "ci95_high", "ratio", "ratio_ci95_low", "ratio_ci95_high"]), "",
        "The global trajectories are participant-equal summaries; points show participant cells and intervals are participant bootstraps. Ratios above one indicate higher FS values.", "",
        "![Global trajectories](figures/figure_01_global_transition_trajectories.png)", "",
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
        markdown_table(did, ["outcome", "estimate", "ci95_low", "ci95_high", "probability_gt_zero", "n_pairs", "n_participants"]), "",
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
    manifold_scores, manifold_loadings, manifold_trajectory, manifold_audit = fit_manifold(
        cells, args.bootstrap_draws, args.seed + 1)
    logistic_models, logistic_predictions, logistic_coefficients = nested_logistic(
        event, args.bootstrap_draws, args.seed + 2)
    state_event, state_associations = state_exploration(state_frame, common)
    matches, match_balance = match_pseudo_onsets(windows, common)
    paired, did = pseudo_difference_in_differences(matches, global_frame, args.bootstrap_draws, args.seed + 3)

    outputs = {
        "global_window_metrics.csv.gz": global_frame,
        "state_window_metrics.csv.gz": state_frame,
        "global_event_bin_analysis_frame.csv.gz": event,
        "global_participant_period_cells.csv": cells,
        "global_trajectory_summary.csv": global_summary,
        "global_contrasts.csv": global_contrasts,
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

    retained = ("demographic_summary.csv", "event_characteristics.csv", "hr_bayesian_cells.csv",
                "coverage_bayesian_cells.csv", "autonomic_level_participant_cells.csv",
                "autonomic_level_bayesian_cells.csv", "autonomic_level_bayesian_contrasts.csv")
    for name in retained:
        source = args.focused_dir / "data" / name
        if source.exists():
            shutil.copy2(source, tables / name)

    configure_style()
    plot_global(cells, global_summary, global_contrasts, figures, args.seed + 4)
    plot_manifold(manifold_scores, manifold_loadings, manifold_audit, figures)
    plot_logistic(logistic_models, logistic_predictions, logistic_coefficients, figures)
    plot_state_exploration(state_event, state_associations, figures, args.seed + 5)
    plot_pseudo(paired, did, match_balance, figures, args.seed + 6)

    counts = {"global_windows": len(global_frame), "state_windows": len(state_frame),
              "events": event.event_key.nunique(), "participants": event.participant.nunique(),
              "interictal_windows": int(global_frame.condition.eq("ii").sum()),
              "preictal_windows": int(global_frame.condition.eq("preict").sum()),
              "autonomic_complete_event_bins": int(event[list(MODEL_FEATURES)].notna().all(axis=1).sum()),
              "matched_pseudo_pairs": len(paired)}
    write_report(out_dir, global_summary, global_contrasts, logistic_models, did,
                 state_associations, manifold_audit, counts)
    summary = counts | {
        "strategy": "global transition grammar first; source states exploratory",
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
        "inputs": {str(args.peak_windows_csv): sha256(args.peak_windows_csv),
                   str(args.common_windows_csv): sha256(args.common_windows_csv)},
        "estimands": {"global_js": "transition-count-weighted JS across six source grammars",
                      "global_kappa": "joint maximum-likelihood baseline-oriented concentration across six sources",
                      "interictal_reference": "participant pooled II, leave-current-window-out for II rows"},
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
    backup = target.with_name(f".{target.name}-previous")
    if backup.exists():
        shutil.rmtree(temporary)
        raise FileExistsError(f"Refusing overwrite because backup exists: {backup}")
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
    backup = target.with_name(f".{target.name}-previous")
    if backup.exists():
        shutil.rmtree(temporary)
        raise FileExistsError(f"Refusing review overwrite because backup exists: {backup}")
    try:
        (temporary / "code").mkdir(); (temporary / "input").mkdir()
        shutil.copy2(Path(__file__), temporary / "code" / Path(__file__).name)
        peak = pd.read_csv(args.peak_windows_csv)
        peak.to_csv(temporary / "input" / "directional_peak_backfit_windows_all.csv.gz",
                    index=False, compression="gzip")
        shutil.copy2(args.common_windows_csv, temporary / "input" / "autonomic_peak_microstate_common_windows.csv.gz")
        shutil.copytree(args.out_dir / "tables", temporary / "tables")
        shutil.copytree(args.out_dir / "figures", temporary / "figures")
        for name in ("RESULTS.md", "ANALYSIS_DECISIONS.csv", "SUMMARY.json", "PUBLICATION_AUDIT.json"):
            shutil.copy2(args.out_dir / name, temporary / name)
        readme = [
            "# Review package", "",
            "This is the complete small package needed to inspect the global-first FS/ES findings.", "",
            "- `RESULTS.md`: concise interpretation and figures.",
            "- `tables/`: exact analysis frames, contrasts, PCA scores/loadings, held-out predictions and exploratory models.",
            "- `input/`: reused K=6 peak-backfit counts and aligned autonomic windows; no microstate refit is needed.",
            "- `code/analyse_global_transition_exploration.py`: the single analysis entry point.", "",
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
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--minimum-transitions", type=int, default=MIN_TRANSITIONS)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
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
    for path in (args.peak_windows_csv, args.common_windows_csv, args.focused_dir):
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
