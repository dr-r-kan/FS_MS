"""Action-conditioned EEG microstate transition-tensor analysis.

This module estimates an empirical action-conditioned transition tensor

    P(next microstate | current microstate, autonomic action)

where autonomic action is represented by four CPI/CSI sign combinations.  A
soft quadrant interpolation is primary: every 60-s window contributes its
transition counts fractionally to the four action corners, with weights that
sum exactly to one.  This avoids arbitrary discontinuities at zero and, unlike
naive temporal upsampling, does not create pseudo-independent observations.

The participant's interictal (II) action tensor is the reference.  Action rows
are partially pooled toward a participant-specific transition grammar modulated
by a leave-participant-out population action effect.  II windows are evaluated
leave-one-window-out.  Preictal windows are evaluated against II only.

Outputs include:
- action-conditioned Jensen-Shannon divergence;
- kappa, a gain on the complete action-conditioned II tensor (kappa=1 preserves
  II route probabilities; <1 flattens them; >1 sharpens them);
- lambda_action, a gain specifically on the action-dependent deviation from the
  participant's action-pooled II grammar (lambda=1 preserves action specificity;
  lambda=0 removes it);
- leave-one-window-out II predictive log-score gain of the action-conditioned
  tensor over an action-pooled transition model;
- participant-cluster bootstrap contrasts and genuine/pseudo-onset sensitivity;
- window-by-source-by-destination probabilities, expected counts, residuals, and
  exploratory route contrasts (the finest valid resolution of 60-s counts);
- publication-grade PNG and vector PDF figures.

The analysis is observational.  It tests an action-conditioned transition model
and temporal-ordering sensitivities; it does not by itself identify causal action
or a literal active-inference B matrix.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import expit


STATES = ("F", "E", "G", "B", "A", "C")
ACTIONS = ("CPI+ / CSI+", "CPI+ / CSI−", "CPI− / CSI+", "CPI− / CSI−")
ACTION_SHORT = ("++", "+−", "−+", "−−")
TIME_BINS = ("-20 to -10", "-10 to -5", "-5 to -2", "-2 to -1", "-1 to 0")
TIME_ORDER = ("II", *TIME_BINS)
COUNT_COLUMNS = tuple(
    f"t_{source}_{destination}"
    for source in range(1, 7)
    for destination in range(1, 7)
)
FS_COLOR = "#0072B2"
ES_COLOR = "#D55E00"
GREY = "#667085"
LIGHT_GREY = "#D0D5DD"
GREEN = "#009E73"
PURPLE = "#7A5195"
ACTION_COLORS = ("#0072B2", "#56B4E9", "#E69F00", "#D55E00")
OFFDIAGONAL_MASK = ~np.eye(6, dtype=bool)


@dataclass(frozen=True)
class ActionTensorConfig:
    minimum_transitions: int = 30
    action_temperature: float = 0.5
    action_mode: str = "soft"
    action_lag_windows: int = 0
    pool_prior_strength: float = 100.0
    action_prior_grid: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0, 50.0)
    bootstrap_draws: int = 2000
    seed: int = 31072026
    pseudo_record_fraction: float = 0.99
    pseudo_native_caliper: float = 10.0
    pseudo_action_caliper: float = 1.0


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.55,
        "grid.alpha": 0.18,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise_path(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace("/", "\\", regex=False).str.lower()


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel(axis: plt.Axes, label: str) -> None:
    axis.text(-0.14, 1.12, label, transform=axis.transAxes, fontsize=10.5,
              fontweight="bold", ha="left", va="bottom")


def markdown_table(frame: pd.DataFrame) -> str:
    use = frame.copy()
    for column in use.select_dtypes(include="number"):
        use[column] = use[column].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    return ("| " + " | ".join(use.columns) + " |\n|" + "|".join(["---"] * len(use.columns)) + "|\n" +
            "\n".join("| " + " | ".join(map(str, row)) + " |" for row in use.itertuples(index=False, name=None)))


def enrich_windows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "participant", "file_path", "condition", "seizure_type", "window_start_s",
        "window_end_s", "window_end_to_ictus_s", *COUNT_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Peak-window input is missing columns: {missing}")
    output = frame.copy()
    output["row_id"] = np.arange(len(output), dtype=int)
    output["file_key"] = normalise_path(output.file_path)
    output["event_key"] = output.file_path.astype(str)
    output["event_type"] = output.seizure_type.astype(str).str.upper().map(
        {"FS": "functional", "ES": "epileptic"})
    output["semiology"] = (output.get("hypo_hyper", pd.Series(index=output.index, dtype=object))
                            .astype(str).str.lower().replace({"nan": np.nan, "": np.nan}))
    output["time_bin"] = np.where(output.condition.eq("ii"), "II", "")
    endpoint = pd.to_numeric(output.window_end_to_ictus_s, errors="coerce")
    for low, high, label in (
        (600, 1200, TIME_BINS[0]),
        (300, 600, TIME_BINS[1]),
        (120, 300, TIME_BINS[2]),
        (60, 120, TIME_BINS[3]),
        (0, 60, TIME_BINS[4]),
    ):
        select = output.condition.eq("preict") & endpoint.between(low, high, inclusive="left")
        output.loc[select, "time_bin"] = label
    return output


def merge_autonomic(windows: pd.DataFrame, common: pd.DataFrame) -> pd.DataFrame:
    required = {"participant", "file_path", "window_start_s", "window_end_s", "csi_z", "cpi_z"}
    missing = sorted(required.difference(common.columns))
    if missing:
        raise ValueError(f"Autonomic input is missing columns: {missing}")
    autonomic = common.copy()
    autonomic["file_key"] = normalise_path(autonomic.file_path)
    keys = ["participant", "file_key", "window_start_s", "window_end_s"]
    columns = keys + [column for column in ("csi_z", "cpi_z", "n_native", "mean_hr_bpm")
                      if column in autonomic.columns]
    autonomic = autonomic[columns].drop_duplicates(keys)
    return windows.merge(autonomic, on=keys, how="left", validate="many_to_one")


def attach_action_values(frame: pd.DataFrame, lag_windows: int) -> pd.DataFrame:
    if lag_windows not in (0, 1):
        raise ValueError("action_lag_windows must be 0 (instantaneous) or 1 (preceding contiguous window)")
    output = frame.copy()
    if lag_windows == 0:
        output["action_csi"] = pd.to_numeric(output.csi_z, errors="coerce")
        output["action_cpi"] = pd.to_numeric(output.cpi_z, errors="coerce")
        output["action_temporal_relation"] = "same 60-s window"
        return output
    output = output.sort_values(["participant", "file_key", "window_start_s", "window_end_s"]).copy()
    grouped = output.groupby(["participant", "file_key"], observed=True, sort=False)
    output["action_csi"] = grouped.csi_z.shift(1)
    output["action_cpi"] = grouped.cpi_z.shift(1)
    previous_end = grouped.window_end_s.shift(1)
    contiguous = np.isclose(
        pd.to_numeric(output.window_start_s, errors="coerce"),
        pd.to_numeric(previous_end, errors="coerce"),
        atol=1e-6,
        equal_nan=False,
    )
    output.loc[~contiguous, ["action_csi", "action_cpi"]] = np.nan
    output["action_temporal_relation"] = "preceding contiguous 60-s window"
    return output.sort_values("row_id").reset_index(drop=True)


def hard_action_labels(cpi: np.ndarray, csi: np.ndarray) -> np.ndarray:
    labels = np.full(len(cpi), "missing", dtype=object)
    valid = np.isfinite(cpi) & np.isfinite(csi)
    labels[valid & (cpi >= 0) & (csi >= 0)] = ACTION_SHORT[0]
    labels[valid & (cpi >= 0) & (csi < 0)] = ACTION_SHORT[1]
    labels[valid & (cpi < 0) & (csi >= 0)] = ACTION_SHORT[2]
    labels[valid & (cpi < 0) & (csi < 0)] = ACTION_SHORT[3]
    return labels


def action_weights(cpi: np.ndarray, csi: np.ndarray, mode: str, temperature: float) -> np.ndarray:
    cpi = np.asarray(cpi, float)
    csi = np.asarray(csi, float)
    output = np.full((len(cpi), 4), np.nan, dtype=float)
    valid = np.isfinite(cpi) & np.isfinite(csi)
    if mode == "hard":
        labels = hard_action_labels(cpi, csi)
        for action, label in enumerate(ACTION_SHORT):
            output[valid & (labels == label), action] = 1.0
        output[valid & np.isnan(output).all(axis=1)] = 0.0
        output[valid] = np.nan_to_num(output[valid], nan=0.0)
    elif mode == "soft":
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("action_temperature must be positive")
        p_cpi_plus = expit(np.clip(cpi[valid] / temperature, -30, 30))
        p_csi_plus = expit(np.clip(csi[valid] / temperature, -30, 30))
        output[valid] = np.column_stack((
            p_cpi_plus * p_csi_plus,
            p_cpi_plus * (1 - p_csi_plus),
            (1 - p_cpi_plus) * p_csi_plus,
            (1 - p_cpi_plus) * (1 - p_csi_plus),
        ))
    else:
        raise ValueError("action_mode must be 'soft' or 'hard'")
    if valid.any() and not np.allclose(output[valid].sum(axis=1), 1.0, atol=1e-10):
        raise AssertionError("Action interpolation weights do not sum to one")
    return output


def counts_cubes(frame: pd.DataFrame) -> np.ndarray:
    cubes = frame.loc[:, COUNT_COLUMNS].to_numpy(float).reshape(-1, 6, 6)
    diagonal = np.arange(6)
    cubes[:, diagonal, diagonal] = 0.0
    return cubes


def row_probabilities(
    counts: np.ndarray,
    prior_probabilities: np.ndarray | None = None,
    prior_strength: float = 0.0,
    jeffreys: float = 0.5,
) -> np.ndarray:
    array = np.asarray(counts, float)
    original_ndim = array.ndim
    if original_ndim == 2:
        array = array[None, ...]
    if array.shape[-2:] != (6, 6):
        raise ValueError("Expected [..., 6, 6] transition counts")
    if prior_probabilities is not None:
        prior = np.asarray(prior_probabilities, float)
        if prior.ndim == 2:
            prior = np.broadcast_to(prior[None, ...], array.shape)
        elif prior.shape != array.shape:
            prior = np.broadcast_to(prior, array.shape)
    else:
        prior = None
    output = np.zeros_like(array, dtype=float)
    for action in range(array.shape[0]):
        for source in range(6):
            allowed = np.arange(6) != source
            x = np.clip(array[action, source, allowed], 0, None)
            if prior is None:
                p0 = np.repeat(1 / 5, 5)
            else:
                p0 = np.clip(prior[action, source, allowed], 1e-12, None)
                p0 = p0 / p0.sum()
            posterior = x + prior_strength * p0 + jeffreys
            output[action, source, allowed] = posterior / posterior.sum()
    return output[0] if original_ndim == 2 else output


def weighted_js_from_probabilities(
    observed_counts: np.ndarray,
    reference_probabilities: np.ndarray,
) -> float:
    counts = np.asarray(observed_counts, float)
    reference = np.asarray(reference_probabilities, float)
    if counts.ndim == 2:
        counts = counts[None, ...]
        reference = reference[None, ...]
    x = counts[..., OFFDIAGONAL_MASK].reshape(-1, 5)
    q = reference[..., OFFDIAGONAL_MASK].reshape(-1, 5)
    n = x.sum(axis=1)
    valid = n > 0
    if not np.any(valid):
        return np.nan
    x = x[valid]; q = q[valid]; n = n[valid]
    p = (x + 0.5) / (n[:, None] + 2.5)
    q = np.clip(q, 1e-12, None)
    q /= q.sum(axis=1, keepdims=True)
    midpoint = (p + q) / 2
    values = 0.5 * np.sum(p * np.log(p / midpoint), axis=1) + \
        0.5 * np.sum(q * np.log(q / midpoint), axis=1)
    return float(np.average(values, weights=n))


def fit_exponential_gain(
    observed_counts: np.ndarray,
    offset_log_probability: np.ndarray,
    effect: np.ndarray,
    start: float = 1.0,
    bounds: tuple[float, float] = (-4.0, 4.0),
    maximum_iterations: int = 30,
) -> float:
    """Fit gain in softmax(offset + gain * effect) by vectorised damped Newton updates."""
    counts = np.asarray(observed_counts, float)
    offset = np.asarray(offset_log_probability, float)
    effect = np.asarray(effect, float)
    if counts.ndim == 2:
        counts = counts[None, ...]
        offset = offset[None, ...]
        effect = effect[None, ...]
    x = counts[..., OFFDIAGONAL_MASK].reshape(-1, 5)
    off = offset[..., OFFDIAGONAL_MASK].reshape(-1, 5)
    eff = effect[..., OFFDIAGONAL_MASK].reshape(-1, 5)
    n = x.sum(axis=1)
    valid = n > 0
    if not np.any(valid):
        return np.nan
    x = x[valid]; off = off[valid]; eff = eff[valid]; n = n[valid]

    def score_hessian(gain: float) -> tuple[float, float, float]:
        logits = off + gain * eff
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        log_probabilities = logits - np.log(exp_logits.sum(axis=1, keepdims=True))
        mean_effect = np.sum(probabilities * eff, axis=1)
        variance_effect = np.sum(probabilities * (eff - mean_effect[:, None]) ** 2, axis=1)
        likelihood = float(np.sum(x * log_probabilities))
        gradient = float(np.sum(x * eff) - np.sum(n * mean_effect))
        hessian = float(-np.sum(n * variance_effect))
        return likelihood, gradient, hessian

    gain = float(np.clip(start, *bounds))
    best_gain = gain
    best_likelihood = score_hessian(gain)[0]
    for _ in range(maximum_iterations):
        likelihood, gradient, hessian = score_hessian(gain)
        if likelihood > best_likelihood:
            best_likelihood, best_gain = likelihood, gain
        if not np.isfinite(hessian) or hessian >= -1e-10:
            break
        step = -gradient / hessian
        if abs(step) < 1e-7:
            return float(gain)
        candidate = float(np.clip(gain + step, *bounds))
        candidate_likelihood = score_hessian(candidate)[0]
        for _ in range(10):
            if candidate_likelihood >= likelihood:
                break
            step *= 0.5
            candidate = float(np.clip(gain + step, *bounds))
            candidate_likelihood = score_hessian(candidate)[0]
        if candidate_likelihood < likelihood:
            break
        gain = candidate
    for candidate in (bounds[0], bounds[1], best_gain, gain):
        likelihood = score_hessian(float(candidate))[0]
        if likelihood > best_likelihood:
            best_likelihood, best_gain = likelihood, float(candidate)
    return float(best_gain)


def log_score(counts: np.ndarray, probabilities: np.ndarray) -> float:
    x = np.asarray(counts, float)[OFFDIAGONAL_MASK].reshape(6, 5)
    p = np.asarray(probabilities, float)[OFFDIAGONAL_MASK].reshape(6, 5)
    n = x.sum(axis=1)
    valid = n > 0
    if not np.any(valid):
        return np.nan
    p = np.clip(p[valid], 1e-12, None)
    p /= p.sum(axis=1, keepdims=True)
    return float(np.sum(x[valid] * np.log(p)) / n[valid].sum())


def pairwise_action_js(probabilities: np.ndarray, source_weights: np.ndarray | None = None) -> np.ndarray:
    output = np.zeros((4, 4), dtype=float)
    if source_weights is None:
        source_weights = np.ones(6) / 6
    source_weights = np.asarray(source_weights, float)
    source_weights = source_weights / source_weights.sum()
    for first in range(4):
        for second in range(first + 1, 4):
            values = []
            for source in range(6):
                allowed = np.arange(6) != source
                p = np.clip(probabilities[first, source, allowed], 1e-12, None)
                q = np.clip(probabilities[second, source, allowed], 1e-12, None)
                p /= p.sum(); q /= q.sum(); midpoint = (p + q) / 2
                values.append(0.5 * np.sum(p * np.log(p / midpoint)) +
                              0.5 * np.sum(q * np.log(q / midpoint)))
            output[first, second] = output[second, first] = float(np.average(values, weights=source_weights))
    return output


class ActionBaselineModel:
    """Partially pooled participant II action-conditioned transition tensors."""

    def __init__(
        self,
        frame: pd.DataFrame,
        cubes: np.ndarray,
        weights: np.ndarray,
        pool_prior_strength: float,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.cubes = np.asarray(cubes, float)
        self.weights = np.asarray(weights, float)
        self.pool_prior_strength = float(pool_prior_strength)
        self.participants = self.frame.participant.astype(str).to_numpy()
        self.conditions = self.frame.condition.astype(str).to_numpy()
        self.ii_indices_by_participant: dict[str, np.ndarray] = {}
        self.action_counts: dict[str, np.ndarray] = {}
        self.pool_counts: dict[str, np.ndarray] = {}
        self.global_action_counts = np.zeros((4, 6, 6), dtype=float)
        for participant, indices in self.frame[self.frame.condition.eq("ii")].groupby(
            self.frame.participant.astype(str), observed=True, sort=False).groups.items():
            index = np.asarray(list(indices), dtype=int)
            self.ii_indices_by_participant[str(participant)] = index
            action = np.einsum("wq,wsd->qsd", self.weights[index], self.cubes[index])
            pool = self.cubes[index].sum(axis=0)
            self.action_counts[str(participant)] = action
            self.pool_counts[str(participant)] = pool
            self.global_action_counts += action
        self.global_pool_counts = self.global_action_counts.sum(axis=0)
        self._population_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._baseline_cache: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}

    def _population_probabilities(self, participant: str) -> tuple[np.ndarray, np.ndarray]:
        if participant in self._population_cache:
            return self._population_cache[participant]
        action = np.maximum(self.global_action_counts - self.action_counts.get(participant, 0), 0)
        pool = np.maximum(self.global_pool_counts - self.pool_counts.get(participant, 0), 0)
        pool_probabilities = row_probabilities(pool, prior_strength=0.0)
        action_probabilities = row_probabilities(action, prior_probabilities=pool_probabilities,
                                                 prior_strength=10.0)
        self._population_cache[participant] = (action_probabilities, pool_probabilities)
        return action_probabilities, pool_probabilities

    def baseline(
        self,
        participant: str,
        action_prior_strength: float,
        exclude_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        participant = str(participant)
        if participant not in self.action_counts:
            raise KeyError(f"No II reference for participant {participant}")
        cache_key = (participant, float(action_prior_strength))
        if exclude_index is None and cache_key in self._baseline_cache:
            baseline, pooled = self._baseline_cache[cache_key]
            return baseline.copy(), pooled.copy()
        direct_action = self.action_counts[participant].copy()
        direct_pool = self.pool_counts[participant].copy()
        if exclude_index is not None:
            direct_action -= self.weights[exclude_index, :, None, None] * self.cubes[exclude_index][None, :, :]
            direct_pool -= self.cubes[exclude_index]
            direct_action = np.maximum(direct_action, 0)
            direct_pool = np.maximum(direct_pool, 0)
        population_action, population_pool = self._population_probabilities(participant)
        participant_pool = row_probabilities(
            direct_pool,
            prior_probabilities=population_pool,
            prior_strength=self.pool_prior_strength,
        )
        log_ratio = np.log(np.clip(population_action, 1e-12, 1)) - np.log(
            np.clip(population_pool[None, :, :], 1e-12, 1))
        hierarchy = np.zeros((4, 6, 6), dtype=float)
        for action in range(4):
            for source in range(6):
                allowed = np.arange(6) != source
                logits = np.log(np.clip(participant_pool[source, allowed], 1e-12, 1)) + \
                    log_ratio[action, source, allowed]
                logits -= np.max(logits)
                p = np.exp(logits); p /= p.sum()
                hierarchy[action, source, allowed] = p
        action_probabilities = row_probabilities(
            direct_action,
            prior_probabilities=hierarchy,
            prior_strength=float(action_prior_strength),
        )
        if exclude_index is None:
            self._baseline_cache[cache_key] = (action_probabilities.copy(), participant_pool.copy())
        return action_probabilities, participant_pool


def prepare_analysis_data(
    peak_windows: Path,
    common_windows: Path,
    config: ActionTensorConfig,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    windows = enrich_windows(pd.read_csv(peak_windows, low_memory=False))
    common = pd.read_csv(common_windows, low_memory=False)
    merged = attach_action_values(merge_autonomic(windows, common), config.action_lag_windows)
    cubes = counts_cubes(merged)
    weights = action_weights(
        pd.to_numeric(merged.action_cpi, errors="coerce").to_numpy(float),
        pd.to_numeric(merged.action_csi, errors="coerce").to_numpy(float),
        config.action_mode,
        config.action_temperature,
    )
    valid = (
        merged.condition.isin(("ii", "preict"))
        & np.isfinite(weights).all(axis=1)
        & (cubes.sum(axis=(1, 2)) >= config.minimum_transitions)
    )
    frame = merged.loc[valid].copy().reset_index(drop=True)
    cubes = cubes[valid.to_numpy()]
    weights = weights[valid.to_numpy()]
    ii_participants = set(frame.loc[frame.condition.eq("ii"), "participant"].astype(str))
    has_reference = frame.participant.astype(str).isin(ii_participants).to_numpy()
    frame = frame.loc[has_reference].copy().reset_index(drop=True)
    cubes = cubes[has_reference]
    weights = weights[has_reference]
    frame["array_index"] = np.arange(len(frame), dtype=int)
    frame["hard_action"] = hard_action_labels(
        pd.to_numeric(frame.action_cpi, errors="coerce").to_numpy(float),
        pd.to_numeric(frame.action_csi, errors="coerce").to_numpy(float),
    )
    for action, label in enumerate(ACTION_SHORT):
        frame[f"weight_{label}"] = weights[:, action]
    if not {"ii", "preict"}.issubset(set(frame.condition)):
        raise ValueError("Eligible data must contain II and preictal windows")
    return frame, cubes, weights


def evaluate_window(
    index: int,
    model: ActionBaselineModel,
    action_prior_strength: float,
    common_exclusion: int | None = None,
    return_probabilities: bool = False,
) -> dict[str, float | np.ndarray]:
    row = model.frame.iloc[index]
    exclusion = common_exclusion if common_exclusion is not None else (index if row.condition == "ii" else None)
    baseline, pooled = model.baseline(str(row.participant), action_prior_strength, exclusion)
    weights = model.weights[index]
    interpolated = np.einsum("q,qsd->sd", weights, baseline)
    js = weighted_js_from_probabilities(model.cubes[index], interpolated)
    interpolated_log = np.log(np.clip(interpolated, 1e-12, 1))
    # The power parameter in softmax(kappa * log p) is kappa itself, not log(kappa).
    # Constrain it to be positive, then store both kappa and log(kappa) explicitly.
    kappa = fit_exponential_gain(
        model.cubes[index],
        np.zeros_like(interpolated_log),
        interpolated_log,
        start=1.0,
        bounds=(math.exp(-4.0), math.exp(4.0)),
    )
    log_kappa = math.log(kappa) if np.isfinite(kappa) and kappa > 0 else np.nan
    pooled_log = np.log(np.clip(pooled, 1e-12, 1))
    action_log_ratios = np.log(np.clip(baseline, 1e-12, 1)) - pooled_log[None, :, :]
    action_effect = np.einsum("q,qsd->sd", weights, action_log_ratios)
    action_lambda = fit_exponential_gain(model.cubes[index], pooled_log, action_effect,
                                         start=1.0, bounds=(-3.0, 3.0))
    conditional_score = log_score(model.cubes[index], interpolated)
    pooled_score = log_score(model.cubes[index], pooled)
    result: dict[str, float | np.ndarray] = {
        "action_js": js,
        "log_action_js": math.log(max(js, 1e-12)),
        "action_log_kappa": log_kappa,
        "action_kappa": kappa,
        "action_lambda": action_lambda,
        "action_log_score": conditional_score,
        "pooled_log_score": pooled_score,
        "action_log_score_gain": conditional_score - pooled_score,
    }
    if return_probabilities:
        result["_action_probabilities"] = interpolated
        result["_pooled_probabilities"] = pooled
    return result


def tune_action_prior(
    model: ActionBaselineModel,
    candidates: Sequence[float],
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """Tune partial-pooling strength using II leave-one-window-out log score.

    The returned participant-by-candidate table is retained so baseline validation
    can nest hyperparameter selection across participants.  This prevents the
    reported predictive gain from reusing the same participants both to choose
    the prior strength and to validate it.
    """
    ii = np.flatnonzero(model.frame.condition.eq("ii").to_numpy())
    rows: list[dict] = []
    participant_rows: list[pd.DataFrame] = []
    for strength in candidates:
        values = []
        for index in ii:
            row = model.frame.iloc[index]
            baseline, pooled = model.baseline(str(row.participant), float(strength), exclude_index=index)
            interpolated = np.einsum("q,qsd->sd", model.weights[index], baseline)
            conditional_score = log_score(model.cubes[index], interpolated)
            pooled_score = log_score(model.cubes[index], pooled)
            values.append({
                "participant": str(row.participant),
                "conditional_log_score": conditional_score,
                "pooled_log_score": pooled_score,
                "gain": conditional_score - pooled_score,
            })
        frame = pd.DataFrame(values)
        participant = (frame.groupby("participant", observed=True)[
            ["conditional_log_score", "pooled_log_score", "gain"]]
            .mean().reset_index())
        participant["action_prior_strength"] = float(strength)
        participant["n_ii_windows"] = frame.groupby("participant", observed=True).size().reindex(
            participant.participant).to_numpy(int)
        participant_rows.append(participant)
        rows.append({
            "action_prior_strength": float(strength),
            "participant_equal_conditional_log_score": participant.conditional_log_score.mean(),
            "participant_equal_pooled_log_score": participant.pooled_log_score.mean(),
            "participant_equal_gain": participant.gain.mean(),
            "n_participants": len(participant),
            "n_ii_windows": len(frame),
        })
    tuning = pd.DataFrame(rows).sort_values("action_prior_strength").reset_index(drop=True)
    best = tuning.loc[tuning.participant_equal_conditional_log_score.idxmax(), "action_prior_strength"]
    tuning["selected"] = tuning.action_prior_strength.eq(best)
    participant_scores = pd.concat(participant_rows, ignore_index=True)
    return float(best), tuning, participant_scores


def nested_baseline_validation(
    participant_candidate_scores: pd.DataFrame,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate action conditioning with participant-nested prior selection.

    For each held-out participant, the action prior strength is selected using
    all *other* participants' II leave-one-window-out conditional log scores.
    The held-out participant's gain at that strength is then used once.  The
    resulting participant-level gains are bootstrapped with participants as the
    resampling unit.
    """
    scores = participant_candidate_scores.copy()
    required = {"participant", "action_prior_strength", "conditional_log_score",
                "pooled_log_score", "gain", "n_ii_windows"}
    if missing := required.difference(scores.columns):
        raise ValueError(f"Nested validation scores are missing columns: {sorted(missing)}")
    rows = []
    for participant in scores.participant.astype(str).unique():
        training = scores[scores.participant.astype(str).ne(participant)]
        mean_training = training.groupby("action_prior_strength", observed=True).conditional_log_score.mean()
        if mean_training.empty:
            continue
        selected = float(mean_training.idxmax())
        test = scores[
            scores.participant.astype(str).eq(participant)
            & scores.action_prior_strength.eq(selected)
        ]
        if len(test) != 1:
            continue
        row = test.iloc[0]
        rows.append({
            "participant": participant,
            "selected_action_prior_strength": selected,
            "mean_log_score_gain": float(row.gain),
            "conditional_log_score": float(row.conditional_log_score),
            "pooled_log_score": float(row.pooled_log_score),
            "n_ii_windows": int(row.n_ii_windows),
        })
    participant = pd.DataFrame(rows)
    if participant.empty:
        raise ValueError("Nested action-prior validation produced no held-out participants")
    rng = np.random.default_rng(seed)
    estimate, low, high, probability = bootstrap_mean(
        participant.mean_log_score_gain.to_numpy(float), rng, draws)
    summary = pd.DataFrame([{
        "estimand": "participant-nested II action-conditioned minus action-pooled log score per transition",
        "estimate": estimate,
        "ci95_low": low,
        "ci95_high": high,
        "probability_gt_zero": probability,
        "n_participants": len(participant),
        "n_ii_windows": int(participant.n_ii_windows.sum()),
        "selection": "for each participant, prior strength selected using all other participants",
    }])
    return participant, summary


def compute_window_metrics(
    model: ActionBaselineModel,
    action_prior_strength: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = [
        "row_id", "array_index", "participant", "file_path", "file_key", "event_key",
        "condition", "event_type", "semiology", "time_bin", "window_index",
        "window_start_s", "window_end_s", "window_end_to_ictus_s", "group",
        "seizure_type", "action_csi", "action_cpi", "hard_action", "n_native",
        "action_temporal_relation",
    ]
    metadata = [column for column in metadata if column in model.frame.columns]
    rows = []
    action_probabilities = np.zeros_like(model.cubes, dtype=float)
    pooled_probabilities = np.zeros_like(model.cubes, dtype=float)
    for index in range(len(model.frame)):
        row = model.frame.iloc[index]
        result = evaluate_window(index, model, action_prior_strength, return_probabilities=True)
        action_probabilities[index] = result.pop("_action_probabilities")
        pooled_probabilities[index] = result.pop("_pooled_probabilities")
        entry = {column: row[column] for column in metadata}
        entry.update(result)
        entry["total_transitions"] = int(model.cubes[index].sum())
        for action, label in enumerate(ACTION_SHORT):
            entry[f"weight_{label}"] = model.weights[index, action]
        rows.append(entry)
    output = pd.DataFrame(rows)
    if output.row_id.duplicated().any():
        raise AssertionError("Window metrics contain duplicated row IDs")
    source, destination = np.where(OFFDIAGONAL_MASK)
    repeated = np.repeat(np.arange(len(model.frame)), len(source))
    source_index = np.tile(source, len(model.frame))
    destination_index = np.tile(destination, len(model.frame))
    counts = model.cubes[:, source, destination].reshape(-1)
    source_totals = model.cubes.sum(axis=2)[:, source].reshape(-1)
    action_probability = action_probabilities[:, source, destination].reshape(-1)
    pooled_probability = pooled_probabilities[:, source, destination].reshape(-1)
    expected = source_totals * action_probability
    route_metadata = [column for column in (
        "row_id", "array_index", "participant", "event_key", "condition", "event_type",
        "semiology", "time_bin", "window_index", "window_start_s", "window_end_s",
        "window_end_to_ictus_s", "group", "action_csi", "action_cpi", "hard_action",
        "action_temporal_relation",
    ) if column in model.frame.columns]
    routes = model.frame.iloc[repeated][route_metadata].reset_index(drop=True)
    routes["source"] = np.asarray(STATES)[source_index]
    routes["destination"] = np.asarray(STATES)[destination_index]
    routes["observed_count"] = counts
    routes["source_total"] = source_totals
    routes["source_has_transitions"] = source_totals > 0
    routes["observed_route_probability"] = np.where(
        source_totals > 0, (counts + 0.5) / (source_totals + 2.5), np.nan)
    routes["action_tensor_probability"] = action_probability
    routes["action_pooled_probability"] = pooled_probability
    routes["expected_count_action"] = expected
    routes["route_log_misfit"] = np.where(
        source_totals > 0, np.log((counts + 0.5) / (expected + 0.5)), np.nan)
    routes["route_log_probability_ratio"] = np.log(
        routes.observed_route_probability / np.clip(action_probability, 1e-12, 1))
    routes["per_transition_action_vs_pooled_log_score_gain"] = np.log(
        np.clip(action_probability, 1e-12, 1) / np.clip(pooled_probability, 1e-12, 1))
    routes["action_log_score_contribution"] = counts * np.log(
        np.clip(action_probability, 1e-12, 1))
    routes["action_vs_pooled_log_score_contribution"] = counts * \
        routes.per_transition_action_vs_pooled_log_score_gain
    variance = expected * (1 - action_probability)
    routes["pearson_residual"] = np.where(
        source_totals > 0, (counts - expected) / np.sqrt(np.clip(variance, 1e-12, None)), np.nan)
    for action, label in enumerate(ACTION_SHORT):
        routes[f"weight_{label}"] = model.weights[repeated, action]
    if len(routes) != 30 * len(output) or routes.groupby("row_id", observed=True).size().ne(30).any():
        raise AssertionError("Route export must contain exactly 30 off-diagonal routes per window")
    return output, routes


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, float)
    output = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return output
    order = valid[np.argsort(values[valid])]
    adjusted = values[order] * len(order) / np.arange(1, len(order) + 1)
    output[order] = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    return output


def route_level_analysis(
    routes: pd.DataFrame,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate route deviations without pretending that counted jumps are timestamped rows."""
    preictal = routes[
        routes.condition.eq("preict")
        & routes.event_type.isin(("functional", "epileptic"))
        & routes.time_bin.isin(TIME_BINS)
    ].copy()
    event = (preictal.groupby(
        ["participant", "event_key", "event_type", "semiology", "time_bin", "source", "destination"],
        observed=True, dropna=False)
        .agg(observed_count=("observed_count", "sum"),
             expected_count_action=("expected_count_action", "sum"),
             n_windows=("row_id", "nunique"))
        .reset_index())
    event = event[event.expected_count_action.gt(0)].copy()
    event["route_log_misfit"] = np.log(
        (event.observed_count + 0.5) / (event.expected_count_action + 0.5))
    cells = (event.groupby(
        ["participant", "event_type", "time_bin", "source", "destination"], observed=True)
        .agg(route_log_misfit=("route_log_misfit", "mean"),
             n_events=("event_key", "nunique"),
             observed_count=("observed_count", "sum"),
             expected_count_action=("expected_count_action", "sum"))
        .reset_index())
    cells["group"] = cells.event_type.map({"functional": "FS", "epileptic": "ES"})
    rng = np.random.default_rng(seed)
    summary_rows = []
    contrast_rows = []
    for time_bin in TIME_BINS:
        for source in STATES:
            for destination in STATES:
                if source == destination:
                    continue
                part = cells[
                    cells.time_bin.eq(time_bin) & cells.source.eq(source)
                    & cells.destination.eq(destination)
                ]
                for group in ("FS", "ES"):
                    values = part.loc[part.group.eq(group), "route_log_misfit"].to_numpy(float)
                    estimate, low, high, probability = bootstrap_mean(values, rng, draws)
                    summary_rows.append({
                        "group": group, "time_bin": time_bin, "source": source,
                        "destination": destination, "estimate": estimate, "ci95_low": low,
                        "ci95_high": high, "probability_gt_zero": probability,
                        "n_participants": int(np.isfinite(values).sum()),
                    })
                estimate, low, high, probability, n_participants, n_fs, n_es = \
                    participant_cluster_difference(
                        part, "route_log_misfit", "FS", "ES", draws, rng)
                contrast_rows.append({
                    "contrast": "FS minus ES", "time_bin": time_bin, "source": source,
                    "destination": destination, "estimate": estimate, "ci95_low": low,
                    "ci95_high": high, "probability_gt_zero": probability,
                    "bootstrap_two_sided_p": 2 * min(probability, 1 - probability),
                    "n_participants": n_participants, "n_fs": n_fs, "n_es": n_es,
                    "estimator": "participant-cluster bootstrap preserving dual-diagnosis overlap",
                })
    summary = pd.DataFrame(summary_rows)
    contrasts = pd.DataFrame(contrast_rows)
    contrasts["bh_fdr"] = bh_fdr(contrasts.bootstrap_two_sided_p)
    return event, cells, summary, contrasts


def participant_cells(window_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = ["log_action_js", "action_log_kappa", "action_lambda", "action_log_score_gain"]
    preictal = window_metrics[
        window_metrics.condition.eq("preict")
        & window_metrics.event_type.isin(("functional", "epileptic"))
        & window_metrics.time_bin.isin(TIME_BINS)
    ].copy()
    event = (preictal.groupby(
        ["participant", "event_key", "event_type", "semiology", "time_bin"],
        observed=True, dropna=False)[outcomes].mean().reset_index())
    cells = (event.groupby(["participant", "event_type", "time_bin"], observed=True)[outcomes]
             .mean().reset_index())
    participant_types = preictal[["participant", "event_type"]].drop_duplicates()
    ii = (window_metrics[window_metrics.condition.eq("ii")]
          .groupby("participant", observed=True)[outcomes].mean().reset_index()
          .merge(participant_types, on="participant", how="inner"))
    ii["time_bin"] = "II"
    cells = pd.concat([ii[cells.columns], cells], ignore_index=True)
    cells["group"] = cells.event_type.map({"functional": "FS", "epileptic": "ES"})
    return event, cells


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float, float, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan, np.nan
    samples = np.mean(values[rng.integers(0, len(values), size=(draws, len(values)))], axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(values.mean()), float(low), float(high), float(np.mean(samples > 0))


def participant_cluster_difference(
    frame: pd.DataFrame,
    outcome: str,
    first: str,
    second: str,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, int, int, int]:
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
    weights = rng.multinomial(n_participants, np.repeat(1 / n_participants, n_participants), size=draws)
    first_denominator = weights @ first_mask
    second_denominator = weights @ second_mask
    valid = (first_denominator > 0) & (second_denominator > 0)
    samples = (weights[valid] @ first_filled) / first_denominator[valid] - \
        (weights[valid] @ second_filled) / second_denominator[valid]
    low, high = np.quantile(samples, (0.025, 0.975))
    return (observed, float(low), float(high), float(np.mean(samples > 0)),
            n_participants, int(first_mask.sum()), int(second_mask.sum()))


def trajectory_summaries(
    cells: pd.DataFrame,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = ("log_action_js", "action_log_kappa", "action_lambda", "action_log_score_gain")
    rng = np.random.default_rng(seed)
    summary_rows = []
    contrast_rows = []
    for outcome in outcomes:
        for group in ("FS", "ES"):
            for time_bin in TIME_ORDER:
                values = cells[cells.group.eq(group) & cells.time_bin.eq(time_bin)][outcome].to_numpy(float)
                estimate, low, high, probability = bootstrap_mean(values, rng, draws)
                summary_rows.append({
                    "outcome": outcome,
                    "group": group,
                    "time_bin": time_bin,
                    "estimate": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "probability_gt_zero": probability,
                    "n_participants": int(np.isfinite(values).sum()),
                })
        for time_bin in TIME_ORDER:
            part = cells[cells.time_bin.eq(time_bin)]
            estimate, low, high, probability, n_participants, n_fs, n_es = participant_cluster_difference(
                part, outcome, "FS", "ES", draws, rng)
            contrast_rows.append({
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
            group_cells = cells[cells.group.eq(group)]
            for time_bin in TIME_BINS:
                pivot = group_cells.pivot(index="participant", columns="time_bin", values=outcome)
                pair = pivot[["II", time_bin]].dropna()
                difference = (pair[time_bin] - pair["II"]).to_numpy(float)
                estimate, low, high, probability = bootstrap_mean(difference, rng, draws)
                contrast_rows.append({
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
    summary = pd.DataFrame(summary_rows)
    contrasts = pd.DataFrame(contrast_rows)
    return summary, contrasts


def baseline_validation(
    window_metrics: pd.DataFrame,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    participant = (window_metrics[window_metrics.condition.eq("ii")]
                   .groupby("participant", observed=True)
                   .agg(mean_log_score_gain=("action_log_score_gain", "mean"),
                        n_ii_windows=("row_id", "size"))
                   .reset_index())
    rng = np.random.default_rng(seed)
    estimate, low, high, probability = bootstrap_mean(
        participant.mean_log_score_gain.to_numpy(float), rng, draws)
    summary = pd.DataFrame([{
        "estimand": "II leave-one-window-out action-conditioned minus action-pooled log score per transition",
        "estimate": estimate,
        "ci95_low": low,
        "ci95_high": high,
        "probability_gt_zero": probability,
        "n_participants": len(participant),
        "n_ii_windows": int(participant.n_ii_windows.sum()),
    }])
    return participant, summary


def action_occupancy(window_metrics: pd.DataFrame) -> pd.DataFrame:
    weight_columns = [f"weight_{label}" for label in ACTION_SHORT]
    pre = window_metrics[
        window_metrics.condition.eq("preict")
        & window_metrics.event_type.isin(("functional", "epileptic"))
        & window_metrics.time_bin.isin(TIME_BINS)
    ].copy()
    pre["group"] = pre.event_type.map({"functional": "FS", "epileptic": "ES"})
    event = (pre.groupby(["participant", "event_key", "group", "time_bin"], observed=True)[weight_columns]
             .mean().reset_index())
    participant = (event.groupby(["participant", "group", "time_bin"], observed=True)[weight_columns]
                   .mean().reset_index())
    ii_types = pre[["participant", "group"]].drop_duplicates()
    ii = (window_metrics[window_metrics.condition.eq("ii")]
          .groupby("participant", observed=True)[weight_columns].mean().reset_index()
          .merge(ii_types, on="participant", how="inner"))
    ii["time_bin"] = "II"
    participant = pd.concat([ii[participant.columns], participant], ignore_index=True)
    long = participant.melt(
        id_vars=["participant", "group", "time_bin"], value_vars=weight_columns,
        var_name="action", value_name="weight")
    long["action"] = long.action.str.replace("weight_", "", regex=False)
    return long


def baseline_tensor_exports(
    model: ActionBaselineModel,
    action_prior_strength: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    probability_rows = []
    separation_rows = []
    for participant in sorted(model.action_counts):
        probabilities, pooled = model.baseline(participant, action_prior_strength)
        source_counts = model.pool_counts[participant].sum(axis=1)
        source_weights = source_counts / source_counts.sum() if source_counts.sum() else np.ones(6) / 6
        separation = pairwise_action_js(probabilities, source_weights)
        for first in range(4):
            for second in range(first + 1, 4):
                separation_rows.append({
                    "participant": participant,
                    "action_1": ACTION_SHORT[first],
                    "action_2": ACTION_SHORT[second],
                    "js": separation[first, second],
                })
        for action in range(4):
            for source in range(6):
                for destination in range(6):
                    if source == destination:
                        continue
                    probability_rows.append({
                        "participant": participant,
                        "action": ACTION_SHORT[action],
                        "source": STATES[source],
                        "destination": STATES[destination],
                        "probability": probabilities[action, source, destination],
                        "pooled_probability": pooled[source, destination],
                        "log_ratio_to_pooled": math.log(max(probabilities[action, source, destination], 1e-12)) -
                            math.log(max(pooled[source, destination], 1e-12)),
                    })
    return pd.DataFrame(probability_rows), pd.DataFrame(separation_rows)


def robust_scale(values: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    centre = values.median()
    scale = (values - centre).abs().median() * 1.4826
    fallback = values.std()
    scale = scale.mask(~np.isfinite(scale) | scale.eq(0), fallback)
    scale = scale.mask(~np.isfinite(scale) | scale.eq(0), 1.0)
    return centre, scale


def match_action_pseudo_onsets(
    frame: pd.DataFrame,
    minimum_record_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = frame.copy()
    source["record_fraction"] = (
        pd.to_numeric(source.window_end_s, errors="coerce") /
        source.groupby("file_path", observed=True).window_end_s.transform("max")
    )
    true = source[
        source.condition.eq("preict")
        & pd.to_numeric(source.window_end_to_ictus_s, errors="coerce").eq(0)
        & source.event_type.isin(("functional", "epileptic"))
    ]
    pseudo = source[source.condition.eq("ii") & source.record_fraction.ge(minimum_record_fraction)]
    features = [
        "record_fraction", "n_gfp_peaks", "n_state_changes", *(f"n_{state}" for state in STATES),
        "n_native", "action_csi", "action_cpi",
    ]
    features = [column for column in features if column in source.columns]
    feature_weights = {"n_native": 4.0, "action_csi": 3.0, "action_cpi": 3.0,
                       "record_fraction": 0.5}
    rows = []
    for participant, targets in true.groupby("participant", observed=True, sort=False):
        pool = pseudo[pseudo.participant.eq(participant)]
        if pool.empty:
            continue
        for action in ACTION_SHORT:
            target_group = targets[targets.hard_action.eq(action)]
            pool_group = pool[pool.hard_action.eq(action)]
            if target_group.empty or pool_group.empty:
                continue
            combined = pd.concat([target_group[features], pool_group[features]], ignore_index=True)
            _, scales = robust_scale(combined.apply(pd.to_numeric, errors="coerce"))
            target_values = target_group[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            pool_values = pool_group[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            delta = (target_values[:, None, :] - pool_values[None, :, :]) / scales.to_numpy(float)
            weights = np.array([feature_weights.get(feature, 1.0) for feature in features])
            delta *= weights
            comparable = np.isfinite(delta).sum(axis=2)
            cost = np.sqrt(np.nansum(delta ** 2, axis=2) / np.maximum(comparable, 1))
            target_positions, pool_positions = linear_sum_assignment(cost)
            for target_position, pool_position in zip(target_positions, pool_positions):
                target = target_group.iloc[target_position]
                candidate = pool_group.iloc[pool_position]
                entry = {
                    "participant": participant,
                    "event_key": target.event_key,
                    "event_type": target.event_type,
                    "semiology": target.semiology,
                    "hard_action": action,
                    "true_row_id": int(target.row_id),
                    "pseudo_row_id": int(candidate.row_id),
                    "true_array_index": int(target.array_index),
                    "pseudo_array_index": int(candidate.array_index),
                    "match_distance": float(cost[target_position, pool_position]),
                    "n_match_features": int(comparable[target_position, pool_position]),
                }
                for feature in features:
                    entry[f"true_{feature}"] = target[feature]
                    entry[f"pseudo_{feature}"] = candidate[feature]
                    entry[f"difference_{feature}"] = target[feature] - candidate[feature]
                rows.append(entry)
    matches = pd.DataFrame(rows)
    if matches.empty:
        raise ValueError("No exact-action participant-matched pseudo-onsets were available")
    balance_rows = []
    for feature in features:
        true_values = pd.to_numeric(matches[f"true_{feature}"], errors="coerce")
        pseudo_values = pd.to_numeric(matches[f"pseudo_{feature}"], errors="coerce")
        pooled_sd = math.sqrt((true_values.var() + pseudo_values.var()) / 2)
        balance_rows.append({
            "feature": feature,
            "standardised_mean_difference":
                float((true_values.mean() - pseudo_values.mean()) / pooled_sd) if pooled_sd > 0 else 0.0,
            "true_mean": true_values.mean(),
            "pseudo_mean": pseudo_values.mean(),
            "n_pairs": int((true_values.notna() & pseudo_values.notna()).sum()),
        })
    return matches, pd.DataFrame(balance_rows)


def caliper_action_pseudo_matches(
    matches: pd.DataFrame,
    native_caliper: float,
    action_caliper: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict pseudo pairs to close autonomic and sampling matches.

    This is a prespecified strict sensitivity, not the primary estimand.  It
    addresses residual native-sample imbalance at the cost of fewer pairs.
    """
    required = {"difference_n_native", "difference_action_csi", "difference_action_cpi"}
    if missing := required.difference(matches.columns):
        raise ValueError(f"Caliper matching columns missing: {sorted(missing)}")
    use = matches[
        pd.to_numeric(matches.difference_n_native, errors="coerce").abs().le(native_caliper)
        & pd.to_numeric(matches.difference_action_csi, errors="coerce").abs().le(action_caliper)
        & pd.to_numeric(matches.difference_action_cpi, errors="coerce").abs().le(action_caliper)
    ].copy()
    if use.empty:
        return use, pd.DataFrame()
    feature_names = [
        column.removeprefix("true_")
        for column in use.columns
        if column.startswith("true_") and f"pseudo_{column.removeprefix('true_')}" in use.columns
        and column.removeprefix("true_") not in {"row_id", "array_index"}
    ]
    balance_rows = []
    for feature in feature_names:
        true_values = pd.to_numeric(use[f"true_{feature}"], errors="coerce")
        pseudo_values = pd.to_numeric(use[f"pseudo_{feature}"], errors="coerce")
        pooled_sd = math.sqrt((true_values.var() + pseudo_values.var()) / 2)
        balance_rows.append({
            "feature": feature,
            "standardised_mean_difference":
                float((true_values.mean() - pseudo_values.mean()) / pooled_sd) if pooled_sd > 0 else 0.0,
            "true_mean": true_values.mean(),
            "pseudo_mean": pseudo_values.mean(),
            "n_pairs": int((true_values.notna() & pseudo_values.notna()).sum()),
            "native_caliper": native_caliper,
            "action_caliper": action_caliper,
        })
    return use, pd.DataFrame(balance_rows)


def weighted_least_squares_beta(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    root = np.sqrt(np.asarray(weights, float))
    coefficients, *_ = np.linalg.lstsq(x * root[:, None], y * root, rcond=None)
    return coefficients


def action_pseudo_did(
    matches: pd.DataFrame,
    model: ActionBaselineModel,
    action_prior_strength: float,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = matches.copy()
    outcomes = ("log_action_js", "action_log_kappa", "action_lambda")
    result_rows = []
    for row in paired.itertuples(index=False):
        true = evaluate_window(row.true_array_index, model, action_prior_strength,
                               common_exclusion=row.pseudo_array_index)
        pseudo = evaluate_window(row.pseudo_array_index, model, action_prior_strength,
                                 common_exclusion=row.pseudo_array_index)
        entry = {"true_row_id": row.true_row_id}
        for outcome in outcomes:
            entry[f"true_{outcome}"] = true[outcome]
            entry[f"pseudo_{outcome}"] = pseudo[outcome]
            entry[f"difference_{outcome}"] = true[outcome] - pseudo[outcome]
        result_rows.append(entry)
    results = pd.DataFrame(result_rows)
    paired = paired.merge(results, on="true_row_id", how="left", validate="one_to_one")

    covariate_candidates = [
        "difference_n_native", "difference_n_gfp_peaks", "difference_n_state_changes",
        "difference_action_csi", "difference_action_cpi",
    ]
    covariates = [column for column in covariate_candidates if column in paired.columns]
    participant = (paired.groupby(["participant", "event_type"], observed=True)
                   [[*(f"difference_{outcome}" for outcome in outcomes), *covariates]]
                   .mean().reset_index())
    participant["is_fs"] = participant.event_type.eq("functional").astype(int)
    for covariate in covariates:
        values = pd.to_numeric(participant[covariate], errors="coerce")
        centre = values.mean(); scale = values.std()
        participant[covariate] = (values - centre) / (scale if np.isfinite(scale) and scale > 0 else 1.0)

    rng = np.random.default_rng(seed)
    rows = []
    for outcome in outcomes:
        column = f"difference_{outcome}"
        complete = participant.dropna(subset=[column]).copy()
        ids = complete.participant.unique()
        participant_lookup = {participant_id: index for index, participant_id in enumerate(ids)}
        row_participant = complete.participant.map(participant_lookup).to_numpy(int)
        cluster_weights = rng.multinomial(len(ids), np.repeat(1 / len(ids), len(ids)), size=draws)
        row_weights = cluster_weights[:, row_participant]
        y = complete[column].to_numpy(float)
        is_fs = complete.is_fs.to_numpy(bool)
        is_es = ~is_fs
        fs_denominator = row_weights[:, is_fs].sum(axis=1)
        es_denominator = row_weights[:, is_es].sum(axis=1)
        valid = (fs_denominator > 0) & (es_denominator > 0)
        samples_unadjusted = (row_weights[valid][:, is_fs] @ y[is_fs]) / fs_denominator[valid] - \
            (row_weights[valid][:, is_es] @ y[is_es]) / es_denominator[valid]
        unadjusted = float(y[is_fs].mean() - y[is_es].mean())

        adjusted_frame = complete[[column, "is_fs", *covariates, "participant"]].replace(
            [np.inf, -np.inf], np.nan).dropna()
        adjusted_ids = adjusted_frame.participant.unique()
        adjusted_lookup = {participant_id: index for index, participant_id in enumerate(adjusted_ids)}
        adjusted_row_participant = adjusted_frame.participant.map(adjusted_lookup).to_numpy(int)
        adjusted_cluster_weights = rng.multinomial(
            len(adjusted_ids), np.repeat(1 / len(adjusted_ids), len(adjusted_ids)), size=draws)
        x = np.column_stack((
            np.ones(len(adjusted_frame)),
            adjusted_frame.is_fs.to_numpy(float),
            adjusted_frame[covariates].to_numpy(float),
        ))
        adjusted_y = adjusted_frame[column].to_numpy(float)
        adjusted_observed = float(weighted_least_squares_beta(
            x, adjusted_y, np.ones(len(adjusted_frame)))[1])
        samples_adjusted = np.empty(draws, dtype=float)
        for draw in range(draws):
            weights = adjusted_cluster_weights[draw, adjusted_row_participant]
            if weights.sum() <= 0:
                samples_adjusted[draw] = np.nan
            else:
                samples_adjusted[draw] = weighted_least_squares_beta(x, adjusted_y, weights)[1]
        samples_adjusted = samples_adjusted[np.isfinite(samples_adjusted)]

        for estimator, estimate, samples in (
            ("unadjusted participant-cluster DiD", unadjusted, np.asarray(samples_unadjusted, float)),
            ("covariate-adjusted participant-cluster DiD", adjusted_observed, samples_adjusted),
        ):
            low, high = np.quantile(samples, (0.025, 0.975))
            rows.append({
                "outcome": outcome,
                "contrast": "[FS genuine - pseudo] - [ES genuine - pseudo]",
                "estimator": estimator,
                "estimate": estimate,
                "ci95_low": float(low),
                "ci95_high": float(high),
                "probability_gt_zero": float(np.mean(samples > 0)),
                "n_pairs": int(paired[column].notna().sum()),
                "n_participants": complete.participant.nunique(),
                "covariates": "; ".join(covariates) if "adjusted" in estimator else "none",
                "reference": "same participant II tensor with matched pseudo-window removed for both members",
            })
    return paired, pd.DataFrame(rows)


def plot_main_action_figure(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    separation: pd.DataFrame,
    figures: Path,
) -> None:
    """Primary mechanistic figure: action definition, baseline separability, and trajectories."""
    configure_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.55), constrained_layout=True)

    axis = axes[0, 0]
    ii = frame[frame.condition.eq("ii")].copy()
    sample = ii.sample(min(len(ii), 1800), random_state=17)
    for action, label in enumerate(ACTION_SHORT):
        part = sample[sample.hard_action.eq(label)]
        axis.scatter(
            part.action_cpi.clip(-3, 3),
            part.action_csi.clip(-3, 3),
            s=5,
            alpha=0.12,
            color=ACTION_COLORS[action],
            rasterized=True,
        )
    axis.axhline(0, color=GREY, linewidth=0.8)
    axis.axvline(0, color=GREY, linewidth=0.8)
    axis.set(
        xlim=(-3, 3),
        ylim=(-3, 3),
        xlabel="CPI (II-standardised)",
        ylabel="CSI (II-standardised)",
        title="Autonomic actions",
    )
    counts = ii.hard_action.value_counts()
    positions = ((1.55, 1.8), (1.55, -1.8), (-1.55, 1.8), (-1.55, -1.8))
    for action, ((x_position, y_position), short) in enumerate(zip(positions, ACTION_SHORT)):
        axis.text(
            x_position,
            y_position,
            f"{short}\n$n$={counts.get(short, 0):,}",
            ha="center",
            va="center",
            color=ACTION_COLORS[action],
            fontweight="bold",
        )
    panel(axis, "A")

    axis = axes[0, 1]
    matrix = np.zeros((4, 4), dtype=float)
    mean_separation = separation.groupby(["action_1", "action_2"], observed=True).js.mean()
    for first in range(4):
        for second in range(first + 1, 4):
            value = mean_separation.get((ACTION_SHORT[first], ACTION_SHORT[second]), np.nan)
            matrix[first, second] = matrix[second, first] = value
    image = axis.imshow(matrix, cmap="viridis", aspect="equal")
    axis.set_xticks(range(4), ACTION_SHORT)
    axis.set_yticks(range(4), ACTION_SHORT)
    axis.set_title("II tensor separation")
    maximum = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0
    for row_index in range(4):
        for column_index in range(4):
            if row_index != column_index:
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6.6,
                    color="white" if matrix[row_index, column_index] > maximum * 0.55 else "black",
                )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Pairwise JS (nats)")
    panel(axis, "B")

    x = np.arange(len(TIME_ORDER))
    labels = ("II", "−20", "−10", "−5", "−2", "−1")
    plot_specs = (
        (axes[0, 2], "log_action_js", "Action-conditioned JS (nats)", True,
         "Action-conditioned divergence", "C"),
        (axes[1, 0], "action_log_kappa", "Transition precision κ", True,
         "Transition precision", "D"),
        (axes[1, 1], "action_lambda", "Action-specificity gain λ", False,
         "Action specificity", "E"),
        (axes[1, 2], "action_log_score_gain", "Action-conditioned log-score gain", False,
         "Action predictive value", "F"),
    )
    for axis, outcome, ylabel, exponentiate, title, letter in plot_specs:
        for group, color, marker in (("FS", FS_COLOR, "o"), ("ES", ES_COLOR, "s")):
            part = (summary[summary.outcome.eq(outcome) & summary.group.eq(group)]
                    .set_index("time_bin").reindex(TIME_ORDER))
            estimate = part.estimate.to_numpy(float)
            low = part.ci95_low.to_numpy(float)
            high = part.ci95_high.to_numpy(float)
            if exponentiate:
                estimate, low, high = np.exp(estimate), np.exp(low), np.exp(high)
            axis.plot(
                x,
                estimate,
                color=color,
                marker=marker,
                linewidth=1.5,
                markersize=4.0,
                label=group,
            )
            axis.fill_between(x, low, high, color=color, alpha=0.14, linewidth=0)
        axis.set_xticks(x, labels)
        axis.set_xlabel("Minutes before event onset")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y")
        if outcome == "action_log_kappa":
            axis.axhline(1, color=GREY, linestyle="--", linewidth=0.9)
        elif outcome == "action_lambda":
            axis.axhline(1, color=GREY, linestyle="--", linewidth=0.9)
            axis.axhline(0, color=LIGHT_GREY, linestyle=":", linewidth=0.9)
        elif outcome == "action_log_score_gain":
            axis.axhline(0, color=GREY, linestyle="--", linewidth=0.9)
        panel(axis, letter)
    axes[0, 2].legend(frameon=False, loc="upper left")
    save_figure(fig, figures, "figure_07_action_conditioned_transition_tensor")


def plot_validation_figure(
    validation_participants: pd.DataFrame,
    validation_summary: pd.DataFrame,
    balance: pd.DataFrame,
    did: pd.DataFrame,
    figures: Path,
    seed: int,
) -> None:
    """Validation figure: out-of-sample II gain, pseudo balance, and matched DiD."""
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), constrained_layout=True)
    rng = np.random.default_rng(seed)

    axis = axes[0]
    values = validation_participants.mean_log_score_gain.to_numpy(float)
    jitter = rng.normal(0, 0.035, len(values))
    axis.scatter(jitter, values, s=9, alpha=0.28, color=GREY, rasterized=True)
    row = validation_summary.iloc[0]
    axis.errorbar(
        [0],
        [row.estimate],
        yerr=[[row.estimate - row.ci95_low], [row.ci95_high - row.estimate]],
        fmt="o",
        color=PURPLE,
        markersize=5,
        capsize=3,
        linewidth=1.4,
    )
    axis.axhline(0, color=GREY, linestyle="--", linewidth=0.8)
    axis.set(
        xlim=(-0.23, 0.23),
        xticks=[],
        ylabel="Held-out log-score gain\nper transition",
        title="Interictal action dependence",
    )
    panel(axis, "A")

    axis = axes[1]
    feature_labels = {
        "record_fraction": "Record position",
        "n_gfp_peaks": "GFP peaks",
        "n_state_changes": "State changes",
        "n_native": "Native autonomic samples",
        "action_csi": "CSI",
        "action_cpi": "CPI",
    }
    use = balance[balance.feature.isin(feature_labels)].copy()
    use["label"] = use.feature.map(feature_labels)
    use = use.set_index("feature").reindex(feature_labels).dropna(subset=["standardised_mean_difference"]).reset_index()
    y = np.arange(len(use))[::-1]
    smd = use.standardised_mean_difference.to_numpy(float)
    colors = [FS_COLOR if abs(value) > 0.25 else GREY for value in smd]
    axis.barh(y, smd, color=colors, alpha=0.8, height=0.65)
    axis.axvline(0, color=GREY, linewidth=0.8)
    axis.axvline(0.1, color=LIGHT_GREY, linestyle=":", linewidth=0.8)
    axis.axvline(-0.1, color=LIGHT_GREY, linestyle=":", linewidth=0.8)
    axis.set_yticks(y, use.label)
    axis.set_xlabel("SMD (genuine − pseudo)")
    axis.set_title("Pseudo-onset balance")
    axis.grid(axis="x")
    panel(axis, "B")

    axis = axes[2]
    labels = {"log_action_js": "log JS", "action_log_kappa": "log κ", "action_lambda": "λ"}
    if did.empty or "estimator" not in did.columns:
        axis.text(0.5, 0.5, "Pseudo-onset analysis unavailable", transform=axis.transAxes,
                  ha="center", va="center")
        axis.set_axis_off()
    else:
        adjusted = did[did.estimator.astype(str).str.startswith("covariate-adjusted")].copy()
        unadjusted = did[did.estimator.astype(str).str.startswith("unadjusted")].copy()
        y = np.arange(len(adjusted))[::-1]
        axis.errorbar(
            adjusted.estimate,
            y,
            xerr=np.vstack((adjusted.estimate - adjusted.ci95_low,
                            adjusted.ci95_high - adjusted.estimate)),
            fmt="o",
            color=GREEN,
            capsize=3,
            linewidth=1.3,
            markersize=4.5,
            label="Adjusted",
        )
        unadjusted = unadjusted.set_index("outcome").reindex(adjusted.outcome)
        axis.scatter(unadjusted.estimate, y, marker="x", color=GREY, s=22, label="Unadjusted")
        axis.axvline(0, color=GREY, linestyle="--", linewidth=0.8)
        axis.set_yticks(y, [labels.get(value, value) for value in adjusted.outcome])
        axis.set_xlabel("FS-specific DiD")
        axis.set_title("Genuine-versus-pseudo DiD")
        axis.legend(frameon=False, fontsize=7, loc="best")
        axis.grid(axis="x")
    panel(axis, "C")
    save_figure(fig, figures, "figure_08_action_tensor_validation_and_pseudo")


def plot_caliper_pseudo_sensitivity(
    primary_balance: pd.DataFrame,
    primary_did: pd.DataFrame,
    caliper_balance: pd.DataFrame,
    caliper_did: pd.DataFrame,
    figures: Path,
) -> None:
    """Supplemental comparison of primary and strict-caliper pseudo-onset analyses."""
    if caliper_balance.empty or caliper_did.empty:
        return
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True,
                             gridspec_kw={"width_ratios": (1.2, 1.0)})
    selected_features = [
        "record_fraction", "n_gfp_peaks", "n_state_changes", "n_native",
        "action_csi", "action_cpi",
    ]
    labels = {
        "record_fraction": "Record position", "n_gfp_peaks": "GFP peaks",
        "n_state_changes": "State changes", "n_native": "Native autonomic samples",
        "action_csi": "CSI", "action_cpi": "CPI",
    }
    y = np.arange(len(selected_features))[::-1]
    axis = axes[0]
    for balance, marker, color, label, offset in (
        (primary_balance, "x", GREY, "Primary", 0.10),
        (caliper_balance, "o", GREEN, "Strict caliper", -0.10),
    ):
        use = balance.set_index("feature").reindex(selected_features)
        axis.scatter(use.standardised_mean_difference, y + offset, marker=marker,
                     color=color, s=28, label=label, zorder=3)
    axis.axvline(0, color=GREY, linewidth=.8)
    axis.axvline(-.1, color=LIGHT_GREY, linestyle=":", linewidth=.8)
    axis.axvline(.1, color=LIGHT_GREY, linestyle=":", linewidth=.8)
    axis.axvline(-.2, color=LIGHT_GREY, linestyle="--", linewidth=.8)
    axis.axvline(.2, color=LIGHT_GREY, linestyle="--", linewidth=.8)
    axis.set_yticks(y, [labels[value] for value in selected_features])
    axis.set_xlabel("SMD (genuine − pseudo)")
    axis.set_title("Balance after caliper restriction")
    axis.grid(axis="x")
    axis.legend(frameon=False, loc="best")
    panel(axis, "A")

    axis = axes[1]
    outcomes = ("log_action_js", "action_log_kappa", "action_lambda")
    outcome_labels = ("log JS", "log κ", "λ")
    y = np.arange(len(outcomes))[::-1]
    for table, marker, color, label, offset in (
        (primary_did, "x", GREY, "Primary adjusted", 0.10),
        (caliper_did, "o", GREEN, "Caliper adjusted", -0.10),
    ):
        use = table[table.estimator.astype(str).str.startswith("covariate-adjusted")]
        use = use.set_index("outcome").reindex(outcomes)
        estimate = use.estimate.to_numpy(float)
        low = use.ci95_low.to_numpy(float)
        high = use.ci95_high.to_numpy(float)
        axis.errorbar(estimate, y + offset,
                      xerr=np.vstack((estimate - low, high - estimate)),
                      fmt=marker, color=color, capsize=2.5, linewidth=1.1,
                      markersize=5, label=label)
    axis.axvline(0, color=GREY, linestyle="--", linewidth=.8)
    axis.set_yticks(y, outcome_labels)
    axis.set_xlabel("FS-specific genuine−pseudo DiD")
    axis.set_title("Mechanistic onset sensitivity")
    axis.grid(axis="x")
    axis.legend(frameon=False, loc="best")
    panel(axis, "B")
    save_figure(fig, figures, "figure_S9_caliper_pseudo_sensitivity")


def plot_baseline_tensor_heatmaps(probabilities: pd.DataFrame, figures: Path) -> None:
    configure_style()
    group = (probabilities.groupby(["action", "source", "destination"], observed=True)
             .log_ratio_to_pooled.mean().reset_index())
    matrices = {}
    maximum = 0.0
    for action in ACTION_SHORT:
        matrix = np.full((6, 6), np.nan)
        part = group[group.action.eq(action)]
        for row in part.itertuples(index=False):
            matrix[STATES.index(row.source), STATES.index(row.destination)] = row.log_ratio_to_pooled
        matrices[action] = matrix
        maximum = max(maximum, float(np.nanmax(np.abs(matrix))))
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.35), constrained_layout=True)
    image = None
    for index, (axis, action) in enumerate(zip(axes, ACTION_SHORT)):
        image = axis.imshow(matrices[action], cmap="coolwarm", vmin=-maximum, vmax=maximum)
        axis.set_xticks(range(6), STATES)
        axis.set_yticks(range(6), STATES if index == 0 else [])
        axis.set_xlabel("Destination")
        if index == 0:
            axis.set_ylabel("Source")
        axis.set_title(f"{action}\n{ACTIONS[index]}", fontsize=8.5)
        for spine in axis.spines.values():
            spine.set_visible(False)
    if image is not None:
        fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02,
                     label="Log probability ratio vs pooled II")
    save_figure(fig, figures, "figure_S7_interictal_action_transition_tensors")


def plot_route_level_misfit(
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    figures: Path,
) -> None:
    """Compact final-minute source-by-destination view at the data's finest valid resolution."""
    configure_style()
    final_summary = summary[summary.time_bin.eq("-1 to 0")]
    final_contrasts = contrasts[contrasts.time_bin.eq("-1 to 0")]
    matrices = []
    for group in ("FS", "ES"):
        matrix = np.full((6, 6), np.nan)
        for row in final_summary[final_summary.group.eq(group)].itertuples(index=False):
            matrix[STATES.index(row.source), STATES.index(row.destination)] = row.estimate
        matrices.append(matrix)
    difference = np.full((6, 6), np.nan)
    significant = np.zeros((6, 6), dtype=bool)
    for row in final_contrasts.itertuples(index=False):
        source = STATES.index(row.source); destination = STATES.index(row.destination)
        difference[source, destination] = row.estimate
        significant[source, destination] = bool(np.isfinite(row.bh_fdr) and row.bh_fdr < 0.05)
    matrices.append(difference)
    maximum = max(float(np.nanmax(np.abs(matrix))) for matrix in matrices)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=True)
    titles = ("FS", "ES", "FS − ES")
    image = None
    for index, (axis, title, matrix) in enumerate(zip(axes, titles, matrices)):
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-maximum, vmax=maximum)
        axis.set_xticks(range(6), STATES)
        axis.set_yticks(range(6), STATES if index == 0 else [])
        axis.set_xlabel("Destination")
        if index == 0:
            axis.set_ylabel("Source")
        axis.set_title(title)
        for spine in axis.spines.values():
            spine.set_visible(False)
        panel(axis, chr(ord("A") + index))
    rows, columns = np.where(significant)
    axes[2].scatter(columns, rows, marker="o", s=14, facecolors="none",
                    edgecolors="black", linewidths=0.8)
    if image is not None:
        fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02,
                     label="Route log misfit: observed vs action-conditioned II")
    fig.suptitle("Final-minute destination-labelled route deviations", fontsize=10)
    save_figure(fig, figures, "figure_10_route_level_action_misfit")


def write_results(
    out_dir: Path,
    config: ActionTensorConfig,
    selected_prior: float,
    tuning: pd.DataFrame,
    validation: pd.DataFrame,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    did: pd.DataFrame,
    caliper_did: pd.DataFrame,
    route_contrasts: pd.DataFrame,
    counts: dict,
) -> None:
    final_summary = summary[summary.time_bin.eq("-1 to 0")]

    def value(outcome: str, group: str, transform=lambda x: x) -> tuple[float, float, float]:
        row = final_summary[final_summary.outcome.eq(outcome) & final_summary.group.eq(group)].iloc[0]
        return transform(row.estimate), transform(row.ci95_low), transform(row.ci95_high)

    fs_js = value("log_action_js", "FS", math.exp)
    es_js = value("log_action_js", "ES", math.exp)
    fs_k = value("action_log_kappa", "FS", math.exp)
    es_k = value("action_log_kappa", "ES", math.exp)
    fs_l = value("action_lambda", "FS")
    es_l = value("action_lambda", "ES")
    fs_gain = value("action_log_score_gain", "FS")
    es_gain = value("action_log_score_gain", "ES")
    final_contrasts = contrasts[
        contrasts.time_bin.eq("-1 to 0") & contrasts.contrast.eq("FS minus ES")
    ]
    if not did.empty and "estimator" in did.columns:
        did_adjusted = did[did.estimator.astype(str).str.startswith("covariate-adjusted")]
    else:
        did_adjusted = pd.DataFrame()
    final_routes = (route_contrasts[route_contrasts.time_bin.eq("-1 to 0")]
                    .sort_values(["bh_fdr", "bootstrap_two_sided_p"]).head(10))

    def reference_statement(name: str, interval: tuple[float, float, float]) -> str:
        estimate, low, high = interval
        if high < 1:
            return f"{name} was below 1 ({estimate:.3f}; 95% interval {low:.3f} to {high:.3f})."
        if low > 1:
            return f"{name} was above 1 ({estimate:.3f}; 95% interval {low:.3f} to {high:.3f})."
        return f"{name} was not clearly different from 1 ({estimate:.3f}; 95% interval {low:.3f} to {high:.3f})."

    lines = [
        "# Action-conditioned transition-tensor analysis", "",
        "## Model", "",
        "The four autonomic actions are CPI+/CSI+, CPI+/CSI−, CPI−/CSI+, and CPI−/CSI−. "
        "The primary model uses soft quadrant interpolation, so every window contributes fractional counts "
        "whose four weights sum to one. This smooths the boundary at zero without creating additional "
        "independent observations.", "",
        f"The action-conditioned II tensor was partially pooled. II-only cross-validation selected an "
        f"action prior strength of **{selected_prior:g}** transitions per action-by-source row from "
        f"{list(config.action_prior_grid)}.", "",
        "The transition-precision parameter is fitted as the positive power κ in "
        "softmax(κ log p). The output column `action_log_kappa` is log(κ), and `action_kappa` is κ. "
        "This explicitly prevents the earlier double-exponentiation error.", "",
        "## Baseline validation", "",
        f"Participant-nested leave-one-window-out II action conditioning changed held-out log score by "
        f"**{validation.estimate.iloc[0]:.6f} per transition** "
        f"(95% participant-bootstrap interval {validation.ci95_low.iloc[0]:.6f} to "
        f"{validation.ci95_high.iloc[0]:.6f}; P[mean>0]={validation.probability_gt_zero.iloc[0]:.3f}).", "",
        "The prior strength was selected using other participants for each held-out participant. A positive held-out gain is required before interpreting CPI/CSI action as carrying information "
        "about the subsequent microstate destination.", "",
        "## Final-minute estimates", "",
        f"- FS action-conditioned JS: {fs_js[0]:.5f} ({fs_js[1]:.5f} to {fs_js[2]:.5f}).",
        f"- ES action-conditioned JS: {es_js[0]:.5f} ({es_js[1]:.5f} to {es_js[2]:.5f}).",
        f"- FS κ: {fs_k[0]:.3f} ({fs_k[1]:.3f} to {fs_k[2]:.3f}).",
        f"- ES κ: {es_k[0]:.3f} ({es_k[1]:.3f} to {es_k[2]:.3f}).",
        f"- FS action-specificity λ: {fs_l[0]:.3f} ({fs_l[1]:.3f} to {fs_l[2]:.3f}).",
        f"- ES action-specificity λ: {es_l[0]:.3f} ({es_l[1]:.3f} to {es_l[2]:.3f}).",
        f"- FS action-conditioned predictive gain: {fs_gain[0]:.6f} "
        f"({fs_gain[1]:.6f} to {fs_gain[2]:.6f}) per transition.",
        f"- ES action-conditioned predictive gain: {es_gain[0]:.6f} "
        f"({es_gain[1]:.6f} to {es_gain[2]:.6f}) per transition.", "",
        "## Prespecified κ pattern", "",
        reference_statement("FS κ", fs_k),
        reference_statement("ES κ", es_k),
        "The code therefore reports the requested FS<1 and ES>1 pattern only if the intervals support it; "
        "it does not encode that pattern as an assumption.", "",
        "## Finest available route resolution", "",
        "`action_route_window_estimates.csv.gz` exports one row for every 60-second window, source state, "
        "and off-diagonal destination (30 routes per window). It includes the observed count, smoothed "
        "observed probability, action-conditioned II probability, expected count, route residuals, and the "
        "per-transition action-versus-pooled log-score increment. The latter is a route/window estimand shared "
        "by all counted jumps on that route; transition order and individual timestamps are unavailable and "
        "were not imputed.", "",
        "Route contrasts are exploratory participant-cluster bootstrap estimates with BH correction across "
        "all route-by-time tests.", "",
        markdown_table(final_routes) if not final_routes.empty else "No route contrasts.", "",
        "## Interpretation guardrail", "",
        "The empirical tensor estimates P(next distinct microstate | current microstate, autonomic action proxy). "
        "It is an observation-level analogue of an action-conditioned transition model, not an identified "
        "latent-state B matrix. Same-window associations do not establish causal direction. The supplied "
        "preceding-contiguous-window sensitivity tests temporal ordering without converting an observational "
        "association into a causal effect.", "",
        "## Final-minute FS−ES contrasts", "",
        markdown_table(final_contrasts) if not final_contrasts.empty else "No final-minute contrasts.", "",
        "## Matched genuine/pseudo-onset difference-in-differences", "",
        markdown_table(did_adjusted) if not did_adjusted.empty else "No matched-pseudo estimates.", "",
        "## Strict caliper-matched sensitivity", "",
        markdown_table(caliper_did) if not caliper_did.empty else "No caliper-matched estimate was available.", "",
        "## Counts", "", json.dumps(counts, indent=2), "",
        "## Tuning table", "", markdown_table(tuning), "",
    ]
    (out_dir / "ACTION_TENSOR_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    peak_windows: Path,
    common_windows: Path,
    out_dir: Path,
    config: ActionTensorConfig,
    make_plots: bool = True,
    run_pseudo: bool = True,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = out_dir / "tables"; figures = out_dir / "figures"
    tables.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    print("[action tensor] preparing data", flush=True)
    frame, cubes, weights = prepare_analysis_data(peak_windows, common_windows, config)
    print(f"[action tensor] eligible windows={len(frame):,}", flush=True)
    model = ActionBaselineModel(frame, cubes, weights, config.pool_prior_strength)
    print("[action tensor] tuning II prior", flush=True)
    selected_prior, tuning, candidate_participant_scores = tune_action_prior(
        model, config.action_prior_grid)
    nested_validation_participants, validation = nested_baseline_validation(
        candidate_participant_scores, config.bootstrap_draws, config.seed + 2)
    print(f"[action tensor] selected prior={selected_prior:g}; computing window metrics", flush=True)
    window_metrics, route_windows = compute_window_metrics(model, selected_prior)
    print("[action tensor] aggregating and bootstrapping", flush=True)
    event, cells = participant_cells(window_metrics)
    summary, contrasts = trajectory_summaries(cells, config.bootstrap_draws, config.seed + 1)
    route_event, route_cells, route_summary, route_contrasts = route_level_analysis(
        route_windows, config.bootstrap_draws, config.seed + 41)
    apparent_validation_participants, apparent_validation = baseline_validation(
        window_metrics, config.bootstrap_draws, config.seed + 22)
    occupancy = action_occupancy(window_metrics)
    baseline_probabilities, separation = baseline_tensor_exports(model, selected_prior)

    if run_pseudo:
        print("[action tensor] matching pseudo-onsets", flush=True)
        matches, balance = match_action_pseudo_onsets(frame, config.pseudo_record_fraction)
        print(f"[action tensor] matched pairs={len(matches):,}; fitting DiD", flush=True)
        paired, did = action_pseudo_did(
            matches, model, selected_prior, config.bootstrap_draws, config.seed + 3)
        caliper_matches, caliper_balance = caliper_action_pseudo_matches(
            matches, config.pseudo_native_caliper, config.pseudo_action_caliper)
        if len(caliper_matches) >= 20 and caliper_matches.participant.nunique() >= 15:
            caliper_paired, caliper_did = action_pseudo_did(
                caliper_matches, model, selected_prior, config.bootstrap_draws, config.seed + 33)
        else:
            caliper_paired = caliper_matches.copy(); caliper_did = pd.DataFrame()
    else:
        paired = pd.DataFrame(); balance = pd.DataFrame(); did = pd.DataFrame()
        caliper_paired = pd.DataFrame(); caliper_balance = pd.DataFrame(); caliper_did = pd.DataFrame()

    outputs = {
        "action_window_metrics.csv.gz": window_metrics,
        "action_event_bin_analysis_frame.csv.gz": event,
        "action_participant_period_cells.csv": cells,
        "action_trajectory_summary.csv": summary,
        "action_contrasts.csv": contrasts,
        "action_route_window_estimates.csv.gz": route_windows,
        "action_route_event_bin_cells.csv.gz": route_event,
        "action_route_participant_cells.csv.gz": route_cells,
        "action_route_summary.csv": route_summary,
        "action_route_contrasts.csv": route_contrasts,
        "action_baseline_tuning.csv": tuning,
        "action_prior_candidate_participant_scores.csv": candidate_participant_scores,
        "action_baseline_validation_participants.csv": nested_validation_participants,
        "action_baseline_validation_summary.csv": validation,
        "action_baseline_validation_apparent_participants.csv": apparent_validation_participants,
        "action_baseline_validation_apparent_summary.csv": apparent_validation,
        "action_quadrant_occupancy.csv": occupancy,
        "action_baseline_probabilities.csv.gz": baseline_probabilities,
        "action_baseline_pairwise_separation.csv": separation,
        "action_matched_pseudo_pairs.csv.gz": paired,
        "action_matched_pseudo_balance.csv": balance,
        "action_matched_pseudo_did.csv": did,
        "action_caliper_matched_pseudo_pairs.csv.gz": caliper_paired,
        "action_caliper_matched_pseudo_balance.csv": caliper_balance,
        "action_caliper_matched_pseudo_did.csv": caliper_did,
    }
    for name, output in outputs.items():
        output.to_csv(tables / name, index=False, compression="gzip" if name.endswith(".gz") else None)

    if make_plots:
        print("[action tensor] rendering figures", flush=True)
        plot_main_action_figure(frame, summary, separation, figures)
        if run_pseudo:
            plot_validation_figure(nested_validation_participants, validation, balance, did, figures, config.seed + 4)
            plot_caliper_pseudo_sensitivity(balance, did, caliper_balance, caliper_did, figures)
        plot_baseline_tensor_heatmaps(baseline_probabilities, figures)
        plot_route_level_misfit(route_summary, route_contrasts, figures)

    counts = {
        "eligible_windows": len(frame),
        "ii_windows": int(frame.condition.eq("ii").sum()),
        "preictal_windows": int(frame.condition.eq("preict").sum()),
        "participants_with_ii_reference": len(model.action_counts),
        "participants_in_preictal_cells": event.participant.nunique(),
        "preictal_events": event.event_key.nunique(),
        "route_window_rows": len(route_windows),
        "route_event_bin_rows": len(route_event),
        "route_participant_cell_rows": len(route_cells),
        "matched_action_pseudo_pairs": len(paired),
        "matched_action_pseudo_participants": paired.participant.nunique() if not paired.empty else 0,
        "caliper_matched_action_pseudo_pairs": len(caliper_paired),
        "caliper_matched_action_pseudo_participants": caliper_paired.participant.nunique() if not caliper_paired.empty else 0,
    }
    write_results(out_dir, config, selected_prior, tuning, validation, summary, contrasts,
                  did, caliper_did, route_contrasts, counts)
    audit = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "code": str(Path(__file__).resolve()),
        "code_sha256": sha256(Path(__file__)),
        "inputs": {str(peak_windows): sha256(peak_windows), str(common_windows): sha256(common_windows)},
        "config": {
            "minimum_transitions": config.minimum_transitions,
            "action_temperature": config.action_temperature,
            "action_mode": config.action_mode,
            "action_lag_windows": config.action_lag_windows,
            "pool_prior_strength": config.pool_prior_strength,
            "action_prior_grid": list(config.action_prior_grid),
            "selected_action_prior_strength": selected_prior,
            "bootstrap_draws": config.bootstrap_draws,
            "seed": config.seed,
            "pseudo_record_fraction": config.pseudo_record_fraction,
            "pseudo_native_caliper": config.pseudo_native_caliper,
            "pseudo_action_caliper": config.pseudo_action_caliper,
        },
        "estimands": {
            "action_js": "transition-count-weighted JS from participant II action-conditioned tensor",
            "action_kappa": "positive power kappa in softmax(kappa * log p); action_log_kappa stores log(kappa)",
            "action_lambda": "gain on action-specific log-probability deviations from participant action-pooled II",
            "baseline_validation": "participant-nested II leave-one-window-out action-conditioned minus action-pooled log score per transition",
            "pseudo_reference": "common participant II tensor with matched pseudo-window removed for both pair members",
            "pseudo_matching": "exact participant and action quadrant; late-record candidate threshold; covariate-adjusted participant-cluster DiD",
            "route_resolution": "60-second window x source x destination; 30 off-diagonal routes per window; counted transitions remain multiplicities because transition timestamps are unavailable",
            "route_misfit": "log((observed route count + 0.5) / (action-conditioned expected route count + 0.5))",
        },
        "software": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scipy", "matplotlib")},
        "counts": counts,
    }
    completed = [path for path in out_dir.rglob("*")
                 if path.is_file() and path.name != "ACTION_TENSOR_AUDIT.json"]
    audit["outputs"] = {str(path.relative_to(out_dir)): sha256(path) for path in completed}
    (out_dir / "ACTION_TENSOR_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return {**counts, "selected_action_prior_strength": selected_prior,
            "baseline_log_score_gain": float(validation.estimate.iloc[0])}


def run_sensitivities(
    peak_windows: Path,
    common_windows: Path,
    out_dir: Path,
    base_config: ActionTensorConfig,
) -> pd.DataFrame:
    specifications = (
        ("primary_soft_same_window", "soft", 0.5, 0),
        ("hard_quadrants_same_window", "hard", 0.5, 0),
        ("soft_temperature_0.25", "soft", 0.25, 0),
        ("soft_temperature_1.0", "soft", 1.0, 0),
        ("soft_preceding_window", "soft", 0.5, 1),
    )
    rows = []
    for index, (label, mode, temperature, lag) in enumerate(specifications):
        config = ActionTensorConfig(
            minimum_transitions=base_config.minimum_transitions,
            action_temperature=temperature,
            action_mode=mode,
            action_lag_windows=lag,
            pool_prior_strength=base_config.pool_prior_strength,
            action_prior_grid=base_config.action_prior_grid,
            bootstrap_draws=min(base_config.bootstrap_draws, 1000),
            seed=base_config.seed + 100 * (index + 1),
            pseudo_record_fraction=base_config.pseudo_record_fraction,
            pseudo_native_caliper=base_config.pseudo_native_caliper,
            pseudo_action_caliper=base_config.pseudo_action_caliper,
        )
        destination = out_dir / label
        result = run_analysis(peak_windows, common_windows, destination, config,
                              make_plots=False, run_pseudo=False)
        summary = pd.read_csv(destination / "tables" / "action_trajectory_summary.csv")
        contrast = pd.read_csv(destination / "tables" / "action_contrasts.csv")
        final = summary[summary.time_bin.eq("-1 to 0")]
        for outcome in ("log_action_js", "action_log_kappa", "action_lambda", "action_log_score_gain"):
            for group in ("FS", "ES"):
                row = final[final.outcome.eq(outcome) & final.group.eq(group)].iloc[0]
                rows.append({
                    "specification": label,
                    "action_mode": mode,
                    "temperature": temperature,
                    "lag_windows": lag,
                    "outcome": outcome,
                    "group_or_contrast": group,
                    "estimate": row.estimate,
                    "ci95_low": row.ci95_low,
                    "ci95_high": row.ci95_high,
                    **result,
                })
            row = contrast[
                contrast.outcome.eq(outcome)
                & contrast.contrast.eq("FS minus ES")
                & contrast.time_bin.eq("-1 to 0")
            ].iloc[0]
            rows.append({
                "specification": label,
                "action_mode": mode,
                "temperature": temperature,
                "lag_windows": lag,
                "outcome": outcome,
                "group_or_contrast": "FS minus ES",
                "estimate": row.estimate,
                "ci95_low": row.ci95_low,
                "ci95_high": row.ci95_high,
                **result,
            })
    output = pd.DataFrame(rows)
    output.to_csv(out_dir / "action_tensor_specification_sensitivity.csv", index=False)
    plot_sensitivity_figure(output, out_dir)
    return output


def plot_sensitivity_figure(sensitivity: pd.DataFrame, directory: Path) -> None:
    """Plot final-minute estimates across action definitions and temporal ordering."""
    configure_style()
    specifications = list(dict.fromkeys(sensitivity.specification.astype(str)))
    labels = {
        "primary_soft_same_window": "Soft\nsame",
        "hard_quadrants_same_window": "Hard\nsame",
        "soft_temperature_0.25": "Soft\nT=.25",
        "soft_temperature_1.0": "Soft\nT=1",
        "soft_preceding_window": "Soft\nlagged",
    }
    outcomes = (
        ("log_action_js", "Action-conditioned JS", True, "A"),
        ("action_log_kappa", "Transition precision κ", True, "B"),
        ("action_lambda", "Action-specificity λ", False, "C"),
        ("action_log_score_gain", "Predictive log-score gain", False, "D"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    x = np.arange(len(specifications))
    for axis, (outcome, ylabel, exponentiate, letter) in zip(axes.flat, outcomes):
        for group, color, marker, offset in (("FS", FS_COLOR, "o", -0.08), ("ES", ES_COLOR, "s", 0.08)):
            part = (sensitivity[
                sensitivity.outcome.eq(outcome) & sensitivity.group_or_contrast.eq(group)
            ].set_index("specification").reindex(specifications))
            estimate = part.estimate.to_numpy(float)
            low = part.ci95_low.to_numpy(float)
            high = part.ci95_high.to_numpy(float)
            if exponentiate:
                estimate, low, high = np.exp(estimate), np.exp(low), np.exp(high)
            axis.errorbar(
                x + offset,
                estimate,
                yerr=np.vstack((estimate - low, high - estimate)),
                fmt=marker,
                color=color,
                capsize=2.5,
                linewidth=1.1,
                markersize=4,
                label=group,
            )
        axis.set_xticks(x, [labels.get(value, value) for value in specifications])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y")
        if outcome in ("action_log_kappa", "action_lambda"):
            axis.axhline(1, color=GREY, linestyle="--", linewidth=0.8)
        if outcome in ("action_lambda", "action_log_score_gain"):
            axis.axhline(0, color=LIGHT_GREY, linestyle=":", linewidth=0.8)
        panel(axis, letter)
    axes[0, 0].legend(frameon=False, loc="best")
    save_figure(fig, directory, "figure_S8_action_tensor_specification_sensitivity")


def self_check() -> None:
    cpi = np.array([1.0, 1.0, -1.0, -1.0, 0.0])
    csi = np.array([1.0, -1.0, 1.0, -1.0, 0.0])
    soft = action_weights(cpi, csi, "soft", 0.5)
    hard = action_weights(cpi, csi, "hard", 0.5)
    assert np.allclose(soft.sum(axis=1), 1)
    assert np.allclose(hard.sum(axis=1), 1)
    counts = np.zeros((4, 6, 6))
    for action in range(4):
        for source in range(6):
            counts[action, source, np.arange(6) != source] = np.array([20, 30, 40, 50, 60]) + action
    probabilities = row_probabilities(counts)
    assert np.allclose(probabilities.sum(axis=2), 1)
    assert np.allclose(np.diagonal(probabilities, axis1=1, axis2=2), 0)
    assert weighted_js_from_probabilities(counts, probabilities) < 1e-3
    log_probabilities = np.log(np.clip(probabilities, 1e-12, 1))
    kappa = fit_exponential_gain(
        counts, np.zeros_like(log_probabilities), log_probabilities,
        bounds=(math.exp(-4.0), math.exp(4.0)),
    )
    assert abs(kappa - 1) < 0.05, kappa
    assert abs(math.log(kappa)) < 0.05
    pooled = row_probabilities(counts.sum(axis=0))
    pooled_tensor = np.broadcast_to(pooled[None, :, :], probabilities.shape)
    lambda_value = fit_exponential_gain(
        counts,
        np.log(np.clip(pooled_tensor, 1e-12, 1)),
        log_probabilities - np.log(np.clip(pooled_tensor, 1e-12, 1)),
    )
    assert abs(lambda_value - 1) < 0.15, lambda_value
    print("action-tensor self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peak-windows-csv", type=Path)
    parser.add_argument("--common-windows-csv", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--minimum-transitions", type=int, default=30)
    parser.add_argument("--action-temperature", type=float, default=0.5)
    parser.add_argument("--action-mode", choices=("soft", "hard"), default="soft")
    parser.add_argument("--action-lag-windows", type=int, choices=(0, 1), default=0)
    parser.add_argument("--pool-prior-strength", type=float, default=100.0)
    parser.add_argument("--action-prior-grid", type=float, nargs="+", default=(2, 5, 10, 20, 50))
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=31072026)
    parser.add_argument("--pseudo-record-fraction", type=float, default=0.99)
    parser.add_argument("--pseudo-native-caliper", type=float, default=10.0)
    parser.add_argument("--pseudo-action-caliper", type=float, default=1.0)
    parser.add_argument("--run-sensitivities", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(); return
    required = (args.peak_windows_csv, args.common_windows_csv, args.out_dir)
    if any(value is None for value in required):
        parser.error("--peak-windows-csv, --common-windows-csv and --out-dir are required")
    for path in (args.peak_windows_csv, args.common_windows_csv):
        if not path.exists():
            parser.error(f"Input does not exist: {path}")
    config = ActionTensorConfig(
        minimum_transitions=args.minimum_transitions,
        action_temperature=args.action_temperature,
        action_mode=args.action_mode,
        action_lag_windows=args.action_lag_windows,
        pool_prior_strength=args.pool_prior_strength,
        action_prior_grid=tuple(args.action_prior_grid),
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
        pseudo_record_fraction=args.pseudo_record_fraction,
        pseudo_native_caliper=args.pseudo_native_caliper,
        pseudo_action_caliper=args.pseudo_action_caliper,
    )
    result = run_analysis(args.peak_windows_csv, args.common_windows_csv, args.out_dir, config)
    if args.run_sensitivities:
        run_sensitivities(args.peak_windows_csv, args.common_windows_csv,
                          args.out_dir / "sensitivities", config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
