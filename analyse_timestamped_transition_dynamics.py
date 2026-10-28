"""Exact transition-time autonomic tensor and peri-ictal manifold analysis.

Every row of the action tensor is built from genuine ordered microstate changes.
CSI/CPI levels and their heartbeat-grid gradients are linearly interpolated only
within observed support. The tensor uses CPI/CSI gradient quadrants; levels are
retained descriptively. Confirmatory group tests are locked unless
--run-confirmatory and a preregistration identifier are supplied together.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import platform
import importlib.metadata
from urllib.parse import urlparse

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.io import loadmat
from scipy.special import expit
from scipy.stats import beta, permutation_test, t as student_t, trim_mean, ttest_ind, ttest_rel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


STATES = ("F", "E", "G", "B", "A", "C")

def fit_to_canonical(fit_labels) -> np.ndarray:
    """Map fitted-state index (1..6 in the MATLAB fit) to the canonical STATES order.

    The labels are DERIVED by the pre-specified template rule and recorded in recovery_audit.json
    (``fit_labels``); no analyst-chosen order is baked into this file.
    """
    labels = tuple(str(label).strip().upper() for label in fit_labels)
    if len(labels) != 6 or sorted(labels) != sorted(STATES):
        raise ValueError(f"fit_labels must be a permutation of {sorted(STATES)}, got {list(labels)}")
    return np.array([STATES.index(label) for label in labels])


ACTIONS = ("dCPI+/dCSI+", "dCPI+/dCSI-", "dCPI-/dCSI+", "dCPI-/dCSI-")
STATE_ACTIONS = ("CPI+/CSI+", "CPI+/CSI-", "CPI-/CSI+", "CPI-/CSI-")
TIME_BINS = ("-20 to -10", "-10 to -5", "-5 to -2", "-2 to -1", "-1 to 0")
TIME_MIDPOINT = dict(zip(TIME_BINS, (-15, -7.5, -3.5, -1.5, -.5)))
MAX_AUTONOMIC_GAP_S = 5.0
PRIOR_STRENGTH = 10.0
# Outcomes: JS enters inference as the NULL-CALIBRATED EXCESS (JS - E0); kappa as log kappa.  Log JS is retained
# as a descriptive column only (it depends on window transition count and soft-weight sharpness).
OUTCOMES = ("ordinary_js_excess", "log_ordinary_kappa", "tensor_js_excess", "log_tensor_kappa")
PRIMARY_OUTCOMES = ("tensor_js_excess", "log_tensor_kappa")
SECONDARY_FS_ES_OUTCOMES = ("ordinary_js_excess", "log_ordinary_kappa")
NULL_DRAWS = 400                 # parametric-bootstrap draws per window for E0
MIN_WINDOW_TRANSITIONS = 30      # per 60-s window (ordinary, and action-supported for the tensor)
MIN_REFERENCE_TRANSITIONS = 300  # interictal reference length (after leaving the evaluated window out)
MIN_EARLY_WINDOWS = 5            # eligible early windows per event
TRIM = 0.2                       # Yuen sensitivity trimming proportion
PERMUTATIONS = 20000             # studentised permutation sensitivity
MIN_DD_PAIRS = 3
MIN_CLUSTERS_PER_SEMIOLOGY = 3
ANALYSIS_SETTINGS = {"prior_strength": 10.0, "max_autonomic_gap_s": 5.0, "null_draws": NULL_DRAWS,
                     "min_window_transitions": MIN_WINDOW_TRANSITIONS,
                     "min_reference_transitions": MIN_REFERENCE_TRANSITIONS,
                     "min_early_windows": MIN_EARLY_WINDOWS, "trim": TRIM, "permutations": PERMUTATIONS,
                     "min_dd_pairs": MIN_DD_PAIRS, "min_clusters_per_semiology": MIN_CLUSTERS_PER_SEMIOLOGY}
OFF_DIAGONAL = ~np.eye(6, dtype=bool)
FS_COLOR, ES_COLOR = "#0072B2", "#D55E00"
HYPO_COLOR, HYPER_COLOR = "#6A3D9A", "#009E73"


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.5, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .8, "grid.alpha": .15, "pdf.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def normalise_path(value: pd.Series | str) -> pd.Series | str:
    if isinstance(value, pd.Series):
        return value.astype(str).str.replace("/", "\\", regex=False).str.lower()
    return str(value).replace("/", "\\").lower()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_text_exact(path: Path, value: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(value)


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _hdf_array(group: h5py.Group, name: str) -> np.ndarray:
    return np.asarray(group[name]).ravel()


def load_transition_cache(path: Path) -> dict[str, np.ndarray | float]:
    try:
        value = loadmat(path, squeeze_me=True, struct_as_record=False)["transition"]
        return {
            "sfreq": float(value.sfreq), "n_samples": int(value.n_samples),
            "sample": np.atleast_1d(value.transition_sample).astype(np.int64),
            "from": np.atleast_1d(value.from_state).astype(np.int8),
            "to": np.atleast_1d(value.to_state).astype(np.int8),
        }
    except NotImplementedError:
        with h5py.File(path, "r") as handle:
            value = handle["transition"]
            return {
                "sfreq": float(_hdf_array(value, "sfreq")[0]),
                "n_samples": int(_hdf_array(value, "n_samples")[0]),
                "sample": _hdf_array(value, "transition_sample").astype(np.int64),
                "from": _hdf_array(value, "from_state").astype(np.int8),
                "to": _hdf_array(value, "to_state").astype(np.int8),
            }


def action_weights(cpi: np.ndarray, csi: np.ndarray, temperature: float = .5) -> np.ndarray:
    output = np.full((len(cpi), 4), np.nan)
    valid = np.isfinite(cpi) & np.isfinite(csi)
    p_cpi = expit(np.clip(cpi[valid] / temperature, -30, 30))
    p_csi = expit(np.clip(csi[valid] / temperature, -30, 30))
    output[valid] = np.column_stack((
        p_cpi * p_csi, p_cpi * (1 - p_csi),
        (1 - p_cpi) * p_csi, (1 - p_cpi) * (1 - p_csi),
    ))
    if valid.any() and not np.allclose(output[valid].sum(axis=1), 1, atol=1e-12):
        raise AssertionError("Action weights do not sum to one")
    return output


def gradient_weights(cpi_gradient: np.ndarray, csi_gradient: np.ndarray) -> np.ndarray:
    """Soft increasing/decreasing quadrants, scaled by each record's typical absolute slope."""
    normalised = []
    for values in (cpi_gradient, csi_gradient):
        valid = np.isfinite(values)
        scale = np.nanmedian(np.abs(values[valid])) if valid.any() else np.nan
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = np.nanstd(values[valid]) if valid.any() else np.nan
        normalised.append(values / scale if np.isfinite(scale) and scale > 1e-12
                          else np.where(valid, 0.0, np.nan))
    return action_weights(normalised[0], normalised[1])


def transition_counts(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    counts = np.zeros((6, 6), dtype=float)
    np.add.at(counts, (source, destination), 1)
    return counts


def tensor_counts(source: np.ndarray, destination: np.ndarray, weights: np.ndarray) -> np.ndarray:
    counts = np.zeros((4, 6, 6), dtype=float)
    valid = np.isfinite(weights).all(axis=1)
    for action in range(4):
        np.add.at(counts[action], (source[valid], destination[valid]), weights[valid, action])
    return counts


def row_probabilities(counts: np.ndarray, prior: np.ndarray | None = None,
                      prior_strength: float = 0, jeffreys: float = .5) -> np.ndarray:
    values = np.asarray(counts, float)
    was_matrix = values.ndim == 2
    if was_matrix:
        values = values[None]
    output = np.zeros_like(values)
    for action in range(len(values)):
        for source in range(6):
            allowed = np.arange(6) != source
            p0 = np.repeat(.2, 5) if prior is None else np.asarray(
                prior if prior.ndim == 2 else prior[action])[source, allowed]
            p0 = p0 / p0.sum()
            alpha = np.clip(values[action, source, allowed], 0, None) + prior_strength * p0 + jeffreys
            output[action, source, allowed] = alpha / alpha.sum()
    return output[0] if was_matrix else output


def weighted_js(counts: np.ndarray, reference: np.ndarray) -> float:
    values = np.asarray(counts, float)
    probabilities = np.asarray(reference, float)
    if values.ndim == 2:
        values, probabilities = values[None], probabilities[None]
    x = values[..., OFF_DIAGONAL].reshape(-1, 5)
    q = probabilities[..., OFF_DIAGONAL].reshape(-1, 5)
    n = x.sum(axis=1)
    valid = n > 0
    if not valid.any():
        return np.nan
    x, q, n = x[valid], q[valid], n[valid]
    p = (x + .5) / (n[:, None] + 2.5)
    q = np.clip(q, 1e-12, None); q /= q.sum(axis=1, keepdims=True)
    midpoint = (p + q) / 2
    js = .5 * np.sum(p * np.log(p / midpoint), axis=1) + .5 * np.sum(q * np.log(q / midpoint), axis=1)
    return float(np.average(js, weights=n))


def fit_kappa(counts: np.ndarray, reference: np.ndarray) -> float:
    values = np.asarray(counts, float)
    probabilities = np.asarray(reference, float)
    if values.ndim == 2:
        values, probabilities = values[None], probabilities[None]
    x = values[..., OFF_DIAGONAL].reshape(-1, 5)
    effect = np.log(np.clip(probabilities[..., OFF_DIAGONAL].reshape(-1, 5), 1e-12, 1))
    n = x.sum(axis=1)
    valid = n > 0
    if not valid.any() or np.all(np.ptp(effect[valid], axis=1) < 1e-12):
        return np.nan
    x, effect, n = x[valid], effect[valid], n[valid]

    def objective(kappa: float) -> tuple[float, float, float]:
        logits = kappa * effect
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits); probability /= probability.sum(axis=1, keepdims=True)
        mean = np.sum(probability * effect, axis=1)
        variance = np.sum(probability * (effect - mean[:, None]) ** 2, axis=1)
        likelihood = float(np.sum(x * np.log(np.clip(probability, 1e-12, 1))))
        gradient = float(np.sum(x * effect) - np.sum(n * mean))
        hessian = float(-np.sum(n * variance))
        return likelihood, gradient, hessian

    lower, upper = math.exp(-4), math.exp(4)
    if objective(lower)[1] <= 0:
        return lower
    if objective(upper)[1] >= 0:
        return upper
    return float(brentq(lambda k: objective(k)[1], lower, upper, xtol=1e-8))


def entropy_concentration(counts: np.ndarray) -> float:
    values = np.asarray(counts, float)
    if values.ndim == 2:
        values = values[None]
    rows, weights = [], []
    for action in range(len(values)):
        for source in range(6):
            allowed = np.arange(6) != source
            n = values[action, source, allowed].sum()
            if n <= 0:
                continue
            p = (values[action, source, allowed] + .5) / (n + 2.5)
            rows.append(1 + np.sum(p * np.log(p)) / np.log(5))
            weights.append(n)
    return float(np.average(rows, weights=weights)) if rows else np.nan


def stable_rng(key: str) -> np.random.Generator:
    """Deterministic generator keyed by a string (recording, window, purpose): reruns are bit-identical."""
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))


def weighted_js_batch(counts: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """``weighted_js`` for a stack of count tensors (B, A, 6, 6) against one reference (A, 6, 6); returns (B,)."""
    x = np.asarray(counts, float)[..., OFF_DIAGONAL].reshape(len(counts), -1, 5)
    q = np.asarray(reference, float)[..., OFF_DIAGONAL].reshape(-1, 5)
    n = x.sum(axis=2)
    p = (x + .5) / (n[..., None] + 2.5)
    q = np.clip(q, 1e-12, None); q /= q.sum(axis=1, keepdims=True)
    midpoint = (p + q) / 2
    js = .5 * np.sum(p * np.log(p / midpoint), axis=2) + .5 * np.sum(q * np.log(q / midpoint), axis=2)
    total = n.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (js * n).sum(axis=1) / total, np.nan)


def null_expected_js(source: np.ndarray, weights: np.ndarray, reference: np.ndarray, key: str,
                     draws: int = NULL_DRAWS) -> float:
    """E0: expected weighted JS if this window's destinations were drawn from the reference.

    Plug-in JS is positively biased by an amount that depends on the window's transition count, its source
    mix and the sharpness of the soft action weights - all of which can differ between intervals or groups for
    reasons unrelated to grammar.  Conditional on the window's OWN source sequence and soft weights
    (``weights``: n x A, rows sum to 1; the ordinary matrix is A=1 with weight 1), destinations are drawn from
    sum_a w_ia * reference[a, s_i, :] with a hash-seeded generator and the same soft-count JS is computed for each
    draw.  ``JS - E0`` is the calibrated outcome used for inference.
    """
    source = np.asarray(source, int)
    weights = np.asarray(weights, float)
    n_actions = weights.shape[1]
    reference = np.asarray(reference, float).reshape(n_actions, 6, 6)
    if len(source) == 0:
        return np.nan
    rows = np.einsum("ia,aid->id", weights, reference[:, source, :])
    rows /= rows.sum(axis=1, keepdims=True)
    cdf = np.cumsum(rows, axis=1)
    cdf[:, -1] = 1.0
    uniform = stable_rng(key).random((draws, len(source)))
    destination = (uniform[:, :, None] >= cdf[None, :, :5]).sum(axis=2)
    flat = (source[None, :] * 6 + destination + np.arange(draws)[:, None] * 36).ravel()
    counts = np.empty((draws, n_actions, 6, 6))
    for action in range(n_actions):
        counts[:, action] = np.bincount(flat, weights=np.tile(weights[:, action], draws),
                                        minlength=draws * 36).reshape(draws, 6, 6)
    return float(np.nanmean(weighted_js_batch(counts, reference)))


def interpolate_native(native: pd.DataFrame, times: np.ndarray) -> tuple[np.ndarray, ...]:
    """Interpolate inside contiguous finite runs; never bridge missing rows or long gaps."""
    levels, gradients = [], []
    for column in ("csi_z", "cpi_z"):
        level = np.full(len(times), np.nan); gradient = level.copy()
        if not native.empty:
            subset = native[["event_time_s", column]].apply(pd.to_numeric, errors="coerce")
            subset = subset.dropna(subset=["event_time_s"]).groupby(
                "event_time_s", as_index=False)[column].mean().sort_values("event_time_s")
            x, y = subset.event_time_s.to_numpy(), subset[column].to_numpy()
            good = np.isfinite(x) & np.isfinite(y)
            starts = np.flatnonzero(good & np.r_[True, ~good[:-1] | (np.diff(x) > MAX_AUTONOMIC_GAP_S)])
            for start in starts:
                stop = start + 1
                while stop < len(x) and good[stop] and x[stop] - x[stop-1] <= MAX_AUTONOMIC_GAP_S:
                    stop += 1
                if stop - start < 2:
                    continue
                xx, yy = x[start:stop], y[start:stop]
                use = (times >= xx[0]) & (times <= xx[-1])
                level[use] = np.interp(times[use], xx, yy)
                gradient[use] = np.interp(times[use], xx, np.gradient(yy, xx))
        levels.append(level); gradients.append(gradient)
    return levels[0], levels[1], gradients[0], gradients[1]


def window_starts(condition: str, duration: float) -> np.ndarray:
    if duration < 60:
        return np.array([])
    if condition == "preict":
        final_start = duration - 60
        number = int(np.floor(min(1140, final_start) / 10)) + 1
        return np.sort(final_start - 10 * np.arange(number))
    return np.arange(0, duration - 59.999, 60)


def time_bin(endpoint: float) -> str | None:
    for low, high, label in ((600, 1200, TIME_BINS[0]), (300, 600, TIME_BINS[1]),
                             (120, 300, TIME_BINS[2]), (60, 120, TIME_BINS[3]), (0, 60.0001, TIME_BINS[4])):
        if low <= endpoint < high:
            return label
    return None


def load_native(path: Path, file_keys: set[str]) -> dict[str, pd.DataFrame]:
    columns = ["file_path", "event_time_s", "csi_z", "cpi_z"]
    parts = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=200_000):
        chunk["file_key"] = normalise_path(chunk.file_path)
        parts.append(chunk[chunk.file_key.isin(file_keys)])
    frame = pd.concat(parts, ignore_index=True)
    return {key: value for key, value in frame.groupby("file_key", sort=False)}


def build_windows(index: pd.DataFrame, native: dict[str, pd.DataFrame], action_cache: Path,
                  fit_labels
                  ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    metadata, ordinary, tensor, action_rows = [], [], [], []
    fit_map = fit_to_canonical(fit_labels)
    action_cache.mkdir(parents=True, exist_ok=True)
    for record_number, row in enumerate(index.itertuples(index=False), 1):
        transition = load_transition_cache(Path(row.cache_file))
        samples = transition["sample"]
        if (not np.isfinite(transition["sfreq"]) or transition["sfreq"] <= 0
                or transition["n_samples"] < 1 or len(samples) != len(transition["from"])
                or len(samples) != len(transition["to"]) or np.any(np.diff(samples) <= 0)
                or np.any(samples < 1) or np.any(samples > transition["n_samples"])
                or not np.isin(transition["from"], np.arange(1, 7)).all()
                or not np.isin(transition["to"], np.arange(1, 7)).all()
                or np.any(transition["from"] == transition["to"])):
            raise ValueError("Invalid ordered genuine-transition cache")
        times = (transition["sample"] - 1) / transition["sfreq"]
        source = fit_map[transition["from"] - 1]
        destination = fit_map[transition["to"] - 1]
        csi, cpi, csi_gradient, cpi_gradient = interpolate_native(
            native.get(row.file_key, pd.DataFrame()), times)
        state_action = action_weights(cpi, csi)
        action = gradient_weights(cpi_gradient, csi_gradient)
        action_file = action_cache / f"{int(row.manifest_index):04d}_{row.participant}_{row.condition}.npz"
        np.savez_compressed(action_file, transition_sample=transition["sample"].astype(np.uint32),
                            sfreq=np.array([transition["sfreq"]]), source=source.astype(np.uint8),
                            destination=destination.astype(np.uint8), csi_z=csi.astype(np.float32),
                            cpi_z=cpi.astype(np.float32), csi_gradient_z_per_s=csi_gradient.astype(np.float32),
                            cpi_gradient_z_per_s=cpi_gradient.astype(np.float32),
                            state_weights=state_action.astype(np.float32),
                            gradient_weights=action.astype(np.float32), action_weights=action.astype(np.float32))
        valid_native = np.isfinite(action).all(axis=1)
        valid_state = np.isfinite(state_action).all(axis=1)
        action_rows.append({"manifest_index": row.manifest_index, "participant": row.participant,
                            "condition": row.condition, "file_path": row.file_path,
                            "source_transition_cache": row.cache_file,
                            "transition_action_cache": str(action_file), "n_transitions": len(times),
                            "n_with_interpolated_action": int(valid_native.sum()),
                            "n_with_interpolated_state": int(valid_state.sum()),
                            "interpolated_action_fraction": float(valid_native.mean()) if len(times) else np.nan,
                            "sfreq": transition["sfreq"]})
        duration = transition["n_samples"] / transition["sfreq"]
        for start in window_starts(row.condition, duration):
            stop = start + 60
            left, right = np.searchsorted(times, (start, stop), side="left")
            if right - left < MIN_WINDOW_TRANSITIONS:
                continue
            sl = slice(left, right)
            counts = transition_counts(source[sl], destination[sl])
            weighted = tensor_counts(source[sl], destination[sl], action[sl])
            valid_action = np.isfinite(action[sl]).all(axis=1)
            endpoint = duration - stop if row.condition == "preict" else np.nan
            metadata.append({
                "row_id": len(metadata), "participant": row.participant, "file_path": row.file_path,
                "file_key": row.file_key, "event_key": row.file_path, "condition": row.condition,
                "group": row.group, "event_type": row.event_type, "semiology": row.semiology,
                "window_start_s": start, "window_end_s": stop, "window_end_to_ictus_s": endpoint,
                "time_bin": time_bin(endpoint) if row.condition == "preict" else "II",
                "n_transitions": right - left, "n_action_transitions": int(valid_action.sum()),
                "action_cache": str(action_file), "window_left": int(left), "window_right": int(right),
                "csi_transition_mean": np.nanmean(csi[sl]) if np.isfinite(csi[sl]).any() else np.nan,
                "cpi_transition_mean": np.nanmean(cpi[sl]) if np.isfinite(cpi[sl]).any() else np.nan,
                "csi_gradient_mean_z_per_s": np.nanmean(csi_gradient[sl]) if np.isfinite(csi_gradient[sl]).any() else np.nan,
                "cpi_gradient_mean_z_per_s": np.nanmean(cpi_gradient[sl]) if np.isfinite(cpi_gradient[sl]).any() else np.nan,
                **{f"state_weight_{name.lower().replace('/', '_').replace('+', 'pos').replace('-', 'neg')}":
                   np.nanmean(state_action[sl, number]) if np.isfinite(state_action[sl, number]).any() else np.nan
                   for number, name in enumerate(STATE_ACTIONS)},
            })
            ordinary.append(counts); tensor.append(weighted)
        if record_number % 25 == 0 or record_number == len(index):
            print(f"[{record_number}/{len(index)}] timestamped records windowed", flush=True)
    if not metadata:
        raise ValueError("No eligible windows: check transition coverage and input files")
    return pd.DataFrame(metadata), np.stack(ordinary), np.stack(tensor), pd.DataFrame(action_rows)


def participant_baselines(metadata: pd.DataFrame, ordinary: np.ndarray, tensor: np.ndarray,
                          prior_strength: float) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    baselines, precision_rows = {}, []
    for participant, positions in metadata[metadata.condition.eq("ii")].groupby("participant", observed=True).groups.items():
        positions = np.asarray(list(positions), int)
        counts = ordinary[positions].sum(axis=0)
        weighted = tensor[positions].sum(axis=0)
        pooled_probability = row_probabilities(counts)
        probability = row_probabilities(weighted, np.broadcast_to(pooled_probability, weighted.shape), prior_strength)
        baselines[str(participant)] = {"ordinary": counts, "tensor": weighted,
                                       "ordinary_probability": pooled_probability, "tensor_probability": probability}
        group = str(metadata.loc[positions[0], "group"])
        for action_number, action_name in enumerate(ACTIONS):
            for source in range(6):
                allowed = np.arange(6) != source
                p0 = pooled_probability[source, allowed]
                alpha = weighted[action_number, source, allowed] + prior_strength * p0 + .5
                total = alpha.sum()
                sd = np.sqrt(alpha * (total - alpha) / (total * total * (total + 1)))
                lower = beta.ppf(.025, alpha, total - alpha)
                upper = beta.ppf(.975, alpha, total - alpha)
                for local, destination in enumerate(np.flatnonzero(allowed)):
                    precision_rows.append({
                        "participant": participant, "group": group, "action": action_name,
                        "source": STATES[source], "destination": STATES[destination],
                        "effective_weighted_transitions": weighted[action_number, source, allowed].sum(),
                        "posterior_probability": alpha[local] / total, "posterior_sd": sd[local],
                        "posterior_95_low": lower[local], "posterior_95_high": upper[local],
                        "posterior_95_width": upper[local] - lower[local],
                    })
    return baselines, pd.DataFrame(precision_rows)


def evaluate_windows(metadata: pd.DataFrame, ordinary: np.ndarray, tensor: np.ndarray,
                     baselines: dict[str, dict[str, np.ndarray]], prior_strength: float) -> pd.DataFrame:
    rows = []
    cached_path, cached = None, None
    for index, meta in metadata.iterrows():
        baseline = baselines.get(str(meta.participant))
        if baseline is None:
            continue
        ordinary_reference = baseline["ordinary"].copy()
        tensor_reference = baseline["tensor"].copy()
        if meta.condition == "ii":
            ordinary_reference -= ordinary[index]
            tensor_reference -= tensor[index]
        if ordinary_reference.sum() < MIN_REFERENCE_TRANSITIONS:
            continue
        ordinary_probability = row_probabilities(ordinary_reference)
        tensor_probability = row_probabilities(
            tensor_reference, np.broadcast_to(ordinary_probability, tensor_reference.shape), prior_strength)
        if meta.action_cache != cached_path:  # windows of one recording are contiguous
            with np.load(meta.action_cache) as handle:
                cached = (handle["source"].astype(int), handle["action_weights"].astype(float))
            cached_path = meta.action_cache
        window = slice(int(meta.window_left), int(meta.window_right))
        source, weights = cached[0][window], cached[1][window]
        key = f"{meta.file_key}|{meta.window_start_s:.9f}"
        tensor_ok = meta.n_action_transitions >= MIN_WINDOW_TRANSITIONS
        valid = np.isfinite(weights).all(axis=1)
        result = meta.to_dict()
        result.update({
            "ordinary_js": weighted_js(ordinary[index], ordinary_probability),
            "ordinary_js_null": null_expected_js(source, np.ones((len(source), 1)), ordinary_probability[None],
                                                 key + "|null-ordinary"),
            "ordinary_kappa": fit_kappa(ordinary[index], ordinary_probability),
            "ordinary_entropy_concentration": entropy_concentration(ordinary[index]),
            "tensor_js": weighted_js(tensor[index], tensor_probability) if tensor_ok else np.nan,
            "tensor_js_null": (null_expected_js(source[valid], weights[valid], tensor_probability, key + "|null-tensor")
                               if tensor_ok else np.nan),
            "tensor_kappa": fit_kappa(tensor[index], tensor_probability) if tensor_ok else np.nan,
            "tensor_entropy_concentration": entropy_concentration(tensor[index]) if tensor_ok else np.nan,
        })
        for stem in ("ordinary", "tensor"):
            result[f"{stem}_js_excess"] = result[f"{stem}_js"] - result[f"{stem}_js_null"]
            result[f"{stem}_kappa_at_bound"] = bool(
                np.isclose(result[f"{stem}_kappa"], math.exp(-4), rtol=0, atol=1e-8)
                or np.isclose(result[f"{stem}_kappa"], math.exp(4), rtol=0, atol=1e-8))
            result[f"log_{stem}_js"] = math.log(max(result[f"{stem}_js"], 1e-12)) if np.isfinite(result[f"{stem}_js"]) else np.nan  # descriptive only
            result[f"log_{stem}_kappa"] = math.log(result[f"{stem}_kappa"]) if result[f"{stem}_kappa"] > 0 else np.nan
        rows.append(result)
        if len(rows) % 5000 == 0:
            print(f"[{len(rows)}] windows evaluated", flush=True)
    return pd.DataFrame(rows)


def baseline_tables(baselines: dict[str, dict[str, np.ndarray]], metadata: pd.DataFrame) -> pd.DataFrame:
    group_by_participant = metadata.drop_duplicates("participant").set_index("participant").group.to_dict()
    rows = []
    for participant, baseline in baselines.items():
        probability = baseline["tensor_probability"]
        for action in range(4):
            for source in range(6):
                for destination in range(6):
                    if source == destination:
                        continue
                    rows.append({"participant": participant, "group": group_by_participant.get(participant),
                                 "action": ACTIONS[action], "source": STATES[source],
                                 "destination": STATES[destination],
                                 "baseline_probability": probability[action, source, destination]})
    return pd.DataFrame(rows)


def manifold(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    use = metrics[metrics.condition.eq("preict") & metrics.time_bin.isin(TIME_BINS)].copy()
    keys = ["participant", "event_key", "event_type", "semiology", "time_bin"]
    features = {
        "ordinary": ["log_ordinary_js", "log_ordinary_kappa", "csi_transition_mean", "cpi_transition_mean"],
        "tensor": ["log_tensor_js", "log_tensor_kappa", "csi_transition_mean", "cpi_transition_mean"],
    }
    score_parts, loading_parts = [], []
    for representation, columns in features.items():
        cells = use.groupby(keys, observed=True, dropna=False, as_index=False)[columns].mean().dropna(subset=columns)
        if len(cells) < 8:
            continue
        standard = StandardScaler().fit_transform(cells[columns])
        model = PCA(n_components=2, random_state=31072026).fit(standard)
        scores = model.transform(standard)
        cells["pc1"], cells["pc2"] = scores[:, 0], scores[:, 1]
        cells["representation"] = representation
        cells["time_midpoint_min"] = cells.time_bin.map(TIME_MIDPOINT)
        score_parts.append(cells)
        for component in range(2):
            for feature, loading in zip(columns, model.components_[component]):
                loading_parts.append({"representation": representation, "component": f"PC{component + 1}",
                                      "feature": feature, "loading": loading,
                                      "explained_variance_ratio": model.explained_variance_ratio_[component]})
    return (pd.concat(score_parts, ignore_index=True) if score_parts else pd.DataFrame(),
            pd.DataFrame(loading_parts))


def _trajectory_panel(axis: plt.Axes, scores: pd.DataFrame, column: str, levels: tuple[str, str],
                      colors: tuple[str, str], title: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for level, color in zip(levels, colors):
        subset = scores[scores[column].astype(str).str.lower().eq(level.lower())]
        for _, event in subset.groupby("event_key", observed=True):
            event = event.sort_values("time_midpoint_min")
            axis.plot(event.pc1, event.pc2, color=color, alpha=.10, lw=.6)
        jitter = rng.normal(0, .025, (len(subset), 2))
        axis.scatter(subset.pc1 + jitter[:, 0], subset.pc2 + jitter[:, 1], s=9, color=color, alpha=.14, linewidth=0)
        centroids = subset.groupby("time_bin", observed=True)[["pc1", "pc2"]].mean().reindex(TIME_BINS).dropna()
        axis.plot(centroids.pc1, centroids.pc2, color=color, lw=2, marker="o", ms=4, label=level.upper())
        for first, second in zip(centroids.to_numpy()[:-1], centroids.to_numpy()[1:]):
            axis.annotate("", xy=second, xytext=first,
                          arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.6, "mutation_scale": 9})
        for label, point in centroids.iterrows():
            axis.text(point.pc1, point.pc2, label.split()[0], color=color, fontsize=6.5, ha="left", va="bottom")
    axis.axhline(0, color="#D0D5DD", lw=.6); axis.axvline(0, color="#D0D5DD", lw=.6)
    axis.set(title=title, xlabel="Manifold axis 1", ylabel="Manifold axis 2")
    axis.legend(frameon=False, loc="best")


def plot_manifold(scores: pd.DataFrame, figures: Path) -> None:
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), constrained_layout=True)
    for row, representation in enumerate(("ordinary", "tensor")):
        use = scores[scores.representation.eq(representation)]
        label = "Ordinary matrix" if representation == "ordinary" else "Gradient tensor"
        _trajectory_panel(axes[row, 0], use, "event_type", ("FS", "ES"), (FS_COLOR, ES_COLOR),
                          f"{label}: FS and ES", 11 + row)
        _trajectory_panel(axes[row, 1], use[use.event_type.eq("FS")], "semiology", ("hypo", "hyper"),
                          (HYPO_COLOR, HYPER_COLOR), f"{label}: FS hypo and hyper", 21 + row)
    fig.suptitle("Peri-ictal movement in transition–autonomic manifold space", fontsize=13, fontweight="bold")
    save_figure(fig, figures, "figure_01_peri_ictal_manifold")


def plot_matrix_trajectories(metrics: pd.DataFrame, figures: Path) -> None:
    configure_style()
    values = ["ordinary_js", "ordinary_kappa", "tensor_js", "tensor_kappa"]
    titles = ["Original matrix JS", "Original matrix κ", "Autonomic-gradient tensor JS", "Autonomic-gradient tensor κ"]
    event = metrics[metrics.condition.eq("preict") & metrics.time_bin.isin(TIME_BINS)].groupby(
        ["event_key", "event_type", "semiology", "time_bin"], observed=True, dropna=False,
        as_index=False)[values].mean()
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 6.2), sharex=True, constrained_layout=True)
    designs = (("event_type", event, ("FS", "ES"), (FS_COLOR, ES_COLOR), "FS versus ES"),
               ("semiology", event[event.event_type.eq("FS")], ("hypo", "hyper"),
                (HYPO_COLOR, HYPER_COLOR), "FS hypo versus hyper"))
    x = np.arange(len(TIME_BINS))
    for row, (column, frame, levels, colors, label) in enumerate(designs):
        for col, (value, title) in enumerate(zip(values, titles)):
            axis = axes[row, col]
            for level, color in zip(levels, colors):
                subset = frame[frame[column].astype(str).str.lower().eq(level.lower())]
                for _, one_event in subset.groupby("event_key", observed=True):
                    one_event = one_event.set_index("time_bin").reindex(TIME_BINS)
                    axis.plot(x, one_event[value], color=color, alpha=.08, lw=.5)
                cells = subset.groupby("time_bin", observed=True)[value].agg(["mean", "sem"]).reindex(TIME_BINS)
                axis.plot(x, cells["mean"], color=color, marker="o", ms=3.5, lw=1.8, label=level.upper())
                axis.fill_between(x, cells["mean"] - cells["sem"], cells["mean"] + cells["sem"], color=color, alpha=.14)
            axis.set_title(title if row == 0 else "")
            axis.grid(axis="y")
            if col == 0: axis.set_ylabel(label)
            axis.set_xticks(x, ["-20\n-10", "-10\n-5", "-5\n-2", "-2\n-1", "-1\n0"])
            if row == 1: axis.set_xlabel("Minutes before onset")
            if col == 3: axis.legend(frameon=False)
    fig.suptitle("Exact transition-matrix dynamics (descriptive; inference withheld)", fontsize=13, fontweight="bold")
    save_figure(fig, figures, "figure_02_matrix_js_kappa_trajectories")


def plot_autonomic_state_trajectories(metrics: pd.DataFrame, figures: Path) -> None:
    configure_style()
    values, titles = ("cpi_transition_mean", "csi_transition_mean"), ("CPI level", "CSI level")
    event = metrics[metrics.condition.eq("preict") & metrics.time_bin.isin(TIME_BINS)].groupby(
        ["event_key", "event_type", "semiology", "time_bin"], observed=True, dropna=False,
        as_index=False)[list(values)].mean()
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 6.0), sharex=True, constrained_layout=True)
    designs = (("event_type", event, ("FS", "ES"), (FS_COLOR, ES_COLOR), "FS versus ES"),
               ("semiology", event[event.event_type.eq("FS")], ("hypo", "hyper"),
                (HYPO_COLOR, HYPER_COLOR), "FS hypo versus hyper"))
    x = np.arange(len(TIME_BINS))
    for row, (column, frame, levels, colors, label) in enumerate(designs):
        for col, (value, title) in enumerate(zip(values, titles)):
            axis = axes[row, col]
            for level, color in zip(levels, colors):
                subset = frame[frame[column].astype(str).str.lower().eq(level.lower())]
                cells = subset.groupby("time_bin", observed=True)[value].agg(["mean", "sem"]).reindex(TIME_BINS)
                axis.plot(x, cells["mean"], color=color, marker="o", ms=3.5, lw=1.8, label=level.upper())
                axis.fill_between(x, cells["mean"] - cells["sem"], cells["mean"] + cells["sem"],
                                  color=color, alpha=.14)
            axis.axhline(0, color="#98A2B3", lw=.7)
            axis.set_title(title if row == 0 else "")
            axis.grid(axis="y")
            if col == 0: axis.set_ylabel(label)
            axis.set_xticks(x, ["-20\n-10", "-10\n-5", "-5\n-2", "-2\n-1", "-1\n0"])
            if row == 1: axis.set_xlabel("Minutes before onset")
            if col == 1: axis.legend(frameon=False)
    fig.suptitle("Autonomic state trajectories (descriptive; inference withheld)", fontsize=12.5,
                 fontweight="bold")
    save_figure(fig, figures, "figure_04_autonomic_state_trajectories")


def plot_baseline_precision(precision: pd.DataFrame, figures: Path) -> None:
    configure_style()
    cells = precision[precision.group.isin(["FS", "ES"])].groupby(
        ["participant", "group", "action"], observed=True, as_index=False).posterior_95_width.median()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True, constrained_layout=True)
    rng = np.random.default_rng(44)
    for axis, group, color in zip(axes, ("FS", "ES"), (FS_COLOR, ES_COLOR)):
        subset = cells[cells.group.eq(group)]
        for action_number, action in enumerate(ACTIONS):
            y = subset[subset.action.eq(action)].posterior_95_width.to_numpy()
            parts = axis.violinplot(y, [action_number], widths=.7, showextrema=False) if len(y) > 1 else None
            if parts:
                for body in parts["bodies"]: body.set_facecolor(color); body.set_alpha(.22)
            axis.scatter(action_number + rng.normal(0, .055, len(y)), y, s=16, color=color, alpha=.55,
                         edgecolor="white", linewidth=.3)
            if len(y): axis.scatter(action_number, np.median(y), marker="D", s=28, color="#101828", zorder=4)
        axis.set_title(f"{group} interictal gradient-tensor precision")
        axis.set_xticks(range(4), ACTIONS, rotation=25, ha="right")
        axis.grid(axis="y")
    axes[0].set_ylabel("Median destination-probability 95% width")
    fig.suptitle("Absolute baseline gradient-tensor precision by participant", fontsize=12.5, fontweight="bold")
    save_figure(fig, figures, "figure_03_baseline_B_tensor_precision")


CONTRASTS = (("FS versus ES", "Welch participant-level t test"),
             ("FS hypo versus hyper",
              "participant-clustered OLS on participant-by-semiology means (CR2, Bell-McCaffrey df)"),
             ("within-DD FS versus ES", "paired participant-level t test"))


def plan_family(outcome: str, estimand: str, contrast: str) -> str:
    """primary (Holm 2) / secondary (Holm 6) / descriptive (estimate and CI only, no hypothesis test)."""
    if estimand == "final_minus_early":
        if contrast == "FS versus ES" and outcome in PRIMARY_OUTCOMES:
            return "primary"
        if contrast == "FS versus ES" and outcome in SECONDARY_FS_ES_OUTCOMES:
            return "secondary"
        if contrast != "FS versus ES" and outcome in PRIMARY_OUTCOMES:
            return "secondary"
    return "descriptive"


MULTIPLICITY = {"primary": "Holm across the two primary tests",
                "secondary": "Holm across the six secondary tests; supporting evidence, cannot rescue a negative primary",
                "descriptive": "none: estimate with pointwise 95% CI, no hypothesis test"}


def confirmatory_plan() -> pd.DataFrame:
    rows = []
    for outcome in OUTCOMES:
        for estimand in ("final_minute", "final_minus_early"):
            for contrast, test in CONTRASTS:
                family = plan_family(outcome, estimand, contrast)
                rows.append({"outcome": outcome, "estimand": estimand, "contrast": contrast, "test": test,
                             "tensor_definition": "soft CPI/CSI gradient quadrants at genuine transition times",
                             "alternative": "two-sided", "family": family, "multiplicity": MULTIPLICITY[family],
                             "status": "WITHHELD_PENDING_PREREGISTRATION"})
    return pd.DataFrame(rows)


def cluster_cr2(y: np.ndarray, design: np.ndarray, groups: np.ndarray, contrast: np.ndarray) -> dict[str, float]:
    """OLS contrast with CR2 cluster-robust variance and Bell-McCaffrey (Satterthwaite) degrees of freedom.

    CR1 with t(G-1) is liberal with few or unbalanced clusters (type I 0.07-0.09 at 8/4 and 9/3 clusters per
    stratum in simulation); CR2 with Satterthwaite df is close to nominal.  The working model is independent
    errors with equal variance.
    """
    y = np.asarray(y, float)
    x = np.asarray(design, float)
    c = np.asarray(contrast, float)
    n = len(y)
    bread = np.linalg.inv(x.T @ x)
    hat = x @ bread @ x.T
    adjust = np.eye(n) - hat
    ids = pd.factorize(groups)[0]
    b_vector = bread @ c
    q = np.zeros((n, n))
    for g in range(ids.max() + 1):
        idx = np.flatnonzero(ids == g)
        values, vectors = np.linalg.eigh(adjust[np.ix_(idx, idx)])
        scale = np.where(values > 1e-10, 1 / np.sqrt(np.clip(values, 1e-10, None)), 0.0)
        inverse_root = vectors @ np.diag(scale) @ vectors.T
        b_g = inverse_root @ x[idx] @ b_vector
        row = b_g @ adjust[idx, :]
        q += np.outer(row, row)
    estimate = float(c @ bread @ x.T @ y)
    variance = float(y @ q @ y)
    trace, trace_square = float(np.trace(q)), float(np.sum(q * q))
    df = trace * trace / trace_square if trace_square > 0 else np.nan
    se = math.sqrt(max(variance, 0.0))
    t_value = estimate / se if se > 0 else np.nan
    p_value = float(2 * student_t.sf(abs(t_value), df)) if np.isfinite(t_value) and np.isfinite(df) else np.nan
    margin = float(student_t.ppf(.975, df) * se) if np.isfinite(df) else np.nan
    return {"estimate": estimate, "se": se, "t": t_value, "p": p_value, "df": df,
            "ci95_low": estimate - margin, "ci95_high": estimate + margin}


def clustered_hypo_hyper(frame: pd.DataFrame) -> dict[str, float | int]:
    """Hyper-minus-hypo from participant-by-semiology means, keeping participants in either or both strata."""
    values = frame[frame.semiology.astype(str).str.lower().isin(("hypo", "hyper"))].copy()
    values = values.dropna(subset=["participant", "value"])
    values["is_hyper"] = values.semiology.astype(str).str.lower().eq("hyper").astype(float)
    counts = values.groupby("semiology", observed=True).size()
    by_stratum = values.groupby("is_hyper", observed=True).participant.nunique()
    overlap = values.groupby("participant", observed=True).is_hyper.nunique().ge(2).sum()
    result = {"estimate": np.nan, "t": np.nan, "p": np.nan, "n_first": int(counts.get("hypo", 0)),
              "n_second": int(counts.get("hyper", 0)), "n_clusters": int(values.participant.nunique()),
              "n_overlap": int(overlap), "df": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    if by_stratum.get(0.0, 0) < MIN_CLUSTERS_PER_SEMIOLOGY or by_stratum.get(1.0, 0) < MIN_CLUSTERS_PER_SEMIOLOGY:
        return result
    design = np.column_stack([np.ones(len(values)), values.is_hyper.to_numpy()])
    fitted = cluster_cr2(values.value.to_numpy(), design, values.participant.to_numpy(), np.array([0.0, 1.0]))
    result.update({key: fitted[key] for key in ("estimate", "t", "p", "df", "ci95_low", "ci95_high")})
    return result


def planned_interval_cells(metrics: pd.DataFrame, final_endpoint: float = 0.0) -> pd.DataFrame:
    """Whole windows within early [-1200,-600]; exactly [-60,0] for final (``final_endpoint`` 60 = [-120,-60)).

    Returns per event/interval the mean of each outcome over its eligible windows and the number of
    eligible windows (``n_<outcome>``).
    """
    keys = ["participant", "event_key", "group", "event_type", "semiology"]
    pre = metrics.loc[metrics.condition.eq("preict")].copy()
    endpoint = pre.window_end_to_ictus_s
    early = endpoint.ge(600) & endpoint.le(1140)
    final = np.isclose(endpoint, final_endpoint, rtol=0, atol=1e-8)
    pre = pre.loc[early | final].copy()
    pre["time_bin"] = np.where(np.isclose(pre.window_end_to_ictus_s, final_endpoint, rtol=0, atol=1e-8),
                               TIME_BINS[-1], TIME_BINS[0])
    grouped = pre.groupby([*keys, "time_bin"], observed=True, dropna=False)
    means = grouped[list(OUTCOMES)].mean().reset_index()
    counts = grouped[list(OUTCOMES)].count().add_prefix("n_").reset_index(drop=True)
    return pd.concat([means, counts], axis=1)


def event_changes(cells: pd.DataFrame, outcome: str, min_early_windows: int = MIN_EARLY_WINDOWS) -> pd.DataFrame:
    """One row per event with finite final and early values and >= ``min_early_windows`` early windows."""
    keys = ["participant", "event_key", "group", "event_type", "semiology"]
    early = cells[cells.time_bin.eq(TIME_BINS[0])][keys + [outcome, f"n_{outcome}"]].rename(
        columns={outcome: "early", f"n_{outcome}": "n_early"})
    final = cells[cells.time_bin.eq(TIME_BINS[-1])][keys + [outcome]].rename(columns={outcome: "final_minute"})
    event = final.merge(early, on=keys, how="inner", validate="one_to_one")
    keep = np.isfinite(event[["early", "final_minute"]]).all(axis=1) & event.n_early.ge(min_early_windows)
    event = event.loc[keep].copy()
    event["final_minus_early"] = event.final_minute - event.early
    return event


def independent_groups(event: pd.DataFrame, estimand: str) -> tuple[pd.Series, pd.Series]:
    table = event[event.group.isin(["FS", "ES"])].groupby(
        ["participant", "group"], observed=True, as_index=False)[estimand].mean().rename(columns={estimand: "value"})
    if table.participant.duplicated().any():
        raise ValueError("Independent FS/ES comparison contains overlapping participants")
    return (table[table.group.eq("FS")].value.dropna().reset_index(drop=True),
            table[table.group.eq("ES")].value.dropna().reset_index(drop=True))


def welch_summary(fs: pd.Series, es: pd.Series) -> dict[str, float | int]:
    estimate = fs.mean() - es.mean()
    estimable = len(fs) > 1 and len(es) > 1
    test = ttest_ind(fs, es, equal_var=False) if estimable else None
    fs_component = fs.var(ddof=1) / len(fs) if len(fs) > 1 else np.nan
    es_component = es.var(ddof=1) / len(es) if len(es) > 1 else np.nan
    standard_error = np.sqrt(fs_component + es_component)
    df = ((fs_component + es_component) ** 2 /
          (fs_component ** 2 / (len(fs) - 1) + es_component ** 2 / (len(es) - 1))) if test else np.nan
    margin = student_t.ppf(.975, df) * standard_error if test else np.nan
    return {"estimate": estimate, "t": test.statistic if test else np.nan, "p": test.pvalue if test else np.nan,
            "n_first": len(fs), "n_second": len(es), "n_clusters": len(fs) + len(es), "n_overlap": 0, "df": df,
            "ci95_low": estimate - margin, "ci95_high": estimate + margin}


def dd_pairs(event: pd.DataFrame, estimand: str) -> pd.DataFrame:
    return (event[event.group.eq("DD")].groupby(["participant", "event_type"], observed=True, as_index=False)[estimand]
            .mean().pivot(index="participant", columns="event_type", values=estimand)
            .reindex(columns=["FS", "ES"]).dropna())


def paired_summary(dd: pd.DataFrame) -> dict[str, float | int]:
    difference = (dd.FS - dd.ES) if len(dd) else pd.Series(dtype=float)
    estimate = difference.mean() if len(dd) else np.nan
    estimable = len(dd) >= MIN_DD_PAIRS
    test = ttest_rel(dd.FS, dd.ES) if estimable else None
    df = len(dd) - 1 if estimable else np.nan
    margin = student_t.ppf(.975, df) * difference.std(ddof=1) / np.sqrt(len(dd)) if estimable else np.nan
    return {"estimate": estimate, "t": test.statistic if test else np.nan, "p": test.pvalue if test else np.nan,
            "n_first": len(dd), "n_second": len(dd), "n_clusters": len(dd), "n_overlap": len(dd), "df": df,
            "ci95_low": estimate - margin, "ci95_high": estimate + margin}


def run_confirmatory(metrics: pd.DataFrame, preregistration_id: str) -> pd.DataFrame:
    cells = planned_interval_cells(metrics)
    plan = confirmatory_plan()
    rows = []
    for outcome in OUTCOMES:
        event = event_changes(cells, outcome)
        for estimand in ("final_minute", "final_minus_early"):
            fs, es = independent_groups(event, estimand)
            semiology = event[event.event_type.eq("FS") & event.semiology.isin(("hypo", "hyper"))].groupby(
                ["participant", "semiology"], observed=True, as_index=False)[estimand].mean().rename(
                    columns={estimand: "value"})
            for contrast, test, direction, summary in (
                    ("FS versus ES", CONTRASTS[0][1], "FS minus ES", welch_summary(fs, es)),
                    ("FS hypo versus hyper", CONTRASTS[1][1], "hyper minus hypo", clustered_hypo_hyper(semiology)),
                    ("within-DD FS versus ES", CONTRASTS[2][1], "FS minus ES",
                     paired_summary(dd_pairs(event, estimand)))):
                rows.append({"preregistration_id": preregistration_id, "outcome": outcome, "estimand": estimand,
                             "contrast": contrast, "test": test, "estimate_direction": direction, **summary})
    result = pd.DataFrame(rows).merge(plan[["outcome", "estimand", "contrast", "family", "multiplicity"]],
                                      on=["outcome", "estimand", "contrast"], validate="one_to_one")
    descriptive = result.family.eq("descriptive")
    result.loc[descriptive, ["t", "p"]] = np.nan  # estimates and CIs only: no hypothesis test is reported
    result["status"] = np.where(descriptive, "descriptive_estimate_only",
                                np.where(np.isfinite(result.p), "estimated", "not_estimable"))
    result["p_holm"] = np.nan
    for family in ("primary", "secondary"):
        mask = result.family.eq(family)
        # Unavailable tests stay in the registered family, conservatively as p=1.
        adjusted = multipletests(result.loc[mask, "p"].fillna(1).to_numpy(), method="holm")[1]
        result.loc[mask, "p_holm"] = np.where(np.isfinite(result.loc[mask, "p"]), adjusted, np.nan)
    result["ci_scope"] = "pointwise_95_not_multiplicity_adjusted"
    order = {"primary": 0, "secondary": 1, "descriptive": 2}
    result["_order"] = result.family.map(order)
    return result.sort_values(["_order", "outcome", "estimand", "contrast"]).drop(columns="_order").reset_index(drop=True)


def primary_sensitivity(metrics: pd.DataFrame) -> pd.DataFrame:
    """Labelled sensitivities for the independent FS-ES change tests (primary and secondary outcomes).

    * Yuen 20% trimmed Welch test (robust to heavy tails and outlying participants);
    * studentised permutation test of the Welch statistic (no normality assumption);
    * the same contrast with the pre-final window [-120,-60) in place of [-60,0), checking that results are not
      driven by onset artefact in the last minute.
    None replaces or rescues the registered tests; no multiplicity adjustment is applied.
    """
    rows = []
    cells = planned_interval_cells(metrics)
    shifted = planned_interval_cells(metrics, final_endpoint=60.0)
    for outcome in (*PRIMARY_OUTCOMES, *SECONDARY_FS_ES_OUTCOMES):
        fs, es = independent_groups(event_changes(cells, outcome), "final_minus_early")
        base = {"outcome": outcome, "estimand": "final_minus_early", "contrast": "FS versus ES",
                "analysis_scope": "sensitivity_not_primary_confirmation", "n_first": len(fs), "n_second": len(es)}
        ok = len(fs) > 2 and len(es) > 2
        trimmed = ttest_ind(fs, es, equal_var=False, trim=TRIM) if ok else None
        rows.append({**base, "analysis": "yuen_20pct_trimmed_welch",
                     "estimate": trim_mean(fs, TRIM) - trim_mean(es, TRIM) if ok else np.nan,
                     "t": trimmed.statistic if ok else np.nan, "p": trimmed.pvalue if ok else np.nan})
        if ok:
            permutation = permutation_test(
                (fs.to_numpy(), es.to_numpy()),
                lambda a, b, axis: ttest_ind(a, b, axis=axis, equal_var=False).statistic,
                permutation_type="independent", n_resamples=PERMUTATIONS, alternative="two-sided",
                random_state=stable_rng(f"permutation|{outcome}"))
            rows.append({**base, "analysis": "studentised_permutation", "estimate": fs.mean() - es.mean(),
                         "t": float(permutation.statistic), "p": float(permutation.pvalue)})
        else:
            rows.append({**base, "analysis": "studentised_permutation", "estimate": np.nan, "t": np.nan, "p": np.nan})
        fs_p, es_p = independent_groups(event_changes(shifted, outcome), "final_minus_early")
        welch = welch_summary(fs_p, es_p)
        rows.append({**base, "analysis": "prefinal_window_-120_to_-60", "n_first": len(fs_p), "n_second": len(es_p),
                     "estimate": welch["estimate"], "t": welch["t"], "p": welch["p"]})
    return pd.DataFrame(rows)


def coverage_table(index: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Pre-outcome eligibility accounting by group and outcome (counts only; no outcome values are compared)."""
    events = index[index.condition.eq("preict")][["participant", "file_path", "group"]].drop_duplicates().rename(
        columns={"file_path": "event_key"})
    cells = planned_interval_cells(metrics)
    produced = set(metrics[metrics.condition.eq("preict")].event_key)
    rows = []
    for outcome in OUTCOMES:
        n_col = f"n_{outcome}"
        final = cells[cells.time_bin.eq(TIME_BINS[-1]) & np.isfinite(cells[outcome])].event_key
        early = cells[cells.time_bin.eq(TIME_BINS[0]) & cells[n_col].ge(MIN_EARLY_WINDOWS) &
                      np.isfinite(cells[outcome])].event_key
        complete = set(final) & set(early)
        for group, subset in events.groupby("group", observed=True):
            keys = set(subset.event_key)
            rows.append({"group": group, "outcome": outcome, "events_in_index": len(keys),
                         "events_with_any_evaluated_window": len(keys & produced),
                         "events_final_window_eligible": len(keys & set(final)),
                         "events_with_min_early_windows": len(keys & set(early)),
                         "events_complete": len(keys & complete),
                         "participants_in_index": subset.participant.nunique(),
                         "participants_complete": subset[subset.event_key.isin(complete)].participant.nunique()})
    return pd.DataFrame(rows)


def sample_summary(metrics: pd.DataFrame) -> dict[str, object]:
    preictal = metrics[metrics.condition.eq("preict")]
    selected = planned_interval_cells(metrics)
    # Counts are outcome-specific; this conservative summary requires all four outcomes.
    selected = selected.loc[np.isfinite(selected[list(OUTCOMES)]).all(axis=1)]
    enough = selected.time_bin.eq(TIME_BINS[-1]) | selected[[f"n_{o}" for o in OUTCOMES]].ge(MIN_EARLY_WINDOWS).all(axis=1)
    selected = selected.loc[enough]
    event_bins = selected.groupby(
        ["participant", "event_key", "group", "event_type", "semiology"],
        observed=True, dropna=False).time_bin.nunique().reset_index(name="n_bins")
    complete = event_bins[event_bins.n_bins.eq(2)]
    fs = complete[complete.event_type.eq("FS") & complete.semiology.isin(("hypo", "hyper"))]
    overlap = fs.groupby("participant", observed=True).semiology.nunique().eq(2).sum()
    dd = (complete[complete.group.eq("DD")].groupby(
        ["participant", "event_type"], observed=True).size().unstack().reindex(columns=["FS", "ES"]).dropna())
    return {
        "analysis_ready_participants": int(metrics.participant.nunique()),
        "participants_by_diagnostic_group": {
            str(key): int(value) for key, value in metrics.groupby("group", observed=True).participant.nunique().items()},
        "preictal_events_by_event_type": {
            str(key): int(value) for key, value in preictal.groupby("event_type", observed=True).event_key.nunique().items()},
        "complete_early_and_final_events_by_event_type": {
            str(key): int(value) for key, value in complete.groupby("event_type", observed=True).event_key.nunique().items()},
        "independent_complete_participants": {
            str(key): int(value) for key, value in complete[complete.group.isin(("FS", "ES"))].groupby(
                "group", observed=True).participant.nunique().items()},
        "fs_complete_events_by_semiology": {
            str(key): int(value) for key, value in fs.groupby("semiology", observed=True).event_key.nunique().items()},
        "fs_complete_participants_by_semiology": {
            str(key): int(value) for key, value in fs.groupby("semiology", observed=True).participant.nunique().items()},
        "fs_semiology_participant_clusters": int(fs.participant.nunique()),
        "fs_participants_in_both_semiology_strata": int(overlap),
        "paired_dd_complete_participants": int(len(dd)),
    }


def osf_preregistration_blurbs(summary: dict[str, object]) -> str:
    text = (Path(__file__).resolve().parent / "PREREGISTRATION.md").read_text(encoding="utf-8")
    fence = chr(96) * 3
    return text + "\n\n## Prepared cohort (computed without group tests)\n\n" + fence + "json\n" + json.dumps(summary, indent=2) + "\n" + fence + "\n"


def post_registration_runbook() -> str:
    return (Path(__file__).resolve().parent / "docs/RUNBOOK.md").read_text(encoding="utf-8")




def runtime_versions() -> dict[str, str]:
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in
        ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "matplotlib", "h5py")}}


def transition_cache_digest(path: Path) -> str:
    """SHA-256 of the transition CONTENT (sampling rate, length, sample indices, from/to states).

    MATLAB v7.3 files embed a creation timestamp, so the same transitions rebuilt later differ at byte level.
    Hashing content makes the data-freeze lock reproducible across independent runs and across a healed cache,
    while any change to a transition still changes the digest.
    """
    cache = load_transition_cache(path)
    digest = hashlib.sha256()
    digest.update(f"{float(cache['sfreq'])!r}|{int(cache['n_samples'])}".encode())
    for key in ("sample", "from", "to"):
        digest.update(np.ascontiguousarray(cache[key]).tobytes())
    return digest.hexdigest()


def direct_cache_hashes(index_path: Path) -> dict[str, str]:
    index = pd.read_csv(index_path, usecols=["cache_file"])
    paths = [Path(str(value)) for value in index.cache_file]
    if any(not p.is_absolute() for p in paths):
        raise ValueError("Transition cache paths must be absolute to remove working-directory ambiguity")
    return {str(p): transition_cache_digest(p) for p in sorted(set(paths))}


def validate_registration_id(value: str) -> None:
    url = urlparse(value)
    if (url.scheme != "https" or url.hostname not in {"osf.io", "doi.org"}
            or not url.path.strip("/") or any(x in value.lower() for x in ("replace", "placeholder", "example"))):
        raise ValueError("Supply the actual immutable registration HTTPS URL at osf.io or doi.org")
    # URL syntax is checked; only the researcher can attest that it is an immutable registration.


def artifact_digest(path: Path) -> str:
    """File hash for the freeze lock; JSON artifacts ignore top-level ``*_utc`` timestamps (rebuilt-identical stages)."""
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except ValueError:
            return sha256(path)
        if isinstance(value, dict):
            value = {k: v for k, v in value.items() if not k.endswith("_utc")}
        return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))
    return sha256(path)


def write_preregistration_lock(output: Path, artifact_paths: dict[str, Path],
                               plan_hash: str, summary: dict[str, object],
                               prehashed: dict[str, str] | None = None,
                               hash_provenance: str = "computed directly from files in this locked run") -> dict[str, object]:
    prehashed = prehashed or {}
    artifacts = {name: {"file_name": path.name,
                        "sha256": prehashed[name] if name in prehashed else artifact_digest(path)}
                 for name, path in artifact_paths.items()}
    lock = {
        "schema": "fs-ms-preregistration-lock/v2",
        "status": "LOCKED_PENDING_PREREGISTRATION",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "hash_algorithm": "SHA-256",
        "confirmatory_statistics_run": False,
        "unlock_condition": "immutable OSF registration plus --run-confirmatory and its public registration identifier",
        "planned_test_count": 24,
        "descriptive_estimate_only_count": 16,
        "primary_test_count": 2,
        "secondary_test_count": 6,
        "analysis_settings": ANALYSIS_SETTINGS,
        "runtime_versions": runtime_versions(),
        "direct_transition_caches": direct_cache_hashes(artifact_paths["transition_index"]),
        "route_level_test_count": 0,
        "locked_plan_sha256": plan_hash,
        "hash_provenance": hash_provenance,
        "prehashed_artifact_roles": sorted(prehashed),
        "sample_summary": summary,
        "artifacts": artifacts,
    }
    lock_path = output / "PREREGISTRATION_LOCK.json"
    write_text_exact(lock_path, json.dumps(lock, indent=2, sort_keys=True) + "\n")
    write_text_exact(output / "PREREGISTRATION_LOCK.sha256",
                     f"{sha256(lock_path)}  PREREGISTRATION_LOCK.json\n")
    return lock


def refresh_lock_from_existing(args: argparse.Namespace) -> None:
    raise RuntimeError("Lock refresh is disabled: corrected measurements require --prepare in a NEW output directory")


def verify_preregistration_lock(output: Path, artifact_paths: dict[str, Path],
                                expected_plan_hash: str) -> tuple[dict[str, object], str]:
    lock_path = output / "PREREGISTRATION_LOCK.json"
    sidecar = output / "PREREGISTRATION_LOCK.sha256"
    if not lock_path.exists() or not sidecar.exists():
        raise RuntimeError("Missing preregistration lock or SHA-256 sidecar; run the locked descriptive pipeline first")
    lock_hash = sha256(lock_path)
    sidecar_hash = sidecar.read_text(encoding="utf-8").strip().split()[0].lower()
    if sidecar_hash != lock_hash:
        raise RuntimeError("PREREGISTRATION_LOCK.sha256 does not match PREREGISTRATION_LOCK.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "fs-ms-preregistration-lock/v2" or lock.get("status") != "LOCKED_PENDING_PREREGISTRATION":
        raise RuntimeError("Preregistration lock schema or status is invalid")
    if lock.get("confirmatory_statistics_run") is not False:
        raise RuntimeError("Preregistration lock indicates that confirmatory statistics were already run")
    if lock.get("planned_test_count") != 24 or lock.get("locked_plan_sha256") != expected_plan_hash:
        raise RuntimeError("Current confirmatory plan does not match the preregistered plan")
    if lock.get("runtime_versions") != runtime_versions():
        raise RuntimeError("Runtime versions differ from registration; restore the locked environment")
    if lock.get("analysis_settings") != ANALYSIS_SETTINGS:
        raise RuntimeError("Analysis settings differ from registration")
    if lock.get("direct_transition_caches") != direct_cache_hashes(artifact_paths["transition_index"]):
        raise RuntimeError("A directly consumed transition cache differs from registration")
    locked_artifacts = lock.get("artifacts", {})
    if set(locked_artifacts) != set(artifact_paths):
        raise RuntimeError("Locked artifact set does not match the current pipeline artifact set")
    mismatches = []
    for name, path in artifact_paths.items():
        if not path.exists() or artifact_digest(path) != locked_artifacts[name].get("sha256"):
            mismatches.append(name)
    if mismatches:
        raise RuntimeError("Preregistration hash mismatch: " + ", ".join(mismatches))
    return lock, lock_hash


def update_review(output: Path, review: Path, scripts: list[Path]) -> None:
    review.mkdir(parents=True, exist_ok=True)
    shutil.copytree(output / "figures", review / "figures", dirs_exist_ok=True)
    (review / "tables").mkdir(exist_ok=True)
    stale_result = review / "tables" / "confirmatory_results.csv"
    if not (output / "tables" / "confirmatory_results.csv").exists() and stale_result.exists():
        stale_result.unlink()
    for source in (output / "tables").iterdir():
        if source.name not in {"timestamped_window_matrices.npz", "timestamped_window_metadata.csv.gz"}:
            shutil.copy2(source, review / "tables" / source.name)
    (review / "code").mkdir(exist_ok=True)
    for script in scripts:
        shutil.copy2(script, review / "code" / script.name)
    root_files = ("README.md", "ANALYSIS_AUDIT.json", "BACKFIT_RECOVERY_AUDIT.json",
                  "OSF_PREREGISTRATION_BLURBS.md", "POST_REGISTRATION_RUNBOOK.md", "PREREGISTRATION_LOCK.json",
                  "PREREGISTRATION_LOCK.sha256", "CONFIRMATORY_EXECUTION.json")
    for name in root_files:
        source = output / name
        destination = review / name
        if source.exists():
            shutil.copy2(source, destination)
        elif destination.exists():
            destination.unlink()
    with (review / "README.md").open("a", encoding="utf-8") as stream:
        stream.write("\n## Review package note\n\nThe large per-transition NPZ caches and full window-matrix NPZ are not duplicated here; their indexed locations and all analysis-ready metric tables are included.\n")
    files = [path for path in review.rglob("*") if path.is_file() and path.name != "SHA256SUMS.csv"]
    pd.DataFrame({"path": [str(path.relative_to(review)) for path in files],
                  "sha256": [sha256(path) for path in files], "bytes": [path.stat().st_size for path in files]}).to_csv(
                      review / "SHA256SUMS.csv", index=False)


def self_check() -> None:
    source = np.array([0, 0, 1, 1, 2]); destination = np.array([1, 2, 0, 2, 0])
    cpi = np.array([-1., -.5, 0, .5, 1]); csi = cpi[::-1]
    weights = action_weights(cpi, csi)
    counts = transition_counts(source, destination); tensor = tensor_counts(source, destination, weights)
    assert counts.sum() == 5 and np.isclose(tensor.sum(), 5)
    native = pd.DataFrame({"event_time_s": [0., 1., 2.], "csi_z": [0., -1., -2.], "cpi_z": [0., 2., 4.]})
    csi_level, cpi_level, csi_gradient, cpi_gradient = interpolate_native(native, np.array([.5, 1.5, 3.]))
    assert np.allclose(csi_level[:2], [-.5, -1.5]) and np.allclose(cpi_level[:2], [1., 3.])
    assert np.allclose(csi_gradient[:2], -1) and np.allclose(cpi_gradient[:2], 2)
    assert np.isnan(csi_level[2]) and np.isnan(cpi_gradient[2])
    gradient = gradient_weights(cpi_gradient, csi_gradient)
    assert np.all(gradient[:2, 1] > gradient[:2, [0, 2, 3]].max(axis=1))
    probability = row_probabilities(counts)
    assert np.allclose(probability.sum(axis=1), 1) and weighted_js(counts, probability) < .01
    dense = np.zeros((6, 6))
    for source_state in range(6):
        dense[source_state, np.arange(6) != source_state] = (20, 30, 40, 50, 60)
    dense_probability = row_probabilities(dense)
    kappa = fit_kappa(dense, dense_probability)
    assert .9 < kappa < 1.1
    baseline_metadata = pd.DataFrame([{"condition": "ii", "participant": "P", "group": "FS"}])
    baseline_tensor = np.broadcast_to(dense / 4, (1, 4, 6, 6)).copy()
    assert "P" in participant_baselines(baseline_metadata, dense[None], baseline_tensor, 10)[0]
    assert len(confirmatory_plan()) == 24
    synthetic = pd.DataFrame({
        "participant": ["P1", "P1", "P2", "P3", "P4", "P4"],
        "semiology": ["hypo", "hyper", "hypo", "hyper", "hypo", "hyper"],
        "value": [1., 2., 1.2, 2.2, .8, 1.8],
    })
    clustered = clustered_hypo_hyper(synthetic)
    assert np.isclose(clustered["estimate"], 1) and clustered["n_clusters"] == 4
    assert clustered["n_overlap"] == 2 and np.isfinite(clustered["p"])
    event_design = [
        ("F1", "FS", "FS", "hypo", 1.), ("F1", "FS", "FS", "hyper", 1.2),
        ("F2", "FS", "FS", "hypo", 1.1), ("F3", "FS", "FS", "hyper", 1.3),
        ("E1", "ES", "ES", np.nan, .7), ("E2", "ES", "ES", np.nan, .8),
        ("E3", "ES", "ES", np.nan, .9),
        ("D1", "DD", "FS", "hypo", 1.), ("D1", "DD", "ES", np.nan, .8),
        ("D2", "DD", "FS", "hyper", 1.2), ("D2", "DD", "ES", np.nan, .9),
        ("D3", "DD", "FS", "hypo", 1.1), ("D3", "DD", "ES", np.nan, .85),
    ]
    confirmatory_rows = []
    for number, (participant, group, event_type, semiology, value) in enumerate(event_design):
        for endpoint in (600., 610., 620., 630., 640., 0.):
            offset = 0. if endpoint else .05 + .01 * (number % 4)
            row = {"participant": participant, "event_key": f"E{number}", "group": group,
                   "event_type": event_type, "semiology": semiology,
                   "condition": "preict", "window_end_to_ictus_s": endpoint}
            row.update({outcome: value + offset for outcome in OUTCOMES})
            confirmatory_rows.append(row)
    synthetic_result = run_confirmatory(pd.DataFrame(confirmatory_rows), "synthetic-preregistration")
    assert len(synthetic_result) == 24
    assert synthetic_result.family.value_counts().to_dict() == {"descriptive": 16, "secondary": 6, "primary": 2}
    assert synthetic_result.loc[synthetic_result.family.eq("descriptive"), ["t", "p", "p_holm"]].isna().all().all()
    synthetic_primary = synthetic_result[
        synthetic_result.outcome.eq("ordinary_js_excess") & synthetic_result.estimand.eq("final_minute")]
    assert synthetic_primary.loc[synthetic_primary.contrast.eq("FS versus ES"), "n_first"].iat[0] == 3
    assert synthetic_primary.loc[synthetic_primary.contrast.eq("FS hypo versus hyper"), "n_overlap"].iat[0] == 1
    assert synthetic_primary.loc[synthetic_primary.contrast.eq("within-DD FS versus ES"), "n_first"].iat[0] == 3
    print("timestamped transition self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition-index", type=Path, required=True)
    parser.add_argument("--autonomic-native", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--plan-lock", type=Path,
                        help="prereg/PLAN_LOCK.json; when given, its hash is bound into the data-freeze lock")
    parser.add_argument("--prior-strength", type=float, default=PRIOR_STRENGTH)
    parser.add_argument("--run-confirmatory", action="store_true")
    parser.add_argument("--preregistration-id", default="")
    parser.add_argument("--refresh-preregistration-lock", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--prepare", action="store_true", help="Explicitly compute descriptive measurements in a fresh directory")
    parser.add_argument("--check-inputs", action="store_true", help="Check input availability only, without data analysis")
    parser.add_argument("--verify-lock", action="store_true", help="Verify the frozen bundle without fitting or testing")
    args = parser.parse_args()
    if args.self_check:
        self_check(); return
    if args.refresh_preregistration_lock:
        if args.run_confirmatory:
            parser.error("--refresh-preregistration-lock cannot be combined with --run-confirmatory")
        try:
            refresh_lock_from_existing(args)
        except RuntimeError as error:
            parser.error(str(error))
        return
    if sum(bool(x) for x in (args.prepare, args.check_inputs, args.verify_lock, args.run_confirmatory)) != 1:
        parser.error("Choose exactly one mode: --check-inputs, --prepare, --verify-lock, or --run-confirmatory")
    if args.prior_strength != PRIOR_STRENGTH:
        parser.error("Prior strength is fixed at 10; changing it requires an explicit protocol amendment")
    if args.run_confirmatory:
        try:
            validate_registration_id(args.preregistration_id.strip())
        except ValueError as error:
            parser.error(str(error))
    if args.check_inputs:
        paths = [args.transition_index, args.autonomic_native, args.manifest,
                 args.transition_index.parent / "recovery_audit.json"]
        missing = [str(p) for p in paths if not p.is_file()]
        print(json.dumps({"mode": "check_inputs_only", "missing_paths": missing, "ready": not missing,
                          "runtime": runtime_versions()}, indent=2))
        if missing:
            raise SystemExit(2)
        return
    for path in (args.transition_index, args.autonomic_native, args.manifest):
        if not path.exists(): parser.error(f"Input does not exist: {path}")
    recovery_audit_path = args.transition_index.parent / "recovery_audit.json"
    if not recovery_audit_path.exists(): parser.error(f"Backfit audit does not exist: {recovery_audit_path}")
    recovery_audit = json.loads(recovery_audit_path.read_text(encoding="utf-8"))
    focused_gate = recovery_audit.get("focused_window_fit_recovery_validation", {})
    if not (recovery_audit.get("compact_export_validated_against_full_backfit") and
            focused_gate.get("transition_matrix_exact_match")):
        parser.error("Backfit recovery did not pass both exact validation gates")
    try:
        fit_to_canonical(recovery_audit.get("fit_labels", ()))
    except ValueError as error:
        parser.error(f"recovery_audit.json lacks valid derived fit_labels (rerun the microstates stage): {error}")

    output, tables, figures = args.out_dir, args.out_dir / "tables", args.out_dir / "figures"
    repo = Path(__file__).resolve().parent
    scripts = [Path(__file__).resolve(), repo / "recover_timestamped_transition_backfits.m",
               repo / "run_pipeline.py",
               repo / "external/VG-microstates/fit_microstate_hierarchical_dataset.m",
               repo / "external/VG-microstates/_internal/matlab/backfit_microstate_timecourse.m",
               repo / "run_vg_microstates.m", repo / "requirements-lock.txt", repo / "docs/STATISTICAL_ANALYSIS_PLAN.md",
               repo / "PREREGISTRATION.md", repo / "docs/RUNBOOK.md"]
    plan = confirmatory_plan()
    plan_text = plan.to_csv(index=False, lineterminator="\n")
    plan_hash = sha256_text(plan_text)
    artifact_paths = {
        "analysis_plan": tables / "confirmatory_analysis_plan_WITHHELD.csv",
        "osf_preregistration_blurbs": output / "OSF_PREREGISTRATION_BLURBS.md",
        "post_registration_runbook": output / "POST_REGISTRATION_RUNBOOK.md",
        "analysis_python": scripts[0], "backfit_exporter_matlab": scripts[1],
        "pipeline_runner": scripts[2], "hierarchy_fit_matlab": scripts[3],
        "backfit_function_matlab": scripts[4], "transition_index": args.transition_index,
        "backfit_recovery_audit": recovery_audit_path, "autonomic_native_input": args.autonomic_native,
        "manifest_input": args.manifest,
        "upstream_runner": scripts[5], "dependencies": scripts[6], "statistical_analysis_plan": scripts[7],
        "preregistration_source": scripts[8], "runbook_source": scripts[9],
        "prepared_metrics": tables / "timestamped_window_metrics.csv.gz",
        "prepared_audit": output / "ANALYSIS_AUDIT.json",
    }
    if args.plan_lock:
        if not args.plan_lock.is_file():
            parser.error(f"Plan lock does not exist: {args.plan_lock}")
        artifact_paths["plan_lock"] = args.plan_lock
    registered_lock = None
    registered_lock_hash = None
    if args.run_confirmatory or args.verify_lock:
        try:
            registered_lock, registered_lock_hash = verify_preregistration_lock(output, artifact_paths, plan_hash)
        except RuntimeError as error:
            parser.error(str(error))

    if args.verify_lock:
        print(json.dumps({"status": "verified_without_statistics", "lock_sha256": registered_lock_hash}))
        return
    if args.run_confirmatory:
        result_path = tables / "confirmatory_results.csv"
        execution_path = output / "CONFIRMATORY_EXECUTION.json"
        if result_path.exists() or execution_path.exists():
            parser.error("A confirmatory result/execution already exists; preserve it rather than overwrite")
        metrics = pd.read_csv(artifact_paths["prepared_metrics"])
        with result_path.open("x", encoding="utf-8", newline="") as stream:
            run_confirmatory(metrics, args.preregistration_id.strip()).to_csv(stream, index=False)
        sensitivity_path = tables / "primary_sensitivity_results.csv"
        sensitivity = primary_sensitivity(metrics)
        with sensitivity_path.open("x", encoding="utf-8", newline="") as stream:
            sensitivity.to_csv(stream, index=False)
        execution = {"schema": "fs-ms-confirmatory-execution/v2", "status": "EXECUTED_AFTER_RESEARCHER_REGISTRATION_ATTESTATION",
            "executed_utc": datetime.now(timezone.utc).isoformat(),
            "preregistration_id": args.preregistration_id.strip(),
            "registration_existence_independently_verified": False,
            "registered_lock_sha256": registered_lock_hash,
            "all_registered_hashes_verified_before_statistics": True,
            "confirmatory_results_sha256": sha256(result_path),
            "primary_sensitivity_sha256": sha256(sensitivity_path)}
        write_text_exact(execution_path, json.dumps(execution, indent=2, sort_keys=True) + "\n")
        if args.review_dir:
            update_review(output, args.review_dir, scripts)
        print(json.dumps(execution, indent=2))
        return
    if output.exists() and any(output.iterdir()):
        parser.error("Preparation requires a NEW empty output directory; existing results and locks are preserved")
    output.mkdir(parents=True, exist_ok=True)
    tables.mkdir(); figures.mkdir()
    initial_input_hashes = {str(p): sha256(p) for p in
                           (args.transition_index, args.autonomic_native, args.manifest, recovery_audit_path)}
    initial_cache_hashes = direct_cache_hashes(args.transition_index)
    shutil.copy2(recovery_audit_path, output / "BACKFIT_RECOVERY_AUDIT.json")
    index = pd.read_csv(args.transition_index)
    if index.file_path.duplicated().any():
        raise ValueError("Duplicate recording identity in transition index")
    if index[["participant", "group"]].drop_duplicates().participant.duplicated().any():
        raise ValueError("Participant has conflicting diagnostic groups")
    index["file_key"] = normalise_path(index.file_path)
    index["condition"] = index.condition.astype(str).str.lower()
    manifest = pd.read_csv(args.manifest)
    manifest["file_key"] = normalise_path(manifest.file_path)
    event_columns = [column for column in ("file_key", "seizure_type", "hypo_hyper") if column in manifest]
    event = manifest[event_columns].drop_duplicates()
    if event.file_key.duplicated().any():
        raise ValueError("Conflicting event labels in manifest")
    index = index.merge(event, on="file_key", how="left", validate="one_to_one")
    index["event_type"] = index.get("seizure_type", pd.Series(index=index.index, dtype=object)).astype(str).str.upper()
    index.loc[~index.event_type.isin(["FS", "ES"]), "event_type"] = np.nan
    index["semiology"] = index.get("hypo_hyper", pd.Series(index=index.index, dtype=object)).astype(str).str.lower()
    index.loc[~index.semiology.isin(["hypo", "hyper"]), "semiology"] = np.nan

    native = load_native(args.autonomic_native, set(index.file_key))
    metadata, ordinary, tensor, action_index = build_windows(
        index, native, output / "transition_action_cache", recovery_audit["fit_labels"])
    baselines, precision = participant_baselines(metadata, ordinary, tensor, args.prior_strength)
    metrics = evaluate_windows(metadata, ordinary, tensor, baselines, args.prior_strength)
    baseline_probability = baseline_tables(baselines, metadata)
    scores, loadings = manifold(metrics)

    metadata.to_csv(tables / "timestamped_window_metadata.csv.gz", index=False, compression="gzip")
    action_index.to_csv(tables / "transition_action_cache_index.csv", index=False)
    np.savez_compressed(tables / "timestamped_window_matrices.npz", ordinary=ordinary.astype(np.uint16),
                        tensor=tensor.astype(np.float32))
    metrics.to_csv(tables / "timestamped_window_metrics.csv.gz", index=False, compression="gzip")
    baseline_probability.to_csv(tables / "baseline_B_tensor_probabilities.csv.gz", index=False, compression="gzip")
    precision.to_csv(tables / "baseline_B_tensor_precision_cells.csv.gz", index=False, compression="gzip")
    precision_summary = precision[precision.group.isin(["FS", "ES"])].groupby(
        ["group", "action"], observed=True).agg(
            n_participants=("participant", "nunique"), median_effective_transitions=("effective_weighted_transitions", "median"),
            median_posterior_95_width=("posterior_95_width", "median"),
            q25_posterior_95_width=("posterior_95_width", lambda x: x.quantile(.25)),
            q75_posterior_95_width=("posterior_95_width", lambda x: x.quantile(.75))).reset_index()
    precision_summary.to_csv(tables / "baseline_B_tensor_precision_summary.csv", index=False)
    scores.to_csv(tables / "peri_ictal_manifold_scores.csv.gz", index=False, compression="gzip")
    loadings.to_csv(tables / "peri_ictal_manifold_loadings.csv", index=False)
    summary = sample_summary(metrics)
    coverage_table(index, metrics).to_csv(tables / "coverage_by_group.csv", index=False)
    if not args.run_confirmatory:
        write_text_exact(tables / "confirmatory_analysis_plan_WITHHELD.csv", plan_text)
        write_text_exact(output / "OSF_PREREGISTRATION_BLURBS.md", osf_preregistration_blurbs(summary))
        write_text_exact(output / "POST_REGISTRATION_RUNBOOK.md", post_registration_runbook())

    if not scores.empty:
        plot_manifold(scores, figures)
    plot_matrix_trajectories(metrics, figures)
    plot_baseline_precision(precision, figures)
    plot_autonomic_state_trajectories(metrics, figures)
    audit = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input_resolution": "genuine ordered microstate-change samples at 1/256 s with heartbeat-resolved autonomic interpolation",
        "transition_timestamps_fabricated": False,
        "autonomic_extrapolation": False,
        "action_tensor": "fractional soft CPI/CSI gradient-quadrant weights assigned at each genuine transition timestamp",
        "gradient_definition": "centred finite-difference slopes on heartbeat times, interpolated without extrapolation; soft signs scaled by record median absolute slope",
        "autonomic_state_description": "CPI/CSI levels, soft level quadrants and peri-ictal level trajectories retained descriptively",
        "canonical_state_order": list(STATES), "action_order": list(ACTIONS),
        "minimum_window_transitions": MIN_WINDOW_TRANSITIONS,
        "minimum_reference_transitions": MIN_REFERENCE_TRANSITIONS, "minimum_early_windows": MIN_EARLY_WINDOWS,
        "js_outcome": "null-calibrated excess JS (JS - E0); E0 = mean JS over hash-seeded parametric-bootstrap "
                      "destinations drawn from the participant reference given the window's own sources and soft weights",
        "null_draws": NULL_DRAWS, "log_js_descriptive_only": True,
        "family_sizes": {"primary": 2, "secondary": 6, "descriptive_estimate_only": 16},
        "route_level_hypothesis_tests": 0,
        "confirmatory_statistics_run": bool(args.run_confirmatory),
        "preregistration_id": args.preregistration_id.strip() if args.run_confirmatory else None,
        "preregistration_lock_sha256": registered_lock_hash,
        "registered_plan_sha256": plan_hash,
        "preregistration_hashes_verified_before_statistics": bool(args.run_confirmatory),
        "hypo_hyper_test": "participant-clustered OLS on participant-by-semiology means, CR2 + Bell-McCaffrey df",
        "n_transition_records": len(index), "n_exact_transitions": int(action_index.n_transitions.sum()),
        "n_transitions_with_interpolated_action": int(action_index.n_with_interpolated_action.sum()),
        "interpolated_action_fraction": float(action_index.n_with_interpolated_action.sum() / action_index.n_transitions.sum()),
        "n_windows": len(metadata), "n_metric_windows": len(metrics),
        "scripts": {str(path): sha256(path) for path in scripts},
        "inputs": {str(path): sha256(path) for path in
                   (args.transition_index, recovery_audit_path, args.autonomic_native, args.manifest)},
    }
    write_text_exact(output / "ANALYSIS_AUDIT.json", json.dumps(audit, indent=2) + "\n")
    if any(sha256(Path(p)) != digest for p, digest in initial_input_hashes.items()):
        raise RuntimeError("Inputs changed during preparation; do not register this incomplete bundle")
    if initial_cache_hashes != direct_cache_hashes(args.transition_index):
        raise RuntimeError("Transition cache changed during preparation; do not register this bundle")
    registered_lock = write_preregistration_lock(output, artifact_paths, plan_hash, summary)
    registered_lock_hash = sha256(output / "PREREGISTRATION_LOCK.json")
    (output / "README.md").write_text(
        "# Timestamped transition analysis\n\n"
        "This run uses every genuine ordered microstate transition at EEG-sample resolution and linearly "
        "interpolates heartbeat-resolved CSI/CPI levels and gradients only within observed support. It reports "
        "original-matrix and autonomic-gradient tensor JS/κ, absolute interictal gradient-tensor precision, and descriptive peri-ictal "
        "manifold movement. No route-level test family is produced. The tensor is observational; temporal "
        "interpolation does not establish causality.\n\n"
        "## Where to start\n\n"
        "- `tables/timestamped_window_metrics.csv.gz`: original-matrix and gradient-tensor JS/κ, CPI/CSI levels and slopes, plus exact-30 sensitivities.\n"
        "- `tables/baseline_B_tensor_precision_summary.csv`: absolute interictal gradient-tensor precision for FS and ES.\n"
        "- `tables/peri_ictal_manifold_scores.csv.gz`: event-by-period 2D movement coordinates.\n"
        "- `transition_action_cache/*.npz`: every genuine transition sample, canonical source/destination, CPI/CSI levels, gradients, descriptive state weights and gradient-tensor weights.\n"
        "- `tables/transition_action_cache_index.csv`: links those caches to source records.\n"
        "- `OSF_PREREGISTRATION_BLURBS.md`: concise, paste-ready text for the OSF form.\n"
        "- `POST_REGISTRATION_RUNBOOK.md`: exact hash-gated command to use after registration.\n"
        "- `PREREGISTRATION_LOCK.json` and `.sha256`: immutable plan/code/input hashes checked before inference.\n\n"
        + ("Confirmatory tests were run under preregistration `" + args.preregistration_id + "`.\n" if args.run_confirmatory else
           "**Confirmatory FS–ES, FS hypo–hyper, and paired within-DD tests are intentionally withheld pending preregistration.**\n"),
        encoding="utf-8")
    if args.review_dir:
        update_review(output, args.review_dir, scripts)
    print(json.dumps({"n_records": len(index), "n_windows": len(metadata),
                      "confirmatory_statistics_run": bool(args.run_confirmatory)}, indent=2))


if __name__ == "__main__":
    main()
