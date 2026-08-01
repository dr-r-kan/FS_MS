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

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import expit
from scipy.stats import beta, ttest_ind, ttest_rel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


STATES = ("F", "E", "G", "B", "A", "C")
FIT_LABELS = ("A", "F", "C", "G", "B", "E")
FIT_TO_CANONICAL = np.array([STATES.index(label) for label in FIT_LABELS])
ACTIONS = ("dCPI+/dCSI+", "dCPI+/dCSI-", "dCPI-/dCSI+", "dCPI-/dCSI-")
STATE_ACTIONS = ("CPI+/CSI+", "CPI+/CSI-", "CPI-/CSI+", "CPI-/CSI-")
TIME_BINS = ("-20 to -10", "-10 to -5", "-5 to -2", "-2 to -1", "-1 to 0")
TIME_MIDPOINT = dict(zip(TIME_BINS, (-15, -7.5, -3.5, -1.5, -.5)))
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
        normalised.append(values / scale if np.isfinite(scale) and scale > 1e-12 else np.zeros_like(values))
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
    if not valid.any():
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

    kappa = 1.0
    for _ in range(30):
        likelihood, gradient, hessian = objective(kappa)
        if hessian >= -1e-10:
            break
        step = -gradient / hessian
        if abs(step) < 1e-7:
            break
        candidate = float(np.clip(kappa + step, math.exp(-4), math.exp(4)))
        for _ in range(10):
            if objective(candidate)[0] >= likelihood:
                break
            step *= .5; candidate = float(np.clip(kappa + step, math.exp(-4), math.exp(4)))
        kappa = candidate
    return float(kappa)


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


def stable_sample(size: int, number: int, key: str) -> np.ndarray:
    if size < number:
        return np.array([], dtype=int)
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    return np.random.default_rng(seed).choice(size, number, replace=False)


def interpolate_native(native: pd.DataFrame, times: np.ndarray) -> tuple[np.ndarray, ...]:
    if native.empty:
        missing = np.full(len(times), np.nan)
        return missing.copy(), missing.copy(), missing.copy(), missing
    levels, gradients = [], []
    for column in ("csi_z", "cpi_z"):
        subset = native[["event_time_s", column]].apply(pd.to_numeric, errors="coerce").dropna()
        subset = subset.groupby("event_time_s", as_index=False)[column].mean().sort_values("event_time_s")
        if len(subset) < 2:
            levels.append(np.full(len(times), np.nan)); gradients.append(np.full(len(times), np.nan)); continue
        x, y = subset.event_time_s.to_numpy(), subset[column].to_numpy()
        levels.append(np.interp(times, x, y, left=np.nan, right=np.nan))
        gradients.append(np.interp(times, x, np.gradient(y, x), left=np.nan, right=np.nan))
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


def build_windows(index: pd.DataFrame, native: dict[str, pd.DataFrame], action_cache: Path
                  ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    metadata, ordinary, tensor, ordinary30, tensor30, action_rows = [], [], [], [], [], []
    action_cache.mkdir(parents=True, exist_ok=True)
    for record_number, row in enumerate(index.itertuples(index=False), 1):
        transition = load_transition_cache(Path(row.cache_file))
        times = (transition["sample"] - 1) / transition["sfreq"]
        source = FIT_TO_CANONICAL[transition["from"] - 1]
        destination = FIT_TO_CANONICAL[transition["to"] - 1]
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
            if right - left < 30:
                continue
            sl = slice(left, right)
            counts = transition_counts(source[sl], destination[sl])
            weighted = tensor_counts(source[sl], destination[sl], action[sl])
            valid_action = np.isfinite(action[sl]).all(axis=1)
            sample = stable_sample(right - left, 30, f"{row.file_key}|{start:.9f}|ordinary")
            valid_positions = np.flatnonzero(valid_action)
            sample_action_local = stable_sample(len(valid_positions), 30, f"{row.file_key}|{start:.9f}|tensor")
            sample_action = valid_positions[sample_action_local] if len(sample_action_local) else np.array([], dtype=int)
            c30 = transition_counts(source[sl][sample], destination[sl][sample])
            t30 = (tensor_counts(source[sl][sample_action], destination[sl][sample_action], action[sl][sample_action])
                   if len(sample_action) else np.full((4, 6, 6), np.nan))
            endpoint = duration - stop if row.condition == "preict" else np.nan
            metadata.append({
                "row_id": len(metadata), "participant": row.participant, "file_path": row.file_path,
                "file_key": row.file_key, "event_key": row.file_path, "condition": row.condition,
                "group": row.group, "event_type": row.event_type, "semiology": row.semiology,
                "window_start_s": start, "window_end_s": stop, "window_end_to_ictus_s": endpoint,
                "time_bin": time_bin(endpoint) if row.condition == "preict" else "II",
                "n_transitions": right - left, "n_action_transitions": int(valid_action.sum()),
                "csi_transition_mean": np.nanmean(csi[sl]) if np.isfinite(csi[sl]).any() else np.nan,
                "cpi_transition_mean": np.nanmean(cpi[sl]) if np.isfinite(cpi[sl]).any() else np.nan,
                "csi_gradient_mean_z_per_s": np.nanmean(csi_gradient[sl]) if np.isfinite(csi_gradient[sl]).any() else np.nan,
                "cpi_gradient_mean_z_per_s": np.nanmean(cpi_gradient[sl]) if np.isfinite(cpi_gradient[sl]).any() else np.nan,
                **{f"state_weight_{name.lower().replace('/', '_').replace('+', 'pos').replace('-', 'neg')}":
                   np.nanmean(state_action[sl, number]) if np.isfinite(state_action[sl, number]).any() else np.nan
                   for number, name in enumerate(STATE_ACTIONS)},
            })
            ordinary.append(counts); tensor.append(weighted); ordinary30.append(c30); tensor30.append(t30)
        if record_number % 25 == 0 or record_number == len(index):
            print(f"[{record_number}/{len(index)}] timestamped records windowed", flush=True)
    return (pd.DataFrame(metadata), np.stack(ordinary), np.stack(tensor),
            np.stack(ordinary30), np.stack(tensor30), pd.DataFrame(action_rows))


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
                     ordinary30: np.ndarray, tensor30: np.ndarray,
                     baselines: dict[str, dict[str, np.ndarray]], prior_strength: float) -> pd.DataFrame:
    rows = []
    for index, meta in metadata.iterrows():
        baseline = baselines.get(str(meta.participant))
        if baseline is None:
            continue
        ordinary_reference = baseline["ordinary"].copy()
        tensor_reference = baseline["tensor"].copy()
        if meta.condition == "ii":
            ordinary_reference -= ordinary[index]
            tensor_reference -= tensor[index]
        if ordinary_reference.sum() < 30:
            continue
        ordinary_probability = row_probabilities(ordinary_reference)
        tensor_probability = row_probabilities(
            tensor_reference, np.broadcast_to(ordinary_probability, tensor_reference.shape), prior_strength)
        result = meta.to_dict()
        result.update({
            "ordinary_js": weighted_js(ordinary[index], ordinary_probability),
            "ordinary_kappa": fit_kappa(ordinary[index], ordinary_probability),
            "ordinary_entropy_concentration": entropy_concentration(ordinary[index]),
            "tensor_js": weighted_js(tensor[index], tensor_probability) if meta.n_action_transitions >= 30 else np.nan,
            "tensor_kappa": fit_kappa(tensor[index], tensor_probability) if meta.n_action_transitions >= 30 else np.nan,
            "tensor_entropy_concentration": entropy_concentration(tensor[index]) if meta.n_action_transitions >= 30 else np.nan,
            "ordinary_js_30": weighted_js(ordinary30[index], ordinary_probability),
            "ordinary_kappa_30": fit_kappa(ordinary30[index], ordinary_probability),
            "tensor_js_30": weighted_js(tensor30[index], tensor_probability) if np.isfinite(tensor30[index]).all() else np.nan,
            "tensor_kappa_30": fit_kappa(tensor30[index], tensor_probability) if np.isfinite(tensor30[index]).all() else np.nan,
        })
        for stem in ("ordinary", "tensor"):
            result[f"log_{stem}_js"] = math.log(max(result[f"{stem}_js"], 1e-12)) if np.isfinite(result[f"{stem}_js"]) else np.nan
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
    return pd.concat(score_parts, ignore_index=True), pd.DataFrame(loading_parts)


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


def confirmatory_plan() -> pd.DataFrame:
    rows = []
    for outcome in ("log_ordinary_js", "log_ordinary_kappa", "log_tensor_js", "log_tensor_kappa"):
        for estimand in ("final_minute", "final_minus_early"):
            for contrast, test in (("FS versus ES", "Welch participant-level t test"),
                                   ("FS hypo versus hyper", "Welch participant-level t test"),
                                   ("within-DD FS versus ES", "paired participant-level t test")):
                rows.append({"outcome": outcome, "estimand": estimand, "contrast": contrast, "test": test,
                             "tensor_definition": "soft CPI/CSI gradient quadrants at genuine transition times",
                             "alternative": "two-sided (direction to be finalised at preregistration)",
                             "multiplicity": "Holm family-wise correction across 24 planned tests",
                             "status": "WITHHELD_PENDING_PREREGISTRATION"})
    return pd.DataFrame(rows)


def run_confirmatory(metrics: pd.DataFrame, preregistration_id: str) -> pd.DataFrame:
    outcomes = ("log_ordinary_js", "log_ordinary_kappa", "log_tensor_js", "log_tensor_kappa")
    keys = ["participant", "event_key", "group", "event_type", "semiology"]
    cells = metrics[metrics.condition.eq("preict") & metrics.time_bin.isin((TIME_BINS[0], TIME_BINS[-1]))].groupby(
        [*keys, "time_bin"], observed=True, dropna=False, as_index=False)[list(outcomes)].mean()
    rows = []
    for outcome in outcomes:
        early = cells[cells.time_bin.eq(TIME_BINS[0])][keys + [outcome]].rename(columns={outcome: "early"})
        final = cells[cells.time_bin.eq(TIME_BINS[-1])][keys + [outcome]].rename(columns={outcome: "final_minute"})
        event = final.merge(early, on=keys, how="inner", validate="one_to_one")
        event["final_minus_early"] = event.final_minute - event.early
        for estimand in ("final_minute", "final_minus_early"):
            participant = event.groupby(
                ["participant", "group", "event_type", "semiology"], observed=True, dropna=False,
                as_index=False)[estimand].mean().rename(columns={estimand: "value"})
            designs = [
                ("FS versus ES", participant[participant.group.isin(["FS", "ES"])], "event_type", "FS", "ES", False),
                ("FS hypo versus hyper", participant[participant.event_type.eq("FS")], "semiology", "hypo", "hyper", False),
            ]
            for name, frame, column, first, second, paired in designs:
                a = frame[frame[column].astype(str).str.lower().eq(first.lower())].value.dropna()
                b = frame[frame[column].astype(str).str.lower().eq(second.lower())].value.dropna()
                test = ttest_ind(a, b, equal_var=False) if len(a) > 1 and len(b) > 1 else None
                rows.append({"preregistration_id": preregistration_id, "outcome": outcome, "estimand": estimand,
                             "contrast": name, "estimate": a.mean() - b.mean(), "t": test.statistic if test else np.nan,
                             "p": test.pvalue if test else np.nan, "n_first": len(a), "n_second": len(b)})
            dd = (participant[participant.group.eq("DD")].groupby(
                ["participant", "event_type"], observed=True, as_index=False).value.mean()
                  .pivot(index="participant", columns="event_type", values="value").dropna())
            test = ttest_rel(dd.FS, dd.ES) if {"FS", "ES"}.issubset(dd) and len(dd) > 1 else None
            rows.append({"preregistration_id": preregistration_id, "outcome": outcome, "estimand": estimand,
                         "contrast": "within-DD FS versus ES", "estimate": (dd.FS - dd.ES).mean() if len(dd) else np.nan,
                         "t": test.statistic if test else np.nan, "p": test.pvalue if test else np.nan,
                         "n_first": len(dd), "n_second": len(dd)})
    result = pd.DataFrame(rows)
    valid = result.p.notna()
    if valid.any():
        p_values = result.loc[valid, "p"].to_numpy()
        order = np.argsort(p_values)
        adjusted_ordered = np.maximum.accumulate((len(p_values) - np.arange(len(p_values))) * p_values[order])
        adjusted = np.empty(len(p_values)); adjusted[order] = np.clip(adjusted_ordered, 0, 1)
        result.loc[valid, "p_holm"] = adjusted
    return result


def update_review(output: Path, review: Path, scripts: list[Path]) -> None:
    review.mkdir(parents=True, exist_ok=True)
    shutil.copytree(output / "figures", review / "figures", dirs_exist_ok=True)
    (review / "tables").mkdir(exist_ok=True)
    for source in (output / "tables").iterdir():
        if source.name not in {"timestamped_window_matrices.npz", "timestamped_window_metadata.csv.gz"}:
            shutil.copy2(source, review / "tables" / source.name)
    (review / "code").mkdir(exist_ok=True)
    for script in scripts:
        shutil.copy2(script, review / "code" / script.name)
    for name in ("README.md", "ANALYSIS_AUDIT.json", "BACKFIT_RECOVERY_AUDIT.json", "PREREGISTRATION_LOCK.json"):
        shutil.copy2(output / name, review / name)
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
    print("timestamped transition self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition-index", type=Path, required=True)
    parser.add_argument("--autonomic-native", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--prior-strength", type=float, default=10)
    parser.add_argument("--run-confirmatory", action="store_true")
    parser.add_argument("--preregistration-id", default="")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(); return
    if args.run_confirmatory and not args.preregistration_id.strip():
        parser.error("--run-confirmatory requires --preregistration-id after preregistration")
    for path in (args.transition_index, args.autonomic_native, args.manifest):
        if not path.exists(): parser.error(f"Input does not exist: {path}")
    recovery_audit_path = args.transition_index.parent / "recovery_audit.json"
    if not recovery_audit_path.exists(): parser.error(f"Backfit audit does not exist: {recovery_audit_path}")
    recovery_audit = json.loads(recovery_audit_path.read_text(encoding="utf-8"))
    focused_gate = recovery_audit.get("focused_window_fit_recovery_validation", {})
    if not (recovery_audit.get("compact_export_validated_against_full_backfit") and
            focused_gate.get("transition_matrix_exact_match")):
        parser.error("Backfit recovery did not pass both exact validation gates")

    output, tables, figures = args.out_dir, args.out_dir / "tables", args.out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(exist_ok=True)
    shutil.copy2(recovery_audit_path, output / "BACKFIT_RECOVERY_AUDIT.json")
    index = pd.read_csv(args.transition_index)
    index["file_key"] = normalise_path(index.file_path)
    index["condition"] = index.condition.astype(str).str.lower()
    manifest = pd.read_csv(args.manifest)
    manifest["file_key"] = normalise_path(manifest.file_path)
    event_columns = [column for column in ("file_key", "seizure_type", "hypo_hyper") if column in manifest]
    event = manifest[event_columns].drop_duplicates("file_key")
    index = index.merge(event, on="file_key", how="left", validate="one_to_one")
    index["event_type"] = index.get("seizure_type", pd.Series(index=index.index, dtype=object)).astype(str).str.upper()
    index.loc[~index.event_type.isin(["FS", "ES"]), "event_type"] = np.nan
    index["semiology"] = index.get("hypo_hyper", pd.Series(index=index.index, dtype=object)).astype(str).str.lower()
    index.loc[~index.semiology.isin(["hypo", "hyper"]), "semiology"] = np.nan

    native = load_native(args.autonomic_native, set(index.file_key))
    metadata, ordinary, tensor, ordinary30, tensor30, action_index = build_windows(
        index, native, output / "transition_action_cache")
    baselines, precision = participant_baselines(metadata, ordinary, tensor, args.prior_strength)
    metrics = evaluate_windows(metadata, ordinary, tensor, ordinary30, tensor30, baselines, args.prior_strength)
    baseline_probability = baseline_tables(baselines, metadata)
    scores, loadings = manifold(metrics)

    metadata.to_csv(tables / "timestamped_window_metadata.csv.gz", index=False, compression="gzip")
    action_index.to_csv(tables / "transition_action_cache_index.csv", index=False)
    np.savez_compressed(tables / "timestamped_window_matrices.npz", ordinary=ordinary.astype(np.uint16),
                        tensor=tensor.astype(np.float32), ordinary_exact30=ordinary30.astype(np.uint8),
                        tensor_exact30=tensor30.astype(np.float32))
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
    plan = confirmatory_plan(); plan.to_csv(tables / "confirmatory_analysis_plan_WITHHELD.csv", index=False)
    lock = {
        "confirmatory_statistics_run": bool(args.run_confirmatory),
        "locked_plan_sha256": sha256(tables / "confirmatory_analysis_plan_WITHHELD.csv"),
        "unlock_condition": "supply both --run-confirmatory and a non-empty --preregistration-id",
        "planned_test_count": len(plan), "route_level_test_count": 0,
    }
    (output / "PREREGISTRATION_LOCK.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    if args.run_confirmatory:
        run_confirmatory(metrics, args.preregistration_id.strip()).to_csv(tables / "confirmatory_results.csv", index=False)

    plot_manifold(scores, figures)
    plot_matrix_trajectories(metrics, figures)
    plot_baseline_precision(precision, figures)
    plot_autonomic_state_trajectories(metrics, figures)
    repo = Path(__file__).resolve().parent
    scripts = [Path(__file__).resolve(), repo / "recover_timestamped_transition_backfits.m",
               repo / "run_timestamped_transition_pipeline.ps1",
               repo / "external/VG-microstates/fit_microstate_hierarchical_dataset.m",
               repo / "external/VG-microstates/_internal/matlab/backfit_microstate_timecourse.m"]
    audit = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input_resolution": "genuine ordered microstate-change samples at 1/256 s with heartbeat-resolved autonomic interpolation",
        "transition_timestamps_fabricated": False,
        "autonomic_extrapolation": False,
        "action_tensor": "fractional soft CPI/CSI gradient-quadrant weights assigned at each genuine transition timestamp",
        "gradient_definition": "centred finite-difference slopes on heartbeat times, interpolated without extrapolation; soft signs scaled by record median absolute slope",
        "autonomic_state_description": "CPI/CSI levels, soft level quadrants and peri-ictal level trajectories retained descriptively",
        "canonical_state_order": list(STATES), "action_order": list(ACTIONS),
        "minimum_transitions": 30, "exact_30_sensitivity": True,
        "route_level_hypothesis_tests": 0,
        "confirmatory_statistics_run": bool(args.run_confirmatory),
        "preregistration_id": args.preregistration_id.strip() if args.run_confirmatory else None,
        "n_transition_records": len(index), "n_exact_transitions": int(action_index.n_transitions.sum()),
        "n_transitions_with_interpolated_action": int(action_index.n_with_interpolated_action.sum()),
        "interpolated_action_fraction": float(action_index.n_with_interpolated_action.sum() / action_index.n_transitions.sum()),
        "n_windows": len(metadata), "n_metric_windows": len(metrics),
        "scripts": {str(path): sha256(path) for path in scripts},
        "inputs": {str(path): sha256(path) for path in
                   (args.transition_index, recovery_audit_path, args.autonomic_native, args.manifest)},
    }
    (output / "ANALYSIS_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
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
        "- `PREREGISTRATION_LOCK.json`: execution status and the hash of the locked test plan.\n\n"
        + ("Confirmatory tests were run under preregistration `" + args.preregistration_id + "`.\n" if args.run_confirmatory else
           "**Confirmatory FS–ES, FS hypo–hyper, and paired within-DD tests are intentionally withheld pending preregistration.**\n"),
        encoding="utf-8")
    if args.review_dir:
        update_review(output, args.review_dir, scripts)
    print(json.dumps({"n_records": len(index), "n_windows": len(metadata),
                      "confirmatory_statistics_run": bool(args.run_confirmatory)}, indent=2))


if __name__ == "__main__":
    main()
