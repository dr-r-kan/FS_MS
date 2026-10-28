"""Checks of stage-C outputs against the KNOWN generating labels of a synthetic dataset (no real data)."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    """SHA-256 of every file below ``root`` (relative POSIX path -> hash)."""
    root = Path(root)
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def _hdf5_array(group, name) -> np.ndarray:
    return np.asarray(group[name])


def global_centers(hierarchy_dir: Path) -> np.ndarray:
    """K x channels fitted global maps from global/global_solution.mat (v7.3 HDF5)."""
    path = Path(hierarchy_dir) / "global" / "global_solution.mat"
    with h5py.File(path, "r") as handle:
        centers = _hdf5_array(handle["Fit"], "centers")
    if centers.shape[0] != 6 and centers.shape[1] == 6:
        centers = centers.T
    return centers.astype(float)


def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def labels_from_generating_maps(hierarchy_dir: Path, truth: dict) -> tuple[list[str], np.ndarray]:
    """Class of each fitted global state according to the GENERATING maps (independent of the template rule)."""
    centers = global_centers(hierarchy_dir)
    base = np.array(truth["base_maps"])
    if centers.shape[1] != base.shape[1]:
        raise AssertionError(f"channel count mismatch: fitted {centers.shape[1]} vs generating {base.shape[1]}")
    correlation = np.abs(_corr(centers, base))
    classes = truth["classes"]
    return [classes[i] for i in correlation.argmax(axis=1)], correlation


def load_transition(path: Path) -> dict:
    value = loadmat(path, squeeze_me=True, struct_as_record=False)["transition"]
    return {
        "peak_sample": np.atleast_1d(value.peak_sample).astype(np.int64),
        "peak_assignment": np.atleast_1d(value.peak_assignment).astype(np.int64),
        "transition_sample": np.atleast_1d(value.transition_sample).astype(np.int64),
        "from_state": np.atleast_1d(value.from_state).astype(np.int64),
        "to_state": np.atleast_1d(value.to_state).astype(np.int64),
        "sfreq": float(value.sfreq), "n_samples": int(value.n_samples),
        "fit_labels": [str(x) for x in np.atleast_1d(value.fit_labels)],
        "file_path": str(value.file_path),
    }


def peak_label_accuracy(backfit_dir: Path, truth_dir: Path, fit_labels: list[str]) -> dict:
    """Fraction of GFP peaks whose fitted label (via fit_labels) equals the generating class at that sample."""
    truth_dir = Path(truth_dir)
    with (Path(backfit_dir) / "transition_index.csv").open(newline="", encoding="utf-8-sig") as handle:
        index = list(csv.DictReader(handle))
    correct = total = 0
    per_record = {}
    classes = sorted(set(fit_labels))
    confusion = np.zeros((len(fit_labels), len(classes)), dtype=int)     # fitted state x generating class
    for row in index:
        stem = Path(row["file_path"]).stem
        with (truth_dir / f"{stem}_segments.csv").open(newline="") as handle:
            segments = list(csv.DictReader(handle))
        starts = np.array([int(seg["start_sample"]) for seg in segments])
        segment_classes = np.array([seg["class"] for seg in segments])
        transition = load_transition(Path(row["cache_file"]))
        zero_based = transition["peak_sample"] - 1
        generating = segment_classes[np.searchsorted(starts, zero_based, side="right") - 1]
        fitted = np.array([fit_labels[i - 1] for i in transition["peak_assignment"]])
        np.add.at(confusion, (transition["peak_assignment"] - 1, [classes.index(c) for c in generating]), 1)
        hit = int((generating == fitted).sum())
        per_record[stem] = hit / max(1, len(fitted))
        correct += hit
        total += len(fitted)
    modal = [classes[int(j)] for j in confusion.argmax(axis=1)]
    return {"accuracy": correct / max(1, total), "n_peaks": total, "per_record": per_record,
            "confusion": confusion, "modal_generating_class_per_state": modal}


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
