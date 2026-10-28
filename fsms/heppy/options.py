"""Locked HEPPy options: loading, validation, normalisation and fingerprint ingredients.

Only the *processing* options (those that can change a preprocessed FIF or an R-peak sidecar)
enter the per-recording fingerprint and the legacy-adoption comparison.  ``qc_rule`` is applied
afterwards to recorded QC evidence and therefore never invalidates computed outputs; ``epoch``,
``hrv`` and ``export`` only matter to downstream HEP analysis, which this stage does not run.
"""
from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

from fsms.atomic import fingerprint

SCHEMA_VERSION = 1
# Legacy runs hard-coded random_state 42 for PyPREP (+ segment index) and ICA; adoption of legacy
# outputs is only valid when the locked seed equals this.
LEGACY_RANDOM_STATE = 42

PACKAGES = ("mne", "pyprep", "mne-icalabel", "neurokit2", "numpy", "scipy", "pandas", "onnxruntime")

# Files whose bytes (line-ending normalised) are part of the per-step fingerprints.
# options.py is deliberately absent: processing options enter the fingerprints as *values* (after
# normalisation), so a change in how they are validated or how ``qc_rule`` is read must not recompute outputs.
STEP_CODE_FILES = {
    "ecg_fit": ("fsms/heppy/ecg_qrs.py", "fsms/heppy/worker.py", "external/HEPPy/ecg_peak_correction.py"),
    "preprocess": ("fsms/heppy/preprocess.py", "fsms/heppy/worker.py", "external/HEPPy/_preprocessing.py"),
    "recovery": ("fsms/heppy/ecg_qrs.py", "fsms/heppy/worker.py", "external/HEPPy/ecg_peak_correction.py"),
}
STEPS = ("ecg_fit", "preprocess", "recovery")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def heppy_dir() -> Path:
    return repo_root() / "external" / "HEPPy"


def ensure_heppy_on_path() -> Path:
    path = heppy_dir()
    if not (path / "_preprocessing.py").exists():
        raise FileNotFoundError(f"HEPPy not found at {path}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str("" if value is None else value).strip().lower() in {"true", "1", "yes", "y", "on"}


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _optional_positive_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    text = _clean(value)
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if number > 0 else None


def asr_cutoff(options: dict) -> float | None:
    """Same semantics as the legacy ``configured_asr_cutoff``."""
    value = options.get("asr", False)
    numeric = _optional_positive_float(value)
    if numeric is not None:
        return numeric
    if truthy(value):
        return _optional_positive_float(options.get("asr_cutoff")) or 20.0
    return None


def cfa_mode(options: dict) -> str:
    return _clean(options.get("remove_cfa_mode") or options.get("cfa_mode") or "remove").lower()


def load_options(path: str | Path) -> dict:
    """Read an options JSON (BOM/whitespace/key-order tolerant)."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def processing_options(options: dict) -> dict:
    """The fully-defaulted options that determine FIF/sidecar content (legacy defaults applied).

    Defaults mirror ``OLD/run_heppy_preprocessing.runtime_config`` and ``OLD/fast_ecg_review``
    (so a legacy ``options_used.json`` with absent keys normalises to the same dict as a locked
    file that spells them out).  Paths and key order never matter.
    """
    filters = options.get("filters") or {}
    montage = options.get("montage") or {}
    tuning = options.get("preprocessing_tuning") or {}
    rr = options.get("adaptive_rr") or {}
    auto = options.get("auto_qrs") or {}
    rec = options.get("rpeak_recovery") or {}
    seeds = options.get("seeds") or {}
    max_chunk = tuning.get("pyprep_max_chunk_size", 8)
    try:
        max_chunk = int(max_chunk) if max_chunk not in (None, "") else None
    except (TypeError, ValueError):
        max_chunk = 8
    min_corr = float(auto.get("min_corr", 0.80))
    return {
        "filters": {
            "hp": float(filters.get("hp", 1.0)),
            "lp": float(filters.get("lp", 100.0)),
            "line_freqs": [float(f) for f in filters.get("line_freqs", (50.0, 100.0))],
        },
        "ecg_channel": _clean((options.get("ecg") or {}).get("channel")),
        "montage": {
            "name": _clean(montage.get("name") or "standard_1020"),
            "rename_to_1020": truthy(montage.get("rename_to_1020", True)),
        },
        "pyprep": truthy(options.get("pyprep", True)),
        "prep_ransac": truthy(options.get("prep_ransac", False)),
        "asr_cutoff": asr_cutoff(options),
        "ica_iclabel": truthy(options.get("ica_iclabel", True)),
        "cfa_mode": cfa_mode(options),
        "target_sfreq": _optional_positive_float(options.get("target_sfreq")),
        "spirometry_channel": _clean((options.get("spirometry") or {}).get("channel")),
        "stim_keep": [str(item) for item in (options.get("stim_keep") or [])],
        "tuning": {
            "pyprep_channel_wise": truthy(tuning.get("pyprep_channel_wise", False)),
            "pyprep_max_chunk_size": max_chunk,
            "pyprep_by_segment": truthy(tuning.get("pyprep_by_segment", False)),
            "pyprep_safe_quantile_patch": truthy(tuning.get("pyprep_safe_quantile_patch", False)),
            "ica_fit_hz": float(tuning.get("ica_fit_hz", 100.0)),
            "iclabel_backend": _clean(tuning.get("iclabel_backend") or "auto").lower(),
        },
        "ecg_settings": {
            "rr_min_bpm": float(rr.get("rr_min_bpm", 35.0)),
            "rr_max_bpm": float(rr.get("rr_max_bpm", 140.0)),
            "rr_outlier_mad": float(rr.get("rr_outlier_mad", 3.5)),
        },
        "auto_qrs": {
            "alpha": float(auto.get("alpha", 0.05)),
            "iterations": int(auto.get("iterations", 2)),
            "min_corr": min_corr,
            "corr_search_s": float(auto.get("corr_search_s", 0.35)),
        },
        "rpeak_recovery": {
            "iterations": int(rec.get("iterations", 2)),
            "max_multiple": int(rec.get("max_multiple", 4)),
            "promac_threshold": float(rec.get("promac_threshold", 0.33)),
            "min_template_corr": float(rec.get("min_template_corr", min_corr)),
        },
        "seeds": {"heppy_random_state": int(seeds.get("heppy_random_state", LEGACY_RANDOM_STATE))},
    }


def validate_options(options: dict) -> list[str]:
    """Return a list of problems (empty = acceptable for this driver)."""
    problems: list[str] = []
    eff = processing_options(options)
    if (options.get("ecg") or {}).get("flip_on_load") or truthy(options.get("flip_flags", False)):
        problems.append("ecg.flip_on_load/flip_flags must be false: this driver has no manual review step")
    if eff["cfa_mode"] != "both":
        problems.append(f"cfa_mode must be 'both' (got {eff['cfa_mode']!r}): the manifest uses both FIF variants")
    mains = options.get("mains_hz")
    if mains is not None:
        expected = [float(mains), 2.0 * float(mains)]
        if eff["filters"]["line_freqs"] != expected:
            problems.append(f"filters.line_freqs {eff['filters']['line_freqs']} inconsistent with mains_hz={mains}")
    rule = options.get("qc_rule")
    if not isinstance(rule, dict) or "rule_id" not in rule:
        problems.append("qc_rule (pre-specified exclusion rule) is missing")
    else:
        from fsms.heppy.qc import rule_problems

        problems.extend(rule_problems(rule))
    if "seeds" not in options or "heppy_random_state" not in (options.get("seeds") or {}):
        problems.append("seeds.heppy_random_state must be explicit")
    return problems


def options_fingerprint(options: dict) -> str:
    return fingerprint(processing_options(options))


def qc_rule_fingerprint(options: dict) -> str:
    return fingerprint(options.get("qc_rule") or {})


def diff_processing_options(a: dict, b: dict) -> list[str]:
    """Dotted names of the processing-option leaves that differ between two *raw* option dicts."""
    left, right = processing_options(a), processing_options(b)
    out: list[str] = []

    def walk(prefix: str, x: Any, y: Any) -> None:
        if isinstance(x, dict) and isinstance(y, dict):
            for key in sorted(set(x) | set(y)):
                walk(f"{prefix}.{key}" if prefix else key, x.get(key), y.get(key))
        elif x != y:
            out.append(prefix)

    walk("", left, right)
    return out


def package_versions() -> dict[str, str]:
    versions = {"python": ".".join(str(part) for part in sys.version_info[:3])}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def code_hash(relative_paths: tuple[str, ...]) -> dict[str, str]:
    """SHA-256 per file with CRLF normalised so git autocrlf cannot change a fingerprint."""
    import hashlib

    root = repo_root()
    out = {}
    for rel in relative_paths:
        data = (root / rel).read_bytes().replace(b"\r\n", b"\n")
        out[rel] = hashlib.sha256(data).hexdigest()
    return out
