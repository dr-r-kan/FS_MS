"""Per-recording sentinels: fingerprints, output validation and skip decisions.

``<heppy-out>/.sentinels/<base>.json`` holds, for each of the three steps (``ecg_fit``,
``preprocess``, ``recovery``), the step fingerprint, status, QC fragment and output manifest.  A
step is skipped only when (a) its recorded fingerprint equals the one recomputed from the inputs
*and* (b) every recorded output still validates.  Failed steps with an unchanged fingerprint are
not retried (deterministic failures stay recorded) unless ``retry_failed`` is set.

EDF identity: ``size`` + ``sha256`` of the clip's content.  To avoid re-hashing large clips on
every run the hash is reused when ``size`` and ``mtime_ns`` equal the values stored in the
sentinel; otherwise the file is re-hashed (so a content change that preserves size *and* mtime is
the one undetectable case; ``--verify-hash`` forces re-hashing).
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fsms.atomic import atomic_write_json, fingerprint, sha256_file
from fsms.heppy import options as opt

SENTINEL_SCHEMA = "heppy_sentinel/1"
ECG_PACKAGES = ("python", "mne", "neurokit2", "numpy", "scipy", "pandas")


def sentinel_path(heppy_out: Path, base: str) -> Path:
    return Path(heppy_out) / ".sentinels" / f"{base}.json"


def read_sentinel(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("schema") == SENTINEL_SCHEMA else None


def write_sentinel(path: Path, data: dict) -> None:
    atomic_write_json(path, data)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def edf_identity(path: Path, previous: dict | None = None, verify_hash: bool = False) -> dict:
    stat = Path(path).stat()
    if (previous and not verify_hash and previous.get("size") == stat.st_size
            and previous.get("mtime_ns") == stat.st_mtime_ns and previous.get("sha256")):
        return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": previous["sha256"]}
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)}


def keyfile_fingerprint(rows: list[dict]) -> str:
    """Fingerprint of the snip-keyfile rows HEPPy reads (segment bounds drive PyPREP chunking)."""
    relevant = sorted(
        (str(r.get("output_edf", "")), str(r.get("output_start_seconds", "")), str(r.get("output_end_seconds", "")))
        for r in rows)
    return fingerprint(relevant)


def step_fingerprints(edf_id: dict, keyfile_fp: str, options: dict, versions: dict[str, str]) -> dict[str, str]:
    """The three step fingerprints and the overall sentinel fingerprint."""
    eff = opt.processing_options(options)
    ecg_only = ("ecg_settings", "auto_qrs", "rpeak_recovery")
    pre_opts = {k: v for k, v in eff.items() if k not in ecg_only}
    edf = {"size": edf_id["size"], "sha256": edf_id["sha256"]}
    ecg_versions = {k: versions.get(k) for k in ECG_PACKAGES}
    fp_a = fingerprint({
        "step": "ecg_fit", "edf": edf, "channel": eff["ecg_channel"],
        "hp": eff["filters"]["hp"], "lp": eff["filters"]["lp"],
        "ecg_settings": eff["ecg_settings"], "auto_qrs": eff["auto_qrs"],
        "code": opt.code_hash(opt.STEP_CODE_FILES["ecg_fit"]), "versions": ecg_versions,
    })
    fp_b = fingerprint({
        "step": "preprocess", "edf": edf, "keyfile_rows": keyfile_fp, "options": pre_opts,
        "seeds": eff["seeds"],
        "code": opt.code_hash(opt.STEP_CODE_FILES["preprocess"]), "versions": versions,
    })
    fp_c = fingerprint({
        "step": "recovery", "fp_ecg_fit": fp_a, "fp_preprocess": fp_b,
        "channel": eff["ecg_channel"], "hp": eff["filters"]["hp"], "lp": eff["filters"]["lp"],
        "ecg_settings": eff["ecg_settings"], "rpeak_recovery": eff["rpeak_recovery"],
        "code": opt.code_hash(opt.STEP_CODE_FILES["recovery"]), "versions": ecg_versions,
    })
    return {"ecg_fit": fp_a, "preprocess": fp_b, "recovery": fp_c,
            "overall": fingerprint({"a": fp_a, "b": fp_b, "c": fp_c})}


# ------------------------------------------------------------------ output validation

def validate_fif(path: Path) -> str | None:
    """None when the FIF header loads without preloading and has EEG channels; else a problem."""
    try:
        import mne

        raw = mne.io.read_raw_fif(str(path), preload=False, verbose="ERROR")
        try:
            n_eeg = len(mne.pick_types(raw.info, eeg=True))
            n_times = int(raw.n_times)
        finally:
            raw.close()
    except Exception as exc:  # unreadable/truncated
        return f"unreadable FIF ({type(exc).__name__}: {str(exc)[:120]})"
    if n_eeg < 1:
        return "FIF has no EEG channels"
    if n_times < 1:
        return "FIF has no samples"
    return None


def validate_sidecar(path: Path, min_rows: int = 0) -> str | None:
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "rpeak_time_s" not in reader.fieldnames:
                return "sidecar lacks rpeak_time_s column"
            n = 0
            for row in reader:
                if not math.isfinite(float(row["rpeak_time_s"])):
                    return "non-finite R-peak time"
                n += 1
    except (OSError, ValueError, KeyError) as exc:
        return f"unreadable sidecar ({type(exc).__name__})"
    return None if n >= min_rows else f"sidecar has {n} < {min_rows} rows"


def validate_output(item: dict, heppy_out: Path) -> str | None:
    """Validate one recorded output item ``{role, path, size, kind}``; None when fine."""
    path = Path(item["path"])
    if not path.is_absolute():
        path = Path(heppy_out) / path
    try:
        size = path.stat().st_size
    except OSError:
        return f"{item['role']}: missing"
    if item.get("size") is not None and size != item["size"]:
        return f"{item['role']}: size {size} != recorded {item['size']}"
    kind = item.get("kind", "file")
    if kind == "fif":
        problem = validate_fif(path)
    elif kind == "sidecar":
        problem = validate_sidecar(path)
    else:
        problem = None if size > 0 else "empty file"
    return f"{item['role']}: {problem}" if problem else None


def output_item(role: str, path: Path, heppy_out: Path, kind: str = "file") -> dict:
    path = Path(path)
    try:
        rel = path.relative_to(heppy_out).as_posix()
    except ValueError:
        rel = str(path)
    return {"role": role, "path": rel, "size": path.stat().st_size, "kind": kind}


def step_is_current(sentinel: dict | None, step: str, fp: str, heppy_out: Path,
                    retry_failed: bool = False) -> tuple[bool, str]:
    """Decide whether ``step`` can be skipped; returns (skip, reason)."""
    entry = ((sentinel or {}).get("steps") or {}).get(step)
    if not entry:
        return False, "no sentinel entry"
    if entry.get("fp") != fp:
        return False, "fingerprint changed"
    if entry.get("status") == "failed":
        return (not retry_failed), ("failed with unchanged fingerprint (use --retry-failed)" if not retry_failed
                                    else "retrying failed step")
    problems = [p for p in (validate_output(item, heppy_out) for item in entry.get("outputs", [])) if p]
    if problems:
        return False, "outputs invalid: " + "; ".join(problems)
    return True, "current"


def plan_steps(sentinel: dict | None, fps: dict[str, str], heppy_out: Path,
               retry_failed: bool = False, force: bool = False) -> tuple[list[str], dict[str, str]]:
    """Steps to run and the reason for each.  ``recovery`` always follows a recomputed inputs."""
    todo: list[str] = []
    reasons: dict[str, str] = {}
    for step in opt.STEPS:
        if force:
            current, why = False, "forced"
        else:
            current, why = step_is_current(sentinel, step, fps[step], heppy_out, retry_failed)
        if current and step == "recovery" and any(s in todo for s in ("ecg_fit", "preprocess")):
            current, why = False, "upstream step recomputed"
        if not current:
            todo.append(step)
        reasons[step] = why
    return todo, reasons


def fresh_sentinel(base: str, edf_id: dict | None, keyfile_fp: str, fps: dict[str, str],
                   options: dict, versions: dict[str, str], previous: dict | None) -> dict:
    """Sentinel skeleton for a (re)run: keep only the steps whose fingerprint still matches."""
    kept = {}
    if previous and not previous.get("adopted"):
        for step, entry in (previous.get("steps") or {}).items():
            if entry.get("fp") == fps.get(step):
                kept[step] = entry
    eff = opt.processing_options(options)
    return {
        "schema": SENTINEL_SCHEMA,
        "base": base,
        "adopted": False,
        "edf": edf_id,
        "keyfile_rows_fingerprint": keyfile_fp,
        "options_fingerprint": opt.options_fingerprint(options),
        "seeds": eff["seeds"],
        "versions": versions,
        "step_fingerprints": {k: v for k, v in fps.items() if k != "overall"},
        "fingerprint": fps["overall"],
        "steps": kept,
    }
