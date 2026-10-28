"""Per-recording QC records and the pre-specified, group-blind exclusion rule (v2).

``assemble_qc`` derives ``heppy_qc/<base>.json`` from the sentinel (single source of truth) and is a
pure function of it, so re-deriving never changes bytes.

The rule has two scopes, both applied from QC evidence only (there is no argument through which a
diagnostic group, seizure type or outcome could enter):

* **recording** (``evaluate_rule``): too few EEG channels after the montage, no ECG channel, no valid
  60-s ECG window, unhandled exception, worker crash, missing outputs;
* **segment** (``evaluate_segments``): only for in-scope segments (II and preictal).  A segment is
  excluded when ANY PyPREP chunk overlapping its time range was skipped/failed or interpolated more
  than the allowed fraction of EEG channels.  Chunks that overlap only out-of-scope segments (ictal,
  postictal, transictal) can never exclude anything.
"""
from __future__ import annotations

from typing import Any

QC_SCHEMA = "heppy_qc/1"
ADOPTED_UNKNOWN = "unknown_adopted"


def _step(sentinel: dict, name: str) -> dict:
    return ((sentinel.get("steps") or {}).get(name)) or {}


def assemble_qc(base: str, sentinel: dict) -> dict:
    """Merge a sentinel's step fragments into the recording's QC record."""
    if sentinel.get("adopted"):
        return _assemble_adopted(base, sentinel)
    a, b, c = _step(sentinel, "ecg_fit"), _step(sentinel, "preprocess"), _step(sentinel, "recovery")
    qa, qb, qc = a.get("qc", {}), b.get("qc", {}), c.get("qc", {})
    errors = []
    for name, entry in (("ecg_fit", a), ("preprocess", b), ("recovery", c)):
        if entry.get("status") == "failed":
            err = entry.get("error") or {}
            errors.append({"step": name, "kind": err.get("kind", "exception"),
                           "type": err.get("type", ""), "message": err.get("message", "")})
    if sentinel.get("crash"):
        crash = sentinel["crash"]
        errors.append({"step": crash.get("last_stage", "worker"), "kind": "crash", "type": "worker_crash",
                       "message": f"exit={crash.get('exit_code')} last_stage={crash.get('last_stage')}"})

    ecg_found = a.get("status") not in (None, "no_ecg_channel")
    final_score = None
    final_source = None
    if c.get("status") == "ok":
        final_score = qc.get("final_score")
        final_source = "recovery"
    elif c.get("status") == "declined":
        final_score = qc.get("before_qc")
        final_source = "autofit_recovery_declined"
    elif a.get("status") == "ok":
        final_score = qa.get("rpeak_score")
        final_source = "autofit"
    valid = None if final_score is None else int(final_score.get("valid_60s_windows", 0))
    total = None if final_score is None else int(final_score.get("total_60s_windows", 0))
    ecg = {
        "found": bool(ecg_found),
        "channel_edf": qa.get("channel"),
        "channel_fif": qc.get("channel"),
        "autofit_status": a.get("status"),
        "recovery_status": c.get("status"),
        "autofit": {k: v for k, v in qa.items() if k not in ("rpeak_score",)},
        "recovery": {k: v for k, v in qc.items() if k not in ("final_score",)},
        "final_source": final_source,
        "final_peak_count": None if final_score is None else int(final_score.get("peak_count", 0)),
        "valid_60s_windows": valid,
        "total_60s_windows": total,
    }
    pre_ok = b.get("status") == "ok"
    outputs_ok = pre_ok and bool(b.get("outputs"))
    status = "failed" if errors else ("ok" if pre_ok else "incomplete")
    return {
        "schema": QC_SCHEMA,
        "base": base,
        "adopted": False,
        "status": status,
        "errors": errors,
        "edf": {"size": (sentinel.get("edf") or {}).get("size"), "sha256": (sentinel.get("edf") or {}).get("sha256")},
        "sfreq": qb.get("sfreq") or qa.get("sfreq"),
        "decim": (qb.get("ica") or {}).get("decim"),
        "n_eeg_input": qb.get("n_eeg_input"),
        "n_eeg_after_montage": qb.get("n_eeg_after_montage"),
        "n_eeg_output": qb.get("n_eeg_output"),
        "pyprep": qb.get("pyprep") or {"status": "not_run", "n_chunks": 0},
        "ica": qb.get("ica") or {},
        "warnings": qb.get("warnings", []),
        "ecg": ecg,
        "outputs_ok": bool(outputs_ok),
        "outputs": {item["role"]: item["path"] for entry in (a, b, c) for item in entry.get("outputs", [])},
        "seeds": sentinel.get("seeds"),
        "versions": sentinel.get("versions"),
        "options_fingerprint": sentinel.get("options_fingerprint"),
        "fingerprint": sentinel.get("fingerprint"),
        "step_fingerprints": sentinel.get("step_fingerprints"),
    }


def _assemble_adopted(base: str, sentinel: dict) -> dict:
    adoption = sentinel.get("adoption") or {}
    qc = dict(adoption.get("qc") or {})
    ecg = qc.get("ecg") or {}
    return {
        "schema": QC_SCHEMA,
        "base": base,
        "adopted": True,
        "status": "ok",
        "errors": [],
        "edf": sentinel.get("edf"),
        "sfreq": qc.get("sfreq"),
        "decim": None,
        "n_eeg_input": None,
        "n_eeg_after_montage": qc.get("n_eeg_output"),
        "n_eeg_output": qc.get("n_eeg_output"),
        "pyprep": qc.get("pyprep") or {"status": ADOPTED_UNKNOWN, "n_chunks": 0},
        "ica": qc.get("ica") or {},
        "warnings": [],
        "ecg": {
            "found": bool(ecg.get("found")),
            "channel_edf": None,
            "channel_fif": ecg.get("channel_fif"),
            "final_source": "legacy_sidecar",
            "final_peak_count": ecg.get("final_peak_count"),
            "valid_60s_windows": ecg.get("valid_60s_windows"),
            "total_60s_windows": ecg.get("total_60s_windows"),
        },
        "outputs_ok": True,
        "outputs": {role: info.get("path") for role, info in (adoption.get("files") or {}).items()},
        "seeds": sentinel.get("seeds"),
        "versions": sentinel.get("versions"),
        "options_fingerprint": sentinel.get("options_fingerprint"),
        "fingerprint": sentinel.get("fingerprint"),
        "step_fingerprints": None,
    }


def _rec(rule: dict) -> dict:
    return rule.get("recording") or {}


def _seg(rule: dict) -> dict:
    return rule.get("segment") or {}


def rule_problems(rule: dict) -> list[str]:
    """Structural check of ``qc_rule`` (v2); empty list = usable."""
    problems = []
    if not isinstance(rule.get("recording"), dict) or not isinstance(rule.get("segment"), dict):
        return ["qc_rule needs 'recording' and 'segment' blocks (rule version 2)"]
    for key in ("min_eeg_channels_after_montage", "require_ecg_channel", "min_valid_ecg_60s_windows",
                "exclude_on_unhandled_exception"):
        if key not in rule["recording"]:
            problems.append(f"qc_rule.recording.{key} missing")
    for key in ("scope_segments", "pyprep_exclude_statuses", "max_interpolated_fraction"):
        if key not in rule["segment"]:
            problems.append(f"qc_rule.segment.{key} missing")
    return problems


def ecg_viable(qc: dict, rule: dict) -> tuple[str, str]:
    """('yes'|'no', reason) for the legacy ``viable`` column, by rule rather than blanket 'yes'."""
    ecg = qc.get("ecg") or {}
    if not ecg.get("found"):
        return "no", "no ECG channel found in clip"
    valid = ecg.get("valid_60s_windows")
    minimum = int(_rec(rule).get("min_valid_ecg_60s_windows", 1))
    if valid is None:
        return "no", "no R-peak sidecar produced"
    if valid < minimum:
        return "no", f"valid_60s_windows={valid} < {minimum}"
    return "yes", f"valid_60s_windows={valid}/{ecg.get('total_60s_windows')}; peaks={ecg.get('final_peak_count')}"


def _sort_reasons(reasons: list[dict], rule: dict) -> list[dict]:
    order = {code: i for i, code in enumerate(rule.get("reason_precedence", []))}
    return sorted(reasons, key=lambda r: (order.get(r["reason_code"], len(order)), r["reason_code"], r["stage"]))


def evaluate_rule(qc: dict, rule: dict) -> dict:
    """Recording-level rule.  Returns reasons (precedence-sorted), flags and evidence gaps.

    Missing evidence (``None``) is *not* an exclusion: it is listed under ``not_evaluable`` so
    adopted legacy recordings, for which some evidence cannot exist, are reported honestly.
    PyPREP outcomes are NOT recording-level criteria (see :func:`evaluate_segments`); a RANSAC
    fallback and the ICA flatten-revert are only flagged.
    """
    rec = _rec(rule)
    reasons: list[dict] = []
    not_evaluable: list[str] = []
    flags: list[str] = []

    def add(code: str, stage: str, evidence: str) -> None:
        reasons.append({"reason_code": code, "stage": stage, "evidence": evidence})

    for err in qc.get("errors", []):
        if err.get("kind") == "crash":
            add("worker_crash", err.get("step") or "worker", err.get("message", ""))
        elif rec.get("exclude_on_unhandled_exception", True):
            add("unhandled_exception", err.get("step", ""), f"{err.get('type', '')}: {str(err.get('message', ''))[:300]}")
    failed = bool(qc.get("errors"))

    if not failed and not qc.get("outputs_ok", False):
        add("missing_outputs", "preprocess", "preprocessing produced no validated FIF outputs")

    n_eeg = qc.get("n_eeg_after_montage")
    minimum = rec.get("min_eeg_channels_after_montage")
    if minimum is not None:
        if n_eeg is None:
            if not failed:
                not_evaluable.append("n_eeg_after_montage")
        elif n_eeg < int(minimum):
            add("too_few_eeg_channels", "montage", f"n_eeg_after_montage={n_eeg} < {minimum}")

    ecg = qc.get("ecg") or {}
    if rec.get("require_ecg_channel", True) and not ecg.get("found"):
        if not (failed and not ecg):  # a crash before any ECG evidence is already covered above
            add("no_ecg_channel", "ecg", "no ECG-like channel (ECG/EKG/CARD/EXG) in clip")
    elif ecg.get("found"):
        valid = ecg.get("valid_60s_windows")
        need = int(rec.get("min_valid_ecg_60s_windows", 0))
        if need:
            if valid is None:
                if not failed:
                    not_evaluable.append("valid_ecg_windows")
            elif valid < need:
                add("ecg_no_valid_window", "ecg", f"valid_60s_windows={valid} < {need}")

    status = (qc.get("pyprep") or {}).get("status")
    if status in set(_seg(rule).get("pyprep_flag_only_statuses", [])):
        flags.append(f"pyprep_{status}")
    if (qc.get("ica") or {}).get("flatten_reverted_any"):
        flags.append("ica_flatten_reverted")

    reasons = _sort_reasons(reasons, rule)
    return {
        "excluded": bool(reasons),
        "reasons": reasons,
        "primary_reason": reasons[0]["reason_code"] if reasons else None,
        "flags": sorted(set(flags)),
        "not_evaluable": sorted(set(not_evaluable)),
    }


def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return min(a1, b1) - max(a0, b0) > 1e-6


def evaluate_segments(qc: dict, rule: dict, segments: list[dict]) -> dict:
    """Segment-level PyPREP rule for the in-scope rows of one clip's snip keyfile.

    ``segments`` are keyfile rows (``segment``, ``output_start_seconds``, ``output_end_seconds`` plus
    identifiers, passed through).  Only segments named in ``rule.segment.scope_segments`` are
    evaluated; chunks lacking time bounds are conservatively taken to cover the whole recording.
    Returns ``{'results': [{'segment': row, 'reasons': [...], 'usable': bool}], 'not_evaluable': [...]}``.
    """
    seg = _seg(rule)
    scope = set(seg.get("scope_segments", []))
    exclude_status = set(seg.get("pyprep_exclude_statuses", []))
    limit = seg.get("max_interpolated_fraction")
    pyprep = qc.get("pyprep") or {}
    status = pyprep.get("status")
    chunks = pyprep.get("chunks") or []
    not_evaluable: list[str] = []
    if status == ADOPTED_UNKNOWN:
        not_evaluable.append("segment_pyprep_status")
    elif not chunks and status in exclude_status:
        not_evaluable.append("segment_pyprep_attribution")   # degraded somewhere, chunk bounds unknown
    if chunks and any(c.get("interpolated_fraction") is None and c.get("status") in ("ok", "ransac_fallback")
                      for c in chunks):
        not_evaluable.append("segment_interpolated_fraction")
    if not chunks and status == "ok" and qc.get("adopted"):
        not_evaluable.append("segment_interpolated_fraction")

    results = []
    for row in segments:
        name = str(row.get("segment", "")).strip()
        if name not in scope:
            continue
        try:
            start, end = float(row["output_start_seconds"]), float(row["output_end_seconds"])
        except (KeyError, ValueError):
            start, end = float("-inf"), float("inf")
        reasons: list[dict] = []
        for chunk in chunks:
            c0, c1 = chunk.get("start_s"), chunk.get("end_s")
            if c0 is not None and c1 is not None and not _overlaps(start, end, float(c0), float(c1)):
                continue
            where = f"chunk{chunk.get('index')}" + (f"@{c0:g}-{c1:g}s" if c0 is not None else "@whole-recording")
            if chunk.get("status") in exclude_status:
                reasons.append({"reason_code": f"segment_pyprep_{chunk['status']}", "stage": "pyprep",
                                "evidence": f"{where} status={chunk['status']}; segment {name} {start:g}-{end:g}s"})
            elif limit is not None and chunk.get("interpolated_fraction") is not None \
                    and chunk["interpolated_fraction"] > float(limit):
                reasons.append({"reason_code": "segment_excess_interpolation", "stage": "pyprep",
                                "evidence": f"{where} interpolated_fraction={chunk['interpolated_fraction']:.3f} > {limit}; "
                                            f"segment {name} {start:g}-{end:g}s"})
        results.append({"segment": row, "reasons": _sort_reasons(reasons, rule), "usable": not reasons})
    return {"results": results, "not_evaluable": sorted(set(not_evaluable))}
