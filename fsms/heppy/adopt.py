"""Read-only adoption of legacy ``heppy_output`` trees (``--adopt-dir``).

A legacy clip is adopted *in place* (no copy; nothing is ever written under the adopt directory)
only when ALL of the following hold:

1. ``logs/headless_preprocessing_summary.csv`` has a row for the clip with ``status == "ok"``;
2. the legacy ``options_used.json`` equals the locked options after normalisation
   (:func:`fsms.heppy.options.processing_options`: key order, whitespace, BOM and paths are
   ignored; only keys that can change a FIF/sidecar count, with legacy defaults applied; the legacy
   hard-coded seed 42 is part of the comparison);
3. both ``raw_fif/<base>_pp_raw.fif`` and ``raw_fif/<base>_pp_raw_keepcfa.fif`` load their headers
   (``preload=False``) and have EEG channels;
4. ``ecg_qc/<base>_rpeaks_corrected.csv`` exists with a usable ``rpeak_time_s`` column (>= 3 peaks).

Paths recorded in the legacy summary are ignored (they usually name a drive that no longer
exists); the conventional layout under the adopt directory is used instead.  PyPREP degradation is
reconstructed from the legacy per-row log (``logs/headless_rows/*_<base>.log``, which captured the
swallowed warnings) when that log is available; otherwise the PyPREP status is ``unknown_adopted``
and the QC rule reports it as *not evaluable* instead of silently passing it.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

from fsms.atomic import sha256_file
from fsms.heppy import options as opt
from fsms.heppy import sentinel as sn
from fsms.heppy.qc import ADOPTED_UNKNOWN


def _read_csv(path: Path) -> list[dict]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def chunk_events_from_log(text: str) -> dict | None:
    """Reconstruct per-chunk PyPREP outcomes from a legacy per-row log (stdout+stderr of the worker).

    Each chunk's fit announces itself with ``HEPPY_STAGE pyprep_fit`` and the swallowed outcome is
    logged right after, so events are attributed to the latest fit announced.  Returns None when
    the log is ambiguous (RANSAC fits announce twice, montage skips announce nothing).
    """
    import re

    chunk = -1
    events: dict[int, str] = {}
    fits = 0
    ambiguous = False
    for line in text.splitlines():
        if line.startswith("HEPPY_STAGE\tpyprep_fit"):
            fits += 1
            chunk += 1
            if ":r=1:" in line:
                ambiguous = True
        elif "PyPREP skipped; too many bad EEG channels" in line:
            events[chunk] = "skipped_toomanybad"
        elif "PyPREP skipped; no EEG channels have usable montage positions" in line:
            ambiguous = True
        elif "PyPREP RANSAC skipped after failure" in line:
            ambiguous = True
        elif "PyPREP failed on segment" in line:
            match = re.search(r"segment (\d+)/(\d+)", line)
            if match:
                events[int(match.group(1)) - 1] = "failed"
            else:
                ambiguous = True
    return None if ambiguous else {"fits": fits, "events": {str(k): v for k, v in events.items()}}


def chunks_from_events(info: dict | None, keyfile_rows: list[dict]) -> list[dict] | None:
    """Chunk records (status + clip-time bounds) when the log's chunk count equals the keyfile's."""
    if not info:
        return None
    bounds = []
    for row in keyfile_rows:
        try:
            bounds.append((float(row["output_start_seconds"]), float(row["output_end_seconds"])))
        except (KeyError, ValueError):
            return None
    bounds.sort()
    if not bounds or info["fits"] != len(bounds):
        return None
    chunks = []
    for index, (start, end) in enumerate(bounds):
        status = info["events"].get(str(index), "ok")
        chunks.append({"index": index, "status": status, "start_s": start, "end_s": end,
                       "interpolated_fraction": None, "n_eeg": None, "error": None})
    return chunks


class AdoptionContext:
    def __init__(self, adopt_dir: Path, locked_options: dict, hash_outputs: bool = False) -> None:
        self.dir = Path(adopt_dir)
        self.hash_outputs = hash_outputs
        self.summary = {str(r.get("base", "")).strip(): r for r in
                        _read_csv(self.dir / "logs" / "headless_preprocessing_summary.csv")}
        self.tsv_rows = self._read_tsv()
        self.options_path = self.dir / "options_used.json"
        self.compat = self._compat(locked_options)

    def _read_tsv(self) -> list[dict]:
        path = self.dir / "logs" / "preprocessing.tsv"
        try:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                return list(csv.DictReader(handle, delimiter="\t"))
        except OSError:
            return []

    def _compat(self, locked: dict) -> dict:
        if not self.options_path.exists():
            return {"compatible": False, "reason": "legacy options_used.json missing", "differing_keys": []}
        try:
            legacy = opt.load_options(self.options_path)
        except ValueError as exc:
            return {"compatible": False, "reason": f"legacy options_used.json unreadable: {exc}", "differing_keys": []}
        diff = opt.diff_processing_options(locked, legacy)
        return {
            "compatible": not diff,
            "reason": "processing options identical after normalisation" if not diff
            else "processing options differ: " + ", ".join(diff),
            "differing_keys": diff,
            "legacy_options_fingerprint": opt.options_fingerprint(legacy),
            "locked_options_fingerprint": opt.options_fingerprint(locked),
        }

    # ---------------------------------------------------------------- per clip
    def paths(self, base: str) -> dict[str, Path]:
        return {
            "raw_fif": self.dir / "raw_fif" / f"{base}_pp_raw.fif",
            "raw_fif_keep": self.dir / "raw_fif" / f"{base}_pp_raw_keepcfa.fif",
            "sidecar": self.dir / "ecg_qc" / f"{base}_rpeaks_corrected.csv",
        }

    def try_adopt(self, base: str, ecg_settings: dict, keyfile_rows: list[dict] | None = None) -> tuple[dict | None, str]:
        """Return (adoption record, "") or (None, reason it cannot be adopted)."""
        if not self.compat["compatible"]:
            return None, f"options not compatible: {self.compat['reason']}"
        row = self.summary.get(base)
        if row is None:
            return None, "no legacy summary row"
        if str(row.get("status", "")).strip() != "ok":
            return None, f"legacy summary status is {row.get('status')!r}, not 'ok'"
        files: dict[str, dict] = {}
        for role, path in self.paths(base).items():
            try:
                stat = path.stat()
            except OSError:
                return None, f"{role} missing: {path.name}"
            info = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if self.hash_outputs:
                info["sha256"] = sha256_file(path)
            files[role] = info
        for role in ("raw_fif", "raw_fif_keep"):
            problem = sn.validate_fif(Path(files[role]["path"]))
            if problem:
                return None, f"{role}: {problem}"
        problem = sn.validate_sidecar(Path(files["sidecar"]["path"]), min_rows=3)
        if problem:
            return None, f"sidecar: {problem}"
        record = {
            "adopt_dir": str(self.dir),
            "legacy_status": "ok",
            "options_compat": self.compat,
            "files": files,
            "legacy_preprocessing_tsv": self._tsv_evidence(base),
            "legacy_row_log": self._log_evidence(base),
        }
        record["qc"] = self._adopted_qc(base, files, record, ecg_settings, keyfile_rows or [])
        return record, ""

    def revalidate(self, record: dict) -> str | None:
        """Re-check a recorded adoption (sizes, mtimes, headers); None when still valid."""
        for role, info in (record.get("files") or {}).items():
            path = Path(info["path"])
            try:
                stat = path.stat()
            except OSError:
                return f"{role} missing"
            if stat.st_size != info.get("size") or stat.st_mtime_ns != info.get("mtime_ns"):
                return f"{role} changed since adoption"
        for role in ("raw_fif", "raw_fif_keep"):
            problem = sn.validate_fif(Path(record["files"][role]["path"]))
            if problem:
                return f"{role}: {problem}"
        return None

    # ---------------------------------------------------------------- evidence
    def _tsv_evidence(self, base: str) -> dict | None:
        out = {}
        for row in self.tsv_rows:
            target = str(row.get("output", ""))
            if str(row.get("status", "")).upper() != "OK" or row.get("stage") != "save":
                continue
            name = Path(target.replace("\\", "/")).name
            if name == f"{base}_pp_raw.fif":
                out["remove"] = {"ica_bad_labels": row.get("ica_bad_labels", ""), "ica_bads": row.get("ica_bads", ""),
                                 "n_events": row.get("n_events", ""), "n_channels": row.get("n_channels", "")}
            elif name == f"{base}_pp_raw_keepcfa.fif":
                out["keep"] = {"ica_bad_labels": row.get("ica_bad_labels", ""), "ica_bads": row.get("ica_bads", ""),
                               "n_events": row.get("n_events", ""), "n_channels": row.get("n_channels", "")}
        return out or None

    def _log_evidence(self, base: str) -> dict | None:
        logs = sorted((self.dir / "logs" / "headless_rows").glob(f"*_{base}.log"))
        if not logs:
            return None
        try:
            text = logs[-1].read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        return {
            "log": logs[-1].name,
            "chunk_events": chunk_events_from_log(text),
            "too_many_bad": text.count("PyPREP skipped; too many bad EEG channels"),
            "no_montage": text.count("PyPREP skipped; no EEG channels have usable montage positions"),
            "ransac_fallback": text.count("PyPREP RANSAC skipped after failure"),
            "segment_failed": text.count("PyPREP failed on segment"),
            "pyprep_fit_stages": text.count("HEPPY_STAGE\tpyprep_fit"),
            "ica_flatten_reverted": text.count("ICA application flattened EEG"),
        }

    def _adopted_qc(self, base: str, files: dict, record: dict, ecg_settings: dict, keyfile_rows: list[dict]) -> dict:
        import mne
        import numpy as np

        from fsms.heppy import ecg_qrs

        raw = mne.io.read_raw_fif(files["raw_fif"]["path"], preload=False, verbose="ERROR")
        try:
            eeg = mne.pick_types(raw.info, eeg=True)
            ecg = mne.pick_types(raw.info, ecg=True)
            sfreq = float(raw.info["sfreq"])
            n_times = int(raw.n_times)
            eeg_names = [raw.ch_names[i] for i in eeg]
            ecg_names = [raw.ch_names[i] for i in ecg]
        finally:
            raw.close()
        times = []
        with Path(files["sidecar"]["path"]).open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                value = float(row["rpeak_time_s"])
                if math.isfinite(value):
                    times.append(value)
        peaks = np.unique(np.rint(np.asarray(times) * sfreq).astype(int))
        peaks = peaks[(peaks > 0) & (peaks < n_times)]
        score = ecg_qrs.rpeak_qc_score(peaks, sfreq, n_times, ecg_settings)
        evidence = record.get("legacy_row_log")
        pyprep: dict = {"status": ADOPTED_UNKNOWN, "n_chunks": 0, "evidence": "no legacy row log"}
        if evidence is not None:
            if evidence["too_many_bad"]:
                status = "skipped_toomanybad"
            elif evidence["no_montage"]:
                status = "skipped_no_montage"
            elif evidence["segment_failed"]:
                status = "failed"
            elif evidence["ransac_fallback"]:
                status = "ransac_fallback"
            else:
                status = "ok"
            pyprep = {"status": status, "n_chunks": evidence["pyprep_fit_stages"],
                      "evidence": f"legacy row log {evidence['log']}",
                      "interpolated_fraction_max": None}
            chunks = chunks_from_events(evidence["chunk_events"], keyfile_rows)
            if chunks is not None:   # per-chunk attribution is only claimed when the log is unambiguous
                pyprep["chunks"] = chunks
                pyprep["chunks_have_bounds"] = True
        ica: dict = {"enabled": True}
        tsv = record.get("legacy_preprocessing_tsv") or {}
        for variant in ("remove", "keep"):
            if variant in tsv:
                labels = [x for x in str(tsv[variant]["ica_bad_labels"]).split(";") if x]
                ica[f"excluded_{variant}"] = dict(sorted(Counter(labels).items()))
                ica[f"n_excluded_{variant}"] = len(labels)
        if evidence is not None:
            ica["flatten_reverted_any"] = bool(evidence["ica_flatten_reverted"])
        return {
            "sfreq": sfreq,
            "n_eeg_output": int(len(eeg)),
            "eeg_channels_output": eeg_names,
            "pyprep": pyprep,
            "ica": ica,
            "ecg": {
                "found": bool(ecg_names),
                "channel_fif": ecg_names[0] if ecg_names else None,
                "final_peak_count": int(score["peak_count"]),
                "valid_60s_windows": int(score["valid_60s_windows"]),
                "total_60s_windows": int(score["total_60s_windows"]),
            },
        }
