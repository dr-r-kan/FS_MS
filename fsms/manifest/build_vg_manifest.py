"""Build the VG-microstates manifest from HEPPy outputs (tracked, root-independent, idempotent).

    python -m fsms.manifest.build_vg_manifest --snip-keyfile CLIPS/snip_keyfile.csv --heppy-out HEPPY_OUT
        --database analysis_database.csv --out-dir VG_DIR [--format mat] [--montage standard_1020]
        [--min-duration-s 10] [--conditions II preict] [--raw-kind remove|keep] [--mode fixture]

Descendant of ``OLD/vg_microstates_pipeline.py`` with the same manifest columns and file naming
(``vg_microstate_manifest_<fmt>.csv``, ``fif_segments/``, ``converted_<fmt>/``).  Changes:

* **No silent drops.**  Every snip-keyfile row that does not reach the manifest is written to
  ``vg_microstate_exclusions.csv`` with a reason code: ``unsupported_segment``,
  ``out_of_scope_condition``, ``heppy_excluded`` (recording-scope row of the HEPPy stage's
  pre-specified QC rule), ``heppy_segment_excluded`` (segment-scope row: an overlapping PyPREP chunk
  was skipped/failed or over-interpolated), ``heppy_output_missing`` (legacy "pending"),
  ``segment_too_short`` and ``missing_database_row``.
* **Labels last.**  Phase 1 (everything above except ``missing_database_row``) uses QC evidence and
  segment timing only; the database is opened in phase 2, so group / seizure labels cannot influence
  which segments survive.
* **Conditions.**  Only ``II`` and ``preict`` (preictal) rows enter the manifest by default: the
  hierarchical VG fit pools GFP peaks over every manifest row, and the preregistered analysis
  (``recover_timestamped_transition_backfits.m`` default ``conditions = {'II','preict'}``,
  ``analyse_timestamped_transition_dynamics.py``) uses only those two.  ictal / postict / transictal
  rows are logged as ``out_of_scope_condition``; ``--conditions`` widens the set explicitly.
* **Idempotent, validated, atomic crops** (sentinel keyed by source-FIF identity + crop bounds +
  code + MNE version; the crop must reload with the expected sample count) and **idempotent,
  atomic conversion** through the unmodified VG converter, always with ``--montage`` so channel
  positions exist.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

from fsms.atomic import atomic_write_bytes, atomic_write_json, fingerprint, sha256_file
from fsms.clips.modes import MODES, enforce_mode

CONDITIONS = {"II": "II", "preictal": "preict", "ictal": "ictal", "postictal": "postict", "transictal": "transictal"}
DEFAULT_CONDITIONS = ("II", "preict")
MANIFEST_COLUMNS = [
    "participant", "condition", "group", "file_path", "source_raw_fif", "crop_start_seconds",
    "crop_end_seconds", "patient_id", "source", "source_sheet", "source_row", "event_type", "event_number",
    "segment_number", "segment", "seizure_type", "hypo_hyper", "event_classification", "source_edf_path",
    "source_migrated_edf_path",
]
EXCLUSION_COLUMNS = ["reason_code", "evidence", "patient_code", "output_edf", "segment", "condition",
                     "source_row", "event_type", "event_number", "segment_number",
                     "crop_start_seconds", "crop_end_seconds"]
REASON_CODES = ("unsupported_segment", "out_of_scope_condition", "heppy_excluded", "heppy_segment_excluded",
                "heppy_output_missing", "segment_too_short", "missing_database_row")
SCHEMA = "vg_manifest/1"
CONVERTER = Path(__file__).resolve().parents[2] / "external" / "VG-microstates" / "_internal" / "python" / "_convert_fif_manifest_to_set.py"


class InfrastructureError(RuntimeError):
    pass


def clean(value) -> str:
    return "" if value is None else str(value).strip()


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", clean(value)).strip("_") or "unknown"


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def csv_bytes(rows: list[dict], fieldnames: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_if_changed(path: Path, data: bytes) -> bool:
    try:
        if Path(path).read_bytes() == data:
            return False
    except OSError:
        pass
    atomic_write_bytes(path, data)
    return True


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def package_version(name: str) -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


# ------------------------------------------------------------------------------ database lookup

def db_keys(row: dict) -> list[tuple[str, ...]]:
    return [
        (clean(row.get("patient_code")), clean(row.get("source")), clean(row.get("source_sheet")),
         clean(row.get("source_row")), clean(row.get("event_type")), clean(row.get("event_number")),
         clean(row.get("segment_number"))),
        (clean(row.get("patient_code")), clean(row.get("source_row")), clean(row.get("event_type")),
         clean(row.get("event_number"))),
        (clean(row.get("patient_code")), clean(row.get("source_row"))),
    ]


def db_lookup(rows: list[dict]) -> dict[tuple[str, ...], dict]:
    lookup: dict[tuple[str, ...], dict] = {}
    for row in rows:
        for key in db_keys(row):
            lookup.setdefault(key, row)
    return lookup


def find_db_row(row: dict, lookup: dict[tuple[str, ...], dict]) -> dict | None:
    for key in db_keys(row):
        if key in lookup:
            return lookup[key]
    return None


# ------------------------------------------------------------------------------ phase 1 (QC only)

def exclusion(code: str, evidence: str, row: dict, condition: str = "", crop: tuple[str, str] | None = None) -> dict:
    return {
        "reason_code": code, "evidence": evidence, "patient_code": clean(row.get("patient_code")),
        "output_edf": clean(row.get("output_edf")), "segment": clean(row.get("segment")), "condition": condition,
        "source_row": clean(row.get("source_row")), "event_type": clean(row.get("event_type")),
        "event_number": clean(row.get("event_number")), "segment_number": clean(row.get("segment_number")),
        "crop_start_seconds": crop[0] if crop else clean(row.get("output_start_seconds")),
        "crop_end_seconds": crop[1] if crop else clean(row.get("output_end_seconds")),
    }


def segment_key(base: str, row: dict) -> tuple[str, str, str, str]:
    return (base, clean(row.get("segment")), clean(row.get("output_start_seconds")),
            clean(row.get("output_end_seconds")))


def load_heppy_state(heppy_out: Path) -> tuple[dict[str, dict], dict[str, list[dict]], dict[tuple, list[dict]]]:
    """(review rows by base, recording-scope exclusions by base, segment-scope exclusions by segment key)."""
    review_path = heppy_out / "ecg_qc_review.csv"
    exclusions_path = heppy_out / "heppy_exclusions.csv"
    if not review_path.exists() or not exclusions_path.exists():
        raise InfrastructureError(
            f"{heppy_out} lacks ecg_qc_review.csv/heppy_exclusions.csv: run the HEPPy stage first")
    reviews = {clean(r.get("base")): r for r in read_csv(review_path) if clean(r.get("base"))}
    recording: dict[str, list[dict]] = defaultdict(list)
    segment: dict[tuple, list[dict]] = defaultdict(list)
    for row in read_csv(exclusions_path):
        base = clean(row.get("base"))
        if clean(row.get("scope")) == "segment":
            segment[segment_key(base, row)].append(row)
        else:   # rows of the first-generation file have no scope column: recording level
            recording[base].append(row)
    return reviews, recording, segment


def source_durations(paths: set[str]) -> dict[str, float]:
    import mne

    out = {}
    for path in sorted(paths):
        raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
        try:
            out[path] = raw.n_times / float(raw.info["sfreq"])
        finally:
            raw.close()
    return out


def plan_qc_phase(snip_rows: list[dict], reviews: dict[str, dict], heppy_excluded: dict[str, list[dict]],
                  raw_kind: str, segment_dir: Path, conditions: tuple[str, ...], min_duration_s: float,
                  durations: dict[str, float] | None = None,
                  heppy_segment_excluded: dict[tuple, list[dict]] | None = None) -> tuple[list[dict], list[dict]]:
    """Phase 1: decide survival from QC evidence and timing only.  Returns (candidates, exclusions)."""
    raw_col = "raw_fif_keep" if raw_kind == "keep" else "raw_fif"
    candidates: list[dict] = []
    excluded: list[dict] = []
    used_paths: set[Path] = set()
    pending: list[tuple[int, dict, str, Path, str]] = []

    for i, row in enumerate(snip_rows, start=1):
        segment = clean(row.get("segment"))
        condition = CONDITIONS.get(segment)
        if not condition:
            excluded.append(exclusion("unsupported_segment", f"segment={segment!r}", row))
            continue
        if condition not in conditions:
            excluded.append(exclusion("out_of_scope_condition",
                                      f"condition {condition!r} not in manifest set {list(conditions)}", row, condition))
            continue
        base = Path(clean(row.get("output_edf"))).stem
        if base in heppy_excluded:
            reasons = sorted({clean(r.get("reason_code")) for r in heppy_excluded[base]})
            excluded.append(exclusion("heppy_excluded", "HEPPy QC rule: " + ",".join(reasons), row, condition))
            continue
        seg_rows = (heppy_segment_excluded or {}).get(segment_key(base, row))
        if seg_rows:
            reasons = sorted({clean(r.get("reason_code")) for r in seg_rows})
            excluded.append(exclusion("heppy_segment_excluded", "HEPPy segment rule: " + ",".join(reasons)
                                      + "; " + "; ".join(sorted({clean(r.get("evidence")) for r in seg_rows}))[:300],
                                      row, condition))
            continue
        review = reviews.get(base, {})
        raw_value = clean(review.get(raw_col))
        if not raw_value:
            excluded.append(exclusion("heppy_output_missing", f"no {raw_col} in HEPPy review for {base}", row, condition))
            continue
        raw_path = Path(raw_value)
        if not raw_path.exists():
            excluded.append(exclusion("heppy_output_missing", f"HEPPy raw FIF not found: {raw_path}", row, condition))
            continue
        pending.append((i, row, condition, raw_path, base))

    if durations is None:
        durations = source_durations({str(item[3]) for item in pending})
    for i, row, condition, raw_path, _base in pending:
        start = float(clean(row.get("output_start_seconds")))
        end = min(float(clean(row.get("output_end_seconds"))), durations[str(raw_path)])
        if end - start < min_duration_s:
            excluded.append(exclusion("segment_too_short",
                                      f"cropped duration {end - start:.2f}s < {min_duration_s:g}s", row, condition))
            continue
        patient = clean(row.get("patient_code"))
        source_row = safe(row.get("source_row"))
        event = safe(row.get("event_number"))
        seg = safe(row.get("segment_number")) if clean(row.get("segment_number")) else f"{i:05d}"
        stem = f"{safe(patient)}_{condition}_row{source_row}_event{event}_seg{seg}"
        out_file = segment_dir / f"{stem}.fif"
        if out_file in used_paths:
            s, e = safe(row.get("output_start_seconds")), safe(row.get("output_end_seconds"))
            out_file = segment_dir / f"{stem}_crop{s}-{e}.fif"
        if out_file in used_paths:
            out_file = segment_dir / f"{stem}_crop{s}-{e}_row{i:05d}.fif"
        used_paths.add(out_file)
        candidates.append({"snip": row, "condition": condition, "file_path": out_file, "source_raw_fif": raw_path,
                           "crop_start": start, "crop_end": end})
    if len({str(c["file_path"]) for c in candidates}) != len(candidates):
        raise ValueError("Segment output paths are not unique")
    return candidates, excluded


# ------------------------------------------------------------------------------ phase 2 (labels)

def join_labels(candidates: list[dict], database_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    lookup = db_lookup(database_rows)
    jobs, excluded = [], []
    for item in candidates:
        row = item["snip"]
        db = find_db_row(row, lookup)
        if db is None:
            excluded.append(exclusion("missing_database_row",
                                      "no analysis-database row for (patient_code, source, sheet, source_row, event)",
                                      row, item["condition"]))
            continue
        jobs.append({
            "participant": clean(row.get("patient_code")),
            "condition": item["condition"],
            "group": clean(db.get("patient_type")) or clean(db.get("patient_type_raw")) or "all",
            "file_path": str(item["file_path"]),
            "source_raw_fif": str(item["source_raw_fif"]),
            "crop_start_seconds": clean(row.get("output_start_seconds")),
            "crop_end_seconds": clean(row.get("output_end_seconds")),
            "patient_id": clean(row.get("patient_id")),
            "source": clean(row.get("source")),
            "source_sheet": clean(row.get("source_sheet")),
            "source_row": clean(row.get("source_row")),
            "event_type": clean(row.get("event_type")),
            "event_number": clean(row.get("event_number")),
            "segment_number": clean(row.get("segment_number")),
            "segment": clean(row.get("segment")),
            "seizure_type": clean(db.get("seizure_type")),
            "hypo_hyper": clean(db.get("hypo_hyper")),
            "event_classification": clean(db.get("event_classification")),
            "source_edf_path": clean(row.get("source_edf_path")),
            "source_migrated_edf_path": clean(row.get("source_migrated_edf_path")),
            "_crop": (item["crop_start"], item["crop_end"]),
        })
    return jobs, excluded


# ------------------------------------------------------------------------------ crops

class IdentityCache:
    """sha256 of source FIFs, reused while (size, mtime_ns) are unchanged."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self.data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}
        self.dirty = False

    def identity(self, file: Path) -> dict:
        stat = Path(file).stat()
        key = str(Path(file).resolve())
        cached = self.data.get(key)
        if cached and cached.get("size") == stat.st_size and cached.get("mtime_ns") == stat.st_mtime_ns:
            return cached
        entry = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(file)}
        self.data[key] = entry
        self.dirty = True
        return entry

    def save(self) -> None:
        if self.dirty:
            atomic_write_json(self.path, self.data)


def crop_fingerprint(source_identity: dict, start: float, end: float, min_duration_s: float) -> str:
    return fingerprint({"schema": SCHEMA, "source_sha256": source_identity["sha256"], "start": start,
                        "end": end, "code": code_sha256(), "mne": package_version("mne")})


def crop_valid(out: Path, sentinel_path: Path, fp: str) -> bool:
    try:
        sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
        if sentinel.get("fingerprint") != fp:
            return False
        stat = out.stat()
        if stat.st_size != sentinel.get("size"):
            return False
        if stat.st_mtime_ns != sentinel.get("mtime_ns") and sha256_file(out) != sentinel.get("sha256"):
            return False
        import mne

        raw = mne.io.read_raw_fif(str(out), preload=False, verbose="ERROR")
        try:
            return int(raw.n_times) == sentinel.get("n_times") and len(mne.pick_types(raw.info, eeg=True)) >= 1
        finally:
            raw.close()
    except Exception:
        return False


def crop_jobs(jobs: list[dict], segment_dir: Path, cache: IdentityCache, min_duration_s: float, force: bool,
              log=print) -> dict[str, str]:
    """Crop each job's FIF idempotently.  Returns file_path -> 'written' | 'unchanged'."""
    import mne

    actions: dict[str, str] = {}
    sentinel_dir = segment_dir / ".sentinels"
    segment_dir.mkdir(parents=True, exist_ok=True)
    by_raw: dict[str, list[dict]] = defaultdict(list)
    for job in jobs:
        by_raw[job["source_raw_fif"]].append(job)
    for raw_path, raw_jobs in by_raw.items():
        identity = cache.identity(Path(raw_path))
        raw = None
        for job in raw_jobs:
            out = Path(job["file_path"])
            start, end = job["_crop"]
            fp = crop_fingerprint(identity, start, end, min_duration_s)
            sentinel_path = sentinel_dir / f"{out.stem}.json"
            if not force and crop_valid(out, sentinel_path, fp):
                actions[str(out)] = "unchanged"
                continue
            if raw is None:
                raw = mne.io.read_raw_fif(raw_path, preload=False, verbose="ERROR")
            part = out.with_name(f".{out.stem}.part-{os.getpid()}.fif")
            part.unlink(missing_ok=True)
            try:
                cropped = raw.copy().crop(tmin=start, tmax=end, include_tmax=False)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cropped.save(part, overwrite=True, verbose="ERROR")
                n_times = int(cropped.n_times)
                os.replace(part, out)
            finally:
                part.unlink(missing_ok=True)
            stat = out.stat()
            atomic_write_json(sentinel_path, {
                "schema": SCHEMA, "fingerprint": fp, "source": str(raw_path), "source_sha256": identity["sha256"],
                "start": start, "end": end, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(out), "n_times": n_times})
            actions[str(out)] = "written"
        if raw is not None:
            raw.close()
    return actions


# ------------------------------------------------------------------------------ conversion

def mat_valid(path: Path, sentinel_path: Path, fp: str) -> bool:
    try:
        sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
        if sentinel.get("fingerprint") != fp:
            return False
        stat = path.stat()
        if stat.st_size != sentinel.get("size"):
            return False
        if stat.st_mtime_ns != sentinel.get("mtime_ns") and sha256_file(path) != sentinel.get("sha256"):
            return False
        import scipy.io as sio

        variables = {name: shape for name, shape, _ in sio.whosmat(path)}
        return "data" in variables and "labels" in variables and "pos" in variables
    except Exception:
        return False


MAT_HEADER_TEXT = b"MATLAB 5.0 MAT-file, Platform: nt, written by the VG converter (header timestamp normalised by fsms)"


def normalise_mat_header(path: Path) -> None:
    """Replace scipy's 116-byte descriptive header (it embeds the wall-clock time) with fixed text.

    The converter's ``savemat`` output differs run to run only in that timestamp; fixing it makes a
    healed/rebuilt ``.mat`` byte-identical to the original.  Version/endian bytes and all data are
    untouched, and MATLAB ignores the descriptive text.
    """
    with open(path, "r+b") as handle:
        head = handle.read(128)
        if not head.startswith(b"MATLAB 5.0 MAT-file"):
            return
        handle.seek(0)
        handle.write(MAT_HEADER_TEXT.ljust(116, b" ")[:116])


def convert_jobs(jobs: list[dict], out_dir: Path, fmt: str, montage: str, force: bool, log=print) -> dict[str, str]:
    """Convert crops to ``fmt`` with the unmodified VG converter; idempotent and atomic."""
    if fmt == "fif":
        return {}
    if fmt not in ("mat", "set"):
        raise ValueError(f"unsupported format {fmt}")
    final_dir = out_dir / f"converted_{fmt}"
    sentinel_dir = final_dir / ".sentinels"
    final_dir.mkdir(parents=True, exist_ok=True)
    converter_sha = sha256_file(CONVERTER)
    actions: dict[str, str] = {}
    todo: list[dict] = []
    fps: dict[str, str] = {}
    for job in jobs:
        crop = Path(job["file_path"])
        crop_sentinel = json.loads((crop.parent / ".sentinels" / f"{crop.stem}.json").read_text(encoding="utf-8"))
        fp = fingerprint({"schema": SCHEMA, "crop_sha256": crop_sentinel["sha256"], "montage": montage,
                          "format": fmt, "converter_sha256": converter_sha, "mne": package_version("mne"),
                          "scipy": package_version("scipy"), "numpy": package_version("numpy")})
        target = final_dir / f"{crop.stem}.{fmt}"
        fps[str(crop)] = fp
        if not force and fmt == "mat" and mat_valid(target, sentinel_dir / f"{crop.stem}.json", fp):
            actions[str(crop)] = "unchanged"
        else:
            todo.append(job)
    if todo:
        stage = out_dir / ".convert_stage"
        shutil.rmtree(stage, ignore_errors=True)   # never reuse a half-finished staging area
        stage.mkdir(parents=True, exist_ok=True)
        subset = stage / "subset_manifest.csv"
        fields = [k for k in MANIFEST_COLUMNS]
        atomic_write_bytes(subset, csv_bytes([{k: j[k] for k in fields} for j in todo], fields))
        command = [sys.executable, str(CONVERTER), str(subset), str(stage / "converted"),
                   "--output-manifest", str(stage / "converted_manifest.csv"), "--format", fmt, "--overwrite"]
        if montage:
            command += ["--montage", montage]
        env = dict(os.environ, PYTHONUTF8="1", MPLBACKEND="Agg")
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        (stage / "converter.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise InfrastructureError(f"VG converter failed (exit {result.returncode}): {result.stderr[-600:]}")
        for job in todo:
            crop = Path(job["file_path"])
            produced = stage / "converted" / f"{crop.stem}.{fmt}"
            target = final_dir / produced.name
            if not produced.exists():
                raise InfrastructureError(f"converter did not produce {produced.name}")
            if fmt == "mat":
                normalise_mat_header(produced)
            os.replace(produced, target)
            if fmt == "set":  # EEGLAB export also writes chanloc sidecars
                for suffix in ("_chanlocs.mat", "_electrodes.tsv"):
                    side = stage / "converted" / f"{crop.stem}{suffix}"
                    if side.exists():
                        os.replace(side, final_dir / side.name)
            stat = target.stat()
            atomic_write_json(sentinel_dir / f"{crop.stem}.json", {
                "schema": SCHEMA, "fingerprint": fps[str(crop)], "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(target)})
            actions[str(crop)] = "written"
        shutil.rmtree(stage, ignore_errors=True)
    return actions


# ------------------------------------------------------------------------------ run

def run(snip_keyfile: Path, heppy_out: Path, database: Path, out_dir: Path, fmt: str = "mat",
        montage: str = "standard_1020", min_duration_s: float = 10.0, raw_kind: str = "remove",
        conditions: tuple[str, ...] = DEFAULT_CONDITIONS, force: bool = False, mode: str = "fixture",
        real_data_verifier=None, log=print) -> dict:
    snip_keyfile, heppy_out, database, out_dir = (Path(p).resolve() for p in (snip_keyfile, heppy_out, database, out_dir))
    enforce_mode(mode, {"snip keyfile (input)": snip_keyfile, "HEPPy output (input)": heppy_out,
                        "database (input)": database, "VG output directory (output)": out_dir}, real_data_verifier)
    for path in (snip_keyfile, database):
        if not path.exists():
            raise InfrastructureError(f"input not found: {path}")
    unknown = [c for c in conditions if c not in CONDITIONS.values()]
    if unknown:
        raise InfrastructureError(f"unknown condition(s) {unknown}; expected {sorted(CONDITIONS.values())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.rglob("*"):
        if stale.is_file() and re.search(r"(\.part-\d+|\.part)(\.fif)?$", stale.name):
            stale.unlink()

    segment_dir = out_dir / "fif_segments"
    reviews, heppy_excluded, heppy_segment_excluded = load_heppy_state(heppy_out)
    snip_rows = read_csv(snip_keyfile)
    candidates, excluded = plan_qc_phase(snip_rows, reviews, heppy_excluded, raw_kind, segment_dir,
                                         tuple(conditions), min_duration_s,
                                         heppy_segment_excluded=heppy_segment_excluded)
    jobs, label_excluded = join_labels(candidates, read_csv(database))   # database first opened here
    excluded += label_excluded

    cache = IdentityCache(out_dir / ".sentinels" / "source_fif_identity.json")
    crop_actions = crop_jobs(jobs, segment_dir, cache, min_duration_s, force, log) if jobs else {}
    cache.save()
    rows = [{k: job[k] for k in MANIFEST_COLUMNS} for job in jobs]
    fif_manifest = out_dir / "vg_microstate_manifest_fif.csv"
    write_if_changed(fif_manifest, csv_bytes(rows, MANIFEST_COLUMNS))
    convert_actions = convert_jobs(jobs, out_dir, fmt, montage, force, log) if jobs else {}
    final_manifest = fif_manifest
    if fmt != "fif":
        final_rows = []
        for job in jobs:
            crop = Path(job["file_path"])
            final_rows.append({**{k: job[k] for k in MANIFEST_COLUMNS},
                               "file_path": str(out_dir / f"converted_{fmt}" / f"{crop.stem}.{fmt}")})
        final_manifest = out_dir / f"vg_microstate_manifest_{fmt}.csv"
        write_if_changed(final_manifest, csv_bytes(final_rows, MANIFEST_COLUMNS))
    excluded_sorted = sorted(excluded, key=lambda r: (REASON_CODES.index(r["reason_code"]), r["patient_code"],
                                                      r["source_row"], r["segment"], r["crop_start_seconds"]))
    write_if_changed(out_dir / "vg_microstate_exclusions.csv", csv_bytes(excluded_sorted, EXCLUSION_COLUMNS))
    counts = Counter(r["reason_code"] for r in excluded)
    summary = {
        "schema": SCHEMA, "format": fmt, "montage": montage, "min_duration_s": min_duration_s,
        "conditions": list(conditions), "raw_kind": raw_kind,
        "keyfile_rows": len(snip_rows), "manifest_rows": len(rows),
        "excluded_by_reason": {code: counts.get(code, 0) for code in REASON_CODES},
        "rows_by_condition": dict(sorted(Counter(r["condition"] for r in rows).items())),
        "rows_by_group_condition": dict(sorted(Counter(f"{r['group']}|{r['condition']}" for r in rows).items())),
        "participants": sorted({r["participant"] for r in rows}),
        "manifest": str(final_manifest),
    }
    atomic_summary = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    write_if_changed(out_dir / "vg_manifest_summary.json", atomic_summary.encode("utf-8"))
    log(f"Manifest: {final_manifest}")
    log(f"Rows: {len(rows)} in manifest; excluded {dict((k, v) for k, v in counts.items())}")
    log(f"Crops: {Counter(crop_actions.values())}; conversions: {Counter(convert_actions.values())}")
    return {"manifest": final_manifest, "rows": rows, "excluded": excluded_sorted, "summary": summary,
            "crop_actions": crop_actions, "convert_actions": convert_actions}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop HEPPy FIFs and build the VG-microstates manifest.")
    parser.add_argument("--snip-keyfile", type=Path, required=True)
    parser.add_argument("--heppy-out", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--format", choices=("mat", "set", "fif"), default="mat")
    parser.add_argument("--montage", default="standard_1020", help="MNE montage passed to the VG converter")
    parser.add_argument("--min-duration-s", type=float, default=10.0)
    parser.add_argument("--raw-kind", choices=("remove", "keep"), default="remove",
                        help="cardiac-removed (remove) or cardiac-kept (keep) HEPPy FIF")
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS),
                        help="conditions to emit (default II preict)")
    parser.add_argument("--force", action="store_true", help="redo every crop/conversion")
    parser.add_argument("--mode", choices=MODES, default="fixture")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args.snip_keyfile, args.heppy_out, args.database, args.out_dir, args.format, args.montage,
            args.min_duration_s, args.raw_kind, tuple(args.conditions), args.force, args.mode,
            log=lambda m: print(m, flush=True))
    except InfrastructureError as exc:
        print(f"INFRASTRUCTURE ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
