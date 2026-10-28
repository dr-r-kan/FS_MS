"""Deterministic builder of a small, clearly SYNTHETIC test study from TUEG EDF headers.

    python -m fsms.fixture.make_tueg_fixture --work-root F:\\FS_MS_fixture_tueg [--seed 20261028]

Everything written lives under ``--work-root``; TUEG files are only opened read-only (headers and
the first MiB are hashed for provenance) and are never modified or copied: the clip stage
(`fsms.clips.snip_clips`, the real tracked code) later reads them through the database's EDF paths.

All diagnostic groups, event types, seizure types, semiologies and onset times below are
**synthetic test labels assigned by this script with no relation to the TUEG patients' real
conditions**.

Composition (14 distinct TUEG patients, chosen by seeded sampling from sorted candidates; the
cohort is larger than the 8 the analysis needs so that, after the pre-specified QC rule has removed
whatever it removes, at least 3 FS, 3 ES and 2 DD participants with a usable II segment and usable
preictal events remain; selection never looks at QC outcomes):

* FS x5: two hypo-only, two hyper-only, one with two events (hypo + hyper);
* ES x5: one event each;
* DD x4: each with one FS event and one ES event (patient_type ``DD``, seizure_type per event).

Timeline per patient: one interictal (II) interval of 600 s at 120-720 s; the first
ictal onset at 1950 s (so a full 1200 s preictal window exists and does not touch II), ictal
duration 30-60 s; multi-event patients have >= 2400 s between the end of one ictal period and the
next onset (so the legacy clip logic keeps a full preictal for both events and creates no
``transictal`` segment).  II starts at 120 s rather than 30 s because TUEG recordings often
contain an artifact-prone opening minutes that makes PyPREP skip any chunk containing them (probed
on 8 TUEG clips: windows starting at 30 s failed ``RobustReference:TooManyBad`` in 3/8, windows
starting at 120 s passed, 8/8); this is a fixed design constant, not a per-recording choice.
Two-event patients (FS-two-event and the DD patients) need longer recordings than single-event ones.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import sys
from pathlib import Path

from fsms.atomic import atomic_write_bytes, atomic_write_json, sha256_text
from fsms.clips.modes import MODES, enforce_mode

FIXTURE_SCHEMA = "fsms_fixture/1"
DEFAULT_TUEG_ROOT = r"F:\TUEG\v1.2.0\edf\01_tcp_ar"
STANDARD_19 = ("FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8",
               "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ")
II_START_S, II_END_S = 120.0, 720.0
FIRST_ONSET_S = 1950.0          # preictal [onset-1200, onset) is complete and does not touch the II interval
EVENT_GAP_S = 2460.0            # >= 2400 so no transictal merge (PREICTAL + POSTICTAL lengths)
POSTICTAL_S = 1200.0
SINGLE_DURATION = (FIRST_ONSET_S + 60.0 + POSTICTAL_S + 120.0, 4500.0)
DOUBLE_DURATION = (FIRST_ONSET_S + 60.0 + EVENT_GAP_S + 60.0 + POSTICTAL_S + 120.0, 9000.0)
COHORT = {"FS_single": 4, "FS_two_events": 1, "ES": 5, "DD": 4}   # 14 patients

# Real-schema analysis_database.csv columns (from the study's database builder).
DB_COLUMNS = [
    "source", "source_sheet", "source_row", "patient_id", "patient_code", "age", "gender", "init",
    "patient_type", "patient_type_raw", "event_type", "event_number", "segment_number", "seizure_type",
    "seizure_type_raw", "hypo_hyper", "file_template", "edf_number", "file_label", "edf_filename",
    "edf_path", "migrated_edf_filename", "migrated_edf_path", "recording_start", "event_date", "edf_start",
    "edf_duration_seconds", "raw_start", "raw_end", "raw_start_seconds", "raw_end_seconds", "start", "end",
    "start_seconds", "end_seconds", "duration_seconds", "time_basis", "time_warning", "event_classification",
    "event_score_fs", "event_description", "notes", "junk",
]


# ---------------------------------------------------------------------------- header scan

def read_header(path: str | os.PathLike[str]) -> dict:
    """Cheap EDF header parse (reads 256 + 256*ns bytes)."""
    with open(path, "rb") as handle:
        fixed = handle.read(256)
        n_records = int(fixed[236:244])
        record_s = float(fixed[244:252])
        ns = int(fixed[252:256])
        body = handle.read(256 * ns)
    labels = [body[i * 16:(i + 1) * 16].decode("latin1").strip() for i in range(ns)]
    offset = 16 * ns + 80 * ns + 8 * ns * 5 + 80 * ns
    samples = [int(body[offset + i * 8: offset + (i + 1) * 8]) for i in range(ns)]
    return {"n_records": n_records, "record_s": record_s, "duration_s": n_records * record_s,
            "labels": labels, "samples": samples, "header_len": 256 + 256 * ns,
            "sfreq": samples[0] / record_s if record_s else 0.0}


def iter_edfs(root: Path):
    """Deterministic (sorted) walk yielding EDF paths; lazily, so a header cap stops early."""
    for group in sorted(os.scandir(root), key=lambda e: e.name):
        if not group.is_dir():
            continue
        for patient in sorted(os.scandir(group.path), key=lambda e: e.name):
            if not patient.is_dir():
                continue
            for session in sorted(os.scandir(patient.path), key=lambda e: e.name):
                if not session.is_dir():
                    continue
                for entry in sorted(os.scandir(session.path), key=lambda e: e.name):
                    if entry.is_file() and entry.name.lower().endswith(".edf"):
                        yield Path(entry.path), patient.name


def scan_headers(root: Path, max_headers: int, cache_path: Path | None = None) -> list[dict]:
    key = {"root": str(root), "max_headers": max_headers, "schema": FIXTURE_SCHEMA}
    if cache_path is not None and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("key") == key:
                return cached["rows"]
        except (OSError, ValueError):
            pass
    rows = []
    for path, patient in iter_edfs(root):
        try:
            header = read_header(path)
        except Exception:
            continue
        rows.append({"path": str(path), "patient": patient, **{k: header[k] for k in
                     ("duration_s", "record_s", "sfreq", "labels", "samples", "header_len")}})
        if len(rows) >= max_headers:
            break
    if cache_path is not None:
        atomic_write_json(cache_path, {"key": key, "rows": rows})
    return rows


def is_candidate(row: dict) -> bool:
    labels = [label.upper() for label in row["labels"]]
    if row["record_s"] != 1.0 or row["sfreq"] not in (250.0, 256.0):
        return False
    ekg = [i for i, label in enumerate(labels) if "EKG" in label or "ECG" in label]
    if not ekg or row["samples"][ekg[0]] != row["samples"][0]:
        return False
    names = set(labels)
    return all(f"EEG {name}-REF" in names for name in STANDARD_19)


# ---------------------------------------------------------------------------- selection

def choose(rows: list[dict], seed: int, cohort: dict[str, int] | None = None) -> dict:
    """Pick patients/files and synthetic roles; pure function of (rows, seed, cohort)."""
    cohort = dict(COHORT if cohort is None else cohort)
    rng = random.Random(seed)
    candidates = sorted((r for r in rows if is_candidate(r)), key=lambda r: (r["patient"], r["path"]))
    doubles = [r for r in candidates if DOUBLE_DURATION[0] <= r["duration_s"] <= DOUBLE_DURATION[1]]
    singles = [r for r in candidates if SINGLE_DURATION[0] <= r["duration_s"] <= SINGLE_DURATION[1]]

    def pick(pool: list[dict], n: int, used: set[str]) -> list[dict]:
        by_patient: dict[str, list[dict]] = {}
        for row in pool:
            if row["patient"] not in used:
                by_patient.setdefault(row["patient"], []).append(row)
        patients = sorted(by_patient)
        if len(patients) < n:
            raise RuntimeError(f"only {len(patients)} suitable TUEG patients for {n} slots; raise --max-headers")
        chosen = rng.sample(patients, n)
        out = [rng.choice(by_patient[p]) for p in chosen]
        used.update(chosen)
        return out

    used: set[str] = set()
    n_two = cohort["FS_two_events"] + cohort["DD"]
    n_one = cohort["FS_single"] + cohort["ES"]
    two_event = pick(doubles, n_two, used)
    one_event = pick(singles, n_one, used)
    roles_two = ["FS_two_events"] * cohort["FS_two_events"] + ["DD"] * cohort["DD"]
    semiology_cycle = rng.sample(["hypo", "hyper"], 2)
    roles_one = [f"FS_single_{semiology_cycle[i % 2]}" for i in range(cohort["FS_single"])] + ["ES"] * cohort["ES"]
    slots = list(zip(roles_two, two_event)) + list(zip(roles_one, one_event))
    slots.sort(key=lambda item: (item[1]["patient"], item[1]["path"]))
    plan = []
    for index, (role, row) in enumerate(slots, start=1):
        code = f"FIX_{index:03d}"
        ictal = [rng.randint(30, 60), rng.randint(30, 60)]
        if role == "FS_two_events":
            order = ["hypo", "hyper"] if rng.random() < 0.5 else ["hyper", "hypo"]
            events = [("FS", order[0]), ("FS", order[1])]
            group = "FS"
        elif role == "DD":
            first = rng.choice(["FS", "ES"])
            semi = rng.choice(["hypo", "hyper"])
            events = [("FS", semi), ("ES", "")] if first == "FS" else [("ES", ""), ("FS", semi)]
            group = "DD"
        elif role.startswith("FS_single"):
            events, group = [("FS", role.rsplit("_", 1)[1])], "FS"
        else:
            events, group = [("ES", "")], "ES"
        onsets = []
        onset = FIRST_ONSET_S
        for k, _event in enumerate(events):
            onsets.append((onset, onset + ictal[k]))
            onset = onset + ictal[k] + EVENT_GAP_S
        plan.append({"code": code, "role": role, "group": group, "events": events, "ictal": onsets, "row": row})
    return {"patients": plan, "cohort": cohort}


def database_rows(plan: dict) -> list[dict]:
    rows = []
    source_row = 2
    for patient in plan["patients"]:
        edf = Path(patient["row"]["path"])
        common = {
            "source": "FIXTURE_TUEG_SYNTHETIC", "source_sheet": "synthetic", "patient_id": patient["row"]["patient"],
            "patient_code": patient["code"], "age": "", "gender": "", "init": "",
            "patient_type": patient["group"], "patient_type_raw": patient["group"],
            "file_template": "", "edf_number": "1", "file_label": edf.stem, "edf_filename": edf.name,
            "edf_path": str(edf), "migrated_edf_filename": edf.name, "migrated_edf_path": str(edf),
            "recording_start": "", "event_date": "", "edf_start": "",
            "edf_duration_seconds": f"{patient['row']['duration_s']:.0f}",
            "time_basis": "fixture_seconds_from_edf_start", "time_warning": "", "event_score_fs": "",
            "notes": "FIXTURE_NOT_REAL_DATA", "junk": "0",
        }
        ii = dict(common, source_row=str(source_row), event_type="II", event_number="0", segment_number="1",
                  seizure_type="", seizure_type_raw="", hypo_hyper="", event_classification="synthetic_interictal",
                  event_description="SYNTHETIC FIXTURE LABEL - NOT REAL DATA")
        ii.update(raw_start_seconds=f"{II_START_S:g}", raw_end_seconds=f"{II_END_S:g}", raw_start="", raw_end="",
                  start="", end="", start_seconds=f"{II_START_S:g}", end_seconds=f"{II_END_S:g}",
                  duration_seconds=f"{II_END_S - II_START_S:g}")
        rows.append(ii)
        source_row += 1
        for number, ((seizure, semiology), (onset, end)) in enumerate(zip(patient["events"], patient["ictal"]), start=1):
            row = dict(common, source_row=str(source_row), event_type="Ict", event_number=str(number),
                       segment_number="", seizure_type=seizure, seizure_type_raw=seizure, hypo_hyper=semiology,
                       event_classification="synthetic_" + seizure,
                       event_description="SYNTHETIC FIXTURE LABEL - NOT REAL DATA")
            row.update(raw_start_seconds=f"{onset:g}", raw_end_seconds=f"{end:g}", raw_start="", raw_end="",
                       start="", end="", start_seconds=f"{onset:g}", end_seconds=f"{end:g}",
                       duration_seconds=f"{end - onset:g}")
            rows.append(row)
            source_row += 1
    return rows


def csv_text(rows: list[dict]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=DB_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def file_provenance(path: Path, header_len: int) -> dict:
    with open(path, "rb") as handle:
        head = handle.read(header_len)
        handle.seek(0)
        first_mb = handle.read(1 << 20)
    return {"path": str(path), "size": path.stat().st_size, "header_sha256": hashlib.sha256(head).hexdigest(),
            "first_mib_sha256": hashlib.sha256(first_mb).hexdigest()}


def write_if_changed(path: Path, data: bytes) -> bool:
    try:
        if path.read_bytes() == data:
            return False
    except OSError:
        pass
    atomic_write_bytes(path, data)
    return True


NOT_REAL_TEXT = """FIXTURE - NOT REAL DATA
=======================

This directory is a synthetic test study built by fsms.fixture.make_tueg_fixture from public TUEG
(Temple University EEG Corpus) recordings.  Every diagnostic group (FS / ES / DD), seizure type,
semiology (hypo / hyper), event time and interictal interval in analysis_database.csv is a
SYNTHETIC TEST LABEL assigned by a script.  It bears no relation to the TUEG patients' real
conditions and carries no scientific meaning.  Do not analyse, report or mix with real study data.
TUEG files are never modified; see fixture_manifest.json for their paths and hashes.
"""


def build(work_root: Path, tueg_root: Path, seed: int, max_headers: int, run_clips: bool = True, log=print) -> dict:
    work_root.mkdir(parents=True, exist_ok=True)
    rows = scan_headers(tueg_root, max_headers, work_root / ".scan_cache.json")
    log(f"{len(rows)} TUEG headers scanned (cap {max_headers})")
    plan = choose(rows, seed)
    db_rows = database_rows(plan)
    db_path = work_root / "analysis_database.csv"
    changed = write_if_changed(db_path, csv_text(db_rows))
    write_if_changed(work_root / "FIXTURE_NOT_REAL_DATA.txt", NOT_REAL_TEXT.encode("utf-8"))

    sources = []
    for patient in plan["patients"]:
        row = patient["row"]
        prov = file_provenance(Path(row["path"]), row["header_len"])
        sources.append({**prov, "tueg_patient": row["patient"], "duration_s": row["duration_s"],
                        "sfreq": row["sfreq"], "n_channels": len(row["labels"]),
                        "fixture_code": patient["code"], "synthetic_role": patient["role"],
                        "synthetic_group": patient["group"],
                        "synthetic_events": [
                            {"seizure_type": s, "hypo_hyper": h, "onset_s": o, "end_s": e}
                            for (s, h), (o, e) in zip(patient["events"], patient["ictal"])]})
    manifest = {
        "schema": FIXTURE_SCHEMA, "fixture": True, "not_real_data": True, "seed": seed,
        "tueg_root": str(tueg_root), "max_headers": max_headers,
        "database_sha256": hashlib.sha256(db_path.read_bytes()).hexdigest(),
        "layout": {"clips": "clips/", "database": "analysis_database.csv", "heppy_output": "heppy_output/",
                   "vg_microstates": "vg_microstates/"},
        "note": "All groups, event types, semiologies and times are synthetic test labels.",
        "design": {
            "cohort": plan["cohort"],
            "selection": "seeded random.Random(seed).sample over sorted candidate patients; never uses QC outcomes",
            "candidate_filter": "01_tcp_ar, 250/256 Hz, EKG channel at EEG rate, all 19 standard scalp channels",
            "single_event_duration_s": list(SINGLE_DURATION), "two_event_duration_s": list(DOUBLE_DURATION),
            "ii_interval_s": [II_START_S, II_END_S], "first_ictal_onset_s": FIRST_ONSET_S,
            "between_events_gap_s": EVENT_GAP_S,
            "why_ii_starts_late": "TUEG openings are artifact-prone: PyPREP skipped (TooManyBad) 30-s-start II chunks "
                                  "in 3/8 probed clips at any length, 120-s-start chunks passed",
            "why_14_patients": "after segment-level QC at least 3 FS, 3 ES and 2 DD (both events) must remain",
        },
        "sources": sources,
    }
    atomic_write_json(work_root / "fixture_manifest.json", manifest)
    log(f"database {'written' if changed else 'unchanged'}: {db_path} ({len(db_rows)} rows, "
        f"{len(plan['patients'])} patients)")
    for patient in plan["patients"]:
        log(f"  {patient['code']} {patient['group']:>2} {patient['role']:<14} TUEG {patient['row']['patient']} "
            f"{Path(patient['row']['path']).name} {patient['row']['duration_s']:.0f}s events={patient['events']}")
    if run_clips:
        from fsms.clips import snip_clips

        actions: dict[str, str] = {}
        clips = work_root / "clips"
        manifest_rows, skipped = snip_clips.write_snips(db_path, clips, clips / "snip_keyfile.csv",
                                                        clips / "snip_skipped.csv", actions=actions)
        log(f"clips: {actions}; {len(manifest_rows)} segments; skipped={len(skipped)}")
    return {"plan": plan, "manifest": manifest}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the synthetic TUEG test study (never real data).")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--tueg-root", type=Path, default=Path(DEFAULT_TUEG_ROOT))
    parser.add_argument("--max-headers", type=int, default=3500, help="cap on EDF headers read")
    parser.add_argument("--no-clips", action="store_true", help="only write the database and manifest")
    parser.add_argument("--mode", choices=MODES, default="fixture")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    enforce_mode(args.mode, {"fixture work root (output)": args.work_root, "TUEG root (read-only input)": args.tueg_root})
    if args.mode != "fixture":
        print("the fixture builder only runs in fixture mode", file=sys.stderr)
        return 2
    if not args.tueg_root.is_dir():
        print(f"TUEG root not found: {args.tueg_root}", file=sys.stderr)
        return 2
    build(args.work_root, args.tueg_root, args.seed, args.max_headers, run_clips=not args.no_clips,
          log=lambda m: print(m, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
