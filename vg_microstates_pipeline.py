from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


CONDITIONS = {"II": "II", "preictal": "preict", "ictal": "ictal", "postictal": "postict", "transictal": "transictal"}


def clean(value) -> str:
    return "" if value is None else str(value).strip()


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", clean(value)).strip("_") or "unknown"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not fieldnames:
        fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_config(path: Path = Path("directories.txt")) -> dict[str, Path]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row[0]: Path(row[1]) for row in csv.reader(handle) if len(row) >= 2 and row[0] and row[1]}


def default_paths() -> dict[str, Path]:
    cfg = read_config()
    work = cfg.get("working_directory", Path.cwd())
    clips = cfg.get("clip_folder", work / "clips")
    return {
        "database": work / "analysis_database.csv" if (work / "analysis_database.csv").exists() else Path("analysis_database.csv"),
        "snip_manifest": clips / "snip_keyfile.csv",
        "heppy_output": work / "heppy_output",
        "out_dir": work / "vg_microstates",
    }


def db_lookup(rows: list[dict]) -> dict[tuple[str, ...], dict]:
    lookup = {}
    for row in rows:
        keys = [
            (
                clean(row.get("patient_code")),
                clean(row.get("source")),
                clean(row.get("source_sheet")),
                clean(row.get("source_row")),
                clean(row.get("event_type")),
                clean(row.get("event_number")),
                clean(row.get("segment_number")),
            ),
            (clean(row.get("patient_code")), clean(row.get("source_row")), clean(row.get("event_type")), clean(row.get("event_number"))),
            (clean(row.get("patient_code")), clean(row.get("source_row"))),
        ]
        for key in keys:
            lookup.setdefault(key, row)
    return lookup


def find_db_row(row: dict, lookup: dict[tuple[str, ...], dict]) -> dict:
    keys = [
        (
            clean(row.get("patient_code")),
            clean(row.get("source")),
            clean(row.get("source_sheet")),
            clean(row.get("source_row")),
            clean(row.get("event_type")),
            clean(row.get("event_number")),
            clean(row.get("segment_number")),
        ),
        (clean(row.get("patient_code")), clean(row.get("source_row")), clean(row.get("event_type")), clean(row.get("event_number"))),
        (clean(row.get("patient_code")), clean(row.get("source_row"))),
    ]
    for key in keys:
        if key in lookup:
            return lookup[key]
    return {}


def review_by_base(rows: list[dict]) -> dict[str, dict]:
    return {clean(row.get("base")): row for row in rows if clean(row.get("base"))}


def build_jobs(
    snip_rows: list[dict],
    review_rows: list[dict],
    database_rows: list[dict],
    raw_kind: str,
    segment_dir: Path,
) -> tuple[list[dict], list[dict], list[dict]]:
    raw_col = "raw_fif_keep" if raw_kind == "keep" else "raw_fif"
    reviews = review_by_base(review_rows)
    lookup = db_lookup(database_rows)
    jobs: list[dict] = []
    pending: list[dict] = []
    skipped: list[dict] = []

    for i, row in enumerate(snip_rows, start=1):
        segment = clean(row.get("segment"))
        condition = CONDITIONS.get(segment)
        if not condition:
            skipped.append({**row, "reason": f"unsupported segment: {segment}"})
            continue

        base = Path(clean(row.get("output_edf"))).stem
        review = reviews.get(base, {})
        raw_value = clean(review.get(raw_col))
        if not raw_value:
            pending.append({**row, "reason": f"missing {raw_col} in HEPPy review", "expected_raw_fif": ""})
            continue
        raw_path = Path(raw_value)
        if not raw_path.exists():
            pending.append({**row, "reason": "HEPPy raw FIF not ready", "expected_raw_fif": str(raw_path)})
            continue

        db = find_db_row(row, lookup)
        patient = clean(row.get("patient_code"))
        source_row = safe(row.get("source_row"))
        event = safe(row.get("event_number"))
        seg = safe(row.get("segment_number")) if clean(row.get("segment_number")) else f"{i:05d}"
        out_file = segment_dir / f"{safe(patient)}_{condition}_row{source_row}_event{event}_seg{seg}.fif"
        jobs.append(
            {
                "participant": patient,
                "condition": condition,
                "group": clean(db.get("patient_type")) or clean(db.get("patient_type_raw")) or "all",
                "file_path": str(out_file),
                "source_raw_fif": str(raw_path),
                "crop_start_seconds": clean(row.get("output_start_seconds")),
                "crop_end_seconds": clean(row.get("output_end_seconds")),
                "patient_id": clean(row.get("patient_id")),
                "source": clean(row.get("source")),
                "source_sheet": clean(row.get("source_sheet")),
                "source_row": clean(row.get("source_row")),
                "event_type": clean(row.get("event_type")),
                "event_number": clean(row.get("event_number")),
                "segment_number": clean(row.get("segment_number")),
                "segment": segment,
                "seizure_type": clean(db.get("seizure_type")),
                "hypo_hyper": clean(db.get("hypo_hyper")),
                "event_classification": clean(db.get("event_classification")),
                "source_edf_path": clean(row.get("source_edf_path")),
                "source_migrated_edf_path": clean(row.get("source_migrated_edf_path")),
            }
        )
    return jobs, pending, skipped


def crop_jobs(jobs: list[dict], overwrite: bool, min_duration_s: float) -> list[dict]:
    import mne

    ready = []
    by_raw: dict[str, list[dict]] = defaultdict(list)
    for job in jobs:
        by_raw[job["source_raw_fif"]].append(job)

    for raw_path, raw_jobs in by_raw.items():
        raw = mne.io.read_raw_fif(raw_path, preload=False, verbose="ERROR")
        duration = raw.n_times / float(raw.info["sfreq"])
        for job in raw_jobs:
            out = Path(job["file_path"])
            start = float(job["crop_start_seconds"])
            end = min(float(job["crop_end_seconds"]), duration)
            if end - start < min_duration_s:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            if overwrite or not out.exists():
                raw.copy().crop(tmin=start, tmax=end, include_tmax=False).save(out, overwrite=True, verbose="ERROR")
            ready.append(job)
    return ready


def convert_manifest(manifest: Path, out_dir: Path, fmt: str, overwrite: bool, montage: str) -> Path:
    if fmt == "fif":
        return manifest
    converter = Path(__file__).resolve().parent / "external" / "VG-microstates" / "_internal" / "python" / "_convert_fif_manifest_to_set.py"
    out_manifest = out_dir / f"vg_microstate_manifest_{fmt}.csv"
    cmd = [sys.executable, str(converter), str(manifest), str(out_dir / f"converted_{fmt}"), "--output-manifest", str(out_manifest), "--format", fmt]
    if overwrite:
        cmd.append("--overwrite")
    if montage:
        cmd.extend(["--montage", montage])
    subprocess.run(cmd, check=True)
    return out_manifest


def matlab_quote(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def run_matlab(manifest: Path, output_dir: Path, tanova_permutations: int) -> None:
    cmd = (
        "addpath(pwd); "
        f"run_vg_microstates({matlab_quote(manifest)}, {matlab_quote(output_dir)}, "
        f"'tanova_permutations', {int(tanova_permutations)});"
    )
    subprocess.run(["matlab", "-batch", cmd], check=True)


def run(args: argparse.Namespace) -> Path | None:
    segment_dir = args.out_dir / "fif_segments"
    jobs, pending, skipped = build_jobs(
        read_csv(args.snip_manifest),
        read_csv(args.heppy_output / "ecg_qc_review.csv"),
        read_csv(args.database),
        args.raw_kind,
        segment_dir,
    )
    write_csv(args.out_dir / "vg_microstate_pending.csv", pending)
    write_csv(args.out_dir / "vg_microstate_skipped.csv", skipped)
    if not jobs:
        print(f"No HEPPy raw FIFs are ready yet. Pending rows: {len(pending)}")
        return None

    ready = crop_jobs(jobs, args.overwrite, args.min_duration_s)
    if not ready:
        print("HEPPy FIFs exist, but no segments survived cropping/duration checks.")
        return None

    fif_manifest = args.out_dir / "vg_microstate_manifest_fif.csv"
    fields = list(ready[0])
    write_csv(fif_manifest, ready, fields)
    final_manifest = convert_manifest(fif_manifest, args.out_dir, args.format, args.overwrite, args.montage)
    print(f"Manifest: {final_manifest}")
    print(f"Rows: {len(ready)} ready, {len(pending)} pending, {len(skipped)} skipped")

    if args.run_matlab:
        run_matlab(final_manifest, args.out_dir / "hierarchical", args.tanova_permutations)
    return final_manifest


def self_check() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw = root / "raw.fif"
        raw.write_text("placeholder", encoding="utf-8")
        snips = [
            {"patient_code": "P1", "patient_id": "1", "output_edf": "P1_snips.edf", "segment": "preictal", "output_start_seconds": "0", "output_end_seconds": "20", "source": "s", "source_sheet": "sh", "source_row": "7", "event_type": "Ict", "event_number": "2", "segment_number": ""},
            {"patient_code": "P1", "patient_id": "1", "output_edf": "P1_snips.edf", "segment": "ictal", "output_start_seconds": "20", "output_end_seconds": "21", "source_row": "7", "event_type": "Ict", "event_number": "2"},
        ]
        reviews = [{"base": "P1_snips", "raw_fif": str(raw), "raw_fif_keep": ""}]
        db = [{"patient_code": "P1", "source": "s", "source_sheet": "sh", "source_row": "7", "event_type": "Ict", "event_number": "2", "patient_type": "FS", "hypo_hyper": "hypo"}]
        jobs, pending, skipped = build_jobs(snips, reviews, db, "remove", root / "segments")
        assert len(jobs) == 2
        assert jobs[0]["condition"] == "preict"
        assert jobs[1]["condition"] == "ictal"
        assert jobs[0]["group"] == "FS"
        assert jobs[0]["hypo_hyper"] == "hypo"
        assert not pending
        assert not skipped


def parse_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser(description="Crop HEPPy preprocessed raw FIFs into VG-microstates condition inputs.")
    parser.add_argument("--database", type=Path, default=defaults["database"])
    parser.add_argument("--snip-manifest", type=Path, default=defaults["snip_manifest"])
    parser.add_argument("--heppy-output", type=Path, default=defaults["heppy_output"])
    parser.add_argument("--out-dir", type=Path, default=defaults["out_dir"])
    parser.add_argument("--raw-kind", choices=("remove", "keep"), default="remove")
    parser.add_argument("--format", choices=("mat", "set", "fif"), default="mat")
    parser.add_argument("--montage", default="")
    parser.add_argument("--min-duration-s", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-matlab", action="store_true")
    parser.add_argument("--tanova-permutations", type=int, default=5000)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        print("self-check passed")
        return
    run(args)


if __name__ == "__main__":
    main()
