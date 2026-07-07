from __future__ import annotations

"""

An mne-free pipeline for snipping EDFs into patient-specific segments with annotations.

"""

import argparse
import csv
import math
import os
import re
import struct
import tempfile
from contextlib import ExitStack
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


PREICTAL_LENGTH = 20 * 60  # 20 minutes in seconds
POSTICTAL_LENGTH = 20 * 60  # 20 minutes in seconds
MIN_ANNOTATION_SAMPLES = 64

SIGNAL_FIELDS = [
    ("labels", 16),
    ("transducers", 80),
    ("physical_dimensions", 8),
    ("physical_mins", 8),
    ("physical_maxes", 8),
    ("digital_mins", 8),
    ("digital_maxes", 8),
    ("prefilterings", 80),
    ("samples_per_record", 8),
    ("reserved", 32),
]


@dataclass
class Edf:
    path: Path
    header: bytes
    patient: str
    recording: str
    start_date: str
    start_time: str
    records: int
    record_duration: float
    record_duration_text: str
    fields: dict[str, list[str]]
    samples: list[int]
    non_annotation: list[int]
    signal_offsets: list[int]
    signal_bytes: list[int]
    record_bytes: int


def clean(value) -> str:
    return "" if value is None else str(value).strip()


def fixed(value, width: int) -> bytes:
    return clean(value).encode("ascii", "replace")[:width].ljust(width)


def read_edf(path: Path) -> Edf:
    with path.open("rb") as handle:
        header = handle.read(256)
        signal_count = int(header[252:256].decode("ascii", "ignore").strip())
        header += handle.read(256 * signal_count)

    fields: dict[str, list[str]] = {}
    pos = 256
    for name, width in SIGNAL_FIELDS:
        fields[name] = [header[pos + i * width : pos + (i + 1) * width].decode("ascii", "ignore").strip() for i in range(signal_count)]
        pos += width * signal_count

    samples = [int(value) for value in fields["samples_per_record"]]
    signal_bytes = [sample_count * 2 for sample_count in samples]
    offsets = []
    offset = 0
    for size in signal_bytes:
        offsets.append(offset)
        offset += size

    non_annotation = [i for i, label in enumerate(fields["labels"]) if "annotation" not in label.lower()]
    return Edf(
        path=path,
        header=header,
        patient=header[8:88].decode("ascii", "ignore").strip(),
        recording=header[88:168].decode("ascii", "ignore").strip(),
        start_date=header[168:176].decode("ascii", "ignore").strip(),
        start_time=header[176:184].decode("ascii", "ignore").strip(),
        records=int(header[236:244].decode("ascii", "ignore").strip()),
        record_duration=float(header[244:252].decode("ascii", "ignore").strip()),
        record_duration_text=header[244:252].decode("ascii", "ignore").strip(),
        fields=fields,
        samples=samples,
        non_annotation=non_annotation,
        signal_offsets=offsets,
        signal_bytes=signal_bytes,
        record_bytes=sum(signal_bytes),
    )


def output_header(
    edf: Edf,
    patient_code: str,
    records: int,
    signal_indexes: list[int],
    annotation_samples: int,
) -> bytes:
    signal_count = len(signal_indexes) + 1
    pieces = [
        fixed("0", 8),
        fixed(patient_code, 80),
        fixed("Snipped segments", 80),
        fixed(edf.start_date, 8),
        fixed(edf.start_time, 8),
        fixed(256 + 256 * signal_count, 8),
        fixed("EDF+C", 44),
        fixed(records, 8),
        fixed(edf.record_duration_text, 8),
        fixed(signal_count, 4),
    ]

    out_fields = {name: [edf.fields[name][i] for i in signal_indexes] for name, _ in SIGNAL_FIELDS}
    out_fields["labels"].append("EDF Annotations")
    out_fields["transducers"].append("")
    out_fields["physical_dimensions"].append("")
    out_fields["physical_mins"].append("-1")
    out_fields["physical_maxes"].append("1")
    out_fields["digital_mins"].append("-32768")
    out_fields["digital_maxes"].append("32767")
    out_fields["prefilterings"].append("")
    out_fields["samples_per_record"].append(str(annotation_samples))
    out_fields["reserved"].append("")

    for name, width in SIGNAL_FIELDS:
        pieces.extend(fixed(value, width) for value in out_fields[name])
    return b"".join(pieces)


def seconds(value) -> float:
    if clean(value) == "":
        raise ValueError("missing seconds")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"bad seconds value: {value}")
    return result


def fmt_time(value: float) -> str:
    if abs(value - round(value)) < 0.000001:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def tal(onset: float, text: str = "") -> bytes:
    text = re.sub(r"[\x00-\x1f]+", " ", text).strip()
    return f"+{fmt_time(onset)}\x14{text}\x14\x00".encode("ascii", "replace")


def annotation_payload(record_onset: float, annotations: list[tuple[float, str]]) -> bytes:
    payload = bytearray(tal(record_onset))
    for onset, text in annotations:
        payload.extend(tal(onset, text))
    return bytes(payload)


def annotation_samples_required(
    records: int,
    record_duration: float,
    annotations: dict[int, list[tuple[float, str]]],
) -> int:
    # EDF+ stores the TAL stream in the two bytes belonging to each annotation
    # sample.  Size this channel from the actual payload instead of truncating it.
    last_onset = max(0.0, (records - 1) * record_duration)
    max_bytes = max(len(tal(0.0)), len(tal(last_onset)))
    for record, items in annotations.items():
        onset = record * record_duration
        max_bytes = max(max_bytes, len(annotation_payload(onset, items)))
    return max(MIN_ANNOTATION_SAMPLES, math.ceil(max_bytes / 2))


def annotation_record(
    record_onset: float,
    annotations: list[tuple[float, str]],
    annotation_samples: int,
) -> bytes:
    capacity = annotation_samples * 2
    payload = annotation_payload(record_onset, annotations)
    if len(payload) > capacity:
        raise ValueError(
            f"Annotation payload is {len(payload)} bytes but the EDF annotation "
            f"channel has capacity {capacity} bytes"
        )
    return payload.ljust(capacity, b"\x00")


def read_config(path: Path = Path("directories.txt")) -> dict[str, Path]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {row[0]: Path(row[1]) for row in csv.reader(handle) if len(row) >= 2 and row[0] and row[1]}


def default_database(config: dict[str, Path]) -> Path:
    path = config.get("working_directory", Path.cwd()) / "analysis_database.csv"
    return path if path.exists() else Path("analysis_database.csv")


def default_output_dir(config: dict[str, Path]) -> Path:
    if "clip_folder" in config:
        return config["clip_folder"]
    if "working_directory" in config:
        return config["working_directory"] / "snipped_edfs"
    return Path("snipped_edfs")


def source_path(row: dict) -> Path:
    fallback = ""
    for name in ("migrated_edf_path", "edf_path"):
        value = clean(row.get(name))
        if value and Path(value).exists():
            return Path(value)
        fallback = fallback or value
    if fallback:
        return Path(fallback)
    raise ValueError(f"missing EDF path for source row {row.get('source_row')}")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", clean(value)).strip("_") or "unknown_patient"


def channel_key(label: str) -> str:
    label = re.sub(r"^(EEG|ECG)\s+", "", clean(label), flags=re.IGNORECASE)
    label = label.replace("-G2", "")
    label = re.sub(r"([+-])-.*$", r"\1", label)
    return label.lower()


def channel_map(edf: Edf) -> dict[str, int]:
    channels = {}
    for signal in edf.non_annotation:
        channels.setdefault(channel_key(edf.fields["labels"][signal]), signal)
    return channels


def common_channels(edfs: list[Edf]) -> tuple[Edf, list[int], dict[Path, list[int]], str]:
    base = edfs[0]
    maps = {edf.path: channel_map(edf) for edf in edfs}
    common = set.intersection(*(set(mapping) for mapping in maps.values()))
    output_keys = [channel_key(base.fields["labels"][signal]) for signal in base.non_annotation if channel_key(base.fields["labels"][signal]) in common]
    if not output_keys:
        raise ValueError(f"No common data channels for patient recordings starting with {base.path}")
    if any(edf.record_duration != base.record_duration for edf in edfs):
        raise ValueError(f"Record duration differs for patient recordings starting with {base.path}")

    by_path = {edf.path: [maps[edf.path][key] for key in output_keys] for edf in edfs}
    for edf in edfs:
        for base_signal, source_signal in zip(by_path[base.path], by_path[edf.path]):
            for field in ("samples_per_record", "physical_dimensions", "physical_mins", "physical_maxes", "digital_mins", "digital_maxes"):
                if edf.fields[field][source_signal] != base.fields[field][base_signal]:
                    raise ValueError(f"Channel scale differs for {base.fields['labels'][base_signal]} in {edf.path}")

    output_channels = ";".join(base.fields["labels"][signal] for signal in by_path[base.path])
    return base, by_path[base.path], by_path, output_channels


def sort_key(row: dict) -> tuple:
    return (str(source_path(row)).lower(), seconds(row["start_seconds"]), seconds(row["end_seconds"]), clean(row.get("event_number")), clean(row.get("segment_number")))


def row_segments(row: dict, edf: Edf) -> list[tuple[str, float, float]]:
    start = seconds(row["start_seconds"])
    end = seconds(row["end_seconds"])
    total = edf.records * edf.record_duration
    if end <= start:
        raise ValueError(f"bad event times in source row {row.get('source_row')}: {start} >= {end}")

    event_type = clean(row.get("event_type"))
    if event_type == "II":
        segments = [("II", start, end)]
    elif event_type == "Ict":
        segments = [
            ("preictal", start - PREICTAL_LENGTH, start),
            ("ictal", start, end),
            ("postictal", end, end + POSTICTAL_LENGTH),
        ]
    else:
        raise ValueError(f"unexpected event type in source row {row.get('source_row')}: {event_type}")

    clipped = []
    for label, left, right in segments:
        left = max(0.0, left)
        right = min(total, right)
        if right > left:
            clipped.append((label, left, right))
    return clipped


def merge_overlapping_ictal(rows: list[dict]) -> list[dict]:
    by_source: dict[Path, list[dict]] = defaultdict(list)
    merged = [row for row in rows if clean(row.get("event_type")) != "Ict"]
    for row in rows:
        if clean(row.get("event_type")) == "Ict":
            by_source[source_path(row)].append(row)

    for source_rows in by_source.values():
        clusters: list[list[dict]] = []
        for row in sorted(source_rows, key=lambda item: (seconds(item["start_seconds"]), seconds(item["end_seconds"]))):
            if not clusters or seconds(row["start_seconds"]) > max(seconds(item["end_seconds"]) for item in clusters[-1]):
                clusters.append([row])
            else:
                clusters[-1].append(row)
        for cluster in clusters:
            outer = dict(cluster[0])
            outer["start_seconds"] = min(seconds(row["start_seconds"]) for row in cluster)
            outer["end_seconds"] = max(seconds(row["end_seconds"]) for row in cluster)
            outer["merged_ictal_source_rows"] = ";".join(filter(None, (clean(row.get("source_row")) for row in cluster)))
            outer["merged_ictal_event_numbers"] = ";".join(filter(None, (clean(row.get("event_number")) for row in cluster)))
            merged.append(outer)
    return sorted(merged, key=sort_key)


def subtract_intervals(start: float, end: float, blocked: list[tuple[float, float]]) -> list[tuple[float, float]]:
    remaining = [(start, end)]
    for left, right in sorted(blocked):
        remaining = [piece for a, b in remaining for piece in ((a, min(b, left)), (max(a, right), b)) if piece[1] > piece[0]]
    return remaining


def patient_segments(rows: list[dict], cache: dict[Path, Edf]):
    """Yield peri-ictal unions first, subtracting them from any II intervals."""
    rows = merge_overlapping_ictal(rows)
    ictal_by_source: dict[Path, list[dict]] = defaultdict(list)
    for row in rows:
        if clean(row.get("event_type")) == "Ict":
            ictal_by_source[source_path(row)].append(row)

    transictal_after: dict[int, dict] = {}
    suppress_pre: set[int] = set()
    for ictal_rows in ictal_by_source.values():
        ictal_rows.sort(key=lambda row: seconds(row["start_seconds"]))
        for previous, following in zip(ictal_rows, ictal_rows[1:]):
            gap = seconds(following["start_seconds"]) - seconds(previous["end_seconds"])
            if 0 < gap < PREICTAL_LENGTH + POSTICTAL_LENGTH:
                transictal_after[id(previous)] = following
                suppress_pre.add(id(following))

    segments = []
    for row in rows:
        if clean(row.get("event_type")) != "Ict":
            continue
        edf = cache[source_path(row)]
        following = transictal_after.get(id(row))
        for label, start, end in row_segments(row, edf):
            if label == "preictal" and id(row) in suppress_pre:
                continue
            if label == "postictal" and following:
                continue
            segments.append((row, label, start, end, None))
            if label == "ictal" and following:
                segments.append((row, "transictal", end, seconds(following["start_seconds"]), following))

    blocked: dict[Path, list[tuple[float, float]]] = defaultdict(list)
    for row, _label, start, end, _following in segments:
        blocked[source_path(row)].append((start, end))
    for row in rows:
        if clean(row.get("event_type")) != "II":
            continue
        edf = cache[source_path(row)]
        for _label, start, end in row_segments(row, edf):
            for left, right in subtract_intervals(start, end, blocked[source_path(row)]):
                segments.append((row, "II", left, right, None))

    yield from sorted(segments, key=lambda item: (str(source_path(item[0])).lower(), item[2], item[3]))


def link_text(row: dict, label: str, source: Path, start: float, end: float) -> str:
    bits = [
        label,
        f"row={clean(row.get('source_row'))}",
        f"event={clean(row.get('event_type'))}{clean(row.get('event_number'))}",
        f"seg={clean(row.get('segment_number')) or label}",
        f"file={source.name}",
        f"src={fmt_time(start)}-{fmt_time(end)}",
    ]
    return " ".join(bits)


def write_patient(patient_code: str, rows: list[dict], out_dir: Path, cache: dict[Path, Edf]) -> list[dict]:
    sorted_rows = sorted(rows, key=sort_key)
    edfs = []
    for row in sorted_rows:
        path = source_path(row)
        if path not in cache:
            cache[path] = read_edf(path)
        if cache[path] not in edfs:
            edfs.append(cache[path])

    base, base_signals, signals_by_path, output_channels = common_channels(edfs)
    pieces = []
    annotations = defaultdict(list)
    manifest = []
    output_record = 0

    for row, label, start, end, following in patient_segments(sorted_rows, cache):
        path = source_path(row)
        edf = cache[path]
        first = max(0, math.floor(start / edf.record_duration))
        last = min(edf.records, math.ceil(end / edf.record_duration))
        if last <= first:
            continue
        out_start = output_record * edf.record_duration
        out_end = out_start + (last - first) * edf.record_duration
        text = link_text(row, label, path, start, end)
        if pieces:
            # MNE recognises EDGE annotations and will split filtering at the
            # hard join rather than convolving across unrelated source periods.
            annotations[output_record].append((out_start, "EDGE boundary"))
            annotations[output_record].append((out_start, "BAD boundary"))
        annotations[output_record].append((out_start, f"{text} START"))
        annotations[output_record + last - first - 1].append((out_end, f"{text} END"))
        pieces.append((edf, signals_by_path[edf.path], first, last))
        manifest.append(
            {
                "patient_code": patient_code,
                "patient_id": clean(row.get("patient_id")),
                "output_edf": f"{safe_name(patient_code)}_snips.edf",
                "output_channel_count": len(base_signals),
                "output_channels": output_channels,
                "segment": label,
                "output_start_seconds": out_start,
                "output_end_seconds": out_end,
                "source_segment_start_seconds": start,
                "source_segment_end_seconds": end,
                "source_record_start_seconds": first * edf.record_duration,
                "source_record_end_seconds": last * edf.record_duration,
                "source": clean(row.get("source")),
                "source_sheet": clean(row.get("source_sheet")),
                "source_row": clean(row.get("source_row")),
                "event_type": clean(row.get("event_type")),
                "event_number": clean(row.get("event_number")),
                "segment_number": clean(row.get("segment_number")),
                "source_edf_path": clean(row.get("edf_path")),
                "source_migrated_edf_path": clean(row.get("migrated_edf_path")),
                "transictal_next_source_row": clean(following.get("source_row")) if following else "",
                "transictal_next_event_number": clean(following.get("event_number")) if following else "",
                "merged_ictal_source_rows": clean(row.get("merged_ictal_source_rows")),
                "merged_ictal_event_numbers": clean(row.get("merged_ictal_event_numbers")),
            }
        )
        output_record = int(round(out_end / edf.record_duration))

    if not pieces:
        return []

    # Quantify source-data duplication without changing the requested output.
    # Closely spaced events can create heavily overlapping 20-minute windows,
    # which are appended independently and can dominate downstream runtime.
    total_source_records = sum(last - first for _edf, _signals, first, last in pieces)
    intervals_by_path: dict[Path, list[tuple[int, int]]] = defaultdict(list)
    for edf, _signals, first, last in pieces:
        intervals_by_path[edf.path].append((first, last))
    unique_source_records = 0
    for intervals in intervals_by_path.values():
        merged: list[list[int]] = []
        for first, last in sorted(intervals):
            if not merged or first > merged[-1][1]:
                merged.append([first, last])
            else:
                merged[-1][1] = max(merged[-1][1], last)
        unique_source_records += sum(last - first for first, last in merged)
    duplication_factor = (
        total_source_records / unique_source_records if unique_source_records else 1.0
    )
    for item in manifest:
        item["source_record_duplication_factor"] = f"{duplication_factor:.6f}"
    if duplication_factor > 1.05:
        print(
            f"WARNING: {patient_code} output repeats source records by a factor "
            f"of {duplication_factor:.2f}; overlapping event windows will increase "
            "preprocessing time and memory."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_name(patient_code)}_snips.edf"
    annotation_samples = annotation_samples_required(
        output_record, base.record_duration, annotations
    )

    # Keep each source file open and seek once per contiguous piece.  The old
    # implementation performed a redundant seek for every EDF record.  Write
    # beside the destination and rename only after the byte count validates, so
    # a stopped run cannot leave a plausible-looking partial EDF.
    temporary = out_path.with_name(f".{out_path.name}.part-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        with ExitStack() as stack:
            sources = {
                edf.path: stack.enter_context(
                    edf.path.open("rb", buffering=4 * 1024 * 1024)
                )
                for edf in edfs
            }
            with temporary.open("wb", buffering=4 * 1024 * 1024) as out:
                header = output_header(
                    base, patient_code, output_record, base_signals, annotation_samples
                )
                out.write(header)
                current = 0
                for edf, signal_indexes, first, last in pieces:
                    source = sources[edf.path]
                    source.seek(len(edf.header) + first * edf.record_bytes)
                    for _record in range(first, last):
                        data = source.read(edf.record_bytes)
                        if len(data) != edf.record_bytes:
                            raise EOFError(f"Short EDF record read from {edf.path}")
                        for signal in signal_indexes:
                            offset = edf.signal_offsets[signal]
                            out.write(data[offset : offset + edf.signal_bytes[signal]])
                        out.write(
                            annotation_record(
                                current * edf.record_duration,
                                annotations[current],
                                annotation_samples,
                            )
                        )
                        current += 1

        output_record_bytes = (
            sum(base.signal_bytes[index] for index in base_signals)
            + annotation_samples * 2
        )
        expected_bytes = len(header) + output_record * output_record_bytes
        actual_bytes = temporary.stat().st_size
        if actual_bytes != expected_bytes:
            raise IOError(
                f"Output EDF byte count mismatch for {patient_code}: "
                f"expected {expected_bytes}, found {actual_bytes}"
            )
        os.replace(temporary, out_path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def skipped_row(row: dict, reason: str) -> dict:
    return {
        "reason": reason,
        "patient_code": clean(row.get("patient_code")),
        "patient_id": clean(row.get("patient_id")),
        "source": clean(row.get("source")),
        "source_sheet": clean(row.get("source_sheet")),
        "source_row": clean(row.get("source_row")),
        "event_type": clean(row.get("event_type")),
        "event_number": clean(row.get("event_number")),
        "segment_number": clean(row.get("segment_number")),
        "edf_filename": clean(row.get("edf_filename")),
        "start_seconds": clean(row.get("start_seconds")),
        "end_seconds": clean(row.get("end_seconds")),
    }


def validate_row(row: dict) -> None:
    source_path(row)
    start = seconds(row.get("start_seconds"))
    end = seconds(row.get("end_seconds"))
    if end <= start:
        raise ValueError(f"bad event times: {start} >= {end}")
    event_type = clean(row.get("event_type"))
    if event_type not in {"II", "Ict"}:
        raise ValueError(f"unexpected event type: {event_type}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_snips(database: Path, out_dir: Path, manifest_path: Path, skipped_path: Path) -> tuple[list[dict], list[dict]]:
    with database.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    by_patient = defaultdict(list)
    skipped = []
    for row in rows:
        try:
            validate_row(row)
        except ValueError as error:
            skipped.append(skipped_row(row, str(error)))
            continue
        by_patient[clean(row.get("patient_code")) or clean(row.get("patient_id"))].append(row)

    cache: dict[Path, Edf] = {}
    manifest = []
    for patient_code, patient_rows in sorted(by_patient.items()):
        try:
            manifest.extend(write_patient(patient_code, patient_rows, out_dir, cache))
        except (ValueError, OSError, EOFError) as error:
            skipped.extend(skipped_row(row, str(error)) for row in patient_rows)

    write_csv(manifest_path, manifest)
    write_csv(skipped_path, skipped)
    return manifest, skipped


def write_fixture_edf(path: Path) -> None:
    fields = {
        "labels": ["EEG Fp1-G2", "EDF Annotations"],
        "transducers": ["", ""],
        "physical_dimensions": ["uV", ""],
        "physical_mins": ["-1", "-1"],
        "physical_maxes": ["1", "1"],
        "digital_mins": ["-32768", "-32768"],
        "digital_maxes": ["32767", "32767"],
        "prefilterings": ["", ""],
        "samples_per_record": ["2", str(MIN_ANNOTATION_SAMPLES)],
        "reserved": ["", ""],
    }
    header = [
        fixed("0", 8),
        fixed("fixture", 80),
        fixed("fixture", 80),
        fixed("01.01.24", 8),
        fixed("00.00.00", 8),
        fixed(256 + 256 * 2, 8),
        fixed("EDF+C", 44),
        fixed(10, 8),
        fixed("1", 8),
        fixed(2, 4),
    ]
    for name, width in SIGNAL_FIELDS:
        header.extend(fixed(value, width) for value in fields[name])

    with path.open("wb") as handle:
        handle.write(b"".join(header))
        for record in range(10):
            handle.write(struct.pack("<hh", record, record + 100))
            handle.write(annotation_record(record, [], MIN_ANNOTATION_SAMPLES))


def self_check() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        edf = root / "source.edf"
        write_fixture_edf(edf)
        database = root / "analysis_database.csv"
        with database.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["patient_code", "patient_id", "event_type", "event_number", "segment_number", "source_row", "start_seconds", "end_seconds", "migrated_edf_path", "edf_path"],
            )
            writer.writeheader()
            writer.writerow({"patient_code": "P_test", "patient_id": "1", "event_type": "II", "event_number": "0", "segment_number": "1", "source_row": "2", "start_seconds": "1", "end_seconds": "3", "migrated_edf_path": str(edf)})
            writer.writerow({"patient_code": "P_test", "patient_id": "1", "event_type": "Ict", "event_number": "1", "source_row": "3", "start_seconds": "5", "end_seconds": "6", "migrated_edf_path": str(edf)})
            writer.writerow({"patient_code": "P_test", "patient_id": "1", "event_type": "Ict", "event_number": "2", "source_row": "4", "start_seconds": "8", "end_seconds": "9", "migrated_edf_path": str(edf)})
            writer.writerow({"patient_code": "P_test", "patient_id": "1", "event_type": "Ict", "event_number": "3", "source_row": "5", "start_seconds": "8.5", "end_seconds": "9.5", "migrated_edf_path": str(edf)})
        manifest, skipped = write_snips(database, root / "out", root / "manifest.csv", root / "skipped.csv")
        out = root / "out" / "P_test_snips.edf"
        snipped = read_edf(out)
        assert out.exists()
        assert snipped.records == 11
        assert snipped.fields["labels"] == ["EEG Fp1-G2", "EDF Annotations"]
        assert [row["segment"] for row in manifest] == ["preictal", "ictal", "transictal", "ictal", "postictal"]
        transictal = next(row for row in manifest if row["segment"] == "transictal")
        assert (transictal["source_segment_start_seconds"], transictal["source_segment_end_seconds"]) == (6.0, 8.0)
        assert transictal["transictal_next_event_number"] == "2"
        final_ictal = [row for row in manifest if row["segment"] == "ictal"][-1]
        assert final_ictal["source_segment_end_seconds"] == 9.5
        assert final_ictal["merged_ictal_source_rows"] == "4;5"
        assert final_ictal["merged_ictal_event_numbers"] == "2;3"
        assert subtract_intervals(0, 10, [(2, 4), (6, 8)]) == [(0, 2), (4, 6), (8, 10)]
        assert skipped == []


def main() -> None:
    parser = argparse.ArgumentParser(description="Create one annotated snipped EDF per patient from analysis_database.csv.")
    parser.add_argument("database", nargs="?", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--skipped", type=Path, default=None)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    self_check()
    if args.self_check:
        print("self-check passed")
        return

    config = read_config()
    database = args.database or default_database(config)
    out_dir = args.out_dir or default_output_dir(config)
    manifest_path = args.manifest or out_dir / "snip_keyfile.csv"
    skipped_path = args.skipped or out_dir / "snip_skipped.csv"
    manifest, skipped = write_snips(database, out_dir, manifest_path, skipped_path)
    print(f"Wrote {len({row['output_edf'] for row in manifest})} patient EDFs to {out_dir}")
    print(f"Wrote {manifest_path} ({len(manifest)} segments)")
    if skipped:
        print(f"Skipped {len(skipped)} rows without EDF paths; wrote {skipped_path}")


if __name__ == "__main__":
    main()
