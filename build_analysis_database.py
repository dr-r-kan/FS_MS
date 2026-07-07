from __future__ import annotations

import csv
from datetime import datetime, time, timedelta
import hashlib
from pathlib import Path
import re
import shutil
import warnings

import pandas as pd


OUT_COLUMNS = [
    "source",
    "source_sheet",
    "source_row",
    "patient_id",
    "patient_code",
    "age",
    "gender",
    "init",
    "patient_type",
    "patient_type_raw",
    "event_type",
    "event_number",
    "segment_number",
    "seizure_type",
    "seizure_type_raw",
    "hypo_hyper",
    "file_template",
    "edf_number",
    "file_label",
    "edf_filename",
    "edf_path",
    "migrated_edf_filename",
    "migrated_edf_path",
    "recording_start",
    "event_date",
    "edf_start",
    "edf_duration_seconds",
    "raw_start",
    "raw_end",
    "raw_start_seconds",
    "raw_end_seconds",
    "start",
    "end",
    "start_seconds",
    "end_seconds",
    "duration_seconds",
    "time_basis",
    "time_warning",
    "event_classification",
    "event_score_fs",
    "event_description",
    "notes",
    "junk",
]


def read_config(path: Path) -> dict[str, Path]:
    with path.open(newline="") as handle:
        return {row[0]: Path(row[1]) for row in csv.reader(handle) if len(row) >= 2}


def empty(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    return isinstance(value, str) and not value.strip()


def text(value) -> str:
    if empty(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def patient_id(value) -> str:
    return text(value)


def stable_code(prefix: str, value) -> str:
    raw = text(value).upper()
    if not raw:
        return ""
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:10]}"


def normal_type(value) -> str:
    raw = text(value)
    key = raw.upper()
    if key in {"NEAD", "FS"}:
        return "FS"
    if key in {"EPILEPSY", "EPILEPTIC SEIZURE", "ES"}:
        return "ES"
    if key in {"DD", "DUAL DIAGNOSIS"}:
        return "DD"
    return raw


def normalize_event_type(value) -> str:
    raw = text(value).upper()
    if raw in {"II", "INTERICTAL"}:
        return "II"
    if raw in {"ICT", "ICTAL"}:
        return "Ict"
    return ""


HYPER_CLASS = re.compile(r"\b(?:hyper\s*motor|auto\s*motor|motor|tonic|clonic|myoclon\w*|gelastic|secondary\s+generali[sz]ation)\b", re.I)
HYPO_CLASS = re.compile(r"\b(?:akinetic|atonic|absen[cs]e|behavio\w*\s+arrest|aura|subjective|sensory|psychic)\b", re.I)
HYPER_DESCRIPTION = re.compile(
    r"\b(?:jerk\w*|tremor\w*|twitch\w*|shak\w*|convuls\w*|stiffen\w*|thrash\w*|kick\w*|"
    r"pedall?\w*|dystoni\w*|automatism\w*|auto\s*motor|fidget\w*|fiddl\w*|grimac\w*|tonic|clonic|"
    r"myoclon\w*|movement\w*|postur\w*)\b|lip\s*smack\w*|head\s*turn\w*",
    re.I,
)
HYPO_DESCRIPTION = re.compile(
    r"\b(?:freez\w*|frozen|motionless|immobile|akinetic|atonic|unresponsive|collapse\w*|"
    r"flop\w*|slump\w*|fell|fall\w*|floor|ground|aura|subjective)\b|behavio\w*\s+arrest|"
    r"no\s+(?:obvious\s+)?clinical\s+manifestations|stops?\s+(?:talking|responding)|appears?\s+to\s+stare",
    re.I,
)


def infer_hypo_hyper(existing, classification, description) -> str:
    current = text(existing).lower()
    if current in {"hypo", "hyper"}:
        return current
    classification = text(classification)
    description = text(description)
    if HYPER_CLASS.search(classification):
        return "hyper"
    if HYPO_CLASS.search(classification):
        return "hypo"
    if HYPER_DESCRIPTION.search(description):
        return "hyper"
    if HYPO_DESCRIPTION.search(description):
        return "hypo"
    return ""


def seconds(value):
    if empty(value):
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        value = value.time()
    if isinstance(value, time):
        return value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1_000_000
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, (int, float)):
        if 0 <= float(value) < 1:
            return round(float(value) * 86400, 6)
        return float(value)

    raw = text(value).replace(";", ":")
    if ":" in raw:
        parts = [float(part) for part in raw.split(":")]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
    try:
        return seconds(float(raw))
    except ValueError:
        return None


def hhmmss(value) -> str:
    secs = seconds(value)
    if secs is None:
        return text(value)
    return seconds_to_hhmmss(secs)


def seconds_to_hhmmss(secs) -> str:
    if secs is None:
        return ""
    hours, rest = divmod(int(round(secs)), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def date_text(value) -> str:
    if empty(value):
        return ""
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return text(value)


def duration(start, end):
    if start is None or end is None:
        return None
    diff = end - start
    return diff + 86400 if diff < 0 else diff


def read_edf_header(path: Path) -> dict:
    with path.open("rb") as handle:
        header = handle.read(256)

    def field(start, end) -> str:
        return header[start:end].decode("ascii", "ignore").strip()

    try:
        start = datetime.strptime(f"{field(168, 176)} {field(176, 184)}", "%d.%m.%y %H.%M.%S")
    except ValueError:
        start = None
    try:
        records = int(field(236, 244))
        record_duration = float(field(244, 252))
        total_duration = records * record_duration if records > 0 else None
    except ValueError:
        total_duration = None
    return {"path": str(path), "start": start, "duration": total_duration}


def clock_seconds_from_edf(label_seconds, edf_start):
    if label_seconds is None or edf_start is None:
        return None
    label_day, label_second = divmod(float(label_seconds), 86400)
    label_dt = datetime.combine(edf_start.date(), time()) + timedelta(days=int(label_day), seconds=label_second)
    relative = (label_dt - edf_start).total_seconds()
    if -300 <= relative < 0:
        return 0
    while relative < -300:
        relative += 86400
    return relative


def inside_edf(start, end, edf_duration):
    if edf_duration in ("", None):
        return True
    values = [value for value in (start, end) if value is not None]
    return all(-1 <= value <= float(edf_duration) + 5 for value in values)


def normalise_times(row: dict, edf_info: dict | None) -> dict:
    raw_start = row["raw_start_seconds"]
    raw_end = row["raw_end_seconds"]
    edf_start = edf_info["start"] if edf_info else None
    edf_duration = edf_info["duration"] if edf_info else None
    row["edf_start"] = edf_start.strftime("%Y-%m-%d %H:%M:%S") if edf_start else ""
    row["edf_duration_seconds"] = edf_duration or ""

    if not edf_info:
        row["time_basis"] = "sheet_relative" if row["source"] == "metaeventmap" else "raw_clock_no_edf"
        row["time_warning"] = "" if row["source"] == "metaeventmap" else "missing_edf_header"
        return row

    raw = (raw_start, raw_end)
    clock = (clock_seconds_from_edf(raw_start, edf_start), clock_seconds_from_edf(raw_end, edf_start))
    if row["source"] == "fs_dd_es":
        chosen, fallback, basis = clock, raw, "edf_header_clock"
    else:
        chosen, fallback, basis = raw, clock, "sheet_relative"

    warning = ""
    if not inside_edf(*chosen, edf_duration) and inside_edf(*fallback, edf_duration):
        chosen = fallback
        basis = "sheet_relative_fallback" if row["source"] == "fs_dd_es" else "edf_header_clock_fallback"
        warning = "primary_times_outside_edf"
    elif not inside_edf(*chosen, edf_duration):
        warning = "times_outside_edf"

    start, end = chosen
    if start is not None and end is not None and end < start:
        end += 86400
    row["start_seconds"] = start
    row["end_seconds"] = end
    row["duration_seconds"] = duration(start, end)
    row["start"] = seconds_to_hhmmss(start) or row["raw_start"]
    row["end"] = seconds_to_hhmmss(end) or row["raw_end"]
    row["time_basis"] = basis
    row["time_warning"] = warning
    return row


def first(row, *names):
    for name in names:
        if name in row and not empty(row[name]):
            return row[name]
    return ""


def edf_number(value):
    raw = text(value)
    if not raw:
        return ""
    try:
        return str(int(float(raw)))
    except ValueError:
        return raw


def file_from_template(template, number, init="") -> str:
    number = text(number)
    template = text(template)
    if template and number:
        index = template.rfind("N")
        if index >= 0:
            return f"{template[:index]}{number}{template[index + 1:]}"
        return f"{template}{number}"
    return f"{text(init)}{number}" if number else text(template or init)


def meta_file_label(init, number, junk="") -> str:
    junk = text(junk)
    if junk:
        return junk
    init = re.sub(r"\s*-\s*JUNK(?:EVENT|ENTRY)\s*$", "", text(init), flags=re.IGNORECASE)
    return file_from_template("", number, init)


def build_edf_index(*roots: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    duplicates = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.edf"):
            key = path.stem.lower()
            if key in index:
                duplicates.add(key)
            else:
                index[key] = read_edf_header(path)
    for key in duplicates:
        index.pop(key, None)
    return index


def add_file_fields(row: dict, edf_paths: dict[str, dict]) -> dict:
    label = text(row.get("file_label"))
    number = text(row.get("edf_number"))
    candidates = [label]
    if label and number and not re.search(r"\d$", label):
        candidates.append(f"{label}{number}")
    if row.get("init") and number and (row["source"] == "metaeventmap" or not row.get("file_template")):
        candidates.append(f"{text(row.get('init'))}{number}")
    if re.search(r"\d+$", label):
        candidates.append(re.sub(r"\d+$", "", label))

    match = None
    for candidate in filter(None, candidates):
        if candidate.lower() in edf_paths:
            label = candidate
            match = edf_paths[candidate.lower()]
            break

    row["file_label"] = label
    row["edf_filename"] = f"{label}.edf" if label else ""
    row["edf_path"] = match["path"] if match else ""
    return normalise_times(row, match)


def event_row(**kwargs) -> dict:
    row = {column: "" for column in OUT_COLUMNS}
    row.update(kwargs)
    start = seconds(row["start"])
    end = seconds(row["end"])
    row["raw_start"] = hhmmss(row["start"])
    row["raw_end"] = hhmmss(row["end"])
    row["raw_start_seconds"] = start
    row["raw_end_seconds"] = end
    row["start"] = row["raw_start"]
    row["end"] = row["raw_end"]
    row["start_seconds"] = start
    row["end_seconds"] = end
    row["duration_seconds"] = duration(start, end)
    row["time_basis"] = "raw_label"
    return row


def build_meta(path: Path, edf_paths: dict[str, dict]) -> list[dict]:
    df = pd.read_excel(path, sheet_name="Sheet1")
    rows = []
    for source_row, row in df.iterrows():
        kind = normalize_event_type(row.get("EventType"))
        if not kind:
            continue
        number = edf_number(row.get("EDF"))
        seizure_raw = row.get("SeizureType") if kind == "Ict" else ""
        junk = text(row.get("Junk"))
        out = event_row(
            source="metaeventmap",
            source_sheet="Sheet1",
            source_row=source_row + 2,
            patient_id=patient_id(row.get("num")),
            age=text(row.get("Age")),
            gender=text(row.get("Gender")),
            init=text(row.get("init")),
            patient_type=normal_type(row.get("PatientType")),
            patient_type_raw=text(row.get("PatientType")),
            event_type=kind,
            event_number=text(row.get("EventNumber")),
            seizure_type=normal_type(seizure_raw),
            seizure_type_raw=text(seizure_raw),
            hypo_hyper=text(row.get("hypo/hyper")) if kind == "Ict" else "",
            edf_number=number,
            file_label=meta_file_label(row.get("init"), number, junk),
            start=row.get("EventStart"),
            end=row.get("EventEnd"),
            event_classification="; ".join(filter(None, map(text, [row.get("Semiology1"), row.get("Semiology2"), row.get("Semiology3")]))),
            event_description=text(row.get("Semiology description from report")),
            notes=text(row.get("notes")),
            junk=junk,
        )
        rows.append(add_file_fields(out, edf_paths))
    return rows


def recording_map(row) -> dict[str, str]:
    values = [row.get("II edf")]
    values.extend(row.get(f"Event {n} EDF") for n in range(1, 8))
    seen: dict[str, str] = {}
    for value in values:
        key = date_text(value)
        if key and key not in seen:
            seen[key] = str(len(seen) + 1)
    return seen


def build_fs(path: Path, edf_paths: dict[str, dict]) -> list[dict]:
    events = pd.read_excel(path, sheet_name="Database_events")
    demographics = pd.read_excel(path, sheet_name="Database_demographics")
    demographics["_patient_id"] = demographics["hospital number"].map(patient_id)
    demographics = demographics.drop_duplicates("_patient_id").set_index("_patient_id")
    rows = []

    for source_row, row in events.iterrows():
        patient_type_raw = text(row.get("category; FS, DD, ES"))
        if patient_type_raw not in {"FS", "DD", "ES"}:
            continue
        pid = patient_id(row.get("hospital number"))
        demo = demographics.loc[pid] if pid in demographics.index else {}
        files = recording_map(row)
        template = text(row.get("FileName"))
        base = dict(
            source="fs_dd_es",
            source_sheet="Database_events",
            source_row=source_row + 2,
            patient_id=pid,
            age=text(demo.get("age", "")),
            gender=text(demo.get("gender", "")),
            init=text(row.get("init") or demo.get("init", "")),
            patient_type=normal_type(patient_type_raw),
            patient_type_raw=patient_type_raw,
            file_template=template,
        )

        ii_recording = date_text(row.get("II edf"))
        ii_number = files.get(ii_recording, "")
        for segment in range(1, 5):
            start = row.get(f"II start {segment}")
            end = row.get(f"II end {segment}")
            if empty(start) and empty(end):
                continue
            out = event_row(
                **base,
                event_type="II",
                event_number="0",
                segment_number=str(segment),
                edf_number=ii_number,
                file_label=file_from_template(template, ii_number, base["init"]),
                recording_start=ii_recording,
                start=start,
                end=end,
            )
            rows.append(add_file_fields(out, edf_paths))

        for event in range(1, 8):
            seizure_raw = text(row.get(f"ES/FS Event {event}"))
            if seizure_raw not in {"FS", "ES"}:
                continue
            recording = date_text(row.get(f"Event {event} EDF"))
            number = files.get(recording, "")
            out = event_row(
                **base,
                event_type="Ict",
                event_number=str(event),
                seizure_type=normal_type(seizure_raw),
                seizure_type_raw=seizure_raw,
                edf_number=number,
                file_label=file_from_template(template, number, base["init"]),
                recording_start=recording,
                event_date=date_text(first(row, f"Event {event} date", f"Event {event}  date")),
                start=row.get(f"Event {event} onset HHMMSS"),
                end=row.get(f"Event {event} offset HHMMSS"),
                event_classification=text(row.get(f"Event {event} classification")),
                event_score_fs=text(row.get(f"Event {event} score (FS)")),
                event_description=text(row.get(f"Event {event} description from report")),
            )
            rows.append(add_file_fields(out, edf_paths))
    return rows


def build_database(config_path: Path) -> pd.DataFrame:
    config = read_config(config_path)
    edf_paths = build_edf_index(config["raw_1"], config["raw_2"])
    meta_rows = build_meta(config["map2"], edf_paths)
    fs_rows = build_fs(config["map1"], edf_paths)
    fs_patient_ids = {row["patient_id"] for row in fs_rows if row["patient_id"]}
    rows = fs_rows + [row for row in meta_rows if row["patient_id"] not in fs_patient_ids]
    df = pd.DataFrame(rows, columns=OUT_COLUMNS)
    if df.empty:
        raise RuntimeError("No event rows were compiled.")
    bad_types = set(df["event_type"]) - {"Ict", "II"}
    if bad_types:
        raise RuntimeError(f"Unexpected event types: {sorted(bad_types)}")
    ictal = df["event_type"].eq("Ict")
    df.loc[ictal, "hypo_hyper"] = df.loc[ictal].apply(
        lambda row: infer_hypo_hyper(row["hypo_hyper"], row["event_classification"], row["event_description"]),
        axis=1,
    )
    df["patient_code"] = df["patient_id"].map(lambda value: stable_code("P", value))
    overlap = set(df.loc[df["source"].eq("fs_dd_es"), "patient_id"]) & set(df.loc[df["source"].eq("metaeventmap"), "patient_id"])
    if overlap:
        raise RuntimeError(f"Patients still duplicated across sources: {sorted(overlap)}")
    if df.loc[df["patient_id"].ne("")].drop_duplicates("patient_id")["patient_code"].duplicated().any():
        raise RuntimeError("Patient code collision.")
    return df.sort_values(["source", "patient_id", "file_label", "event_type", "event_number", "segment_number"])


def migrated_filename(source_path: str) -> str:
    digest = hashlib.sha1(str(Path(source_path)).lower().encode("utf-8")).hexdigest()[:12]
    return f"FSMS_{digest}.edf"


def inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def assert_migration_target_safe(edf_dir: Path, raw_roots) -> None:
    for raw_root in raw_roots:
        if inside(edf_dir, Path(raw_root)):
            raise RuntimeError(f"Refusing to write migrated EDFs inside raw EDF folder: {raw_root}")


def join_unique(values) -> str:
    return ";".join(sorted({text(value) for value in values if text(value)}))


def copy_idempotent(source: Path, target: Path) -> str:
    if source.resolve() == target.resolve():
        raise RuntimeError(f"Refusing to overwrite original EDF: {source}")
    source_size = source.stat().st_size
    if target.exists() and target.stat().st_size == source_size:
        return "exists"

    existed = target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f"{target.name}.tmp")
    if temp.exists():
        temp.unlink()
    shutil.copy2(source, temp)
    if temp.stat().st_size != source_size:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"Copy size mismatch: {source} -> {target}")
    temp.replace(target)
    return "replaced" if existed else "copied"


def migrate_edfs(df: pd.DataFrame, edf_dir: Path, raw_roots=()) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert_migration_target_safe(edf_dir, raw_roots)
    df = df.copy()
    df["migrated_edf_filename"] = ""
    df["migrated_edf_path"] = ""

    rows = []
    migrated_by_path: dict[str, tuple[str, str]] = {}
    source_paths = sorted(path for path in df["edf_path"].fillna("").unique() if path)
    for source_path in source_paths:
        source = Path(source_path)
        new_name = migrated_filename(source_path)
        target = edf_dir / new_name
        status = copy_idempotent(source, target)
        migrated_by_path[source_path] = (new_name, str(target))

        subset = df[df["edf_path"].eq(source_path)]
        rows.append(
            {
                "migrated_edf_filename": new_name,
                "migrated_edf_path": str(target),
                "source_edf_path": source_path,
                "source_edf_filename": source.name,
                "source_file_label": source.stem,
                "old_init": join_unique(subset["init"]),
                "old_file_template": join_unique(subset["file_template"]),
                "old_file_label": join_unique(subset["file_label"]),
                "old_edf_filename": join_unique(subset["edf_filename"]),
                "patient_id": join_unique(subset["patient_id"]),
                "patient_code": join_unique(subset["patient_code"]),
                "source": join_unique(subset["source"]),
                "patient_type": join_unique(subset["patient_type"]),
                "edf_start": join_unique(subset["edf_start"]),
                "edf_duration_seconds": join_unique(subset["edf_duration_seconds"]),
                "source_size_bytes": source.stat().st_size,
                "copy_status": status,
            }
        )

    if len({name for name, _ in migrated_by_path.values()}) != len(migrated_by_path):
        raise RuntimeError("Migrated EDF filename collision.")

    for source_path, (new_name, target) in migrated_by_path.items():
        match = df["edf_path"].eq(source_path)
        df.loc[match, "migrated_edf_filename"] = new_name
        df.loc[match, "migrated_edf_path"] = target
    return df, pd.DataFrame(rows).sort_values("migrated_edf_filename")


def build_patient_keyfile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient_id, subset in df.groupby("patient_id", dropna=False):
        rows.append(
            {
                "patient_code": join_unique(subset["patient_code"]),
                "patient_id": text(patient_id),
                "source": join_unique(subset["source"]),
                "patient_type": join_unique(subset["patient_type"]),
                "old_init": join_unique(subset["init"]),
                "old_file_template": join_unique(subset["file_template"]),
                "old_file_label": join_unique(subset["file_label"]),
                "event_rows": len(subset),
            }
        )
    return pd.DataFrame(rows).sort_values("patient_code")


def self_check() -> None:
    edf = {"path": "demo.edf", "start": datetime(2024, 1, 1, 23, 59), "duration": 300}
    fs = normalise_times(event_row(source="fs_dd_es", start="00:01:00", end="00:02:00"), edf)
    meta = normalise_times(event_row(source="metaeventmap", start="00:01:00", end="00:02:00"), edf)
    assert fs["start_seconds"] == 120 and fs["end_seconds"] == 180
    assert meta["start_seconds"] == 60 and meta["end_seconds"] == 120
    assert migrated_filename(r"G:\Chalfont\Example1.edf") == migrated_filename(r"g:\chalfont\example1.edf")
    assert infer_hypo_hyper("", "Focal motor with loss of awareness", "") == "hyper"
    assert infer_hypo_hyper("", "Aura; Subjective Sensory", "") == "hypo"
    assert infer_hypo_hyper("", "", "She froze and remained motionless") == "hypo"
    assert infer_hypo_hyper("", "", "Generalised shaking and limb jerking") == "hyper"
    assert infer_hypo_hyper("hypo", "Tonic-Clonic", "") == "hypo"
    assert infer_hypo_hyper("", "Focal aware", "") == ""


def main() -> None:
    warnings.filterwarnings("ignore", message="Data Validation extension is not supported")
    self_check()
    config_path = Path("directories.txt")
    config = read_config(config_path)
    output_dir = config.get("working_directory", Path.cwd())
    edf_dir = output_dir / "migrated_edfs"

    df = build_database(config_path)
    df, edf_key = migrate_edfs(df, edf_dir, raw_roots=[config["raw_1"], config["raw_2"]])
    patient_key = build_patient_keyfile(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths = [Path("analysis_database.csv")]
    edf_key_paths = [Path("edf_keyfile.csv")]
    patient_key_paths = [Path("patient_keyfile.csv")]
    if output_dir.resolve() != Path.cwd().resolve():
        out_paths.append(output_dir / "analysis_database.csv")
        edf_key_paths.append(output_dir / "edf_keyfile.csv")
        patient_key_paths.append(output_dir / "patient_keyfile.csv")

    for path in out_paths:
        df.to_csv(path, index=False)
    for path in edf_key_paths:
        edf_key.to_csv(path, index=False)
    for path in patient_key_paths:
        patient_key.to_csv(path, index=False)

    print(f"Wrote {out_paths[-1]} ({len(df)} rows)")
    print(f"Wrote {edf_key_paths[-1]} ({len(edf_key)} EDFs)")
    print(f"Wrote {patient_key_paths[-1]} ({len(patient_key)} patients)")
    print(df.groupby(["source", "event_type"]).size().to_string())
    print(edf_key["copy_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
