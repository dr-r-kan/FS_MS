from __future__ import annotations

import argparse
import csv
import faulthandler
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(Path(tempfile.gettempdir()) / "mne_home"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_cache"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba_cache"))

# Native numerical libraries otherwise create their own thread pools.  On
# Windows this both increases peak RAM and makes mixed MKL/OpenMP runtimes more
# likely to fail.  HEPPY_NATIVE_THREADS can be raised deliberately after the
# pipeline is stable and measured.
_NATIVE_THREADS = os.environ.get("HEPPY_NATIVE_THREADS", "1")
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TBB_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_var] = _NATIVE_THREADS
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["MKL_DYNAMIC"] = "FALSE"


def _disable_windows_error_dialogs() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        sem_failcriticalerrors = 0x0001
        sem_nogpfault_errorbox = 0x0002
        sem_noopenfile_errorbox = 0x8000
        ctypes.windll.kernel32.SetErrorMode(
            sem_failcriticalerrors | sem_nogpfault_errorbox | sem_noopenfile_errorbox
        )
    except Exception:
        pass


_disable_windows_error_dialogs()
try:
    faulthandler.enable(all_threads=True)
except Exception:
    pass

REVIEW_COLS = [
    "base", "edf_path", "raw_fif", "raw_fif_keep", "qc_png", "viable",
    "true_rr_s", "true_hr_bpm", "beats_per_gap", "notes", "flip",
]


def clean(value) -> str:
    return "" if value is None else str(value).strip()


def truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return clean(value).lower() in {"true", "1", "yes", "y", "on"}


def optional_positive_float(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    text = clean(value)
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if number > 0 else None


def configured_asr_cutoff(options: dict) -> float | None:
    value = options.get("asr", False)
    numeric = optional_positive_float(value)
    if numeric is not None:
        return numeric
    if truthy(value):
        return optional_positive_float(options.get("asr_cutoff")) or 20.0
    return None


def read_config(path: Path = Path("directories.txt")) -> dict[str, Path]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row[0]: Path(row[1]) for row in csv.reader(handle) if len(row) >= 2 and row[0] and row[1]}


def default_heppy_output() -> Path:
    cfg = read_config()
    return cfg.get("working_directory", Path.cwd()) / "heppy_output"


def read_options(heppy_output: Path, options_path: Path | None = None) -> dict:
    candidates = [options_path] if options_path else [heppy_output / "options_used.json", Path("options_used.json")]
    for path in candidates:
        if path and path.exists():
            return json.loads(path.read_text(encoding="utf-8-sig"))
    return {}


def apply_overrides(options: dict, args: argparse.Namespace) -> dict:
    options = dict(options)
    if args.cfa_mode:
        options["remove_cfa_mode"] = args.cfa_mode
        options["cfa_mode"] = args.cfa_mode
    return options


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for col in REVIEW_COLS:
            row.setdefault(col, "")
    return rows


def _atomic_write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = list(dict.fromkeys(REVIEW_COLS + [key for row in rows for key in row]))
    _atomic_write_csv(path, fieldnames, rows)


def write_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    _atomic_write_csv(path, fieldnames, rows)


def prepare_snips(args: argparse.Namespace) -> None:
    from snip_clips import default_database, default_output_dir, read_config as read_snip_config, write_snips

    config = read_snip_config()
    database = args.database or default_database(config)
    snip_dir = args.snip_dir or default_output_dir(config)
    manifest = snip_dir / "snip_keyfile.csv"
    skipped = snip_dir / "snip_skipped.csv"
    rows, skipped_rows = write_snips(database, snip_dir, manifest, skipped)

    review_csv = args.review_csv or args.heppy_output / "ecg_qc_review.csv"
    old = {clean(row.get("base")): row for row in read_rows(review_csv)} if review_csv.exists() else {}
    bases = sorted({Path(row["output_edf"]).stem for row in rows})
    review_rows = []
    for base in bases:
        row = old.get(base, {})
        row.update({"base": base, "edf_path": str(snip_dir / f"{base}.edf")})
        review_rows.append(row)
    write_rows(review_csv, review_rows)
    args.review_csv = review_csv
    args.snip_dir = snip_dir
    print(f"Prepared {len(bases)} transient patient EDFs; skipped rows={len(skipped_rows)}", flush=True)


def cfa_mode(options: dict) -> str:
    return clean(options.get("remove_cfa_mode") or options.get("cfa_mode") or "remove").lower()


def local_venv_python() -> Path:
    return Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"


def missing_modules(names: list[str]) -> list[str]:
    return [name for name in names if importlib.util.find_spec(name) is None]


def ensure_runtime(options: dict) -> None:
    needed = ["mne", "neurokit2", "numpy", "pandas"]
    if truthy(options.get("pyprep", True)):
        needed.append("pyprep")
    if configured_asr_cutoff(options) is not None:
        needed.append("asrpy")
    if truthy(options.get("ica_iclabel", True)):
        needed.append("mne_icalabel")

    missing = missing_modules(needed)
    if not missing:
        return

    venv = local_venv_python()
    if venv.exists() and os.environ.get("HEPPY_PREPROCESS_REEXEC") != "1" and Path(sys.executable).resolve() != venv.resolve():
        env = os.environ.copy()
        env["HEPPY_PREPROCESS_REEXEC"] = "1"
        raise SystemExit(subprocess.run([str(venv), *sys.argv], env=env).returncode)
    raise SystemExit(f"Missing {', '.join(missing)} in {sys.executable}. Run .\\.venv\\Scripts\\python.exe run_heppy_preprocessing.py")


def runtime_config(options: dict, heppy_output: Path) -> SimpleNamespace:
    filters = options.get("filters") or {}
    montage = options.get("montage") or {}
    spirometry = options.get("spirometry") or {}
    tuning = options.get("preprocessing_tuning") or {}
    line_freqs = filters.get("line_freqs", (50.0, 100.0))
    asr_cutoff = configured_asr_cutoff(options)

    max_chunk = tuning.get("pyprep_max_chunk_size", 8)
    try:
        max_chunk = int(max_chunk) if max_chunk not in (None, "") else None
    except (TypeError, ValueError):
        max_chunk = 8

    return SimpleNamespace(
        output_root=str(heppy_output),
        target_sfreq=options.get("target_sfreq"),
        # Historically this field was used as both a switch and the ASR cutoff.
        # Keep it numeric so True is never interpreted as cutoff=1.
        use_asr=asr_cutoff,
        asr_cutoff=asr_cutoff,
        use_pyprep=truthy(options.get("pyprep", True)),
        remove_cfa_mode=cfa_mode(options),
        remove_cfa=cfa_mode(options) != "keep",
        log_file=str(heppy_output / "logs" / "preprocessing.tsv"),
        ref_chs="eeg",
        reref_chs="eeg",
        high_pass=float(filters.get("hp", 1.0)),
        low_pass=float(filters.get("lp", 100.0)),
        prep_ransac=truthy(options.get("prep_ransac", False)),
        line_freqs=tuple(float(freq) for freq in line_freqs),
        montage_name=clean(montage.get("name") or "standard_1020"),
        rename_to_1020=truthy(montage.get("rename_to_1020", True)),
        use_ica=truthy(options.get("ica_iclabel", True)),
        spirometry_channel=clean(spirometry.get("channel")),
        pyprep_channel_wise=truthy(tuning.get("pyprep_channel_wise", False)),
        pyprep_max_chunk_size=max_chunk,
        pyprep_by_segment=truthy(tuning.get("pyprep_by_segment", False)),
        pyprep_safe_quantile_patch=truthy(tuning.get("pyprep_safe_quantile_patch", False)),
        ica_fit_hz=float(tuning.get("ica_fit_hz", 100.0)),
        iclabel_backend=clean(tuning.get("iclabel_backend") or "auto").lower(),
    )


def import_heppy_preprocessing():
    heppy_dir = Path(__file__).resolve().parent / "external" / "HEPPy"
    sys.path.insert(0, str(heppy_dir))
    from _preprocessing import preprocess_edf, preprocess_edf_both, set_runtime_config  # type: ignore

    return preprocess_edf, preprocess_edf_both, set_runtime_config


def output_paths(row: dict, raw_dir: Path, mode: str) -> dict[str, Path]:
    base = Path(clean(row.get("edf_path")) or clean(row.get("base"))).stem
    if mode == "both":
        return {
            "raw_fif": raw_dir / f"{base}_pp_raw.fif",
            "raw_fif_keep": raw_dir / f"{base}_pp_raw_keepcfa.fif",
        }
    suffix = "" if mode != "keep" else "_keepcfa"
    key = "raw_fif" if mode != "keep" else "raw_fif_keep"
    return {key: raw_dir / f"{base}_pp_raw{suffix}.fif"}


def selected_rows(rows: list[dict], allow_pending: bool, process_pending: bool, process_all: bool) -> list[tuple[int, dict]]:
    if process_all:
        return [(idx, row) for idx, row in enumerate(rows)]
    pending = [row for row in rows if not clean(row.get("viable"))]
    if pending and not (allow_pending or process_pending):
        raise SystemExit(
            f"ECG review still has {len(pending)} pending rows. Finish review first, "
            "or pass --allow-pending to process current yes rows only."
        )
    selected = []
    for idx, row in enumerate(rows):
        value = clean(row.get("viable")).lower()
        if value in {"yes", "y", "true", "1"} or (process_pending and value == ""):
            selected.append((idx, row))
    return selected


def error_bases(summary_path: Path) -> set[str]:
    if not summary_path.exists():
        return set()
    with summary_path.open(newline="", encoding="utf-8-sig") as handle:
        return {clean(row.get("base")) for row in csv.DictReader(handle) if clean(row.get("status")) == "error"}


def process_one(args: argparse.Namespace, row_idx: int, ordinal: int | None = None, total: int | None = None) -> int:
    heppy_output = args.heppy_output
    review_csv = args.review_csv or heppy_output / "ecg_qc_review.csv"
    options = apply_overrides(read_options(heppy_output, args.options), args)
    rows = read_rows(review_csv)
    row = rows[row_idx]

    ensure_runtime(options)
    print("HEPPY_STAGE\timports_begin", flush=True)
    preprocess_edf, preprocess_edf_both, set_runtime_config = import_heppy_preprocessing()
    print("HEPPY_STAGE\timports_complete", flush=True)
    set_runtime_config(runtime_config(options, heppy_output))

    mode = cfa_mode(options)
    raw_dir = heppy_output / "raw_fif"
    log_path = heppy_output / "logs" / "preprocessing.tsv"
    spirometry_dir = heppy_output / "spirometry"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    spirometry_dir.mkdir(parents=True, exist_ok=True)

    ecg_channel = clean((options.get("ecg") or {}).get("channel")) or None
    stim_keep = options.get("stim_keep") or None
    spirometry_channel = clean((options.get("spirometry") or {}).get("channel")) or None

    edf_path = Path(clean(row.get("edf_path")))
    base = clean(row.get("base")) or edf_path.stem
    outs = output_paths(row, raw_dir, mode)
    for col, path in outs.items():
        rows[row_idx][col] = str(path)
    if not getattr(args, "worker_no_review_write", False):
        write_rows(review_csv, rows)

    label = f"[{ordinal}/{total}] " if ordinal and total else ""
    print(f"{label}{base}", flush=True)
    if not edf_path.exists():
        raise FileNotFoundError(edf_path)
    # Do not trust existence alone.  A native crash can leave a truncated FIF.
    # The preprocessing functions validate cached files and replace unreadable
    # ones using an atomic temporary-file rename.
    if mode == "both":
        preprocess_edf_both(
            str(edf_path),
            output_path_remove=str(outs["raw_fif"]),
            output_path_keep=str(outs["raw_fif_keep"]),
            redo=bool(args.redo),
            flip_ecg=truthy(row.get("flip")),
            ecg_channel=ecg_channel,
            stim_keep=stim_keep,
            spirometry_channel=spirometry_channel,
            spirometry_dir=spirometry_dir,
            logging_path=str(log_path),
        )
    else:
        key = "raw_fif_keep" if mode == "keep" else "raw_fif"
        preprocess_edf(
            str(edf_path),
            output_path=str(outs[key]),
            redo=bool(args.redo),
            remove_cfa_override=(mode != "keep"),
            flip_ecg=truthy(row.get("flip")),
            ecg_channel=ecg_channel,
            stim_keep=stim_keep,
            spirometry_channel=spirometry_channel,
            spirometry_dir=spirometry_dir,
            logging_path=str(log_path),
        )
    return 0


def child_command(args: argparse.Namespace, row_idx: int, ordinal: int, total: int) -> list[str]:
    python = local_venv_python() if local_venv_python().exists() else Path(sys.executable)
    cmd = [
        str(python),
        str(Path(__file__).resolve()),
        "--worker-row", str(row_idx),
        "--worker-ordinal", str(ordinal),
        "--worker-total", str(total),
        "--heppy-output", str(args.heppy_output),
    ]
    if args.review_csv:
        cmd += ["--review-csv", str(args.review_csv)]
    if args.options:
        cmd += ["--options", str(args.options)]
    if args.redo:
        cmd.append("--redo")
    if args.cfa_mode:
        cmd += ["--cfa-mode", args.cfa_mode]
    cmd.append("--worker-no-review-write")
    return cmd


def describe_exit_code(code: int) -> str:
    unsigned = code & 0xFFFFFFFF
    names = {
        0xC0000005: "access_violation",
        0xC0000017: "no_memory",
        0xC0000409: "stack_buffer_overrun",
        0xC0000374: "heap_corruption",
    }
    label = names.get(unsigned, "")
    suffix = f" ({label})" if label else ""
    return f"{code} / 0x{unsigned:08X}{suffix}"


def last_stage(log_text: str) -> str:
    for line in reversed(log_text.splitlines()):
        if line.startswith("HEPPY_STAGE\t"):
            return line.split("\t", 1)[1].strip()
    return "unknown"


def run_child(cmd: list[str], log_path: Path, echo: bool = True) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = None
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                if echo:
                    print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return proc.wait()
        except KeyboardInterrupt:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            raise


def process(args: argparse.Namespace) -> int:
    heppy_output = args.heppy_output
    review_csv = args.review_csv or heppy_output / "ecg_qc_review.csv"
    summary_path = args.summary or heppy_output / "logs" / "headless_preprocessing_summary.csv"
    options = apply_overrides(read_options(heppy_output, args.options), args)
    rows = read_rows(review_csv)
    if args.retry_errors:
        bases = error_bases(summary_path)
        jobs = [(idx, row) for idx, row in enumerate(rows) if (clean(row.get("base")) or Path(clean(row.get("edf_path"))).stem) in bases]
    else:
        jobs = selected_rows(rows, args.allow_pending, args.process_pending, args.process_all)
    if args.limit:
        jobs = jobs[: args.limit]
    if not jobs:
        print("No viable ECG rows selected; nothing to preprocess.", flush=True)
        return 0

    mode = cfa_mode(options)
    raw_dir = heppy_output / "raw_fif"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = []
    run_jobs = []
    workers = max(1, int(getattr(args, "workers", 1) or 1))

    for ordinal, (row_idx, row) in enumerate(jobs, start=1):
        edf_path = Path(clean(row.get("edf_path")))
        base = clean(row.get("base")) or edf_path.stem
        outs = output_paths(row, raw_dir, mode)
        for col, path in outs.items():
            rows[row_idx][col] = str(path)

        print(f"[{ordinal}/{len(jobs)}] {base}", flush=True)
        record = {"base": base, "edf_path": str(edf_path), "status": "", "error": ""}
        record.update({col: str(path) for col, path in outs.items()})
        summary.append(record)
        run_jobs.append((ordinal, row_idx, row, outs, record))

    write_rows(review_csv, rows)
    write_summary(summary_path, summary)

    def run_job(job):
        ordinal, row_idx, row, outs, record = job
        edf_path = Path(clean(row.get("edf_path")))
        base = clean(row.get("base")) or edf_path.stem
        try:
            if not edf_path.exists():
                raise FileNotFoundError(edf_path)
            record["status"] = "running"
            row_log = heppy_output / "logs" / "headless_rows" / f"{ordinal:03d}_{base}.log"
            code = run_child(child_command(args, row_idx, ordinal, len(jobs)), row_log, echo=workers == 1)
            if code == 0 and all(path.exists() for path in outs.values()):
                record["status"] = "ok"
                record["error"] = ""
                if (args.snip_first and not args.keep_snips) or args.delete_source_on_success:
                    edf_path.unlink()
                    print(f"  Removed transient snip: {edf_path}", flush=True)
            else:
                log_text = row_log.read_text(encoding="utf-8", errors="replace") if row_log.exists() else ""
                if "No EEG channels remain" in log_text or "All channels would be dropped" in log_text:
                    record["status"] = "skipped_no_eeg"
                    record["error"] = f"no_eeg_channels; log={row_log}"
                else:
                    record["status"] = "error"
                    stage = last_stage(log_text)
                    record["error"] = (
                        f"child_exit={describe_exit_code(code)}; "
                        f"last_stage={stage}; log={row_log}"
                    )
        except KeyboardInterrupt:
            record["status"] = "interrupted"
            record["error"] = "KeyboardInterrupt"
            raise
        except BaseException as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
        return ordinal, base, record

    if workers == 1:
        for job in run_jobs:
            ordinal, base, record = run_job(job)
            if record["error"]:
                prefix = "SKIP" if record["status"] == "skipped_no_eeg" else "ERROR"
                print(f"  {prefix}: {record['error']}", flush=True)
            write_summary(summary_path, summary)
    else:
        print(f"Running {len(run_jobs)} files with {workers} workers; per-row output is in headless_rows logs.", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_job, job) for job in run_jobs]
            for future in as_completed(futures):
                ordinal, base, record = future.result()
                label = "OK" if record["status"] == "ok" else record["status"].upper()
                print(f"[{ordinal}/{len(jobs)}] {base}: {label}", flush=True)
                if record["error"]:
                    print(f"  {record['error']}", flush=True)
                write_summary(summary_path, summary)

    errors = sum(1 for row in summary if row["status"] == "error")
    print(f"Finished {len(summary)} files; errors={errors}; mode={mode}; workers={workers}", flush=True)
    return 1 if errors else 0


def self_check() -> None:
    assert cfa_mode({"cfa_mode": "both"}) == "both"
    assert cfa_mode(apply_overrides({}, argparse.Namespace(cfa_mode="both"))) == "both"
    assert truthy("yes") and truthy(True) and not truthy("no")
    assert configured_asr_cutoff({"asr": True}) == 20.0
    assert configured_asr_cutoff({"asr": 25}) == 25.0
    assert configured_asr_cutoff({"asr": False}) is None
    assert "access_violation" in describe_exit_code(-1073741819)
    assert runtime_config({"pyprep": True}, Path(".")).use_pyprep is True
    assert runtime_config({"pyprep": False}, Path(".")).use_pyprep is False
    assert runtime_config({}, Path(".")).prep_ransac is False
    assert runtime_config({"prep_ransac": True}, Path(".")).prep_ransac is True
    rows = [{"viable": "yes"}, {"viable": "no"}, {"viable": ""}]
    assert len(selected_rows(rows, allow_pending=True, process_pending=False, process_all=False)) == 1
    assert len(selected_rows(rows, allow_pending=False, process_pending=True, process_all=False)) == 2
    assert len(selected_rows(rows, allow_pending=False, process_pending=False, process_all=True)) == 3
    summary_check = Path(tempfile.gettempdir()) / "heppy_summary_self_check.csv"
    summary_check.write_text("base,status\nok_row,ok\nbad_row,error\n", encoding="utf-8")
    try:
        assert error_bases(summary_check) == {"bad_row"}
    finally:
        summary_check.unlink(missing_ok=True)
    log = Path(tempfile.gettempdir()) / "heppy_child_self_check.log"
    assert run_child([sys.executable, "-c", "raise SystemExit(7)"], log) == 7
    print("self-check OK")


def log_fatal(args: argparse.Namespace, exc: BaseException) -> None:
    log_dir = args.heppy_output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "headless_preprocessing_fatal.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now(timezone.utc).isoformat()}\t{type(exc).__name__}: {exc}\n")
        handle.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HEPPy EEG preprocessing from the ECG review CSV.")
    parser.add_argument("--heppy-output", type=Path, default=default_heppy_output())
    parser.add_argument("--review-csv", type=Path)
    parser.add_argument("--options", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Number of EDF child processes to run at once.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cfa-mode", choices=("remove", "keep", "both"), help="Override options_used.json CFA mode.")
    parser.add_argument("--allow-pending", action="store_true", help="Ignore blank ECG review rows, but still process only viable=yes rows.")
    parser.add_argument("--process-pending", action="store_true", help="Treat blank ECG review rows as viable. Use only after external ECG fixing.")
    parser.add_argument("--process-all", action="store_true", help="Process every review row, regardless of viable status.")
    parser.add_argument("--retry-errors", action="store_true", help="Process only rows marked error in the previous summary.")
    parser.add_argument("--snip-first", action="store_true", help="Create patient snips before preprocessing and remove each after its FIF output succeeds.")
    parser.add_argument("--database", type=Path, help="Analysis database used by --snip-first.")
    parser.add_argument("--snip-dir", type=Path, help="Transient EDF directory used by --snip-first.")
    parser.add_argument("--keep-snips", action="store_true", help="Keep EDFs created by --snip-first.")
    parser.add_argument("--delete-source-on-success", action="store_true", help="Delete each input EDF only after all requested FIF outputs are verified.")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--worker-row", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-ordinal", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-total", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-no-review-write", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    try:
        if args.worker_row is not None:
            code = process_one(args, args.worker_row, args.worker_ordinal, args.worker_total)
        else:
            if args.snip_first:
                prepare_snips(args)
            code = process(args)
    except BaseException as exc:
        log_fatal(args, exc)
        raise
    raise SystemExit(code)


if __name__ == "__main__":
    main()
