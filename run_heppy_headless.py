from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from run_heppy_preprocessing import prepare_snips, read_config, read_rows, write_rows


ROOT = Path(__file__).resolve().parent


def path_value(config: dict, name: str, default: Path, config_dir: Path) -> Path:
    path = Path(config.get(name) or default)
    return path if path.is_absolute() else config_dir / path


def run(command: list[str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"completed": []}


def complete_stage(path: Path, state: dict, stage: str) -> None:
    state["completed"] = list(dict.fromkeys([*state.get("completed", []), stage]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)


def all_preprocessed(review_csv: Path, mode: str) -> bool:
    if not review_csv.exists():
        return False
    selected = [row for row in read_rows(review_csv) if str(row.get("viable", "")).lower() == "yes"]
    if not selected:
        return False
    for row in selected:
        if mode in {"remove", "both"} and not Path(row.get("raw_fif") or "missing").exists():
            return False
        if mode in {"keep", "both"} and not Path(row.get("raw_fif_keep") or "missing").exists():
            return False
    return True


def missing_inputs(review_csv: Path, mode: str) -> list[dict]:
    missing = []
    for row in read_rows(review_csv):
        if str(row.get("viable", "")).lower() != "yes" or Path(row.get("edf_path") or "missing").exists():
            continue
        remove_ready = Path(row.get("raw_fif") or "missing").exists()
        keep_ready = Path(row.get("raw_fif_keep") or "missing").exists()
        if (mode in {"remove", "both"} and not remove_ready) or (mode in {"keep", "both"} and not keep_ready):
            missing.append(row)
    return missing


def restore_missing_snips(database: Path, snip_dir: Path, missing: list[dict]) -> None:
    from collections import defaultdict
    from snip_clips import write_patient

    wanted = {row["base"].removesuffix("_snips") for row in missing}
    grouped = defaultdict(list)
    with database.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("patient_code") in wanted:
                grouped[row["patient_code"]].append(row)
    absent = wanted - set(grouped)
    if absent:
        raise RuntimeError(f"Cannot restore missing snips for: {sorted(absent)}")
    cache = {}
    for patient_code in sorted(wanted):
        write_patient(patient_code, grouped[patient_code], snip_dir, cache)
        print(f"Restored transient snip: {patient_code}", flush=True)


def summary_has_errors(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return any(row.get("status") == "error" for row in csv.DictReader(handle))


def compensate_missed_beats(times: np.ndarray, min_bpm: float, max_bpm: float, tolerance: float = 0.20) -> tuple[np.ndarray, int, int]:
    """Split long RR gaps only when they closely match an integer number of local beats."""
    rr = np.diff(np.asarray(times, dtype=float))
    low, high = 60.0 / max_bpm, 60.0 / min_bpm
    plausible = rr[(rr >= low) & (rr <= high)]
    fallback = float(np.median(plausible)) if len(plausible) else float(np.median(rr))
    corrected: list[float] = []
    imputed = excluded = 0
    for index, gap in enumerate(rr):
        nearby = rr[max(0, index - 5):index + 6]
        nearby = nearby[(nearby >= low) & (nearby <= high)]
        expected = float(np.median(nearby)) if len(nearby) else fallback
        count = max(1, int(round(gap / expected))) if expected > 0 else 1
        split = gap / count
        if count > 1 and low <= split <= high and abs(split - expected) <= tolerance * expected:
            corrected.extend([split] * count)
            imputed += count - 1
        elif low <= gap <= high:
            corrected.append(float(gap))
        else:
            excluded += 1
    return np.asarray(corrected), imputed, excluded


def hrv_metrics(rr: np.ndarray, frequency: bool, nonlinear: bool) -> dict:
    if len(rr) < 2:
        return {}
    samples = np.rint(np.r_[0.0, np.cumsum(rr)] * 1000.0).astype(int)
    try:
        import neurokit2 as nk

        peaks = {"ECG_R_Peaks": samples}
        out = nk.hrv_time(peaks, sampling_rate=1000, show=False).iloc[0].to_dict()
        if frequency and len(rr) >= 60:
            out.update(nk.hrv_frequency(peaks, sampling_rate=1000, show=False).iloc[0].to_dict())
        if nonlinear and len(rr) >= 20:
            out.update(nk.hrv_nonlinear(peaks, sampling_rate=1000, show=False).iloc[0].to_dict())
        return out
    except Exception:
        diff = np.diff(rr)
        return {
            "HRV_MeanNN": float(np.mean(rr) * 1000),
            "HRV_SDNN": float(np.std(rr, ddof=1) * 1000),
            "HRV_RMSSD": float(np.sqrt(np.mean(diff**2)) * 1000),
            "HRV_pNN50": float(np.mean(np.abs(diff) > 0.05) * 100),
            "HR_Mean": float(60 / np.mean(rr)),
        }


def compute_hrv(config: dict, review_csv: Path, output: Path) -> None:
    adaptive = config.get("adaptive_rr") or {}
    hrv = config.get("hrv") or {}
    rows = []
    for review in read_rows(review_csv):
        sidecar = output / "ecg_qc" / f"{review['base']}_rpeaks_corrected.csv"
        if not sidecar.exists():
            continue
        with sidecar.open(newline="", encoding="utf-8-sig") as handle:
            times = np.asarray([float(row["rpeak_time_s"]) for row in csv.DictReader(handle)], dtype=float)
        times = np.unique(times[np.isfinite(times)])
        rr, imputed, excluded = compensate_missed_beats(
            times,
            float(adaptive.get("rr_min_bpm", 35)),
            float(adaptive.get("rr_max_bpm", 140)),
            float(hrv.get("missed_beat_tolerance", 0.20)),
        )
        row = {
            "base": review["base"], "n_detected_beats": len(times),
            "n_hrv_intervals": len(rr), "n_imputed_beats": imputed,
            "n_excluded_intervals": excluded,
            "imputed_fraction": imputed / max(len(rr), 1),
            "quality": "review" if imputed / max(len(rr), 1) > float(hrv.get("review_imputed_fraction", 0.05)) or excluded else "ok",
        }
        row.update(hrv_metrics(rr, bool(hrv.get("frequency_domain", True)), bool(hrv.get("nonlinear", False))))
        rows.append(row)
    if rows:
        target = output / "hrv" / "hrv_metrics.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
            writer.writeheader()
            writer.writerows(rows)


def export_epochs(config: dict, review_csv: Path, output: Path) -> None:
    import mne

    from external.HEPPy.ecg_peak_correction import corrected_rpeaks_path, load_corrected_rpeaks

    epoch = config.get("epoch") or {}
    tmin, tmax = float(epoch.get("tmin", -0.2)), float(epoch.get("tmax", 0.6))
    minimum = float(epoch.get("min_duration_s", 0.8))
    if tmax - tmin < minimum:
        raise ValueError(f"Epoch window is {tmax - tmin:.3f}s; minimum is {minimum:.3f}s")
    reject = {"eeg": float(epoch.get("amp_reject_uv", 150)) * 1e-6}
    baseline = epoch.get("baseline")
    baseline = tuple(baseline) if baseline is not None else None
    epoch_dir = output / "epochs"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for row in read_rows(review_csv):
        for variant, value in (("remove", row.get("raw_fif")), ("keep", row.get("raw_fif_keep"))):
            raw_path = Path(value) if value else None
            if not raw_path or not raw_path.exists():
                continue
            raw = None
            try:
                raw = mne.io.read_raw_fif(raw_path, preload=True, verbose="ERROR")
                peaks = load_corrected_rpeaks(corrected_rpeaks_path(output / "ecg_qc", row["base"]), raw.info["sfreq"], raw.n_times)
                if peaks is None or len(peaks) < 3:
                    raise ValueError("missing corrected R peaks")
                events = np.column_stack([peaks + raw.first_samp, np.zeros(len(peaks), int), np.ones(len(peaks), int)])
                epochs = mne.Epochs(raw, events, {"HEP": 1}, tmin=tmin, tmax=tmax, baseline=baseline,
                                    picks="eeg", preload=True, reject=reject, reject_by_annotation=True,
                                    on_missing="ignore", verbose="ERROR")
                name = row["base"] + ("_keepcfa" if variant == "keep" else "")
                target = epoch_dir / f"{name}_epo.fif"
                epochs.save(target, overwrite=True, verbose="ERROR")
                log.append({"base": row["base"], "variant": variant, "status": "ok", "detected": len(peaks),
                            "saved": len(epochs), "dropped": len(peaks) - len(epochs), "file": str(target)})
            except Exception as exc:
                log.append({"base": row["base"], "variant": variant, "status": "error", "reason": f"{type(exc).__name__}: {exc}"})
            finally:
                if raw is not None:
                    raw.close()
    if log:
        with (epoch_dir / "epoch_log.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in log for key in row)))
            writer.writeheader()
            writer.writerows(log)


def self_check() -> None:
    rr, imputed, excluded = compensate_missed_beats(np.array([0, 1, 2, 4, 5], float), 35, 140)
    assert np.allclose(rr, 1.0) and imputed == 1 and excluded == 0
    import mne
    from external.HEPPy.ecg_peak_correction import corrected_rpeaks_path, save_corrected_rpeaks

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = root / "output"
        raw_path = output / "raw_fif" / "fixture_pp_raw.fif"
        raw_path.parent.mkdir(parents=True)
        data = np.zeros((1, 1000))
        data[0, 500] = 500e-6
        raw = mne.io.RawArray(data, mne.create_info(["Cz"], 100, "eeg"), verbose="ERROR")
        raw.save(raw_path, overwrite=True, verbose="ERROR")
        raw.close()
        peaks = np.array([10, 100, 200, 300, 400, 500, 600, 700, 800, 980])
        save_corrected_rpeaks(corrected_rpeaks_path(output / "ecg_qc", "fixture"), peaks, 100)
        review = output / "ecg_qc_review.csv"
        write_rows(review, [{"base": "fixture", "raw_fif": str(raw_path), "raw_fif_keep": ""}])
        export_epochs({"epoch": {"tmin": -0.2, "tmax": 0.6, "amp_reject_uv": 150, "min_duration_s": 0.8}}, review, output)
        assert (output / "epochs" / "fixture_epo.fif").exists()
        saved = mne.read_epochs(output / "epochs" / "fixture_epo.fif", preload=True, verbose="ERROR")
        assert len(saved) < len(peaks)
        del saved
        state_path = root / "state.json"
        state = load_state(state_path)
        complete_stage(state_path, state, "fixture")
        assert "fixture" in load_state(state_path)["completed"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless HEPPy: clip, fit/correct R peaks, HRV, preprocess, and epoch.")
    parser.add_argument("config", nargs="?", type=Path)
    parser.add_argument("--redo", action="store_true", help="Rerun every stage and overwrite cached outputs.")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("self-check passed")
        return
    if not args.config:
        parser.error("config JSON is required")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    config_dir = config_path.parent
    directories = read_config()
    work = directories.get("working_directory", config_dir)
    output = path_value(config, "heppy_output", work / "heppy_output", config_dir)
    review_csv = output / "ecg_qc_review.csv"
    snip_dir = path_value(config, "snip_dir", directories.get("clip_folder", work / "clips"), config_dir)
    database = path_value(config, "database", work / "analysis_database.csv", config_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "logs" / "headless_state.json"
    state = load_state(state_path)

    def skip(stage: str, inferred: bool = False) -> bool:
        done = not args.redo and (stage in state.get("completed", []) or inferred)
        if done:
            print(f"SKIP {stage}: already complete (use --redo to rerun)", flush=True)
            complete_stage(state_path, state, stage)
        return done

    mode = str(config.get("remove_cfa_mode") or config.get("cfa_mode") or "remove").lower()
    manifest = snip_dir / "snip_keyfile.csv"
    print("STAGE 1/7 clipping", flush=True)
    missing = missing_inputs(review_csv, mode) if review_csv.exists() else []
    if not args.redo and manifest.exists() and review_csv.exists() and missing:
        restore_missing_snips(database, snip_dir, missing)
        complete_stage(state_path, state, "clipping")
    elif not skip("clipping", manifest.exists() and review_csv.exists()):
        prepare_snips(SimpleNamespace(database=database, snip_dir=snip_dir, review_csv=review_csv, heppy_output=output))
        complete_stage(state_path, state, "clipping")

    print("STAGE 2-3/7 R-peak fitting and correction", flush=True)
    auto = config.get("auto_qrs") or {}
    report = output / "ecg_qc" / "auto_qrs_report.csv"
    command = [sys.executable, str(ROOT / "fast_ecg_review.py"), "--heppy-output", str(output), "--review-csv", str(review_csv),
               "--auto-fit", "all", "--auto-save", "--auto-mark-viable", "--auto-plot", "--auto-report", str(report),
               "--auto-alpha", str(auto.get("alpha", 0.05)), "--auto-iterations", str(auto.get("iterations", 2)),
               "--auto-min-corr", str(auto.get("min_corr", 0.80)), "--auto-corr-search-s", str(auto.get("corr_search_s", 0.35))]
    if auto.get("from_detected", False):
        command.append("--auto-from-detected")
    if args.redo:
        command.append("--auto-from-detected")
    if not skip("initial_rpeaks", report.exists() and review_csv.exists()):
        run(command)
        complete_stage(state_path, state, "initial_rpeaks")
    errors = {}
    if report.exists():
        with report.open(newline="", encoding="utf-8-sig") as handle:
            errors = {row.get("base", ""): row.get("error", "") for row in csv.DictReader(handle) if row.get("error")}
    review_rows = read_rows(review_csv)
    for row in review_rows:
        row["qc_png"] = str(output / "ecg_qc" / "auto_qrs_plots" / f"{row['base']}_auto_qrs_qc.png")
        error = errors.get(row["base"], "")
        if "No ECG channel found" in error:
            row["viable"] = "no"
            row["notes"] = f"skipped: {error}"
        else:
            row["notes"] = "automatic QRS fit; retrospective review recommended"
    write_rows(review_csv, review_rows)

    print("STAGE 4/7 preprocessing", flush=True)
    command = [sys.executable, str(ROOT / "run_heppy_preprocessing.py"), "--heppy-output", str(output),
               "--review-csv", str(review_csv), "--options", str(config_path), "--workers", str(config.get("workers", 1))]
    if not config.get("keep_snips", False):
        command.append("--delete-source-on-success")
    if args.redo:
        command.append("--redo")
    summary = output / "logs" / "headless_preprocessing_summary.csv"
    if not args.redo and summary_has_errors(summary):
        command.append("--retry-errors")
    if not skip("preprocessing", all_preprocessed(review_csv, mode)):
        run(command)
        complete_stage(state_path, state, "preprocessing")

    print("STAGE 5/7 missed R-peak recovery", flush=True)
    recovery = config.get("rpeak_recovery") or {}
    recovery_report = output / "ecg_qc" / "missed_rpeak_recovery_report.csv"
    recovery_command = [
        sys.executable, str(ROOT / "fast_ecg_review.py"), "--heppy-output", str(output), "--review-csv", str(review_csv),
        "--recover-missed", "all", "--auto-report", str(recovery_report),
        "--recover-iterations", str(recovery.get("iterations", 2)),
        "--recover-max-multiple", str(recovery.get("max_multiple", 4)),
        "--recover-promac-threshold", str(recovery.get("promac_threshold", 0.33)),
        "--auto-min-corr", str(recovery.get("min_template_corr", auto.get("min_corr", 0.80))),
        "--auto-iterations", str(auto.get("iterations", 2)),
        "--auto-alpha", str(auto.get("alpha", 0.05)),
    ]
    if not skip("missed_rpeak_recovery", recovery_report.exists()):
        run(recovery_command)
        complete_stage(state_path, state, "missed_rpeak_recovery")

    print("STAGE 6/7 adaptive HRV", flush=True)
    hrv_path = output / "hrv" / "hrv_metrics.csv"
    if not skip("hrv", hrv_path.exists()):
        compute_hrv(config, review_csv, output)
        complete_stage(state_path, state, "hrv")

    print("STAGE 7/7 epochs", flush=True)
    epoch_log = output / "epochs" / "epoch_log.csv"
    if not skip("epochs", epoch_log.exists()):
        export_epochs(config, review_csv, output)
        complete_stage(state_path, state, "epochs")
    print(f"Complete: raw FIFs={output / 'raw_fif'} epochs={output / 'epochs'} ECG review={review_csv}", flush=True)


if __name__ == "__main__":
    main()
