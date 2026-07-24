from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(Path(tempfile.gettempdir()) / "mne_home"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_cache"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba_cache"))

ECG_TOKENS = ("ECG", "EKG", "CARD", "EXG")
REVIEW_COLS = [
    "base", "edf_path", "raw_fif", "raw_fif_keep", "qc_png", "viable",
    "true_rr_s", "true_hr_bpm", "beats_per_gap", "notes", "flip",
]


def clean(value) -> str:
    return "" if value is None else str(value).strip()


def local_venv_python() -> Path:
    root = Path(__file__).resolve().parent
    return root / ".venv" / "Scripts" / "python.exe"


def missing_modules(names: list[str]) -> list[str]:
    return [name for name in names if importlib.util.find_spec(name) is None]


def ensure_runtime(qrs_mode: bool) -> None:
    needed = ["matplotlib"] + (["mne", "neurokit2"] if qrs_mode else [])
    missing = missing_modules(needed)
    if not missing:
        return
    venv = local_venv_python()
    if venv.exists() and os.environ.get("FAST_ECG_REVIEW_REEXEC") != "1" and Path(sys.executable).resolve() != venv.resolve():
        env = os.environ.copy()
        env["FAST_ECG_REVIEW_REEXEC"] = "1"
        raise SystemExit(subprocess.run([str(venv), *sys.argv], env=env).returncode)
    raise SystemExit(f"Missing {', '.join(missing)} in {sys.executable}. Run .\\.venv\\Scripts\\python.exe fast_ecg_review.py")


def read_config(path: Path = Path("directories.txt")) -> dict[str, Path]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row[0]: Path(row[1]) for row in csv.reader(handle) if len(row) >= 2 and row[0] and row[1]}


def default_heppy_output() -> Path:
    cfg = read_config()
    return cfg.get("working_directory", Path.cwd()) / "heppy_output"


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for col in REVIEW_COLS:
            row.setdefault(col, "")
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = list(dict.fromkeys(REVIEW_COLS + [k for row in rows for k in row]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_dicts(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_png(row: dict, heppy_output: Path) -> Path:
    qc_png = clean(row.get("qc_png"))
    if qc_png:
        path = Path(qc_png)
        if path.exists():
            return path
    return heppy_output / "ecg_qc" / f"{clean(row.get('base'))}_ecg_qc.png"


def truthy(value) -> bool:
    return clean(value).lower() in {"true", "1", "yes", "y"}


def same_channel(a: str, b: str) -> bool:
    return str(a).casefold() == str(b).casefold() or canonical_channel(a).casefold() == canonical_channel(b).casefold()


def canonical_channel(ch: str) -> str:
    name = re.sub(r"^\s*EEG\s+", "", str(ch).strip(), flags=re.IGNORECASE)
    parts = [p.strip() for p in name.split("-") if p.strip()]
    if len(parts) > 1:
        name = parts[1] if parts[0].isdigit() else parts[0]
    return name.upper().replace("FP", "Fp").replace("Z", "z")


def read_options(heppy_output: Path) -> dict:
    for path in (heppy_output / "options_used.json", Path("options_used.json")):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def ecg_settings(heppy_output: Path) -> dict:
    opts = read_options(heppy_output)
    return {
        "channel": clean((opts.get("ecg") or {}).get("channel")),
        "hp": float((opts.get("filters") or {}).get("hp", 0.5)),
        "lp": float((opts.get("filters") or {}).get("lp", 45.0)),
        "rr_min_bpm": float((opts.get("adaptive_rr") or {}).get("rr_min_bpm", 35.0)),
        "rr_max_bpm": float((opts.get("adaptive_rr") or {}).get("rr_max_bpm", 140.0)),
        "rr_outlier_mad": float((opts.get("adaptive_rr") or {}).get("rr_outlier_mad", 3.5)),
    }


def source_path(row: dict) -> Path | None:
    for col in ("raw_fif", "raw_fif_keep", "edf_path"):
        value = clean(row.get(col))
        if value and Path(value).exists():
            return Path(value)
    return None


def read_raw_any(path: Path, preload=False):
    import mne

    suffix = path.suffix.lower()
    if suffix == ".bdf":
        return mne.io.read_raw_bdf(str(path), preload=preload, verbose="ERROR")
    if suffix == ".edf":
        return mne.io.read_raw_edf(str(path), preload=preload, verbose="ERROR")
    if suffix == ".fif":
        return mne.io.read_raw_fif(str(path), preload=preload, verbose="ERROR")
    if suffix == ".vhdr":
        return mne.io.read_raw_brainvision(str(path), preload=preload, verbose="ERROR")
    raise ValueError(f"Unsupported file type: {path}")


def pick_ecg(raw, preferred: str = "") -> str:
    import mne

    if preferred:
        for ch in raw.ch_names:
            if same_channel(ch, preferred):
                return ch
    idx = mne.pick_types(raw.info, ecg=True)
    if len(idx):
        return raw.ch_names[idx[0]]
    for ch in raw.ch_names:
        if any(token in ch.upper() for token in ECG_TOKENS):
            return ch
    raise RuntimeError("No ECG channel found; pass --ecg-channel.")


def detect_rpeaks(signal: np.ndarray, sfreq: float, settings: dict) -> np.ndarray:
    import neurokit2 as nk

    _, info = nk.ecg_process(ecg_signal=signal, sampling_rate=sfreq)
    return np.unique(np.asarray(info["ECG_R_Peaks"], dtype=int))


def adaptive_fix(rpeaks: np.ndarray, sfreq: float, settings: dict) -> np.ndarray:
    if rpeaks.size < 3:
        return rpeaks
    rr = np.diff(rpeaks) / float(sfreq)
    bpm = 60.0 / rr
    ok = (bpm >= settings["rr_min_bpm"]) & (bpm <= settings["rr_max_bpm"])
    rr_med = float(np.median(rr))
    mad = float(np.median(np.abs(rr - rr_med))) + 1e-12
    ok &= np.abs(0.6745 * (rr - rr_med) / mad) <= settings["rr_outlier_mad"]
    kept = [int(rpeaks[0])]
    for i, good in enumerate(ok):
        if not good:
            gap = (int(rpeaks[i + 1]) - kept[-1]) / float(sfreq)
            for _ in range(max(1, int(round(gap / rr_med)) - 1)):
                kept.append(int(round(kept[-1] + rr_med * sfreq)))
        kept.append(int(rpeaks[i + 1]))
    return np.unique(np.asarray(kept, dtype=int))


def missing_peak_estimates(rpeaks: np.ndarray, sfreq: float, settings: dict, max_multiple: int = 4) -> tuple[list[int], int]:
    rpeaks = np.asarray(rpeaks, dtype=int)
    if len(rpeaks) < 3:
        return [], 0
    rr = np.diff(rpeaks) / float(sfreq)
    low = 60.0 / float(settings.get("rr_max_bpm", 140.0))
    high = 60.0 / float(settings.get("rr_min_bpm", 35.0))
    usual_values = rr[(rr >= low) & (rr <= high)]
    if not len(usual_values):
        return [], 0
    usual = float(np.median(usual_values))
    estimates = []
    uncertain = 0
    for left, right in zip(rpeaks, rpeaks[1:]):
        gap = (right - left) / float(sfreq)
        multiple = int(round(gap / usual))
        if multiple < 2:
            continue
        if multiple > max_multiple or abs(gap / multiple - usual) > 0.20 * usual:
            uncertain += 1
            continue
        estimates.extend(int(round(left + step * (right - left) / multiple)) for step in range(1, multiple))
    return estimates, uncertain


def recover_missing_rpeaks(
    signal: np.ndarray,
    sfreq: float,
    rpeaks: np.ndarray,
    settings: dict,
    min_corr: float = 0.80,
    iterations: int = 2,
    promac_threshold: float = 0.33,
    max_multiple: int = 4,
) -> tuple[np.ndarray, dict]:
    import neurokit2 as nk

    cleaned = clean_ecg_signal(signal, sfreq)
    promac = np.asarray(nk.ecg_findpeaks(
        cleaned,
        sampling_rate=sfreq,
        method="promac",
        promac_methods=["neurokit", "elgendi", "kalidas"],
        threshold=promac_threshold,
    )["ECG_R_Peaks"], dtype=int)
    current = np.unique(np.asarray(rpeaks, dtype=int))
    stats = {"estimated": 0, "recovered": 0, "no_promac_candidate": 0, "template_rejected": 0, "uncertain_gaps": 0}
    min_rr = int(round((60.0 / float(settings.get("rr_max_bpm", 140.0))) * sfreq))

    for _ in range(max(1, int(iterations))):
        estimates, uncertain = missing_peak_estimates(current, sfreq, settings, max_multiple=max_multiple)
        stats["uncertain_gaps"] += uncertain
        if not estimates:
            break
        stats["estimated"] += len(estimates)
        rr = np.diff(current) / sfreq
        low, high = 60.0 / float(settings.get("rr_max_bpm", 140.0)), 60.0 / float(settings.get("rr_min_bpm", 35.0))
        usual = float(np.median(rr[(rr >= low) & (rr <= high)]))
        search = max(1, int(round(0.20 * usual * sfreq)))
        _, waves, _ = qrs_waves_from_cleaned(cleaned, sfreq, current, -0.2, 0.2)
        template = np.nanmean(normalise_waves(waves), axis=0)
        accepted = []
        for estimate in estimates:
            candidates = promac[np.abs(promac - estimate) <= search]
            candidates = candidates[[np.all(np.abs(current - candidate) >= min_rr) for candidate in candidates]]
            if not len(candidates):
                stats["no_promac_candidate"] += 1
                continue
            refined = [best_template_alignment(cleaned, sfreq, int(candidate), template, min(0.04, 0.20 * usual)) for candidate in candidates]
            peak, corr = max(refined, key=lambda item: item[1] if np.isfinite(item[1]) else -np.inf)
            if not np.isfinite(corr) or corr < min_corr:
                stats["template_rejected"] += 1
                continue
            if np.all(np.abs(np.r_[current, accepted] - peak) >= min_rr):
                accepted.append(peak)
                stats["recovered"] += 1
        if not accepted:
            break
        current = np.unique(np.r_[current, accepted]).astype(int)
    return current, stats


def rpeak_qc_score(rpeaks: np.ndarray, sfreq: float, n_samples: int, settings: dict) -> dict:
    """Summarise physiologically usable 60-s coverage without changing peaks."""
    peaks = np.unique(np.asarray(rpeaks, dtype=int)) / float(sfreq)
    duration = n_samples / float(sfreq)
    total_windows = max(1, int(np.floor(duration / 60.0)))
    valid_windows = 0
    if len(peaks) >= 3:
        rr = np.diff(peaks)
        bpm = 60.0 / rr
        mids = (peaks[:-1] + peaks[1:]) / 2
        for index in range(total_windows):
            lo, hi = index * 60.0, (index + 1) * 60.0
            use = (mids >= lo) & (mids < hi)
            if not np.any(use):
                continue
            coverage = np.maximum(0, np.minimum(peaks[1:], hi) - np.maximum(peaks[:-1], lo)).sum() / 60.0
            if (coverage >= .80 and np.max(rr[use]) <= 5.0 and
                    np.all((bpm[use] >= settings["rr_min_bpm"]) & (bpm[use] <= settings["rr_max_bpm"]))):
                valid_windows += 1
    return {
        "peak_count": int(len(peaks)),
        "first_peak_s": float(peaks[0]) if len(peaks) else np.nan,
        "last_peak_s": float(peaks[-1]) if len(peaks) else np.nan,
        "duration_s": duration,
        "valid_60s_windows": valid_windows,
        "total_60s_windows": total_windows,
        "valid_60s_fraction": valid_windows / total_windows,
    }


def write_recovery_sidecar(path: Path, rpeaks: np.ndarray, sfreq: float,
                           detected: np.ndarray, reviewed: np.ndarray, tolerance_s: float = .05) -> dict:
    """Write peak times plus their auditable source; only gap-added peaks are recovered."""
    tolerance = max(1, int(round(tolerance_s * sfreq)))
    detected = np.asarray(detected, dtype=int)
    reviewed = np.asarray(reviewed, dtype=int)
    rows, counts = [], {"reviewed": 0, "full_record_detector": 0, "gap_recovered": 0}
    for peak in np.unique(np.asarray(rpeaks, dtype=int)):
        near_reviewed = len(reviewed) and np.min(np.abs(reviewed - peak)) <= tolerance
        near_detected = len(detected) and np.min(np.abs(detected - peak)) <= tolerance
        source = "reviewed" if near_reviewed else ("full_record_detector" if near_detected else "gap_recovered")
        counts[source] += 1
        rows.append({"rpeak_time_s": peak / float(sfreq), "rpeak_source": source,
                     "is_gap_recovered": int(source == "gap_recovered")})
    write_dicts(path, rows)
    return counts


def heppy_peak_module():
    sys.path.insert(0, str(Path(__file__).resolve().parent / "external" / "HEPPy"))
    from ecg_peak_correction import (  # type: ignore
        adaptive_refit_excluded_rpeaks,
        corrected_rpeaks_path,
        delete_excluded_rpeaks,
        load_corrected_rpeaks,
        save_corrected_rpeaks,
    )

    return corrected_rpeaks_path, load_corrected_rpeaks, save_corrected_rpeaks, delete_excluded_rpeaks, adaptive_refit_excluded_rpeaks


def clean_ecg_signal(signal: np.ndarray, sfreq: float) -> np.ndarray:
    try:
        import neurokit2 as nk

        return np.asarray(nk.ecg_clean(signal, sampling_rate=sfreq), dtype=float)
    except Exception:
        return np.asarray(signal, dtype=float)


def qrs_waves_from_cleaned(cleaned: np.ndarray, sfreq: float, rpeaks: np.ndarray, left_s: float = -0.2, right_s: float = 0.35) -> tuple[np.ndarray, np.ndarray, list[int]]:
    left = int(round(left_s * sfreq))
    right = int(round(right_s * sfreq))
    x = np.arange(left, right) / float(sfreq)
    waves = []
    kept = []
    for idx, peak in enumerate(np.asarray(rpeaks, dtype=int)):
        start, stop = int(peak) + left, int(peak) + right
        if 0 <= start and stop <= len(cleaned):
            waves.append(cleaned[start:stop])
            kept.append(idx)
    return x, np.asarray(waves, dtype=float), kept


def qrs_waves(signal: np.ndarray, sfreq: float, rpeaks: np.ndarray, left_s: float = -0.2, right_s: float = 0.35) -> tuple[np.ndarray, np.ndarray, list[int]]:
    return qrs_waves_from_cleaned(clean_ecg_signal(signal, sfreq), sfreq, rpeaks, left_s, right_s)


def normalise_waves(waves: np.ndarray) -> np.ndarray:
    waves = np.asarray(waves, dtype=float)
    centred = waves - np.nanmedian(waves, axis=1, keepdims=True)
    scale = np.nanmax(np.abs(centred), axis=1, keepdims=True)
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    return centred / scale


def outlier_beats(waves: np.ndarray, kept_indices: list[int], alpha: float) -> tuple[list[int], np.ndarray]:
    if len(waves) < 10:
        return [], np.zeros(len(waves), dtype=float)
    norm = normalise_waves(waves)
    lo = np.nanpercentile(norm, 100 * alpha / 2.0, axis=0)
    hi = np.nanpercentile(norm, 100 * (1.0 - alpha / 2.0), axis=0)
    outside_fraction = np.mean((norm < lo) | (norm > hi), axis=1)
    zero_idx = norm.shape[1] // 2
    centre = norm[:, zero_idx]
    mean_centre = float(np.nanmean(centre))
    centre_bad = np.zeros(len(norm), dtype=bool)
    if np.isfinite(mean_centre) and abs(mean_centre) > 1e-9:
        centre_bad = (np.abs(centre) < 0.20 * abs(mean_centre)) | ((centre * mean_centre) < 0)
    bad_mask = (outside_fraction > alpha) | centre_bad
    bad = [kept_indices[i] for i in np.flatnonzero(bad_mask)]
    return bad, outside_fraction


def template_correlations(norm: np.ndarray, template: np.ndarray) -> np.ndarray:
    norm = np.asarray(norm, dtype=float)
    template = np.asarray(template, dtype=float)
    finite = np.isfinite(template)
    if norm.ndim != 2 or not finite.any():
        return np.full(len(norm), np.nan)
    x = norm[:, finite]
    t = template[finite] - np.nanmean(template[finite])
    x = x - np.nanmean(x, axis=1, keepdims=True)
    denom = np.sqrt(np.nansum(x * x, axis=1) * np.nansum(t * t))
    out = np.full(len(norm), np.nan)
    ok = np.isfinite(denom) & (denom > 0)
    out[ok] = np.nansum(x[ok] * t, axis=1) / denom[ok]
    return out


def best_template_alignment(cleaned: np.ndarray, sfreq: float, peak: int, template: np.ndarray, search_s: float) -> tuple[int, float]:
    left = int(round(-0.2 * sfreq))
    right = int(round(0.2 * sfreq))
    win = max(1, int(round(float(search_s) * sfreq)))
    step = max(1, int(round(0.004 * sfreq)))
    candidates = list(range(int(peak) - win, int(peak) + win + 1, step))
    if int(peak) not in candidates:
        candidates.append(int(peak))
    candidates = sorted(set(c for c in candidates if 0 <= c + left and c + right <= len(cleaned)))
    if not candidates:
        return int(peak), np.nan
    waves = np.asarray([cleaned[c + left:c + right] for c in candidates], dtype=float)
    corr = template_correlations(normalise_waves(waves), template)
    if not np.isfinite(corr).any():
        return int(peak), np.nan
    best = int(np.nanargmax(corr))
    return int(candidates[best]), float(corr[best])


def align_flagged_beats(
    signal: np.ndarray,
    sfreq: float,
    rpeaks: np.ndarray,
    peak_ids: np.ndarray,
    flagged: list[int],
    handled_ids: set[int],
    min_corr: float,
    search_s: float,
    settings: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if not flagged:
        return np.asarray(rpeaks, dtype=int), np.asarray(peak_ids, dtype=int), {"aligned": 0, "corr_failed": 0, "repeat_dropped": 0, "duplicate_dropped": 0}

    cleaned = clean_ecg_signal(signal, sfreq)
    _, waves, kept = qrs_waves_from_cleaned(cleaned, sfreq, rpeaks, -0.2, 0.2)
    norm = normalise_waves(waves)
    flagged_set = set(int(i) for i in flagged)
    good_rows = [i for i, peak_idx in enumerate(kept) if peak_idx not in flagged_set]
    if len(good_rows) < 10:
        good_rows = list(range(len(norm)))
    template = np.nanmean(norm[good_rows], axis=0) if good_rows else np.zeros(norm.shape[1] if norm.ndim == 2 else 0)

    out = np.asarray(rpeaks, dtype=int).copy()
    ids = np.asarray(peak_ids, dtype=int).copy()
    drop: set[int] = set()
    stats = {"aligned": 0, "corr_failed": 0, "repeat_dropped": 0, "duplicate_dropped": 0}
    min_rr = int(round((60.0 / float(settings.get("rr_max_bpm", 140.0))) * float(sfreq)))

    for idx in sorted(flagged_set):
        if idx < 0 or idx >= len(out):
            continue
        peak_id = int(ids[idx])
        if peak_id in handled_ids:
            drop.add(idx)
            stats["repeat_dropped"] += 1
            continue

        best_peak, best_corr = best_template_alignment(cleaned, sfreq, int(out[idx]), template, search_s)
        handled_ids.add(peak_id)
        if not np.isfinite(best_corr) or best_corr < float(min_corr):
            drop.add(idx)
            stats["corr_failed"] += 1
            continue

        neighbours = np.flatnonzero(np.abs(out - best_peak) < min_rr)
        neighbours = [int(i) for i in neighbours if int(i) != idx and int(i) not in drop]
        if neighbours:
            drop.add(idx)
            stats["duplicate_dropped"] += 1
            continue

        if int(best_peak) != int(out[idx]):
            stats["aligned"] += 1
        out[idx] = int(best_peak)

    keep = np.ones(len(out), dtype=bool)
    if drop:
        keep[list(drop)] = False
    out = out[keep]
    ids = ids[keep]
    order = np.argsort(out)
    return out[order], ids[order], stats


def save_auto_qc_plot(path: Path, signal: np.ndarray, sfreq: float, rpeaks: np.ndarray, alpha: float, title: str) -> None:
    import matplotlib.pyplot as plt

    x, waves, _ = qrs_waves(signal, sfreq, rpeaks, -0.2, 0.2)
    norm = normalise_waves(waves)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(norm):
        if len(norm) > 300:
            rng = np.random.default_rng(1)
            draw = norm[rng.choice(len(norm), size=300, replace=False)]
        else:
            draw = norm
        ax.plot(x, draw.T, color="0.25", alpha=0.04, linewidth=0.5)
        lo = np.nanpercentile(norm, 100 * alpha / 2.0, axis=0)
        hi = np.nanpercentile(norm, 100 * (1.0 - alpha / 2.0), axis=0)
        ax.fill_between(x, lo, hi, color="tab:blue", alpha=0.20, label=f"{int((1-alpha)*100)}% envelope")
        ax.plot(x, np.nanmean(norm, axis=0), color="crimson", linewidth=2, label="mean QRS")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Seconds from R peak")
    ax.set_ylabel("Normalised ECG")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def auto_fit_row(
    review_csv: Path,
    heppy_output: Path,
    row: dict,
    settings: dict,
    alpha: float,
    iterations: int,
    save: bool,
    plot: bool,
    plot_dir: Path,
    mark_viable: bool,
    min_corr: float,
    corr_search_s: float,
    from_detected: bool,
) -> dict:
    editor = QrsEditor(review_csv, heppy_output, row, settings)
    editor.load(use_saved=not from_detected)
    original = editor.rpeaks.copy()
    _, _, save_corrected, delete_excluded, _ = heppy_peak_module()

    history = []
    align_history = []
    current = original
    peak_ids = np.arange(len(current), dtype=int)
    handled_ids: set[int] = set()
    for i in range(max(0, int(iterations))):
        _, waves, kept = qrs_waves(editor.signal, editor.sfreq, current, -0.2, 0.2)
        bad, outside = outlier_beats(waves, kept, alpha)
        history.append(len(bad))
        if not bad:
            break
        current, peak_ids, stats = align_flagged_beats(
            editor.signal, editor.sfreq, current, peak_ids, bad, handled_ids, min_corr, corr_search_s, settings
        )
        align_history.append(stats)

    _, waves, kept = qrs_waves(editor.signal, editor.sfreq, current, -0.2, 0.2)
    final_bad, outside = outlier_beats(waves, kept, alpha)
    fixed = delete_excluded(current, final_bad)
    plot_path = plot_dir / f"{clean(row.get('base'))}_auto_qrs_qc.png"
    if save:
        save_corrected(editor.sidecar, fixed, editor.sfreq)
        if mark_viable:
            editor.update_viable("yes")
    if plot:
        save_auto_qc_plot(
            plot_path,
            editor.signal,
            editor.sfreq,
            fixed,
            alpha,
            f"{row.get('base')} | final peaks={len(fixed)} | dropped={len(final_bad)} ({100.0 * len(final_bad) / max(len(current), 1):.2f}%)",
        )

    return {
        "base": clean(row.get("base")),
        "source": editor.source_name,
        "channel": editor.channel,
        "alpha": alpha,
        "iterations": iterations,
        "min_corr": min_corr,
        "corr_search_s": corr_search_s,
        "from_detected": bool(from_detected),
        "saved": bool(save),
        "marked_viable": bool(save and mark_viable),
        "original_peaks": int(len(original)),
        "after_refit_peaks": int(len(current)),
        "after_align_peaks": int(len(current)),
        "final_peaks": int(len(fixed)),
        "refit_flagged_iter1": int(history[0]) if len(history) > 0 else 0,
        "refit_flagged_iter2": int(history[1]) if len(history) > 1 else 0,
        "aligned_iter1": int(align_history[0]["aligned"]) if len(align_history) > 0 else 0,
        "aligned_iter2": int(align_history[1]["aligned"]) if len(align_history) > 1 else 0,
        "corr_failed_iter1": int(align_history[0]["corr_failed"]) if len(align_history) > 0 else 0,
        "corr_failed_iter2": int(align_history[1]["corr_failed"]) if len(align_history) > 1 else 0,
        "repeat_dropped_iter1": int(align_history[0]["repeat_dropped"]) if len(align_history) > 0 else 0,
        "repeat_dropped_iter2": int(align_history[1]["repeat_dropped"]) if len(align_history) > 1 else 0,
        "duplicate_dropped_iter1": int(align_history[0]["duplicate_dropped"]) if len(align_history) > 0 else 0,
        "duplicate_dropped_iter2": int(align_history[1]["duplicate_dropped"]) if len(align_history) > 1 else 0,
        "dropped_after_refit": int(len(final_bad)),
        "drop_percent": round(100.0 * len(final_bad) / max(len(current), 1), 3),
        "median_outside_fraction": round(float(np.nanmedian(outside)) if len(outside) else 0.0, 4),
        "max_outside_fraction": round(float(np.nanmax(outside)) if len(outside) else 0.0, 4),
        "sidecar": str(editor.sidecar),
        "plot": str(plot_path) if plot else "",
    }


class PngReviewer:
    def __init__(self, review_csv: Path, heppy_output: Path, start: int = 0):
        self.review_csv = review_csv
        self.heppy_output = heppy_output
        self.rows = read_rows(review_csv)
        blank = next((i for i, row in enumerate(self.rows) if not clean(row.get("viable"))), 0)
        self.index = max(0, min(start if start else blank, len(self.rows) - 1))

    def save(self) -> None:
        write_rows(self.review_csv, self.rows)

    def show(self) -> None:
        import matplotlib.pyplot as plt

        self.fig, self.ax = plt.subplots(figsize=(13, 9))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()
        plt.show()

    def draw(self) -> None:
        import matplotlib.pyplot as plt

        self.ax.clear()
        row = self.rows[self.index]
        png = row_png(row, self.heppy_output)
        if png.exists():
            self.ax.imshow(plt.imread(png))
            self.ax.axis("off")
        else:
            self.ax.text(0.5, 0.5, f"Missing PNG:\n{png}", ha="center", va="center", transform=self.ax.transAxes)
        done = sum(1 for row in self.rows if clean(row.get("viable")))
        title = (
            f"{self.index + 1}/{len(self.rows)} {row.get('base')} | viable={row.get('viable') or '-'} "
            f"flip={row.get('flip')} | done={done}\n"
            "keys: y yes, n no, f flip, right/space next, left/b prev, q quit"
        )
        self.ax.set_title(title)
        self.fig.canvas.draw_idle()

    def mark(self, value: str) -> None:
        self.rows[self.index]["viable"] = value
        self.save()
        self.step(1)

    def step(self, delta: int) -> None:
        self.index = max(0, min(self.index + delta, len(self.rows) - 1))
        self.draw()

    def on_key(self, event) -> None:
        key = event.key
        if key == "y":
            self.mark("yes")
        elif key == "n":
            self.mark("no")
        elif key == "f":
            row = self.rows[self.index]
            row["flip"] = "False" if truthy(row.get("flip")) else "True"
            self.save()
            self.draw()
        elif key in {"right", " "}:
            self.step(1)
        elif key in {"left", "b"}:
            self.step(-1)
        elif key == "q":
            import matplotlib.pyplot as plt

            self.save()
            plt.close(self.fig)


class QrsEditor:
    def __init__(self, review_csv: Path, heppy_output: Path, row: dict, settings: dict):
        self.review_csv = review_csv
        self.heppy_output = heppy_output
        self.row = row
        self.settings = settings
        self.selected: set[int] = set()

    def load(self, use_saved: bool = True) -> None:
        import mne

        corrected_path, load_corrected, *_ = heppy_peak_module()
        src = source_path(self.row)
        if src is None:
            raise FileNotFoundError("No EDF/FIF source path exists for this row.")
        raw = read_raw_any(src, preload=False)
        ch = pick_ecg(raw, self.settings["channel"])
        raw.pick([ch])
        raw.load_data()
        if truthy(self.row.get("flip")):
            raw._data[0] *= -1
        # ECG channels are often typed as misc; MNE's default data picks then select nothing.
        raw.filter(l_freq=self.settings["hp"], h_freq=self.settings["lp"], picks="all", verbose="ERROR")
        self.sfreq = float(raw.info["sfreq"])
        self.signal = raw.get_data()[0].astype(float)
        sidecar = corrected_path(self.heppy_output / "ecg_qc", clean(self.row.get("base")))
        saved = load_corrected(sidecar, self.sfreq, len(self.signal)) if use_saved else None
        self.rpeaks = saved if saved is not None and len(saved) >= 3 else detect_rpeaks(self.signal, self.sfreq, self.settings)
        self.sidecar = sidecar
        self.source_name = src.name
        self.channel = ch
        try:
            raw.close()
        except Exception:
            pass
        mne.set_log_level("WARNING")

    def show(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RectangleSelector

        self.load()
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.selector = RectangleSelector(self.ax, self.on_select, useblit=True, button=[1], interactive=False)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()
        plt.show()

    def draw(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        self.ax.clear()
        self.x, self.waves, self.kept_indices = qrs_waves(self.signal, self.sfreq, self.rpeaks)
        if len(self.waves) == 0:
            self.ax.text(0.5, 0.5, "No QRS waves available.", ha="center", va="center", transform=self.ax.transAxes)
            self.fig.canvas.draw_idle()
            return
        segments = np.stack([np.column_stack([self.x, wave]) for wave in self.waves])
        self.ax.add_collection(LineCollection(segments, colors=(0.1, 0.1, 0.1, 0.08), linewidths=0.5))
        selected_mask = [i for i, idx in enumerate(self.kept_indices) if idx in self.selected]
        if selected_mask:
            self.ax.add_collection(LineCollection(segments[selected_mask], colors=(1.0, 0.1, 0.05, 0.7), linewidths=1.2))
        self.ax.plot(self.x, np.nanmean(self.waves, axis=0), color="crimson", linewidth=2)
        self.ax.axvline(0, color="black", linestyle="--", linewidth=1)
        self.ax.autoscale()
        self.ax.grid(alpha=0.2)
        self.ax.set_title(
            f"{self.row.get('base')} | {self.source_name} | {self.channel} | peaks={len(self.rpeaks)} selected={len(self.selected)}\n"
            "drag box to select; keys: r refit selected, d delete selected, a accept, c clear, y viable yes, n viable no, q quit"
        )
        self.ax.set_xlabel("seconds from detected R peak")
        self.fig.canvas.draw_idle()

    def on_select(self, start, end) -> None:
        x0, x1 = sorted((start.xdata, end.xdata))
        y0, y1 = sorted((start.ydata, end.ydata))
        inside = (self.x >= x0) & (self.x <= x1)
        if not np.any(inside):
            return
        hit = np.where(np.any((self.waves[:, inside] >= y0) & (self.waves[:, inside] <= y1), axis=1))[0]
        self.selected |= {self.kept_indices[i] for i in hit}
        self.draw()

    def save_peaks(self) -> None:
        _, _, save_corrected, *_ = heppy_peak_module()
        save_corrected(self.sidecar, self.rpeaks, self.sfreq)

    def update_viable(self, value: str) -> None:
        rows = read_rows(self.review_csv)
        for row in rows:
            if clean(row.get("base")) == clean(self.row.get("base")):
                row["viable"] = value
                break
        write_rows(self.review_csv, rows)
        self.row["viable"] = value

    def on_key(self, event) -> None:
        key = event.key
        if key == "c":
            self.selected.clear()
            self.draw()
        elif key == "a":
            self.save_peaks()
            self.update_viable("yes")
            self.draw()
        elif key in {"y", "n"}:
            self.update_viable("yes" if key == "y" else "no")
            self.draw()
        elif key == "d" and self.selected:
            _, _, _, delete_excluded, _ = heppy_peak_module()
            self.rpeaks = delete_excluded(self.rpeaks, sorted(self.selected))
            self.selected.clear()
            self.save_peaks()
            self.draw()
        elif key == "r" and self.selected:
            *_, refit = heppy_peak_module()
            self.rpeaks = refit(self.signal, self.rpeaks, self.sfreq, sorted(self.selected))
            self.selected.clear()
            self.save_peaks()
            self.draw()
        elif key == "q":
            import matplotlib.pyplot as plt

            plt.close(self.fig)


def find_row(rows: list[dict], target: str) -> dict:
    if target.isdigit():
        idx = int(target) - 1
        if 0 <= idx < len(rows):
            return rows[idx]
    for row in rows:
        if clean(row.get("base")) == target:
            return row
    raise SystemExit(f"No review row found for {target!r}")


def self_check() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        review = root / "ecg_qc_review.csv"
        rows = [{"base": "A", "qc_png": "", "viable": "", "flip": "False"}]
        write_rows(review, rows)
        loaded = read_rows(review)
        assert loaded[0]["base"] == "A"
        assert canonical_channel("EEG ECG1+-G2") == "ECG1+"
        assert np.array_equal(
            adaptive_fix(np.array([0, 100, 200, 1000]), 100.0, {"rr_min_bpm": 35, "rr_max_bpm": 140, "rr_outlier_mad": 3.5}),
            np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]),
        )
        waves = np.zeros((12, 5), dtype=float)
        waves[:, 2] = 1.0
        waves[:, 3] = -0.5
        waves[0, 2] = -1.0
        waves[1, 2] = 0.05
        waves[1, 3] = 1.0
        bad, _ = outlier_beats(waves, list(range(12)), 0.05)
        assert 0 in bad and 1 in bad
        signal = np.zeros(500, dtype=float)
        qrs = np.zeros(40, dtype=float)
        qrs[20] = 1.0
        qrs[25] = -1.0
        signal[180:220] = qrs
        aligned, ids, stats = align_flagged_beats(
            signal,
            100.0,
            np.array([200, 210]),
            np.array([0, 1]),
            [1],
            set(),
            0.0,
            0.2,
            {"rr_max_bpm": 140},
        )
        assert np.array_equal(aligned, np.array([200]))
        assert np.array_equal(ids, np.array([0]))
        assert stats["duplicate_dropped"] == 1
        estimates, uncertain = missing_peak_estimates(np.array([0, 100, 400, 500]), 100.0, {"rr_min_bpm": 35, "rr_max_bpm": 140})
        assert estimates == [200, 300] and uncertain == 0
        qc = rpeak_qc_score(np.arange(50, 6050, 100), 100.0, 6100, {"rr_min_bpm": 35, "rr_max_bpm": 140})
        assert qc["valid_60s_windows"] == 1 and qc["valid_60s_fraction"] == 1
        sources = write_recovery_sidecar(root / "peaks.csv", np.array([100, 200, 300]), 100.0,
                                         np.array([100, 200]), np.array([100]))
        assert sources == {"reviewed": 1, "full_record_detector": 1, "gap_recovered": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast external ECG/QRS reviewer for HEPPy outputs.")
    parser.add_argument("--heppy-output", type=Path, default=default_heppy_output())
    parser.add_argument("--review-csv", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0, help="1-based row index for PNG reviewer.")
    parser.add_argument("--qrs", default="", help="Open fast QRS editor for 1-based row index or base name.")
    parser.add_argument("--auto-fit", default="", help="Semi-auto fit row/base/all using QRS envelope plus centre-sample checks.")
    parser.add_argument("--auto-alpha", type=float, default=0.05, help="QRS envelope alpha and outside-sample fraction.")
    parser.add_argument("--auto-iterations", type=int, default=2)
    parser.add_argument("--auto-min-corr", type=float, default=0.80, help="Drop flagged beats whose best template correlation is below this.")
    parser.add_argument("--auto-corr-search-s", type=float, default=0.35, help="Seconds either side to search when aligning flagged beats to the template.")
    parser.add_argument("--auto-from-detected", action="store_true", help="Ignore saved corrected peaks and start auto-fit from detector output.")
    parser.add_argument("--auto-save", action="store_true", help="Write corrected r-peaks; default is dry-run.")
    parser.add_argument("--auto-mark-viable", action="store_true", help="Mark auto-saved rows viable=yes.")
    parser.add_argument("--auto-plot", action="store_true", help="Write compact corrected-QRS QC PNGs.")
    parser.add_argument("--auto-plot-dir", type=Path, default=None)
    parser.add_argument("--auto-report", type=Path, default=None)
    parser.add_argument("--recover-missed", default="", help="Recover missed peaks for row/base/all using ProMAC candidates plus template matching.")
    parser.add_argument("--recover-iterations", type=int, default=2)
    parser.add_argument("--recover-start", type=int, default=1, help="1-based first review row for --recover-missed all.")
    parser.add_argument("--recover-stop", type=int, default=0, help="Inclusive last review row; zero means the final row.")
    parser.add_argument("--recover-max-multiple", type=int, default=4, choices=(2, 3, 4))
    parser.add_argument("--recover-promac-threshold", type=float, default=0.33)
    parser.add_argument("--ecg-channel", default="", help="Override ECG channel.")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        print("self-check passed")
        return
    ensure_runtime(bool(args.qrs or args.auto_fit or args.recover_missed))
    review_csv = args.review_csv or args.heppy_output / "ecg_qc_review.csv"
    settings = ecg_settings(args.heppy_output)
    if args.ecg_channel:
        settings["channel"] = args.ecg_channel
    if args.auto_fit:
        rows = read_rows(review_csv)
        targets = rows if args.auto_fit.lower() == "all" else [find_row(rows, args.auto_fit)]
        plot_dir = args.auto_plot_dir or args.heppy_output / "ecg_qc" / "auto_qrs_plots"
        report = []
        for row in targets:
            try:
                result = auto_fit_row(
                    review_csv,
                    args.heppy_output,
                    row,
                    settings,
                    args.auto_alpha,
                    args.auto_iterations,
                    args.auto_save,
                    args.auto_plot,
                    plot_dir,
                    args.auto_mark_viable,
                    args.auto_min_corr,
                    args.auto_corr_search_s,
                    args.auto_from_detected,
                )
            except Exception as exc:
                result = {"base": clean(row.get("base")), "error": str(exc)}
            report.append(result)
            if "error" in result:
                print(f"{result['base']}: ERROR {result['error']}")
            else:
                print(
                    f"{result['base']}: {result['original_peaks']} -> {result['after_refit_peaks']} -> {result['final_peaks']} peaks; "
                    f"drop {result['dropped_after_refit']} ({result['drop_percent']}%); "
                    f"iter flags {result['refit_flagged_iter1']}, {result['refit_flagged_iter2']}; "
                    f"aligned {result['aligned_iter1']}, {result['aligned_iter2']}; "
                    f"{'saved' if args.auto_save else 'dry-run'}"
                )
        out = args.auto_report or args.heppy_output / "ecg_qc" / "auto_qrs_report.csv"
        write_dicts(out, report)
        print(f"report: {out}")
        return
    if args.recover_missed:
        rows = read_rows(review_csv)
        targets = rows if args.recover_missed.lower() == "all" else [find_row(rows, args.recover_missed)]
        if args.recover_missed.lower() == "all":
            first = max(1, args.recover_start)
            last = len(rows) if args.recover_stop <= 0 else min(len(rows), args.recover_stop)
            targets = rows[first - 1:last]
        report = []
        for row in targets:
            try:
                editor = QrsEditor(review_csv, args.heppy_output, row, settings)
                editor.load(use_saved=True)
                saved = np.asarray(editor.rpeaks, dtype=int)
                detected = detect_rpeaks(editor.signal, editor.sfreq, settings)
                recovered, stats = recover_missing_rpeaks(
                    editor.signal, editor.sfreq, detected, settings,
                    min_corr=args.auto_min_corr,
                    iterations=args.recover_iterations,
                    promac_threshold=args.recover_promac_threshold,
                    max_multiple=args.recover_max_multiple,
                )
                before_qc = rpeak_qc_score(saved, editor.sfreq, len(editor.signal), settings)
                after_qc = rpeak_qc_score(recovered, editor.sfreq, len(editor.signal), settings)
                if after_qc["valid_60s_windows"] < before_qc["valid_60s_windows"]:
                    raise RuntimeError(
                        f"full-record recovery reduced valid 60-s windows "
                        f"({before_qc['valid_60s_windows']} to {after_qc['valid_60s_windows']})"
                    )
                _, _, save_corrected, *_ = heppy_peak_module()
                save_corrected(editor.sidecar, recovered, editor.sfreq)
                source_counts = write_recovery_sidecar(editor.sidecar, recovered, editor.sfreq, detected, saved)
                plot_dir = args.auto_plot_dir or args.heppy_output / "ecg_qc" / "recovery_qc_plots"
                save_auto_qc_plot(
                    plot_dir / f"{clean(row.get('base'))}_recovered_qrs_qc.png",
                    editor.signal, editor.sfreq, recovered, args.auto_alpha,
                    f"{row.get('base')} | full-record detector plus gap recovery | peaks={len(recovered)}",
                )
                result = {
                    "base": clean(row.get("base")), **stats,
                    "saved_before": len(saved), "full_record_detected": len(detected), "final": len(recovered),
                    **{f"final_source_{key}": value for key, value in source_counts.items()},
                    **{f"before_{key}": value for key, value in before_qc.items()},
                    **{f"after_{key}": value for key, value in after_qc.items()},
                    "sidecar": str(editor.sidecar),
                }
            except Exception as exc:
                result = {"base": clean(row.get("base")), "error": str(exc)}
            report.append(result)
            print(result, flush=True)
        out = args.auto_report or args.heppy_output / "ecg_qc" / "missed_rpeak_recovery_report.csv"
        write_dicts(out, report)
        print(f"report: {out}")
        return
    if args.qrs:
        row = find_row(read_rows(review_csv), args.qrs)
        QrsEditor(review_csv, args.heppy_output, row, settings).show()
    else:
        PngReviewer(review_csv, args.heppy_output, args.start - 1).show()


if __name__ == "__main__":
    main()
