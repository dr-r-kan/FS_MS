"""Automatic ECG R-peak fit and missed-peak recovery (headless, deterministic).

The algorithms are the legacy ``OLD/fast_ecg_review.py`` ones, copied without numerical change:
``detect_rpeaks``, ``missing_peak_estimates``, ``recover_missing_rpeaks``, ``rpeak_qc_score``,
``write_recovery_sidecar``, ``clean_ecg_signal``, ``qrs_waves_from_cleaned``, ``normalise_waves``,
``outlier_beats``, ``template_correlations``, ``best_template_alignment``, ``align_flagged_beats``
and ``save_auto_qc_plot`` (``adaptive_fix`` is not used by either legacy stage and is omitted).

What differs (all engineering, none of it signal processing):

* no GUI/``QrsEditor``/``directories.txt``/cwd dependence;
* ``auto_fit`` **always** starts from detector output.  The legacy default re-loaded the previous
  sidecar and re-applied outlier deletion on every re-run, so peaks drifted; here the previous
  sidecar is never an input;
* ``recover`` starts from the fit written by ``auto_fit`` of the same fingerprint (kept as
  ``<base>_rpeaks_autofit.csv``) rather than from whatever ``_rpeaks_corrected.csv`` holds, so the
  ``rpeak_source`` labels are reproducible (legacy re-runs labelled every peak ``reviewed``);
* all writes are atomic.
"""
from __future__ import annotations

import csv
import io
import os
from pathlib import Path

import numpy as np

from fsms.atomic import atomic_write_bytes

ECG_TOKENS = ("ECG", "EKG", "CARD", "EXG")


def clean(value) -> str:
    return "" if value is None else str(value).strip()


def canonical_channel(ch: str) -> str:
    import re

    name = re.sub(r"^\s*EEG\s+", "", str(ch).strip(), flags=re.IGNORECASE)
    parts = [p.strip() for p in name.split("-") if p.strip()]
    if len(parts) > 1:
        name = parts[1] if parts[0].isdigit() else parts[0]
    return name.upper().replace("FP", "Fp").replace("Z", "z")


def same_channel(a: str, b: str) -> bool:
    return str(a).casefold() == str(b).casefold() or canonical_channel(a).casefold() == canonical_channel(b).casefold()


def read_raw_any(path: Path, preload=False):
    import mne

    suffix = Path(path).suffix.lower()
    if suffix == ".bdf":
        return mne.io.read_raw_bdf(str(path), preload=preload, verbose="ERROR")
    if suffix == ".edf":
        return mne.io.read_raw_edf(str(path), preload=preload, verbose="ERROR")
    if suffix == ".fif":
        return mne.io.read_raw_fif(str(path), preload=preload, verbose="ERROR")
    if suffix == ".vhdr":
        return mne.io.read_raw_brainvision(str(path), preload=preload, verbose="ERROR")
    raise ValueError(f"Unsupported file type: {path}")


class NoEcgChannel(RuntimeError):
    """Raised when a recording has no ECG-like channel (a normal, rule-handled outcome)."""


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
    raise NoEcgChannel("No ECG channel found; pass --ecg-channel.")


def load_ecg(path: Path, preferred_channel: str, hp: float, lp: float, flip: bool = False) -> dict:
    """Legacy ``QrsEditor.load`` minus sidecar handling: ECG channel, flipped?, band-passed."""
    raw = read_raw_any(Path(path), preload=False)
    ch = pick_ecg(raw, preferred_channel)
    raw.pick([ch])
    raw.load_data()
    if flip:
        raw._data[0] *= -1
    # ECG channels are often typed as misc; MNE's default data picks then select nothing.
    raw.filter(l_freq=hp, h_freq=lp, picks="all", verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    signal = raw.get_data()[0].astype(float)
    try:
        raw.close()
    except Exception:
        pass
    return {"signal": signal, "sfreq": sfreq, "channel": ch, "source": Path(path).name}


def detect_rpeaks(signal: np.ndarray, sfreq: float, settings: dict) -> np.ndarray:
    import neurokit2 as nk

    _, info = nk.ecg_process(ecg_signal=signal, sampling_rate=sfreq)
    return np.unique(np.asarray(info["ECG_R_Peaks"], dtype=int))


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


def csv_bytes(rows: list[dict]) -> bytes:
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_dicts(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    atomic_write_bytes(path, csv_bytes(rows))


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


def save_corrected_atomic(path: Path, rpeaks: np.ndarray, sfreq: float) -> None:
    """HEPPy's ``save_corrected_rpeaks`` (same bytes) written through a temporary + os.replace."""
    from ecg_peak_correction import save_corrected_rpeaks  # HEPPy, on sys.path via options.ensure_heppy_on_path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        save_corrected_rpeaks(temporary, rpeaks, sfreq)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_sidecar_peaks(path: Path, sfreq: float, n_samples: int) -> np.ndarray | None:
    """HEPPy ``load_corrected_rpeaks`` semantics (samples within (0, n_samples))."""
    from ecg_peak_correction import load_corrected_rpeaks

    return load_corrected_rpeaks(Path(path), sfreq, n_samples)


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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, waves, _ = qrs_waves(signal, sfreq, rpeaks, -0.2, 0.2)
    norm = normalise_waves(waves)
    path = Path(path)
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
    temporary = path.with_name(f".{path.stem}.part-{os.getpid()}{path.suffix}")
    try:
        fig.savefig(temporary, dpi=150)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(fig)


def auto_fit(signal: np.ndarray, sfreq: float, settings: dict, alpha: float, iterations: int,
             min_corr: float, corr_search_s: float) -> tuple[np.ndarray, dict]:
    """Legacy ``auto_fit_row`` body, always started from detector output (no stale sidecar)."""
    original = detect_rpeaks(signal, sfreq, settings)
    history = []
    align_history = []
    current = original
    peak_ids = np.arange(len(current), dtype=int)
    handled_ids: set[int] = set()
    for _ in range(max(0, int(iterations))):
        _, waves, kept = qrs_waves(signal, sfreq, current, -0.2, 0.2)
        bad, _outside = outlier_beats(waves, kept, alpha)
        history.append(len(bad))
        if not bad:
            break
        current, peak_ids, stats = align_flagged_beats(
            signal, sfreq, current, peak_ids, bad, handled_ids, min_corr, corr_search_s, settings
        )
        align_history.append(stats)

    _, waves, kept = qrs_waves(signal, sfreq, current, -0.2, 0.2)
    final_bad, outside = outlier_beats(waves, kept, alpha)
    from ecg_peak_correction import delete_excluded_rpeaks

    fixed = delete_excluded_rpeaks(current, final_bad)

    def hist(items, key, i):
        return int(items[i][key]) if len(items) > i else 0

    result = {
        "alpha": alpha,
        "iterations": iterations,
        "min_corr": min_corr,
        "corr_search_s": corr_search_s,
        "original_peaks": int(len(original)),
        "after_refit_peaks": int(len(current)),
        "final_peaks": int(len(fixed)),
        "refit_flagged_iter1": int(history[0]) if len(history) > 0 else 0,
        "refit_flagged_iter2": int(history[1]) if len(history) > 1 else 0,
        "aligned_iter1": hist(align_history, "aligned", 0),
        "aligned_iter2": hist(align_history, "aligned", 1),
        "corr_failed_iter1": hist(align_history, "corr_failed", 0),
        "corr_failed_iter2": hist(align_history, "corr_failed", 1),
        "repeat_dropped_iter1": hist(align_history, "repeat_dropped", 0),
        "repeat_dropped_iter2": hist(align_history, "repeat_dropped", 1),
        "duplicate_dropped_iter1": hist(align_history, "duplicate_dropped", 0),
        "duplicate_dropped_iter2": hist(align_history, "duplicate_dropped", 1),
        "dropped_after_refit": int(len(final_bad)),
        "drop_percent": round(100.0 * len(final_bad) / max(len(current), 1), 3),
        "median_outside_fraction": round(float(np.nanmedian(outside)) if len(outside) else 0.0, 4),
        "max_outside_fraction": round(float(np.nanmax(outside)) if len(outside) else 0.0, 4),
    }
    return np.asarray(fixed, dtype=int), result


def recover(signal: np.ndarray, sfreq: float, saved: np.ndarray, settings: dict, params: dict,
            alpha: float, iterations_auto: int) -> dict:
    """Legacy ``--recover-missed`` body for one recording; returns peaks, sources and statistics.

    ``saved`` is the auto-fit result (legacy: the sidecar left by the auto-fit stage).  Recovery
    starts from the *full-record detector* output, exactly as legacy, and is declined when it
    would reduce the number of valid 60-s windows relative to ``saved``.
    """
    saved = np.asarray(saved, dtype=int)
    detected = detect_rpeaks(signal, sfreq, settings)
    recovered, stats = recover_missing_rpeaks(
        signal, sfreq, detected, settings,
        min_corr=params["min_template_corr"],
        iterations=params["iterations"],
        promac_threshold=params["promac_threshold"],
        max_multiple=params["max_multiple"],
    )
    before_qc = rpeak_qc_score(saved, sfreq, len(signal), settings)
    after_qc = rpeak_qc_score(recovered, sfreq, len(signal), settings)
    declined = after_qc["valid_60s_windows"] < before_qc["valid_60s_windows"]
    return {"saved": saved, "detected": detected, "recovered": recovered, "stats": stats,
            "before_qc": before_qc, "after_qc": after_qc, "declined": declined}
