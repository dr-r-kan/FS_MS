"""Cardiac autonomic native table (CSI/CPI) from corrected R-peak sidecars.

This is the only upstream input the preregistered transition analysis takes from the ECG.  The numerical
routines are verbatim ports of the legacy implementations so that results are unchanged:

* ``robust_cov_hrv``, ``poincare_features``, ``native_indices``  <- analyse_candia_autonomic.py
  (Python port of the Candia-Rivera et al. robust CSI/CPI MATLAB code, 15-s windows)
* ``read_rpeaks_with_recovery_flags``                            <- analyse_ace_cardioneural.py
* ``standardise_autonomic_to_ii``, record loop                   <- analyse_focused_fs_ms.py (read_ecg)

``tests/test_autonomic.py`` asserts equality with the legacy functions on synthetic R-peak trains.
The legacy script additionally fitted many exploratory Bayesian models; none of that is needed here.
Records that cannot support the indices are excluded with a recorded reason, never silently.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from .atomic import atomic_write_json, sha256_file

NATIVE_SCHEMA = "continuous-source-native-v4"  # same label as the legacy cache it reproduces
MIN_RR_BPM, MAX_RR_BPM = 35.0, 140.0
CONDITIONS = ("ii", "preict", "transictal")


def robust_cov_hrv(x: np.ndarray) -> np.ndarray:
    """Python equivalent of the authors' robustCovHRV.m shrinkage covariance."""
    x = np.asarray(x, float)
    if x.ndim != 2 or len(x) < 4 or not np.isfinite(x).all():
        raise ValueError("robust covariance needs at least four finite observations")
    n, p = x.shape
    xc = x - x.mean(axis=0)

    def varcov(z):
        s = np.cov(z, rowvar=False, ddof=1)
        outer = z[:, :, None] * z[:, None, :]
        vs = np.var(outer, axis=0, ddof=1) * n / (n - 1) ** 2
        return np.atleast_2d(s), np.atleast_2d(vs)

    s, vs = varcov(xc)
    v, vv = np.diag(s), np.diag(vs)
    target = np.median(v)
    denominator = np.sum((v - target) ** 2)
    lam = np.clip(np.sum(vv) / denominator, 0, 1) if denominator > 0 else 1.0
    shrunk_v = (1 - lam) * v + lam * target
    sd = xc.std(axis=0, ddof=1)
    if np.any(sd <= 0):
        return np.diag(np.maximum(shrunk_v, 0))
    r, vr = varcov(xc / sd)
    lower = np.tril_indices(p, -1)
    denominator = np.sum(r[lower] ** 2)
    lam = np.clip(np.sum(vr[lower]) / denominator, 0, 1) if denominator > 0 else 1.0
    r = (1 - lam) * r
    np.fill_diagonal(r, 1)
    d = np.diag(np.sqrt(np.maximum(shrunk_v, 0)))
    return d @ r @ d


def poincare_features(ibi: np.ndarray) -> tuple[float, float, float]:
    points = np.column_stack([ibi[:-1], ibi[1:]])
    eigenvalues = np.maximum(np.linalg.eigvalsh(robust_cov_hrv(points)), 0)
    centre = trim_mean(points, .025, axis=0)
    return float(np.sqrt(eigenvalues[0])), float(np.sqrt(eigenvalues[1])), float(np.linalg.norm(centre))


def native_indices(peaks: np.ndarray, recovered_peaks: np.ndarray, window_s: float = 15.0) -> pd.DataFrame:
    peaks = np.asarray(peaks, float)
    recovered_peaks = np.asarray(recovered_peaks, bool)
    rr, t_rr = np.diff(peaks), peaks[1:]
    interval_recovered = recovered_peaks[:-1] | recovered_peaks[1:]
    physiological = np.isfinite(rr) & (rr >= 60 / MAX_RR_BPM) & (rr <= 60 / MIN_RR_BPM)
    base_pairs = physiological[:-1] & physiological[1:]
    if base_pairs.sum() < 20:
        raise ValueError("fewer than 20 physiological IBI pairs")
    base = np.column_stack([rr[:-1][base_pairs], rr[1:][base_pairs]])
    base_eigen = np.maximum(np.linalg.eigvalsh(robust_cov_hrv(base)), 0)
    sd01, sd02 = np.sqrt(base_eigen)
    d0 = float(np.linalg.norm(trim_mean(base, .025, axis=0)))
    rows = []
    eligible = np.flatnonzero(t_rr > t_rr[0] + window_s)
    for i in eligible[:-1]:  # matches the published MATLAB implementation
        left = int(np.searchsorted(t_rr, t_rr[i] - window_s, side="left"))
        use = np.arange(left, i + 1)
        if (len(use) < 7 or t_rr[use[-1]] - t_rr[use[0]] < .8 * window_s or
                not physiological[use].all() or interval_recovered[use].mean() > .10):
            continue
        local_rr = rr[use]
        first_difference = np.diff(local_rr)
        second_difference = np.diff(first_difference)
        sd1, sd2, distance = poincare_features(local_rr)
        rows.append((np.median(t_rr[use]), t_rr[use[0]], t_rr[use[-1]], sd1, sd2, distance,
                     len(use), interval_recovered[use].mean(), local_rr.mean(), 60 / local_rr.mean(),
                     first_difference.mean(), second_difference.mean() if len(second_difference) else np.nan,
                     np.sqrt(np.mean(first_difference ** 2))))
    out = pd.DataFrame(rows, columns=["event_time_s", "window_start_s", "window_end_s", "sd1", "sd2",
                                      "ccd", "n_ibi", "recovered_fraction", "mean_rr_s", "mean_hr_bpm",
                                      "rr_first_difference_s", "rr_acceleration_s", "rmssd_s"])
    if len(out) < 30:
        raise ValueError("fewer than 30 valid 15-second estimates")
    for column, reference in (("sd1", sd01), ("sd2", sd02), ("ccd", d0)):
        out[f"{column}_recentred"] = out[column] - out[column].mean() + reference
    out["cpi_hr_component"] = out["ccd_recentred"]
    out["csi_hr_component"] = 2 * out["ccd_recentred"].mean() - out["ccd_recentred"]
    out["cpi_hrv_component"] = 10 * out["sd1_recentred"] + 1
    out["csi_hrv_component"] = out["sd2_recentred"] + 1
    out["cpi"] = out["cpi_hr_component"] + out["cpi_hrv_component"]
    out["csi"] = out["csi_hr_component"] + out["csi_hrv_component"]
    return out


def read_rpeaks_with_recovery_flags(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.array([], dtype=float), np.array([], dtype=bool)
    data = pd.read_csv(path)
    times = pd.to_numeric(data.get("rpeak_time_s"), errors="coerce")
    keep = times.notna()
    data = data.loc[keep].copy()
    data["rpeak_time_s"] = times[keep].to_numpy(float)
    data = data.sort_values("rpeak_time_s")
    if "is_gap_recovered" in data:
        recovered = pd.to_numeric(data["is_gap_recovered"], errors="coerce").fillna(0).astype(bool).to_numpy()
    else:
        recovered = np.zeros(len(data), dtype=bool)
    return data["rpeak_time_s"].to_numpy(float), recovered


def sidecar_for(source_raw_fif: str) -> Path:
    """<heppy_out>/raw_fif/<base>_pp_raw[_keepcfa].fif  ->  <heppy_out>/ecg_qc/<base>_rpeaks_corrected.csv.

    Resolved from the FIF location so adopted (in-place, read-only) legacy HEPPy output works unchanged.
    """
    fif = Path(str(source_raw_fif))
    base = fif.name
    for suffix in ("_pp_raw_keepcfa.fif", "_pp_raw.fif", ".fif"):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    return fif.parent.parent / "ecg_qc" / f"{base}_rpeaks_corrected.csv"


def standardise_autonomic_to_ii(native: pd.DataFrame) -> pd.DataFrame:
    out = native.copy()
    baseline = out[out.condition.eq("ii")].groupby("participant", observed=True)
    for index in ("csi", "cpi", "mean_rr_s", "mean_hr_bpm", "rr_first_difference_s",
                  "rr_acceleration_s", "rmssd_s"):
        mean = baseline[index].mean()
        sd = baseline[index].std()
        out[f"{index}_ii_mean"] = out.participant.map(mean)
        out[f"{index}_ii_sd"] = out.participant.map(sd)
        out[f"{index}_z"] = (out[index] - out[f"{index}_ii_mean"]) / out[f"{index}_ii_sd"].where(out[f"{index}_ii_sd"].gt(0))
    out["rr_surprise"] = np.nan
    for participant, part in out.groupby("participant", observed=True):
        part = part.sort_values(["file_path", "event_time_s"])
        lag1 = part.groupby("file_path", observed=True).mean_rr_s.shift(1)
        lag2 = part.groupby("file_path", observed=True).mean_rr_s.shift(2)
        design = np.column_stack([np.ones(len(part)), lag1, lag2])
        eligible = part.condition.eq("ii").to_numpy() & np.isfinite(design).all(axis=1)
        if eligible.sum() < 30:
            continue
        beta = np.linalg.lstsq(design[eligible], part.mean_rr_s.to_numpy(float)[eligible], rcond=None)[0]
        residual = part.mean_rr_s.to_numpy(float) - design @ beta
        variance = np.var(residual[eligible], ddof=3)
        if not np.isfinite(variance) or variance <= 0:
            continue
        out.loc[part.index, "rr_surprise"] = .5 * np.log(2 * np.pi * variance) + residual ** 2 / (2 * variance)
    baseline = out[out.condition.eq("ii")].groupby("participant", observed=True)
    mean, sd = baseline.rr_surprise.mean(), baseline.rr_surprise.std()
    out["rr_surprise_z"] = ((out.rr_surprise - out.participant.map(mean)) /
                            out.participant.map(sd).where(out.participant.map(sd).gt(0)))
    return out


def build_native(manifest: pd.DataFrame, sidecar_resolver=sidecar_for) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-record native CSI/CPI estimates (cropped to each manifest segment) and a reason-coded QC table."""
    records = (manifest[manifest.condition.astype(str).str.lower().isin(CONDITIONS)]
               .sort_values(["source_raw_fif", "crop_start_seconds", "file_path"]))
    native_rows, qc_rows = [], []
    cached_sidecar = None
    cached_peaks = cached_recovered = cached_native = None
    cached_error = ""
    for row in records.itertuples(index=False):
        sidecar = sidecar_resolver(row.source_raw_fif)
        if str(sidecar) != cached_sidecar:
            cached_sidecar = str(sidecar)
            cached_peaks, cached_recovered = read_rpeaks_with_recovery_flags(sidecar)
            cached_error = ""
            try:
                cached_native = native_indices(cached_peaks, cached_recovered)
            except ValueError as error:
                cached_native = None
                cached_error = str(error)
        start, stop = float(row.crop_start_seconds), float(row.crop_end_seconds)
        duration = stop - start
        keep = (cached_peaks >= start) & (cached_peaks <= stop)
        audit = {"participant": row.participant, "file_path": row.file_path, "condition": row.condition,
                 "rpeak_sidecar": str(sidecar), "sidecar_exists": Path(sidecar).exists(),
                 "n_peaks": int(keep.sum()), "autonomic_valid": False, "reason": ""}
        if keep.sum() < 3:
            audit["reason"] = "no_corrected_rpeaks" if not audit["sidecar_exists"] else "fewer than three R peaks"
        else:
            try:
                if cached_native is None:
                    raise ValueError(cached_error)
                native = cached_native[(cached_native.window_start_s >= start) &
                                       (cached_native.window_end_s <= stop)].copy()
                if len(native) < 30:
                    raise ValueError("fewer than 30 complete 15-second estimates inside segment")
                for column in ("event_time_s", "window_start_s", "window_end_s"):
                    native[column] -= start
                native.insert(0, "participant", row.participant)
                native.insert(1, "file_path", row.file_path)
                native.insert(2, "condition", str(row.condition).lower())
                native["record_duration_s"] = duration
                native_rows.append(native)
                audit["autonomic_valid"] = True
            except ValueError as error:
                audit["reason"] = str(error)
        qc_rows.append(audit)
    if not native_rows:
        raise ValueError("No manifest record produced valid autonomic estimates; see the QC reasons: "
                         + "; ".join(sorted({r["reason"] for r in qc_rows if r["reason"]})))
    return pd.concat(native_rows, ignore_index=True), pd.DataFrame(qc_rows)


def run(manifest_csv: Path, out_dir: Path) -> dict:
    manifest = pd.read_csv(manifest_csv)
    needed = {"participant", "condition", "file_path", "source_raw_fif", "crop_start_seconds", "crop_end_seconds"}
    missing = needed - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest lacks required columns: {sorted(missing)}")
    native, qc = build_native(manifest)
    native = standardise_autonomic_to_ii(native)
    out_dir.mkdir(parents=True, exist_ok=True)
    native_path = out_dir / "autonomic_native_ii_standardised.csv.gz"
    qc_path = out_dir / "autonomic_native_qc.csv"
    for path, frame, kwargs in ((native_path, native, {"compression": "gzip"}), (qc_path, qc, {})):
        temporary = path.with_name(path.name + ".part")
        frame.to_csv(temporary, index=False, **kwargs)
        temporary.replace(path)
    sidecars = sorted({r for r in qc.rpeak_sidecar[qc.sidecar_exists]})
    audit = {"schema": NATIVE_SCHEMA, "manifest_records": int(len(manifest)),
             "records_valid": int(qc.autonomic_valid.sum()), "records_excluded": int((~qc.autonomic_valid).sum()),
             "exclusion_reasons": qc.loc[~qc.autonomic_valid, "reason"].value_counts().to_dict(),
             "participants_with_ii_baseline": int(native[native.condition.eq("ii")].participant.nunique()),
             "rpeak_sidecar_sha256": {s: sha256_file(Path(s)) for s in sidecars}}
    atomic_write_json(out_dir / "autonomic_audit.json", audit)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run(args.manifest, args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
