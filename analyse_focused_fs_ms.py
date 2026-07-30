"""Focused FS/ES cardiac and microstate analysis requested for the final paper set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit

import analyse_ace_cardioneural as ace
import analyse_transition_priorities as priorities
from analyse_bayesian_statewise import effective_sample_size, split_rhat
from analyse_candia_autonomic import native_indices
import analyse_statewise_transition_recovery as statewise
from analyse_preictal_publication import attach_strata
from microstate_labels import load_global_labels


ALL_STATES = ("F", "E", "G", "B", "A", "C")
STATE_REFERENCE_MIN_CORRELATION = 0.50
FUNCTIONAL = ("FS functional", "DD functional")
EPILEPTIC = ("ES epileptic", "DD epileptic")
ECG_CACHE_SCHEMA = "continuous-source-native-v4"
ANALYSIS_SCHEMA = "focused-c-reference-050-v6"


def inv_gamma(rng: np.random.Generator, shape: float, scale: float) -> float:
    return float(1 / rng.gamma(shape, 1 / scale))


def mixed_chain(y: np.ndarray, x: np.ndarray, participant: np.ndarray, event: np.ndarray | None,
                iterations: int, burn: int, thin: int, seed: int, prior_var: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, p = x.shape
    n_participants = participant.max() + 1
    n_events = event.max() + 1 if event is not None else 0
    beta = np.linalg.solve(x.T @ x + np.eye(p) / prior_var, x.T @ y)
    u = np.zeros(n_participants); v = np.zeros(n_events)
    sigma2 = max(float(np.var(y - x @ beta)), .05)
    participant_var = event_var = max(sigma2 / 4, .05)
    saved = []
    for iteration in range(iterations):
        adjusted = y - u[participant] - (v[event] if event is not None else 0)
        precision = x.T @ x / sigma2 + np.eye(p) / prior_var
        mean = np.linalg.solve(precision, x.T @ adjusted / sigma2)
        chol = np.linalg.cholesky(precision)
        beta = mean + np.linalg.solve(chol.T, rng.normal(size=p))

        adjusted = y - x @ beta - (v[event] if event is not None else 0)
        counts = np.bincount(participant, minlength=n_participants)
        variance = 1 / (counts / sigma2 + 1 / participant_var)
        mean = variance * np.bincount(participant, weights=adjusted, minlength=n_participants) / sigma2
        u = mean + rng.normal(size=n_participants) * np.sqrt(variance)

        if event is not None:
            adjusted = y - x @ beta - u[participant]
            counts = np.bincount(event, minlength=n_events)
            variance = 1 / (counts / sigma2 + 1 / event_var)
            mean = variance * np.bincount(event, weights=adjusted, minlength=n_events) / sigma2
            v = mean + rng.normal(size=n_events) * np.sqrt(variance)

        residual = y - x @ beta - u[participant] - (v[event] if event is not None else 0)
        sigma2 = inv_gamma(rng, 2 + n / 2, .1 + float(residual @ residual) / 2)
        participant_var = inv_gamma(rng, 2 + n_participants / 2, .1 + float(u @ u) / 2)
        if event is not None:
            event_var = inv_gamma(rng, 2 + n_events / 2, .1 + float(v @ v) / 2)
        if iteration >= burn and (iteration - burn) % thin == 0:
            variance_parameters = [np.sqrt(participant_var)]
            if event is not None:
                variance_parameters.append(np.sqrt(event_var))
            variance_parameters.append(np.sqrt(sigma2))
            saved.append(np.r_[beta, variance_parameters])
    return np.asarray(saved)


def fit_bayes(frame: pd.DataFrame, outcome: str, predictors: list[str], name: str, data_dir: Path,
              args, *, intercept: bool = True, event_column: str | None = "event_key", prior_var: float = 16) -> dict:
    required = [outcome, "participant", *predictors] + ([event_column] if event_column else [])
    use = frame.dropna(subset=required).copy()
    if use.participant.nunique() < 8 or len(use) <= len(predictors) + 10:
        raise ValueError(f"{name}: insufficient eligible observations")
    x = use[predictors].to_numpy(float)
    names = list(predictors)
    if intercept:
        x = np.column_stack([np.ones(len(use)), x]); names.insert(0, "Intercept")
    y = use[outcome].to_numpy(float)
    participant = pd.factorize(use.participant, sort=True)[0]
    event = pd.factorize(use[event_column], sort=True)[0] if event_column else None
    name_seed = int(hashlib.sha256(name.encode()).hexdigest()[:6], 16)
    def sample(iterations: int, burn: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        result = np.stack([mixed_chain(y, x, participant, event, iterations, burn, args.thin,
                                       args.seed + chain * 1009 + name_seed, prior_var)
                           for chain in range(args.chains)])
        return result, split_rhat(result), effective_sample_size(result)
    iterations, burn = args.iterations, args.burn
    chains, rhat, ess = sample(iterations, burn)
    diagnostic_names = names + ["participant_sd"] + (["event_sd"] if event is not None else []) + ["residual_sd"]
    failed = not np.isfinite(rhat).all() or rhat.max() > 1.05 or np.nanmin(ess) < 100
    if failed:
        worst = int(np.nanargmax(rhat))
        iterations, burn = iterations * 2, burn * 2
        print(f"Retrying {name} with {iterations} iterations: max R-hat {np.nanmax(rhat):.3f} "
              f"({diagnostic_names[worst]}), "
              f"min ESS {np.nanmin(ess):.1f}", flush=True)
        chains, rhat, ess = sample(iterations, burn)
    if not np.isfinite(rhat).all() or rhat.max() > 1.05 or np.nanmin(ess) < 100:
        worst = int(np.nanargmax(rhat)); lowest = int(np.nanargmin(ess))
        raise RuntimeError(f"{name}: MCMC failed (max R-hat {np.nanmax(rhat):.3f} "
                           f"for {diagnostic_names[worst]}, min ESS {np.nanmin(ess):.1f} "
                           f"for {diagnostic_names[lowest]})")
    safe = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    np.savez_compressed(data_dir / f"posterior_{safe}.npz", draws=chains, parameter_names=np.asarray(names))
    return {"name": name, "frame": use, "draws": chains[:, :, :len(names)].reshape(-1, len(names)),
            "names": names, "rhat": rhat[:len(names)], "ess": ess[:len(names)],
            "iterations": iterations, "burn": burn}


def fit_conjugate(frame: pd.DataFrame, outcome: str, predictors: list[str], name: str,
                  args, *, prior_var: float) -> dict:
    """Exact Gaussian regression posterior for participant-aggregated descriptive cells."""
    use = frame.dropna(subset=[outcome, "participant", *predictors]).copy()
    if use.participant.nunique() < 4:
        raise ValueError(f"{name}: fewer than four eligible participants")
    x = np.column_stack([np.ones(len(use)), use[predictors].to_numpy(float)])
    names = ["Intercept", *predictors]; y = use[outcome].to_numpy(float)
    precision = x.T @ x + np.eye(x.shape[1]) / prior_var
    mean = np.linalg.solve(precision, x.T @ y)
    shape = 2 + len(y) / 2
    scale = .1 + .5 * max(float(y @ y - mean @ precision @ mean), 1e-10)
    chol = np.linalg.cholesky(precision)
    n_saved = max(1, (args.iterations - args.burn + args.thin - 1) // args.thin)
    chains = []
    name_seed = int(hashlib.sha256(name.encode()).hexdigest()[:6], 16)
    for chain in range(args.chains):
        rng = np.random.default_rng(args.seed + chain * 1009 + name_seed)
        sigma2 = 1 / rng.gamma(shape, 1 / scale, size=n_saved)
        noise = rng.normal(size=(n_saved, x.shape[1])) @ np.linalg.inv(chol)
        beta = mean + noise * np.sqrt(sigma2)[:, None]
        chains.append(beta)
    chains = np.stack(chains); rhat = split_rhat(chains); ess = effective_sample_size(chains)
    if not np.isfinite(rhat).all() or rhat.max() > 1.05 or np.nanmin(ess) < 100:
        raise RuntimeError(f"{name}: exact posterior draw check failed (max R-hat {rhat.max():.3f}, min ESS {ess.min():.1f})")
    return {"name": name, "frame": use, "draws": chains.reshape(-1, len(names)), "names": names,
            "rhat": rhat, "ess": ess, "iterations": args.iterations, "burn": args.burn}


def descriptive_cell_summaries(frame: pd.DataFrame, outcome: str, analysis: str, args, *,
                               prior_var: float, transform=None) -> list[dict]:
    rows = []
    for functional in (0, 1):
        for hypo in (0, 1):
            for ictal in (0, 1):
                cell = frame[(frame.is_functional == functional) & (frame.is_hypo == hypo) &
                             (frame.is_ictal == ictal)]
                part = cell.groupby("participant", observed=True).agg(
                    **{outcome: (outcome, "mean")}, is_dd=("is_dd", "first")).reset_index()
                part["is_dd_centered"] = part.is_dd - part.is_dd.mean()
                label = f"{'FS' if functional else 'ES'} {'hypo' if hypo else 'hyper'} {'ictal' if ictal else 'preictal'}"
                model = fit_conjugate(part, outcome, ["is_dd_centered"], f"{analysis}: {label}", args,
                                      prior_var=prior_var)
                rows.append(contrast(model, {"Intercept": 1}, label, transform))
    return rows


def contrast(model: dict, weights: dict[str, float], label: str, transform=None) -> dict:
    vector = np.zeros(len(model["names"]))
    for name, weight in weights.items():
        vector[model["names"].index(name)] = weight
    values = model["draws"] @ vector
    if transform is not None:
        values = transform(values)
    low, median, high = np.quantile(values, [.025, .5, .975])
    return {"analysis": model["name"], "contrast": label, "median": median, "ci95_low": low, "ci95_high": high,
            "probability_gt_zero": float(np.mean(values > 0)), "probability_lt_zero": float(np.mean(values < 0)),
            "n_rows": len(model["frame"]), "n_participants": model["frame"].participant.nunique(),
            "max_rhat": float(np.max(model["rhat"])), "min_ess": float(np.min(model["ess"]))}


def add_event_metadata(frame: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    metadata = manifest[["file_key", "stratum", "hypo_hyper", "event_number", "source_row"]].copy()
    out = frame.copy(); out["file_key"] = normalise_path(out.file_path)
    out = out.drop(columns=[column for column in ("stratum", "hypo_hyper") if column in out]).merge(
        metadata, on="file_key", how="left", validate="many_to_one")
    out["semiology"] = ace.normalise_semiology(out.hypo_hyper)
    out["event_type"] = np.select([out.stratum.isin(FUNCTIONAL), out.stratum.isin(EPILEPTIC)],
                                  ["functional", "epileptic"], default="interictal")
    out["is_functional"] = out.event_type.eq("functional").astype(int)
    out["is_hypo"] = out.semiology.eq("hypo").astype(int)
    out["is_dd"] = out.stratum.astype(str).str.startswith("DD").astype(int)
    number = pd.to_numeric(out.event_number, errors="coerce").fillna(-1).astype(int).astype(str)
    source = pd.to_numeric(out.source_row, errors="coerce").fillna(-1).astype(int).astype(str)
    out["event_key"] = out.participant.astype(str) + "_" + number + "_" + source
    return out


def normalise_path(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace("/", "\\", regex=False).str.lower()


def cell_design(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[tuple, str]]:
    out = frame.copy(); mapping = {}
    for functional in (0, 1):
        for hypo in (0, 1):
            for ictal in (0, 1):
                key = (functional, hypo, ictal)
                name = f"cell_{functional}_{hypo}_{ictal}"
                out[name] = ((out.is_functional == functional) & (out.is_hypo == hypo) & (out.is_ictal == ictal)).astype(int)
                mapping[key] = name
    return out, list(mapping.values()), mapping


def read_ecg(manifest: pd.DataFrame, ecg_dir: Path, original_ecg_dir: Path, data_dir: Path,
             reuse: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    native_path = data_dir / "autonomic_native.csv.gz"; hr_path = data_dir / "dense_rr_hr.csv.gz"; qc_path = data_dir / "ecg_qc.csv"
    schema_path = data_dir / "ecg_cache_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.exists() else {}
    if (reuse and schema.get("schema") == ECG_CACHE_SCHEMA and native_path.exists() and
            hr_path.exists() and qc_path.exists()):
        return pd.read_csv(native_path), pd.read_csv(hr_path), pd.read_csv(qc_path)
    native_rows, hr_rows, qc_rows = [], [], []
    records = (manifest[manifest.condition.astype(str).str.lower().isin(("ii", "preict", "transictal"))]
               .sort_values(["source_raw_fif", "crop_start_seconds", "file_path"]))
    cached_sidecar = None; cached_peaks = cached_recovered = cached_native = None; cached_native_error = ""
    for number, row in enumerate(records.itertuples(index=False), 1):
        sidecar = ace.rpeak_file(ecg_dir, row.source_raw_fif)
        if not sidecar.exists(): sidecar = ace.rpeak_file(original_ecg_dir, row.source_raw_fif)
        if str(sidecar) != cached_sidecar:
            cached_sidecar = str(sidecar)
            cached_peaks, cached_recovered = ace.read_rpeaks_with_recovery_flags(sidecar)
            cached_native_error = ""
            try:
                cached_native = native_indices(cached_peaks, cached_recovered)
            except ValueError as error:
                cached_native = None; cached_native_error = str(error)
        peaks, recovered = cached_peaks, cached_recovered
        start, stop = float(row.crop_start_seconds), float(row.crop_end_seconds); duration = stop - start
        keep = (peaks >= start) & (peaks <= stop); local = peaks[keep] - start; flags = recovered[keep]
        audit = {"participant": row.participant, "file_path": row.file_path, "condition": row.condition,
                 "rpeak_sidecar": str(sidecar), "n_peaks": len(local), "hr_valid": False, "autonomic_valid": False, "reason": ""}
        if len(local) >= 3:
            rr = np.diff(local); midpoint = (local[:-1] + local[1:]) / 2
            interval_recovered = flags[:-1] | flags[1:]
            valid = np.isfinite(rr) & (rr >= 60 / 140) & (rr <= 60 / 35) & ~interval_recovered
            phase = str(row.condition).lower(); phase = "ictal" if phase == "transictal" else phase
            if phase in ("ii", "preict", "ictal"):
                window = valid & ((midpoint >= duration - 60) if phase == "preict" else
                                  ((midpoint < 60) if phase == "ictal" else np.ones(len(midpoint), bool)))
                if window.sum() >= 20 and np.ptp(midpoint[window]) >= 40:
                    mean_hr = 60 * window.sum() / rr[window].sum()
                    hr_rows.append({"participant": row.participant, "file_path": row.file_path, "condition": row.condition,
                                    "phase": phase, "mean_hr_bpm": mean_hr, "n_rr": int(window.sum())})
                    audit["hr_valid"] = True
            try:
                if cached_native is None:
                    raise ValueError(cached_native_error)
                native = cached_native[(cached_native.window_start_s >= start) &
                                       (cached_native.window_end_s <= stop)].copy()
                if len(native) < 30:
                    raise ValueError("fewer than 30 complete 15-second estimates inside segment")
                for time_column in ("event_time_s", "window_start_s", "window_end_s"):
                    native[time_column] -= start
                native.insert(0, "participant", row.participant); native.insert(1, "file_path", row.file_path)
                native.insert(2, "condition", str(row.condition).lower()); native["record_duration_s"] = duration
                native_rows.append(native); audit["autonomic_valid"] = True
            except ValueError as error:
                audit["reason"] = str(error)
        else:
            audit["reason"] = "fewer than three R peaks"
        qc_rows.append(audit)
        if number % 25 == 0: print(f"ECG records {number}/{len(records)}", flush=True)
    native = pd.concat(native_rows, ignore_index=True); hr = pd.DataFrame(hr_rows); qc = pd.DataFrame(qc_rows)
    native.to_csv(native_path, index=False, compression="gzip"); hr.to_csv(hr_path, index=False, compression="gzip"); qc.to_csv(qc_path, index=False)
    schema_path.write_text(json.dumps({"schema": ECG_CACHE_SCHEMA,
                                      "native_method": "full source-record RR train; complete windows cropped to segment",
                                      "dense_hr_periods": ["ii", "preict", "ictal"]}, indent=2),
                           encoding="utf-8")
    return native, hr, qc


def standardise_autonomic_to_ii(native: pd.DataFrame) -> pd.DataFrame:
    out = native.copy(); baseline = out[out.condition.eq("ii")].groupby("participant", observed=True)
    for index in ("csi", "cpi", "mean_rr_s", "mean_hr_bpm", "rr_first_difference_s",
                  "rr_acceleration_s", "rmssd_s"):
        mean = baseline[index].mean(); sd = baseline[index].std()
        out[f"{index}_ii_mean"] = out.participant.map(mean); out[f"{index}_ii_sd"] = out.participant.map(sd)
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


def simple_hr(hr: pd.DataFrame, manifest: pd.DataFrame, data_dir: Path, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    event = add_event_metadata(hr[~hr.phase.eq("ii")], manifest)
    event = event[event.event_type.isin(("functional", "epileptic")) & event.semiology.notna()].copy()
    group = ["participant", "event_type", "semiology", "is_functional", "is_hypo", "is_dd", "phase"]
    event = event.groupby(group, observed=True).mean_hr_bpm.mean().reset_index()

    cells = manifest[(manifest.condition.astype(str).str.lower().eq("preict")) &
                     manifest.stratum.isin(FUNCTIONAL + EPILEPTIC) & manifest.semiology.notna()][
        ["participant", "stratum", "semiology"]].drop_duplicates()
    cells["event_type"] = np.where(cells.stratum.isin(FUNCTIONAL), "functional", "epileptic")
    cells["is_functional"] = cells.event_type.eq("functional").astype(int)
    cells["is_hypo"] = cells.semiology.eq("hypo").astype(int)
    cells["is_dd"] = cells.stratum.astype(str).str.startswith("DD").astype(int)
    interictal = (hr[hr.phase.eq("ii")].groupby(["participant", "file_path"], observed=True)
                  .mean_hr_bpm.mean().groupby("participant").mean().rename("mean_hr_bpm").reset_index()
                  .merge(cells.drop(columns="stratum"), on="participant", how="inner"))
    interictal["phase"] = "ii"
    participant = pd.concat([interictal[group + ["mean_hr_bpm"]], event], ignore_index=True)
    participant.to_csv(data_dir / "hr_participant_period_cells.csv", index=False)

    rows = []
    periods = (("ii", "interictal"), ("preict", "preictal"), ("ictal", "ictal"))
    for functional in (0, 1):
        for hypo in (0, 1):
            for period, period_label in periods:
                part = participant[(participant.is_functional.eq(functional)) &
                                   (participant.is_hypo.eq(hypo)) & participant.phase.eq(period)].copy()
                part["is_dd_centered"] = part.is_dd - part.is_dd.mean()
                label = f"{'FS' if functional else 'ES'} {'hypo' if hypo else 'hyper'} {period_label}"
                model = fit_conjugate(part, "mean_hr_bpm", ["is_dd_centered"],
                                      f"dense RR HR cell mean: {label}", args, prior_var=10000)
                rows.append(contrast(model, {"Intercept": 1}, label) |
                            {"group": "FS" if functional else "ES", "semiology": "hypo" if hypo else "hyper",
                             "period": period_label})
    summary = pd.DataFrame(rows); summary.to_csv(data_dir / "hr_bayesian_cells.csv", index=False)
    return participant, summary


def simple_coverage(results_dir: Path, manifest: pd.DataFrame, data_dir: Path, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(results_dir / "participant_condition_state_backfit_metrics.csv")
    frame = frame[frame.backfit_method.eq("hard") & frame.condition.astype(str).str.lower().isin(("preict", "transictal"))]
    frame = add_event_metadata(frame, manifest)
    frame = frame[frame.event_type.isin(("functional", "epileptic")) & frame.semiology.notna() & frame.template_label.isin(ALL_STATES)].copy()
    frame = frame[pd.to_numeric(frame.template_match_abs_correlation, errors="coerce").ge(STATE_REFERENCE_MIN_CORRELATION)]
    frame["is_ictal"] = frame.condition.astype(str).str.lower().eq("transictal").astype(int)
    frame["logit_coverage"] = np.log(np.clip(frame.occupancy, 1e-5, 1 - 1e-5) / np.clip(1 - frame.occupancy, 1e-5, 1))
    summaries = []
    for state in ALL_STATES:
        rows = descriptive_cell_summaries(frame[frame.template_label.eq(state)], "logit_coverage",
                                          f"coverage state {state} cell mean", args, prior_var=25, transform=expit)
        for result in rows:
            result["microstate"] = state; summaries.append(result)
    summary = pd.DataFrame(summaries); summary.to_csv(data_dir / "coverage_bayesian_cells.csv", index=False)
    return frame, summary


def autonomic_trends(native: pd.DataFrame, manifest: pd.DataFrame, data_dir: Path, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = ["csi", "cpi"]
    preictal = add_event_metadata(native[native.condition.eq("preict")], manifest)
    preictal["relative_s"] = preictal.event_time_s - preictal.record_duration_s
    preictal = preictal[preictal.relative_s.between(-60, 0) &
                        preictal.event_type.isin(("functional", "epileptic")) & preictal.semiology.notna()]
    event_group = ["participant", "event_key", "event_type", "semiology", "is_functional", "is_hypo", "is_dd"]
    preictal = preictal.groupby(event_group, observed=True)[values].mean().reset_index()
    cell_group = ["participant", "event_type", "semiology", "is_functional", "is_hypo", "is_dd"]
    preictal = preictal.groupby(cell_group, observed=True)[values].mean().reset_index()
    preictal["period"] = "preictal"

    cells = manifest[(manifest.condition.astype(str).str.lower().eq("preict")) &
                     manifest.stratum.isin(FUNCTIONAL + EPILEPTIC) & manifest.semiology.notna()][
        ["participant", "stratum", "semiology"]].drop_duplicates()
    cells["event_type"] = np.where(cells.stratum.isin(FUNCTIONAL), "functional", "epileptic")
    cells["is_functional"] = cells.event_type.eq("functional").astype(int)
    cells["is_hypo"] = cells.semiology.eq("hypo").astype(int)
    cells["is_dd"] = cells.stratum.astype(str).str.startswith("DD").astype(int)
    interictal = (native[native.condition.eq("ii")].groupby(["participant", "file_path"], observed=True)[values].mean()
                  .groupby("participant").mean().reset_index()
                  .merge(cells.drop(columns="stratum"), on="participant", how="inner"))
    interictal["period"] = "interictal"

    participant = pd.concat([interictal[cell_group + ["period", *values]],
                             preictal[cell_group + ["period", *values]]], ignore_index=True)
    matched = (participant.groupby(["participant", "event_type", "semiology"], observed=True).period.nunique()
               .loc[lambda count: count.eq(2)].rename("matched").reset_index())
    participant = participant.merge(matched, on=["participant", "event_type", "semiology"], how="inner").drop(columns="matched")
    participant.to_csv(data_dir / "autonomic_level_participant_cells.csv", index=False)

    participant["is_preictal"] = participant.period.eq("preictal").astype(int)
    participant["is_dd_centered"] = participant.is_dd - participant.is_dd.mean()
    for index in ("csi", "cpi"):
        participant[f"log_{index}"] = np.log(participant[index].where(participant[index].gt(0)))
    cell_names = {}
    for functional in (0, 1):
        for hypo in (0, 1):
            for preict in (0, 1):
                name = f"cell_{functional}_{hypo}_{preict}"; cell_names[(functional, hypo, preict)] = name
                participant[name] = (participant.is_functional.eq(functional) & participant.is_hypo.eq(hypo) &
                                     participant.is_preictal.eq(preict)).astype(int)

    def weights(coefficient) -> dict[str, float]:
        return {name: float(coefficient(*key)) for key, name in cell_names.items() if coefficient(*key)}

    contrast_specs = (
        ("FS minus ES (marginal)", lambda f, h, p: (1 if f else -1) / 4),
        ("Hypomotor minus hypermotor (marginal)", lambda f, h, p: (1 if h else -1) / 4),
        ("Preictal minus interictal (marginal)", lambda f, h, p: (1 if p else -1) / 4),
        ("FS-ES difference in preictal change", lambda f, h, p: (1 if f == p else -1) / 2),
        ("Hypo-hyper difference in preictal change", lambda f, h, p: (1 if h == p else -1) / 2),
        ("Group by semiology interaction", lambda f, h, p: (1 if f == h else -1) / 2),
        ("Group by semiology by period interaction", lambda f, h, p: 1 if (f + h + p) % 2 else -1),
    )
    cell_rows, contrast_rows = [], []
    for index in ("csi", "cpi"):
        model = fit_bayes(participant, f"log_{index}", [*cell_names.values(), "is_dd_centered"],
                          f"{index.upper()} autonomic group semiology period levels", data_dir, args,
                          intercept=False, event_column=None, prior_var=25)
        for (functional, hypo, preict), name in cell_names.items():
            period = "preictal" if preict else "interictal"
            label = f"{'FS' if functional else 'ES'} {'hypo' if hypo else 'hyper'} {period}"
            cell = participant[(participant.is_functional.eq(functional)) &
                               (participant.is_hypo.eq(hypo)) & participant.is_preictal.eq(preict)]
            cell_rows.append(contrast(model, {name: 1}, label, transform=np.exp) |
                             {"index": index.upper(), "group": "FS" if functional else "ES",
                               "semiology": "hypo" if hypo else "hyper", "period": period,
                               "n_rows": len(cell), "n_participants": cell.participant.nunique(),
                               "probability_gt_zero": np.nan, "probability_lt_zero": np.nan})
        for label, coefficient in contrast_specs:
            result = contrast(model, weights(coefficient), label) | {"index": index.upper()}
            result |= {"ratio_median": math.exp(result["median"]),
                       "ratio_ci95_low": math.exp(result["ci95_low"]),
                       "ratio_ci95_high": math.exp(result["ci95_high"])}
            contrast_rows.append(result)
        for functional in (0, 1):
            for hypo in (0, 1):
                label = f"{'FS' if functional else 'ES'} {'hypo' if hypo else 'hyper'}: preictal minus interictal"
                cell = participant[(participant.is_functional.eq(functional)) & participant.is_hypo.eq(hypo)]
                result = (contrast(model, {cell_names[(functional, hypo, 1)]: 1,
                                           cell_names[(functional, hypo, 0)]: -1}, label) |
                          {"index": index.upper(), "n_rows": len(cell),
                           "n_participants": cell.participant.nunique()})
                result |= {"ratio_median": math.exp(result["median"]),
                           "ratio_ci95_low": math.exp(result["ci95_low"]),
                           "ratio_ci95_high": math.exp(result["ci95_high"])}
                contrast_rows.append(result)
    cells_summary, contrasts = pd.DataFrame(cell_rows), pd.DataFrame(contrast_rows)
    cells_summary.to_csv(data_dir / "autonomic_level_bayesian_cells.csv", index=False)
    contrasts.to_csv(data_dir / "autonomic_level_bayesian_contrasts.csv", index=False)
    return participant, contrasts


def transition_data(results_dir: Path, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = tuple(load_global_labels(results_dir)); baselines = statewise.interictal_baselines(results_dir / "participant_condition_transition_metrics.csv")
    windows, late, _, _ = statewise.preictal_metrics(results_dir, manifest, labels, baselines, 5)
    post, _ = statewise.postictal_metrics(results_dir, manifest, labels, baselines, 5)
    return windows[windows.template_label.eq("C")], late[late.template_label.eq("C")], post[post.template_label.eq("C")]


def state_reference_qc(results_dir: Path) -> pd.DataFrame:
    metrics = pd.read_csv(results_dir / "participant_condition_state_backfit_metrics.csv",
                          usecols=["participant", "template_label", "template_match_abs_correlation"])
    metrics["template_match_abs_correlation"] = pd.to_numeric(metrics.template_match_abs_correlation, errors="coerce")
    qc = (metrics.groupby(["participant", "template_label"], observed=True).template_match_abs_correlation
          .median().reset_index())
    qc["retained"] = qc.template_match_abs_correlation.ge(STATE_REFERENCE_MIN_CORRELATION)
    return qc


def dd_transition_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    dd = frame[frame.stratum.str.startswith("DD")].copy()
    dd["event_type"] = np.where(dd.is_functional.eq(1), "functional", "epileptic")
    counts = dd.groupby(["participant", "event_type"], observed=True).size().unstack(fill_value=0)
    counts = counts.reindex(columns=["epileptic", "functional"], fill_value=0)
    counts = counts[counts.gt(0).all(axis=1)]
    means = (dd.groupby(["participant", "event_type"], observed=True)
             [["js_from_II", "log_js", "kappa", "log_kappa", "is_hypo_raw", "log_transitions"]]
             .mean().unstack("event_type").reindex(counts.index))
    pairs = pd.DataFrame({"participant": counts.index,
                          "n_epileptic_events": counts.epileptic.to_numpy(int),
                          "n_functional_events": counts.functional.to_numpy(int)})
    for source, name in (("js_from_II", "js"), ("log_js", "log_js"),
                         ("kappa", "kappa"), ("log_kappa", "log_kappa"),
                         ("is_hypo_raw", "hypo_fraction"), ("log_transitions", "log_transitions")):
        pairs[f"{name}_epileptic"] = means[(source, "epileptic")].to_numpy(float)
        pairs[f"{name}_functional"] = means[(source, "functional")].to_numpy(float)
        pairs[f"{name}_fs_minus_es"] = pairs[f"{name}_functional"] - pairs[f"{name}_epileptic"]
    return pairs


def main_transition_models(windows: pd.DataFrame, data_dir: Path, args) -> tuple[pd.DataFrame, dict]:
    frame = windows[windows.window_end_to_ictus_s.eq(0) & windows.semiology.notna()].copy()
    frame["is_functional"] = frame.stratum.isin(FUNCTIONAL).astype(int); frame["is_hypo_raw"] = frame.semiology.eq("hypo").astype(float)
    frame["is_hypo"] = frame.is_hypo_raw - frame.is_hypo_raw.mean(); frame["is_dd"] = frame.stratum.str.startswith("DD").astype(float)
    frame["is_dd"] -= frame.is_dd.mean(); frame["functional_x_hypo"] = frame.is_functional * frame.is_hypo
    frame["log_transitions"] = np.log(frame.outgoing_transitions.clip(lower=1)); frame["log_transitions"] -= frame.log_transitions.mean()
    frame["log_js"] = np.log(frame.js_from_II.where(frame.js_from_II.gt(0)))
    rows, models = [], {}
    predictors = ["is_functional", "is_hypo", "functional_x_hypo", "is_dd", "log_transitions"]
    for outcome in ("log_js", "log_kappa"):
        model = fit_bayes(frame, outcome, predictors, f"C-origin {outcome} FS ES and semiology", data_dir, args,
                          event_column=None, prior_var=16); models[outcome] = model
        rows.append(contrast(model, {"is_functional": 1}, "FS minus ES"))
        rows.append(contrast(model, {"is_hypo": 1, "functional_x_hypo": 1}, "Hypomotor minus hypermotor within FS"))
    summary = pd.DataFrame(rows); summary["outcome"] = ["JS", "JS", "kappa", "kappa"]
    summary.to_csv(data_dir / "c_origin_primary_bayesian.csv", index=False)

    non_dd = frame[~frame.stratum.str.startswith("DD")].copy()
    non_dd["is_hypo"] = non_dd.is_hypo_raw - non_dd.is_hypo_raw.mean()
    non_dd["functional_x_hypo"] = non_dd.is_functional * non_dd.is_hypo
    non_dd["log_transitions"] = np.log(non_dd.outgoing_transitions.clip(lower=1))
    non_dd["log_transitions"] -= non_dd.log_transitions.mean()
    pairs = dd_transition_pairs(frame)
    for covariate in ("hypo_fraction", "log_transitions"):
        delta = f"{covariate}_fs_minus_es"
        pairs[f"{delta}_centered"] = pairs[delta] - pairs[delta].mean()
    pairs.to_csv(data_dir / "c_origin_dd_paired_values.csv", index=False)
    subsidiary_rows = []
    for outcome, label in (("log_js", "JS"), ("log_kappa", "kappa")):
        model = fit_bayes(non_dd, outcome, ["is_functional", "is_hypo", "functional_x_hypo", "log_transitions"],
                          f"C-origin {outcome} FS ES excluding DD", data_dir, args,
                          event_column=None, prior_var=16)
        models[f"{outcome}_non_dd"] = model
        subsidiary_rows.append(contrast(model, {"is_functional": 1}, "FS minus ES excluding DD") | {"outcome": label})
        difference = f"{outcome}_fs_minus_es"
        paired = pairs.dropna(subset=[difference])
        model = fit_conjugate(paired, difference,
                              ["hypo_fraction_fs_minus_es_centered", "log_transitions_fs_minus_es_centered"],
                              f"C-origin {outcome} DD paired FS ES", args, prior_var=16)
        models[f"{outcome}_dd_paired"] = model
        subsidiary_rows.append(contrast(model, {"Intercept": 1}, "FS minus ES within DD (paired)") | {"outcome": label})
    subsidiary = pd.DataFrame(subsidiary_rows)
    subsidiary["ratio_median"] = np.exp(subsidiary["median"])
    subsidiary["ratio_ci95_low"] = np.exp(subsidiary.ci95_low)
    subsidiary["ratio_ci95_high"] = np.exp(subsidiary.ci95_high)
    subsidiary.to_csv(data_dir / "c_origin_subsidiary_bayesian.csv", index=False)
    return frame, models


def recovery_pairs(late: pd.DataFrame, post: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    pre = add_event_metadata(late, manifest); after = add_event_metadata(post, manifest)
    keys = ["participant", "event_key", "semiology", "is_hypo", "is_dd"]
    pre = pre[pre.event_type.eq("functional")].groupby(keys, observed=True).js_from_II.mean().rename("pre_js").reset_index()
    after = after[after.event_type.eq("functional")].groupby(keys, observed=True).js_from_II.mean().rename("post_js").reset_index()
    paired = pre.merge(after, on=keys, validate="one_to_one"); paired["log_post_pre_js"] = np.log(paired.post_js / paired.pre_js)
    return paired[np.isfinite(paired[["pre_js", "post_js", "log_post_pre_js"]]).all(axis=1)].copy()


def recovery_model(late: pd.DataFrame, post: pd.DataFrame, manifest: pd.DataFrame, data_dir: Path, args) -> pd.DataFrame:
    paired = recovery_pairs(late, post, manifest)
    paired.to_csv(data_dir / "c_origin_js_recovery_paired_events.csv", index=False)
    paired["is_hypo_centered"] = paired.is_hypo - paired.is_hypo.mean(); paired["is_dd_centered"] = paired.is_dd - paired.is_dd.mean()
    model = fit_bayes(paired, "log_post_pre_js", ["is_hypo_centered", "is_dd_centered"], "C-origin JS postictal recovery",
                      data_dir, args, event_column=None, prior_var=16)
    rows = [contrast(model, {"Intercept": 1}, "Overall postictal minus preictal"),
            contrast(model, {"Intercept": 1, "is_hypo_centered": -paired.is_hypo.mean()}, "Hypermotor postictal minus preictal"),
            contrast(model, {"Intercept": 1, "is_hypo_centered": 1 - paired.is_hypo.mean()}, "Hypomotor postictal minus preictal"),
            contrast(model, {"is_hypo_centered": 1}, "Hypomotor minus hypermotor recovery")]
    summary = pd.DataFrame(rows); summary.to_csv(data_dir / "c_origin_js_recovery.csv", index=False)
    return summary


def peak_metrics(table: pd.DataFrame) -> pd.DataFrame:
    transitions = [f"t_{i}_{j}" for i in range(1, 7) for j in range(1, 7)]
    baseline = {}
    for participant, part in table[table.condition.eq("ii")].groupby("participant", observed=True):
        counts = part[transitions].sum().to_numpy(float).reshape(6, 6); probs = counts + .5; np.fill_diagonal(probs, 0)
        probs /= probs.sum(axis=1, keepdims=True); baseline[str(participant)] = (probs, counts.sum(axis=1))
    rows = []
    for row in table.itertuples(index=False):
        if row.n_gfp_peaks < 30 or str(row.participant) not in baseline: continue
        counts = np.array([[getattr(row, f"t_{i}_{j}") for j in range(1, 7)] for i in range(1, 7)], float)
        b0, ii_n = baseline[str(row.participant)]; enough = counts[5].sum() >= 5 and ii_n[5] >= 5
        logk = ace.fit_kappa(counts, b0, (5,), 5)[0] if enough else np.nan
        p = (counts[5, :5] + .5) / (counts[5, :5].sum() + 2.5)
        q = b0[5, :5] / b0[5, :5].sum()
        logodds = {f"c_logodds_{ALL_STATES[index]}": math.log((p[index] / p[4]) / (q[index] / q[4]))
                   for index in range(4)} if enough else {f"c_logodds_{ALL_STATES[index]}": np.nan for index in range(4)}
        rows.append({"participant": row.participant, "file_path": row.file_path, "event_key": row.event_key,
                     "condition": row.condition, "event_type": row.event_type, "semiology": row.semiology,
                     "window_index": row.window_index, "window_start_s": row.window_start_s, "window_end_s": row.window_end_s,
                     "window_end_to_ictus_s": row.window_end_to_ictus_s, "c_expression": row.n_C / row.n_gfp_peaks,
                     "c_log_kappa": logk, "c_js": ace.js_divergence(counts, b0, (5,)) if enough else np.nan,
                     "c_outgoing": counts[5].sum(), **logodds})
    return pd.DataFrame(rows)


def common_windows(peaks: pd.DataFrame, native: pd.DataFrame) -> pd.DataFrame:
    lookup = {key: value for key, value in native.groupby(["participant", "file_path"], observed=True)}; rows = []
    for row in peaks.itertuples(index=False):
        part = lookup.get((row.participant, row.file_path))
        if part is None: continue
        use = part[part.event_time_s.between(row.window_start_s, row.window_end_s, inclusive="left")]
        if len(use) < 30 or use.event_time_s.max() - use.event_time_s.min() < 45: continue
        item = row._asdict()
        for column in ("csi_z", "cpi_z", "mean_rr_s_z", "mean_hr_bpm_z", "rr_first_difference_s_z",
                       "rr_acceleration_s_z", "rmssd_s_z", "rr_surprise_z", "mean_hr_bpm"):
            item[column] = use[column].mean()
        item["n_native"] = len(use); rows.append(item)
    return pd.DataFrame(rows)


def correlation_models(common: pd.DataFrame, functional_participants: set[str], data_dir: Path, args) -> pd.DataFrame:
    frame = common[common.participant.astype(str).isin(functional_participants) & common.condition.isin(("ii", "preict"))].copy()
    indices = {"csi_z": "CSI", "cpi_z": "CPI", "mean_rr_s_z": "mean RR",
               "rr_first_difference_s_z": "RR first difference", "rr_acceleration_s_z": "RR acceleration",
               "rmssd_s_z": "RMSSD", "rr_surprise_z": "RR surprise"}
    frame["is_preict"] = frame.condition.eq("preict").astype(int)
    for index in indices:
        frame[f"{index}_preict"] = frame[index] * frame.is_preict
    frame["log_transitions"] = np.log(frame.c_outgoing.clip(lower=1))
    frame["log_transitions"] -= frame.log_transitions.mean(); frame["log_js"] = np.log(frame.c_js.where(frame.c_js.gt(0)))
    frame["is_dd_centered"] = frame.is_dd - frame.is_dd.mean()
    rows = []
    for outcome in ("log_js", "c_log_kappa"):
        predictors = [*indices, "is_preict", *(f"{index}_preict" for index in indices),
                      "log_transitions", "is_dd_centered"]
        model = fit_bayes(frame, outcome, predictors,
                          f"FS C-origin {outcome} autonomic association", data_dir, args, prior_var=16)
        for index, label in indices.items():
            rows.append(contrast(model, {index: 1}, f"{label} association in II") | {"outcome": outcome})
            rows.append(contrast(model, {index: 1, f"{index}_preict": 1}, f"{label} association preictally") | {"outcome": outcome})
            rows.append(contrast(model, {f"{index}_preict": 1}, f"{label} preictal minus II association") | {"outcome": outcome})
    summary = pd.DataFrame(rows); summary.to_csv(data_dir / "fs_autonomic_c_origin_correlations.csv", index=False)
    return summary


def within_record_z(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy(); grouped = out.groupby(["participant", "file_path"], observed=True)
    for column in columns:
        out[f"{column}_wz"] = (out[column] - grouped[column].transform("mean")) / grouped[column].transform("std").where(lambda value: value > 0)
    return out


def make_lags(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for _, part in frame.groupby(["participant", "file_path"], observed=True, sort=False):
        part = part.sort_values("window_start_s").copy()
        gap = part.window_start_s.diff();
        for column in columns: part[f"{column}_lag"] = part[column].shift(1)
        rows.append(part[gap.eq(60)])
    return pd.concat(rows, ignore_index=True)


def flow_models(common: pd.DataFrame, functional_participants: set[str], data_dir: Path, args) -> pd.DataFrame:
    variables = ["c_expression", "c_js", "c_log_kappa", "csi_z", "cpi_z"]
    frame = within_record_z(common[common.condition.isin(("ii", "preict"))], variables)
    z = [f"{name}_wz" for name in variables]; lagged = make_lags(frame, z)
    rows = []
    specs = (("autonomic to C coverage", "c_expression_wz", ["c_expression_wz_lag", "csi_z_wz_lag", "cpi_z_wz_lag"], ("csi_z_wz_lag", "cpi_z_wz_lag")),
             ("autonomic to C-origin JS", "c_js_wz", ["c_js_wz_lag", "csi_z_wz_lag", "cpi_z_wz_lag"], ("csi_z_wz_lag", "cpi_z_wz_lag")),
             ("autonomic to C-origin kappa", "c_log_kappa_wz", ["c_log_kappa_wz_lag", "csi_z_wz_lag", "cpi_z_wz_lag"], ("csi_z_wz_lag", "cpi_z_wz_lag")),
             ("C coverage to C-origin JS", "c_js_wz", ["c_js_wz_lag", "c_expression_wz_lag"], ("c_expression_wz_lag",)),
             ("C coverage to C-origin kappa", "c_log_kappa_wz", ["c_log_kappa_wz_lag", "c_expression_wz_lag"], ("c_expression_wz_lag",)),
             ("C-origin grammar to C coverage", "c_expression_wz", ["c_expression_wz_lag", "c_js_wz_lag", "c_log_kappa_wz_lag"], ("c_js_wz_lag", "c_log_kappa_wz_lag")),
             ("C coverage to CSI", "csi_z_wz", ["csi_z_wz_lag", "cpi_z_wz_lag", "c_expression_wz_lag"], ("c_expression_wz_lag",)),
             ("C coverage to CPI", "cpi_z_wz", ["cpi_z_wz_lag", "csi_z_wz_lag", "c_expression_wz_lag"], ("c_expression_wz_lag",)),
             ("C-origin grammar to CSI", "csi_z_wz", ["csi_z_wz_lag", "cpi_z_wz_lag", "c_js_wz_lag", "c_log_kappa_wz_lag"], ("c_js_wz_lag", "c_log_kappa_wz_lag")),
             ("C-origin grammar to CPI", "cpi_z_wz", ["cpi_z_wz_lag", "csi_z_wz_lag", "c_js_wz_lag", "c_log_kappa_wz_lag"], ("c_js_wz_lag", "c_log_kappa_wz_lag")))
    target_labels = {"csi_z_wz_lag": "CSI", "cpi_z_wz_lag": "CPI",
                     "c_expression_wz_lag": "C coverage", "c_js_wz_lag": "C-origin JS",
                     "c_log_kappa_wz_lag": "C-origin kappa"}
    for label, outcome, predictors, targets in specs:
        model = fit_bayes(lagged, outcome, predictors, f"all participants flow {label}", data_dir, args, prior_var=9)
        for target in targets: rows.append(contrast(model, {target: 1}, f"All participants: {label} ({target_labels[target]})"))

    fs = lagged[lagged.participant.astype(str).isin(functional_participants)].copy()
    fs["is_last_minute"] = (fs.condition.eq("preict") & fs.window_end_to_ictus_s.eq(0)).astype(int)
    fs = fs[fs.condition.eq("ii") | fs.is_last_minute.eq(1)]
    for label, outcome, predictors, targets in specs:
        expanded = list(predictors) + ["is_last_minute"]
        for predictor in predictors:
            name = f"last_x_{predictor}"; fs[name] = fs.is_last_minute * fs[predictor]; expanded.append(name)
        model = fit_bayes(fs, outcome, expanded, f"FS II versus last-minute flow {label}", data_dir, args, prior_var=9)
        for target in targets:
            target_label = target_labels[target]
            rows.append(contrast(model, {target: 1}, f"FS II: {label} ({target_label})"))
            rows.append(contrast(model, {target: 1, f"last_x_{target}": 1}, f"FS last minute: {label} ({target_label})"))
            rows.append(contrast(model, {f"last_x_{target}": 1}, f"FS last minute minus II: {label} ({target_label})"))
    summary = pd.DataFrame(rows); summary.to_csv(data_dir / "bayesian_predictive_flow.csv", index=False)
    return summary


def demographics(manifest: pd.DataFrame, database_path: Path, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    participants = manifest[["participant", "patient_id", "group"]].drop_duplicates("participant").copy()
    database = pd.read_csv(database_path, low_memory=False)
    demo = database[["patient_id", "age", "gender"]].drop_duplicates("patient_id")
    participants["patient_id"] = participants.patient_id.astype(str); demo["patient_id"] = demo.patient_id.astype(str)
    participants = participants.merge(demo, on="patient_id", how="left"); participants["age"] = pd.to_numeric(participants.age, errors="coerce")
    participants["gender"] = participants.gender.astype(str).str.strip().str.lower().replace({"nan": "unknown", "f": "female", "m": "male"})
    rows = []
    for group, part in participants.groupby("group", observed=True):
        age = part.age.dropna(); genders = part.gender.value_counts()
        rows.append({"group": group, "participants": len(part), "age_available": len(age), "age_median": age.median(),
                     "age_q1": age.quantile(.25), "age_q3": age.quantile(.75), "female": genders.get("female", 0),
                     "male": genders.get("male", 0), "gender_unknown": genders.get("unknown", 0)})
    summary = pd.DataFrame(rows)
    events = (manifest[manifest.condition.astype(str).str.lower().isin(("preict", "transictal"))]
              .assign(semiology=ace.normalise_semiology(manifest.hypo_hyper))
              .groupby(["group", "condition", "seizure_type", "semiology"], observed=True, dropna=False)
              .agg(records=("file_path", "size"), participants=("participant", "nunique")).reset_index())
    participants.to_csv(data_dir / "participant_demographics.csv", index=False); summary.to_csv(data_dir / "demographic_summary.csv", index=False)
    events.to_csv(data_dir / "event_characteristics.csv", index=False)
    return summary, events


def prepare_preictal_kappa_js(windows: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    frame = windows[np.isfinite(windows.log_kappa) & np.isfinite(windows.js_from_II)].copy()
    frame["group"] = np.select([frame.stratum.isin(FUNCTIONAL), frame.stratum.isin(EPILEPTIC)], ["FS", "ES"], default="")
    frame = frame[frame.group.ne("") & frame.semiology.isin(("hyper", "hypo"))]
    event = (frame.groupby(["participant", "file_path", "group", "semiology"], observed=True)
             [["log_kappa", "js_from_II"]].mean().reset_index())
    participant = (event.groupby(["participant", "group", "semiology"], observed=True)
                   .agg(log_kappa=("log_kappa", "mean"), js_from_II=("js_from_II", "mean"),
                        n_events=("file_path", "nunique")).reset_index())
    participant["kappa"] = np.exp(participant.log_kappa)
    assert participant.groupby(["participant", "group", "semiology"], observed=True).size().max() == 1
    participant = participant.sort_values(["group", "semiology", "participant"])
    participant.to_csv(data_dir / "preictal_kappa_js_participant.csv", index=False)

    rows = []
    for group in ("FS", "ES"):
        for semiology in ("hyper", "hypo"):
            part = participant[participant.group.eq(group) & participant.semiology.eq(semiology)]
            if len(part) < 3:
                continue
            slope, intercept = np.polyfit(part.log_kappa, part.js_from_II, 1)
            rows.append({"group": group, "semiology": semiology, "n_participants": len(part),
                         "slope_per_log_kappa": slope, "intercept": intercept,
                         "pearson_r": np.corrcoef(part.log_kappa, part.js_from_II)[0, 1]})
    pd.DataFrame(rows).to_csv(data_dir / "preictal_kappa_js_lines.csv", index=False)
    return participant


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    use = frame[columns].copy()
    for column in use.select_dtypes(include="number"):
        use[column] = use[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    return "| " + " | ".join(columns) + " |\n|" + "|".join(["---"] * len(columns)) + "|\n" + "\n".join(
        "| " + " | ".join(map(str, row)) + " |" for row in use.itertuples(index=False, name=None))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def self_check() -> None:
    rng = np.random.default_rng(6); rows = []
    for participant in range(30):
        u = rng.normal(0, .3)
        for event in range(3):
            x = rng.normal(); rows.append({"participant": f"p{participant}", "event_key": f"p{participant}_{event}",
                                           "x": x, "y": u + .6 * x + rng.normal(0, .4)})
    frame = pd.DataFrame(rows)
    class Args: iterations=2200; burn=700; thin=1; chains=4; seed=9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); model = fit_bayes(frame, "y", ["x"], "self check", Path.cwd(), Args())
    result = contrast(model, {"x": 1}, "x"); assert result["median"] > .35 and result["probability_gt_zero"] > .98
    dd = pd.DataFrame([
        {"participant": "p0", "stratum": "DD epileptic", "is_functional": 0, "js_from_II": .1,
         "log_js": math.log(.1), "kappa": 1., "log_kappa": 0., "is_hypo_raw": 0., "log_transitions": 1.},
        {"participant": "p0", "stratum": "DD functional", "is_functional": 1, "js_from_II": .2,
         "log_js": math.log(.2), "kappa": 2., "log_kappa": math.log(2), "is_hypo_raw": 1., "log_transitions": 2.},
        {"participant": "p1", "stratum": "DD functional", "is_functional": 1, "js_from_II": .3,
         "log_js": math.log(.3), "kappa": 1., "log_kappa": 0., "is_hypo_raw": 0., "log_transitions": 1.},
    ])
    pairs = dd_transition_pairs(dd)
    assert pairs.participant.tolist() == ["p0"] and abs(pairs.log_js_fs_minus_es.iloc[0] - math.log(2)) < 1e-12
    priorities.self_check()
    Path("posterior_self_check.npz").unlink(missing_ok=True)


def run(args) -> dict:
    data_dir = args.out_dir / "data"; data_dir.mkdir(parents=True, exist_ok=True)
    manifest = attach_strata(pd.read_csv(args.manifest)); manifest["semiology"] = ace.normalise_semiology(manifest.hypo_hyper)
    native, hr, ecg_qc = read_ecg(manifest, args.ecg_qc_dir, args.original_ecg_qc_dir, data_dir, args.reuse_ecg)
    native = standardise_autonomic_to_ii(native); native.to_csv(data_dir / "autonomic_native_ii_standardised.csv.gz", index=False, compression="gzip")
    demographics_table, event_table = demographics(manifest, args.demographics_csv, data_dir)
    hr_cells, hr_summary = simple_hr(hr, manifest, data_dir, args)
    _, coverage_summary = simple_coverage(args.results_dir, manifest, data_dir, args)
    autonomic_cells, autonomic_summary = autonomic_trends(native, manifest, data_dir, args)
    reference_qc = state_reference_qc(args.results_dir); reference_qc.to_csv(data_dir / "state_reference_qc.csv", index=False)
    valid_c_participants = set(reference_qc.loc[reference_qc.template_label.eq("C") & reference_qc.retained, "participant"].astype(str))
    windows, late, post = transition_data(args.results_dir, manifest)
    windows = windows[windows.participant.astype(str).isin(valid_c_participants)]
    late = late[late.participant.astype(str).isin(valid_c_participants)]
    post = post[post.participant.astype(str).isin(valid_c_participants)]
    windows.to_csv(data_dir / "c_origin_preictal_windows.csv", index=False); late.to_csv(data_dir / "c_origin_late_preictal.csv", index=False); post.to_csv(data_dir / "c_origin_postictal.csv", index=False)
    prepare_preictal_kappa_js(windows, data_dir)
    main_transition_models(windows, data_dir, args)
    transition_summary = pd.read_csv(data_dir / "c_origin_primary_bayesian.csv")
    subsidiary_summary = pd.read_csv(data_dir / "c_origin_subsidiary_bayesian.csv")
    recovery_summary = recovery_model(late, post, manifest, data_dir, args)
    peak_table = add_event_metadata(pd.read_csv(args.peak_windows_csv), manifest)
    peaks = peak_metrics(peak_table)
    peaks = peaks[peaks.participant.astype(str).isin(valid_c_participants)]
    common = common_windows(peaks, native)
    dd_participants = set(manifest.loc[manifest.group.astype(str).str.upper().eq("DD"), "participant"].astype(str))
    functional_participants = set(manifest.loc[manifest.stratum.isin(FUNCTIONAL), "participant"].astype(str))
    common["is_dd"] = common.participant.astype(str).isin(dd_participants).astype(int)
    common["is_functional_participant"] = common.participant.astype(str).isin(functional_participants).astype(int)
    common.to_csv(data_dir / "autonomic_peak_microstate_common_windows.csv.gz", index=False, compression="gzip")
    association_summary = correlation_models(common, functional_participants, data_dir, args)
    flow_summary = flow_models(common, functional_participants, data_dir, args)
    feedback_summary = priorities.run(peak_table, reference_qc, common, data_dir, args.transition_draws,
                                      args.seed + 104729)
    raw_count_summary = pd.read_csv(data_dir / "raw_count_c_origin_bayesian.csv")
    specificity_summary = pd.read_csv(data_dir / "all_state_raw_count_specificity_bayesian.csv")
    route_summary = pd.read_csv(data_dir / "c_origin_route_destination_bayesian.csv")
    pseudo_summary = pd.read_csv(data_dir / "matched_interictal_pseudo_onset_bayesian.csv")
    matched_recovery_summary = pd.read_csv(data_dir / "c_origin_recovery_matched_bayesian.csv")
    trajectory_summary = pd.read_csv(data_dir / "c_origin_preictal_binned_trajectory_bayesian.csv")

    main_js = transition_summary[(transition_summary.outcome.eq("JS")) & transition_summary.contrast.eq("FS minus ES")].iloc[0]
    main_kappa = transition_summary[(transition_summary.outcome.eq("kappa")) & transition_summary.contrast.eq("FS minus ES")].iloc[0]
    semio_js = transition_summary[(transition_summary.outcome.eq("JS")) & transition_summary.contrast.str.startswith("Hypomotor")].iloc[0]
    non_dd_js = subsidiary_summary[(subsidiary_summary.outcome.eq("JS")) &
                                   subsidiary_summary.contrast.str.contains("excluding DD")].iloc[0]
    non_dd_kappa = subsidiary_summary[(subsidiary_summary.outcome.eq("kappa")) &
                                      subsidiary_summary.contrast.str.contains("excluding DD")].iloc[0]
    dd_js = subsidiary_summary[(subsidiary_summary.outcome.eq("JS")) &
                               subsidiary_summary.contrast.str.contains("within DD")].iloc[0]
    dd_kappa = subsidiary_summary[(subsidiary_summary.outcome.eq("kappa")) &
                                  subsidiary_summary.contrast.str.contains("within DD")].iloc[0]
    recovery = recovery_summary[recovery_summary.contrast.str.startswith("Overall")].iloc[0]
    report = ["# Focused FS/ES cardiac and microstate results", "", f"Completed: {datetime.now(timezone.utc).isoformat()}", "",
              "## Cohort and event characteristics", "", markdown_table(demographics_table, list(demographics_table.columns)), "",
              markdown_table(event_table, list(event_table.columns)), "", "## Descriptive comparisons", "",
              "### Dense RR-derived heart rate", "", markdown_table(hr_summary, ["contrast", "median", "ci95_low", "ci95_high", "n_participants"]), "",
              "![Dense HR](figures/dense_rr_hr.png)", "", "### Microstate coverage", "",
              markdown_table(coverage_summary, ["microstate", "contrast", "median", "ci95_low", "ci95_high", "n_participants"]), "",
              "![Coverage](figures/microstate_coverage.png)", "", "### Robust interictal and preictal autonomic levels", "",
              "Raw positive CSI/CPI levels retain between-patient information; Gaussian inference is on the log scale. Ictal estimates are excluded because of ECG noise. Interictal records and final-minute preictal events are first averaged equally within each participant cell.", "",
              markdown_table(autonomic_summary, ["index", "contrast", "median", "ci95_low", "ci95_high", "ratio_median", "ratio_ci95_low", "ratio_ci95_high", "probability_gt_zero", "n_participants"]), "",
              "![Robust interictal and preictal autonomic levels](figures/robust_autonomic_interictal_preictal.png)", "", "## Primary hypothesis tests", "",
              f"- Final-minute C-origin JS, FS minus ES: posterior log effect {main_js['median']:.3f} (95% CrI {main_js.ci95_low:.3f} to {main_js.ci95_high:.3f}); P(FS>ES)={main_js.probability_gt_zero:.3f}.",
              f"- Final-minute C-origin kappa, FS minus ES: posterior log effect {main_kappa['median']:.3f} (95% CrI {main_kappa.ci95_low:.3f} to {main_kappa.ci95_high:.3f}); P(FS>ES)={main_kappa.probability_gt_zero:.3f}.",
              f"- C-origin JS, hypomotor minus hypermotor FS: {semio_js['median']:.3f} (95% CrI {semio_js.ci95_low:.3f} to {semio_js.ci95_high:.3f}); P(hypo>hyper)={semio_js.probability_gt_zero:.3f}.",
              f"- Postictal correction of C-origin JS: log(post/pre) {recovery['median']:.3f} (95% CrI {recovery.ci95_low:.3f} to {recovery.ci95_high:.3f}); P(post<pre)={recovery.probability_lt_zero:.3f}.", "",
              markdown_table(transition_summary, ["outcome", "contrast", "median", "ci95_low", "ci95_high", "probability_gt_zero", "n_participants"]), "",
              "![C-origin primary](figures/c_origin_primary.png)", "", "![C-origin boxplots](figures/c_origin_final_minute_boxplots.png)", "",
              "![C-origin JS recovery boxplots](figures/c_origin_js_recovery_boxplots.png)", "",
              "### Raw-count and equal-information sensitivity", "",
              f"The confirmatory sensitivity starts from the five raw C-exit counts in each 60-second window. Jeffreys-Dirichlet draws propagate finite-count uncertainty for both the event window and participant-specific interictal reference; hierarchical Bayesian-bootstrap weights represent event and participant sampling. A second analysis thins both distributions to {priorities.FIXED_EXITS} exits.", "",
              markdown_table(raw_count_summary, ["estimand", "outcome", "contrast", "median", "ci95_low", "ci95_high", "probability_gt_zero", "n_participants"]), "",
              "![Raw-count C-origin analyses](figures/raw_count_c_origin.png)", "",
              "### Route, concentration and destination decomposition", "",
              markdown_table(route_summary, ["outcome", "destination", "contrast", "median", "ci95_low", "ci95_high", "probability_gt_zero"]), "",
              "![C-origin route and destinations](figures/c_origin_route_destinations.png)", "",
              "### Source-state specificity", "",
              markdown_table(specificity_summary, ["outcome", "source_state", "contrast", "median", "ci95_low", "ci95_high", "probability_gt_zero"]), "",
              "![All-state specificity](figures/all_state_specificity.png)", "",
              "### Matched interictal pseudo-onsets and equal-duration recovery", "",
              "Pseudo-onsets are matched within participant on recording time, GFP-peak count, C exits, C expression and autonomic-window availability. Movement/EMG measures were not present in the exports and therefore were not represented as if matched.", "",
              markdown_table(pseudo_summary, ["contrast", "median", "ci95_low", "ci95_high", "probability_gt_zero", "n_participants"]), "",
              markdown_table(matched_recovery_summary, ["estimand", "outcome", "contrast", "median", "ci95_low", "ci95_high", "probability_lt_zero", "n_participants"]), "",
              "![Matched pseudo-onsets](figures/matched_pseudo_onsets.png)", "", "### Time-resolved preictal trajectory", "",
              markdown_table(trajectory_summary[trajectory_summary.contrast.str.contains("FS")],
                             ["time_bin", "outcome", "contrast", "median", "ci95_low", "ci95_high", "probability_gt_zero", "n_participants"]), "",
              "![Preictal physiology trajectory](figures/preictal_physiology_trajectory.png)", "",
              "## Subsidiary FS versus ES comparisons", "",
              "The non-DD models repeat the pooled adjusted model after excluding DD. The within-DD models use one FS-minus-ES log difference per complete patient pair, adjusted for within-pair differences in semiology composition and transition count.", "",
              f"- Excluding DD, the FS/ES JS ratio was {non_dd_js.ratio_median:.3f} (95% CrI {non_dd_js.ratio_ci95_low:.3f} to {non_dd_js.ratio_ci95_high:.3f}); P(FS>ES)={non_dd_js.probability_gt_zero:.3f}.",
              f"- Excluding DD, the FS/ES kappa ratio was {non_dd_kappa.ratio_median:.3f} (95% CrI {non_dd_kappa.ratio_ci95_low:.3f} to {non_dd_kappa.ratio_ci95_high:.3f}); P(FS>ES)={non_dd_kappa.probability_gt_zero:.3f}.",
              f"- Within {int(dd_js.n_participants)} paired DD patients, the adjusted FS/ES JS ratio was {dd_js.ratio_median:.3f} (95% CrI {dd_js.ratio_ci95_low:.3f} to {dd_js.ratio_ci95_high:.3f}); P(FS>ES)={dd_js.probability_gt_zero:.3f}.",
              f"- Within {int(dd_kappa.n_participants)} paired DD patients, the adjusted FS/ES kappa ratio was {dd_kappa.ratio_median:.3f} (95% CrI {dd_kappa.ratio_ci95_low:.3f} to {dd_kappa.ratio_ci95_high:.3f}); P(FS>ES)={dd_kappa.probability_gt_zero:.3f}.", "",
              markdown_table(subsidiary_summary, ["outcome", "contrast", "median", "ci95_low", "ci95_high", "ratio_median", "ratio_ci95_low", "ratio_ci95_high", "probability_gt_zero", "n_participants"]), "",
              "![Subsidiary C-origin comparisons](figures/c_origin_subsidiary_raw.png)", "",
              "### Descriptive preictal κ–JS relationship", "",
              "Points are participant means after equal averaging within event and then across events; lines are descriptive regressions on log κ.", "",
              "![Preictal kappa versus JS](figures/preictal_kappa_vs_js.png)", "", "## Autonomic association with C-origin transitions in FS", "",
              "CSI and CPI are standardised to each participant's genuine interictal distribution. C-origin metrics here use the same GFP-peak backfit estimator in II and preictal records.", "",
              markdown_table(association_summary, ["outcome", "contrast", "median", "ci95_low", "ci95_high", "probability_gt_zero", "n_participants"]), "",
              "![Associations](figures/fs_autonomic_c_origin_correlations.png)", "", "![CPI versus kappa](figures/fs_cpi_kappa_regression.png)", "", "## Postictal recovery", "",
              markdown_table(recovery_summary, ["contrast", "median", "ci95_low", "ci95_high", "probability_lt_zero", "n_participants"]), "",
              "![Recovery](figures/c_origin_js_recovery.png)", "", "## Exploratory predictive information flow", "",
              "These reciprocal one-window lag models now distinguish C coverage from C-origin JS and kappa. They test temporal prediction after each outcome's own prior value and do not establish causal physiological direction.", "",
              markdown_table(flow_summary, ["contrast", "median", "ci95_low", "ci95_high", "probability_gt_zero", "n_participants"]), "",
              "![Flow](figures/bayesian_predictive_flow.png)", "", "![CSI predicts C](figures/fs_csi_to_c_regression.png)", "",
              f"Label-specific analyses require participant-reference absolute correlation >= {STATE_REFERENCE_MIN_CORRELATION:.2f}; {len(valid_c_participants)}/{manifest.participant.nunique()} participants met this criterion for C. All intervals are 95% Bayesian credible intervals. HR and coverage descriptive cells use exact conjugate Gaussian posteriors after equal event averaging within participant and include cohort adjustment. Autonomic level, hypothesis and window models use participant and, where identifiable, event random intercepts. Strong directional evidence is conventionally P>0.975 or P<0.025; estimates below that threshold remain uncertain."]
    (args.out_dir / "RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    summary = {"participants": int(manifest.participant.nunique()), "c_reference_valid_participants": len(valid_c_participants),
               "ecg_records_valid": int(ecg_qc.autonomic_valid.sum()),
               "hr_periods": sorted(hr_cells.phase.unique().tolist()),
               "autonomic_periods": sorted(autonomic_cells.period.unique().tolist()),
               "autonomic_level_participants": int(autonomic_cells.participant.nunique()),
               "common_windows": len(common), "main_c_js_probability_fs_gt_es": float(main_js.probability_gt_zero),
               "main_c_kappa_probability_fs_gt_es": float(main_kappa.probability_gt_zero),
               "non_dd_c_js_probability_fs_gt_es": float(non_dd_js.probability_gt_zero),
               "dd_paired_c_js_probability_fs_gt_es": float(dd_js.probability_gt_zero),
               "dd_complete_pairs": int(dd_js.n_participants),
               "recovery_probability_post_lt_pre": float(recovery.probability_lt_zero),
               **feedback_summary}
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (data_dir / "analysis_schema.json").write_text(json.dumps({"schema": ANALYSIS_SCHEMA,
        "state_reference_min_abs_correlation": STATE_REFERENCE_MIN_CORRELATION,
        "dense_hr_periods": ["ii", "preict", "ictal"],
        "autonomic_level_periods": ["interictal", "preictal"],
        "autonomic_level_model": "log raw CSI/CPI",
        "raw_count_transition_model": "Jeffreys-Dirichlet count posterior plus hierarchical event/participant Bayesian bootstrap",
        "primary_preictal_window": "final non-overlapping 60 seconds",
        "fixed_exit_sensitivity": priorities.FIXED_EXITS}, indent=2), encoding="utf-8")
    labels = pd.read_csv(args.results_dir / "global" / "global_manual_labels.csv").sort_values("state_index")
    label_text = ", ".join(f"{int(row.state_index)}={row.label}" for row in labels.itertuples())
    provenance_table = pd.read_csv(args.results_dir / "hierarchical_fit_summary.csv")
    provenance = provenance_table[provenance_table.level.ne("global")].iloc[0]
    failures = ecg_qc[~ecg_qc.autonomic_valid.astype(bool)].reason.value_counts()
    failure_text = "; ".join(f"{reason}: {count}" for reason, count in failures.items()) or "none"
    code_files = [Path(__file__), Path(__file__).with_name("analyse_candia_autonomic.py"),
                  Path(__file__).with_name("analyse_transition_priorities.py"),
                  Path(__file__).with_name("export_directional_peak_windows.m"),
                  Path(__file__).with_name("run_ace_manual_pipeline.ps1"), Path(__file__).with_name("run_vg_microstates.m")]
    code_hashes = "; ".join(f"{path.name}={file_sha256(path)}" for path in code_files)
    (args.out_dir / "AUDIT.txt").write_text("\n".join(["Focused FS_MS analysis", f"Completed UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Hierarchy: {args.results_dir}", f"Manifest: {args.manifest}", f"Peak windows: {args.peak_windows_csv}",
        f"Manual global labels in fitted-state order: {label_text}",
        f"SPM_MS commit: {provenance.spm_ms_commit}; SPM: {provenance.spm_version}; SPM commit: {provenance.spm_commit}; prior mode: {provenance.prior_mode}; prior weight: {provenance.spm_prior_weight}",
        "Descriptive HR and coverage cells: exact conjugate Gaussian posterior after equal event averaging within participant, with centred DD-cohort adjustment",
        "Autonomic level and hypothesis/window models: Bayesian Gaussian mixed models with participant and eligible event random intercepts", f"Chains={args.chains}; iterations={args.iterations}; burn={args.burn}; thin={args.thin}; seed={args.seed}",
        "Convergence rule: R-hat <=1.05 and ESS >=100 for every fixed and variance parameter; one deterministic doubled-length retry is allowed, otherwise the run stops",
        "Dense RR HR periods: all eligible interictal intervals, final preictal minute, and first ictal minute; records and events are equally averaged within participant cells",
        "CSI/CPI levels: raw robust trailing 15-second indices from continuous source-record R-peak trains; log-Gaussian II/final-preictal factorial model; ictal excluded",
        "CSI/CPI association models: standardised to each participant's interictal distribution",
        "Subsidiary transitions: non-DD adjusted refit plus equal-patient paired DD log differences adjusted for within-pair semiology composition and transition count",
        f"State-reference QC: absolute correlation >= {STATE_REFERENCE_MIN_CORRELATION:.2f} required for every label-specific analysis; C retained {len(valid_c_participants)}/{manifest.participant.nunique()} participants",
        f"ECG autonomic QC: {int(ecg_qc.autonomic_valid.astype(bool).sum())}/{len(ecg_qc)} records retained; exclusions: {failure_text}",
        "Primary transition estimator: full-sample hard backfit; exploratory II/preict estimator: GFP-peak hard backfit",
        f"Raw-count sensitivity: {args.transition_draws} Jeffreys-Dirichlet/Bayesian-bootstrap draws; fixed-duration final 60 seconds and fixed-{priorities.FIXED_EXITS}-C-exit estimands",
        "Pseudo-onsets: within-participant interictal matches on time, GFP peaks, C exits, C expression and autonomic availability; movement/EMG unavailable",
        f"Code SHA-256: {code_hashes}", f"Run log: {args.out_dir / 'pipeline_run.log'}", json.dumps(summary, indent=2)]) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path); parser.add_argument("--manifest", type=Path); parser.add_argument("--ecg-qc-dir", type=Path)
    parser.add_argument("--original-ecg-qc-dir", type=Path); parser.add_argument("--demographics-csv", type=Path)
    parser.add_argument("--peak-windows-csv", type=Path); parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--chains", type=int, default=4); parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--burn", type=int, default=1200); parser.add_argument("--thin", type=int, default=2); parser.add_argument("--seed", type=int, default=26072026)
    parser.add_argument("--transition-draws", type=int, default=1200)
    parser.add_argument("--reuse-ecg", action="store_true"); parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check: self_check(); print("self-check passed"); return
    if not all((args.results_dir, args.manifest, args.ecg_qc_dir, args.original_ecg_qc_dir, args.demographics_csv, args.peak_windows_csv, args.out_dir)):
        parser.error("all path arguments are required")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
