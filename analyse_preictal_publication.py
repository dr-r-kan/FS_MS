"""Publication-facing rolling analysis of functional-seizure microstates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import t, ttest_rel
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.multitest import multipletests


def norm_path(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace("/", "\\", regex=False).str.lower()


def attach_strata(manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest.copy()
    group = out["group"].astype(str).str.strip().str.upper()
    seizure = out["seizure_type"].astype(str).str.strip().str.upper()
    out["stratum"] = np.select(
        [group.eq("FS") & seizure.eq("FS"), group.eq("DD") & seizure.eq("FS"),
         group.eq("ES") & seizure.eq("ES"), group.eq("DD") & seizure.eq("ES")],
        ["FS functional", "DD functional", "ES epileptic", "DD epileptic"], default="other")
    out["file_key"] = norm_path(out["file_path"])
    return out


def rolling_one_group(frame: pd.DataFrame, source: str, scheme: str, window: int, step: int) -> list[dict]:
    frame = frame.sort_values("minutes_to_ictus", ascending=False)
    x = -pd.to_numeric(frame["minutes_to_ictus"], errors="coerce").to_numpy(float) * 60
    rows = []
    for end_to in range(0, 1200 - window + 1, step):
        keep = (x >= end_to) & (x < end_to + window)
        part = frame.loc[keep]
        duration = pd.to_numeric(part["bin_duration_s"], errors="coerce").fillna(0).to_numpy(float)
        if duration.sum() < .9 * window:
            continue
        coverage = pd.to_numeric(part["coverage"], errors="coerce").to_numpy(float)
        state_exposure = np.nansum(coverage * duration)
        row = {
            "source": source, "scheme": scheme, "window_s": window, "step_s": step,
            "window_end_to_ictus_s": end_to, "minutes_to_ictus": -end_to / 60,
            "coverage": state_exposure / duration.sum(),
            "template_match_abs_correlation": pd.to_numeric(part["template_match_abs_correlation"], errors="coerce").mean()
            if "template_match_abs_correlation" in part else np.nan,
        }
        if "occurrence_count" in part:
            occurrences = pd.to_numeric(part["occurrence_count"], errors="coerce").to_numpy(float)
            total_occurrences = np.nansum(occurrences)
            row["occurrence_rate_hz"] = total_occurrences / duration.sum()
            if "mean_dwell_s" in part:
                dwell = pd.to_numeric(part["mean_dwell_s"], errors="coerce").to_numpy(float)
                valid_dwell = np.isfinite(dwell) & np.isfinite(occurrences) & (occurrences > 0)
                denom_dwell = np.nansum(occurrences[valid_dwell])
                row["mean_dwell_s"] = (np.nansum(dwell[valid_dwell] * occurrences[valid_dwell]) / denom_dwell
                                       if denom_dwell > 0 else np.nan)
        for metric in ("mean_gfp", "mean_topographic_gmd"):
            if metric in part:
                value = pd.to_numeric(part[metric], errors="coerce").to_numpy(float)
                valid = np.isfinite(value) & np.isfinite(coverage)
                denom = np.nansum(coverage[valid] * duration[valid])
                row[metric] = np.nansum(value[valid] * coverage[valid] * duration[valid]) / denom if denom > 0 else np.nan
        rows.append(row)
    return rows


def filter_event_window_outliers(event: pd.DataFrame) -> pd.DataFrame:
    out = event.copy()
    keys = ["participant", "file_path", "source", "scheme", "backfit_method", "template_label", "stratum"]
    for metric in ("coverage", "mean_gfp"):
        if metric not in out:
            continue
        raw = f"{metric}_raw"
        flag = f"{metric}_outlier"
        out[raw] = pd.to_numeric(out[metric], errors="coerce")
        grouped = out.groupby(keys, observed=True, dropna=False)[raw]
        median = grouped.transform("median")
        sd = grouped.transform("std")
        count = grouped.transform("count")
        out[flag] = (out[raw].notna() & count.ge(4) & sd.gt(0) &
                     out[raw].sub(median).abs().gt(3 * sd))
        out.loc[out[flag], metric] = np.nan
    return out


def rolling_outlier_audit(event: pd.DataFrame) -> pd.DataFrame:
    group = ["metric", "scheme", "source", "template_label", "backfit_method", "stratum"]
    rows = []
    for metric in ("coverage", "mean_gfp"):
        raw, flag = f"{metric}_raw", f"{metric}_outlier"
        if raw not in event or flag not in event:
            continue
        part = event.assign(metric=metric, finite_window=event[raw].notna(), outlying_window=event[flag].fillna(False))
        audit = (part.groupby(group, observed=True, dropna=False)
                 .agg(finite_windows=("finite_window", "sum"), outlying_windows=("outlying_window", "sum"),
                      events=("file_path", "nunique"), participants=("participant", "nunique")).reset_index())
        audit["retained_windows"] = audit["finite_windows"] - audit["outlying_windows"]
        rows.append(audit)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def validate_event_window_outliers(event: pd.DataFrame) -> None:
    keys = ["participant", "file_path", "source", "scheme", "backfit_method", "template_label", "stratum"]
    for metric in ("coverage", "mean_gfp"):
        raw, flag = f"{metric}_raw", f"{metric}_outlier"
        if raw not in event or flag not in event:
            raise ValueError(f"Missing outlier audit fields for {metric}")
        grouped = event.groupby(keys, observed=True, dropna=False)[raw]
        median, sd, count = grouped.transform("median"), grouped.transform("std"), grouped.transform("count")
        expected = event[raw].notna() & count.ge(4) & sd.gt(0) & event[raw].sub(median).abs().gt(3 * sd)
        observed = event[flag].fillna(False).astype(bool)
        if not observed.equals(expected):
            raise ValueError(f"{metric} outlier flags do not match the locked one-pass within-event rule")
        if event.loc[observed, metric].notna().any():
            raise ValueError(f"Flagged {metric} values were not excluded")
        retained = event[raw].notna() & ~observed
        if not np.allclose(event.loc[retained, metric], event.loc[retained, raw], equal_nan=True):
            raise ValueError(f"Unflagged {metric} values changed during filtering")


def build_rolling(time_csv: Path, source: str, targets: set[str], metadata: pd.DataFrame,
                  include_primary: bool = True, include_final: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = pd.read_csv(time_csv, nrows=0).columns.tolist()
    wanted = [c for c in ["participant", "condition", "group", "file_path", "backfit_method",
                          "template_label", "bin_from_ictus", "bin_duration_s", "coverage", "mean_gfp",
                          "mean_topographic_gmd", "occurrence_count", "mean_dwell_s", "minutes_to_ictus",
                          "template_match_abs_correlation"] if c in columns]
    data = pd.read_csv(time_csv, usecols=wanted)
    data["template_label"] = data["template_label"].astype(str).str.strip().str.upper()
    data = data[data["template_label"].isin(targets) & data["backfit_method"].isin(["hard", "gaussian_mixture"])].copy()
    data["file_key"] = norm_path(data["file_path"])
    data = data.merge(metadata[["file_key", "stratum", "hypo_hyper", "seizure_type"]], on="file_key", how="left", validate="many_to_one")
    keys = ["participant", "file_path", "file_key", "backfit_method", "template_label", "stratum", "hypo_hyper", "seizure_type"]
    schemes = []
    if include_primary:
        schemes.append(("20min_60s_step10s", 60, 10))
    if include_final:
        schemes.append(("final60s_10s_step1s", 10, 1))
    output: list[dict] = []
    for key, frame in data.groupby(keys, observed=True, dropna=False, sort=False):
        base = dict(zip(keys, key))
        for scheme, window, step in schemes:
            part = frame[(-frame["minutes_to_ictus"] * 60).le(1200 if window == 60 else 60)]
            output.extend(base | row for row in rolling_one_group(part, source, scheme, window, step))
    event = pd.DataFrame(output)
    if event.empty:
        return event, event
    event = filter_event_window_outliers(event)
    value_cols = [c for c in ["coverage", "occurrence_rate_hz", "mean_dwell_s", "mean_gfp",
                              "mean_topographic_gmd", "template_match_abs_correlation",
                              "coverage_raw", "mean_gfp_raw"] if c in event]
    group_cols = ["participant", "stratum", "backfit_method", "template_label", "source", "scheme",
                  "window_s", "step_s", "window_end_to_ictus_s", "minutes_to_ictus"]
    participant = event.groupby(group_cols, observed=True, dropna=False)[value_cols].mean().reset_index()
    participant["n_events"] = event.groupby(group_cols, observed=True, dropna=False).size().to_numpy()
    for flag in ("coverage_outlier", "mean_gfp_outlier"):
        if flag in event:
            participant[f"{flag}_events"] = event.groupby(group_cols, observed=True, dropna=False)[flag].sum().to_numpy()
    return participant, event


def piecewise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    tmin = out["minutes_to_ictus"].astype(float)
    out["early_elapsed_min"] = np.clip(tmin + 20, 0, 15)
    out["late_elapsed_min"] = np.clip(tmin + 5, 0, 5)
    return out


def fit_scalar_models(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    terms, contrasts = [], []
    factors = ["source", "template_label", "analysis_role", "backfit_method", "stratum", "scheme"]
    for key, frame in data.groupby(factors, observed=True, dropna=False):
        meta = dict(zip(factors, key))
        if meta["source"] == "participant" and "template_match_abs_correlation" in frame:
            frame = frame[pd.to_numeric(frame["template_match_abs_correlation"], errors="coerce").ge(.5)].copy()
            meta["template_quality_filter"] = "abs_r>=0.50"
        else:
            meta["template_quality_filter"] = "not_applicable"
        frame = piecewise_columns(frame)
        for metric in [c for c in ["coverage", "occurrence_rate_hz", "mean_dwell_s", "mean_gfp",
                                    "mean_topographic_gmd"] if c in frame]:
            fit_data = frame.dropna(subset=[metric]).copy()
            n = fit_data["participant"].nunique()
            if n < 15 or len(fit_data) < 30 or fit_data["minutes_to_ictus"].nunique() < 3:
                terms.append(meta | {"metric": metric, "n_participants": n, "status": "descriptive: insufficient n/time points", "term": "", "p_value": np.nan})
                continue
            if metric == "coverage":
                fit_data["outcome"] = np.log(np.clip(fit_data[metric], 1e-5, 1 - 1e-5) / np.clip(1 - fit_data[metric], 1e-5, 1))
                scale = "logit coverage"
            elif metric == "occurrence_rate_hz":
                fit_data["outcome"] = np.log(fit_data[metric].clip(lower=0) + .5 / fit_data["window_s"])
                scale = "log occurrence rate (half-event correction)"
            else:
                fit_data = fit_data[fit_data[metric].gt(0)].copy()
                fit_data["outcome"] = np.log(fit_data[metric])
                scale = f"log {metric}"
            try:
                fit = smf.gee("outcome ~ early_elapsed_min + late_elapsed_min", "participant", fit_data,
                              family=sm.families.Gaussian(), cov_struct=Exchangeable()).fit()
                ci = fit.conf_int()
                converged = bool(fit.converged)
                for term in ("early_elapsed_min", "late_elapsed_min"):
                    terms.append(meta | {"metric": metric, "scale": scale, "n_participants": n, "n_rows": len(fit_data),
                                         "term": term, "estimate": fit.params[term], "se": fit.bse[term],
                                         "ci_low": ci.loc[term, 0], "ci_high": ci.loc[term, 1],
                                         "p_value": fit.pvalues[term] if converged else np.nan,
                                         "status": "ok" if converged else "warning: not_converged"})
            except Exception as exc:
                terms.append(meta | {"metric": metric, "n_participants": n, "status": f"failed: {exc}", "term": "", "p_value": np.nan})
            early = fit_data[fit_data["minutes_to_ictus"].le(-15)].groupby("participant")[metric].mean()
            late = fit_data[fit_data["minutes_to_ictus"].ge(-5)].groupby("participant")[metric].mean()
            paired = pd.concat([early.rename("early"), late.rename("late")], axis=1).dropna()
            if len(paired) >= 3:
                test = ttest_rel(paired["late"], paired["early"])
                diff = paired["late"] - paired["early"]
                se = diff.std(ddof=1) / math.sqrt(len(diff))
                crit = t.ppf(.975, len(diff) - 1)
                contrasts.append(meta | {"metric": metric, "n_participants": len(diff), "early_mean": paired["early"].mean(),
                                         "late_mean": paired["late"].mean(), "late_minus_early": diff.mean(),
                                         "ci_low": diff.mean() - crit * se, "ci_high": diff.mean() + crit * se,
                                         "p_value": test.pvalue})
    terms = pd.DataFrame(terms)
    contrasts = pd.DataFrame(contrasts)
    for result in (terms, contrasts):
        result["p_fdr"] = np.nan
        if "p_value" not in result:
            continue
        valid = result["p_value"].notna()
        if valid.any():
            result.loc[valid, "p_fdr"] = multipletests(result.loc[valid, "p_value"], method="fdr_bh")[1]
    return terms, contrasts


def parse_numbers(value: object) -> np.ndarray:
    if pd.isna(value) or not str(value).strip():
        return np.array([], float)
    return np.fromstring(str(value), sep=";")


def interval_union_exposure(starts: np.ndarray, ends: np.ndarray, lo: float, hi: float) -> float:
    intervals = sorted((max(lo, a), min(hi, b)) for a, b in zip(starts, ends) if b > lo and a < hi)
    total, cursor = 0.0, lo
    for a, b in intervals:
        a = max(a, cursor)
        if b > a:
            total += b - a
            cursor = b
    return total


def build_cardiac(cardiac_csv: Path, metadata: pd.DataFrame, quality: pd.DataFrame,
                  start_ms: int = 180, end_ms: int = 500) -> pd.DataFrame:
    use = ["participant", "condition", "group", "file_path", "backfit_method", "cardiac_phase_available",
           "template_label", "duration_s", "state_occurrence_times_s", "rpeak_times_relative_s"]
    data = pd.read_csv(cardiac_csv, usecols=use)
    data = data[data["condition"].astype(str).str.lower().isin(["preict", "preictal"]) & data["backfit_method"].eq("hard") &
                data["cardiac_phase_available"].eq(1) & data["template_label"].isin(["A", "C", "E"])].copy()
    data["file_key"] = norm_path(data["file_path"])
    data = data.merge(metadata[["file_key", "stratum"]], on="file_key", how="left", validate="many_to_one")
    data = data.merge(quality, on=["participant", "template_label"], how="left", validate="many_to_one")
    data = data[pd.to_numeric(data["template_match_abs_correlation"], errors="coerce").ge(.5)].copy()
    rows = []
    for row in data.itertuples(index=False):
        duration = float(row.duration_s)
        occurrences = np.sort(parse_numbers(row.state_occurrence_times_s))
        rpeaks = parse_numbers(row.rpeak_times_relative_s)
        starts, ends = rpeaks + start_ms / 1000, rpeaks + end_ms / 1000
        merged = []
        for a, b in zip(starts, ends):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        merged_starts = np.array([x[0] for x in merged], float)
        merged_ends = np.array([x[1] for x in merged], float)
        phase_index = np.searchsorted(merged_starts, occurrences, side="right") - 1
        occurrence_in_sys = (phase_index >= 0) & (occurrences < merged_ends[np.maximum(phase_index, 0)])
        systolic_cumsum = np.r_[0, np.cumsum(occurrence_in_sys)]
        for end_to in range(0, 1141, 10):
            hi, lo = duration - end_to, duration - end_to - 60
            if lo < -1e-6:
                continue
            exposure_sys = np.maximum(0, np.minimum(merged_ends, hi) - np.maximum(merged_starts, lo)).sum()
            first, last = np.searchsorted(occurrences, [lo, hi])
            systolic_count = int(systolic_cumsum[last] - systolic_cumsum[first])
            total_count = int(last - first)
            for phase, count, exposure in (("systole", systolic_count, exposure_sys),
                                            ("diastole", total_count - systolic_count, 60 - exposure_sys)):
                rows.append({"participant": row.participant, "file_path": row.file_path, "stratum": row.stratum,
                             "template_label": row.template_label, "phase": phase, "count": int(count),
                             "exposure_s": exposure, "window_end_to_ictus_s": end_to,
                             "minutes_to_ictus": -end_to / 60, "systolic_window_start_ms": start_ms,
                             "systolic_window_end_ms": end_ms})
    return pd.DataFrame(rows)


def fit_cardiac_models(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    factors = ["template_label", "stratum", "systolic_window_start_ms", "systolic_window_end_ms"]
    for key, frame in data.groupby(factors, observed=True):
        state, stratum, start_ms, end_ms = key
        n = frame["participant"].nunique()
        if n < 15:
            rows.append({"template_label": state, "stratum": stratum, "systolic_window_start_ms": start_ms,
                         "systolic_window_end_ms": end_ms, "n_participants": n, "status": "descriptive: n<15", "term": "", "p_value": np.nan})
            continue
        frame = (frame.groupby(["participant", "phase", "window_end_to_ictus_s", "minutes_to_ictus"], observed=True)
                 [["count", "exposure_s"]].mean().reset_index())
        frame = piecewise_columns(frame)
        frame = frame[frame["exposure_s"].gt(0)].copy()
        frame["is_systole"] = frame["phase"].eq("systole").astype(int)
        try:
            fit = smf.gee("count ~ is_systole * (early_elapsed_min + late_elapsed_min)", "participant", frame,
                          family=sm.families.Poisson(), cov_struct=sm.cov_struct.Independence(),
                          offset=np.log(frame["exposure_s"])).fit(maxiter=200)
            ci = fit.conf_int()
            converged = bool(fit.converged)
            for term in fit.params.index:
                if term == "Intercept":
                    continue
                rows.append({"template_label": state, "stratum": stratum, "systolic_window_start_ms": start_ms,
                             "systolic_window_end_ms": end_ms, "n_participants": n, "n_rows": len(frame),
                             "term": term, "estimate_log_rate": fit.params[term], "rate_ratio": np.exp(fit.params[term]),
                             "ci_low": np.exp(ci.loc[term, 0]), "ci_high": np.exp(ci.loc[term, 1]),
                              "p_value": fit.pvalues[term] if converged else np.nan,
                              "status": "ok" if converged else "warning: not_converged"})
        except Exception as exc:
            rows.append({"template_label": state, "stratum": stratum, "systolic_window_start_ms": start_ms,
                         "systolic_window_end_ms": end_ms, "n_participants": n, "status": f"failed: {exc}", "term": "", "p_value": np.nan})
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    if "p_value" not in out or "term" not in out:
        return out
    for _, idx in out.groupby(["systolic_window_start_ms", "systolic_window_end_ms"], observed=True).groups.items():
        valid = out.index.isin(idx) & out["p_value"].notna() & out["term"].str.contains(":", na=False)
        if valid.any():
            out.loc[valid, "p_holm"] = multipletests(out.loc[valid, "p_value"], method="holm")[1]
    return out


def plot_scalar(data: pd.DataFrame, out_dir: Path) -> None:
    primary = data[(data["scheme"].eq("20min_60s_step10s")) & data["backfit_method"].eq("hard")]
    primary = primary[(primary["source"].ne("participant")) |
                      pd.to_numeric(primary["template_match_abs_correlation"], errors="coerce").ge(.5)]
    for metric in [c for c in ["coverage", "occurrence_rate_hz", "mean_dwell_s",
                                "mean_topographic_gmd", "mean_gfp"] if c in primary]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
        for ax, (source, state) in zip(axes.ravel(), [("global", "C"), ("global", "E"),
                                                      ("participant", "C"), ("participant", "E")]):
            panel = primary[(primary["source"].eq(source)) & primary["template_label"].eq(state)]
            for stratum, frame in panel.groupby("stratum", observed=True):
                if "functional" not in stratum:
                    continue
                summary = frame.groupby("minutes_to_ictus")[metric].agg(["mean", "std", "count"]).reset_index()
                sem = summary["std"] / np.sqrt(summary["count"])
                ax.plot(summary["minutes_to_ictus"], summary["mean"], label=stratum)
                ax.fill_between(summary["minutes_to_ictus"], summary["mean"] - 1.96 * sem,
                                summary["mean"] + 1.96 * sem, alpha=.18)
            ax.axvline(-5, color=".55", ls="--", lw=1)
            ax.set_title(f"{source.title()} {state}")
            ax.set_xlabel("Minutes to seizure onset")
            ax.grid(alpha=.2)
        axes[0, 0].set_ylabel(metric.replace("_", " ").title())
        axes[1, 0].set_ylabel(metric.replace("_", " ").title())
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
        fig.suptitle(f"Preictal rolling {metric.replace('_', ' ')} (participant-weighted mean and 95% CI)", fontweight="bold")
        fig.tight_layout(rect=(0, .07, 1, .94))
        fig.savefig(out_dir / f"rolling_{metric}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def cohort_audit(manifest: pd.DataFrame) -> pd.DataFrame:
    return (manifest.groupby(["stratum", "group", "condition", "seizure_type"], observed=True, dropna=False)
            .agg(n_records=("file_path", "size"), n_participants=("participant", "nunique")).reset_index())


def self_check() -> None:
    frame = pd.DataFrame({"bin_from_ictus": np.arange(1, 61), "minutes_to_ictus": -(np.arange(1, 61) - .5) / 60,
                          "bin_duration_s": 1,
                          "coverage": .25, "occurrence_count": 2, "mean_dwell_s": .1,
                          "mean_gfp": 2.0, "mean_topographic_gmd": .5})
    rows = rolling_one_group(frame, "global", "test", 60, 10)
    assert len(rows) == 1 and abs(rows[0]["coverage"] - .25) < 1e-12
    assert abs(rows[0]["occurrence_rate_hz"] - 2.0) < 1e-12
    assert abs(rows[0]["mean_dwell_s"] - .1) < 1e-12
    outliers = pd.DataFrame({
        "participant": "P1", "file_path": "event", "source": "global", "scheme": "test",
        "backfit_method": "hard", "template_label": "C", "stratum": "FS functional",
        "coverage": [.2] * 20 + [1.0], "mean_gfp": [2.] * 19 + [20., 2.]})
    cleaned = filter_event_window_outliers(outliers)
    assert cleaned["coverage_outlier"].sum() == 1 and pd.isna(cleaned.loc[20, "coverage"])
    assert cleaned["mean_gfp_outlier"].sum() == 1 and pd.isna(cleaned.loc[19, "mean_gfp"])
    assert cleaned.loc[19, "coverage"] == .2 and cleaned.loc[20, "mean_gfp"] == 2.
    boundary = outliers.iloc[:9].copy()
    boundary["file_path"] = "boundary"
    boundary["coverage"] = [0.] * 8 + [3.]
    boundary["mean_gfp"] = 1.
    boundary_cleaned = filter_event_window_outliers(boundary)
    assert not boundary_cleaned.loc[boundary_cleaned.index[-1], "coverage_outlier"]
    assert boundary_cleaned.loc[boundary_cleaned.index[-1], "coverage"] == 3.
    assert abs(interval_union_exposure(np.array([0., .2]), np.array([.4, .6]), 0., 1.) - .6) < 1e-12
    manifest = attach_strata(pd.DataFrame({"group": ["FS", "DD"], "seizure_type": ["FS", "ES"],
                                           "file_path": ["A", "B"]}))
    assert manifest["stratum"].tolist() == ["FS functional", "DD epileptic"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cardiac-csv", type=Path)
    parser.add_argument("--final-minute-results-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("self-check passed")
        return
    if not args.results_dir or not args.manifest or not args.out_dir:
        parser.error("--results-dir, --manifest, and --out-dir are required")
    data_dir, figure_dir = args.out_dir / "data", args.out_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    manifest = attach_strata(pd.read_csv(args.manifest))
    if not manifest["file_key"].is_unique:
        raise ValueError("Corrected manifest still contains duplicate file paths")
    cohort_audit(manifest).to_csv(data_dir / "cohort_audit.csv", index=False)
    global_roll, global_event = build_rolling(args.results_dir / "global_preictal_time_metrics.csv", "global", {"C", "E", "G"}, manifest,
                                              include_primary=True, include_final=args.final_minute_results_dir is None)
    participant_roll, participant_event = build_rolling(args.results_dir / "participant_condition_preictal_time_metrics.csv", "participant", {"C", "E"}, manifest,
                                                        include_primary=True, include_final=args.final_minute_results_dir is None)
    if args.final_minute_results_dir:
        final_global_roll, final_global_event = build_rolling(args.final_minute_results_dir / "global_preictal_time_metrics.csv",
                                                               "global", {"C", "E", "G"}, manifest, include_primary=False, include_final=True)
        final_participant_roll, final_participant_event = build_rolling(args.final_minute_results_dir / "participant_condition_preictal_time_metrics.csv",
                                                                         "participant", {"C", "E"}, manifest, include_primary=False, include_final=True)
        global_roll = pd.concat([global_roll, final_global_roll], ignore_index=True)
        participant_roll = pd.concat([participant_roll, final_participant_roll], ignore_index=True)
        global_event = pd.concat([global_event, final_global_event], ignore_index=True)
        participant_event = pd.concat([participant_event, final_participant_event], ignore_index=True)
    rolling = pd.concat([global_roll, participant_roll], ignore_index=True)
    rolling_event = pd.concat([global_event, participant_event], ignore_index=True)
    validate_event_window_outliers(rolling_event)
    rolling["analysis_role"] = np.select(
        [rolling["source"].eq("global") & rolling["template_label"].eq("C"),
         rolling["source"].eq("participant") & rolling["template_label"].eq("E"),
         rolling["source"].eq("global") & rolling["template_label"].eq("E"),
         rolling["source"].eq("participant") & rolling["template_label"].eq("C")],
        ["co_primary", "co_primary", "global_E_sensitivity", "participant_C_sensitivity"],
        default="exploratory")
    terms, contrasts = fit_scalar_models(rolling)
    terms.to_csv(data_dir / "rolling_scalar_gee_terms.csv", index=False)
    contrasts.to_csv(data_dir / "rolling_early_late_contrasts.csv", index=False)
    rolling_outlier_audit(rolling_event).to_csv(data_dir / "rolling_window_outlier_audit.csv", index=False)
    plot_scalar(rolling, figure_dir)
    cardiac_csv = args.cardiac_csv or args.results_dir / "participant_ce_cardiac_phase_metrics.csv"
    covariance = pd.read_csv(args.results_dir / "hierarchical_state_covariance_metrics.csv",
                             usecols=["level", "participant", "template_label", "template_match_abs_correlation"])
    quality = (covariance[covariance["level"].eq("participant")]
               .groupby(["participant", "template_label"], observed=True)["template_match_abs_correlation"].mean().reset_index())
    cardiac = pd.concat([build_cardiac(cardiac_csv, manifest, quality, 180, 500),
                         build_cardiac(cardiac_csv, manifest, quality, 200, 400)], ignore_index=True)
    cardiac_models = fit_cardiac_models(cardiac)
    cardiac_models.to_csv(data_dir / "rolling_cardiac_phase_gee_terms.csv", index=False)
    subject_map = {participant: f"S{i:03d}" for i, participant in enumerate(sorted(manifest["participant"].unique()), 1)}
    rolling_export = rolling.assign(subject_id=rolling["participant"].map(subject_map)).drop(columns="participant")
    rolling_export.to_csv(data_dir / "rolling_microstate_metrics_participant_weighted.csv", index=False)
    rolling_event_map = {path: f"R{i:04d}" for i, path in enumerate(sorted(rolling_event["file_path"].unique()), 1)}
    rolling_event_export = rolling_event.assign(subject_id=rolling_event["participant"].map(subject_map),
                                                 event_id=rolling_event["file_path"].map(rolling_event_map)).drop(columns=["participant", "file_path", "file_key"])
    rolling_event_export.to_csv(data_dir / "rolling_microstate_metrics_event.csv", index=False)
    event_map = {path: f"E{i:04d}" for i, path in enumerate(sorted(cardiac["file_path"].unique()), 1)}
    cardiac_export = cardiac.assign(subject_id=cardiac["participant"].map(subject_map),
                                     event_id=cardiac["file_path"].map(event_map)).drop(columns=["participant", "file_path"])
    cardiac_export.to_csv(data_dir / "rolling_cardiac_phase_metrics.csv", index=False)
    dictionary = pd.DataFrame([
        ("rolling_microstate_metrics_participant_weighted.csv", "One row per participant/window/state/method after equal averaging of that participant's events."),
        ("rolling_microstate_metrics_event.csv", "Deidentified event-level rolling metrics with raw values and metric-specific median-plus/minus-3-SD outlier flags."),
        ("rolling_window_outlier_audit.csv", "Retained and excluded GFP/coverage window counts by analysis cell."),
        ("rolling_scalar_gee_terms.csv", "Participant-clustered GEE piecewise slopes; early=-20 to -5 min, late=-5 to 0 min."),
        ("rolling_early_late_contrasts.csv", "Paired participant contrast: final 5 min minus earliest 5 min."),
        ("rolling_cardiac_phase_metrics.csv", "Event-window systolic/diastolic occurrence counts and exposure from corrected R-peaks."),
        ("rolling_cardiac_phase_gee_terms.csv", "Exposure-offset Poisson GEE; Holm correction on phase-by-time interactions."),
        ("variable_dictionary.csv", "Column definitions and units for the rolling and cardiac analysis tables."),
    ], columns=["file", "description"])
    dictionary.to_csv(data_dir / "data_dictionary.csv", index=False)
    variables = pd.DataFrame([
        ("subject_id", "identifier", "Deidentified participant identifier; mapping is not exported."),
        ("event_id", "identifier", "Deidentified event identifier; mapping is not exported."),
        ("stratum", "category", "FS functional, DD functional, ES epileptic or DD epileptic analysis stratum."),
        ("source", "category", "Global or participant-template backfit source."),
        ("analysis_role", "category", "Co-primary, sensitivity or exploratory state/source role."),
        ("scheme", "category", "Rolling-window definition."),
        ("template_label", "category", "Canonical microstate label."),
        ("backfit_method", "category", "Hard or Gaussian-mixture backfit."),
        ("minutes_to_ictus", "minutes", "Window endpoint relative to annotated seizure onset; zero is onset."),
        ("coverage", "proportion", "Fraction of window assigned to the state."),
        ("coverage_raw", "proportion", "Coverage before within-event median-plus/minus-3-SD filtering."),
        ("coverage_outlier", "boolean", "True when raw coverage is more than three sample SDs from its within-event median."),
        ("occurrence_rate_hz", "events/second", "Number of state onsets divided by window duration."),
        ("mean_dwell_s", "seconds", "Occurrence-weighted mean duration of contiguous state runs."),
        ("mean_gfp", "input signal units", "State-weighted mean global field power."),
        ("mean_gfp_raw", "input signal units", "Mean GFP before within-event median-plus/minus-3-SD filtering."),
        ("mean_gfp_outlier", "boolean", "True when raw mean GFP is more than three sample SDs from its within-event median."),
        ("mean_topographic_gmd", "dimensionless", "Polarity-invariant global map dissimilarity to the template; lower is closer."),
        ("template_match_abs_correlation", "correlation", "Absolute spatial correlation between fitted state and canonical template."),
        ("phase", "category", "Systole or diastole under the specified R-peak window."),
        ("count", "events", "State-onset count in a phase/window."),
        ("exposure_s", "seconds", "Time-at-risk in the phase/window."),
        ("estimate", "model scale", "GEE coefficient on the stated transformed outcome scale."),
        ("rate_ratio", "ratio", "Exponentiated Poisson GEE coefficient."),
        ("p_fdr", "probability", "Benjamini-Hochberg-adjusted p-value."),
        ("p_holm", "probability", "Holm-adjusted p-value for cardiac phase-by-time interactions."),
    ], columns=["variable", "unit_or_type", "description"])
    variables.to_csv(data_dir / "variable_dictionary.csv", index=False)
    primary_terms = terms[(terms["analysis_role"].eq("co_primary")) &
                          (terms["scheme"].eq("20min_60s_step10s")) & terms["backfit_method"].eq("hard") &
                          terms["stratum"].astype(str).str.contains("functional", na=False)]
    primary_cardiac = cardiac_models[(cardiac_models["systolic_window_start_ms"].eq(180)) &
                                     (cardiac_models["systolic_window_end_ms"].eq(500)) &
                                     cardiac_models["stratum"].astype(str).str.contains("functional", na=False) &
                                     cardiac_models["term"].astype(str).str.contains(":", na=False)]
    summary = {"manifest_rows": len(manifest), "manifest_unique_paths": int(manifest["file_key"].nunique()),
               "global_template_labels": sorted(covariance.loc[covariance["level"].eq("global"), "template_label"].dropna().astype(str).unique().tolist()),
               "rolling_rows": len(rolling), "rolling_participants": int(rolling["participant"].nunique()),
               "rolling_event_rows": len(rolling_event),
               "coverage_outlying_windows": int(rolling_event.get("coverage_outlier", pd.Series(dtype=bool)).sum()),
               "mean_gfp_outlying_windows": int(rolling_event.get("mean_gfp_outlier", pd.Series(dtype=bool)).sum()),
               "cardiac_rows": len(cardiac), "cardiac_participants": int(cardiac["participant"].nunique()),
               "significant_scalar_fdr": int(terms["p_fdr"].lt(.05).sum()),
               "significant_scalar_fdr_primary_functional": int(primary_terms["p_fdr"].lt(.05).sum()),
               "significant_cardiac_holm_interactions": int(cardiac_models["p_holm"].lt(.05).sum()),
               "significant_cardiac_holm_primary_functional": int(primary_cardiac["p_holm"].lt(.05).sum())}
    (data_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
