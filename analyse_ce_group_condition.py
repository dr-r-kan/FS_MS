"""Group x condition analysis of microstate C/E coverage and GFP."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian
from statsmodels.stats.multitest import multipletests


SOURCES = {
    "participant": "participant_condition_state_backfit_metrics.csv",
    "global": "global_state_backfit_metrics.csv",
}
TIME_SOURCES = {
    "participant": "participant_condition_preictal_time_metrics.csv",
    "global": "global_preictal_time_metrics.csv",
}
STATES = ("C", "E")
METRICS = ("coverage", "mean_gfp")
SEMIOLOGY_CONDITIONS = ("preictal", "postictal")
BASELINE_METRICS = ("coverage_change_from_II", "mean_gfp_change_from_II")
CONDITION_NAMES = {"preict": "preictal", "postict": "postictal"}
SEIZURE_TYPE_NAMES = {"FS": "NEAD", "NEAD": "NEAD", "ES": "ES"}


def default_manifest(results_dir: Path) -> Path:
    for name in ("vg_microstate_manifest_mat.csv", "vg_microstate_manifest_set.csv", "vg_microstate_manifest_fif.csv"):
        path = results_dir.parent / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No VG manifest found beside {results_dir}")


def add_semiology(data: pd.DataFrame, manifest_csv: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_csv, usecols=lambda column: column in {"file_path", "hypo_hyper"})
    if not {"file_path", "hypo_hyper"}.issubset(manifest.columns):
        raise ValueError(f"{manifest_csv} must contain file_path and hypo_hyper")
    manifest["_path"] = manifest["file_path"].astype(str).str.replace("/", "\\", regex=False).str.lower()
    data = data.copy()
    data["_path"] = data["file_path"].astype(str).str.replace("/", "\\", regex=False).str.lower()
    return data.merge(manifest[["_path", "hypo_hyper"]].drop_duplicates("_path"), on="_path", how="left")


def load_backfits(results_dir: Path, method: str, manifest_csv: Path) -> pd.DataFrame:
    frames = []
    required = {
        "participant", "group", "condition", "template_label", "backfit_method",
        "backfit_available", "occupancy", "mean_gfp", "duration_s", "file_path",
    }
    for source, filename in SOURCES.items():
        path = results_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        frame["source"] = source
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No backfit metric tables found in {results_dir}")

    data = pd.concat(frames, ignore_index=True)
    data = add_semiology(data, manifest_csv)
    data["template_label"] = data["template_label"].astype(str).str.strip().str.upper()
    data["condition"] = data["condition"].replace(CONDITION_NAMES)
    data["hypo_hyper"] = data["hypo_hyper"].astype(str).str.strip().str.lower()
    for column in ("backfit_available", "occupancy", "mean_gfp", "duration_s"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    keep = (
        data["backfit_method"].eq(method)
        & data["backfit_available"].eq(1)
        & data["template_label"].isin(STATES)
    )
    data = data.loc[keep].copy()
    if data.empty:
        raise ValueError(f"No available {method!r} backfits labelled C or E")
    return data


def load_preictal_time_backfits(results_dir: Path, manifest_csv: Path, method: str) -> pd.DataFrame:
    frames = []
    for source, filename in TIME_SOURCES.items():
        path = results_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["source"] = source
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    data = add_semiology(pd.concat(frames, ignore_index=True), manifest_csv)
    data["template_label"] = data["template_label"].astype(str).str.strip().str.upper()
    data["hypo_hyper"] = data["hypo_hyper"].astype(str).str.strip().str.lower()
    data["seizure_type"] = data["group"].astype(str).str.strip().str.upper().map(SEIZURE_TYPE_NAMES)
    for column in ("coverage", "mean_gfp", "minutes_to_ictus", "bin_duration_s"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data[
        data["backfit_method"].eq(method)
        & data["template_label"].isin(STATES)
        & data["condition"].isin(("preict", "preictal"))
        & data["bin_duration_s"].ge(30)
        & data["seizure_type"].isin(("ES", "NEAD"))
    ].copy()


def participant_condition_metrics(data: pd.DataFrame, extra_keys: tuple[str, ...] = ()) -> pd.DataFrame:
    rows = []
    keys = ["source", "participant", "group", "condition", "template_label", *extra_keys]
    for key, group in data.groupby(keys, dropna=False, sort=True):
        duration = group["duration_s"].clip(lower=0).fillna(0).to_numpy(float)
        coverage = group["occupancy"].to_numpy(float)
        gfp = group["mean_gfp"].to_numpy(float)
        valid_coverage = np.isfinite(coverage) & (duration > 0)
        coverage_mean = np.average(coverage[valid_coverage], weights=duration[valid_coverage]) if valid_coverage.any() else np.nan
        gfp_weight = duration * np.nan_to_num(coverage, nan=0.0)
        valid_gfp = np.isfinite(gfp) & (gfp_weight > 0)
        gfp_mean = np.average(gfp[valid_gfp], weights=gfp_weight[valid_gfp]) if valid_gfp.any() else np.nan
        rows.append((*key, coverage_mean, gfp_mean, len(group), duration.sum()))
    return pd.DataFrame(rows, columns=[*keys, *METRICS, "n_records", "duration_s"])


def group_condition_summary(unit: pd.DataFrame) -> pd.DataFrame:
    long = unit.melt(
        id_vars=["source", "participant", "group", "condition", "template_label"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    summary = (
        long.groupby(["source", "template_label", "metric", "group", "condition"], dropna=False)["value"]
        .agg(n_participants="count", mean="mean", sd="std")
        .reset_index()
    )
    summary["sem"] = summary["sd"] / np.sqrt(summary["n_participants"])
    summary["ci95"] = 1.96 * summary["sem"]
    return summary


def gee_results(unit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    term_rows, coefficient_rows = [], []
    long = unit.melt(
        id_vars=["source", "participant", "group", "condition", "template_label"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value", "participant", "group", "condition"])
    for (source, state, metric), data in long.groupby(["source", "template_label", "metric"]):
        prefix = {"source": source, "template_label": state, "metric": metric, "n": len(data), "n_participants": data["participant"].nunique()}
        if data["group"].nunique() < 2 or data["condition"].nunique() < 2 or data["participant"].nunique() < 4:
            term_rows.append({**prefix, "term": "", "note": "insufficient group, condition, or participant levels"})
            continue
        try:
            fit = smf.gee(
                "value ~ C(group) * C(condition)", data=data, groups="participant",
                cov_struct=Exchangeable(), family=Gaussian(),
            ).fit()
            terms = fit.wald_test_terms(scalar=True).table
            for term, result in terms.iterrows():
                term_rows.append({
                    **prefix, "term": term, "chi2": float(result["statistic"]),
                    "df": int(result["df_constraint"]), "p_value": float(result["pvalue"]), "note": "",
                })
            ci = fit.conf_int()
            for name in fit.params.index:
                coefficient_rows.append({
                    **prefix, "coefficient": name, "estimate": float(fit.params[name]),
                    "se": float(fit.bse[name]), "ci_low": float(ci.loc[name, 0]),
                    "ci_high": float(ci.loc[name, 1]), "p_value": float(fit.pvalues[name]),
                })
        except Exception as exc:
            term_rows.append({**prefix, "term": "", "note": str(exc)})
    terms = pd.DataFrame(term_rows)
    terms["p_fdr_bh"] = np.nan
    if "p_value" in terms:
        tested = terms["term"].ne("Intercept") & pd.to_numeric(terms["p_value"], errors="coerce").notna()
        if tested.any():
            terms.loc[tested, "p_fdr_bh"] = multipletests(terms.loc[tested, "p_value"], method="fdr_bh")[1]
    return terms, pd.DataFrame(coefficient_rows)


def baseline_correct_from_ii(unit: pd.DataFrame) -> pd.DataFrame:
    keys = ["source", "participant", "group", "template_label"]
    data = unit[unit["group"].isin(("FS", "ES"))].copy()
    baseline = data[data["condition"].astype(str).str.upper().eq("II")][keys + list(METRICS)].rename(
        columns={metric: f"{metric}_II" for metric in METRICS}
    )
    corrected = data[data["condition"].isin(SEMIOLOGY_CONDITIONS)].merge(
        baseline, on=keys, how="inner", validate="many_to_one"
    )
    for metric, corrected_metric in zip(METRICS, BASELINE_METRICS):
        corrected[f"raw_{metric}"] = corrected[metric]
        corrected[corrected_metric] = corrected[metric] - corrected[f"{metric}_II"]
    return corrected


def semiology_summary(unit: pd.DataFrame) -> pd.DataFrame:
    long = unit.melt(
        id_vars=["source", "participant", "group", "condition", "template_label"],
        value_vars=BASELINE_METRICS, var_name="metric", value_name="value",
    ).dropna(subset=["value"])
    summary = (
        long.groupby(["source", "template_label", "metric", "group", "condition"])["value"]
        .agg(n_participants="count", mean="mean", sd="std").reset_index()
    )
    summary["sem"] = summary["sd"] / np.sqrt(summary["n_participants"])
    summary["ci95"] = 1.96 * summary["sem"]
    return summary


def semiology_gee_results(unit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    term_rows, coefficient_rows = [], []
    long = unit.melt(
        id_vars=["source", "participant", "group", "condition", "template_label"],
        value_vars=BASELINE_METRICS, var_name="metric", value_name="value",
    ).dropna(subset=["value", "participant", "condition"])
    for (source, state, metric), data in long.groupby(["source", "template_label", "metric"]):
        prefix = {"source": source, "template_label": state, "metric": metric, "n": len(data), "n_participants": data["participant"].nunique()}
        if data["group"].nunique() < 2 or data["condition"].nunique() < 2 or data["participant"].nunique() < 4:
            term_rows.append({**prefix, "term": "", "note": "insufficient FS/ES, condition, or participant levels"})
            continue
        try:
            fit = smf.gee(
                "value ~ C(group) * C(condition)", data=data, groups="participant",
                cov_struct=Exchangeable(), family=Gaussian(),
            ).fit()
            terms = fit.wald_test_terms(scalar=True).table
            for term, result in terms.iterrows():
                term_rows.append({
                    **prefix, "term": term, "chi2": float(result["statistic"]),
                    "df": int(result["df_constraint"]), "p_value": float(result["pvalue"]), "note": "",
                })
            ci = fit.conf_int()
            for name in fit.params.index:
                coefficient_rows.append({
                    **prefix, "coefficient": name, "estimate": float(fit.params[name]),
                    "se": float(fit.bse[name]), "ci_low": float(ci.loc[name, 0]),
                    "ci_high": float(ci.loc[name, 1]), "p_value": float(fit.pvalues[name]),
                })
        except Exception as exc:
            term_rows.append({**prefix, "term": "", "note": str(exc)})
    terms = pd.DataFrame(term_rows)
    terms["p_fdr_bh"] = np.nan
    if "p_value" in terms:
        tested = terms["term"].ne("Intercept") & pd.to_numeric(terms["p_value"], errors="coerce").notna()
        if tested.any():
            terms.loc[tested, "p_fdr_bh"] = multipletests(terms.loc[tested, "p_value"], method="fdr_bh")[1]
    return terms, pd.DataFrame(coefficient_rows)


def preictal_gradient_analysis(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["source", "participant", "seizure_type", "hypo_hyper", "template_label", "bin_from_ictus"]
    unit = data.groupby(keys, dropna=False, as_index=False)[["minutes_to_ictus", *METRICS]].mean()
    long = unit.melt(
        id_vars=[*keys, "minutes_to_ictus"], value_vars=METRICS,
        var_name="metric", value_name="value",
    ).dropna(subset=["value", "minutes_to_ictus"])
    term_rows, coefficient_rows = [], []
    for (source, state, metric), frame in long.groupby(["source", "template_label", "metric"]):
        prefix = {"source": source, "template_label": state, "metric": metric, "n": len(frame), "n_participants": frame["participant"].nunique()}
        if frame["minutes_to_ictus"].nunique() < 2 or frame["participant"].nunique() < 4:
            term_rows.append({**prefix, "term": "", "note": "insufficient preictal time range or participants"})
            continue
        has_type = frame["seizure_type"].nunique() > 1
        has_semiology = frame["hypo_hyper"].isin(("hypo", "hyper")).sum() and frame["hypo_hyper"].nunique() > 1
        if has_type and has_semiology:
            formula = "value ~ minutes_to_ictus * C(seizure_type) * C(hypo_hyper)"
        elif has_type:
            formula = "value ~ minutes_to_ictus * C(seizure_type)"
        elif has_semiology:
            formula = "value ~ minutes_to_ictus * C(hypo_hyper)"
        else:
            formula = "value ~ minutes_to_ictus"
        try:
            fit = smf.gee(
                formula, data=frame, groups="participant",
                cov_struct=Exchangeable(), family=Gaussian(),
            ).fit()
            terms = fit.wald_test_terms(scalar=True).table
            for term, result in terms.iterrows():
                term_rows.append({
                    **prefix, "term": term, "chi2": float(result["statistic"]),
                    "df": int(result["df_constraint"]), "p_value": float(result["pvalue"]), "note": "",
                })
            ci = fit.conf_int()
            for name in fit.params.index:
                coefficient_rows.append({
                    **prefix, "coefficient": name, "estimate": float(fit.params[name]),
                    "se": float(fit.bse[name]), "ci_low": float(ci.loc[name, 0]),
                    "ci_high": float(ci.loc[name, 1]), "p_value": float(fit.pvalues[name]),
                })
        except Exception as exc:
            term_rows.append({**prefix, "term": "", "note": str(exc)})
    terms = pd.DataFrame(term_rows)
    terms["p_fdr_bh"] = np.nan
    if "p_value" in terms:
        tested = terms["term"].ne("Intercept") & pd.to_numeric(terms["p_value"], errors="coerce").notna()
        if tested.any():
            terms.loc[tested, "p_fdr_bh"] = multipletests(terms.loc[tested, "p_value"], method="fdr_bh")[1]
    summary = (
        long.groupby(["source", "template_label", "metric", "seizure_type", "hypo_hyper", "bin_from_ictus"], dropna=False)
        .agg(minutes_to_ictus=("minutes_to_ictus", "mean"), n_participants=("participant", "nunique"), mean=("value", "mean"), sd=("value", "std"))
        .reset_index()
    )
    summary["sem"] = summary["sd"] / np.sqrt(summary["n_participants"])
    summary["ci95"] = 1.96 * summary["sem"]
    return unit, summary, terms, pd.DataFrame(coefficient_rows)


def plot_summary(summary: pd.DataFrame, out_file: Path) -> None:
    row_keys = [(source, state) for source in SOURCES for state in STATES if ((summary["source"] == source) & (summary["template_label"] == state)).any()]
    fig, axes = plt.subplots(len(row_keys), len(METRICS), squeeze=False, figsize=(11, 3.2 * len(row_keys)), facecolor="white")
    for row, (source, state) in enumerate(row_keys):
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            panel = summary[(summary["source"] == source) & (summary["template_label"] == state) & (summary["metric"] == metric)]
            conditions = sorted(panel["condition"].astype(str).unique())
            for group_name, values in panel.groupby("group", sort=True):
                values = values.set_index(values["condition"].astype(str)).reindex(conditions)
                ax.errorbar(conditions, values["mean"], yerr=values["ci95"], marker="o", capsize=3, label=str(group_name))
            ax.set_title(f"{source.title()} backfit | state {state} | {'Coverage' if metric == 'coverage' else 'Mean GFP'}")
            ax.set_xlabel("Condition")
            ax.set_ylabel("Proportion" if metric == "coverage" else "GFP (native units)")
            ax.grid(axis="y", alpha=0.2)
            ax.legend(title="Group", frameon=False)
    fig.suptitle("Microstates C and E: group × condition", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, facecolor="white")
    plt.close(fig)


def plot_semiology(summary: pd.DataFrame, out_file: Path) -> None:
    row_keys = [(source, state) for source in SOURCES for state in STATES if ((summary["source"] == source) & (summary["template_label"] == state)).any()]
    fig, axes = plt.subplots(len(row_keys), len(BASELINE_METRICS), squeeze=False, figsize=(11, 3.2 * len(row_keys)), facecolor="white")
    for row, (source, state) in enumerate(row_keys):
        for col, metric in enumerate(BASELINE_METRICS):
            ax = axes[row, col]
            panel = summary[(summary["source"] == source) & (summary["template_label"] == state) & (summary["metric"] == metric)]
            conditions = [condition for condition in SEMIOLOGY_CONDITIONS if condition in set(panel["condition"])]
            for group_name, values in panel.groupby("group", sort=True):
                values = values.set_index("condition").reindex(conditions)
                ax.errorbar(conditions, values["mean"], yerr=values["ci95"], marker="o", capsize=3, label=str(group_name))
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"{source.title()} backfit | state {state} | {'Coverage' if metric.startswith('coverage') else 'Mean GFP'}")
            ax.set_xlabel("Condition")
            ax.set_ylabel("Change from individual II baseline")
            ax.grid(axis="y", alpha=0.2)
            ax.legend(title="Group", frameon=False)
    fig.suptitle("Microstates C and E: FS vs ES change from individual II baseline", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, facecolor="white")
    plt.close(fig)


def plot_preictal_gradient(summary: pd.DataFrame, out_file: Path) -> None:
    row_keys = [(source, state) for source in SOURCES for state in STATES if ((summary["source"] == source) & (summary["template_label"] == state)).any()]
    fig, axes = plt.subplots(len(row_keys), len(METRICS), squeeze=False, figsize=(11, 3.2 * len(row_keys)), facecolor="white")
    for row, (source, state) in enumerate(row_keys):
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            panel = summary[(summary["source"] == source) & (summary["template_label"] == state) & (summary["metric"] == metric)]
            for (seizure_type, semiology), values in panel.groupby(["seizure_type", "hypo_hyper"], sort=True):
                values = values.sort_values("minutes_to_ictus")
                ax.errorbar(values["minutes_to_ictus"], values["mean"], yerr=values["ci95"], marker="o", capsize=2, label=f"{seizure_type} | {semiology}")
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(f"{source.title()} backfit | state {state} | {'Coverage' if metric == 'coverage' else 'Mean GFP'}")
            ax.set_xlabel("Minutes to ictal onset")
            ax.set_ylabel("Proportion" if metric == "coverage" else "GFP (native units)")
            ax.grid(axis="y", alpha=0.2)
            ax.legend(title="Seizure type | semiology", frameon=False)
    fig.suptitle("Microstates C and E: gradient through the preictal period", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, facecolor="white")
    plt.close(fig)


def analyse(results_dir: Path, out_dir: Path, method: str = "hard", manifest_csv: Path | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = manifest_csv or default_manifest(results_dir)
    backfits = load_backfits(results_dir, method, manifest_csv)
    unit = participant_condition_metrics(backfits)
    summary = group_condition_summary(unit)
    terms, coefficients = gee_results(unit)
    unit.to_csv(out_dir / "ce_participant_condition_metrics.csv", index=False)
    summary.to_csv(out_dir / "ce_group_condition_summary.csv", index=False)
    terms.to_csv(out_dir / "ce_group_condition_gee_terms.csv", index=False)
    coefficients.to_csv(out_dir / "ce_group_condition_gee_coefficients.csv", index=False)
    plot_summary(summary, out_dir / "ce_group_condition.png")

    semiology_unit = baseline_correct_from_ii(unit)
    if not semiology_unit.empty:
        semiology_descriptives = semiology_summary(semiology_unit)
        semiology_terms, semiology_coefficients = semiology_gee_results(semiology_unit)
        semiology_unit.to_csv(out_dir / "ce_semiology_participant_condition_metrics.csv", index=False)
        semiology_descriptives.to_csv(out_dir / "ce_semiology_condition_summary.csv", index=False)
        semiology_terms.to_csv(out_dir / "ce_semiology_gee_terms.csv", index=False)
        semiology_coefficients.to_csv(out_dir / "ce_semiology_gee_coefficients.csv", index=False)
        (
            semiology_unit.groupby(["source", "condition", "group"])["participant"]
            .nunique().rename("n_participants").reset_index()
            .to_csv(out_dir / "ce_semiology_condition_counts.csv", index=False)
        )
        plot_semiology(semiology_descriptives, out_dir / "ce_semiology_by_condition.png")

    time_data = load_preictal_time_backfits(results_dir, manifest_csv, method)
    if not time_data.empty:
        gradient_unit, gradient_summary, gradient_terms, gradient_coefficients = preictal_gradient_analysis(time_data)
        gradient_unit.to_csv(out_dir / "ce_preictal_gradient_participant_bins.csv", index=False)
        gradient_summary.to_csv(out_dir / "ce_preictal_gradient_summary.csv", index=False)
        gradient_terms.to_csv(out_dir / "ce_preictal_gradient_gee_terms.csv", index=False)
        gradient_coefficients.to_csv(out_dir / "ce_preictal_gradient_gee_coefficients.csv", index=False)
        plot_preictal_gradient(gradient_summary, out_dir / "ce_preictal_gradient.png")


def self_check() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rng = np.random.default_rng(4)
        participants = [f"P{i:02d}" for i in range(16)]
        rows = []
        manifest_rows = []
        for participant_index, participant in enumerate(participants):
            group = "FS" if participant_index < 8 else "ES"
            semiology = "hypo" if participant_index % 2 == 0 else "hyper"
            for condition_index, condition in enumerate(("II", *SEMIOLOGY_CONDITIONS)):
                file_path = str(root / f"{participant}_{condition}.mat")
                manifest_rows.append({"file_path": file_path, "hypo_hyper": semiology})
                for state_index, state in enumerate(STATES):
                    rows.append({
                        "participant": participant, "group": group, "condition": condition,
                        "file_path": file_path,
                        "template_label": state, "backfit_method": "hard", "backfit_available": 1,
                        "occupancy": 0.2 + 0.01 * participant_index + 0.03 * condition_index + 0.01 * state_index + rng.normal(0, 0.02),
                        "mean_gfp": 1.0 + 0.05 * participant_index + 0.2 * condition_index + 0.05 * state_index + rng.normal(0, 0.1),
                        "duration_s": 10,
                    })
        frame = pd.DataFrame(rows)
        for filename in SOURCES.values():
            frame.to_csv(root / filename, index=False)
        manifest = root / "vg_microstate_manifest_mat.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
        time_rows = []
        for participant_index, participant in enumerate(participants):
            group = "FS" if participant_index < 8 else "ES"
            file_path = str(root / f"{participant}_preictal.mat")
            for state_index, state in enumerate(STATES):
                for bin_from_ictus in range(1, 6):
                    minutes = -(bin_from_ictus - 0.5)
                    time_rows.append({
                        "participant": participant, "group": group, "condition": "preict",
                        "file_path": file_path, "backfit_method": "hard", "template_label": state,
                        "bin_from_ictus": bin_from_ictus, "minutes_to_ictus": minutes, "bin_duration_s": 60,
                        "coverage": 0.3 + 0.01 * minutes + 0.01 * state_index + rng.normal(0, 0.01),
                        "mean_gfp": 1.2 + 0.03 * minutes + 0.05 * state_index + rng.normal(0, 0.03),
                    })
        time_frame = pd.DataFrame(time_rows)
        for filename in TIME_SOURCES.values():
            time_frame.to_csv(root / filename, index=False)
        analyse(root, root / "analysis", manifest_csv=manifest)
        terms = pd.read_csv(root / "analysis" / "ce_group_condition_gee_terms.csv")
        assert set(terms["template_label"]) == set(STATES)
        assert (root / "analysis" / "ce_group_condition.png").exists()
        semiology_terms = pd.read_csv(root / "analysis" / "ce_semiology_gee_terms.csv")
        assert semiology_terms["term"].str.contains(r"C\(group\)", na=False).any()
        semiology_unit = pd.read_csv(root / "analysis" / "ce_semiology_participant_condition_metrics.csv")
        assert set(BASELINE_METRICS).issubset(semiology_unit.columns)
        assert (root / "analysis" / "ce_semiology_by_condition.png").exists()
        gradient_terms = pd.read_csv(root / "analysis" / "ce_preictal_gradient_gee_terms.csv")
        assert gradient_terms["term"].str.contains("minutes_to_ictus", na=False).any()
        assert (root / "analysis" / "ce_preictal_gradient.png").exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory containing hierarchical backfit CSVs")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None, help="VG manifest containing file_path and hypo_hyper")
    parser.add_argument("--method", default="hard")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        print("self-check passed")
        return
    out_dir = args.out_dir or args.results_dir / "ce_group_condition_analysis"
    analyse(args.results_dir, out_dir, args.method, args.manifest)
    print(f"C/E group x condition analysis: {out_dir}")


if __name__ == "__main__":
    main()
