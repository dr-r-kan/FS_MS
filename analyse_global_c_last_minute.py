"""Final-minute Gaussian-mixture trajectory analysis for global microstate C."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


METHOD = "gaussian_mixture"
STATE = "C"
GROUPS = ("FS", "ES")
SEMIOLOGY = ("hypo", "hyper")
METRICS = ("coverage", "mean_gfp")


def default_manifest(results_dir: Path) -> Path:
    for name in ("vg_microstate_manifest_mat.csv", "vg_microstate_manifest_set.csv", "vg_microstate_manifest_fif.csv"):
        path = results_dir.parent / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No VG manifest found beside {results_dir}")


def normalise_path(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace("/", "\\", regex=False).str.lower()


def load_data(results_dir: Path, manifest_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    time_path = results_dir / "global_preictal_time_metrics.csv"
    baseline_path = results_dir / "global_state_backfit_metrics.csv"
    time = pd.read_csv(time_path)
    baseline = pd.read_csv(baseline_path)
    manifest = pd.read_csv(manifest_csv, usecols=lambda c: c in {"file_path", "hypo_hyper"})
    if not {"file_path", "hypo_hyper"}.issubset(manifest.columns):
        raise ValueError(f"{manifest_csv} must contain file_path and hypo_hyper")

    time["_path"] = normalise_path(time["file_path"])
    manifest["_path"] = normalise_path(manifest["file_path"])
    time = time.merge(manifest[["_path", "hypo_hyper"]].drop_duplicates("_path"), on="_path", how="left")
    for frame in (time, baseline):
        frame["template_label"] = frame["template_label"].astype(str).str.strip().str.upper()
        frame["group"] = frame["group"].astype(str).str.strip().str.upper()
    time["hypo_hyper"] = time["hypo_hyper"].astype("string").str.strip().str.lower()
    for column in ("minutes_to_ictus", "bin_duration_s", *METRICS):
        time[column] = pd.to_numeric(time[column], errors="coerce")
    for column in ("backfit_available", "occupancy", "mean_gfp", "duration_s"):
        baseline[column] = pd.to_numeric(baseline[column], errors="coerce")

    time = time[
        time["backfit_method"].eq(METHOD)
        & time["template_label"].eq(STATE)
        & time["condition"].astype(str).str.lower().isin(("preict", "preictal"))
        & time["group"].isin(GROUPS)
        & time["hypo_hyper"].isin(SEMIOLOGY)
        & time["minutes_to_ictus"].between(-1, 0, inclusive="both")
    ].copy()
    if time.empty:
        raise ValueError(
            "No final-minute Gaussian-mixture rows for global C. Rerun the cached backfit "
            "with 'backfit_time_bin_s', 1 before running this script."
        )
    if time["minutes_to_ictus"].nunique() < 3:
        raise ValueError("Fewer than three final-minute time bins; rerun backfit with 'backfit_time_bin_s', 1.")

    baseline = baseline[
        baseline["backfit_method"].eq(METHOD)
        & baseline["backfit_available"].eq(1)
        & baseline["template_label"].eq(STATE)
        & baseline["condition"].astype(str).str.upper().eq("II")
        & baseline["group"].isin(GROUPS)
    ].copy()
    return time, baseline


def individual_ii_baselines(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (participant, group), part in data.groupby(["participant", "group"], sort=True):
        duration = part["duration_s"].clip(lower=0).fillna(0).to_numpy(float)
        coverage = part["occupancy"].to_numpy(float)
        gfp = part["mean_gfp"].to_numpy(float)
        valid_coverage = np.isfinite(coverage) & (duration > 0)
        ii_coverage = np.average(coverage[valid_coverage], weights=duration[valid_coverage]) if valid_coverage.any() else np.nan
        gfp_weights = duration * np.nan_to_num(coverage, nan=0.0)
        valid_gfp = np.isfinite(gfp) & (gfp_weights > 0)
        ii_gfp = np.average(gfp[valid_gfp], weights=gfp_weights[valid_gfp]) if valid_gfp.any() else np.nan
        rows.append((participant, group, ii_coverage, ii_gfp))
    return pd.DataFrame(rows, columns=["participant", "group", "ii_coverage", "ii_mean_gfp"])


def participant_bins(time: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    # Average seizures within participant and time bin so participants, not records, are the analysis units.
    unit = (
        time.groupby(["participant", "group", "hypo_hyper", "minutes_to_ictus"], observed=True)[list(METRICS)]
        .mean().reset_index()
    )
    unit = unit.merge(baselines, on=["participant", "group"], how="inner", validate="many_to_one")
    unit["coverage_change_from_II"] = unit["coverage"] - unit["ii_coverage"]
    unit["mean_gfp_change_from_II"] = unit["mean_gfp"] - unit["ii_mean_gfp"]
    unit["seconds_to_ictus"] = unit["minutes_to_ictus"] * 60
    unit["time_10s"] = unit["seconds_to_ictus"] / 10
    return unit


def descriptive_summary(unit: pd.DataFrame) -> pd.DataFrame:
    value_columns = (*METRICS, "coverage_change_from_II", "mean_gfp_change_from_II")
    long = unit.melt(
        id_vars=["participant", "group", "hypo_hyper", "minutes_to_ictus", "seconds_to_ictus"],
        value_vars=value_columns, var_name="metric", value_name="value",
    ).dropna(subset=["value"])
    summary = (
        long.groupby(["metric", "group", "hypo_hyper", "minutes_to_ictus", "seconds_to_ictus"], observed=True)["value"]
        .agg(n_participants="count", mean="mean", sd="std").reset_index()
    )
    summary["ci95"] = 1.96 * summary["sd"] / np.sqrt(summary["n_participants"])
    return summary


def participant_estimands(unit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (participant, group, semiology), part in unit.groupby(["participant", "group", "hypo_hyper"], observed=True):
        early, final = part[part["seconds_to_ictus"] <= -50], part[part["seconds_to_ictus"] >= -10]
        if early.empty or final.empty:
            continue
        for metric in METRICS:
            rows.append((participant, group, semiology, metric, "final_minus_first_10s",
                         final[metric].mean() - early[metric].mean()))
            rows.append((participant, group, semiology, metric, "final_10s_minus_II",
                         final[f"{metric}_change_from_II"].mean()))
    return pd.DataFrame(rows, columns=["participant", "group", "hypo_hyper", "metric", "estimand", "value"])


def estimand_stats(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    term_rows, contrast_rows = [], []
    rhs = "C(group, Treatment(reference='FS')) * C(hypo_hyper, Treatment(reference='hypo'))"
    for (metric, estimand), frame in data.groupby(["metric", "estimand"], observed=True):
        prefix = {"metric": metric, "estimand": estimand, "n_participants": frame["participant"].nunique()}
        fit = smf.ols(f"value ~ {rhs}", data=frame).fit(cov_type="HC3")
        for term, result in fit.wald_test_terms(scalar=True).table.iterrows():
            term_rows.append({**prefix, "term": term, "F": float(result["statistic"]),
                              "df": int(result["df_constraint"]), "p_value": float(result["pvalue"])})

        names = list(fit.params.index)
        group_name = next(name for name in names if "group" in name and ":" not in name)
        semiology_name = next(name for name in names if "hypo_hyper" in name and ":" not in name)
        interaction_name = next(name for name in names if ":" in name)
        vectors = {
            "FS hypo vs zero": {"Intercept": 1},
            "ES hypo vs zero": {"Intercept": 1, group_name: 1},
            "FS hyper vs zero": {"Intercept": 1, semiology_name: 1},
            "ES hyper vs zero": {"Intercept": 1, group_name: 1, semiology_name: 1, interaction_name: 1},
            "ES minus FS | hypo": {group_name: 1},
            "ES minus FS | hyper": {group_name: 1, interaction_name: 1},
            "FS/ES difference differs by semiology": {interaction_name: 1},
        }
        for label, weights in vectors.items():
            contrast = np.array([weights.get(name, 0) for name in names], dtype=float)
            test = fit.t_test(contrast)
            contrast_rows.append({**prefix, "contrast": label, "estimate": np.asarray(test.effect).item(),
                                  "se": np.asarray(test.sd).item(), "ci_low": float(test.conf_int()[0, 0]),
                                  "ci_high": float(test.conf_int()[0, 1]), "p_value": float(test.pvalue)})
    terms, contrasts = pd.DataFrame(term_rows), pd.DataFrame(contrast_rows)
    for frame in (terms, contrasts):
        frame["p_fdr"] = multipletests(frame["p_value"], method="fdr_bh")[1]
    return terms, contrasts


def plot(summary: pd.DataFrame, out_file: Path) -> None:
    panels = (("coverage", "Coverage"), ("coverage_change_from_II", "Coverage change from II"),
              ("mean_gfp", "Mean GFP"), ("mean_gfp_change_from_II", "Mean GFP change from II"))
    colours = {("FS", "hypo"): "#0072B2", ("FS", "hyper"): "#56B4E9", ("ES", "hypo"): "#D55E00", ("ES", "hyper"): "#E69F00"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, facecolor="white")
    for ax, (metric, title) in zip(axes.flat, panels):
        panel = summary[summary["metric"].eq(metric)]
        for (group, semiology), values in panel.groupby(["group", "hypo_hyper"], observed=True, sort=True):
            values = values.sort_values("seconds_to_ictus")
            x, mean, ci = values["seconds_to_ictus"], values["mean"], values["ci95"]
            ax.plot(x, mean, color=colours[(group, semiology)], label=f"{group} | {semiology}")
            ax.fill_between(x, mean - ci, mean + ci, color=colours[(group, semiology)], alpha=.15)
        if "change_from_II" in metric:
            ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Seconds to ictal onset")
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Global microstate C: Gaussian-mixture trajectory in the final preictal minute", fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_file, dpi=300, facecolor="white")
    plt.close(fig)


def analyse(results_dir: Path, out_dir: Path, manifest_csv: Path | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    time, baseline = load_data(results_dir, manifest_csv or default_manifest(results_dir))
    unit = participant_bins(time, individual_ii_baselines(baseline))
    if unit.empty:
        raise ValueError("No participants had both eligible preictal data and an individual II baseline.")
    summary = descriptive_summary(unit)
    estimands = participant_estimands(unit)
    terms, contrasts = estimand_stats(estimands)
    unit.to_csv(out_dir / "global_c_last_minute_gaussian_participant_bins.csv", index=False)
    summary.to_csv(out_dir / "global_c_last_minute_gaussian_summary.csv", index=False)
    estimands.to_csv(out_dir / "global_c_last_minute_gaussian_participant_estimands.csv", index=False)
    terms.to_csv(out_dir / "global_c_last_minute_gaussian_model_terms.csv", index=False)
    contrasts.to_csv(out_dir / "global_c_last_minute_gaussian_interpretable_contrasts.csv", index=False)
    (unit.groupby(["group", "hypo_hyper"])["participant"].nunique().rename("n_participants").reset_index()
     .to_csv(out_dir / "global_c_last_minute_gaussian_counts.csv", index=False))
    plot(summary, out_dir / "global_c_last_minute_gaussian.png")


def self_check() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, rng = Path(tmp), np.random.default_rng(9)
        time_rows, baseline_rows, manifest_rows = [], [], []
        for i in range(24):
            participant, group = f"P{i:02d}", ("FS" if i < 12 else "ES")
            semiology = pd.NA if i == 23 else ("hypo" if i % 2 == 0 else "hyper")
            baseline_rows.append({"participant": participant, "group": group, "condition": "II", "template_label": "C",
                                  "backfit_method": METHOD, "backfit_available": 1, "occupancy": .25 + .002*i,
                                  "mean_gfp": 1 + .01*i, "duration_s": 60})
            path = str(root / f"{participant}_preict.mat")
            manifest_rows.append({"file_path": path, "hypo_hyper": semiology})
            for b in range(6):
                seconds = -55 + 10*b
                time_rows.append({"participant": participant, "group": group, "condition": "preict", "file_path": path,
                                  "template_label": "C", "backfit_method": METHOD, "minutes_to_ictus": seconds/60,
                                  "bin_duration_s": 10, "coverage": .3 + .005*b + .01*(group == "ES") + rng.normal(0, .01),
                                  "mean_gfp": 1.1 + .01*b + rng.normal(0, .02)})
        pd.DataFrame(time_rows).to_csv(root / "global_preictal_time_metrics.csv", index=False)
        pd.DataFrame(baseline_rows).to_csv(root / "global_state_backfit_metrics.csv", index=False)
        manifest = root / "vg_microstate_manifest_mat.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
        analyse(root, root / "out", manifest)
        unit = pd.read_csv(root / "out" / "global_c_last_minute_gaussian_participant_bins.csv")
        assert unit["participant"].nunique() == 23 and unit["minutes_to_ictus"].nunique() == 6
        estimands = pd.read_csv(root / "out" / "global_c_last_minute_gaussian_participant_estimands.csv")
        assert set(estimands["estimand"]) == {"final_minus_first_10s", "final_10s_minus_II"}
        assert (root / "out" / "global_c_last_minute_gaussian_interpretable_contrasts.csv").exists()
        assert (root / "out" / "global_c_last_minute_gaussian.png").exists()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("self-check passed")
        return
    out_dir = args.out_dir or args.results_dir / "global_c_last_minute_gaussian"
    analyse(args.results_dir, out_dir, args.manifest)
    print(f"Global C final-minute analysis: {out_dir}")


if __name__ == "__main__":
    main()
