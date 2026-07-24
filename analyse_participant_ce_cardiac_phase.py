"""Test whether participant-wise microstate C/E cardiac-phase ratios predict clinical labels."""

from __future__ import annotations

import argparse
import tempfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from analyse_ce_group_condition import default_manifest
from analyse_global_c_cardiac_phase import apply_systolic_window, aggregate


PREDICTIONS = (
    ("seizure_type", "event_type", "ES", "NEAD", ("preictal",)),
    ("semiology", "semiology", "Hypermotor", "Hypomotor", ("preictal",)),
    ("participant_diagnosis", "group", "ES", "FS", ("II", "preictal", "postictal")),
    ("preictal_vs_interictal", "condition", "preictal", "II", ("II", "preictal")),
    ("postictal_vs_preictal", "condition", "postictal", "preictal", ("preictal", "postictal")),
)


def load_data(results_dir: Path, manifest_csv: Path, start_ms: float, end_ms: float) -> pd.DataFrame:
    path = results_dir / "participant_ce_cardiac_phase_metrics.csv"
    data = pd.read_csv(path)
    required = {
        "participant", "group", "condition", "file_path", "template_label",
        "systolic_occurrences", "diastolic_occurrences",
        "systolic_exposure_s", "diastolic_exposure_s",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    data["template_label"] = data["template_label"].astype(str).str.upper().str.strip()
    data = data[data["template_label"].isin(("C", "E"))].copy()
    if set(data["template_label"]) != {"C", "E"}:
        raise ValueError(f"{path} must contain participant-wise C and E rows")
    data = apply_systolic_window(data, start_ms, end_ms)

    manifest = pd.read_csv(manifest_csv, usecols=lambda c: c in {"file_path", "seizure_type", "hypo_hyper"})
    if not {"file_path", "seizure_type", "hypo_hyper"}.issubset(manifest):
        raise ValueError(f"{manifest_csv} must contain file_path, seizure_type, and hypo_hyper")
    normalise = lambda values: values.astype(str).str.replace("/", "\\", regex=False).str.lower()
    data["_path"] = normalise(data["file_path"])
    manifest["_path"] = normalise(manifest["file_path"])
    data = data.merge(
        manifest[["_path", "seizure_type", "hypo_hyper"]].drop_duplicates("_path"),
        on="_path", how="left",
    )
    data["group"] = data["group"].astype(str).str.upper().str.strip()
    data["condition"] = data["condition"].astype(str).str.lower().str.strip().replace(
        {"ii": "II", "preict": "preictal", "postict": "postictal"}
    )
    data["event_type"] = data["seizure_type"].astype("string").str.upper().str.strip().replace({"FS": "NEAD"})
    data["semiology"] = data["hypo_hyper"].astype("string").str.lower().str.strip().replace(
        {"hypo": "Hypomotor", "hyper": "Hypermotor"}
    )
    numeric = ["systolic_occurrences", "diastolic_occurrences", "systolic_exposure_s", "diastolic_exposure_s"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    available = pd.to_numeric(
        data["cardiac_phase_available"] if "cardiac_phase_available" in data else pd.Series(1, index=data.index),
        errors="coerce",
    ).eq(1)
    return data[available & data[numeric].notna().all(axis=1)].copy()


def ce_ratios(data: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    unit = aggregate(data, [*keys, "template_label"])
    wide = unit.pivot_table(index=keys, columns="template_label", values="log_rate_ratio").reset_index()
    if not {"C", "E"}.issubset(wide):
        return pd.DataFrame(columns=[*keys, "log_ratio_C", "log_ratio_E"])
    return wide.rename(columns={"C": "log_ratio_C", "E": "log_ratio_E"}).dropna(
        subset=["log_ratio_C", "log_ratio_E"]
    )


def prediction_units(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for name, outcome, positive, negative, conditions in PREDICTIONS:
        subset = data[data["condition"].isin(conditions)].copy()
        keys = ["participant", outcome]
        if name == "semiology":
            keys.append("event_type")
        if name == "postictal_vs_preictal":
            keys.append("event_type")
            subset = subset[subset["event_type"].isin(("ES", "NEAD"))]
        unit = ce_ratios(subset[subset[outcome].isin((positive, negative))], keys)
        unit["outcome"] = unit[outcome].eq(positive).astype(int)
        unit["outcome_name"] = name
        frames[name] = unit
    return frames


def fit_prediction(name: str, data: pd.DataFrame, positive: str, negative: str) -> pd.DataFrame:
    prefix = {
        "prediction": name, "positive_class": positive, "reference_class": negative,
        "n_observations": len(data), "n_participants": data["participant"].nunique(),
        "n_positive": int(data["outcome"].sum()) if len(data) else 0,
    }
    if len(data) < 8 or data["outcome"].nunique() < 2 or data["participant"].nunique() < 4:
        return pd.DataFrame([{**prefix, "term": "", "status": "insufficient observations, classes, or participants"}])
    frame = data.copy()
    for state in ("C", "E"):
        source = f"log_ratio_{state}"
        sd = frame[source].std()
        if not np.isfinite(sd) or sd == 0:
            return pd.DataFrame([{**prefix, "term": "", "status": f"no variation in {source}"}])
        frame[f"z_{state}"] = (frame[source] - frame[source].mean()) / sd
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerfectSeparationWarning)
            if len(frame) == frame["participant"].nunique():
                fit = smf.glm("outcome ~ z_C + z_E", data=frame, family=Binomial()).fit()
                model = "participant-level logistic regression"
            else:
                fit = smf.gee(
                    "outcome ~ z_C + z_E", groups="participant", data=frame,
                    family=Binomial(), cov_struct=Exchangeable(),
                ).fit()
                model = "participant-clustered logistic GEE"
    except Exception as error:
        return pd.DataFrame([{**prefix, "term": "", "status": f"model failed: {error}"}])
    ci = fit.conf_int()
    rows = []
    for term in ("z_C", "z_E"):
        estimate, low, high = float(fit.params[term]), float(ci.loc[term, 0]), float(ci.loc[term, 1])
        rows.append({
            **prefix, "model": model, "term": term, "log_odds": estimate, "std_error": float(fit.bse[term]),
            "odds_ratio_per_sd": np.exp(estimate), "ci_low": np.exp(low), "ci_high": np.exp(high),
            "statistic": float(fit.tvalues[term]), "p_value": float(fit.pvalues[term]), "status": "ok",
        })
    return pd.DataFrame(rows)


def analyse(results_dir: Path, out_dir: Path, manifest_csv: Path | None = None,
            start_ms: float = 180, end_ms: float = 500) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(results_dir, manifest_csv or default_manifest(results_dir), start_ms, end_ms)
    units = prediction_units(data)
    results = pd.concat([
        fit_prediction(name, units[name], positive, negative)
        for name, _, positive, negative, _ in PREDICTIONS
    ], ignore_index=True)
    results["p_holm"] = np.nan
    valid = results["p_value"].notna() if "p_value" in results else pd.Series(False, index=results.index)
    if valid.any():
        results.loc[valid, "p_holm"] = multipletests(results.loc[valid, "p_value"], method="holm")[1]
    results["systolic_window_start_ms"] = start_ms
    results["systolic_window_end_ms"] = end_ms
    pd.concat(units.values(), ignore_index=True).to_csv(out_dir / "participant_ce_prediction_units.csv", index=False)
    results.to_csv(out_dir / "participant_ce_prediction_models.csv", index=False)

    plotted = results[results["status"].eq("ok")].copy()
    if not plotted.empty:
        fig, ax = plt.subplots(figsize=(8, max(4, .45 * len(plotted))), facecolor="white")
        y = np.arange(len(plotted))
        estimate = plotted["odds_ratio_per_sd"].to_numpy(float)
        ax.errorbar(estimate, y, xerr=[estimate-plotted["ci_low"], plotted["ci_high"]-estimate], fmt="o", capsize=3)
        ax.set_yticks(y, plotted["prediction"] + " | " + plotted["term"].str.removeprefix("z_"))
        ax.axvline(1, color="black", linewidth=.8)
        ax.set_xscale("log"); ax.invert_yaxis(); ax.grid(axis="x", alpha=.2)
        ax.set_xlabel("Odds ratio per 1-SD increase in log systolic/diastolic rate ratio")
        fig.tight_layout()
        fig.savefig(out_dir / "participant_ce_prediction_forest.png", dpi=300, facecolor="white")
        plt.close(fig)


def self_check() -> None:
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); rows = []; manifest = []
        for i in range(24):
            participant = f"P{i:02d}"; group = "FS" if i < 12 else "ES"
            event_type = "FS" if group == "FS" else "ES"
            for condition in ("II", "preict", "postict"):
                for semiology in ("hypo", "hyper"):
                    path = str(root / f"{participant}_{condition}_{semiology}.mat")
                    manifest.append({"file_path": path, "seizure_type": event_type, "hypo_hyper": semiology})
                    for state in ("C", "E"):
                        effect = (group == "ES") + (semiology == "hyper") + (condition == "postict")
                        count = 8 + effect + (state == "E") + rng.integers(0, 9)
                        rows.append({
                            "participant": participant, "group": group, "condition": condition, "file_path": path,
                            "template_label": state, "cardiac_phase_available": 1,
                            "systolic_occurrences": count, "diastolic_occurrences": 30 + i % 5,
                            "systolic_exposure_s": 20, "diastolic_exposure_s": 80,
                        })
        pd.DataFrame(rows).to_csv(root / "participant_ce_cardiac_phase_metrics.csv", index=False)
        pd.DataFrame(manifest).to_csv(root / "manifest.csv", index=False)
        analyse(root, root / "out", root / "manifest.csv")
        result = pd.read_csv(root / "out" / "participant_ce_prediction_models.csv")
        assert {"z_C", "z_E"}.issubset(set(result.loc[result["status"].eq("ok"), "term"]))
        assert (root / "out" / "participant_ce_prediction_forest.png").exists()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--systolic-start-ms", type=float, default=180)
    parser.add_argument("--systolic-end-ms", type=float, default=500)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(); print("self-check passed"); return
    out_dir = args.out_dir or args.results_dir / "participant_ce_cardiac_phase_prediction"
    analyse(args.results_dir, out_dir, args.manifest, args.systolic_start_ms, args.systolic_end_ms)
    print(f"Participant C/E cardiac-phase prediction analysis: {out_dir}")


if __name__ == "__main__":
    main()
