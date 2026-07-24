"""Microstate systolic/diastolic occurrence-rate ratios with ten planned comparisons."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t
from statsmodels.stats.multitest import multipletests

from analyse_ce_group_condition import default_manifest


NAMES = [
    "People: ES only vs FS only", "People: DD vs FS only", "People: DD vs ES only",
    "Events: Preict functional seizure vs preict epileptic seizure",
    "Functional events (FS or DD): II vs Preict", "Epileptic events (ES or DD): II vs Preict",
    "Within DD: Preict epileptic vs preict functional event",
    "Epileptic events (ES or DD): Preict vs Postict", "Functional events (FS or DD): Preict vs Postict",
    "Functional events (FS or DD): Preict Hypomotor vs Preict Hypermotor",
]


def _times(value) -> np.ndarray:
    if pd.isna(value) or not str(value).strip():
        return np.empty(0)
    return np.fromstring(str(value), sep=";")


def apply_systolic_window(data: pd.DataFrame, start_ms: float, end_ms: float) -> pd.DataFrame:
    if not 0 <= start_ms < end_ms:
        raise ValueError("Systolic interval must satisfy 0 <= start_ms < end_ms")
    onset_column = "state_occurrence_times_s" if "state_occurrence_times_s" in data else "c_occurrence_times_s"
    required = {onset_column, "rpeak_times_relative_s", "duration_s"}
    if not required.issubset(data.columns):
        return data
    data = data.copy()
    phase_columns = ["systolic_occurrences", "diastolic_occurrences", "systolic_exposure_s", "diastolic_exposure_s"]
    data[phase_columns] = data[phase_columns].astype(float)
    start_s, end_s = start_ms / 1000, end_ms / 1000
    for index, row in data.iterrows():
        duration = float(row["duration_s"]); onsets = np.sort(_times(row[onset_column])); rpeaks = _times(row["rpeak_times_relative_s"])
        intervals = sorted((max(0, r+start_s), min(duration, r+end_s)) for r in rpeaks if r+end_s > 0 and r+start_s < duration)
        merged = []
        for left, right in intervals:
            if right <= left: continue
            if merged and left <= merged[-1][1]: merged[-1] = (merged[-1][0], max(merged[-1][1], right))
            else: merged.append((left, right))
        exposure = sum(right-left for left, right in merged)
        count = sum(np.searchsorted(onsets, right, side="left")-np.searchsorted(onsets, left, side="left") for left, right in merged)
        available = duration > exposure > 0 and len(rpeaks) > 0
        data.loc[index, phase_columns] = (
            count, len(onsets)-count, exposure, duration-exposure) if available else (np.nan, np.nan, np.nan, np.nan)
        data.at[index, "cardiac_phase_available"] = int(available)
    data["systolic_window_start_ms"] = start_ms; data["systolic_window_end_ms"] = end_ms
    return data


def aggregate(data: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    sums = (data.groupby(keys, observed=True)[
        ["systolic_occurrences", "diastolic_occurrences", "systolic_exposure_s", "diastolic_exposure_s"]
    ].sum().reset_index())
    sums["systolic_rate_hz"] = sums["systolic_occurrences"] / sums["systolic_exposure_s"]
    sums["diastolic_rate_hz"] = sums["diastolic_occurrences"] / sums["diastolic_exposure_s"]
    sums["rate_ratio"] = sums["systolic_rate_hz"] / sums["diastolic_rate_hz"]
    sums["log_rate_ratio"] = (
        np.log((sums["systolic_occurrences"] + .5) / sums["systolic_exposure_s"])
        - np.log((sums["diastolic_occurrences"] + .5) / sums["diastolic_exposure_s"])
    )
    return sums


def load_units(results_dir: Path, manifest_csv: Path, start_ms: float, end_ms: float,
               metrics_file: str = "global_c_cardiac_phase_metrics.csv", state: str = "C") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(results_dir / metrics_file)
    if "template_label" not in data:
        data["template_label"] = "C"
    data["template_label"] = data["template_label"].astype(str).str.upper().str.strip()
    data = data[data["template_label"].eq(state.upper())].copy()
    if data.empty:
        raise ValueError(f"No participant-wise microstate {state.upper()} rows in {metrics_file}")
    data = apply_systolic_window(data, start_ms, end_ms)
    manifest = pd.read_csv(manifest_csv, usecols=lambda c: c in {"file_path", "seizure_type", "hypo_hyper"})
    normalise = lambda x: x.astype(str).str.replace("/", "\\", regex=False).str.lower()
    data["_path"] = normalise(data["file_path"]); manifest["_path"] = normalise(manifest["file_path"])
    data = data.merge(manifest[["_path", "seizure_type", "hypo_hyper"]].drop_duplicates("_path"), on="_path", how="left")
    data["group"] = data["group"].astype(str).str.upper().str.strip()
    data["condition"] = data["condition"].astype(str).str.lower().str.strip().replace({"ii": "II", "preict": "preictal", "postict": "postictal"})
    data["event_type"] = data["seizure_type"].astype("string").str.upper().str.strip().replace({"FS": "NEAD"})
    data["semiology"] = data["hypo_hyper"].astype("string").str.lower().str.strip().replace({"hypo": "Hypomotor", "hyper": "Hypermotor"})
    data[["event_type", "semiology"]] = data[["event_type", "semiology"]].fillna("")
    numeric = ["systolic_occurrences", "diastolic_occurrences", "systolic_exposure_s", "diastolic_exposure_s"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data = data[data["cardiac_phase_available"].eq(1) & data[numeric].notna().all(axis=1)].copy()
    main = aggregate(data, ["participant", "group", "condition"])
    event = aggregate(data, ["participant", "group", "condition", "event_type"])
    semiology = aggregate(data, ["participant", "group", "condition", "event_type", "semiology"])
    return main, event, semiology


def planned_sets(main: pd.DataFrame, event: pd.DataFrame, semiology: pd.DataFrame):
    def group_pair(a, b):
        subset = main[main["group"].isin((a, b))]
        return aggregate(subset, ["participant", "group"])
    sets = [
        (group_pair("FS", "ES"), "group", "FS", "ES", "independent"),
        (group_pair("FS", "DD"), "group", "FS", "DD", "independent"),
        (group_pair("ES", "DD"), "group", "ES", "DD", "independent"),
        (event[event["condition"].eq("preictal") & event["event_type"].isin(("NEAD", "ES"))], "event_type", "NEAD", "ES", "clustered"),
        (event[event["group"].isin(("FS", "DD")) & (event["condition"].eq("II") | (event["condition"].eq("preictal") & event["event_type"].eq("NEAD")))], "condition", "II", "preictal", "paired"),
        (event[event["group"].isin(("ES", "DD")) & (event["condition"].eq("II") | (event["condition"].eq("preictal") & event["event_type"].eq("ES")))], "condition", "II", "preictal", "paired"),
        (event[event["group"].eq("DD") & event["condition"].eq("preictal") & event["event_type"].isin(("NEAD", "ES"))], "event_type", "NEAD", "ES", "paired"),
        (event[event["group"].isin(("ES", "DD")) & event["event_type"].eq("ES") & event["condition"].isin(("preictal", "postictal"))], "condition", "preictal", "postictal", "paired"),
        (event[event["group"].isin(("FS", "DD")) & event["event_type"].eq("NEAD") & event["condition"].isin(("preictal", "postictal"))], "condition", "preictal", "postictal", "paired"),
        (semiology[semiology["group"].isin(("FS", "DD")) & semiology["condition"].eq("preictal") & semiology["event_type"].eq("NEAD") & semiology["semiology"].isin(("Hypomotor", "Hypermotor"))], "semiology", "Hypomotor", "Hypermotor", "paired"),
    ]
    return sets


def comparison(name: str, data: pd.DataFrame, factor: str, a: str, b: str, design: str) -> dict:
    if design == "paired":
        wide = data.pivot_table(index="participant", columns=factor, values="log_rate_ratio", aggfunc="mean").dropna(subset=[a, b])
        difference = (wide[b] - wide[a]).to_numpy(float); x, y = wide[a].to_numpy(float), wide[b].to_numpy(float)
        n_a = n_b = len(difference); estimate = difference.mean() if len(difference) else np.nan
        se = difference.std(ddof=1) / np.sqrt(len(difference)) if len(difference) > 1 else np.nan
        df = len(difference) - 1; statistic = estimate / se if se > 0 else np.nan
        test = "paired t-test on log occurrence-rate ratio"
    elif design == "clustered":
        frame=data[data[factor].isin((a,b))].dropna(subset=["log_rate_ratio"]).copy()
        fit=smf.ols(f"log_rate_ratio ~ C({factor}, Treatment(reference={a!r}))",data=frame).fit(
            cov_type="cluster",cov_kwds={"groups":frame["participant"]})
        key=next(name for name in fit.params.index if name!="Intercept")
        x=frame.loc[frame[factor].eq(a),"log_rate_ratio"].to_numpy(float); y=frame.loc[frame[factor].eq(b),"log_rate_ratio"].to_numpy(float)
        n_a=frame.loc[frame[factor].eq(a),"participant"].nunique(); n_b=frame.loc[frame[factor].eq(b),"participant"].nunique()
        estimate=float(fit.params[key]); se=float(fit.bse[key]); df=frame["participant"].nunique()-1
        statistic=estimate/se if se>0 else np.nan; test="participant-clustered regression on log occurrence-rate ratio"
    else:
        x = data.loc[data[factor].eq(a), "log_rate_ratio"].dropna().to_numpy(float)
        y = data.loc[data[factor].eq(b), "log_rate_ratio"].dropna().to_numpy(float)
        n_a, n_b = len(x), len(y); estimate = y.mean() - x.mean() if n_a and n_b else np.nan
        va, vb = x.var(ddof=1), y.var(ddof=1)
        se = np.sqrt(va/n_a + vb/n_b) if n_a > 1 and n_b > 1 else np.nan
        df = (va/n_a + vb/n_b)**2 / ((va/n_a)**2/(n_a-1) + (vb/n_b)**2/(n_b-1)) if se > 0 else np.nan
        statistic = estimate / se if se > 0 else np.nan
        test = "Welch t-test on log occurrence-rate ratio"
    p_value = 2*t.sf(abs(statistic), df) if np.isfinite(statistic) and df > 0 else np.nan
    critical = t.ppf(.975, df) if np.isfinite(df) and df > 0 else np.nan
    return {"comparison": name, "level_a": a, "level_b": b, "design": design, "test": test,
            "n_a": n_a, "n_b": n_b, "mean_log_ratio_a": np.mean(x) if len(x) else np.nan,
            "mean_log_ratio_b": np.mean(y) if len(y) else np.nan, "geometric_rate_ratio_a": np.exp(np.mean(x)) if len(x) else np.nan,
            "geometric_rate_ratio_b": np.exp(np.mean(y)) if len(y) else np.nan, "difference_b_minus_a_log": estimate,
            "ratio_of_rate_ratios_b_over_a": np.exp(estimate), "ci_low_log": estimate-critical*se,
            "ci_high_log": estimate+critical*se, "ci_low_ratio_of_ratios": np.exp(estimate-critical*se),
            "ci_high_ratio_of_ratios": np.exp(estimate+critical*se), "statistic": statistic, "df": df, "p_value": p_value}


def analyse(results_dir: Path, out_dir: Path, manifest_csv: Path | None = None, start_ms: float = 180, end_ms: float = 500,
            metrics_file: str = "global_c_cardiac_phase_metrics.csv", states: tuple[str, ...] = ("C",), prefix: str = "global_c") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    unit_tables = [[], [], []]; stat_tables = []
    for state in states:
        main, event, semiology = load_units(results_dir, manifest_csv or default_manifest(results_dir), start_ms, end_ms, metrics_file, state)
        for frame, collection in zip((main, event, semiology), unit_tables):
            frame.insert(0, "template_label", state.upper()); collection.append(frame)
        state_stats = pd.DataFrame([comparison(name, *spec) for name, spec in zip(NAMES, planned_sets(main, event, semiology))])
        state_stats.insert(0, "template_label", state.upper()); stat_tables.append(state_stats)
    main, event, semiology = (pd.concat(parts, ignore_index=True) for parts in unit_tables)
    stats = pd.concat(stat_tables, ignore_index=True)
    stats["systolic_window_start_ms"] = start_ms; stats["systolic_window_end_ms"] = end_ms
    stats["p_holm"] = np.nan; valid = stats["p_value"].notna()
    if valid.any(): stats.loc[valid, "p_holm"] = multipletests(stats.loc[valid, "p_value"], method="holm")[1]
    main.to_csv(out_dir / f"{prefix}_cardiac_phase_participant_condition.csv", index=False)
    event.to_csv(out_dir / f"{prefix}_cardiac_phase_participant_event_type.csv", index=False)
    semiology.to_csv(out_dir / f"{prefix}_cardiac_phase_participant_semiology.csv", index=False)
    stats.to_csv(out_dir / f"{prefix}_cardiac_phase_planned_comparisons.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, max(6, .42*len(stats))), facecolor="white")
    y = np.arange(len(stats)); estimate = stats["ratio_of_rate_ratios_b_over_a"]
    ax.errorbar(estimate, y, xerr=[estimate-stats["ci_low_ratio_of_ratios"], stats["ci_high_ratio_of_ratios"]-estimate], fmt="o", capsize=3)
    labels = stats["template_label"] + " | " + stats["comparison"] if len(states) > 1 else stats["comparison"]
    ax.axvline(1, color="black", linewidth=.8); ax.set_xscale("log"); ax.set_yticks(y, labels)
    ax.invert_yaxis(); ax.set_xlabel("Ratio of systolic/diastolic occurrence-rate ratios (B / A)"); ax.grid(axis="x", alpha=.2)
    fig.tight_layout(); fig.savefig(out_dir / f"{prefix}_cardiac_phase_forest.png", dpi=300, facecolor="white"); plt.close(fig)


def self_check() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, rows, manifest = Path(tmp), [], []
        for group in ("FS", "ES", "DD"):
            for i in range(8):
                participant = f"{group}{i}"
                for condition in ("II", "preict", "postict"):
                    types = ("FS", "ES") if group == "DD" and condition != "II" else (("FS",) if group == "FS" else (("ES",) if group == "ES" else ("",)))
                    for event_type in types:
                        sems = ("hypo", "hyper") if group == "FS" and condition == "preict" else ("hypo",)
                        for sem in sems:
                            path = str(root/f"{participant}_{condition}_{event_type}_{sem}.mat")
                            extra=((condition!="II")*(i%3)+(condition=="postict")*(i%2)
                                   +(sem=="hyper")*(i%2)+(event_type=="ES")*(i%2))
                            cache_onsets="0.2;0.7;1.3;2.4;3.1"+(";2.2" if extra%2 else "")+(";0.8" if extra%3==2 else "")
                            rows.append({"participant":participant,"group":group,"condition":condition,"file_path":path,
                                         "cardiac_phase_available":1,"systolic_occurrences":12+i+extra,"diastolic_occurrences":30+i,
                                         "systolic_exposure_s":20,"diastolic_exposure_s":80,"duration_s":100,
                                         "c_occurrence_times_s":cache_onsets,"rpeak_times_relative_s":"0;1;2;3"})
                            manifest.append({"file_path":path,"seizure_type":event_type,"hypo_hyper":sem})
        pd.DataFrame(rows).to_csv(root/"global_c_cardiac_phase_metrics.csv",index=False)
        pd.DataFrame(manifest).to_csv(root/"vg_microstate_manifest_mat.csv",index=False)
        analyse(root,root/"out",root/"vg_microstate_manifest_mat.csv",180,500)
        result=pd.read_csv(root/"out"/"global_c_cardiac_phase_planned_comparisons.csv")
        assert list(result["comparison"])==NAMES
        assert result["p_value"].notna().all(), result.loc[result["p_value"].isna(), ["comparison","n_a","n_b"]]
        cached = pd.read_csv(root/"global_c_cardiac_phase_metrics.csv")
        cached["template_label"] = "C"; cached["state_occurrence_times_s"] = cached["c_occurrence_times_s"]
        e_cached = cached.copy(); e_cached["template_label"] = "E"
        pd.concat((cached, e_cached), ignore_index=True).to_csv(root/"participant_ce_cardiac_phase_metrics.csv", index=False)
        analyse(root,root/"out_ce",root/"vg_microstate_manifest_mat.csv",180,500,
                "participant_ce_cardiac_phase_metrics.csv",("C","E"),"participant_ce")
        ce_result=pd.read_csv(root/"out_ce"/"participant_ce_cardiac_phase_planned_comparisons.csv")
        assert len(ce_result)==20 and set(ce_result["template_label"])=={"C","E"}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--results-dir",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path); parser.add_argument("--manifest",type=Path)
    parser.add_argument("--metrics-file",default="global_c_cardiac_phase_metrics.csv")
    parser.add_argument("--states",nargs="+",default=["C"]); parser.add_argument("--prefix",default="global_c")
    parser.add_argument("--systolic-start-ms",type=float,default=180); parser.add_argument("--systolic-end-ms",type=float,default=500)
    parser.add_argument("--self-check",action="store_true")
    args=parser.parse_args()
    if args.self_check: self_check(); print("self-check passed"); return
    out=args.out_dir or args.results_dir/f"{args.prefix}_cardiac_phase_{args.systolic_start_ms:g}_{args.systolic_end_ms:g}ms"
    analyse(args.results_dir,out,args.manifest,args.systolic_start_ms,args.systolic_end_ms,
            args.metrics_file,tuple(args.states),args.prefix)
    print(f"Cardiac-phase analysis: {out}")


if __name__ == "__main__": main()
