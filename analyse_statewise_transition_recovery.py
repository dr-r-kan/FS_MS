"""State-wise transition precision/JS and postictal recovery without refitting microstates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import analyse_ace_cardioneural as ace
from analyse_preictal_publication import attach_strata, piecewise_columns
from microstate_labels import load_global_labels


def transition_matrix(frame: pd.DataFrame) -> np.ndarray:
    counts = np.zeros((6, 6), dtype=float)
    for row in frame.itertuples(index=False):
        i, j = int(row.state_from_index) - 1, int(row.state_to_index) - 1
        if 0 <= i < 6 and 0 <= j < 6 and i != j:
            counts[i, j] += float(row.transition_count)
    return counts


def interictal_baselines(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    columns = ["participant", "condition", "backfit_method", "state_from_index", "state_to_index", "transition_count"]
    table = pd.read_csv(path, usecols=columns)
    table = table[table["condition"].astype(str).str.upper().eq("II") & table["backfit_method"].eq("hard")]
    output = {}
    for participant, frame in table.groupby("participant", observed=True):
        counts = transition_matrix(frame)
        probs = counts + 0.5
        np.fill_diagonal(probs, 0)
        probs /= probs.sum(axis=1, keepdims=True)
        output[str(participant)] = (probs, counts.sum(axis=1))
    return output


def state_metrics(counts: np.ndarray, b0: np.ndarray, ii_n: np.ndarray,
                  labels: tuple[str, ...], minimum: int) -> list[dict]:
    rows = []
    for i, label in enumerate(labels):
        observed = int(counts[i].sum())
        enough = observed >= minimum and ii_n[i] >= minimum
        log_kappa, _ = ace.fit_kappa(counts, b0, (i,), minimum) if enough else (np.nan, observed)
        rows.append({
            "state_index": i + 1,
            "template_label": label,
            "log_kappa": log_kappa,
            "kappa": math.exp(log_kappa) if np.isfinite(log_kappa) else np.nan,
            "abs_log_kappa": abs(log_kappa) if np.isfinite(log_kappa) else np.nan,
            "js_from_II": ace.js_divergence(counts, b0, (i,)) if enough else np.nan,
            "outgoing_transitions": observed,
            "ii_outgoing_transitions": int(ii_n[i]),
        })
    return rows


def preictal_metrics(results_dir: Path, manifest: pd.DataFrame, labels: tuple[str, ...],
                     baselines: dict[str, tuple[np.ndarray, np.ndarray]], minimum: int):
    columns = ["participant", "file_path", "backfit_method", "template_label", "bin_from_ictus", "collapsed_sequence"]
    table = pd.read_csv(results_dir / "participant_condition_preictal_time_metrics.csv", usecols=columns)
    table = table[table["backfit_method"].eq("hard") & table["template_label"].eq(labels[0])].copy()
    table["file_key"] = table["file_path"].astype(str).str.replace("/", "\\", regex=False).str.lower()
    table = table.merge(manifest[["file_key", "stratum", "hypo_hyper"]], on="file_key", how="left", validate="many_to_one")
    windows, late = [], []
    for (participant, file_path), frame in table.groupby(["participant", "file_path"], observed=True, sort=False):
        baseline = baselines.get(str(participant))
        if baseline is None:
            continue
        b0, ii_n = baseline
        info = frame.iloc[0]
        sequences = {int(row.bin_from_ictus): ace.parse_sequence(row.collapsed_sequence)
                     for row in frame.itertuples(index=False)}
        max_bin = max(sequences, default=0)
        for start in range(1, max_bin - 4):
            bins = list(range(start + 5, start - 1, -1))
            if not all(value in sequences for value in bins):
                continue
            counts = ace.transition_counts(ace.collapse([state for value in bins for state in sequences[value]]))
            meta = {"participant": participant, "file_path": file_path, "stratum": info.stratum,
                    "hypo_hyper": info.hypo_hyper, "phase": "preictal_60s",
                    "window_end_to_ictus_s": (start - 1) * 10, "minutes_to_ictus": -(start - 1) / 6}
            windows.extend(meta | row for row in state_metrics(counts, b0, ii_n, labels, minimum))
        late_bins = list(range(30, 0, -1))
        if all(value in sequences for value in late_bins):
            counts = ace.transition_counts(ace.collapse([state for value in late_bins for state in sequences[value]]))
            meta = {"participant": participant, "file_path": file_path, "stratum": info.stratum,
                    "hypo_hyper": info.hypo_hyper, "phase": "late_preictal_5min"}
            late.extend(meta | row for row in state_metrics(counts, b0, ii_n, labels, minimum))
    return ace.add_cohort(pd.DataFrame(windows)), ace.add_cohort(pd.DataFrame(late))


def postictal_metrics(results_dir: Path, manifest: pd.DataFrame, labels: tuple[str, ...],
                      baselines: dict[str, tuple[np.ndarray, np.ndarray]], minimum: int) -> pd.DataFrame:
    columns = ["participant", "condition", "file_path", "backfit_method", "valid_samples", "sfreq",
               "state_from_index", "state_to_index", "transition_count"]
    table = pd.read_csv(results_dir / "participant_condition_transition_metrics.csv", usecols=columns)
    table = table[table["condition"].astype(str).str.lower().eq("postict") & table["backfit_method"].eq("hard")].copy()
    table["file_key"] = table["file_path"].astype(str).str.replace("/", "\\", regex=False).str.lower()
    table = table.merge(manifest[["file_key", "stratum", "hypo_hyper"]], on="file_key", how="left", validate="many_to_one")
    rows = []
    for (participant, file_path), frame in table.groupby(["participant", "file_path"], observed=True, sort=False):
        baseline = baselines.get(str(participant))
        if baseline is None:
            continue
        b0, ii_n = baseline
        info = frame.iloc[0]
        counts = transition_matrix(frame)
        duration = float(info.valid_samples) / float(info.sfreq) if float(info.sfreq) > 0 else np.nan
        meta = {"participant": participant, "file_path": file_path, "stratum": info.stratum,
                "hypo_hyper": info.hypo_hyper, "phase": "postictal_full_record", "duration_s": duration}
        rows.extend(meta | row for row in state_metrics(counts, b0, ii_n, labels, minimum))
    return ace.add_cohort(pd.DataFrame(rows))


def interaction_term(parameters, prefix: str) -> str:
    hits = [name for name in parameters if prefix in name and "semiology" in name and "hypo" in name]
    if len(hits) != 1:
        raise KeyError(f"Expected one {prefix} by semiology interaction, found {hits}")
    return hits[0]


def preictal_models(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = ["participant", "stratum", "cohort", "semiology", "template_label",
             "window_end_to_ictus_s", "minutes_to_ictus"]
    participant = windows.groupby(group, observed=True, dropna=False)[["log_kappa", "js_from_II"]].mean().reset_index()
    participant = piecewise_columns(participant)
    rows = []
    formula = "{outcome} ~ early_elapsed_min + late_elapsed_min + C(semiology, Treatment('hyper')) + C(cohort, Treatment('FS')) + late_elapsed_min:C(semiology, Treatment('hyper'))"
    for label in ace.LABELS:
        frame = participant[participant["template_label"].eq(label) & participant["stratum"].isin(ace.FUNCTIONAL) & participant["semiology"].notna()]
        for outcome in ("log_kappa", "js_from_II"):
            try:
                fit, _ = ace.fit_gee(frame, formula.format(outcome=outcome), f"state {label} preictal {outcome}")
                interaction = interaction_term(fit.params.index, "late_elapsed_min:")
                contrasts = (
                    ace.linear_contrast(fit, {"late_elapsed_min": 1}, "hypermotor late slope"),
                    ace.linear_contrast(fit, {"late_elapsed_min": 1, interaction: 1}, "hypomotor late slope"),
                    ace.linear_contrast(fit, {interaction: 1}, "hypomotor minus hypermotor late slope"),
                )
                rows.extend({"template_label": label, "outcome": outcome, **value} for value in contrasts)
            except (KeyError, ValueError, np.linalg.LinAlgError) as exc:
                rows.append({"template_label": label, "outcome": outcome, "contrast": "model_failed", "error": str(exc)})
    contrasts = pd.DataFrame(rows)
    contrasts["p_bh"] = ace.bh(contrasts.get("p", pd.Series(index=contrasts.index, dtype=float)))
    return participant, contrasts


def recovery_models(late: pd.DataFrame, postictal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    phases = pd.concat([late, postictal], ignore_index=True, sort=False)
    group = ["participant", "stratum", "cohort", "semiology", "template_label", "phase"]
    participant = phases.groupby(group, observed=True, dropna=False)[["log_kappa", "abs_log_kappa", "js_from_II"]].mean().reset_index()
    pair = ["participant", "stratum", "cohort", "semiology", "template_label"]
    complete = participant.groupby(pair, observed=True, dropna=False)["phase"].transform("nunique").eq(2)
    participant = participant[complete].copy()
    participant["postictal"] = participant["phase"].eq("postictal_full_record").astype(int)
    rows = []
    formula = "{outcome} ~ postictal + C(semiology, Treatment('hyper')) + C(cohort, Treatment('FS')) + postictal:C(semiology, Treatment('hyper'))"
    for label in ace.LABELS:
        frame = participant[participant["template_label"].eq(label) & participant["stratum"].isin(ace.FUNCTIONAL) & participant["semiology"].notna()]
        for outcome in ("abs_log_kappa", "js_from_II"):
            try:
                fit, _ = ace.fit_gee(frame, formula.format(outcome=outcome), f"state {label} recovery {outcome}")
                interaction = interaction_term(fit.params.index, "postictal:")
                contrasts = (
                    ace.linear_contrast(fit, {"postictal": 1}, "hypermotor postictal minus late-preictal"),
                    ace.linear_contrast(fit, {"postictal": 1, interaction: 1}, "hypomotor postictal minus late-preictal"),
                    ace.linear_contrast(fit, {interaction: 1}, "hypomotor minus hypermotor recovery"),
                )
                rows.extend({"template_label": label, "outcome": outcome, **value} for value in contrasts)
            except (KeyError, ValueError, np.linalg.LinAlgError) as exc:
                rows.append({"template_label": label, "outcome": outcome, "contrast": "model_failed", "error": str(exc)})
    contrasts = pd.DataFrame(rows)
    contrasts["p_bh"] = ace.bh(contrasts.get("p", pd.Series(index=contrasts.index, dtype=float)))
    descriptive = participant.groupby(["template_label", "semiology", "phase"], observed=True).agg(
        participants=("participant", "nunique"), log_kappa_mean=("log_kappa", "mean"),
        abs_log_kappa_mean=("abs_log_kappa", "mean"), js_from_II_mean=("js_from_II", "mean")).reset_index()
    return participant, contrasts, descriptive


def self_check() -> None:
    labels = ("F", "E", "G", "B", "A", "C")
    frame = pd.DataFrame({"state_from_index": [5, 5], "state_to_index": [6, 2], "transition_count": [7, 3]})
    counts = transition_matrix(frame)
    assert counts[4, 5] == 7 and counts[4, 1] == 3 and counts.sum() == 10
    b0 = np.full((6, 6), 0.2); np.fill_diagonal(b0, 0)
    metrics = state_metrics(counts, b0, np.full(6, 20), labels, 5)
    assert len(metrics) == 6 and metrics[4]["outgoing_transitions"] == 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--minimum-transitions", type=int, default=10)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(); print("self-check passed"); return
    if not all((args.results_dir, args.manifest, args.out_dir)):
        parser.error("--results-dir, --manifest and --out-dir are required")
    labels = load_global_labels(args.results_dir)
    ace.LABELS = labels
    manifest = attach_strata(pd.read_csv(args.manifest))
    if not manifest["file_key"].is_unique:
        raise ValueError("Manifest file paths are not unique")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    baselines = interictal_baselines(args.results_dir / "participant_condition_transition_metrics.csv")
    windows, late = preictal_metrics(args.results_dir, manifest, labels, baselines, args.minimum_transitions)
    postictal = postictal_metrics(args.results_dir, manifest, labels, baselines, args.minimum_transitions)
    preictal_participant, preictal_contrasts = preictal_models(windows)
    recovery_participant, recovery_contrasts, recovery_descriptive = recovery_models(late, postictal)
    outputs = {
        "statewise_preictal_transition_windows.csv": windows,
        "statewise_preictal_transition_participant.csv": preictal_participant,
        "statewise_preictal_transition_gee_contrasts.csv": preictal_contrasts,
        "statewise_recovery_event_phases.csv": pd.concat([late, postictal], ignore_index=True, sort=False),
        "statewise_recovery_participant.csv": recovery_participant,
        "statewise_recovery_gee_contrasts.csv": recovery_contrasts,
        "statewise_recovery_descriptives.csv": recovery_descriptive,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.out_dir / name, index=False)
    summary = {"labels_by_state": dict(enumerate(labels, 1)), "II_baseline_participants": len(baselines),
               "minimum_outgoing_transitions_per_state_and_phase": args.minimum_transitions,
               "preictal_events": int(windows["file_path"].nunique()),
               "postictal_records": int(postictal["file_path"].nunique()),
               "postictal_estimand": "whole-record postictal operator versus final-five-minute preictal operator",
               "microstate_refitting_performed": False}
    (args.out_dir / "statewise_transition_recovery_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
