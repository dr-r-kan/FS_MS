"""Collect deidentified, paper-useful outputs from a completed hierarchy run."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def copy_if_present(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def anonymise_hierarchy(source: Path, destination: Path) -> None:
    frame = pd.read_csv(source)
    participants = sorted(p for p in frame.get("participant", pd.Series(dtype=str)).dropna().astype(str).unique() if p)
    subject_map = {participant: f"S{i:03d}" for i, participant in enumerate(participants, 1)}
    if "participant" in frame:
        frame["participant"] = frame["participant"].map(subject_map).fillna("")
    if "level" in frame and "name" in frame:
        participant_rows = frame["level"].astype(str).eq("participant")
        frame.loc[participant_rows, "name"] = frame.loc[participant_rows, "name"].map(subject_map)
    frame = frame.drop(columns=["fit_path"], errors="ignore")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    root, package = args.results_dir, args.package
    data, figures, provenance = package / "data", package / "figures", package / "provenance"
    data.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    provenance.mkdir(parents=True, exist_ok=True)

    anonymise_hierarchy(root / "hierarchical_fit_summary.csv", data / "hierarchical_fit_summary.csv")
    covariance = root / "hierarchical_state_covariance_metrics.csv"
    if covariance.exists():
        anonymise_hierarchy(covariance, data / "hierarchical_state_covariance_metrics.csv")

    copies = {
        "omnibus_analysis/tanova/tanova_results.csv": "data/tanova_results.csv",
        "omnibus_analysis/tanova/bayesian_tanova_results.csv": "data/bayesian_tanova_results.csv",
        "omnibus_analysis/tanova/tanova_methods_notes.txt": "provenance/tanova_methods_notes.txt",
        "omnibus_analysis/bhattacharyya_tanova/bhattacharyya_tanova_results.csv": "data/bhattacharyya_tanova_results.csv",
        "omnibus_analysis/bhattacharyya_tanova/bhattacharyya_tanova_methods.txt": "provenance/bhattacharyya_tanova_methods.txt",
        "omnibus_analysis/global_topoplots/global_microstate_templates.csv": "data/global_microstate_templates.csv",
        "omnibus_analysis/global_topoplots/global_microstate_templates.png": "figures/global_microstate_templates.png",
        "omnibus_analysis/tanova/plots/tanova_pmax_heatmap.png": "figures/tanova_pmax_heatmap.png",
        "participant_ce_transition_analysis/group_condition_edge_model_terms.csv": "data/transition_group_condition_edge_model_terms.csv",
        "participant_ce_transition_analysis/group_condition_edge_model_coefficients.csv": "data/transition_group_condition_edge_model_coefficients.csv",
        "participant_ce_transition_analysis/group_condition_matrix_summary.csv": "data/transition_group_condition_matrix_summary.csv",
        "ce_group_condition_analysis/ce_group_condition_summary.csv": "data/ce_group_condition_summary.csv",
        "ce_group_condition_analysis/ce_group_condition_gee_terms.csv": "data/ce_group_condition_gee_terms.csv",
        "ce_group_condition_analysis/ce_group_condition_gee_coefficients.csv": "data/ce_group_condition_gee_coefficients.csv",
        "ce_group_condition_analysis/ce_group_condition.png": "figures/ce_group_condition.png",
        "ce_group_condition_analysis/ce_preictal_gradient_summary.csv": "data/ce_preictal_gradient_summary.csv",
        "ce_group_condition_analysis/ce_preictal_gradient_gee_terms.csv": "data/ce_preictal_gradient_gee_terms.csv",
        "ce_group_condition_analysis/ce_preictal_gradient.png": "figures/ce_preictal_gradient.png",
        "participant_ce_cardiac_phase_prediction/participant_ce_prediction_models.csv": "data/participant_ce_cardiac_prediction_models.csv",
        "participant_ce_cardiac_phase_prediction/participant_ce_prediction_forest.png": "figures/participant_ce_cardiac_prediction_forest.png",
    }
    for source, destination in copies.items():
        copy_if_present(root / source, package / destination)

    dictionary_path = data / "data_dictionary.csv"
    existing = pd.read_csv(dictionary_path) if dictionary_path.exists() else pd.DataFrame(columns=["file", "description"])
    additions = pd.DataFrame([
        ("cohort_audit.csv", "Aggregate record and participant counts by diagnostic group, condition and seizure type; contains no direct identifiers."),
        ("hierarchical_fit_summary.csv", "Deidentified global/group/condition/participant fit counts, K and template-alignment quality."),
        ("hierarchical_state_covariance_metrics.csv", "Deidentified topology-aware covariance and canonical-template fit metrics by hierarchy node and state."),
        ("tanova_results.csv", "Frequentist permutation TANOVA with within-effect max-state correction."),
        ("bayesian_tanova_results.csv", "Bayesian-bootstrap TANOVA effect uncertainty and ROPE probabilities."),
        ("bhattacharyya_tanova_results.csv", "Channel/state Bhattacharyya distances, permutation nulls and Bayesian-FDR indicators."),
        ("global_microstate_templates.csv", "Global template maps and canonical labels."),
        ("transition_group_condition_edge_model_terms.csv", "Canonical C/E transition-model omnibus terms with multiplicity correction."),
        ("transition_group_condition_edge_model_coefficients.csv", "Canonical C/E transition-model coefficients."),
        ("transition_group_condition_matrix_summary.csv", "Group/condition C/E transition summaries."),
        ("ce_group_condition_summary.csv", "Participant-template C/E descriptive summaries by group and condition."),
        ("ce_group_condition_gee_terms.csv", "Participant-template C/E group-by-condition GEE terms."),
        ("ce_group_condition_gee_coefficients.csv", "Participant-template C/E group-by-condition GEE coefficients."),
        ("ce_preictal_gradient_summary.csv", "Legacy within-preictal C/E gradient sensitivity summary."),
        ("ce_preictal_gradient_gee_terms.csv", "Legacy within-preictal C/E gradient sensitivity terms."),
        ("participant_ce_cardiac_prediction_models.csv", "Aggregate participant C/E cardiac-phase prediction model estimates."),
    ], columns=["file", "description"])
    combined = pd.concat([existing, additions], ignore_index=True).drop_duplicates("file", keep="first")
    combined = combined[combined["file"].map(lambda name: (data / str(name)).exists())]
    combined.sort_values("file").to_csv(dictionary_path, index=False)


if __name__ == "__main__":
    main()
