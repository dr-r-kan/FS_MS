"""Assemble the versioned A/C/E paper package and write its SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from microstate_labels import load_global_labels


SPM_MS_COMMIT = "0dbde1146f6777412dce19edfa4962cd45157fca"
SPM_COMMIT = "04db1be762077630772ee660a470a0fd63d4ff43"


def copy_file(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree(source: Path, target: Path, patterns: tuple[str, ...] | None = None) -> None:
    if not source.exists():
        return
    for path in source.rglob("*"):
        if not path.is_file() or (patterns and not any(path.match(pattern) for pattern in patterns)):
            continue
        copy_file(path, target / path.relative_to(source))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(root: Path) -> None:
    manifest = root / "SHA256_MANIFEST.txt"
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p != manifest):
        rows.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}  {path.stat().st_size}")
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")


def assemble(repository: Path, hierarchy: Path, analysis: Path, morphology: Path,
             ecg: Path, manifest_csv: Path, output: Path) -> None:
    label_order = load_global_labels(hierarchy)
    output.mkdir(parents=True, exist_ok=True)
    for subdir in ("figures", "useful_data/activity_cardioneural", "useful_data/morphology",
                   "useful_data/hierarchy", "useful_data/ecg_qc", "frozen_code", "provenance"):
        (output / subdir).mkdir(parents=True, exist_ok=True)

    copy_tree(analysis / "data", output / "useful_data" / "activity_cardioneural")
    copy_tree(morphology / "data", output / "useful_data" / "morphology")
    copy_tree(analysis / "figures", output / "figures")
    copy_tree(morphology / "figures", output / "figures")

    hierarchy_files = (
        "hierarchical_fit_summary.csv", "hierarchical_state_covariance_metrics.csv",
        "common_channels.csv", "common_channel_positions.csv", "normalised_input_manifest.csv",
        "participant_condition_state_backfit_metrics.csv", "participant_condition_state_pairwise_backfit_metrics.csv",
        "participant_condition_record_backfit_summary.csv",
        "participant_condition_preictal_time_metrics.csv", "participant_condition_transition_metrics.csv",
        "global_state_backfit_metrics.csv", "global_state_pairwise_backfit_metrics.csv",
        "global_record_backfit_summary.csv", "global_preictal_time_metrics.csv", "global_transition_metrics.csv",
        "participant_ace_cardiac_phase_metrics.csv", "global_ace_cardiac_phase_metrics.csv",
        "preictal_window_refit_maps.csv", "preictal_window_refit_maps.audit.json",
        "hierarchy_validation.json", "cache_integrity_sha256_audit.txt", "matlab_run.log",
    )
    for name in hierarchy_files:
        copy_file(hierarchy / name, output / "useful_data" / "hierarchy" / name)
    copy_file(hierarchy / "global" / "global_centers.csv",
              output / "useful_data" / "hierarchy" / "global_K6_manual_centers.csv")
    copy_file(hierarchy / "global" / "global_centers.png",
              output / "figures" / "global_K6_manual_labels.png")
    copy_file(hierarchy / "global" / "global_manual_labels.csv",
              output / "useful_data" / "hierarchy" / "global_manual_labels.csv")
    copy_file(hierarchy / "global" / "global_manual_labels.mat",
              output / "useful_data" / "hierarchy" / "global_manual_labels.mat")
    for log_file in hierarchy.parent.glob(f"{hierarchy.name}_matlab*.log"):
        copy_file(log_file, output / "provenance" / "runtime_logs" / log_file.name)
    copy_tree(hierarchy / "omnibus_analysis", output / "figures" / "hierarchy", ("*.png", "*.csv", "*.json"))
    copy_file(manifest_csv, output / "useful_data" / "hierarchy" / manifest_csv.name)

    copy_tree(ecg, output / "useful_data" / "ecg_qc", ("*_rpeaks_corrected.csv", "*report*.csv", "*.json"))
    copy_tree(ecg / "recovery_qc_plots", output / "figures" / "ecg_qc", ("*.png",))

    code_files = (
        "run_vg_microstates.m", "analyse_preictal_publication.py", "analyse_preictal_morphology_rft.py",
        "analyse_ace_cardioneural.py", "analyse_statewise_transition_recovery.py", "fast_ecg_review.py", "build_ace_report.py",
        "combine_window_refits.py", "validate_hierarchy.py", "assemble_ace_package.py", "microstate_labels.py",
        "analysis_parameters.json",
    )
    for name in code_files:
        copy_file(repository / name, output / "frozen_code" / name)
    vg = repository / "external" / "VG-microstates"
    for relative in (
        "fit_microstate_hierarchical_dataset.m", "_internal/matlab/microstate_utilities.m",
        "_internal/matlab/spm_mix_prior.m", "_internal/matlab/test_spm_mix_prior.m",
        "_internal/matlab/test_fs_spm_prior_hierarchy.m", "config/microstate_config.json",
    ):
        copy_file(vg / relative, output / "frozen_code" / "VG-microstates" / relative)

    parameters = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manual_labels_by_state": dict(enumerate(label_order, 1)),
        "K": 6,
        "spm_prior_mode": "spm_mix_prior_component_means",
        "spm_prior_weight": 1,
        "spm_ms_commit": SPM_MS_COMMIT,
        "spm_version": "SPM25",
        "spm_commit": SPM_COMMIT,
        "window_refit": {"length_s": 60, "step_s": 30, "time_window_s": 1200,
                         "minimum_real_peaks": 30, "minimum_state_peaks": 5,
                         "record_duration_source": "source MAT sample count divided by sfreq"},
        "morphology": {"states": ["A", "C", "E"], "event_minimum_windows": 12,
                       "event_minimum_span_minutes": 5, "tfce_E": 0.5, "tfce_H": 2,
                       "participant_sign_flips": 5000,
                       "joint_fields": "electrodes by A/C/E by FS-functional/DD-functional"},
        "activity": {"primary_balance": "C/A", "gate_state": "E", "piecewise_knot_min": -5,
                     "outlier_rule": "one-pass event median +/- 3 sample SD; coverage and GFP independently"},
        "cardiac": {"window_s": 60, "rr_coverage_min": 0.80, "maximum_gap_s": 5,
                    "maximum_gap_recovered_interval_fraction": 0.10, "instantaneous_hr_bpm": [35, 140],
                    "baseline_minutes": [-20, -5], "baseline_summary": "event mean across valid 60-second HR windows",
                    "hypermotor_tost_bpm_per_min": [-1, 1]},
        "primary_multiplicity": "Holm across C/A late slope, hypomotor HR slope and pooled B-like precision trend",
        "secondary_multiplicity": "BH for scalar secondary tests; maximum-T/TFCE FWER for morphology",
        "hierarchy_dir": str(hierarchy), "analysis_dir": str(analysis),
        "morphology_dir": str(morphology), "ecg_dir": str(ecg), "manifest": str(manifest_csv),
    }
    (output / "provenance" / "locked_analysis_parameters.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8")
    provenance = [
        "CURRENT SPM_MS HIERARCHICAL PRIOR PROVENANCE",
        f"SPM_MS commit: {SPM_MS_COMMIT}", f"SPM25 commit: {SPM_COMMIT}",
        "Parent maps and child GFP maps share the child-derived PCA/projective transform.",
        "Child call: spm_mix_prior(features,K,0,struct('means',prior_features,'mean_weight',1)).",
        "No parent map is inserted into observations. beta_0=1 for all six component means.",
        f"Manual global state mapping: {dict(enumerate(label_order, 1))}.",
        "Global fit uses standard spm_mix. Hierarchy and window caches were regenerated for this analysis.",
    ]
    (output / "provenance" / "SPM_MS_SPM_PROVENANCE.txt").write_text("\n".join(provenance) + "\n", encoding="utf-8")
    write_manifest(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--hierarchy", type=Path)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--morphology", type=Path)
    parser.add_argument("--ecg", type=Path)
    parser.add_argument("--manifest-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-only", action="store_true")
    args = parser.parse_args()
    if args.hash_only:
        write_manifest(args.output); return
    if not all((args.repository, args.hierarchy, args.analysis, args.morphology, args.ecg, args.manifest_csv)):
        parser.error("all source path arguments are required unless --hash-only is used")
    assemble(args.repository, args.hierarchy, args.analysis, args.morphology, args.ecg, args.manifest_csv, args.output)


if __name__ == "__main__":
    main()
