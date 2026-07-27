#!/usr/bin/env python3
"""Run the omnibus post-processing suite for simulated microstate experiments."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "_analysis"
SPM_CRITERION = "free_energy_elbow"
REMOVED_CRITERIA = {"elbow_sil_combined", "calinski_harabasz_score", "free_energy_covariance"}
REMOVED_K_CANDIDATE_COLUMNS = {
    "calinski_harabasz",
    "CalinskiHarabasz",
    "calinski_harabasz_norm",
    "score_calinski_harabasz",
    "selected_by_calinski_harabasz",
    "score_elbow_sil_combined",
    "selected_by_elbow_sil_combined",
    "score_free_energy_covariance",
    "selected_by_free_energy_covariance",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all simulation post-processing reports.")
    parser.add_argument("results_dir", nargs="?", type=Path, default=ROOT / "outputs" / "simulations" / "results")
    parser.add_argument("--comparison-csv", type=Path, default=None)
    parser.add_argument("--k-candidate-csv", type=Path, default=None)
    parser.add_argument("--config-file", type=Path, default=ROOT / "config" / "microstate_config.json")
    parser.add_argument("--template-file", type=Path, default=ROOT / "MetaMaps_2023_06.set")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--benchmark-n-boot", type=int, default=2000)
    parser.add_argument("--benchmark-n-perm", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-mixed-models", action="store_true")
    parser.add_argument("--skip-confusion-outputs", action="store_true")
    parser.add_argument("--matlab-topoplot", action="store_true", help="Render the optional MATLAB/EEGLAB recovery topoplot.")
    parser.add_argument("--skip-matlab-topoplot", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def run_step(label: str, command: list[str]) -> None:
    print(f"[simulation omnibus] {label}")
    subprocess.run(command, cwd=ROOT, check=True)


def script(name: str) -> str:
    return str(ANALYSIS_DIR / name)


def matlab_string(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def criterion_key(value: object) -> str:
    return "_".join(str(value).strip().lower().replace("-", "_").replace(" ", "_").split("_"))


def filter_comparison_csv(source: Path, target: Path) -> None:
    with source.open("r", encoding="utf-8", newline="") as src, target.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            return
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(row for row in reader if criterion_key(row.get("criterion", "")) not in REMOVED_CRITERIA)


def filter_k_candidate_csv(source: Path, target: Path) -> None:
    with source.open("r", encoding="utf-8", newline="") as src, target.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            return
        fields = [field for field in reader.fieldnames if field not in REMOVED_K_CANDIDATE_COLUMNS]
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in reader)


def filtered_input_csvs(comparison_csv: Path, k_candidate_csv: Path, output_dir: Path) -> tuple[Path, Path]:
    filtered_dir = output_dir / "filtered_inputs"
    filtered_dir.mkdir(parents=True, exist_ok=True)
    filtered_comparison = filtered_dir / comparison_csv.name
    filtered_k_candidate = filtered_dir / k_candidate_csv.name
    filter_comparison_csv(comparison_csv, filtered_comparison)
    filter_k_candidate_csv(k_candidate_csv, filtered_k_candidate)
    return filtered_comparison, filtered_k_candidate


def run_matlab_recovery_topoplot(results_dir: Path, comparison_csv: Path, template_file: Path, output_dir: Path, seed: int) -> None:
    output_file = output_dir / "comparison" / "simulated_microstate_method_comparison_topoplot.png"
    batch = (
        "setup_microstate_paths; "
        "plot_simulated_microstate_method_comparison("
        f"'output_dir',{matlab_string(results_dir.parent)},"
        f"'results_csv',{matlab_string(comparison_csv)},"
        f"'template_file',{matlab_string(template_file)},"
        f"'spm_criterion',{matlab_string(SPM_CRITERION)},"
        "'K_true',7,"
        f"'seed',{int(seed)},"
        f"'output_file',{matlab_string(output_file)}"
        ");"
    )
    run_step("MATLAB topoplot recovery figure", ["matlab", "-batch", batch])


def main() -> None:
    args = parse_args()
    if args.self_test:
        for name in (
            "_analyse_comparison_results.py",
            "_simulation_methods_report.py",
            "_simulation_descriptive_results.py",
            "_simulation_benchmark_results.py",
            "_simulation_confusion_matrices_report.py",
            "_simulation_pipeline_figure.py",
        ):
            assert (ANALYSIS_DIR / name).is_file(), name
        assert (ANALYSIS_DIR / "matlab" / "plot_simulated_microstate_method_comparison.m").is_file()
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            comparison = tmp / "comparison_results.csv"
            k_candidates = tmp / "k_candidate_metrics.csv"
            comparison.write_text(
                "method,criterion,value\n"
                "spm_vb,free_energy_elbow,1\n"
                "spm_vb,elbow_sil_combined,2\n"
                "spm_vb,calinski_harabasz_score,3\n"
                "spm_vb,free_energy_covariance,4\n",
                encoding="utf-8",
            )
            k_candidates.write_text(
                "score_free_energy_elbow,score_elbow_sil_combined,score_calinski_harabasz,score_free_energy_covariance,selected_by_free_energy_covariance,calinski_harabasz\n"
                "1,2,3,4,5,6\n",
                encoding="utf-8",
            )
            filtered_comparison, filtered_k_candidates = filtered_input_csvs(comparison, k_candidates, tmp / "out")
            assert "free_energy_elbow" in filtered_comparison.read_text(encoding="utf-8")
            assert "elbow_sil_combined" not in filtered_comparison.read_text(encoding="utf-8")
            assert "free_energy_covariance" not in filtered_comparison.read_text(encoding="utf-8")
            header = filtered_k_candidates.read_text(encoding="utf-8").splitlines()[0]
            assert "score_free_energy_elbow" in header
            assert "score_elbow_sil_combined" not in header
            assert "score_calinski_harabasz" not in header
            assert "score_free_energy_covariance" not in header
        print("self-test passed")
        return

    results_dir = args.results_dir
    comparison_csv = args.comparison_csv or results_dir / "comparison_results.csv"
    k_candidate_csv = args.k_candidate_csv or results_dir / "k_candidate_metrics.csv"
    output_dir = args.output_dir or results_dir.parent / "omnibus_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv, k_candidate_csv = filtered_input_csvs(comparison_csv, k_candidate_csv, output_dir)
    run_step(
        "simulation workflow figure",
        [
            sys.executable,
            script("_simulation_pipeline_figure.py"),
            "--comparison-csv",
            str(comparison_csv),
            "--template-file",
            str(args.template_file),
            "--output-dir",
            str(results_dir.parent / "plots"),
            "--seed",
            str(args.seed),
        ],
    )

    run_step(
        "methods summary",
        [
            sys.executable,
            script("_simulation_methods_report.py"),
            "--config-file",
            str(args.config_file),
            "--comparison-csv",
            str(comparison_csv),
            "--k-candidate-csv",
            str(k_candidate_csv),
            "--output-html",
            str(output_dir / "simulation_methods_summary.html"),
        ],
    )
    comparison_cmd = [
        sys.executable,
        script("_analyse_comparison_results.py"),
        str(results_dir),
        "--results-csv",
        str(comparison_csv),
        "--output-dir",
        str(output_dir / "comparison"),
        "--template-file",
        str(args.template_file),
        "--n-boot",
        str(args.n_boot),
        "--seed",
        str(args.seed),
        "--skip-microstate-map-plot",
        "--skip-confusion-pair-plots",
    ]
    if args.skip_mixed_models:
        comparison_cmd.append("--skip-mixed-models")
    if args.skip_confusion_outputs:
        comparison_cmd.append("--skip-confusion-outputs")
    run_step("bootstrap/statistical comparison", comparison_cmd)
    if args.matlab_topoplot and not args.skip_matlab_topoplot:
        run_matlab_recovery_topoplot(results_dir, comparison_csv, args.template_file, output_dir, args.seed)
    run_step(
        "descriptive plots",
        [
            sys.executable,
            script("_simulation_descriptive_results.py"),
            "--comparison-csv",
            str(comparison_csv),
            "--k-candidate-csv",
            str(k_candidate_csv),
            "--output-html",
            str(output_dir / "simulation_descriptive_results.html"),
        ],
    )
    run_step(
        "paired benchmark",
        [
            sys.executable,
            script("_simulation_benchmark_results.py"),
            "--comparison-csv",
            str(comparison_csv),
            "--output-html",
            str(output_dir / "simulation_benchmark_results.html"),
            "--spm-criterion",
            SPM_CRITERION,
            "--n-boot",
            str(args.benchmark_n_boot),
            "--n-perm",
            str(args.benchmark_n_perm),
            "--seed",
            str(args.seed),
        ],
    )
    run_step(
        "K-selection confusion matrices",
        [
            sys.executable,
            script("_simulation_confusion_matrices_report.py"),
            "--comparison-csv",
            str(comparison_csv),
            "--output-html",
            str(output_dir / "simulation_confusion_matrices.html"),
        ],
    )

    manifest = output_dir / "README.txt"
    manifest.write_text(
        "Simulation omnibus analysis\n"
        f"comparison_csv: {comparison_csv}\n"
        f"k_candidate_csv: {k_candidate_csv}\n"
        f"VB-GMM criterion: {SPM_CRITERION}\n"
        "Excluded criteria: elbow_sil_combined, calinski_harabasz_score, free_energy_covariance.\n"
        "Display labels use VB-GMM for spm_vb and K-Means for kmeans_koenig.\n"
        "The optional microstate recovery topography figure is skipped by default; pass --matlab-topoplot to render it.\n",
        encoding="utf-8",
    )
    print(f"Simulation omnibus outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
