"""Run the action-conditioned transition and directionality analyses end-to-end.

Example
-------
python code/run_mechanistic_extension.py --out-root rerun --bootstrap-draws 2000

The input files default to ``../input`` relative to this script.  Primary action
analysis uses soft four-quadrant interpolation in CPI/CSI action space, exact
same-window autonomic summaries, participant-nested II validation, and a
participant-cluster bootstrap.  ``--run-lagged`` additionally evaluates the
preceding contiguous action window without rerunning pseudo-onset matching.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from analyse_action_conditioned_transition_tensor import (
    ActionTensorConfig, markdown_table, run_analysis as run_action, self_check as action_self_check,
)
from analyse_transition_to_autonomic_action import (
    run as run_directionality, self_check as directionality_self_check,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_optional_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def result_summary(root: Path) -> None:
    validation = pd.read_csv(root / "action_tensor/tables/action_baseline_validation_summary.csv").iloc[0]
    trajectory = pd.read_csv(root / "action_tensor/tables/action_trajectory_summary.csv")
    contrasts = pd.read_csv(root / "action_tensor/tables/action_contrasts.csv")
    final = contrasts[contrasts.time_bin.eq("-1 to 0") & contrasts.contrast.eq("FS minus ES")]
    routes = pd.read_csv(root / "action_tensor/tables/action_route_contrasts.csv")
    final_routes = (routes[routes.time_bin.eq("-1 to 0")]
                    .sort_values(["bh_fdr", "bootstrap_two_sided_p"]).head(10))
    lagged_path = root / "lagged_sensitivity/tables/action_contrasts.csv"
    lagged = read_optional_csv(lagged_path)
    if not lagged.empty:
        lagged = lagged[lagged.time_bin.eq("-1 to 0") & lagged.contrast.eq("FS minus ES")]
    did = read_optional_csv(root / "action_tensor/tables/action_matched_pseudo_did.csv")
    if not did.empty:
        did = did[did.estimator.eq("covariate-adjusted participant-cluster DiD")]
    caliper = read_optional_csv(root / "action_tensor/tables/action_caliper_matched_pseudo_did.csv")
    if not caliper.empty:
        caliper = caliper[caliper.estimator.eq("covariate-adjusted participant-cluster DiD")]
    direction = pd.read_csv(root / "directionality/tables/transition_to_next_action_contrasts.csv")
    direction_available = direction.dropna(subset=["ci95_low", "ci95_high"])
    direction_supported = bool(((direction_available.ci95_low > 0) |
                                (direction_available.ci95_high < 0)).any())
    group_rows = []
    for outcome, transform in (("log_action_js", np.exp), ("action_log_kappa", np.exp),
                               ("action_lambda", lambda value: value),
                               ("action_log_score_gain", lambda value: value)):
        for group in ("FS", "ES"):
            row = trajectory[(trajectory.outcome.eq(outcome)) & trajectory.group.eq(group) &
                             trajectory.time_bin.eq("-1 to 0")].iloc[0]
            group_rows.append({"outcome": outcome, "group": group,
                               "estimate": transform(row.estimate),
                               "ci95_low": transform(row.ci95_low),
                               "ci95_high": transform(row.ci95_high)})
    lines = [
        "# Action-conditioned mechanistic extension", "",
        "## Scope", "",
        "The available observations are 60-second destination-labelled transition-count matrices and "
        "60-second CSI/CPI summaries. The primary analysis therefore uses soft interpolation in CPI/CSI "
        "action space: four non-negative weights sum to one, so action resolution improves without "
        "creating extra observations. This is an observational window-level model, not an instantaneous "
        "or causal transition-by-transition analysis.", "",
        "A genuinely instantaneous extension requires:", "",
        "- the timestamp of every microstate transition;",
        "- continuous or heartbeat-resolved CSI/CPI;",
        "- CSI/CPI interpolation only at genuine transition timestamps;",
        "- a continuous-time or semi-Markov model retaining dwell times.", "",
        "The finest valid export is nevertheless destination-labelled: every eligible 60-second window is "
        "expanded to its 30 off-diagonal source-to-destination routes with observed counts, model probabilities, "
        "expected counts, residuals, and route/window per-transition log-score increments. Counted jumps are not "
        "treated as separately timestamped observations.", "",
        "## Interictal validation", "",
        f"Action conditioning improved participant-held-out interictal log score by "
        f"**{validation.estimate:.6f} per transition** "
        f"(95% interval {validation.ci95_low:.6f} to {validation.ci95_high:.6f}; "
        f"{int(validation.n_participants)} participants).", "",
        "## Final-minute action-conditioned estimates", "",
        markdown_table(pd.DataFrame(group_rows)), "",
        "## Final-minute FS−ES contrasts", "",
        markdown_table(final[["outcome", "estimate", "ci95_low", "ci95_high", "probability_gt_zero",
                              "n_participants"]]), "",
        "## Exploratory final-minute route contrasts", "",
        markdown_table(final_routes[["source", "destination", "estimate", "ci95_low", "ci95_high",
                                     "bootstrap_two_sided_p", "bh_fdr", "n_participants"]]), "",
        "## Matched pseudo-onset sensitivity", "",
        markdown_table(did[["outcome", "estimate", "ci95_low", "ci95_high", "probability_gt_zero", "n_pairs",
                            "n_participants"]]) if not did.empty else
        "No matched pseudo-onset estimates were available.", "",
        "Strict-caliper estimates are exploratory because matching retains a selected smaller subset:", "",
        markdown_table(caliper[["outcome", "estimate", "ci95_low", "ci95_high", "probability_gt_zero", "n_pairs",
                                "n_participants"]]) if not caliper.empty else
        "No strict-caliper estimate was available.", "",
        "## Temporal sensitivities", "",
        "The preceding-contiguous-window specification gives:", "",
        (markdown_table(lagged[["outcome", "estimate", "ci95_low", "ci95_high", "probability_gt_zero",
                                "n_participants"]]) if not lagged.empty else
         "Not run; pass `--run-lagged` to add this sensitivity."), "",
        ("At least one reverse transition-to-next-action interval excluded zero."
         if direction_supported else
         "No reverse transition-to-next-action FS−ES interval excluded zero."),
        "The final 60-second transition window has no subsequent within-record autonomic window and is "
        "therefore intentionally unavailable for that reverse model.", "",
        "## Interpretation", "",
        "The action tensor is empirically testable only as P(next distinct microstate | current microstate, "
        "60-second autonomic action proxy). Positive interictal validation establishes reproducible association, "
        "not causal action selection. Primary pseudo-onset, strict-caliper, and reverse-direction estimates "
        "remain sensitivity analyses.", "",
        "![Action-conditioned transition tensor](action_tensor/figures/figure_07_action_conditioned_transition_tensor.png)", "",
        "![Validation and pseudo-onset sensitivity](action_tensor/figures/figure_08_action_tensor_validation_and_pseudo.png)", "",
        "![Destination-labelled route deviations](action_tensor/figures/figure_10_route_level_action_misfit.png)", "",
        "![Reverse temporal prediction](directionality/figures/figure_09_transition_to_next_action_directionality.png)", "",
    ]
    (root / "MECHANISTIC_EXTENSION_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def append_decisions(root: Path) -> None:
    path = root / "MECHANISTIC_ANALYSIS_DECISIONS.csv"
    decisions = pd.DataFrame([
        ("autonomic action representation", "add", "Four CPI/CSI corners with soft action-space interpolation"),
        ("action-conditioned baseline", "add", "Participant II next-distinct-state tensor with nested validation"),
        ("transition precision", "add", "Fit positive kappa directly in softmax(kappa log p)"),
        ("action specificity", "add", "Lambda isolates action-dependent deviation from the action-pooled grammar"),
        ("route-level resolution", "add", "Window x source x destination export; no invented transition timestamps"),
        ("temporal direction", "add sensitivity", "Preceding-action and reverse next-action models; observational only"),
        ("instantaneous causal model", "defer", "Requires transition timestamps, continuous CSI/CPI and dwell-time modelling"),
    ], columns=("analysis", "decision", "reason"))
    decisions.to_csv(path, index=False)


def write_audit(root: Path, scripts: tuple[Path, ...], inputs: tuple[Path, ...], results: dict) -> None:
    files = [path for folder in ("action_tensor", "lagged_sensitivity", "directionality")
             for path in (root / folder).rglob("*") if path.is_file()]
    files += [root / name for name in ("RUN_SUMMARY.json", "MECHANISTIC_EXTENSION_RESULTS.md",
                                       "MODEL_SPECIFICATION.md", "MECHANISTIC_ANALYSIS_DECISIONS.csv")]
    audit = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input_resolution": "60-second transition counts and 60-second autonomic summaries",
        "instantaneous_analysis_performed": False,
        "finest_route_resolution": "60-second window x source x destination (30 off-diagonal routes)",
        "soft_interpolation": "four action-space weights sum to one; no temporal upsampling",
        "scripts": {str(path.resolve()): sha256(path) for path in scripts},
        "inputs": {str(path.resolve()): sha256(path) for path in inputs},
        "results": results,
        "outputs": {str(path.relative_to(root)): sha256(path) for path in files},
    }
    (root / "MECHANISTIC_EXTENSION_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def update_review(root: Path, review: Path, scripts: tuple[Path, ...]) -> None:
    review.mkdir(parents=True, exist_ok=True)
    for folder in ("action_tensor", "lagged_sensitivity", "directionality"):
        shutil.copytree(root / folder, review / folder, dirs_exist_ok=True)
    (review / "code").mkdir(exist_ok=True)
    for script in scripts:
        shutil.copy2(script, review / "code" / script.name)
    for name in ("RUN_SUMMARY.json", "MECHANISTIC_EXTENSION_RESULTS.md", "MECHANISTIC_EXTENSION_AUDIT.json",
                 "MODEL_SPECIFICATION.md", "MECHANISTIC_ANALYSIS_DECISIONS.csv"):
        shutil.copy2(root / name, review / name)
    (review / "MECHANISTIC_EXTENSION_README.md").write_text(
        "# Mechanistic extension review package\n\n"
        "Start with `MECHANISTIC_EXTENSION_RESULTS.md`; exact tables, figures, code, input files and "
        "provenance are included. The model is window-level and observational.\n", encoding="utf-8")
    files = [path for path in review.rglob("*") if path.is_file() and path.name != "SHA256SUMS.csv"]
    pd.DataFrame({"path": [str(path.relative_to(review)) for path in files],
                  "sha256": [sha256(path) for path in files],
                  "bytes": [path.stat().st_size for path in files]}).to_csv(review / "SHA256SUMS.csv", index=False)


def main() -> None:
    here = Path(__file__).resolve().parent
    package = here.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peak-windows-csv", type=Path,
                        default=package / "input/directional_peak_backfit_windows_all.csv.gz")
    parser.add_argument("--common-windows-csv", type=Path,
                        default=package / "input/autonomic_peak_microstate_common_windows.csv.gz")
    parser.add_argument("--out-root", type=Path, required=False, default=package / "rerun")
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=31072026)
    parser.add_argument("--run-lagged", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        action_self_check(); directionality_self_check()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            append_decisions(root); append_decisions(root)
            decisions = pd.read_csv(root / "MECHANISTIC_ANALYSIS_DECISIONS.csv")
            assert decisions.analysis.is_unique and "instantaneous causal model" in set(decisions.analysis)
        print("mechanistic wrapper self-check passed")
        return
    for path in (args.peak_windows_csv, args.common_windows_csv):
        if not path.exists():
            parser.error(f"Input does not exist: {path}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    primary = ActionTensorConfig(
        minimum_transitions=30,
        action_temperature=0.5,
        action_mode="soft",
        action_lag_windows=0,
        pool_prior_strength=100.0,
        action_prior_grid=(2.0, 5.0, 10.0, 20.0, 50.0),
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
        pseudo_record_fraction=0.99,
        pseudo_native_caliper=10.0,
        pseudo_action_caliper=1.0,
    )
    results = {
        "action_tensor": run_action(
            args.peak_windows_csv, args.common_windows_csv,
            args.out_root / "action_tensor", primary,
            make_plots=True, run_pseudo=True,
        )
    }
    if args.run_lagged:
        lagged = ActionTensorConfig(
            minimum_transitions=30,
            action_temperature=0.5,
            action_mode="soft",
            action_lag_windows=1,
            pool_prior_strength=100.0,
            action_prior_grid=(10.0,),
            bootstrap_draws=max(1000, args.bootstrap_draws // 2),
            seed=args.seed + 100,
            pseudo_record_fraction=0.99,
            pseudo_native_caliper=10.0,
            pseudo_action_caliper=1.0,
        )
        results["lagged_sensitivity"] = run_action(
            args.peak_windows_csv, args.common_windows_csv,
            args.out_root / "lagged_sensitivity", lagged,
            make_plots=False, run_pseudo=False,
        )
    results["directionality"] = run_directionality(
        args.peak_windows_csv, args.common_windows_csv,
        args.out_root / "directionality", 30, 0.0003,
        args.bootstrap_draws, args.seed + 200,
    )
    (args.out_root / "RUN_SUMMARY.json").write_text(json.dumps(results, indent=2))
    shutil.copy2(package / "MODEL_SPECIFICATION.md", args.out_root / "MODEL_SPECIFICATION.md")
    result_summary(args.out_root)
    append_decisions(args.out_root)
    scripts = (Path(__file__).resolve(),
               (here / "analyse_action_conditioned_transition_tensor.py").resolve(),
               (here / "analyse_transition_to_autonomic_action.py").resolve())
    write_audit(args.out_root, scripts, (args.peak_windows_csv, args.common_windows_csv), results)
    if args.review_dir is not None:
        update_review(args.out_root, args.review_dir, scripts)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
