#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "outputs" / "simulations" / "results"
CONFIG_FILE = ROOT / "config" / "microstate_config.json"
COMPARISON_CSV = RESULTS_DIR / "comparison_results.csv"
K_CANDIDATE_CSV = RESULTS_DIR / "k_candidate_metrics.csv"
OUTPUT_HTML = RESULTS_DIR / "simulation_methods_report.html"

SPM_LABEL = "VB-GMM"
KM_LABEL = "K-Means"


def main() -> None:
    args = parse_args()
    config = load_json(args.config_file)
    comparison = pd.read_csv(args.comparison_csv)
    candidates = pd.read_csv(args.k_candidate_csv) if args.k_candidate_csv.is_file() else pd.DataFrame()

    html = render_html(
        args=args,
        config=config,
        comparison=comparison,
        candidates=candidates,
    )
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html, encoding="utf-8")
    print(args.output_html)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a methods-section numerical summary for the simulation benchmark.")
    parser.add_argument("--config-file", type=Path, default=CONFIG_FILE)
    parser.add_argument("--comparison-csv", type=Path, default=COMPARISON_CSV)
    parser.add_argument("--k-candidate-csv", type=Path, default=K_CANDIDATE_CSV)
    parser.add_argument("--output-html", type=Path, default=OUTPUT_HTML)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("self-test passed")
        raise SystemExit(0)
    return args


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def render_html(*, args: argparse.Namespace, config: dict, comparison: pd.DataFrame, candidates: pd.DataFrame) -> str:
    comparison = add_k_error_columns(comparison)
    sim_cfg = config.get("simulation", {})
    cards = summary_cards(sim_cfg, comparison, candidates)
    design = design_table(sim_cfg, comparison)
    montages = montage_table(comparison)
    methods = method_table(comparison)
    criteria = criterion_table(comparison)
    outcomes = outcome_table(comparison)
    backfit = backfit_table(sim_cfg, comparison)
    data_files = data_file_table(args, comparison, candidates)
    prose = copy_ready_methods(sim_cfg, comparison, candidates)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Simulation methods numerical summary</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    h1 {{ margin-bottom: 4px; }}
    h2 {{ margin-top: 32px; }}
    .meta {{ color: #52616b; margin-top: 0; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 24px 0; }}
    .card {{ border: 1px solid #d9e2ec; padding: 14px; border-radius: 6px; }}
    .card b {{ display: block; font-size: 22px; margin-bottom: 4px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 14px 0 24px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 7px 8px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child, td:nth-child(2) {{ text-align: left; }}
    th {{ background: #f3f6f8; }}
    li {{ margin: 6px 0; }}
    .copy {{ border-left: 4px solid #155e75; padding: 10px 14px; background: #f7fafc; }}
  </style>
</head>
<body>
  <h1>Simulation methods numerical summary</h1>
  <p class="meta">Source files: {escape(str(args.config_file))}, {escape(str(args.comparison_csv))}, {escape(str(args.k_candidate_csv))}</p>
  <div class="cards">{cards}</div>
  <h2>Copy-Ready Methods Numbers</h2>
  <div class="copy">{prose}</div>
  <h2>Simulation Design</h2>
  {design}
  <h2>Montages</h2>
  {montages}
  <h2>Methods and Criteria</h2>
  {methods}
  {criteria}
  <h2>Outcomes</h2>
  {outcomes}
  <h2>Backfitting and Diagnostics</h2>
  {backfit}
  <h2>Data Files</h2>
  {data_files}
</body>
</html>
"""


def add_k_error_columns(comparison: pd.DataFrame) -> pd.DataFrame:
    comparison = comparison.copy()
    if {"K_estimated", "K_true"}.issubset(comparison.columns):
        k_delta = pd.to_numeric(comparison["K_estimated"], errors="coerce") - pd.to_numeric(
            comparison["K_true"], errors="coerce"
        )
        comparison["K_sq_error"] = k_delta.pow(2)
    elif "K_error" in comparison.columns:
        comparison["K_sq_error"] = pd.to_numeric(comparison["K_error"], errors="coerce").pow(2)
    return comparison


def summary_cards(sim_cfg: dict, comparison: pd.DataFrame, candidates: pd.DataFrame) -> str:
    blocks = generated_eeg_blocks(comparison)
    rows = [
        ("Generated EEG conditions", f"{blocks:,}"),
        ("Comparison rows", f"{len(comparison):,}"),
        ("K-candidate rows", f"{len(candidates):,}" if not candidates.empty else "not found"),
        ("Duration", f"{num(sim_cfg.get('duration_s'))} s"),
        ("Sampling rate", f"{num(sim_cfg.get('sfreq'))} Hz"),
        ("Samples per simulation", f"{int(sim_cfg.get('duration_s', 0) * sim_cfg.get('sfreq', 0)):,}"),
    ]
    return "".join(f"<div class=\"card\"><b>{escape(value)}</b>{escape(label)}</div>" for label, value in rows)


def design_table(sim_cfg: dict, comparison: pd.DataFrame) -> str:
    rows = [
        ("Replicates", values_from_config_or_data(sim_cfg, "reps", comparison, "rep")),
        ("True K values", values_from_config_or_data(sim_cfg, "K_true_vals", comparison, "K_true")),
        ("SNR levels (dB)", values_from_config_or_data(sim_cfg, "SNR_dbs", comparison, "SNR_dB")),
        ("Overlap probabilities", values_from_config_or_data(sim_cfg, "overlap_probs", comparison, "overlap_prob")),
        ("Overlap window", f"{value_range(sim_cfg.get('overlap_ms_range', []))} ms"),
        ("Overlap strength", num(sim_cfg.get("overlap_strength"))),
        ("K candidates", values_from_config_or_data(sim_cfg, "K_candidates", comparison, "K_estimated")),
        ("Duration", f"{num(sim_cfg.get('duration_s'))} s"),
        ("Sampling frequency", f"{num(sim_cfg.get('sfreq'))} Hz"),
        ("Samples before backfit downsampling", f"{int(sim_cfg.get('duration_s', 0) * sim_cfg.get('sfreq', 0)):,}"),
        ("Generated EEG conditions", f"{generated_eeg_blocks(comparison):,}"),
        ("Montage-specific paired units", f"{comparison[unit_columns()].drop_duplicates().shape[0]:,}"),
    ]
    return kv_table(rows)


def montage_table(comparison: pd.DataFrame) -> str:
    rows = []
    for (montage, n_leads), group in comparison.groupby(["montage_type", "n_leads"], dropna=False):
        rows.append(
            {
                "montage_type": montage,
                "n_leads": n_leads,
                "generated_eeg_blocks": group[block_columns()].drop_duplicates().shape[0],
                "comparison_rows": len(group),
                "method_criterion_pairs": group[["method", "criterion"]].drop_duplicates().shape[0],
            }
        )
    rows = pd.DataFrame(rows).sort_values(["n_leads", "montage_type"], ascending=[False, True])
    rows["montage"] = rows["montage_type"].astype(str) + " (" + rows["n_leads"].astype(int).astype(str) + " ch)"
    return rows[["montage", "generated_eeg_blocks", "comparison_rows", "method_criterion_pairs"]].to_html(index=False, escape=True)


def method_table(comparison: pd.DataFrame) -> str:
    rows = []
    for method, group in comparison.groupby("method", dropna=False):
        unique_fits = group[unit_columns() + ["method", "runtime_s"]].drop_duplicates()
        rows.append(
            {
                "method": method_label(method),
                "criteria": group["criterion"].nunique(),
                "unique_fits": len(unique_fits),
                "result_rows": len(group),
                "median_runtime_s": unique_fits["runtime_s"].median(),
                "mean_runtime_s": unique_fits["runtime_s"].mean(),
            }
        )
    rows = pd.DataFrame(rows)
    for col in ["median_runtime_s", "mean_runtime_s"]:
        rows[col] = rows[col].map(lambda x: f"{x:.2f}")
    return rows.to_html(index=False, escape=True)


def criterion_table(comparison: pd.DataFrame) -> str:
    comparison = comparison.copy()
    comparison["signed_K_error"] = pd.to_numeric(comparison["K_estimated"], errors="coerce") - pd.to_numeric(comparison["K_true"], errors="coerce")
    comparison["K_sq_error"] = comparison["signed_K_error"].pow(2)
    comparison["under_selected"] = comparison["signed_K_error"] < 0
    comparison["over_selected"] = comparison["signed_K_error"] > 0
    rows = (
        comparison.groupby(["method", "criterion"], dropna=False)
        .agg(
            rows=("fit_id", "size"),
            mean_selected_K=("K_estimated", "mean"),
            mean_squared_K_error=("K_sq_error", "mean"),
            recovered_state_F1=("f1_score", "mean"),
            sensitivity=("sensitivity", "mean"),
            precision=("precision", "mean"),
            exact_K_recovery=("K_correct", "mean"),
            under_selected=("under_selected", "mean"),
            over_selected=("over_selected", "mean"),
        )
        .reset_index()
        .sort_values(["method", "recovered_state_F1"], ascending=[True, False])
    )
    rows["method"] = rows["method"].map(method_label)
    rows["criterion"] = rows["criterion"].map(clean_label)
    rows["mean_selected_K"] = rows["mean_selected_K"].map(lambda x: f"{x:.2f}")
    rows["mean_squared_K_error"] = rows["mean_squared_K_error"].map(lambda x: f"{x:.3f}")
    for col in ["recovered_state_F1", "sensitivity", "precision", "exact_K_recovery", "under_selected", "over_selected"]:
        rows[col] = rows[col].map(lambda x: f"{100*x:.1f}%")
    return rows.to_html(index=False, escape=True)


def outcome_table(comparison: pd.DataFrame) -> str:
    outcome_cols = [
        "K_correct",
        "K_error",
        "K_sq_error",
        "mean_recovery_matched",
        "mean_recovery_padded",
        "sensitivity",
        "precision",
        "f1_score",
        "cluster_identity_accuracy",
        "backfit_mix_label_top1_accuracy",
        "backfit_mix_label_weight_mae",
        "runtime_s",
    ]
    rows = []
    for col in outcome_cols:
        if col not in comparison:
            continue
        values = pd.to_numeric(comparison[col], errors="coerce").dropna()
        rows.append(
            {
                "outcome": clean_label(col),
                "non_missing": f"{len(values):,}",
                "mean": f"{values.mean():.3f}",
                "median": f"{values.median():.3f}",
                "min": f"{values.min():.3f}",
                "max": f"{values.max():.3f}",
            }
        )
    return pd.DataFrame(rows).to_html(index=False, escape=True)


def backfit_table(sim_cfg: dict, comparison: pd.DataFrame) -> str:
    downsample = pd.to_numeric(comparison.get("backfit_downsample_factor", pd.Series(dtype=float)), errors="coerce")
    n_samples = pd.to_numeric(comparison.get("backfit_n_samples", pd.Series(dtype=float)), errors="coerce")
    n_original = pd.to_numeric(comparison.get("backfit_n_samples_original", pd.Series(dtype=float)), errors="coerce")
    rows = [
        ("Backfit diagnostics enabled", str(bool(sim_cfg.get("compute_backfit_diagnostics")))),
        ("Backfit detail files saved", str(bool(sim_cfg.get("save_backfit_details")))),
        ("Configured downsample factor", num(sim_cfg.get("backfit_downsample_factor"))),
        ("Observed downsample factors", join_values(downsample.dropna().unique())),
        ("Observed original samples", join_values(n_original.dropna().unique())),
        ("Observed backfit samples", join_values(n_samples.dropna().unique())),
        ("Rows with mixture backfit available", f"{int(pd.to_numeric(comparison.get('backfit_mix_available', 0), errors='coerce').fillna(0).sum()):,}"),
        ("Diagnostic MAT files", f"{comparison['backfit_diagnostic_file'].replace('', np.nan).dropna().nunique():,}" if "backfit_diagnostic_file" in comparison else "0"),
    ]
    return kv_table(rows)


def data_file_table(args: argparse.Namespace, comparison: pd.DataFrame, candidates: pd.DataFrame) -> str:
    rows = [
        ("Configuration", args.config_file, ""),
        ("Comparison results", args.comparison_csv, f"{len(comparison):,} rows x {comparison.shape[1]:,} columns"),
        ("K-candidate metrics", args.k_candidate_csv, f"{len(candidates):,} rows x {candidates.shape[1]:,} columns" if not candidates.empty else "not found"),
    ]
    return pd.DataFrame(rows, columns=["File", "Path", "Shape"]).to_html(index=False, escape=True)


def copy_ready_methods(sim_cfg: dict, comparison: pd.DataFrame, candidates: pd.DataFrame) -> str:
    blocks = generated_eeg_blocks(comparison)
    units = comparison[unit_columns()].drop_duplicates().shape[0]
    montage_bits = []
    for row in montage_table_data(comparison):
        montage_bits.append(f"{row['montage']} ({row['blocks']} generated EEG conditions)")
    criteria = criterion_count_text(comparison)
    text = [
        f"Simulations used {blocks:,} generated EEG conditions from {num(sim_cfg.get('reps'))} replicates, true K values {values_from_config_or_data(sim_cfg, 'K_true_vals', comparison, 'K_true')}, SNR levels {values_from_config_or_data(sim_cfg, 'SNR_dbs', comparison, 'SNR_dB')} dB, and overlap probabilities {values_from_config_or_data(sim_cfg, 'overlap_probs', comparison, 'overlap_prob')}.",
        f"Each simulation lasted {num(sim_cfg.get('duration_s'))} s at {num(sim_cfg.get('sfreq'))} Hz ({int(sim_cfg.get('duration_s', 0) * sim_cfg.get('sfreq', 0)):,} samples), with overlap events spanning {value_range(sim_cfg.get('overlap_ms_range', []))} ms at strength {num(sim_cfg.get('overlap_strength'))}.",
        f"The same generated EEGs were evaluated separately for {', '.join(montage_bits)}; montage-specific units totalled {units:,}.",
        f"Model selection considered K candidates {values_from_config_or_data(sim_cfg, 'K_candidates', comparison, 'K_estimated')}. The results table contains {len(comparison):,} method-criterion rows; {criteria}.",
        "Criterion choice is evaluated against recoverable structure: sensitivity measures recovered true states, precision penalises extra estimated states, recovered-state F1 balances both, squared K error gives the nonlinear K-count cost, and literal exact-K recovery is retained as a diagnostic.",
        f"K-candidate diagnostics contain {len(candidates):,} rows." if not candidates.empty else "K-candidate diagnostics were not available.",
    ]
    return "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in text) + "</ul>"


def montage_table_data(comparison: pd.DataFrame) -> list[dict]:
    out = []
    for (montage, n_leads), group in comparison.groupby(["montage_type", "n_leads"], dropna=False):
        out.append(
            {
                "montage": f"{montage} ({int(n_leads)} ch)",
                "n_leads": int(n_leads),
                "blocks": group[block_columns()].drop_duplicates().shape[0],
            }
        )
    return sorted(out, key=lambda row: (-row["n_leads"], row["montage"]))


def criterion_count_text(comparison: pd.DataFrame) -> str:
    chunks = []
    for method, group in comparison.groupby("method", dropna=False):
        chunks.append(f"{method_label(method)} contributed {group['criterion'].nunique()} criterion set(s)")
    return "; ".join(chunks)


def values_from_config_or_data(sim_cfg: dict, config_key: str, df: pd.DataFrame, data_col: str) -> str:
    value = sim_cfg.get(config_key)
    if value is None:
        value = sorted(pd.Series(df[data_col]).dropna().unique().tolist())
    if isinstance(value, int) and config_key == "reps":
        value = list(range(1, value + 1))
    if isinstance(value, (list, tuple)):
        return join_values(value)
    return num(value)


def generated_eeg_blocks(df: pd.DataFrame) -> int:
    return df[block_columns()].drop_duplicates().shape[0]


def block_columns() -> list[str]:
    return [
        "rep",
        "K_true",
        "SNR_dB",
        "overlap_prob",
        "overlap_strength",
        "overlap_ms_min",
        "overlap_ms_max",
        "true_template_labels",
        "true_template_indices",
    ]


def unit_columns() -> list[str]:
    return block_columns() + ["montage_type", "n_leads"]


def kv_table(rows: list[tuple[str, object]]) -> str:
    return pd.DataFrame(rows, columns=["Item", "Value"]).to_html(index=False, escape=True)


def join_values(values) -> str:
    vals = list(values)
    try:
        vals = sorted(vals)
    except TypeError:
        pass
    return ", ".join(num(v) for v in vals)


def value_range(values) -> str:
    vals = list(values)
    if not vals:
        return ""
    return f"{num(vals[0])}-{num(vals[-1])}"


def num(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(x)) if x.is_integer() else f"{x:g}"


def method_label(value: object) -> str:
    text = str(value).lower().replace("_", " ")
    if "spm" in text:
        return SPM_LABEL
    if "kmeans" in text or "k means" in text:
        return KM_LABEL
    return clean_label(value)


def clean_label(value: object) -> str:
    return str(value).replace("_", " ").replace("-", " ").title().replace("Spm", "SPM").replace("F1", "F1")


def self_test() -> None:
    sim_cfg = {
        "reps": 2,
        "K_true_vals": [4],
        "SNR_dbs": [0],
        "K_candidates": [2, 3, 4],
        "duration_s": 10,
        "sfreq": 100,
        "overlap_probs": [0],
        "overlap_ms_range": [10, 40],
        "overlap_strength": 0.5,
    }
    rows = []
    for rep in [1, 2]:
        for montage, n_leads in [("full", 71), ("limited", 12)]:
            rows.append(
                {
                    "fit_id": len(rows) + 1,
                    "rep": rep,
                    "method": "spm_vb",
                    "criterion": "icl",
                    "K_true": 4,
                    "SNR_dB": 0,
                    "overlap_prob": 0,
                    "overlap_strength": 0.5,
                    "overlap_ms_min": 10,
                    "overlap_ms_max": 40,
                    "true_template_labels": "A|B|C|D",
                    "true_template_indices": "[1 2 3 4]",
                    "montage_type": montage,
                    "n_leads": n_leads,
                    "K_estimated": 4,
                    "K_correct": 1,
                    "K_error": 0,
                    "sensitivity": 1,
                    "precision": 1,
                    "f1_score": 1,
                    "runtime_s": 1.2,
                    "backfit_mix_available": 1,
                    "backfit_diagnostic_file": "x.mat",
                }
            )
    df = pd.DataFrame(rows)
    html = render_html(
        args=argparse.Namespace(
            config_file=Path("config.json"),
            comparison_csv=Path("comparison.csv"),
            k_candidate_csv=Path("k.csv"),
            output_html=Path("out.html"),
        ),
        config={"simulation": sim_cfg},
        comparison=df,
        candidates=pd.DataFrame({"x": [1, 2]}),
    )
    assert "Koenig" not in html
    assert "Generated EEG conditions" in html


if __name__ == "__main__":
    main()
