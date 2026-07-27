#!/usr/bin/env python3
"""Plot TAS/STAI/YFAS correlations with microstate E coverage, GFP, and covariance."""

from __future__ import annotations

import csv
import argparse
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


CURRENT_ROOT = Path("outputs/hierarchical_microstates")
LEGACY_ROOT = Path("outputs/hierarchical_microstates_OLD")
OUTPUT_ROOTS = (CURRENT_ROOT, LEGACY_ROOT)


def first_existing_output(relative_path: str) -> Path:
    relative = Path(relative_path)
    for root in OUTPUT_ROOTS:
        path = root / relative
        if path.exists():
            return path
    return CURRENT_ROOT / relative


def output_root_for(path: Path) -> Path:
    resolved = path.resolve()
    for root in OUTPUT_ROOTS:
        try:
            resolved.relative_to(root.resolve())
            return root
        except ValueError:
            continue
    return CURRENT_ROOT


BEHAVIOUR_CSV = first_existing_output("behavioural_backfit_analysis/psychometric_raw_matrix.csv")
BACKFIT_FEATURE_CSV = first_existing_output("behavioural_backfit_analysis/method_hard/backfit_feature_matrix.csv")
STATE_METRICS_CSV = first_existing_output("participant_condition_state_backfit_metrics.csv")
COVARIANCE_METRICS_CSV = first_existing_output("hierarchical_state_covariance_metrics.csv")
AXIS_DYNAMICS_CSV = first_existing_output("participant_condition_state_axis_dynamics.csv")
ACTIVE_ROOT = output_root_for(BEHAVIOUR_CSV)
OUTPUT_DIR = ACTIVE_ROOT / "behavioural_backfit_analysis" / "tas_stai_yfas_state_e_plots"

TANOVA_PERMUTATIONS = 5000
TANOVA_RANDOM_SEED = 1
INTERACTION_MIN_N = 8
COVARIANCE_TOKENS = ("covariance", "cov_", "logdet")
AXIAL_TOKENS = (
    "axis_",
    "angular_",
    "perpendicular_",
    "residual_",
    "precession_",
    "pca_",
)
LONG_TABLE_ID_COLS = {
    "participant",
    "condition",
    "group",
    "file_path",
    "fit_path",
    "name",
    "level",
    "backfit_result_source",
    "backfit_method",
    "backfit_available",
    "state_index",
    "template_label",
    "K_estimated",
}
TANOVA_FIELDS = [
    "state_label",
    "state_index",
    "n_participants",
    "n_conditions",
    "conditions",
    "condition_counts",
    "statistic_dgfp_rms",
    "p_value",
    "p_value_max_state",
    "p_value_fdr_bh",
    "n_permutations",
    "status",
    "note",
]
INTERACTION_FIELDS = [
    "psychometry",
    "covariance_metric",
    "axial_metric",
    "n",
    "interaction_beta",
    "interaction_t",
    "interaction_p",
    "interaction_p_fdr_bh",
    "model_r2",
    "status",
    "note",
]

TAS_COL = "emotion_and_personality_test_battery_lemon_tas__tas_overallscore"
STAI_COL = "emotion_and_personality_test_battery_lemon_stai_g_x2__stai_trait_anxiety"
YFAS_COL = "emotion_and_personality_test_battery_lemon_yfas__yfas_symptom_count"
BEHAVIOUR_SOURCE_COLS = {
    "tas_overall": TAS_COL,
    "stai_trait": STAI_COL,
    "yfas_symptom_count": YFAS_COL,
}
BEHAVIOUR_LABELS = {
    "tas_overall": ("TAS", "TAS overall"),
    "stai_trait": ("STAI", "STAI trait anxiety"),
    "yfas_symptom_count": ("YFAS", "YFAS symptom count"),
}
TAS_STAI_BEHAVIOURS = [
    ("tas_overall", "TAS"),
    ("stai_trait", "STAI"),
]
STATE_COVARIANCE_METRICS = [
    "covariance_trace",
    "covariance_trace_per_feature",
    "covariance_logdet",
    "covariance_logdet_per_feature",
]
PRIMARY_STATE_COVARIANCE_METRICS = [
    "covariance_trace_per_feature",
    "covariance_logdet_per_feature",
]
STATE_COVARIANCE_LABELS = {
    "covariance_trace": "covariance trace",
    "covariance_trace_per_feature": "covariance trace/feature",
    "covariance_logdet": "covariance logdet",
    "covariance_logdet_per_feature": "covariance logdet/feature",
}
STATE_COVARIANCE_CONDITION_FIELDS = [
    "state_label",
    "covariance_metric",
    "n_pairs",
    "ec_mean",
    "eo_mean",
    "mean_delta_eo_minus_ec",
    "paired_t",
    "paired_t_p",
    "paired_t_p_fdr_bh",
    "wilcoxon_statistic",
    "wilcoxon_p",
    "wilcoxon_p_fdr_bh",
    "status",
    "note",
]


def parse_float(value: object) -> float:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "na", "n/a", "none", "null"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_table_if_exists(path: Path) -> list[dict[str, str]]:
    return read_table(path) if path.exists() else []


def index_by_participant(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["participant"]: row for row in rows if row.get("participant")}


def merge_behaviour_backfit() -> list[dict[str, float | str]]:
    behaviour = index_by_participant(read_table(BEHAVIOUR_CSV))
    backfit = index_by_participant(read_table(BACKFIT_FEATURE_CSV))
    out: list[dict[str, float | str]] = []
    for participant in sorted(set(behaviour) & set(backfit)):
        row: dict[str, float | str] = {"participant": participant}
        for score_col, source_col in BEHAVIOUR_SOURCE_COLS.items():
            row[score_col] = parse_float(behaviour[participant].get(source_col))
        for feature in (
            "state__EC__E__occupancy",
            "state__EO__E__occupancy",
            "delta__EO_minus_EC__E__occupancy",
            "state__EC__E__mean_gfp",
            "state__EO__E__mean_gfp",
            "delta__EO_minus_EC__E__mean_gfp",
        ):
            row[feature] = parse_float(backfit[participant].get(feature))
        out.append(row)
    return out


def merge_behaviour_state_covariance() -> list[dict[str, float | str]]:
    out: dict[str, dict[str, float | str]] = {}
    for participant, behaviour in index_by_participant(read_table(BEHAVIOUR_CSV)).items():
        row: dict[str, float | str] = {"participant": participant}
        for score_col, source_col in BEHAVIOUR_SOURCE_COLS.items():
            row[score_col] = parse_float(behaviour.get(source_col))
        out[participant] = row

    for row in read_table_if_exists(COVARIANCE_METRICS_CSV):
        if row.get("level") != "participant_condition":
            continue
        participant = str(row.get("participant", "")).strip()
        condition = str(row.get("condition", "")).strip()
        label = str(row.get("template_label", "")).strip()
        if not participant or condition not in {"EC", "EO"} or label not in {"A", "E"}:
            continue
        target = out.setdefault(participant, {"participant": participant})
        for metric in STATE_COVARIANCE_METRICS:
            value = parse_float(row.get(metric))
            if math.isfinite(value):
                target[f"state__{condition}__{label}__{metric}"] = value

    for target in out.values():
        for label in ("A", "E"):
            for metric in STATE_COVARIANCE_METRICS:
                ec_col = f"state__EC__{label}__{metric}"
                eo_col = f"state__EO__{label}__{metric}"
                ec = parse_float(target.get(ec_col))
                eo = parse_float(target.get(eo_col))
                if math.isfinite(ec) and math.isfinite(eo):
                    target[f"delta__EO_minus_EC__{label}__{metric}"] = eo - ec
    return list(out.values())


def state_covariance_metric_specs(label: str, metrics: list[str] | None = None) -> list[tuple[str, str]]:
    specs = []
    for metric in metrics or STATE_COVARIANCE_METRICS:
        metric_label = STATE_COVARIANCE_LABELS[metric]
        specs.extend(
            [
                (f"state__EC__{label}__{metric}", f"state {label} {metric_label} EC"),
                (f"state__EO__{label}__{metric}", f"state {label} {metric_label} EO"),
                (f"delta__EO_minus_EC__{label}__{metric}", f"state {label} {metric_label} EO-EC"),
            ]
        )
    return specs


def state_e_long_rows() -> list[dict[str, object]]:
    rows = []
    for row in read_table(STATE_METRICS_CSV):
        if row.get("backfit_method") != "hard":
            continue
        if row.get("template_label") != "E":
            continue
        if row.get("condition") not in {"EC", "EO"}:
            continue
        occupancy = parse_float(row.get("occupancy"))
        if not math.isfinite(occupancy):
            continue
        rows.append(
            {
                "participant": row["participant"],
                "condition_eo": 1.0 if row["condition"] == "EO" else 0.0,
                "occupancy": occupancy,
            }
        )
    return rows


def mixed_model_coverage_estimates() -> dict[str, float]:
    """Simple random-intercept estimate for state E coverage, adjusted for condition."""
    rows = state_e_long_rows()
    if not rows:
        return {}

    y = np.array([float(row["occupancy"]) for row in rows], dtype=float)
    x = np.column_stack([np.ones(len(rows)), [float(row["condition_eo"]) for row in rows]])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta

    by_participant: dict[str, list[float]] = defaultdict(list)
    for row, resid in zip(rows, residual):
        by_participant[str(row["participant"])].append(float(resid))

    diffs = [
        vals[1] - vals[0]
        for vals in by_participant.values()
        if len(vals) == 2 and all(math.isfinite(v) for v in vals)
    ]
    sigma_e2 = float(np.var(diffs, ddof=1) / 2.0) if len(diffs) >= 2 else float(np.var(residual, ddof=1))
    means = np.array([np.mean(vals) for vals in by_participant.values()], dtype=float)
    mean_n = np.array([len(vals) for vals in by_participant.values()], dtype=float)
    sigma_b2 = max(0.0, float(np.var(means, ddof=1) - np.mean(sigma_e2 / mean_n)))

    fixed_average_condition = float(beta[0] + 0.5 * beta[1])
    estimates = {}
    for participant, vals in by_participant.items():
        n_i = len(vals)
        shrinkage = sigma_b2 / (sigma_b2 + sigma_e2 / n_i) if sigma_b2 > 0 else 0.0
        estimates[participant] = fixed_average_condition + shrinkage * float(np.mean(vals))
    return estimates


def attach_mixed_estimate(rows: list[dict[str, float | str]]) -> None:
    estimates = mixed_model_coverage_estimates()
    for row in rows:
        row["mixed_model_state_e_coverage"] = estimates.get(str(row["participant"]), np.nan)


def finite_xy(rows: list[dict[str, float | str]], x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([parse_float(row.get(x_col)) for row in rows], dtype=float)
    y = np.array([parse_float(row.get(y_col)) for row in rows], dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def correlation_row(rows: list[dict[str, float | str]], behaviour: str, metric: str) -> dict[str, object]:
    x, y = finite_xy(rows, behaviour, metric)
    out: dict[str, object] = {
        "behaviour": behaviour,
        "microstate_metric": metric,
        "n": int(x.size),
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
    }
    if x.size >= 3 and np.std(x) > 0 and np.std(y) > 0:
        pearson = stats.pearsonr(x, y)
        spearman = stats.spearmanr(x, y)
        out.update(
            {
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
            }
        )
    return out


def format_value(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.10g}"
    return value


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    fieldnames = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key)) for key in fieldnames})


def plot_panel(ax: plt.Axes, rows: list[dict[str, float | str]], x_col: str, y_col: str, title: str) -> None:
    x, y = finite_xy(rows, x_col, y_col)
    ax.scatter(x, y, s=28, alpha=0.78, color="#2b6cb0", edgecolors="white", linewidths=0.4)
    if x.size >= 3 and np.std(x) > 0 and np.std(y) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax.plot(xx, slope * xx + intercept, color="#9b2c2c", linewidth=1.5)
        r_value, p_value = stats.pearsonr(x, y)
        ax.text(
            0.03,
            0.97,
            f"n={x.size}\nr={r_value:.2f}, p={p_value:.3g}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#d0d0d0"},
        )
    else:
        ax.text(0.5, 0.5, "not enough data", transform=ax.transAxes, ha="center", va="center")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(BEHAVIOUR_LABELS[x_col][1])
    ax.grid(True, alpha=0.25)


def save_grid(
    rows: list[dict[str, float | str]],
    filename: str,
    metrics: list[tuple[str, str]],
    ylabel: str,
    behaviours: list[tuple[str, str]] | None = None,
) -> None:
    behaviours = behaviours or [(col, labels[0]) for col, labels in BEHAVIOUR_LABELS.items()]
    fig, axes = plt.subplots(
        len(behaviours), len(metrics), figsize=(5.2 * len(metrics), 4.1 * len(behaviours)), squeeze=False
    )
    for row_idx, (behaviour_col, behaviour_label) in enumerate(behaviours):
        for col_idx, (metric_col, metric_label) in enumerate(metrics):
            ax = axes[row_idx][col_idx]
            plot_panel(ax, rows, behaviour_col, metric_col, f"{behaviour_label} vs {metric_label}")
            if col_idx == 0:
                ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close(fig)


def paired_condition_values(
    rows: list[dict[str, float | str]], label: str, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    ec_col = f"state__EC__{label}__{metric}"
    eo_col = f"state__EO__{label}__{metric}"
    ec = np.array([parse_float(row.get(ec_col)) for row in rows], dtype=float)
    eo = np.array([parse_float(row.get(eo_col)) for row in rows], dtype=float)
    keep = np.isfinite(ec) & np.isfinite(eo)
    return ec[keep], eo[keep]


def covariance_condition_stat_row(rows: list[dict[str, float | str]], label: str, metric: str) -> dict[str, object]:
    ec, eo = paired_condition_values(rows, label, metric)
    out: dict[str, object] = {
        "state_label": label,
        "covariance_metric": metric,
        "n_pairs": int(ec.size),
        "ec_mean": float(np.mean(ec)) if ec.size else np.nan,
        "eo_mean": float(np.mean(eo)) if eo.size else np.nan,
        "mean_delta_eo_minus_ec": float(np.mean(eo - ec)) if ec.size else np.nan,
        "paired_t": np.nan,
        "paired_t_p": np.nan,
        "paired_t_p_fdr_bh": np.nan,
        "wilcoxon_statistic": np.nan,
        "wilcoxon_p": np.nan,
        "wilcoxon_p_fdr_bh": np.nan,
        "status": "skipped",
        "note": "",
    }
    if ec.size < 2:
        out["note"] = "Fewer than two paired EC/EO observations."
        return out
    diff = eo - ec
    if float(np.std(diff, ddof=0)) > 0:
        paired = stats.ttest_rel(eo, ec)
        out["paired_t"] = float(paired.statistic)
        out["paired_t_p"] = float(paired.pvalue)
    if np.any(diff != 0):
        try:
            wilcoxon = stats.wilcoxon(eo, ec)
            out["wilcoxon_statistic"] = float(wilcoxon.statistic)
            out["wilcoxon_p"] = float(wilcoxon.pvalue)
        except ValueError as exc:
            out["note"] = str(exc)
    out["status"] = "ok"
    return out


def state_a_covariance_condition_stats(rows: list[dict[str, float | str]]) -> list[dict[str, object]]:
    out = [covariance_condition_stat_row(rows, "A", metric) for metric in STATE_COVARIANCE_METRICS]
    t_fdr = bh_fdr([parse_float(row["paired_t_p"]) for row in out])
    w_fdr = bh_fdr([parse_float(row["wilcoxon_p"]) for row in out])
    for row, t_p, w_p in zip(out, t_fdr, w_fdr):
        row["paired_t_p_fdr_bh"] = t_p
        row["wilcoxon_p_fdr_bh"] = w_p
    return out


def plot_state_a_covariance_ec_eo(rows: list[dict[str, float | str]], stat_rows: list[dict[str, object]]) -> None:
    stat_by_metric = {str(row["covariance_metric"]): row for row in stat_rows}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
    for ax, metric in zip(axes.ravel(), STATE_COVARIANCE_METRICS):
        ec, eo = paired_condition_values(rows, "A", metric)
        if ec.size:
            ax.boxplot([ec, eo], tick_labels=["EC", "EO"], showfliers=False)
            x_ec = np.full(ec.shape, 1.0)
            x_eo = np.full(eo.shape, 2.0)
            ax.scatter(x_ec, ec, s=16, alpha=0.6, color="#2b6cb0", edgecolors="white", linewidths=0.25)
            ax.scatter(x_eo, eo, s=16, alpha=0.6, color="#9b2c2c", edgecolors="white", linewidths=0.25)
            stat_row = stat_by_metric.get(metric, {})
            p_value = parse_float(stat_row.get("paired_t_p"))
            ax.text(
                0.03,
                0.97,
                f"n={ec.size}\npaired t p={p_value:.3g}" if math.isfinite(p_value) else f"n={ec.size}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#d0d0d0"},
            )
        else:
            ax.text(0.5, 0.5, "not enough data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(f"state A {STATE_COVARIANCE_LABELS[metric]}")
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Microstate A covariance: EC vs EO")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUTPUT_DIR / "state_a_covariance_ec_vs_eo.png", dpi=200)
    plt.close(fig)


def bh_fdr(p_values: list[float]) -> list[float]:
    p = np.array(p_values, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size == 0:
        return out.tolist()
    vals = p[finite]
    order = np.argsort(vals)
    ranked = vals[order] * finite.size / np.arange(1, finite.size + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out[finite[order]] = np.clip(adjusted, 0.0, 1.0)
    return out.tolist()


def read_numeric_matrix(path: Path) -> np.ndarray:
    rows = []
    if not path.exists():
        return np.empty((0, 0), dtype=float)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.reader(handle):
            parsed = [parse_float(value) for value in raw]
            if parsed and all(math.isfinite(value) for value in parsed):
                rows.append(parsed)
    lengths = {len(row) for row in rows}
    if not rows or len(lengths) != 1:
        return np.empty((0, 0), dtype=float)
    return np.array(rows, dtype=float)


def normalise_topomap(values: np.ndarray) -> np.ndarray | None:
    x = np.array(values, dtype=float)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return None
    x = x - float(np.mean(x))
    rms = float(np.sqrt(np.mean(x**2)))
    if not math.isfinite(rms) or rms <= 0:
        return None
    return x / rms


def participants_with_ec_eo(root: Path) -> list[str]:
    conditions: dict[str, set[str]] = defaultdict(set)
    for row in read_table_if_exists(root / "normalised_input_manifest.csv"):
        participant = str(row.get("participant", "")).strip()
        condition = str(row.get("condition", "")).strip()
        if participant and condition in {"EC", "EO"}:
            conditions[participant].add(condition)
    return sorted(participant for participant, values in conditions.items() if {"EC", "EO"}.issubset(values))


def state_label(index: int) -> str:
    return chr(ord("A") + index) if 0 <= index < 26 else f"state_{index + 1}"


def tanova_skip_row(note: str, n_participants: int = 0) -> dict[str, object]:
    return {
        "state_label": "",
        "state_index": "",
        "n_participants": n_participants,
        "n_conditions": 2,
        "conditions": "EC;EO",
        "condition_counts": "",
        "statistic_dgfp_rms": np.nan,
        "p_value": np.nan,
        "p_value_max_state": np.nan,
        "p_value_fdr_bh": np.nan,
        "n_permutations": TANOVA_PERMUTATIONS,
        "status": "skipped",
        "note": note,
    }


def covariance_state_indices(root: Path) -> dict[tuple[str, str, str], dict[str, int]]:
    out: dict[tuple[str, str, str], dict[str, int]] = defaultdict(dict)
    for row in read_table_if_exists(root / "hierarchical_state_covariance_metrics.csv"):
        label = str(row.get("template_label", "")).strip()
        index = int(parse_float(row.get("state_index"))) - 1 if math.isfinite(parse_float(row.get("state_index"))) else -1
        if not label or index < 0:
            continue
        key = (
            str(row.get("level", "")).strip(),
            str(row.get("participant", "")).strip(),
            str(row.get("condition", "")).strip(),
        )
        out[key][label] = index
    return out


def global_center_maps_by_label(root: Path, indices: dict[tuple[str, str, str], dict[str, int]]) -> dict[str, np.ndarray]:
    matrix = read_numeric_matrix(root / "global" / "global_centers.csv")
    label_indices = indices.get(("global", "", ""), {})
    out = {}
    for label, index in label_indices.items():
        if matrix.size and index < matrix.shape[0]:
            topo = normalise_topomap(matrix[index])
            if topo is not None:
                out[label] = topo
    return out


def load_labelled_center(
    matrix_path: Path,
    label: str,
    label_indices: dict[str, int],
    global_maps: dict[str, np.ndarray],
) -> np.ndarray | None:
    matrix = read_numeric_matrix(matrix_path)
    index = label_indices.get(label)
    if matrix.size == 0 or index is None or index >= matrix.shape[0]:
        return None
    topo = normalise_topomap(matrix[index])
    if topo is None:
        return None
    reference = global_maps.get(label)
    if reference is not None and reference.shape == topo.shape and float(np.dot(topo, reference)) < 0:
        topo = -topo
    return topo


def paired_condition_tanova_statistic(differences: np.ndarray) -> float:
    if differences.size == 0:
        return np.nan
    mean_difference = np.mean(differences, axis=0)
    return float(np.sqrt(np.mean(mean_difference**2)))


def hierarchical_condition_tanova_rows(root: Path) -> list[dict[str, object]]:
    participants = participants_with_ec_eo(root)
    if not participants:
        return [tanova_skip_row(f"No paired EC/EO participants found in {root / 'normalised_input_manifest.csv'}.")]

    indices = covariance_state_indices(root)
    global_maps = global_center_maps_by_label(root, indices)
    labels = sorted(global_maps) or sorted(
        {
            label
            for (level, _, condition), by_label in indices.items()
            if level == "participant_condition" and condition in {"EC", "EO"}
            for label in by_label
        }
    )
    if not labels:
        return [tanova_skip_row("No template-label/state-index mapping was found for condition TANOVA.", len(participants))]

    paired_by_label: dict[str, list[tuple[str, np.ndarray]]] = {label: [] for label in labels}
    for participant in participants:
        for label in labels:
            ec = load_labelled_center(
                root / "participant_conditions" / participant / "EC" / "participant_condition_centers.csv",
                label,
                indices.get(("participant_condition", participant, "EC"), {}),
                global_maps,
            )
            eo = load_labelled_center(
                root / "participant_conditions" / participant / "EO" / "participant_condition_centers.csv",
                label,
                indices.get(("participant_condition", participant, "EO"), {}),
                global_maps,
            )
            if ec is not None and eo is not None and ec.shape == eo.shape:
                paired_by_label[label].append((participant, eo - ec))

    usable_labels = [label for label in labels if len(paired_by_label[label]) >= 2]
    if not usable_labels:
        return [tanova_skip_row("No states had at least two paired EC/EO participant-condition maps.", len(participants))]

    observed = np.array(
        [paired_condition_tanova_statistic(np.vstack([diff for _, diff in paired_by_label[label]])) for label in usable_labels],
        dtype=float,
    )

    all_participants = sorted({participant for label in usable_labels for participant, _ in paired_by_label[label]})
    rng = np.random.default_rng(TANOVA_RANDOM_SEED)
    null = np.empty((TANOVA_PERMUTATIONS, len(usable_labels)), dtype=float)
    for permutation in range(TANOVA_PERMUTATIONS):
        signs = {participant: rng.choice([-1.0, 1.0]) for participant in all_participants}
        for index, label in enumerate(usable_labels):
            signed = np.vstack([signs[participant] * diff for participant, diff in paired_by_label[label]])
            null[permutation, index] = paired_condition_tanova_statistic(signed)

    max_null = np.nanmax(null, axis=1)
    p_values = []
    p_max_values = []
    for index, statistic in enumerate(observed):
        if math.isfinite(float(statistic)):
            p_values.append(float((1 + np.sum(null[:, index] >= statistic)) / (TANOVA_PERMUTATIONS + 1)))
            p_max_values.append(float((1 + np.sum(max_null >= statistic)) / (TANOVA_PERMUTATIONS + 1)))
        else:
            p_values.append(np.nan)
            p_max_values.append(np.nan)
    p_fdr = bh_fdr(p_values)

    rows = []
    for index, label in enumerate(usable_labels):
        n_pairs = len(paired_by_label[label])
        state_index_value = indices.get(("global", "", ""), {}).get(label, index) + 1
        rows.append(
            {
                "state_label": label,
                "state_index": state_index_value,
                "n_participants": n_pairs,
                "n_conditions": 2,
                "conditions": "EC;EO",
                "condition_counts": f"EC:{n_pairs};EO:{n_pairs}",
                "statistic_dgfp_rms": float(observed[index]),
                "p_value": p_values[index],
                "p_value_max_state": p_max_values[index],
                "p_value_fdr_bh": p_fdr[index],
                "n_permutations": TANOVA_PERMUTATIONS,
                "status": "ok",
                "note": "Paired EC/EO sign-flip permutation on participant-condition centre maps.",
            }
        )
    return rows


def column_matches(column: str, tokens: tuple[str, ...]) -> bool:
    lower = column.lower()
    return any(token in lower for token in tokens)


def candidate_numeric_columns(rows: list[dict[str, str]], tokens: tuple[str, ...]) -> list[str]:
    if not rows:
        return []
    columns = list(rows[0])
    out = []
    for column in columns:
        if column in LONG_TABLE_ID_COLS or not column_matches(column, tokens):
            continue
        finite = sum(math.isfinite(parse_float(row.get(column))) for row in rows)
        if finite >= INTERACTION_MIN_N:
            out.append(column)
    return out


def clean_feature_part(value: object) -> str:
    text = str(value or "").strip()
    return text.replace(" ", "_").replace("/", "_").replace("\\", "_")


def long_feature_name(prefix: str, row: dict[str, str], metric: str) -> str:
    parts = [prefix]
    for column in ("level", "backfit_method", "condition"):
        value = clean_feature_part(row.get(column))
        if value:
            parts.append(value)
    label = clean_feature_part(row.get("template_label"))
    if not label:
        index = clean_feature_part(row.get("state_index"))
        label = f"state_{index}" if index else ""
    if label:
        parts.append(label)
    parts.append(metric)
    return "__".join(parts)


def add_flat_candidate_features(
    out: dict[str, dict[str, float | str]], path: Path, tokens: tuple[str, ...]
) -> None:
    rows = read_table_if_exists(path)
    columns = candidate_numeric_columns(rows, tokens)
    if not columns:
        return
    for participant, row in index_by_participant(rows).items():
        target = out.setdefault(participant, {"participant": participant})
        for column in columns:
            value = parse_float(row.get(column))
            if math.isfinite(value):
                target[column] = value


def add_long_candidate_features(
    out: dict[str, dict[str, float | str]], path: Path, prefix: str, tokens: tuple[str, ...]
) -> None:
    rows = read_table_if_exists(path)
    columns = candidate_numeric_columns(rows, tokens)
    if not columns:
        return
    for row in rows:
        participant = str(row.get("participant", "")).strip()
        if not participant:
            continue
        target = out.setdefault(participant, {"participant": participant})
        for column in columns:
            value = parse_float(row.get(column))
            if math.isfinite(value):
                target[long_feature_name(prefix, row, column)] = value


def interaction_feature_rows() -> list[dict[str, float | str]]:
    out: dict[str, dict[str, float | str]] = {}
    for row in read_table(BEHAVIOUR_CSV):
        participant = str(row.get("participant", "")).strip()
        if not participant:
            continue
        target: dict[str, float | str] = {"participant": participant}
        for score_col, source_col in BEHAVIOUR_SOURCE_COLS.items():
            target[score_col] = parse_float(row.get(source_col))
        out[participant] = target

    add_flat_candidate_features(out, BACKFIT_FEATURE_CSV, COVARIANCE_TOKENS)
    add_flat_candidate_features(out, BACKFIT_FEATURE_CSV, AXIAL_TOKENS)
    add_long_candidate_features(out, COVARIANCE_METRICS_CSV, "covariance", COVARIANCE_TOKENS)
    add_long_candidate_features(out, AXIS_DYNAMICS_CSV, "axial", AXIAL_TOKENS)
    return list(out.values())


def standardise(values: np.ndarray) -> np.ndarray | None:
    sd = float(np.std(values, ddof=0))
    if not math.isfinite(sd) or sd <= 0:
        return None
    return (values - float(np.mean(values))) / sd


def interaction_regression_row(
    rows: list[dict[str, float | str]], psychometry: str, covariance_metric: str, axial_metric: str
) -> dict[str, object] | None:
    p = np.array([parse_float(row.get(psychometry)) for row in rows], dtype=float)
    c = np.array([parse_float(row.get(covariance_metric)) for row in rows], dtype=float)
    y = np.array([parse_float(row.get(axial_metric)) for row in rows], dtype=float)
    keep = np.isfinite(p) & np.isfinite(c) & np.isfinite(y)
    if int(np.sum(keep)) < INTERACTION_MIN_N:
        return None

    zp = standardise(p[keep])
    zc = standardise(c[keep])
    zy = standardise(y[keep])
    if zp is None or zc is None or zy is None:
        return None

    x = np.column_stack([np.ones(zp.size), zp, zc, zp * zc])
    if np.linalg.matrix_rank(x) < x.shape[1]:
        return None
    beta, *_ = np.linalg.lstsq(x, zy, rcond=None)
    residual = zy - x @ beta
    dof = int(zp.size - x.shape[1])
    if dof <= 0:
        return None
    sigma2 = float(np.sum(residual**2) / dof)
    xtx_inv = np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    if not math.isfinite(float(se[3])) or float(se[3]) <= 0:
        return None
    t_value = float(beta[3] / se[3])
    p_value = float(2.0 * stats.t.sf(abs(t_value), dof))
    ss_total = float(np.sum((zy - np.mean(zy)) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / ss_total if ss_total > 0 else np.nan
    return {
        "psychometry": psychometry,
        "covariance_metric": covariance_metric,
        "axial_metric": axial_metric,
        "n": int(zp.size),
        "interaction_beta": float(beta[3]),
        "interaction_t": t_value,
        "interaction_p": p_value,
        "interaction_p_fdr_bh": np.nan,
        "model_r2": r2,
        "status": "ok",
        "note": "OLS model: z(axial) ~ z(psychometry) * z(covariance).",
    }


def interaction_skip_row(note: str) -> dict[str, object]:
    return {
        "psychometry": "",
        "covariance_metric": "",
        "axial_metric": "",
        "n": 0,
        "interaction_beta": np.nan,
        "interaction_t": np.nan,
        "interaction_p": np.nan,
        "interaction_p_fdr_bh": np.nan,
        "model_r2": np.nan,
        "status": "skipped",
        "note": note,
    }


def psychometry_covariance_axial_interaction_rows() -> list[dict[str, object]]:
    rows = interaction_feature_rows()
    if not rows:
        return [interaction_skip_row("No psychometry rows were available.")]
    all_columns = sorted(set().union(*(row.keys() for row in rows)))
    covariance_columns = [column for column in all_columns if column_matches(column, COVARIANCE_TOKENS)]
    axial_columns = [column for column in all_columns if column_matches(column, AXIAL_TOKENS)]
    if not covariance_columns or not axial_columns:
        missing = []
        if not covariance_columns:
            missing.append(str(COVARIANCE_METRICS_CSV))
        if not axial_columns:
            missing.append(str(AXIS_DYNAMICS_CSV))
        return [interaction_skip_row("No covariance and/or axial candidate columns found. Expected optional CSVs: " + "; ".join(missing))]

    out = []
    for psychometry in BEHAVIOUR_LABELS:
        for covariance_metric in covariance_columns:
            for axial_metric in axial_columns:
                row = interaction_regression_row(rows, psychometry, covariance_metric, axial_metric)
                if row is not None:
                    out.append(row)
    if not out:
        return [interaction_skip_row(f"No non-degenerate interactions had n >= {INTERACTION_MIN_N}.")]
    fdr = bh_fdr([float(row["interaction_p"]) for row in out])
    for row, adjusted in zip(out, fdr):
        row["interaction_p_fdr_bh"] = adjusted
    out.sort(key=lambda row: (parse_float(row.get("interaction_p")), row["psychometry"], row["covariance_metric"], row["axial_metric"]))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused TAS/STAI/YFAS analysis for microstate E and optional covariance/axis outputs.")
    parser.add_argument("--behaviour-csv", type=Path, default=BEHAVIOUR_CSV)
    parser.add_argument("--backfit-feature-csv", type=Path, default=BACKFIT_FEATURE_CSV)
    parser.add_argument("--state-metrics-csv", type=Path, default=STATE_METRICS_CSV)
    parser.add_argument("--covariance-metrics-csv", type=Path, default=COVARIANCE_METRICS_CSV)
    parser.add_argument("--axis-dynamics-csv", type=Path, default=AXIS_DYNAMICS_CSV)
    parser.add_argument("--hierarchical-root", type=Path, default=ACTIVE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def configure_paths(args: argparse.Namespace) -> None:
    global BEHAVIOUR_CSV, BACKFIT_FEATURE_CSV, STATE_METRICS_CSV
    global COVARIANCE_METRICS_CSV, AXIS_DYNAMICS_CSV, ACTIVE_ROOT, OUTPUT_DIR

    BEHAVIOUR_CSV = args.behaviour_csv
    BACKFIT_FEATURE_CSV = args.backfit_feature_csv
    STATE_METRICS_CSV = args.state_metrics_csv
    COVARIANCE_METRICS_CSV = args.covariance_metrics_csv
    AXIS_DYNAMICS_CSV = args.axis_dynamics_csv
    ACTIVE_ROOT = args.hierarchical_root
    OUTPUT_DIR = args.output_dir


def main() -> None:
    args = parse_args()
    if args.self_test:
        assert state_label(0) == "A"
        assert state_label(26) == "state_27"
        print("self-test passed")
        return
    configure_paths(args)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = merge_behaviour_backfit()
    attach_mixed_estimate(rows)
    covariance_rows = merge_behaviour_state_covariance()

    coverage_metrics = [
        ("state__EC__E__occupancy", "state E coverage EC"),
        ("state__EO__E__occupancy", "state E coverage EO"),
        ("delta__EO_minus_EC__E__occupancy", "state E coverage EO-EC"),
    ]
    gfp_metrics = [
        ("state__EC__E__mean_gfp", "state E GFP EC"),
        ("state__EO__E__mean_gfp", "state E GFP EO"),
        ("delta__EO_minus_EC__E__mean_gfp", "state E GFP EO-EC"),
    ]
    mixed_metrics = [("mixed_model_state_e_coverage", "mixed-model state E coverage")]

    save_grid(rows, "tas_stai_yfas_vs_state_e_coverage.png", coverage_metrics, "State E coverage")
    save_grid(rows, "tas_stai_yfas_vs_state_e_gfp.png", gfp_metrics, "State E mean GFP")
    save_grid(
        rows, "tas_stai_yfas_vs_mixed_model_state_e_coverage.png", mixed_metrics, "State E coverage estimate"
    )
    state_e_covariance_metrics = state_covariance_metric_specs("E", PRIMARY_STATE_COVARIANCE_METRICS)
    save_grid(
        covariance_rows,
        "tas_stai_vs_state_e_covariance.png",
        state_e_covariance_metrics,
        "State E covariance",
        behaviours=TAS_STAI_BEHAVIOURS,
    )

    stat_rows = []
    for behaviour in BEHAVIOUR_LABELS:
        for metric, _ in [*coverage_metrics, *gfp_metrics, *mixed_metrics]:
            stat_rows.append(correlation_row(rows, behaviour, metric))
    write_csv(OUTPUT_DIR / "tas_stai_yfas_state_e_correlations.csv", stat_rows)

    covariance_stat_rows = []
    for behaviour, _ in TAS_STAI_BEHAVIOURS:
        for metric, _ in state_e_covariance_metrics:
            covariance_stat_rows.append(correlation_row(covariance_rows, behaviour, metric))
    write_csv(OUTPUT_DIR / "tas_stai_state_e_covariance_correlations.csv", covariance_stat_rows)

    state_a_covariance_stats = state_a_covariance_condition_stats(covariance_rows)
    write_csv(
        OUTPUT_DIR / "state_a_covariance_ec_vs_eo_stats.csv",
        state_a_covariance_stats,
        STATE_COVARIANCE_CONDITION_FIELDS,
    )
    plot_state_a_covariance_ec_eo(covariance_rows, state_a_covariance_stats)

    tanova_rows = hierarchical_condition_tanova_rows(ACTIVE_ROOT)
    write_csv(OUTPUT_DIR / "hierarchical_condition_tanova_by_state.csv", tanova_rows, TANOVA_FIELDS)
    stale_group_tanova = OUTPUT_DIR / "hierarchical_group_tanova_by_state.csv"
    if stale_group_tanova.exists():
        stale_group_tanova.unlink()

    interaction_rows = psychometry_covariance_axial_interaction_rows()
    write_csv(
        OUTPUT_DIR / "tas_stai_yfas_covariance_axial_interactions.csv",
        interaction_rows,
        INTERACTION_FIELDS,
    )

    note = (
        "Mixed-model state E coverage is a simple random-intercept estimate from "
        "hard-backfit EC/EO state E occupancy adjusted for condition. The exported "
        "gaussian_mixture backfit rows are unavailable in the current CSVs, so no "
        "gaussian-mixture coverage plot was generated.\n\n"
        "tas_stai_state_e_covariance_correlations.csv and "
        "tas_stai_vs_state_e_covariance.png test TAS/STAI relationships with "
        "participant-condition microstate E covariance metrics from "
        "hierarchical_state_covariance_metrics.csv.\n\n"
        "state_a_covariance_ec_vs_eo_stats.csv and state_a_covariance_ec_vs_eo.png "
        "compare paired EC and EO microstate A covariance metrics.\n\n"
        "hierarchical_condition_tanova_by_state.csv is a paired EC-vs-EO "
        "topographic permutation test on participant-condition hierarchical maps. "
        "Maps are common-average centred, RMS-normalised, keyed by template label, "
        "and sign-aligned to the global template before testing.\n\n"
        "tas_stai_yfas_covariance_axial_interactions.csv tests OLS interaction terms "
        "of the form z(axial measure) ~ z(psychometry) * z(covariance measure). "
        "This table is populated when hierarchical_state_covariance_metrics.csv and "
        "participant_condition_state_axis_dynamics.csv are present.\n"
    )
    (OUTPUT_DIR / "README.txt").write_text(note, encoding="utf-8")
    print(f"Wrote plots and analysis tables to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
