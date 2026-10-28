"""Prospective transition-to-autonomic-action directionality analysis.

This sensitivity asks the reverse temporal question from the action-conditioned
transition tensor: do microstate transition probabilities in one 60-s window
improve prediction of the *next* CPI/CSI action quadrant after accounting for
the current action quadrant?

Models are trained only on interictal (II) windows and evaluated with
participant-held-out cross-validation.  The null model contains current action;
the full model adds 24 destination-labelled transition log-ratio features,
centred within participant on their II transition grammar.  The outcome is the
incremental held-out log score of the full model over the current-action null.

This remains observational.  Temporal precedence and held-out prediction are
necessary checks for a directional account but do not identify causality.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

from analyse_action_conditioned_transition_tensor import (
    STATES, TIME_BINS, TIME_ORDER, FS_COLOR, ES_COLOR, GREY, LIGHT_GREY,
    configure_style, save_figure, panel, enrich_windows, merge_autonomic,
    hard_action_labels, counts_cubes, markdown_table,
)

ACTION_SHORT = ("++", "+−", "−+", "−−")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transition_alr_features(cubes: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Create destination-labelled ALR features without pooling coordinate identities."""
    features: list[np.ndarray] = []
    names: list[str] = []
    for source in range(6):
        destinations = [destination for destination in range(6) if destination != source]
        counts = cubes[:, source, destinations] + 0.5
        probabilities = counts / counts.sum(axis=1, keepdims=True)
        reference = destinations[-1]
        for position, destination in enumerate(destinations[:-1]):
            features.append(np.log(probabilities[:, position] / probabilities[:, -1]))
            names.append(f"{STATES[source]}→{STATES[destination]} vs {STATES[reference]}")
    return np.column_stack(features), names


def one_hot(index: np.ndarray) -> np.ndarray:
    output = np.zeros((len(index), 4), dtype=float)
    output[np.arange(len(index)), index.astype(int)] = 1.0
    return output


def prepare_rows(peak_windows: Path, common_windows: Path, minimum_transitions: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    windows = enrich_windows(pd.read_csv(peak_windows, low_memory=False))
    common = pd.read_csv(common_windows, low_memory=False)
    frame = merge_autonomic(windows, common)
    frame["action_csi"] = pd.to_numeric(frame.csi_z, errors="coerce")
    frame["action_cpi"] = pd.to_numeric(frame.cpi_z, errors="coerce")
    labels = hard_action_labels(frame.action_cpi.to_numpy(float), frame.action_csi.to_numpy(float))
    mapping = {label: index for index, label in enumerate(ACTION_SHORT)}
    frame["current_action"] = pd.Series(labels, index=frame.index).map(mapping)

    # Determine the next action only across contiguous windows in the same file.
    frame = frame.sort_values(["participant", "file_key", "window_start_s", "window_end_s"]).copy()
    grouped = frame.groupby(["participant", "file_key"], observed=True, sort=False)
    frame["next_action"] = grouped.current_action.shift(-1)
    next_start = grouped.window_start_s.shift(-1)
    contiguous = np.isclose(
        pd.to_numeric(frame.window_end_s, errors="coerce"),
        pd.to_numeric(next_start, errors="coerce"),
        atol=1e-6,
        equal_nan=False,
    )
    frame.loc[~contiguous, "next_action"] = np.nan
    frame = frame.sort_values("row_id").reset_index(drop=True)

    cubes = counts_cubes(frame)
    transitions = cubes.sum(axis=(1, 2))
    valid = (
        frame.current_action.notna().to_numpy()
        & frame.next_action.notna().to_numpy()
        & (transitions >= minimum_transitions)
        & frame.condition.isin(("ii", "preict")).to_numpy()
        & (frame.condition.eq("ii").to_numpy() | frame.time_bin.isin(TIME_BINS).to_numpy())
    )
    frame = frame.loc[valid].copy().reset_index(drop=True)
    cubes = cubes[valid]
    transitions = transitions[valid]
    features, feature_names = transition_alr_features(cubes)

    # Express transitions as within-participant deviations from the participant II grammar.
    centred = np.full_like(features, np.nan)
    participants = frame.participant.astype(str).to_numpy()
    conditions = frame.condition.astype(str).to_numpy()
    for participant in np.unique(participants):
        positions = np.flatnonzero(participants == participant)
        ii_positions = positions[conditions[positions] == "ii"]
        if len(ii_positions):
            centred[positions] = features[positions] - np.nanmean(features[ii_positions], axis=0)
    complete = np.isfinite(centred).all(axis=1)
    frame = frame.loc[complete].copy().reset_index(drop=True)
    frame["total_transitions"] = transitions[complete]
    return frame, centred[complete], cubes[complete], feature_names


def fit_predictions(frame: pd.DataFrame, features: np.ndarray, regularisation_c: float, seed: int) -> pd.DataFrame:
    groups = frame.participant.astype(str).to_numpy()
    conditions = frame.condition.astype(str).to_numpy()
    current = frame.current_action.to_numpy(int)
    following = frame.next_action.to_numpy(int)
    current_features = one_hot(current)
    splitter = GroupKFold(n_splits=min(10, len(np.unique(groups))), shuffle=True, random_state=seed)
    predictions: list[dict] = []
    dummy = np.zeros((len(frame), 1))
    for train_all, test in splitter.split(dummy, following, groups):
        train = train_all[conditions[train_all] == "ii"]
        if len(train) < 80 or np.unique(following[train]).size < 4:
            continue
        scaler = RobustScaler(quantile_range=(25, 75)).fit(features[train])
        transition_train = scaler.transform(features[train])
        transition_test = scaler.transform(features[test])
        null_train = current_features[train]
        null_test = current_features[test]
        full_train = np.column_stack((null_train, transition_train))
        full_test = np.column_stack((null_test, transition_test))

        train_groups = groups[train]
        sizes = pd.Series(train_groups).map(pd.Series(train_groups).value_counts()).to_numpy(float)
        sample_weight = 1.0 / sizes
        sample_weight *= len(sample_weight) / sample_weight.sum()
        null_model = LogisticRegression(C=regularisation_c, max_iter=2500, solver="lbfgs", random_state=seed)
        full_model = LogisticRegression(C=regularisation_c, max_iter=2500, solver="lbfgs", random_state=seed)
        null_model.fit(null_train, following[train], sample_weight=sample_weight)
        full_model.fit(full_train, following[train], sample_weight=sample_weight)
        null_probability = null_model.predict_proba(null_test)
        full_probability = full_model.predict_proba(full_test)
        null_classes = {int(label): position for position, label in enumerate(null_model.classes_)}
        full_classes = {int(label): position for position, label in enumerate(full_model.classes_)}
        for local, position in enumerate(test):
            target = int(following[position])
            p_null = float(null_probability[local, null_classes[target]])
            p_full = float(full_probability[local, full_classes[target]])
            row = frame.iloc[position]
            predictions.append({
                "participant": row.participant,
                "event_key": row.event_key,
                "condition": row.condition,
                "event_type": row.event_type,
                "time_bin": row.time_bin,
                "row_id": int(row.row_id),
                "current_action": ACTION_SHORT[int(current[position])],
                "next_action": ACTION_SHORT[target],
                "null_probability": p_null,
                "full_probability": p_full,
                "transition_to_next_action_log_gain": math.log(max(p_full, 1e-12)) - math.log(max(p_null, 1e-12)),
                "total_transitions": float(row.total_transitions),
            })
    output = pd.DataFrame(predictions)
    if output.empty:
        raise ValueError("No participant-held-out predictions were generated")
    return output


def participant_cells(predictions: pd.DataFrame) -> pd.DataFrame:
    outcome = "transition_to_next_action_log_gain"
    pre = predictions[
        predictions.condition.eq("preict")
        & predictions.event_type.isin(("functional", "epileptic"))
        & predictions.time_bin.isin(TIME_BINS)
    ]
    event = (pre.groupby(["participant", "event_key", "event_type", "time_bin"], observed=True)[outcome]
             .mean().reset_index())
    pre_cells = (event.groupby(["participant", "event_type", "time_bin"], observed=True)[outcome]
                 .mean().reset_index())
    ii = (predictions[predictions.condition.eq("ii")]
          .groupby("participant", observed=True)[outcome].mean().reset_index())
    types = pre_cells[["participant", "event_type"]].drop_duplicates()
    ii = ii.merge(types, on="participant", how="inner")
    ii["time_bin"] = "II"
    cells = pd.concat([ii[pre_cells.columns], pre_cells], ignore_index=True)
    cells["group"] = cells.event_type.map({"functional": "FS", "epileptic": "ES"})
    return cells


def cluster_difference(frame: pd.DataFrame, draws: int, rng: np.random.Generator) -> tuple[float, float, float, float, int, int, int]:
    outcome = "transition_to_next_action_log_gain"
    pivot = frame.pivot_table(index="participant", columns="group", values=outcome, aggfunc="mean")
    for group in ("FS", "ES"):
        if group not in pivot:
            pivot[group] = np.nan
    fs = pivot.FS.to_numpy(float); es = pivot.ES.to_numpy(float)
    fs_mask = np.isfinite(fs).astype(float); es_mask = np.isfinite(es).astype(float)
    fs_filled = np.nan_to_num(fs); es_filled = np.nan_to_num(es)
    n = len(pivot)
    weights = rng.multinomial(n, np.repeat(1 / n, n), size=draws)
    fs_den = weights @ fs_mask; es_den = weights @ es_mask
    good = (fs_den > 0) & (es_den > 0)
    samples = (weights[good] @ fs_filled) / fs_den[good] - (weights[good] @ es_filled) / es_den[good]
    low, high = np.quantile(samples, (0.025, 0.975))
    return (float(np.nanmean(fs) - np.nanmean(es)), float(low), float(high), float(np.mean(samples > 0)),
            n, int(fs_mask.sum()), int(es_mask.sum()))


def summarise(cells: pd.DataFrame, draws: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcome = "transition_to_next_action_log_gain"
    rng = np.random.default_rng(seed)
    summary_rows = []
    contrast_rows = []
    for group in ("FS", "ES"):
        for time_bin in TIME_ORDER:
            values = cells[cells.group.eq(group) & cells.time_bin.eq(time_bin)][outcome].dropna().to_numpy(float)
            if len(values):
                samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
                low, high = np.quantile(samples, (0.025, 0.975))
                estimate = float(values.mean()); probability = float(np.mean(samples > 0))
            else:
                estimate = low = high = probability = np.nan
            summary_rows.append({"group": group, "time_bin": time_bin, "estimate": estimate,
                                 "ci95_low": low, "ci95_high": high,
                                 "probability_gt_zero": probability, "n_participants": len(values)})
    for time_bin in TIME_ORDER:
        use = cells[cells.time_bin.eq(time_bin)]
        if use.empty or not {"FS", "ES"}.issubset(set(use.group)):
            contrast_rows.append({"contrast": "FS minus ES", "time_bin": time_bin,
                                  "estimate": np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
                                  "probability_gt_zero": np.nan, "n_participants": use.participant.nunique(),
                                  "n_fs": int(use.group.eq("FS").sum()), "n_es": int(use.group.eq("ES").sum())})
            continue
        estimate, low, high, probability, n, n_fs, n_es = cluster_difference(use, draws, rng)
        contrast_rows.append({"contrast": "FS minus ES", "time_bin": time_bin,
                              "estimate": estimate, "ci95_low": low, "ci95_high": high,
                              "probability_gt_zero": probability, "n_participants": n,
                              "n_fs": n_fs, "n_es": n_es})
    return pd.DataFrame(summary_rows), pd.DataFrame(contrast_rows)


def plot_directionality(summary: pd.DataFrame, contrasts: pd.DataFrame, figures: Path) -> None:
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.65), constrained_layout=True,
                             gridspec_kw={"width_ratios": (0.9, 1.4, 1.2)})
    ax = axes[0]
    ax.axis("off")
    ax.text(0.46, 0.86, "Current action", ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=.35", fc="#F2F4F7", ec=GREY, lw=.8))
    ax.text(0.46, 0.52, "Microstate\ntransition grammar", ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=.35", fc="#F2F4F7", ec=GREY, lw=.8))
    ax.text(0.46, 0.18, "Next action", ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=.35", fc="#F2F4F7", ec=GREY, lw=.8))
    ax.annotate("", xy=(0.46, 0.67), xytext=(0.46, 0.76),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.0))
    ax.text(0.78, 0.715, "conditions\ntransitions", ha="center", va="center", fontsize=7, color=GREY)
    ax.annotate("", xy=(0.46, 0.30), xytext=(0.46, 0.41),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.0))
    ax.text(0.79, 0.355, "prospective\ntest", ha="center", va="center", fontsize=7, color=GREY)
    ax.text(0.46, 0.015, "Null model also contains current action", ha="center", va="bottom",
            fontsize=6.4, color=GREY)
    panel(ax, "A")

    x = np.arange(len(TIME_ORDER) - 1)  # final minute cannot predict a following window
    labels = ("II", "−20", "−10", "−5", "−2")
    ax = axes[1]
    for group, color, marker in (("FS", FS_COLOR, "o"), ("ES", ES_COLOR, "s")):
        use = summary[summary.group.eq(group) & summary.time_bin.isin(TIME_ORDER[:-1])].set_index("time_bin").reindex(TIME_ORDER[:-1])
        estimate = use.estimate.to_numpy(float); low = use.ci95_low.to_numpy(float); high = use.ci95_high.to_numpy(float)
        ax.plot(x, estimate, color=color, marker=marker, linewidth=1.3, markersize=4, label=group)
        ax.fill_between(x, low, high, color=color, alpha=.14, linewidth=0)
    ax.axhline(0, color=GREY, linestyle="--", linewidth=.8)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Minutes before event onset")
    ax.set_ylabel("Transition → next-action\nheld-out log-score gain")
    ax.set_title("Prospective directionality")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="best")
    panel(ax, "B")

    ax = axes[2]
    use = contrasts[contrasts.time_bin.isin(TIME_ORDER[:-1])].set_index("time_bin").reindex(TIME_ORDER[:-1])
    y = np.arange(len(use))[::-1]
    estimate = use.estimate.to_numpy(float); low = use.ci95_low.to_numpy(float); high = use.ci95_high.to_numpy(float)
    ax.errorbar(estimate, y, xerr=np.vstack((estimate-low, high-estimate)), fmt="o", color="#7A5195",
                capsize=2.5, linewidth=1.1, markersize=4)
    ax.axvline(0, color=GREY, linestyle="--", linewidth=.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("FS − ES log-score gain")
    ax.set_title("Group difference")
    ax.grid(axis="x")
    panel(ax, "C")
    save_figure(fig, figures, "figure_09_transition_to_next_action_directionality")


def run(peak_windows: Path, common_windows: Path, out_dir: Path, minimum_transitions: int,
        regularisation_c: float, draws: int, seed: int) -> dict:
    tables = out_dir / "tables"; figures = out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    frame, features, _, feature_names = prepare_rows(peak_windows, common_windows, minimum_transitions)
    predictions = fit_predictions(frame, features, regularisation_c, seed)
    cells = participant_cells(predictions)
    summary, contrasts = summarise(cells, draws, seed + 1)
    predictions.to_csv(tables / "transition_to_next_action_predictions.csv.gz", index=False, compression="gzip")
    cells.to_csv(tables / "transition_to_next_action_participant_cells.csv", index=False)
    summary.to_csv(tables / "transition_to_next_action_summary.csv", index=False)
    contrasts.to_csv(tables / "transition_to_next_action_contrasts.csv", index=False)
    pd.DataFrame({"feature": feature_names}).to_csv(tables / "transition_feature_names.csv", index=False)
    plot_directionality(summary, contrasts, figures)
    ii = summary[summary.time_bin.eq("II")]
    lines = [
        "# Prospective transition-to-autonomic-action sensitivity", "",
        "The null model predicts the next CPI/CSI quadrant from the current quadrant. The full model adds 24 destination-labelled microstate-transition log-ratio features. Training used II windows only and all predictions were participant-held-out.", "",
        "A positive incremental log score would indicate that current microstate transitions predict the next autonomic action beyond current autonomic action. This temporal analysis is still observational and does not identify causality.", "",
        "## II validation", "", markdown_table(ii), "",
        "## Trajectory", "", markdown_table(summary), "",
        "## FS−ES contrasts", "", markdown_table(contrasts), "",
        "The final 60-s transition window cannot predict a subsequent within-record action window, so the −1 to 0 bin is intentionally unavailable.", "",
    ]
    (out_dir / "DIRECTIONALITY_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    counts = {"eligible_rows": len(frame), "predicted_rows": len(predictions),
              "participants": predictions.participant.nunique(),
              "ii_predicted_rows": int(predictions.condition.eq("ii").sum()),
              "preictal_predicted_rows": int(predictions.condition.eq("preict").sum())}
    audit = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "code": str(Path(__file__).resolve()), "code_sha256": sha256(Path(__file__)),
        "inputs": {str(peak_windows): sha256(peak_windows), str(common_windows): sha256(common_windows)},
        "config": {"minimum_transitions": minimum_transitions, "regularisation_c": regularisation_c,
                   "bootstrap_draws": draws, "seed": seed},
        "estimand": "participant-held-out incremental log score for transition features predicting the next action quadrant beyond current action, trained on II only",
        "software": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scikit-learn", "matplotlib")},
        "counts": counts,
    }
    completed = [path for path in out_dir.rglob("*")
                 if path.is_file() and path.name != "DIRECTIONALITY_AUDIT.json"]
    audit["outputs"] = {str(path.relative_to(out_dir)): sha256(path) for path in completed}
    (out_dir / "DIRECTIONALITY_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return counts


def self_check() -> None:
    cubes = np.ones((3, 6, 6), float)
    cubes[:, np.arange(6), np.arange(6)] = 0
    features, names = transition_alr_features(cubes)
    assert features.shape == (3, 24) and len(names) == 24
    assert np.allclose(features, 0)
    print("directionality self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peak-windows-csv", type=Path)
    parser.add_argument("--common-windows-csv", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--minimum-transitions", type=int, default=30)
    parser.add_argument("--regularisation-c", type=float, default=0.0003)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=31072026)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(); return
    if any(value is None for value in (args.peak_windows_csv, args.common_windows_csv, args.out_dir)):
        parser.error("--peak-windows-csv, --common-windows-csv and --out-dir are required")
    for path in (args.peak_windows_csv, args.common_windows_csv):
        if not path.exists():
            parser.error(f"Input does not exist: {path}")
    result = run(args.peak_windows_csv, args.common_windows_csv, args.out_dir,
                 args.minimum_transitions, args.regularisation_c, args.bootstrap_draws, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
