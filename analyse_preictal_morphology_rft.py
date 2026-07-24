"""Permutation maximum-field and TFCE inference for preictal morphology fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from analyse_preictal_publication import attach_strata, norm_path


def scalp_grid(positions: np.ndarray, fwhm_deg: float, grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    axis = np.linspace(-1, 1, grid_size)
    xx, yy = np.meshgrid(axis, axis)
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    inside = (xy[:, 0] ** 2 + xy[:, 1] ** 2 <= 1) & (Delaunay(pos[:, :2]).find_simplex(xy) >= 0)
    xy = xy[inside]
    xyz = np.column_stack([xy, np.sqrt(np.maximum(0, 1 - np.sum(xy**2, axis=1)))])
    distance = np.arccos(np.clip(xyz @ pos.T, -1, 1))
    sigma = np.deg2rad(fwhm_deg) / np.sqrt(8 * np.log(2))
    weights = np.exp(-(distance**2) / (2 * sigma**2))
    weights /= weights.sum(axis=1, keepdims=True)
    return xy, xyz, weights


def electrode_neighbours(positions: np.ndarray) -> list[np.ndarray]:
    pos = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    neighbours = [set() for _ in range(len(pos))]
    for triangle in Delaunay(pos[:, :2]).simplices:
        for first, second in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[0], triangle[2])):
            neighbours[first].add(int(second)); neighbours[second].add(int(first))
    return [np.array(sorted(values), dtype=int) for values in neighbours]


def tfce_score(statistic: np.ndarray, neighbours: list[np.ndarray], extent_power: float = .5,
               height_power: float = 2.) -> np.ndarray:
    def one_sided(values: np.ndarray) -> np.ndarray:
        score = np.zeros_like(values, dtype=float)
        previous = 0.
        for level in np.unique(values[values > 0]):
            active = values >= level
            increment = (level ** (height_power + 1) - previous ** (height_power + 1)) / (height_power + 1)
            remaining = set(np.flatnonzero(active))
            while remaining:
                component, stack = [], [remaining.pop()]
                while stack:
                    node = stack.pop(); component.append(node)
                    linked = [int(x) for x in neighbours[node] if active[x] and x in remaining]
                    for linked_node in linked:
                        remaining.remove(linked_node); stack.append(linked_node)
                score[component] += len(component) ** extent_power * increment
            previous = float(level)
        return score

    return one_sided(statistic) - one_sided(-statistic)


def tangent_maps(data: pd.DataFrame, amp_cols: list[str], ref_cols: list[str]) -> pd.DataFrame:
    out = data.copy()
    maps = out[amp_cols].to_numpy(float)
    refs = out[ref_cols].to_numpy(float)
    valid = np.isfinite(maps).all(axis=1) & np.isfinite(refs).all(axis=1)
    dot = np.full(len(out), np.nan)
    tangent = np.full_like(maps, np.nan)
    if valid.any():
        signed = np.sum(maps[valid] * refs[valid], axis=1)
        aligned = maps[valid] * np.where(signed < 0, -1., 1.)[:, None]
        cosine = np.clip(np.abs(signed), 0., 1.)
        theta = np.arccos(cosine)
        factor = np.divide(theta, np.sin(theta), out=np.ones_like(theta), where=np.abs(theta) > 1e-10)
        tangent[valid] = factor[:, None] * (aligned - cosine[:, None] * refs[valid])
        dot[valid] = theta
        if not np.allclose(np.sum(tangent[valid] * refs[valid], axis=1), 0, atol=2e-6):
            raise ValueError("Tangent maps are not orthogonal to participant reference maps")
    out.loc[:, amp_cols] = tangent
    out["geodesic_theta_rad"] = dot
    return out


def participant_betas(data: pd.DataFrame, amp_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_keys = ["participant", "file_path", "stratum", "backfit_result_source", "template_label", "analysis_role"]
    event_rows = []
    for key, frame in data.groupby(event_keys, observed=True, dropna=False, sort=False):
        time = pd.to_numeric(frame["minutes_to_ictus"], errors="coerce").to_numpy(float)
        values = frame[amp_cols].to_numpy(float)
        theta = pd.to_numeric(frame["geodesic_theta_rad"], errors="coerce").to_numpy(float)
        valid = np.isfinite(time) & np.isfinite(values).all(axis=1) & np.isfinite(theta)
        if valid.sum() < 12 or np.ptp(time[valid]) < 5:
            continue
        design = np.column_stack([np.ones(valid.sum()), time[valid]])
        if np.linalg.matrix_rank(design) < 2:
            continue
        slope = np.linalg.lstsq(design, values[valid], rcond=None)[0][1]
        theta_slope = float(np.linalg.lstsq(design, theta[valid], rcond=None)[0][1])
        event_rows.append(dict(zip(event_keys, key)) |
                          {"term": "linear_minutes_to_ictus", "n_windows": int(valid.sum()),
                           "time_span_min": float(np.ptp(time[valid])),
                           "geodesic_beta_rad_per_min": theta_slope,
                           "mean_window_match_abs_correlation": float(pd.to_numeric(
                               frame.loc[valid, "window_match_abs_correlation"], errors="coerce").mean())} |
                          dict(zip(amp_cols, slope)))
    events = pd.DataFrame(event_rows)
    if events.empty:
        return events, events
    participant_keys = ["participant", "stratum", "backfit_result_source", "template_label", "analysis_role", "term"]
    outcomes = [*amp_cols, "geodesic_beta_rad_per_min"]
    participants = events.groupby(participant_keys, observed=True, dropna=False)[outcomes].mean().reset_index()
    counts = (events.groupby(participant_keys, observed=True, dropna=False)
              .agg(n_events=("file_path", "size"), total_windows=("n_windows", "sum"),
                   median_time_span_min=("time_span_min", "median"),
                   mean_window_match_abs_correlation=("mean_window_match_abs_correlation", "mean")).reset_index())
    participants = participants.merge(counts, on=participant_keys, validate="one_to_one")
    return participants, events


def validate_window_refits(data: pd.DataFrame, amp_cols: list[str], ref_cols: list[str]) -> None:
    """Fail early if the production refit no longer matches the locked estimand."""
    if not data["fit_kind"].eq("window_refit").all():
        raise ValueError("Morphology input must contain window-refitted state maps")
    if not data["prior_source"].eq("participant_full_k6").all():
        raise ValueError("Window refits do not identify the locked participant K=6 prior")
    if not data["prior_mode"].eq("spm_mix_prior_component_means").all():
        raise ValueError("Window refits do not use current SPM_MS component-mean priors")
    if not np.allclose(pd.to_numeric(data["spm_prior_weight"], errors="coerce"), 1):
        raise ValueError("Window refits do not use spm_prior_weight=1")
    if not data["spm_ms_commit"].eq("0dbde1146f6777412dce19edfa4962cd45157fca").all():
        raise ValueError("Window refits were not produced by the pinned SPM_MS revision")
    if not data["K"].eq(6).all():
        raise ValueError("Window refits must all use K=6")
    valid = data[data["fit_status"].eq("ok")].copy()
    if valid.empty:
        raise ValueError("No successful window-refitted maps")
    if (valid["window_peak_count"] < 30).any() or (valid["state_peak_count"] < 5).any():
        raise ValueError("Successful maps violate the locked peak thresholds")
    if not np.allclose(valid["window_duration_s"], 60):
        raise ValueError("Successful maps do not use 60-second windows")
    if ((valid["minutes_to_ictus"] + valid["window_duration_s"] / 120) > 1e-8).any():
        raise ValueError("A successful window extends beyond seizure onset")
    maps = valid[amp_cols].to_numpy(float)
    refs = valid[ref_cols].to_numpy(float)
    if not np.isfinite(maps).all() or not np.allclose(np.linalg.norm(maps, axis=1), 1, atol=2e-6):
        raise ValueError("Successful maps are not finite unit-norm topographies")
    if not np.allclose(maps.mean(axis=1), 0, atol=2e-7):
        raise ValueError("Successful maps are not average-referenced topographies")
    if not np.isfinite(refs).all() or not np.allclose(np.linalg.norm(refs, axis=1), 1, atol=2e-6):
        raise ValueError("Participant reference maps are not finite unit-norm topographies")
    paired = valid[valid["template_label"].isin(["A", "C", "E"])]
    duplicate_match = paired.duplicated(["participant", "file_path", "window_index_from_ictus",
                                         "window_state_index"], keep=False)
    if duplicate_match.any():
        raise ValueError("A/C/E states are not matched one-to-one within a window")
    for _, trajectory in paired.groupby(["participant", "file_path", "template_label"], observed=True):
        trajectory = trajectory.sort_values("window_index_from_ictus")
        if len(trajectory) > 1 and not np.all(np.diff(trajectory["minutes_to_ictus"]) < 0):
            raise ValueError("Window index/timing order is inconsistent")


def t_field(values: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    return np.divide(mean, sd / np.sqrt(n), out=np.zeros_like(mean), where=sd > 0)


def permutation_family(fields: list[dict], participant_order: list[str], permutations: int,
                       rng: np.random.Generator) -> tuple[np.ndarray, list[dict]]:
    index = {participant: i for i, participant in enumerate(participant_order)}
    signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, len(participant_order)))
    max_null = np.zeros(permutations)
    prepared = []
    for field in fields:
        values = field["values"]
        local_signs = signs[:, [index[p] for p in field["participants"]]]
        n = values.shape[0]
        perm_mean = local_signs @ values / n
        sum_sq = np.sum(values**2, axis=0)
        perm_var = np.maximum((sum_sq - n * perm_mean**2) / max(n - 1, 1), 0)
        perm_t = np.divide(perm_mean, np.sqrt(perm_var / n), out=np.zeros_like(perm_mean), where=perm_var > 0)
        observed_t = t_field(values)
        field["observed_t"] = observed_t
        field["p_uncorrected"] = (1 + np.sum(np.abs(perm_t) >= np.abs(observed_t), axis=0)) / (permutations + 1)
        max_null = np.maximum(max_null, np.max(np.abs(perm_t), axis=1))
        prepared.append(field)
    for field in prepared:
        field["p_fwer"] = (1 + np.sum(max_null[:, None] >= np.abs(field["observed_t"]), axis=0)) / (permutations + 1)
    return max_null, prepared


def permutation_tfce_family(fields: list[dict], participant_order: list[str], permutations: int,
                            rng: np.random.Generator, neighbours: list[np.ndarray],
                            extent_power: float, height_power: float) -> tuple[np.ndarray, list[dict]]:
    index = {participant: i for i, participant in enumerate(participant_order)}
    signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, len(participant_order)))
    max_null = np.zeros(permutations)
    prepared = []
    for field in fields:
        values = field["channel_values"]
        local_signs = signs[:, [index[p] for p in field["participants"]]]
        n = values.shape[0]
        perm_mean = local_signs @ values / n
        sum_sq = np.sum(values**2, axis=0)
        perm_var = np.maximum((sum_sq - n * perm_mean**2) / max(n - 1, 1), 0)
        perm_t = np.divide(perm_mean, np.sqrt(perm_var / n), out=np.zeros_like(perm_mean), where=perm_var > 0)
        perm_tfce = np.vstack([tfce_score(row, neighbours, extent_power, height_power) for row in perm_t])
        field["tfce_t"] = t_field(values)
        field["tfce_score"] = tfce_score(field["tfce_t"], neighbours, extent_power, height_power)
        field["p_tfce_uncorrected"] = ((1 + np.sum(np.abs(perm_tfce) >= np.abs(field["tfce_score"]), axis=0)) /
                                         (permutations + 1))
        max_null = np.maximum(max_null, np.max(np.abs(perm_tfce), axis=1))
        prepared.append(field)
    for field in prepared:
        field["p_tfce_fwer"] = ((1 + np.sum(max_null[:, None] >= np.abs(field["tfce_score"]), axis=0)) /
                                  (permutations + 1))
    return max_null, prepared


def plot_family(fields: list[dict], xy: np.ndarray, output: Path, title: str,
                beta_key: str = "beta_grid", p_key: str = "p_fwer",
                colorbar_label: str = "Smoothed morphology beta / min", show_sensors: bool = False) -> None:
    if not fields:
        return
    strata = [s for s in ("Pooled functional", "FS functional", "DD functional")
               if any(f["stratum"] == s for f in fields)]
    columns = ["A", "C", "E"]
    lookup = {(f["stratum"], f["template_label"]): f for f in fields}
    limit = max(np.nanmax(np.abs(f[beta_key])) for f in fields) or 1
    fig, axes = plt.subplots(len(strata), 3, figsize=(11, 3.55 * len(strata)), squeeze=False)
    triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1])
    contour = None
    for r, stratum in enumerate(strata):
        for c, label in enumerate(columns):
            ax = axes[r, c]
            field = lookup.get((stratum, label))
            ax.set_aspect("equal"); ax.axis("off")
            if field is None:
                continue
            contour = ax.tricontourf(triangulation, field[beta_key], levels=21, cmap="RdBu_r", vmin=-limit, vmax=limit)
            significant = field[p_key] < .05
            if show_sensors:
                ax.scatter(xy[:, 0], xy[:, 1], s=10, facecolors="white", edgecolors="0.35", linewidths=.5)
            if significant.any():
                ax.scatter(xy[significant, 0], xy[significant, 1], s=8, c="black")
            ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="black", linewidth=.8))
            ax.plot([-.08, 0, .08], [.99, 1.08, .99], color="black", linewidth=.8)
            ax.set_title(f"{stratum}\nwindow-refitted participant {label}", fontsize=9)
    fig.subplots_adjust(top=.84, bottom=.07, left=.03, right=.88, hspace=.38, wspace=.08)
    if contour is not None:
        fig.colorbar(contour, ax=axes.ravel().tolist(), shrink=.75, label=colorbar_label)
    fig.suptitle(title, y=.98, fontweight="bold")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_analysis(morphology_csv: Path, positions_csv: Path, manifest_csv: Path, out_dir: Path,
                 permutations: int, fwhm_deg: float, grid_size: int, seed: int,
                 tfce_extent_power: float = .5, tfce_height_power: float = 2.) -> dict:
    data_dir, figure_dir = out_dir / "data", out_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True); figure_dir.mkdir(parents=True, exist_ok=True)
    positions = pd.read_csv(positions_csv).sort_values("channel_index")
    header = pd.read_csv(morphology_csv, nrows=0)
    amp_cols = header.filter(regex=r"^amp_").columns.tolist()
    ref_cols = header.filter(regex=r"^ref_amp_").columns.tolist()
    if len(amp_cols) != len(positions):
        raise ValueError(f"Morphology has {len(amp_cols)} amplitudes but geometry has {len(positions)} channels")
    if len(ref_cols) != len(positions):
        raise ValueError(f"Morphology has {len(ref_cols)} reference amplitudes but geometry has {len(positions)} channels")
    expected_amp_cols = [f"amp_{str(label).lower()}" for label in positions["channel_label"]]
    if [column.lower() for column in amp_cols] != expected_amp_cols:
        raise ValueError("Morphology amplitude columns do not match channel geometry order")
    wanted = ["participant", "file_path", "backfit_result_source", "fit_kind", "fit_status", "fit_message", "template_label",
              "prior_source", "prior_mode", "spm_prior_weight", "parent_map_sha256", "spm_ms_commit", "spm_version",
              "spm_commit", "K", "prior_state_index", "window_state_index",
              "template_match_abs_correlation", "window_match_abs_correlation", "window_index_from_ictus",
              "window_start_s", "window_end_s", "minutes_to_ictus", "window_duration_s",
              "window_peak_count", "state_peak_count", *amp_cols, *ref_cols]
    morphology = pd.read_csv(morphology_csv, usecols=wanted, low_memory=False)
    validate_window_refits(morphology, amp_cols, ref_cols)
    subject_map = {p: f"S{i:03d}" for i, p in enumerate(sorted(morphology["participant"].unique()), 1)}
    event_map = {path: f"E{i:04d}" for i, path in enumerate(sorted(morphology["file_path"].unique()), 1)}
    manifest = attach_strata(pd.read_csv(manifest_csv))
    morphology["file_key"] = norm_path(morphology["file_path"])
    morphology = morphology.merge(manifest[["file_key", "stratum"]], on="file_key", how="left", validate="many_to_one")
    morphology["template_label"] = morphology["template_label"].astype(str).str.upper()
    window_export = morphology.assign(subject_id=morphology["participant"].map(subject_map),
                                      event_id=morphology["file_path"].map(event_map)).drop(
                                          columns=["participant", "file_path", "file_key"])
    window_export["fit_message"] = (window_export["fit_message"].astype("string")
                                    .str.replace(r"P_[0-9a-f]{10}", "[participant]", regex=True)
                                    .str.replace(r"[A-Za-z]:[\\/][^\r\n]*", "[path]", regex=True))
    window_export.to_csv(data_dir / "morphology_window_refit_maps.csv", index=False)
    window_refit_rows = len(morphology)
    successful_window_refit_rows = int(morphology["fit_status"].eq("ok").sum())
    window_refit_events = int(morphology["file_path"].nunique())
    window_refit_status_counts = {str(k): int(v) for k, v in morphology["fit_status"].value_counts(dropna=False).items()}
    morphology["analysis_role"] = np.where(
        morphology["backfit_result_source"].eq("participant") & morphology["template_label"].isin(["A", "C", "E"]),
        "co_primary", "exclude")
    morphology = morphology[morphology["stratum"].isin(["FS functional", "DD functional"]) &
                            morphology["analysis_role"].ne("exclude") & morphology["fit_status"].eq("ok") &
                            morphology["template_match_abs_correlation"].ge(.5)].copy()
    morphology = tangent_maps(morphology, amp_cols, ref_cols)
    betas, event_betas = participant_betas(morphology, amp_cols)
    if betas.empty:
        raise ValueError("No participant morphology betas were estimable")
    if (event_betas["n_windows"] < 12).any() or (event_betas["time_span_min"] < 5).any():
        raise ValueError("An exported event slope violates the locked window-count or span threshold")
    if not np.isfinite(betas[amp_cols].to_numpy(float)).all():
        raise ValueError("Participant aggregation produced non-finite morphology betas")
    participant_keys = ["participant", "stratum", "backfit_result_source", "template_label", "analysis_role", "term"]
    beta_cols = [*amp_cols, "geodesic_beta_rad_per_min"]
    expected = event_betas.groupby(participant_keys, observed=True)[beta_cols].mean().reset_index()
    observed = betas[participant_keys + beta_cols].sort_values(participant_keys).reset_index(drop=True)
    expected = expected.sort_values(participant_keys).reset_index(drop=True)
    if not observed[participant_keys].equals(expected[participant_keys]) or not np.allclose(
            observed[beta_cols], expected[beta_cols], atol=1e-12):
        raise ValueError("Participant betas are not equal-weight event averages")
    beta_export = betas.assign(subject_id=betas["participant"].map(subject_map)).drop(columns="participant")
    beta_export.to_csv(data_dir / "morphology_participant_channel_betas.csv", index=False)
    event_export = event_betas.assign(subject_id=event_betas["participant"].map(subject_map),
                                      event_id=event_betas["file_path"].map(event_map)).drop(columns=["participant", "file_path"])
    event_export.to_csv(data_dir / "morphology_event_channel_betas.csv", index=False)
    inference_betas = pd.concat([betas.assign(stratum="Pooled functional"), betas], ignore_index=True)
    geodesic_fields = []
    for (stratum, label), frame in inference_betas.groupby(["stratum", "template_label"], observed=True):
        if len(frame) >= 15:
            geodesic_fields.append({"stratum": stratum, "template_label": label,
                                    "participants": frame["participant"].tolist(),
                                    "values": frame[["geodesic_beta_rad_per_min"]].to_numpy(float),
                                    "n_participants": len(frame)})
    family_specs = (("primary", {"Pooled functional"}),
                    ("sensitivity", {"FS functional", "DD functional"}))
    geodesic_rows = []
    geodesic_rng = np.random.default_rng(seed)
    for family, strata_keep in family_specs:
        family_fields = [f for f in geodesic_fields if f["stratum"] in strata_keep]
        participants = sorted({p for f in family_fields for p in f["participants"]})
        _, tested = permutation_family(family_fields, participants, permutations, geodesic_rng)
        for field in tested:
            geodesic_rows.append({"family": family, "stratum": field["stratum"],
                                  "template_label": field["template_label"],
                                  "n_participants": field["n_participants"],
                                  "geodesic_beta_rad_per_min": float(field["values"].mean()),
                                  "t": float(field["observed_t"][0]),
                                  "p_uncorrected": float(field["p_uncorrected"][0]),
                                  "p_fwer": float(field["p_fwer"][0])})
    pd.DataFrame(geodesic_rows).to_csv(data_dir / "morphology_geodesic_results.csv", index=False)
    position_values = positions[["x", "y", "z"]].to_numpy(float)
    position_values /= np.linalg.norm(position_values, axis=1, keepdims=True)
    xy, xyz, weights = scalp_grid(position_values, fwhm_deg, grid_size)
    neighbours = electrode_neighbours(position_values)
    rng = np.random.default_rng(seed)
    all_fields = []
    keys = ["analysis_role", "stratum", "backfit_result_source", "template_label", "term"]
    for key, frame in inference_betas.groupby(keys, observed=True, sort=False):
        if len(frame) < 15:
            continue
        channel_betas = frame[amp_cols].to_numpy(float)
        grid_betas = channel_betas @ weights.T
        meta = dict(zip(keys, key))
        all_fields.append(meta | {"source": "participant_window_refit", "participants": frame["participant"].tolist(),
                                  "values": grid_betas, "beta_grid": grid_betas.mean(axis=0),
                                  "channel_values": channel_betas, "channel_beta": channel_betas.mean(axis=0),
                                  "n_participants": len(frame)})
    results, peaks = [], []
    for family, strata_keep in family_specs:
        family_fields = [f for f in all_fields if f["stratum"] in strata_keep]
        participants = sorted({p for f in family_fields for p in f["participants"]})
        max_null, family_fields = permutation_family(family_fields, participants, permutations, rng)
        if not np.isfinite(max_null).all():
            raise ValueError("Gaussian maximum-field permutations produced non-finite values")
        threshold = float(np.quantile(max_null, .95))
        for field in family_fields:
            for i in range(len(xy)):
                results.append({"family": family, "analysis_role": field["analysis_role"], "stratum": field["stratum"],
                                "source": field["source"], "template_label": field["template_label"], "term": field["term"],
                                "n_participants": field["n_participants"], "grid_index": i + 1, "x": xy[i, 0], "y": xy[i, 1],
                                "z": xyz[i, 2], "beta_per_min": field["beta_grid"][i], "t": field["observed_t"][i],
                                "p_uncorrected": field["p_uncorrected"][i], "p_fwer": field["p_fwer"][i],
                                "significant_fwer": int(field["p_fwer"][i] < .05), "fwhm_deg": fwhm_deg,
                                "family_max_t_threshold_05": threshold, "n_permutations": permutations})
            peak = int(np.argmax(np.abs(field["observed_t"])))
            nearest = int(np.argmax(xyz[peak] @ (positions[["x", "y", "z"]].to_numpy(float) /
                                                  np.linalg.norm(positions[["x", "y", "z"]].to_numpy(float), axis=1, keepdims=True)).T))
            peaks.append({"family": family, "analysis_role": field["analysis_role"], "stratum": field["stratum"],
                          "source": field["source"], "template_label": field["template_label"], "term": field["term"],
                          "n_participants": field["n_participants"], "peak_beta_per_min": field["beta_grid"][peak],
                          "peak_t": field["observed_t"][peak], "peak_p_uncorrected": field["p_uncorrected"][peak],
                          "peak_p_fwer": field["p_fwer"][peak], "peak_x": xy[peak, 0], "peak_y": xy[peak, 1],
                          "peak_z": xyz[peak, 2],
                          "peak_nearest_channel": positions.iloc[nearest]["channel_label"],
                          "significant_grid_points": int(np.sum(field["p_fwer"] < .05)), "fwhm_deg": fwhm_deg,
                          "family_max_t_threshold_05": threshold, "n_permutations": permutations})
        plot_family(family_fields, xy, figure_dir / f"morphology_rft_{family}_beta_maps.png",
                    "Window-refitted preictal morphology: Gaussian maximum-field sensitivity")
    result_frame, peak_frame = pd.DataFrame(results), pd.DataFrame(peaks)
    result_frame.to_csv(data_dir / "morphology_rft_grid_results.csv", index=False)
    peak_frame.to_csv(data_dir / "morphology_rft_peak_summary.csv", index=False)
    tfce_results, tfce_peaks = [], []
    tfce_rng = np.random.default_rng(seed)
    adjacency_edges = sum(len(values) for values in neighbours) // 2
    for family, strata_keep in family_specs:
        family_fields = [f for f in all_fields if f["stratum"] in strata_keep]
        participants = sorted({p for f in family_fields for p in f["participants"]})
        max_null, family_fields = permutation_tfce_family(family_fields, participants, permutations, tfce_rng,
                                                          neighbours, tfce_extent_power, tfce_height_power)
        if not np.isfinite(max_null).all():
            raise ValueError("TFCE permutations produced non-finite values")
        threshold = float(np.quantile(max_null, .95))
        for field in family_fields:
            for i, channel in positions.reset_index(drop=True).iterrows():
                tfce_results.append({"family": family, "analysis_role": field["analysis_role"],
                                     "stratum": field["stratum"], "source": field["source"],
                                     "template_label": field["template_label"], "term": field["term"],
                                     "n_participants": field["n_participants"], "channel_index": i + 1,
                                     "channel_label": channel["channel_label"], "x": position_values[i, 0],
                                     "y": position_values[i, 1], "z": position_values[i, 2],
                                     "beta_per_min": field["channel_beta"][i], "t": field["tfce_t"][i],
                                     "tfce_score": field["tfce_score"][i],
                                     "p_tfce_uncorrected": field["p_tfce_uncorrected"][i],
                                     "p_tfce_fwer": field["p_tfce_fwer"][i],
                                     "significant_tfce_fwer": int(field["p_tfce_fwer"][i] < .05),
                                     "tfce_extent_power": tfce_extent_power, "tfce_height_power": tfce_height_power,
                                     "adjacency_edges": adjacency_edges,
                                     "family_max_tfce_threshold_05": threshold, "n_permutations": permutations})
            peak = int(np.argmax(np.abs(field["tfce_score"])))
            tfce_peaks.append({"family": family, "analysis_role": field["analysis_role"],
                               "stratum": field["stratum"], "source": field["source"],
                               "template_label": field["template_label"], "term": field["term"],
                               "n_participants": field["n_participants"],
                               "peak_beta_per_min": field["channel_beta"][peak], "peak_t": field["tfce_t"][peak],
                               "peak_tfce_score": field["tfce_score"][peak],
                               "peak_p_tfce_uncorrected": field["p_tfce_uncorrected"][peak],
                               "peak_p_tfce_fwer": field["p_tfce_fwer"][peak],
                               "peak_channel": positions.iloc[peak]["channel_label"],
                               "significant_electrodes": int(np.sum(field["p_tfce_fwer"] < .05)),
                               "tfce_extent_power": tfce_extent_power, "tfce_height_power": tfce_height_power,
                               "adjacency_edges": adjacency_edges,
                               "family_max_tfce_threshold_05": threshold, "n_permutations": permutations})
        plot_family(family_fields, position_values[:, :2], figure_dir / f"morphology_tfce_{family}_beta_maps.png",
                    "Window-refitted preictal morphology: primary electrode-graph TFCE",
                    "channel_beta", "p_tfce_fwer", "Electrode morphology beta / min", True)
    tfce_result_frame, tfce_peak_frame = pd.DataFrame(tfce_results), pd.DataFrame(tfce_peaks)
    tfce_result_frame.to_csv(data_dir / "morphology_tfce_channel_results.csv", index=False)
    tfce_peak_frame.to_csv(data_dir / "morphology_tfce_peak_summary.csv", index=False)
    dictionary_path = data_dir / "data_dictionary.csv"
    dictionary = pd.DataFrame([
        ("morphology_window_refit_maps.csv", "Deidentified rolling-window K=6 refits with current SPM_MS component-mean priors; A/C/E are inferential states."),
        ("morphology_event_channel_betas.csv", "Deidentified event-level tangent-space channel slopes from window-refitted A/C/E maps."),
        ("morphology_participant_channel_betas.csv", "Deidentified participant-level channel slopes after equal averaging of event slopes."),
        ("morphology_geodesic_results.csv", "Polarity-invariant geodesic-deviation slopes with joint maximum-T inference over A/C/E."),
        ("morphology_rft_grid_results.csv", "Gaussian-smoothed scalp-grid beta and t fields with joint permutation maximum-field FWER p-values."),
        ("morphology_rft_peak_summary.csv", "Peak location and corrected inference for each morphology slope field."),
        ("morphology_tfce_channel_results.csv", "Electrode-graph TFCE scores and joint permutation-FWER results for participant morphology slopes."),
        ("morphology_tfce_peak_summary.csv", "Peak electrode and joint TFCE-FWER inference for each morphology slope field."),
        ("morphology_rft_summary.json", "Run-level morphology counts and significance summary."),
    ], columns=["file", "description"])
    dictionary.sort_values("file").to_csv(dictionary_path, index=False)
    variable_path = data_dir / "variable_dictionary.csv"
    variables = pd.DataFrame([
        ("amp_*", "normalised amplitude or tangent amplitude/min", "Window map amplitude in the refit table; tangent-space channel slope per minute in event/participant beta tables."),
        ("ref_amp_*", "normalised amplitude", "Participant full-record reference map used as the component-mean prior and tangent point."),
        ("prior_source", "category", "Identity of the locked parent solution used to regularise each window fit."),
        ("prior_mode", "category", "Current SPM_MS component-mean prior mechanism; no parent maps are appended to observations."),
        ("spm_prior_weight", "precision weight", "SPM_MS component-mean prior weight beta_0; locked to one."),
        ("parent_map_sha256", "SHA-256", "Hash of the six participant reference maps used for the window fit."),
        ("fit_status", "category", "Window/state fit outcome; non-ok rows retain failure/QC information and missing amplitudes."),
        ("window_match_abs_correlation", "correlation", "Absolute spatial correlation between a window-refitted state and its participant prior after one-to-one matching."),
        ("n_windows", "count", "Valid rolling-window maps contributing to an event-level channel slope."),
        ("time_span_min", "minutes", "Time span covered by valid maps in an event-level channel slope."),
        ("geodesic_beta_rad_per_min", "radians/min", "Slope of polarity-invariant angular deviation from the participant reference map."),
        ("beta_per_min", "normalised tangent amplitude/min", "Mean participant tangent-space beta after Gaussian scalp smoothing."),
        ("p_fwer", "probability", "Two-sided permutation maximum-field p-value, jointly corrected over the scalp and all fields in the named family."),
        ("fwhm_deg", "degrees", "Gaussian kernel full width at half maximum on the unit scalp sphere."),
        ("tfce_score", "TFCE statistic", "Signed exact-integral TFCE score from the electrode t field."),
        ("p_tfce_fwer", "probability", "Two-sided maximum-TFCE permutation p-value, jointly corrected over electrodes and fields."),
    ], columns=["variable", "unit_or_type", "description"])
    variables.sort_values("variable").to_csv(variable_path, index=False)
    summary = {"participants": int(betas["participant"].nunique()), "participant_beta_rows": len(betas),
               "window_refit_rows": window_refit_rows, "successful_window_refit_rows": successful_window_refit_rows,
               "window_refit_events": window_refit_events, "window_refit_status_counts": window_refit_status_counts,
               "event_beta_rows": len(event_betas),
               "grid_points": len(xy), "fwhm_deg": fwhm_deg, "permutations": permutations,
               "significant_primary_fields": int((peak_frame["family"].eq("primary") & peak_frame["peak_p_fwer"].lt(.05)).sum()),
               "significant_sensitivity_fields": int((peak_frame["family"].eq("sensitivity") & peak_frame["peak_p_fwer"].lt(.05)).sum()),
               "significant_primary_grid_points": int(result_frame.loc[result_frame["family"].eq("primary"), "significant_fwer"].sum()),
               "significant_sensitivity_grid_points": int(result_frame.loc[result_frame["family"].eq("sensitivity"), "significant_fwer"].sum()),
               "tfce_electrodes": len(positions), "tfce_adjacency_edges": adjacency_edges,
               "tfce_extent_power": tfce_extent_power, "tfce_height_power": tfce_height_power,
               "tfce_significant_primary_fields": int((tfce_peak_frame["family"].eq("primary") &
                                                        tfce_peak_frame["peak_p_tfce_fwer"].lt(.05)).sum()),
               "tfce_significant_sensitivity_fields": int((tfce_peak_frame["family"].eq("sensitivity") &
                                                            tfce_peak_frame["peak_p_tfce_fwer"].lt(.05)).sum()),
               "tfce_significant_primary_electrodes": int(tfce_result_frame.loc[tfce_result_frame["family"].eq("primary"),
                                                                                 "significant_tfce_fwer"].sum()),
               "tfce_significant_sensitivity_electrodes": int(tfce_result_frame.loc[
                   tfce_result_frame["family"].eq("sensitivity"), "significant_tfce_fwer"].sum())}
    (data_dir / "morphology_rft_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def self_check() -> None:
    pos = np.array([[-.7, 0, .7], [.7, 0, .7], [0, -.7, .7], [0, .7, .7], [0, 0, 1.]])
    xy, _, weights = scalp_grid(pos, 45, 17)
    assert len(xy) > 20 and np.allclose(weights.sum(axis=1), 1)
    rng = np.random.default_rng(3)
    values = rng.normal(0, .05, (24, len(xy))) + weights[:, 4]
    field = {"values": values, "participants": [f"S{i}" for i in range(24)]}
    _, tested = permutation_family([field], field["participants"], 511, rng)
    assert tested[0]["p_fwer"].min() < .05
    neighbours = [np.array([1]), np.array([0, 2]), np.array([1])]
    assert tfce_score(np.array([3., 3., 0.]), neighbours)[0] > tfce_score(np.array([3., 0., 0.]), neighbours)[0] > 0
    channel_values = rng.normal(0, .05, (24, 3)); channel_values[:, :2] += 1
    tfce_field = {"channel_values": channel_values, "participants": [f"S{i}" for i in range(24)]}
    _, tfce_tested = permutation_tfce_family([tfce_field], tfce_field["participants"], 511, rng, neighbours, .5, 2.)
    assert tfce_tested[0]["p_tfce_fwer"].min() < .05
    time = np.linspace(-10, 0, 21)
    synthetic = pd.DataFrame({"participant": "P1", "file_path": "event1", "stratum": "FS functional",
                              "backfit_result_source": "participant", "template_label": "C",
                              "analysis_role": "co_primary", "minutes_to_ictus": time,
                              "window_match_abs_correlation": .8,
                              "geodesic_theta_rad": .2 + .01 * time,
                              "amp_a": .1 + .02 * time, "amp_b": -.1 - .03 * time})
    participant, event = participant_betas(synthetic, ["amp_a", "amp_b"])
    assert len(event) == 1 and len(participant) == 1
    assert abs(event.iloc[0]["amp_a"] - .02) < 1e-12 and abs(event.iloc[0]["amp_b"] + .03) < 1e-12
    assert abs(event.iloc[0]["geodesic_beta_rad_per_min"] - .01) < 1e-12
    reference = np.array([[-1., 0., 1.]]) / np.sqrt(2)
    direction = np.array([[1., -2., 1.]]) / np.sqrt(6)
    angles = np.array([.1, .2, .3])
    maps = np.cos(angles)[:, None] * reference + np.sin(angles)[:, None] * direction
    maps[1] *= -1
    topo = pd.DataFrame(maps, columns=["amp_1", "amp_2", "amp_3"])
    topo[["ref_amp_1", "ref_amp_2", "ref_amp_3"]] = np.repeat(reference, 3, axis=0)
    tangent = tangent_maps(topo, ["amp_1", "amp_2", "amp_3"], ["ref_amp_1", "ref_amp_2", "ref_amp_3"])
    assert np.allclose(tangent["geodesic_theta_rad"], angles)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--morphology-csv", type=Path)
    parser.add_argument("--positions-csv", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--fwhm-deg", type=float, default=45.0)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--tfce-extent-power", type=float, default=.5)
    parser.add_argument("--tfce-height-power", type=float, default=2.)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(); print("self-check passed"); return
    if not all((args.morphology_csv, args.positions_csv, args.manifest, args.out_dir)):
        parser.error("--morphology-csv, --positions-csv, --manifest and --out-dir are required")
    print(json.dumps(run_analysis(args.morphology_csv, args.positions_csv, args.manifest, args.out_dir,
                                  args.permutations, args.fwhm_deg, args.grid_size, args.seed,
                                  args.tfce_extent_power, args.tfce_height_power), indent=2))


if __name__ == "__main__":
    main()
