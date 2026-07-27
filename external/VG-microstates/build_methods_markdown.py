#!/usr/bin/env python3
"""Generate METHODS.md from the microstate simulation config."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = {
    "paths": {
        "template_file": "MetaMaps_2023_06.set",
        "simulation_output_dir": "outputs/simulations",
        "koenig_code_dir": "_internal/matlab/Koenig_code",
        "spm_mixture_paths": ["~/spm/toolbox/mixture", "~/Downloads/spm/toolbox/mixture"],
    },
    "simulation": {
        "out_dir": "outputs/simulations",
        "reps": 8,
        "K_true_vals": [4, 5, 6, 7],
        "SNR_dbs": [-9, -3, 1, 0, 1, 3],
        "K_candidates": list(range(2, 11)),
        "duration_s": 300,
        "sfreq": 250,
        "n_workers": 12,
        "montages": ["full", "10-20-20", "10-20-12"],
        "overlap_probs": [0, 0.5, 1.0],
        "overlap_ms_range": [10, 40],
        "overlap_strength": 0.5,
        "compute_backfit_diagnostics": True,
        "save_backfit_details": True,
        "backfit_downsample_factor": 5,
        "template_alignment_strong_threshold": 0.5,
        "ecological_profile": True,
        "randomize_true_templates": True,
        "true_template_pool_K": 7,
        "clean_sanity_profile": True,
        "clean_sanity_snr_db_threshold": 40,
        "validate_simulation": False,
        "preprocessing": {
            "apply_average_reference": False,
            "spatial_filter": "none",
            "reject_gfp_peak_outliers": False,
        },
        "methods": ["spm_vb", "kmeans_koenig"],
        "criteria": [
            "silhouette",
            "free_energy",
            "log_likelihood",
            "bic",
            "icl",
            "free_energy_elbow",
            "gev",
            "calinski_harabasz_score",
            "covariance",
            "covariance_elbow",
            "elbow_sil_combined",
            "free_energy_covariance",
        ],
        "microstate_amplitude_uv": 30,
        "background_noise_rms_uv": 10,
        "mean_dur_ms": 80,
        "map_jitter_fraction": 0.15,
        "gfp_envelope_eta": 0.2,
        "inject_artifacts": True,
        "real_background_file": "",
    },
}

PREPROCESS_DEFAULTS = {
    "apply_average_reference": False,
    "filter_band": [2, 20],
    "spatial_filter": "none",
    "spatial_filter_neighbours": 6,
    "spatial_filter_strength": 1,
    "gfp_peak_min_distance": 3,
    "gfp_peak_threshold_schedule": [0.50, 0.60, 0.70, 0.80, 0.90],
    "reject_gfp_peak_outliers": False,
    "gfp_outlier_mad_multiplier": 6,
    "gfp_outlier_upper_quantile": 0.995,
    "min_peak_count_after_gfp_rejection": 3,
}

SUPPORTED_CRITERIA = {
    "spm_vb": [
        "silhouette",
        "free_energy",
        "log_likelihood",
        "ll",
        "bic",
        "icl",
        "free_energy_elbow",
        "gev",
        "gfp",
        "calinski_harabasz_score",
        "covariance",
        "covariance_elbow",
        "elbow_sil_combined",
        "free_energy_covariance",
    ],
    "kmeans_koenig": ["silhouette"],
}

METHOD_NAMES = {
    "spm_vb": "VB-GMM",
    "kmeans_koenig": "K-Means",
}

MONTAGE_COUNTS = {
    "full": "template-dependent full MetaMaps montage; current generator is written for 71 channels",
    "10-20-20": "19 active 10-20 leads: Fp1, Fp2, F7, F3, Fz, F4, F8, T7/T3, C3, Cz, C4, T8/T4, P7/T5, P3, Pz, P4, P8/T6, O1, O2",
    "10-20-12": "12 clinical leads: Fp1, Fp2, F3, F4, Fz, C3, C4, P3, P4, Pz, O1, O2",
}


def recursive_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = recursive_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        user_config = json.load(handle)
    return recursive_merge(DEFAULT_CONFIG, user_config)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:g}"
    return str(value)


def fmt_list(values: Any, unit: str = "") -> str:
    vals = as_list(values)
    if not vals:
        return "none"
    suffix = f" {unit}" if unit else ""
    return ", ".join(f"{fmt_value(v)}{suffix}" for v in vals)


def fmt_range(values: Any, unit: str = "") -> str:
    vals = as_list(values)
    if len(vals) >= 2:
        suffix = f" {unit}" if unit else ""
        return f"{fmt_value(vals[0])}-{fmt_value(vals[-1])}{suffix}"
    return fmt_list(vals, unit)


def fmt_bool(value: Any) -> str:
    return "enabled" if bool(value) else "disabled"


def method_criteria_map(methods: list[str], requested: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for method in methods:
        supported = SUPPORTED_CRITERIA.get(method, [])
        out[method] = [criterion for criterion in requested if criterion in supported]
    return out


def count_downsampled(n_samples: int, factor: int) -> int:
    if n_samples <= 0:
        return 0
    factor = max(1, int(factor))
    return ((n_samples - 1) // factor) + 1


def bullet(lines: list[str], level: int, text: str = "") -> None:
    prefix = "  " * level + "- "
    lines.append(prefix + text)


def build_methods_markdown(config: dict[str, Any], config_path: Path) -> str:
    paths = config.get("paths", {})
    sim = config.get("simulation", {})
    preproc = recursive_merge(PREPROCESS_DEFAULTS, sim.get("preprocessing", {}))

    reps = as_list(sim.get("rep_vals")) or list(range(1, int(sim["reps"]) + 1))
    k_true_vals = as_list(sim["K_true_vals"])
    snrs = as_list(sim["SNR_dbs"])
    overlaps = as_list(sim["overlap_probs"])
    k_candidates = as_list(sim["K_candidates"])
    montages = [str(x) for x in as_list(sim["montages"])]
    methods = [str(x) for x in as_list(sim.get("methods", ["spm_vb", "kmeans_koenig"]))]
    requested_criteria = [str(x) for x in as_list(sim.get("criteria", []))]
    criteria_by_method = method_criteria_map(methods, requested_criteria)
    n_pairs = sum(len(v) for v in criteria_by_method.values())

    duration_s = float(sim["duration_s"])
    sfreq = float(sim["sfreq"])
    n_samples = int(round(duration_s * sfreq))
    n_eeg_conditions = len(reps) * len(k_true_vals) * len(snrs) * len(overlaps)
    n_method_fits = n_eeg_conditions * len(montages) * len(methods)
    n_result_rows = n_eeg_conditions * len(montages) * n_pairs
    n_k_candidate_rows = n_eeg_conditions * len(montages) * len(methods) * len(k_candidates)
    backfit_factor = int(sim.get("backfit_downsample_factor", 5))
    backfit_samples = count_downsampled(n_samples, backfit_factor)

    lines: list[str] = [
        "# Methods",
        "",
        "- This document is generated by `build_methods_markdown.py` from `" + str(config_path) + "`.",
        "- It describes the current implementation in `simulated_ms_retrieval_experiment.m`, `generate_microstate_eeg.m`, `fit_microstate_spm_vb.m`, `fit_microstate_kmeans_koenig.m`, and the shared utility functions.",
        "- Regenerate it after changing the simulation config with: `python3 build_methods_markdown.py --config-file config/microstate_config.json --output-file METHODS.md`.",
        "",
        "## 1. VB-GMM Microstate Method",
    ]

    bullet(lines, 0, "Implementation and inputs")
    bullet(lines, 1, "The VB-GMM fitting entry point is `fit_microstate_spm_vb.m`.")
    bullet(lines, 1, "The method is run for each candidate state count `K` in `" + fmt_list(k_candidates) + "`.")
    bullet(lines, 1, "The simulation pipeline calls the method once per EEG/montage with the default selection criterion `elbow_sil_combined`, but it saves all candidate-K metric arrays and then reapplies each supported criterion downstream.")
    bullet(lines, 1, "SPM's mixture toolbox must expose `spm_mix`; the pipeline searches an explicit `spm_path`, the `SPM_MIXTURE_PATH` and `SPM_PATH` environment variables, the configured paths `" + fmt_list(paths.get("spm_mixture_paths")) + "`, and paths inferred from `spm.m`.")
    bullet(lines, 1, "The fitted data are the same GFP-peak maps used by all compared methods, not a method-specific resampling of the EEG.")
    bullet(lines, 0, "Shared preprocessing before VB fitting")
    bullet(lines, 1, "The noisy montage-specific EEG is read from `Sim.X_noisy`.")
    bullet(lines, 1, "Average reference is `" + fmt_bool(preproc["apply_average_reference"]) + "` for the simulation run.")
    bullet(lines, 1, "Spatial filtering is `" + str(preproc["spatial_filter"]) + "` with `" + str(preproc["spatial_filter_neighbours"]) + "` neighbours and strength `" + fmt_value(preproc["spatial_filter_strength"]) + "` if a spatial filter is enabled.")
    bullet(lines, 1, "The fitting signal is zero-phase FFT bandpass filtered over `" + fmt_range(preproc["filter_band"], "Hz") + "`.")
    bullet(lines, 1, "Global field power is computed as the across-channel root-mean-square of the demeaned signal at each sample.")
    bullet(lines, 1, "GFP peaks are local maxima separated by at least `" + fmt_value(preproc["gfp_peak_min_distance"]) + "` samples; quantile thresholds are tried in order `" + fmt_list(preproc["gfp_peak_threshold_schedule"]) + "` until at least `" + fmt_value(preproc["min_peak_count_after_gfp_rejection"]) + "` peaks remain.")
    bullet(lines, 1, "GFP peak outlier rejection is `" + fmt_bool(preproc["reject_gfp_peak_outliers"]) + "`; if enabled it uses median + `" + fmt_value(preproc["gfp_outlier_mad_multiplier"]) + "` scaled MAD with a `" + fmt_value(preproc["gfp_outlier_upper_quantile"]) + "` quantile fallback.")
    bullet(lines, 1, "Peak maps are row-demeaned and normalized to unit Euclidean norm before clustering; polarity is therefore represented by map direction rather than amplitude.")
    bullet(lines, 0, "Feature space used by `spm_mix`")
    bullet(lines, 1, "The normalized peak maps are first reduced by PCA.")
    bullet(lines, 1, "The PCA dimensionality is the smallest dimension explaining at least 99.9% of variance, capped by estimated rank, `N - 1`, `D - 1`, and 8 dimensions, where `N` is the number of peak maps and `D` is the number of channels.")
    bullet(lines, 1, "PCA scores are standardized dimension-wise before the polarity-invariant embedding.")
    bullet(lines, 1, "To make `x` and `-x` equivalent, each PCA score vector is normalized and represented as the upper triangle of the outer product `x x'`.")
    bullet(lines, 1, "Off-diagonal outer-product entries are multiplied by `sqrt(2)` so Euclidean distance in the vectorized embedding preserves the Frobenius inner product.")
    bullet(lines, 1, "Constant outer-product dimensions are dropped and retained dimensions are standardized.")
    bullet(lines, 0, "VB mixture fit")
    bullet(lines, 1, "For each `K`, VB-GMM calls `spm_mix(features, K, 0)`; the implementation uses the isotropic covariance mode to avoid singular covariance failures in the projective feature space.")
    bullet(lines, 1, "A candidate fit is treated as invalid if SPM does not return a finite non-zero free energy, state means, and finite covariance entries.")
    bullet(lines, 1, "For a valid candidate, peak maps are assigned to mixture components by maximizing `log(prior_k) + log N(x | mean_k, covariance_k)`.")
    bullet(lines, 1, "Topographic cluster centers are recovered in the original normalized scalp-map space, not in the PCA/projective space.")
    bullet(lines, 1, "Center recovery uses a Koenig-style polarity-invariant update: for each assigned cluster, the first eigenvector of the cluster map cross-product matrix is used as the center, then row demeaned and unit-normalized.")
    bullet(lines, 1, "Recovered centers are refined for up to 25 iterations by reassigning maps to the center with the largest absolute correlation, repairing empty clusters where needed.")
    bullet(lines, 1, "Potential polarity-duplicate centers are flagged when the absolute center correlation is at least 0.85.")
    bullet(lines, 0, "VB-GMM evidence and per-K quantities")
    bullet(lines, 1, "The raw SPM free energy from `spm_mix` is stored separately as `spm_pca_free_energy_vals`.")
    bullet(lines, 1, "The primary `free_energy_vals` used for model selection are recomputed in a full sensor-space, polarity-invariant outer-product embedding.")
    bullet(lines, 1, "The matrix-space evidence model uses a diagonal Gaussian model aligned to the final labels for each K.")
    bullet(lines, 1, "The matrix-space log likelihood is the summed log marginal likelihood over all peak maps.")
    bullet(lines, 1, "The number of parameters is `(K - 1) + 2 * K * intrinsic_dim`, where `intrinsic_dim` is the estimated rank of the scalp-map matrix minus one, lower-bounded at 1.")
    bullet(lines, 1, "The matrix-space BIC-form evidence is `log_likelihood - 0.5 * n_parameters * log(n_peak_maps)` and is used as `free_energy_vals` and `bic_vals`.")
    bullet(lines, 1, "ICL is computed as BIC minus the assignment entropy.")
    bullet(lines, 1, "GEV is computed from absolute map-center similarity weighted by original, non-normalized GFP peak map power.")
    bullet(lines, 1, "Silhouette is the mean sample silhouette using the polarity-insensitive distance `1 - |cosine_similarity|`.")
    bullet(lines, 1, "Within-cluster sum of squares is the sum of squared `1 - |correlation|` distances to the assigned topographic center.")
    bullet(lines, 1, "Calinski-Harabasz score is computed in the fitted feature space.")
    bullet(lines, 1, "SPM covariance summaries include mean/median covariance trace, mean/median log determinant, and mean log determinant per feature dimension.")
    bullet(lines, 0, "VB-GMM model-selection criteria")
    bullet(lines, 1, "The benchmark requests these VB-GMM criteria: `" + fmt_list(criteria_by_method.get("spm_vb", [])) + "`.")
    bullet(lines, 1, "`silhouette`: selects the maximum polarity-insensitive silhouette; when more than four valid K values are available, endpoint K values are excluded from this selector.")
    bullet(lines, 1, "`free_energy`: selects the maximum matrix-space BIC/free-energy value.")
    bullet(lines, 1, "`log_likelihood`: selects the maximum matrix-space log likelihood.")
    bullet(lines, 1, "`bic`: selects the maximum BIC-form evidence.")
    bullet(lines, 1, "`icl`: selects the maximum ICL value after entropy penalty.")
    bullet(lines, 1, "`free_energy_elbow`: normalizes K and free energy to 0-1 and selects the K with the largest perpendicular distance from the line between the first and last valid free-energy points.")
    bullet(lines, 1, "`gev`: selects the maximum global explained variance.")
    bullet(lines, 1, "`calinski_harabasz_score`: selects the maximum Calinski-Harabasz score.")
    bullet(lines, 1, "`covariance`: selects the tightest mixture covariance using the first non-degenerate covariance metric in this order: log determinant per dimension, log determinant, trace mean, trace median.")
    bullet(lines, 1, "`covariance_elbow`: applies the same endpoint-line elbow rule to the decreasing covariance curve.")
    bullet(lines, 1, "`elbow_sil_combined`: first finds the free-energy elbow K, then scores each K as `0.6 * exp(-abs(K - K_elbow)) + 0.4 * ((silhouette + 1) / 2)`.")
    bullet(lines, 1, "`free_energy_covariance`: scores each K as `0.45 * free_energy_elbow_norm + 0.35 * covariance_elbow_norm + 0.15 * covariance_tightness_norm + 0.05 * free_energy_norm`.")

    lines.extend(["", "## 2. K-Means Parameterisation"])
    bullet(lines, 0, "Implementation and dependency")
    bullet(lines, 1, "The traditional baseline is implemented in `fit_microstate_kmeans_koenig.m`.")
    bullet(lines, 1, "It calls Thomas Koenig's MICROSTATELAB `eeg_kMeans` implementation from the configured directory `" + str(paths.get("koenig_code_dir")) + "`.")
    bullet(lines, 1, "The expected external files are `eeg_kMeans.m`, `L2NormDim.m`, `mywaitbar.m`, and `popFitMSMaps.m`.")
    bullet(lines, 0, "Input maps and candidate K values")
    bullet(lines, 1, "K-Means uses exactly the same preprocessed GFP peak maps as VB-GMM.")
    bullet(lines, 1, "The candidate K set is `" + fmt_list(k_candidates) + "`.")
    bullet(lines, 1, "The method is fit once for every candidate K; model-selection criteria are applied after all candidates are available.")
    bullet(lines, 0, "K-Means fitting parameters")
    bullet(lines, 1, "`eeg_kMeans` is called as `eeg_kMeans(maps_norm, K, 20, n_maps, flags)`.")
    bullet(lines, 1, "The third argument sets 20 random restarts for each candidate K.")
    bullet(lines, 1, "`n_maps` is passed as the maximum number of maps, so all retained GFP peak maps are available to the Koenig routine.")
    bullet(lines, 1, "`flags` is the empty string; in this implementation that is the polarity-insensitive resting-state microstate mode based on absolute map correlation.")
    bullet(lines, 1, "For each K the function stores the returned centers, labels, loading vector, and per-cluster explained variance.")
    bullet(lines, 0, "K-Means scoring and selected benchmark criterion")
    bullet(lines, 1, "GEV is computed with original, non-normalized peak maps and absolute correlation to each center.")
    bullet(lines, 1, "Silhouette uses the same polarity-insensitive distance as VB-GMM: `1 - |cosine_similarity|`.")
    bullet(lines, 1, "Within-cluster sum of squares is the sum of squared `1 - |correlation|` distances for assigned peak maps.")
    bullet(lines, 1, "The current benchmark exposes only `" + fmt_list(criteria_by_method.get("kmeans_koenig", [])) + "` for traditional K-Means; unsupported requested criteria are skipped by design.")
    bullet(lines, 1, "In the simulation result extraction helper, K-Means silhouette selection excludes endpoint K values when more than four K candidates are present; with the current candidates this means the final selector searches K = `" + fmt_list(k_candidates[1:-1]) + "`.")
    bullet(lines, 1, "K-Means map recovery is evaluated with the same partial topographic alignment routine used for VB-GMM.")

    lines.extend(["", "## 3. Simulated EEG Generation"])
    bullet(lines, 0, "Generation entry point and record dimensions")
    bullet(lines, 1, "Synthetic EEG is generated by `generate_microstate_eeg.m` once per replicate, true K, SNR, and overlap condition.")
    bullet(lines, 1, "Each generated record lasts `" + fmt_value(duration_s) + "` s at `" + fmt_value(sfreq) + "` Hz, giving `" + f"{n_samples:,}" + "` samples before montage reduction.")
    bullet(lines, 1, "The full channel geometry and canonical maps are read from `" + str(paths.get("template_file")) + "`.")
    bullet(lines, 1, "The full montage is generated first; reduced montages are made by channel selection from the same generated signal.")
    bullet(lines, 0, "True map source and template selection")
    bullet(lines, 1, "Canonical MetaMaps templates are loaded with `load_metamaps_templates`, which returns one zero-mean/unit-norm map per state and alphabetically sorted labels.")
    bullet(lines, 1, "The template pool K is `" + fmt_value(sim.get("true_template_pool_K")) + "`.")
    bullet(lines, 1, "Random true-template selection is `" + fmt_bool(sim.get("randomize_true_templates")) + "`.")
    bullet(lines, 1, "When randomization is enabled, each replicate/K pair draws `K_true` sorted template indices from the pool using seed `42 + rep * 1000 + K_true * 100`.")
    bullet(lines, 1, "The selected unit-norm templates are scaled to `" + fmt_value(sim.get("microstate_amplitude_uv")) + "` uV before temporal generation.")
    bullet(lines, 0, "State sequence and clean microstate signal")
    bullet(lines, 1, "The random generator seed for each generated EEG is `42 + rep * 1000 + K_true * 100 + round((SNR_dB + 10) * 10)`.")
    bullet(lines, 1, "The initial state is sampled uniformly from the true states.")
    bullet(lines, 1, "State dwell times are geometric with switch probability `1 / mean_dur_samples`; the configured mean dwell is `" + fmt_value(sim.get("mean_dur_ms")) + "` ms.")
    bullet(lines, 1, "The next state is sampled uniformly from all states except the current state.")
    bullet(lines, 1, "Each segment uses the selected canonical map plus smooth spatial jitter.")
    bullet(lines, 1, "Spatial jitter is drawn as `map_jitter_fraction * microstate_amplitude`, with the current fraction `" + fmt_value(sim.get("map_jitter_fraction")) + "` and an RBF spatial kernel length scale of 0.25 over normalized channel coordinates.")
    bullet(lines, 1, "Clean segment activity is the segment topography multiplied by a segment GFP envelope.")
    bullet(lines, 0, "GFP envelope profile")
    bullet(lines, 1, "The ecological profile is `" + fmt_bool(sim.get("ecological_profile")) + "`.")
    bullet(lines, 1, "With the ecological profile enabled, each segment contains smooth GFP packets with approximately `1 + Poisson(max(0, 3 * dwell_seconds - 0.4))` packet peaks.")
    bullet(lines, 1, "Each packet is a Gaussian burst with randomized center, 18-43 ms width, and log-normal amplitude.")
    bullet(lines, 1, "A sinusoidal alpha-range modulation between 8 and 12 Hz is applied, a sinusoidal taper lowers activity at segment boundaries, and the segment polarity sign can flip.")
    bullet(lines, 1, "The clean sanity profile flag is `" + fmt_bool(sim.get("clean_sanity_profile")) + "`, but it activates only when overlap probability is 0 and SNR is at least `" + fmt_value(sim.get("clean_sanity_snr_db_threshold")) + "` dB.")
    bullet(lines, 1, "With the current SNR set `" + fmt_list(snrs, "dB") + "`, the clean sanity profile does not activate.")
    bullet(lines, 0, "Temporal overlap model")
    bullet(lines, 1, "Overlap probabilities are `" + fmt_list(overlaps) + "` per state boundary.")
    bullet(lines, 1, "The configured overlap window range is `" + fmt_range(sim.get("overlap_ms_range"), "ms") + "` with strength `" + fmt_value(sim.get("overlap_strength")) + "`.")
    bullet(lines, 1, "If a boundary is selected for overlap, the implementation uses the maximum feasible overlap up to the configured upper bound after enforcing the lower-bound feasibility check; it is not a uniform draw across the range.")
    bullet(lines, 1, "The next state fades into the tail of the current segment and the previous state fades out across the start of the next segment.")
    bullet(lines, 1, "State weights are mixed with the same cross-fade weights, clipped non-negative, and normalized to sum to one per sample.")
    bullet(lines, 1, "Dominant true labels `z_true` are the argmax of the true state-weight matrix after overlap mixing.")
    bullet(lines, 0, "Background noise, SNR, and artifacts")
    bullet(lines, 1, "The target background RMS before SNR scaling is `" + fmt_value(sim.get("background_noise_rms_uv")) + "` uV.")
    bullet(lines, 1, "If no real background file is supplied, the ecological synthetic background is used.")
    bullet(lines, 1, "The synthetic background combines channel-wise pink noise, 8-13 Hz alpha-band noise, and 1-4 Hz slow noise.")
    bullet(lines, 1, "Background channels are spatially mixed with covariance `exp(-distance^2 / (2 * 0.35^2)) + 0.03 I`, demeaned per sample, and normalized to the target RMS.")
    bullet(lines, 1, "Noise is scaled to the requested SNR by `sqrt((10^(-SNR_dB/10) * signal_power) / noise_power)` and added to the clean EEG.")
    bullet(lines, 1, "Artifact injection is `" + fmt_bool(sim.get("inject_artifacts")) + "` by default.")
    bullet(lines, 1, "If `artefact_template.set` is found, the generator injects one 0.5 s artifact snippet per approximately 10 s of simulated EEG; each snippet is randomly positioned, faded in/out, and scaled to 50-100% of the current signal standard deviation.")
    bullet(lines, 1, "If the artifact template cannot be loaded, generation warns and continues without artifact snippets.")
    bullet(lines, 0, "Montage reduction")
    for montage in montages:
        bullet(lines, 1, "`" + montage + "`: " + MONTAGE_COUNTS.get(montage, "custom montage handled by `select_montage_subset.m`"))
    bullet(lines, 1, "Montage matching is case-insensitive and accepts T3/T4/T5/T6 aliases for T7/T8/P7/P8 in the 10-20 montage.")
    bullet(lines, 1, "The same clean EEG, noisy EEG, true maps, channel locations, and labels are reduced together so each montage preserves the correct ground truth.")

    lines.extend(["", "## 4. Simulation Setup and Metrics"])
    bullet(lines, 0, "Experimental grid")
    bullet(lines, 1, "Output root: `" + str(sim.get("out_dir", paths.get("simulation_output_dir", "outputs/simulations"))) + "`.")
    bullet(lines, 1, "Replicates: `" + fmt_list(reps) + "` (`" + f"{len(reps):,}" + "` replicate values).")
    bullet(lines, 1, "True K values: `" + fmt_list(k_true_vals) + "`.")
    bullet(lines, 1, "SNR values: `" + fmt_list(snrs, "dB") + "`.")
    bullet(lines, 1, "Overlap probabilities: `" + fmt_list(overlaps) + "`.")
    bullet(lines, 1, "K candidates for fitting: `" + fmt_list(k_candidates) + "`.")
    bullet(lines, 1, "Montages: `" + fmt_list(montages) + "`.")
    bullet(lines, 1, "Methods: `" + fmt_list([METHOD_NAMES.get(m, m) for m in methods]) + "`.")
    bullet(lines, 1, "Parallel worker target: `" + fmt_value(sim.get("n_workers")) + "`.")
    bullet(lines, 0, "Derived run counts for the current config")
    bullet(lines, 1, "Generated EEG conditions: `" + f"{n_eeg_conditions:,}" + "` = `" + f"{len(reps):,}" + "` replicates x `" + f"{len(k_true_vals):,}" + "` true-K values x `" + f"{len(snrs):,}" + "` SNR values x `" + f"{len(overlaps):,}" + "` overlap settings.")
    bullet(lines, 1, "Method fits: `" + f"{n_method_fits:,}" + "` = generated EEG conditions x `" + f"{len(montages):,}" + "` montages x `" + f"{len(methods):,}" + "` methods.")
    bullet(lines, 1, "Supported method-criterion pairs: `" + f"{n_pairs:,}" + "`.")
    for method, criteria in criteria_by_method.items():
        bullet(lines, 2, "`" + method + "` contributes `" + f"{len(criteria):,}" + "` criterion rows per EEG/montage: `" + fmt_list(criteria) + "`.")
    bullet(lines, 1, "Expected comparison-result rows if all fits succeed: `" + f"{n_result_rows:,}" + "` = generated EEG conditions x montages x supported method-criterion pairs.")
    bullet(lines, 1, "Expected candidate-K rows if all fits succeed: `" + f"{n_k_candidate_rows:,}" + "` = generated EEG conditions x montages x methods x `" + f"{len(k_candidates):,}" + "` K candidates.")
    bullet(lines, 0, "Pipeline execution")
    bullet(lines, 1, "For each EEG condition, the full generated EEG is created once and then reused across all requested montages.")
    bullet(lines, 1, "For each montage, each method is fit once across all K candidates.")
    bullet(lines, 1, "After fitting, each valid method-criterion pair selects K from the stored per-K arrays and extracts the selected topographic solution.")
    bullet(lines, 1, "Checkpointing is enabled by default in the MATLAB pipeline; checkpoints are keyed by condition and invalidated when the non-runtime configuration changes.")
    bullet(lines, 1, "Simulation QC validation is `" + fmt_bool(sim.get("validate_simulation")) + "`.")
    bullet(lines, 1, "JSON topography export is enabled by the MATLAB default unless explicitly disabled at runtime.")
    bullet(lines, 0, "Main output files")
    bullet(lines, 1, "`comparison_results.csv`: one row per EEG condition, montage, method, and supported criterion.")
    bullet(lines, 1, "`k_candidate_metrics.csv`: one row per candidate K for each EEG condition, montage, and method.")
    bullet(lines, 1, "`k_selection_summary_by_method_criterion.csv`: grouped mean and standard deviation of K recovery by method and criterion.")
    bullet(lines, 1, "`analysis_info.json`: run metadata, config, channel labels, channel positions, and the number of generated EEG conditions.")
    bullet(lines, 1, "`microstates_json/`: selected maps and metadata for plotting when JSON export is enabled.")
    bullet(lines, 1, "`backfit_diagnostics/`: per-solution backfit diagnostic `.mat` files when backfit diagnostics and detail saving are enabled.")
    bullet(lines, 0, "K-selection and map-recovery metrics in `comparison_results.csv`")
    bullet(lines, 1, "`K_estimated`: selected K for the method/criterion row.")
    bullet(lines, 1, "`K_correct`: exact K recovery indicator, `K_estimated == K_true`.")
    bullet(lines, 1, "`K_error`: absolute K error, `abs(K_true - K_estimated)`.")
    bullet(lines, 1, "`K_gap`: signed K gap, `K_true - K_estimated`.")
    bullet(lines, 1, "`n_maps`: number of GFP peak maps retained for fitting.")
    bullet(lines, 1, "`n_matched`: number of estimated maps matched to true maps by rectangular Hungarian assignment.")
    bullet(lines, 1, "Topographic similarity for matching is absolute cosine similarity by default.")
    bullet(lines, 1, "The alignment threshold is 0 in `microstate_partial_alignment`, so every optimal non-dummy match is retained.")
    bullet(lines, 1, "`mean_recovery_matched`: mean absolute cosine similarity across matched pairs.")
    bullet(lines, 1, "`mean_recovery_padded`: matched similarities padded with zeros to `max(K_true, K_estimated)` before averaging, penalizing missing or extra states.")
    bullet(lines, 1, "`sensitivity`: `n_matched / K_true`.")
    bullet(lines, 1, "`precision`: `n_matched / K_estimated`.")
    bullet(lines, 1, "`f1_score`: harmonic mean of sensitivity and precision.")
    bullet(lines, 1, "`recovery_01`, `recovery_02`, `recovery_03`: first three matched topographic similarities, padded with NaN if unavailable.")
    bullet(lines, 1, "`runtime_s`: method runtime for the selected solution extraction row.")
    bullet(lines, 0, "Canonical template identity metrics")
    bullet(lines, 1, "True and estimated maps are aligned to the canonical MetaMaps template using strong-threshold `" + fmt_value(sim.get("template_alignment_strong_threshold")) + "`.")
    bullet(lines, 1, "`cluster_identity_accuracy`: fraction of all possible true/estimated states whose matched canonical labels agree, normalized by `max(K_true, K_estimated)`.")
    bullet(lines, 1, "`cluster_identity_accuracy_matched`: label agreement fraction among matched topographic pairs only.")
    bullet(lines, 1, "`cluster_n_label_matches`: number of matched pairs with the same canonical label.")
    bullet(lines, 1, "`cluster_mean_matched_similarity`: mean topographic similarity across the matched pairs used for label identity.")
    bullet(lines, 0, "Backfit diagnostics")
    bullet(lines, 1, "Backfit diagnostics are `" + fmt_bool(sim.get("compute_backfit_diagnostics")) + "`.")
    bullet(lines, 1, "Saved backfit detail files are `" + fmt_bool(sim.get("save_backfit_details")) + "`.")
    bullet(lines, 1, "Backfit metrics use downsample factor `" + fmt_value(backfit_factor) + "`, so a `" + f"{n_samples:,}" + "` sample record contributes up to `" + f"{backfit_samples:,}" + "` diagnostic samples.")
    bullet(lines, 1, "Hard backfitting uses winner-take-all absolute topographic correlation at each sample for simulated fits.")
    bullet(lines, 1, "Mixture backfitting is available for simulated fits through transition-aware soft weighting from fitted maps; for real VB-GMM fits it falls back to Gaussian peak-spread soft assignment when peak-map labels are available.")
    bullet(lines, 1, "`backfit_overlap_fraction`: fraction of diagnostic samples with more than one non-zero true state weight.")
    bullet(lines, 1, "`backfit_coverage_corr` and `backfit_coverage_spearman`: Pearson and Spearman correlations between true and estimated canonical-label coverage.")
    bullet(lines, 1, "`backfit_coverage_mae`, `backfit_coverage_rmse`, and `backfit_coverage_l1`: absolute, root-mean-square, and L1 errors in canonical-label coverage.")
    bullet(lines, 1, "`backfit_hard_cluster_top1_accuracy`: sample-wise accuracy of the top predicted fitted cluster after projecting estimated clusters to true state indices.")
    bullet(lines, 1, "`backfit_hard_label_top1_accuracy`: sample-wise accuracy of the top canonical template label.")
    bullet(lines, 1, "`backfit_hard_cluster_top1_accuracy_overlap` and `backfit_hard_label_top1_accuracy_overlap`: the same accuracies restricted to overlap samples.")
    bullet(lines, 1, "`backfit_hard_label_weight_mae` and overlap counterpart: mean absolute error between estimated and true canonical-label weight vectors.")
    bullet(lines, 1, "Mixture backfit fields mirror the hard-backfit fields with prefix `backfit_mix_*` and include an availability flag.")
    bullet(lines, 1, "Overlap pair-accuracy fields test whether the active true state pair is recovered among the strongest predicted weights on overlap samples.")
    bullet(lines, 0, "Per-K candidate metrics in `k_candidate_metrics.csv`")
    bullet(lines, 1, "Each row records replicate, true K, candidate K, SNR, overlap probability, montage, channel count, method, and edge-candidate indicators.")
    bullet(lines, 1, "Candidate metrics include free energy, SPM PCA free energy, log likelihood, BIC, ICL, assignment entropy, parameter count, silhouette, GEV, Calinski-Harabasz score, WSS, and covariance summaries.")
    bullet(lines, 1, "Normalized columns rescale finite values to 0-1 within the fitted candidate-K curve.")
    bullet(lines, 1, "`selected_by_*` indicator columns mark which K would be selected by each criterion.")

    lines.append("")
    return "\n".join(lines)


def self_test() -> None:
    cfg = recursive_merge(DEFAULT_CONFIG, {"simulation": {"reps": 2, "SNR_dbs": [0], "overlap_probs": [0, 0.5]}})
    text = build_methods_markdown(cfg, Path("config.json"))
    assert "Generated EEG conditions: `16`" in text
    assert "`spm_vb` contributes `12`" in text
    assert "`kmeans_koenig` contributes `1`" in text
    assert "Expected comparison-result rows if all fits succeed: `624`" in text


def main() -> None:
    parser = argparse.ArgumentParser(description="Build METHODS.md from the microstate simulation config.")
    parser.add_argument("--config-file", type=Path, default=ROOT / "config" / "microstate_config.json")
    parser.add_argument("--output-file", type=Path, default=ROOT / "METHODS.md")
    parser.add_argument("--self-test", action="store_true", help="Run a small generator self-check and exit.")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    config = load_config(args.config_file)
    markdown = build_methods_markdown(config, args.config_file)
    args.output_file.write_text(markdown, encoding="utf-8")
    print(f"Wrote {args.output_file}")


if __name__ == "__main__":
    main()
