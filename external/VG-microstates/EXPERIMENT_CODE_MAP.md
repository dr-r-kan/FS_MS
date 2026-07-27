# Experiment Code Map

This document maps the public experiment entry points, their internal
dependencies, and the omnibus scripts used to analyse outputs.

Internal files are kept under underscore-prefixed folders such as `_internal/`
and `_analysis/`. MATLAB helper filenames are not individually prefixed because
MATLAB requires a callable function to live in a file with the same name.
`analyze_single_eeg_file.m` is intentionally public because it is a useful
standalone entry point.

The public plotting labels are `VB-GMM` for `spm_vb` and `K-Means` for
`kmeans_koenig`.

## Public Entry Points

| File | Role |
| --- | --- |
| `simulated_ms_retrieval_experiment.m` | Runs the synthetic microstate recovery benchmark across true K, SNR, overlap, montage, method, and K-selection criteria. |
| `fit_microstate_hierarchical_dataset.m` | Runs the dataset-wise global/group/condition/participant/participant-condition VB-GMM hierarchy, currently used for LEMON-style test datasets and general EEG manifests. |
| `cardiac_cycle_and_microstates/fit_microstates_heppy_global_group_condition.m` | HEPPy-specific wrapper that prepares condition data, runs the hierarchy, exports clusters, and backfits global templates to records. |
| `analyze_single_eeg_file.m` | Standalone single-record workflow for real EEG. |
| `analyse_simulation_outputs.py` | Public omnibus post-processing entry point for simulation outputs. |
| `analyse_hierarchical_outputs.m` | Public omnibus topoplot and TANOVA entry point for hierarchy MAT outputs. |
| `analyse_lemon_outputs.py` | Public omnibus behavioural/psychometry entry point for LEMON/hierarchical outputs. |
| `cardiac_cycle_and_microstates/analyse_heppy_outputs.py` | Public omnibus downstream statistics/report entry point for HEPPy outputs. |

## Dependency Map

### `simulated_ms_retrieval_experiment.m`

This synthetic benchmark driver generates EEG once per simulation condition,
reduces it to requested montages, fits each method once, reapplies all supported
selection criteria, and writes comparison tables, per-K metrics, JSON
topographies, and optional backfit diagnostics.

| Dependency | What it does |
| --- | --- |
| `setup_microstate_paths.m` | Adds `_internal/matlab`, `_analysis/matlab`, `_internal/matlab/Koenig_code`, and SPM-BSD compatibility helpers to the MATLAB path. |
| `_internal/matlab/microstate_utilities.m` | Shared configuration, path resolution, preprocessing, map normalisation, filtering, GFP peak extraction, channel utilities, plotting helpers, and SPM path setup. |
| `config/microstate_config.json` | Central defaults for paths, preprocessing, simulation design, hierarchical fitting, and plotting. |
| `_internal/matlab/generate_microstate_eeg.m` | Generates canonical-template synthetic EEG with state dwell times, GFP envelopes, spatial jitter, ecological background noise, optional artefacts, and overlap between adjacent states. |
| `_internal/matlab/load_metamaps_templates.m` and `MetaMaps_2023_06.set` | Load canonical template maps and montage geometry for simulation and alignment. |
| `_internal/matlab/select_montage_subset.m` | Reduces the full template montage to supported 10-20 subsets while preserving labels, positions, channel locations, and maps. |
| `_internal/matlab/fit_microstate_spm_vb.m` | Fits VB-GMM mixture models for all candidate K values in a polarity-invariant feature space and stores per-K solutions and metrics. |
| `_internal/matlab/select_spm_vb_k_by_criterion.m` | Applies K-selection criteria including free energy, ICL, BIC, covariance, elbows, GEV, silhouette, and hybrids. |
| `_internal/matlab/extract_microstate_solution_for_k.m` | Builds a criterion-specific selected result from the stored per-K fit output. |
| `_internal/matlab/fit_microstate_kmeans_koenig.m` | Runs the vendored MICROSTATELAB K-Means baseline and computes comparable GEV, silhouette, WSS, and recovery metrics. |
| `_internal/matlab/Koenig_code/*.m` | Vendored upstream MICROSTATELAB functions used by the K-Means baseline. |
| `_internal/matlab/microstate_partial_alignment.m` | Computes polarity-invariant rectangular map matching and recovery metrics when estimated K differs from true K. |
| `_internal/matlab/align_microstates_to_template.m` | Aligns estimated and true maps to MetaMaps labels using channel labels or coordinate interpolation. |
| `_internal/matlab/backfit_microstate_timecourse.m` | Backfits fitted states to the full record using hard absolute-correlation assignments and soft mixture weights where available. |
| `_internal/matlab/compute_simulation_backfit_diagnostics.m` | Compares simulated backfit outputs with known state weights and canonical labels, including overlap-specific metrics and confusion matrices. |
| `_internal/matlab/save_microstate_json.m` | Writes selected microstate maps, metrics, metadata, and template labels to plotting-friendly JSON. |

External requirements include MATLAB, EEGLAB, SPM `spm_mix`, the MATLAB
Parallel Computing Toolbox for parallel runs, and the vendored MICROSTATELAB
files.

### `fit_microstate_hierarchical_dataset.m`

This active dataset-wise fitter reads a folder or CSV manifest, harmonises
channels, extracts GFP peaks, fits a global VB-GMM model, then fits group,
condition, participant, and participant-condition nodes seeded from parent
templates.

| Dependency | What it does |
| --- | --- |
| `setup_microstate_paths.m` | Adds internal MATLAB helper folders. |
| `_internal/matlab/microstate_utilities.m` | Shared config, preprocessing, channel handling, scalp filtering, interpolation support, and SPM path setup. |
| `config/microstate_config.json` | Defaults for input/output paths, preprocessing, K candidates, priors, map caps, template alignment, and plotting. |
| `_internal/matlab/fit_microstate_spm_vb.m` | Global and child VB-GMM fitting engine. |
| `_internal/matlab/select_spm_vb_k_by_criterion.m` | Selects K for global and child fits. |
| `_internal/matlab/align_microstates_to_template.m` | Aligns fitted hierarchy nodes to MetaMaps labels and flips map polarity for consistent reporting. |
| `_internal/matlab/load_metamaps_templates.m` and `MetaMaps_2023_06.set` | Provide canonical labels and channel geometry. |
| `_internal/matlab/backfit_microstate_timecourse.m` | Backfits participant-condition or participant templates to full records and supports hard and Gaussian-mixture metrics. |

External requirements include MATLAB, SPM `spm_mix`, EEGLAB for `.set` files
and topoplots, and optionally FieldTrip, EEGLAB FileIO/Biosig, or MNE-MATLAB
for additional EEG formats such as FIF.

### `cardiac_cycle_and_microstates/fit_microstates_heppy_global_group_condition.m`

This HEPPy wrapper finds the current raw FIF layout, prepares
condition-specific records, converts data when MATLAB cannot read FIF, runs the
hierarchical fitter, saves cluster exports, and backfits global templates to
participant-condition records.

| Dependency | What it does |
| --- | --- |
| `setup_microstate_paths.m` | Adds internal MATLAB helper folders from the repository root. |
| `cardiac_cycle_and_microstates/_prepare_heppy_condition_fifs.py` | Uses MNE annotations/event codes to crop and concatenate condition-specific FIF files from exact `suj_N_pp_raw.fif` inputs. |
| `_internal/python/_convert_fif_manifest_to_set.py` | Converts FIF manifest rows to `.mat` or `.set` plus channel-location sidecars for MATLAB compatibility. |
| `fit_microstate_hierarchical_dataset.m` | Performs the VB-GMM hierarchy once the HEPPy manifest is prepared. |
| `_internal/matlab/microstate_utilities.m` | Supplies preprocessing and spatial filtering helpers for global-template backfitting. |

External Python requirements include `mne`, `numpy`, and `pandas`. External
MATLAB requirements mirror the hierarchical fitter.

### `analyze_single_eeg_file.m`

This standalone workflow loads one `.set` or `.mat` file, applies configured
preprocessing, fits the requested method, optionally aligns to MetaMaps,
backfits the time course, writes JSON, and plots topographies.

| Dependency | What it does |
| --- | --- |
| `setup_microstate_paths.m` | Adds internal MATLAB helper folders. |
| `_internal/matlab/microstate_utilities.m` | Configuration, EEG preprocessing, channel labels, positions, scalp masks, and SPM setup. |
| `_internal/matlab/fit_microstate_spm_vb.m` | Default VB-GMM fitting engine. |
| `_internal/matlab/fit_microstate_kmeans_koenig.m` | Optional K-Means engine. |
| `_internal/matlab/align_microstates_to_template.m` | Optional MetaMaps alignment. |
| `_internal/matlab/backfit_microstate_timecourse.m` | Full-record state backfitting. |
| `_internal/matlab/save_microstate_json.m` | JSON export. |
| `_internal/matlab/plot_real_eeg_microstate_topographies.m` | Topographic plot generation from JSON and the original EEG montage. |

## Omnibus Output Analyses

### Simulation Outputs

Use:

```bash
python3 analyse_simulation_outputs.py outputs/simulations/results
```

| Internal file | Inputs | What it contributes | Degeneracy handled by omnibus |
| --- | --- | --- | --- |
| `_analysis/_analyse_comparison_results.py` | `comparison_results.csv`, optional `k_candidate_metrics.csv`, JSON, and backfit `.mat` files | Broad bootstrap/statistical comparison, optional mixed models, plots, and backfit confusion outputs. | Canonical simulation analysis core; the omnibus disables its Python-rendered topography plot. |
| `_analysis/_simulation_methods_report.py` | `comparison_results.csv`, `k_candidate_metrics.csv`, config JSON | Methods-section summary with design counts, methods, criteria, outcomes, and files. | Shares method/criterion counting with other reports. |
| `_analysis/_simulation_descriptive_results.py` | `comparison_results.csv`, `k_candidate_metrics.csv` | Descriptive criterion, K error, F1, SNR, true-K, and overlap plots. | Overlaps with broad comparison and benchmark summaries. |
| `_analysis/_simulation_benchmark_results.py` | `comparison_results.csv` | Paired VB-GMM versus K-Means benchmark with bootstrapped/permutation contrasts by montage. | Focused view of metrics also present in broad reports. |
| `_analysis/_simulation_confusion_matrices_report.py` | `comparison_results.csv` | K-selection confusion matrices for each method and criterion. | Overlaps with confusion outputs from the broad analysis. |
| `_analysis/matlab/plot_simulated_microstate_method_comparison.m` | `comparison_results.csv` plus JSON exports | Representative recovery plot with true, K-Means, and VB-GMM maps rendered using EEGLAB `topoplot`. | Called by `analyse_simulation_outputs.py` to avoid Python topography orientation problems. |

The older split-report workflow is now internal. `analyse_simulation_outputs.py`
is the public route that combines the best pieces and keeps labels consistent.

### Hierarchical and LEMON Outputs

Use MATLAB topographic/statistical analysis:

```matlab
analyse_hierarchical_outputs('outputs/hierarchical_microstates/hierarchical_microstate_results.mat');
```

Use Python behavioural/psychometry analysis:

```bash
python3 analyse_lemon_outputs.py
```

| File | Inputs | What it contributes | Degeneracy handled by omnibus |
| --- | --- | --- | --- |
| `_analysis/matlab/plot_microstate_global_states.m` | `hierarchical_microstate_results.mat` | Global fitted templates using EEGLAB `topoplot`. | Called by `analyse_hierarchical_outputs.m`. |
| `_analysis/matlab/plot_microstate_split_by_group.m` | `hierarchical_microstate_results.mat` | Group-split templates using `topoplot`. | Shares plotting boilerplate with global/condition plotters. |
| `_analysis/matlab/plot_microstate_split_by_condition.m` | `hierarchical_microstate_results.mat` | Condition-split templates using `topoplot`. | Shares plotting boilerplate with global/group plotters. |
| `_analysis/matlab/run_microstate_hierarchical_tanova.m` | `hierarchical_microstate_results.mat` | Participant-level topographic permutation/Bayesian-bootstrap summaries for hierarchy effects. | Called by `analyse_hierarchical_outputs.m`. |
| `_analysis/_analyze_lemon_backfit_behaviour.py` | Backfit state/record CSVs plus LEMON behavioural data | Participant-level backfit features, psychometric PCA, correlations, and demographic tests. | Broad behavioural analysis core. |
| `_analysis/_analyse_lemon_state_e_psychometry.py` | Behaviour/backfit feature CSVs, state metrics, optional covariance and axis-dynamics CSVs | Focused TAS/STAI/YFAS state-E plots, group-only hierarchical TANOVA by state, and psychometry-by-covariance interactions on axial measures. | Supersedes the earlier focused TAS/STAI state-E plotting script. |
| `compute_spm_bsd_spectral_parameters.m` | EEG folder/manifest and optional microstate results MAT | SPM-BSD aperiodic/periodic spectra by participant-condition and optionally by microstate activation. | Separate spectral side analysis, not a core dependency of the three main entry points. |
| `_internal/matlab/compat/spm_bsd/*.m` | SPM-BSD compatibility setup | Compatibility wrappers for older/newer SPM-BSD expectations. | Internal support for spectral analysis only. |

### HEPPy Outputs

Use:

```bash
python3 cardiac_cycle_and_microstates/analyse_heppy_outputs.py
```

| Internal file | Inputs | What it contributes | Degeneracy handled by omnibus |
| --- | --- | --- | --- |
| `cardiac_cycle_and_microstates/_analysis/_heppy_microstate_publication_stats.py` | HEPPy MATLAB output and original HEPPy FIF files | Main downstream HEPPy statistics: group/condition effects, cardiac-cycle modulation, heartbeat-tapping metrics, and beat-level models. | Canonical HEPPy stats core. |
| `cardiac_cycle_and_microstates/_analysis/_compile_heppy_microstate_report.py` | Output of the stats script | HTML report from HEPPy tables and figures. | Report layer over the stats core. |
| `cardiac_cycle_and_microstates/prepare_hmm_data.py` | HEPPy global backfit sequences and source FIFs | Sample-level microstate/R-peak tables for HMM or Markov analyses. | Separate sequence-analysis export. |
| `cardiac_cycle_and_microstates/sequence_analysis.py` | Sample-level microstate/R-peak CSV or parquet | Bayesian input-dependent Markov analysis around R peaks. | Experimental sequence-analysis path. |
| `cardiac_cycle_and_microstates/hep_surprisal.py` | Conditioned FIFs and optional microstate backfit outputs | Experimental R-locked RR-surprisal and aperiodic EEG modelling. | Separate from the main microstate fitting pipeline. |

## Superseded Public Surface

The standalone simulation report scripts, the focused TAS/STAI/YFAS plotting
script, and the HEPPy stats/report pair are no longer intended as direct public
entry points. They are retained as internal modules under `_analysis/` or
`cardiac_cycle_and_microstates/_analysis/` and are called by the omnibus
wrappers above.

Legacy MATLAB-only analyses such as `_analysis/matlab/analyze_montage_robustness.m`,
`_analysis/matlab/summarize_first_line_spm_vb_metrics.m`,
`_analysis/matlab/plot_simulated_microstate_method_comparison.m`,
`_analysis/matlab/plot_simulated_backfit_confusion_summary.m`, and
`_analysis/matlab/write_simulation_backfit_reports.m` remain available for
narrow follow-up figures or checks, but the public simulation post-processing
route is `analyse_simulation_outputs.py`.
