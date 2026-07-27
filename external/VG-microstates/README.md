# SPM Microstate Analysis

MATLAB and Python tools for EEG microstate fitting, template alignment,
hierarchical dataset modelling, cardiac-cycle analyses, and simulation-based
method comparison.

The public-facing method labels are:

- `VB-GMM` for the SPM variational Bayesian Gaussian-mixture microstate fitter.
- `K-Means` for the vendored MICROSTATELAB K-Means baseline - though this is
  dependent on keeping the required code in a local folder.

Defaults for paths, preprocessing, hierarchical fitting, simulation, and
plotting are kept in `config/microstate_config.json`.

## Public Entry Points

### Simulation Benchmark

```matlab
T = simulated_ms_retrieval_experiment();
```

This runs the synthetic recovery benchmark across true K, SNR, overlap,
montage, method, and K-selection criteria.

Run the omnibus post-processing suite with:

```bash
python3 analyse_simulation_outputs.py outputs/simulations/results
```

The omnibus script calls the internal reports in `_analysis/` and writes a
single analysis folder containing methods summaries, descriptive plots,
paired benchmarks, bootstrap/statistical comparisons, and K-selection
confusion matrices. The representative microstate recovery topography is
rendered through MATLAB/EEGLAB `topoplot`, not Python interpolation.

### Hierarchical Dataset Fit

```matlab
[HResults, results_mat] = fit_microstate_hierarchical_dataset('LEMON', ...
    'output_dir', 'outputs/hierarchical_microstates');
```

This is the dataset-wise global/group/condition/participant/participant-condition
VB-GMM hierarchy. It accepts a folder of supported EEG files or a CSV manifest.
A manifest must contain `file_path`; `participant`, `condition`, and `group`
are optional.

For a CSV manifest:

```matlab
[HResults, results_mat] = fit_microstate_hierarchical_csv('manifest.csv', ...
    'output_dir', 'outputs/my_hierarchical_microstates');
```

If MATLAB cannot load FIF files directly, convert a manifest first:

```bash
python3 _internal/python/_convert_fif_manifest_to_set.py labelled_eeg_files.csv outputs/dreams_set --overwrite
```

Topographic summaries and hierarchy-level TANOVA can be run with:

```matlab
analyse_hierarchical_outputs('outputs/hierarchical_microstates/hierarchical_microstate_results.mat');
```

LEMON behavioural/psychometry post-processing can be run with:

```bash
python3 analyse_lemon_outputs.py
```

### Single EEG Analysis

```matlab
[Results, json_file] = analyze_single_eeg_file('subject_EC.set', ...
    'method', 'spm_vb', ...
    'criterion', 'elbow_sil_combined', ...
    'align_template', true);
```

This standalone workflow loads one `.set` or `.mat` file, applies configured
preprocessing, fits microstates, aligns to MetaMaps when requested, saves JSON,
and writes topographic plots.

## Main Outputs

Hierarchical runs write:

- `hierarchical_microstate_results.mat`: full `HResults` structure.
- `hierarchical_fit_summary.csv`: global/group/condition/participant summaries.
- `common_channels.csv`: final montage after filtering and interpolation.
- `pooled_gfp_peak_manifest.csv`: retained GFP peaks and provenance.
- `participant_condition_state_backfit_metrics.csv`: per-state backfit metrics.
- `participant_condition_state_pairwise_backfit_metrics.csv`: pairwise state metrics.
- `participant_condition_record_backfit_summary.csv`: per-record backfit summary.
- `participant_condition_state_axis_dynamics.csv`: optional per-state active-axis dynamics.
- `global/`, `groups/`, `conditions/`, `participants/`, and
  `participant_conditions/`: fitted node exports and topoplots.

## Repository Layout

- `simulated_ms_retrieval_experiment.m`: synthetic benchmark.
- `fit_microstate_hierarchical_dataset.m`: direct dataset-wise hierarchy.
- `cardiac_cycle_and_microstates/fit_microstates_heppy_global_group_condition.m`: HEPPy entry point.
- `analyze_single_eeg_file.m`: standalone single-record entry point.
- `analyse_simulation_outputs.py`: simulation omnibus analysis.
- `analyse_hierarchical_outputs.m`: hierarchical topoplots and TANOVA.
- `analyse_lemon_outputs.py`: LEMON behavioural/psychometry omnibus analysis.
- `cardiac_cycle_and_microstates/analyse_heppy_outputs.py`: HEPPy downstream omnibus analysis.
- `_internal/matlab/`: MATLAB helper functions used by the entry points.
- `_internal/matlab/Koenig_code/`: vendored MICROSTATELAB K-Means files.
- `_internal/python/`: Python conversion helpers.
- `_analysis/`: internal simulation, LEMON, and MATLAB post-processing modules.
- `config/`: repository defaults.
- `outputs/`: generated results; ignored by git.

## Koenig Code Provenance

Any code copied exactly from Thomas Koenig's MICROSTATELAB repository lives in
`_internal/matlab/Koenig_code/`. Local files outside that folder should contain
only project-specific wrappers, adaptations, or reimplementations.

Vendored upstream files currently used:

- `_internal/matlab/Koenig_code/eeg_kMeans.m`
- `_internal/matlab/Koenig_code/pop_FitMSMaps.m`
- `_internal/matlab/Koenig_code/L2NormDim.m`
- `_internal/matlab/Koenig_code/mywaitbar.m`

Upstream source: <https://github.com/ThomasKoenigBern/microstates>

## Dependencies

- MATLAB R2020a or newer; developed with R2025a.
- EEGLAB for `.set` loading and `topoplot`.
- SPM mixture toolbox (`spm_mix`) for VB-GMM fitting.
- MATLAB Parallel Computing Toolbox for parallel simulation runs.
- Python 3 with the scientific stack used by the analysis scripts.
- MNE, NumPy, and pandas for HEPPy/FIF preparation and conversion.

## License

MIT License. See `LICENSE`.
