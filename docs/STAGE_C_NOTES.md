# Stage C notes: headless VG-microstates hierarchy and exact transitions

Stage C turns a manifest of (converted) EEG files into `transition_index.csv`, `transition_cache/*.mat` and
`recovery_audit.json` for `analyse_timestamped_transition_dynamics.py`. It runs headless under `matlab -batch`,
with no interactive step, no hard-coded path and no call to Python.

## Commands

```powershell
# whole stage (the orchestrator passes a FRESH, versioned out_dir)
matlab -batch "addpath('<repo>\fsms\matlab'); run_study_hierarchy('<manifest.csv>','<out_dir>');"

# individual pieces (all take the hierarchy folder produced by run_vg_microstates)
matlab -batch "addpath('<repo>'); run_vg_microstates('<analysis_manifest.csv>','<hierarchy_dir>');"
matlab -batch "addpath('<repo>'); export_directional_peak_windows('<hierarchy_dir>','<analysis_manifest.csv>','<windows.csv>');"
matlab -batch "addpath('<repo>'); recover_timestamped_transition_backfits('<analysis_manifest.csv>','<hierarchy_dir>','<backfit_dir>','validation_windows_csv','<windows.csv>');"

# tests (MATLAB tests need FSMS_RUN_SLOW=1; the rest are pure Python)
.venv\Scripts\python.exe -B -m unittest tests.test_labelling -v
matlab -batch "addpath('<repo>\tests\matlab'); test_prespecified_labels('<scratch dir>')"
```

`run_study_hierarchy` options: `validation_windows_csv` (destination of the freshly exported windows CSV),
`conditions`, `min_peaks_per_record` (20), `min_peaks_per_participant` (60), `validation_records` (`'all'`),
`full_backfit_validation` (`'first_per_participant'`), `max_records`, `hierarchy_args`, `real_study_mode`. Do not use
`max_records` or `hierarchy_args` for the registered run (they are recorded in the fingerprint and audit).

## Restricted-data guard

`fsms/matlab/fsms_assert_unrestricted.m` is the MATLAB twin of `fsms.guard` (roots parsed from `fsms/guard.py`; textual
comparison only, so the check never touches restricted storage). Every entry point (`run_study_hierarchy`, `run_vg_microstates`,
the manifest reader, window export, transition recovery, labelling) refuses a restricted manifest, EEG file or output folder with
`FSMS:RestrictedData` before any `isfile`/read. It is lifted only while `run_study_hierarchy(..., 'real_study_mode', true)` runs;
the orchestrator must pass that only after `fsms.lock` has verified the registration record. It does not resolve subst/junction
aliases (the Python guard does, and the orchestrator calls it first).

## The pre-specified naming rule `template_prespecified_v2`

Frozen in `fsms/matlab/template_prespecified_v2_constants.m`; implemented in `template_prespecified_v2.m`. v2 supersedes v1
(which stopped the stage whenever an assigned |r| was below 0.5): the registered outcomes (JS, kappa over all six states) are
invariant to how fitted states are named, the names only permute states consistently and label descriptive tables, so halting
the confirmatory analysis on template similarity would conflate a descriptive naming-quality indicator with inference.

1. |r| (polarity-invariant) between each of the six global fitted maps and the MetaMaps_2023_06 K=7 maps. Channels are
   matched by label when at least 16 labels match, otherwise the fitted maps are interpolated (inverse-distance weighting
   on unit-sphere coordinates, MNE RAS) to the template electrodes. `first_common_channels` is never accepted.
2. Class set `{A,B,C,E,F,G}`; class D is excluded a priori.
3. Exact optimal one-to-one assignment maximising the summed |r| (all 720 permutations). Ties (total within 1e-9) go to
   the lexicographically smallest assignment vector (entry k = class index of fitted state k, order A,B,C,E,F,G).
4. Quality flag (descriptive, **never a stop**): `labelling_quality = 'ok'` when every assigned |r| >= 0.5, otherwise
   `'below_threshold'`, with the per-state |r| list and the states below 0.5 recorded in `labelling_audit.json` and
   `recovery_audit.json` (`labelling_quality`, `labelling_states_below_threshold`, `labelling_correlations.assigned_abs_r`).
   Audit text: *State names are nominal; primary and secondary outcomes do not depend on them; below_threshold means
   descriptive per-state names are unreliable.* The runner-up margin stays recorded.
5. Technical failures stop the stage (`global/LABELLING_REJECTED.json`, no label files, `FSMS:LabellingRejected`; no fallback):
   channel-matching mode not `labels`/`coordinates_idw_to_template` (e.g. `first_common_channels`), fewer than 8 fitted-map
   channels in the comparison, non-finite maps or correlations, fewer or more fitted states than classes, non-unique labels,
   or tied optimal assignments (identical fitted maps: the naming is ill-defined).
   Labels and audit are ALWAYS written when none of these applies, including for `below_threshold`.

Outputs in `hierarchy/global/`: `global_manual_labels.csv` (`state_index,label,selected_utc,source`; source is
`template_prespecified_v2`), `global_manual_labels.mat` (what the vendored hierarchy re-reads) and `labelling_audit.json`
(rule id, template and script SHA-256, |r| matrices, assignment, `labelling_quality`, `states_below_threshold`, margin to the
runner-up assignment, next-best class per state, channel mode, MATLAB/SPM versions). The fitted-state order depends on the data and the RNG: it is always **read** from these
files (or `recovery_audit.json: fit_labels`), never assumed.

Mechanics (no vendored code is modified): pass 1 runs the vendored hierarchy with `global_only` and no template, so the cached
global solution keeps its raw fitted order; the rule then writes `global_manual_labels.mat` with `reference_centers` equal to
that solution; pass 2 runs the full hierarchy, which re-uses the cached global fit and takes the saved labels. The global fit
is identical to a single-pass run.

## What the two audit gates prove

* `compact_export_validated_against_full_backfit`: for the first record of every participant (all records with
  `full_backfit_validation='all'`) the compact GFP-peak transition list equals, sample for sample, the full hard backfit.
* `focused_window_fit_recovery_validation.transition_matrix_exact_match`: the windows CSV was exported from the hierarchy GFP-peak
  caches and the participant solutions. The gate re-extracts the GFP peaks **from the raw EEG file** (independent of the cache),
  checks they equal the cache, maps fitted states to the presentation order through the global labels (independent of the
  mapping used by the exporter) and recomputes the peak/state/transition counts of **every** exported window of those records
  with a separate counting loop. It proves the cache is the deterministic product of the manifest file and the frozen
  preprocessing (no stale or mismatched cache) and that exporter and transition stage use the same state mapping. It does not
  validate the labelling itself (see the labelling audit) or the VB fit. Any mismatch aborts the stage; the field is only ever
  written as `true`.

## Exclusions

GFP peaks are extracted once in `screening_peaks/` (small global fit, discarded). A record with fewer than 20 peaks is excluded
(`RECORD_TOO_FEW_GFP_PEAKS`); a participant whose remaining records hold fewer than 60 (10 x K) peaks is excluded with all of its
records (`PARTICIPANT_TOO_FEW_GFP_PEAKS`). Reasons go to `exclusions.csv`; the study continues on `analysis_manifest.csv`. If
nothing is excluded the screening peak caches are reused (atomic copies), otherwise extraction is repeated on the retained set.
If no record remains or a group/condition holds fewer than 60 peaks the stage stops (`FSMS:TooFewMapsAfterExclusion`). Missing or
unreadable EEG files are errors, never exclusions. The thresholds are mechanical feasibility limits chosen before looking at any
data; they never use labels, maps or outcomes.

## Idempotency and protections

* `STAGE_C_INPUTS.json` (manifest SHA-256, parameters, SHA-256 of every vendored/owned MATLAB source, the template and the config,
  MATLAB release, SPM version/commit) is written first. A non-empty folder without it, with another manifest hash
  (`FSMS:StaleOutputDir`) or with other parameters/sources/environment (`FSMS:StageInputsChanged`) is refused.
* `STAGE_C_DONE.json` is written last with SHA-256 of every key output and hierarchy solution and size of every transition
  cache. A re-run with the same inputs verifies it and skips (no file is touched). Missing/changed derived outputs
  (transition caches, index, audit, windows CSV) are healed: the stage resumes, every cache is revalidated (record path,
  participant centres, label order, preprocessing hash, full-backfit flag) and bad ones are rebuilt. The hierarchy is not
  re-run when healing (the sentinel has just verified its files). A changed frozen file (manifests, screening, exclusions,
  labels, hierarchy solutions) is an error (`FSMS:DoneSentinelInconsistent`).
* New outputs are written to a temporary file and renamed. The vendored hierarchy writes its own caches non-atomically; an
  interrupted run is resumed or refused, never mixed with other inputs.
* `STAGE_C_FAILED.json` records the error identifier/message of a failed stage.

## `recovery_audit.json` (schema_version 2)

Keeps every legacy key (`manual_labels_by_fitted_state`, `n_manifest_records`, `n_global_peaks`, `n_timestamped_records`,
`n_transitions`, both gate fields, `transition_time_resolution`, ...) and adds `schema_version`, `stage`, `manifest_csv`/`manifest_sha256`
(the STUDY manifest as given to `run_study_hierarchy`), `analysis_manifest_csv`/`analysis_manifest_sha256` (the retained-records
manifest actually analysed), `hierarchy_dir`, `fit_labels` (six labels in fitted-state order), `labelling_rule`, `labelling_quality`
(`ok`|`below_threshold`), `labelling_states_below_threshold` (list of `{state_index, label, assigned_abs_r}`), `labelling_semantics`,
`labelling_correlations`
(`matrix_abs`, `columns`, `assigned_abs_r`, `next_best_abs_r`, `strong_threshold`, `channel_match_mode`, `template_sha256`),
`labelling_audit_sha256`, `n_full_backfit_validated_records`, `full_backfit_validation_scope`, `state_index_convention`,
`conditions`, `matlab_version`, `matlab_release`, `spm` (dir, version, git commit, `spm_mix` hashes), `script_sha256` (list of
`{path, sha256}`), `script_fingerprint`, `created_utc`, and `provenance` (stage parameters, exclusion counts and hashes, input
fingerprint). The gate sub-object also carries `n_records_reextracted`, `n_windows_checked`, `n_windows_mismatched`,
`peak_cache_reproduced_from_raw`, `max_abs_map_difference`, `validation_windows_sha256`. Transition caches store **fitted-state**
indices; `fit_labels[i-1]` is the class of state i. `microstate_labels.validate_recovery_audit_schema` checks all of this.

## Synthetic test data

`tests/matlab/make_synthetic_dataset.py` writes converter-schema `.mat` files (MNE standard_1020 positions, 256 Hz), a manifest and
the generating truth. The generating maps are the MetaMaps A,B,C,E,F,G classes sampled at the electrodes and then nudged apart
(`--max-class-correlation 0.55`, retaining |r| >= 0.95 with the sampled template): sampled at 19 channels, classes C and E have
|r| = 0.91, which no clustering fit can separate, so without this step a label mismatch would reflect the fit, not the rule.
`--edge-cases` appends a too-short record, a too-little-data participant and 45-s records without a full 60-s window.

## Limits

* The default VG preprocessing (12-neighbour spatial smoothing, 2-20 Hz band) is heavy for 19-21 channels and is unchanged; it
  blurs class differences, so classes that are nearly collinear on a small montage (MetaMaps C and E) may not be separable by
  the fit. The rule then assigns by best total |r| and, if a state falls below 0.5, flags `below_threshold`.
* The labelling checks the fitted map against the template, not that the clusters are physiologically real.
* **The |r| >= 0.5 flag has weak specificity on a 19-channel montage** (one reason v2 no longer enforces it).
  `tests/matlab/rule_null_check.m` (100 sets, seed 11): six iid Gaussian maps get `ok` 0% of the time, the same maps after the
  pipeline's default smoothing 37% (median weakest assigned |r| 0.44), template maps plus noise 100% (median weakest |r|
  0.84-0.91). In an ad-hoc full-pipeline run over `make_synthetic_dataset.py --maps random` seeds 1-6, four of six were `ok` (seed 2 is the
  `below_threshold` case used by the test suite). `ok` therefore does not
  prove the clusters are microstates, and `below_threshold` is not evidence against the inference; the registered outcomes do
  not use the names. The margin to the runner-up assignment is recorded but is not a criterion (the optimal assignment is not
  greedy, so assigned-minus-next-best is often negative even for correct names).
* The gates test consistency and reproducibility, not the validity of GFP-peak microstate analysis.
* The vendored code is unmodified. Its own non-atomic cache writes and its hard-coded `Fit.spm_commit` string are untouched;
  the audit records the real SPM git commit and file hashes.
* `export_global_manual_preview.m` was deleted (manual labelling is obsolete); `legacy/run_ace_manual_pipeline.ps1` still
  calls it.
