# Pre-specified upstream processing rules

Everything that decides which recordings, segments and participants reach the statistics is fixed here and in
code/config **before any study data are processed**, is applied identically to every group, never reads diagnostic
labels or outcomes, and writes reason-coded exclusion tables. Detailed stage notes: `docs/STAGE_B_NOTES.md`
(clips → HEPPy → manifest) and `docs/STAGE_C_NOTES.md` (VG-microstates). Thresholds below are the current values in
`config/heppy_options.json`, `fsms/matlab/template_prespecified_v2_constants.m` and `fsms/autonomic.py`.

## 1. Clips

Existing clips and `snip_keyfile.csv` are adopted as they are (the real clips are never rewritten). Segment types:
II, preictal (20 min before onset), ictal, postictal, transictal. Only **II** and **preictal** segments enter the
analysis; ictal/postictal/transictal rows are logged as out of scope and never influence any fit or exclusion.

## 2. HEPPy preprocessing (`heppy_recording_exclusion_v2`, group-blind)

Processing (unchanged from the algorithms that produced earlier outputs): band-pass 0.5–45 Hz, mains notch,
PyPREP (RANSAC off; per-segment), standard-10-20 montage, average reference, extended-infomax ICA (seed 42) with
ICLabel, cardiac-field-artefact-removed and -kept FIFs, automatic R-peak fit and missed-peak recovery. Explicit seeds,
single native threads and `PYTHONHASHSEED=0`; outputs are byte-reproducible.

Exclusions (reason-coded in `heppy_exclusions.csv`, counted in `heppy_summary.json`):

* **Recording level:** fewer than 15 EEG channels after the montage; no ECG channel; no valid 60-s R-R window;
  unhandled exception; worker crash; missing outputs.
* **Segment level, in-scope segments (II, preictal) only:** a segment is excluded if any PyPREP chunk overlapping it
  failed (`skipped_toomanybad`, `skipped_no_montage`, `failed`) or interpolated more than 30 % of its EEG channels.
  A RANSAC fallback is flagged only. **Failures confined to ictal, postictal or transictal chunks never exclude
  anything**, so artefact-laden out-of-scope periods cannot discard a participant's usable data.
* A participant left without a usable II segment has no reference grammar and drops out downstream by the stats
  stage's reference minimum; a participant keeps each preictal event whose segment survives.
* Previously produced HEPPy output is adopted **in place and read-only** only if its recorded options equal the locked
  options (processing keys only) and the files validate; evidence missing from adopted recordings is reported as
  `not_evaluable`, never silently passed.

## 3. Manifest and conversion

Rows enter the manifest only after QC; labels (group, seizure type, semiology) are attached afterwards from the
clinical database. Every dropped row appears in `vg_microstate_exclusions.csv` with a code. EEG is converted with the
standard-10-20 montage so channel positions exist for template matching.

## 4. Microstates (VG-microstates, K = 6)

* K fixed at 6 (VB-GMM fit, seeded). GFP-peak feasibility: a record with fewer than 20 peaks is excluded; a participant
  whose remaining records hold fewer than 60 peaks is excluded; the stage stops if no records remain or a group/condition
  holds fewer than 60 peaks. Reasons go to `exclusions.csv`.
* **Labelling rule `template_prespecified` (no hand labelling).** Exact optimal one-to-one assignment of the six
  fitted global maps to MetaMaps_2023_06 classes {A, B, C, E, F, G} (class D excluded a priori) maximising summed
  polarity-invariant |r|; ties → lexicographically smallest assignment. **State names are nominal:** every registered
  outcome (JS, κ over all six states) is invariant to how states are named, so template similarity cannot be an
  exclusion criterion for the inference. Therefore the rule **stops the run only for technical failures** that make the
  template comparison meaningless (channel matching by `first_common_channels`, too few common channels, fewer fitted
  states than classes, non-finite maps, ill-defined assignment; evidence in `global/LABELLING_REJECTED.json`). Weak
  similarity (any assigned |r| < 0.5) does **not** stop the run: it sets `labelling_quality = below_threshold` with the
  per-state |r| in `labelling_audit.json` and `recovery_audit.json` and it is reported with the results; descriptive
  per-state names are then unreliable. Fitted-state order is read from `recovery_audit.json: fit_labels`, never assumed.
* Two validation gates must pass before statistics: the compact GFP-peak transition list equals the full hard backfit
  sample-for-sample, and the independently re-extracted peak/state/transition counts of every exported window equal the
  exported windows. They test consistency and reproducibility, not validity of GFP-peak microstate analysis.

## 5. ECG-derived CSI/CPI

Corrected R-peak sidecars → robust 15-s CSI/CPI (Candia-Rivera). A record needs ≥ 3 R-peaks in its segment, ≥ 20
physiological R-R pairs (35–140 bpm) in the whole-record train, and ≥ 30 complete 15-s estimates inside the segment;
estimates with > 10 % gap-recovered intervals or non-physiological R-R are skipped. Records failing this have no autonomic
support, so tensor outcomes are unavailable for them (ordinary-matrix outcomes still are). Reasons are in
`autonomic_native_qc.csv`. Standardisation uses each participant's interictal distribution.

## 6. What stops the run

Unregistered/unlocked plan or changed locked file; a technical labelling failure; an infrastructure error in any stage; the
backfit validation gate failing; existing confirmatory results (never overwritten). Nothing is silently skipped:
every exclusion is a row in a table.
