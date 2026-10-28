# Stage B notes: clips -> HEPPy -> VG manifest (idempotent, fixture-tested)

Scope: `fsms/clips/`, `fsms/heppy/`, `fsms/manifest/`, `fsms/fixture/`, `config/heppy_options*.json`.
All development testing used **TUEG test EDFs only**; no restricted real-study path was read, listed,
opened or written (every CLI calls `fsms.guard.assert_unrestricted` on its paths in `--mode fixture`,
the default; `--mode real` is gated by `fsms.lock.real_data_unlock_problems()`).

## Run it (each stage is independently re-runnable)

```powershell
$py = '.\.venv\Scripts\python.exe'; $w = 'F:\FS_MS_fixture_tueg'
& $py -m fsms.fixture.make_tueg_fixture --work-root $w                 # synthetic db + clips (real snip_clips)
& $py -m fsms.clips.snip_clips --database $w\analysis_database.csv --out-dir $w\clips [--manifest F --skipped F --force]
& $py -m fsms.heppy.run_heppy_stage --clips-dir $w\clips --heppy-out $w\heppy_output `
      --options config\heppy_options_fixture.json --workers 4 [--limit N] [--only BASE ...] [--adopt-dir LEGACY] [--retry-failed] [--force]
& $py -m fsms.manifest.build_vg_manifest --snip-keyfile $w\clips\snip_keyfile.csv --heppy-out $w\heppy_output `
      --database $w\analysis_database.csv --out-dir $w\vg_microstates --format mat --montage standard_1020 --min-duration-s 10
& $py -B -m unittest discover -s tests -v                              # fast tests; FSMS_RUN_SLOW=1 adds the TUEG end-to-end test
```

Real-study options are `config/heppy_options.json` (normalises to the identical processing options as the legacy
`options_used.json`, so legacy outputs stay adoptable); the fixture uses `config/heppy_options_fixture.json`
(identical except 60 Hz mains: `line_freqs [60, 120]`).

## Design

**Clips** (`fsms/clips/snip_clips.py`) - `OLD/snip_clips.py` with the segment logic and EDF+ writer unchanged
(output EDF and `snip_keyfile.csv` are byte-identical to the legacy writer on the same input; tested). Changes: no
`directories.txt`/cwd (explicit `--database/--out-dir/--manifest/--skipped`); per-patient sentinel
`.sentinels/<patient>.json` = fingerprint of (clip-relevant database columns of that patient's rows - diagnostic
labels deliberately excluded, source EDF name+size+mtime_ns, SHA-256 of the code); an EDF is rewritten only if the
fingerprint changed or the output fails validation (size, exact header bytes, recorded SHA-256); atomic CSV/EDF writes;
stale `.part` cleanup; `self_check` only via `--self-check`.

**HEPPy stage** (`run_heppy_stage` + `worker`, one worker process per clip) - steps in legacy order and with legacy
algorithms: `ecg_fit` (auto R-peak fit on the clip EDF) -> `preprocess` (HEPPy `preprocess_edf_both`: `_pp_raw.fif` and
`_pp_raw_keepcfa.fif`) -> `recovery` (missed-peak recovery on the *preprocessed FIF's* ECG channel, as legacy stage 5).
Each step has its own fingerprint inside `.sentinels/<base>.json` (EDF size+SHA-256, keyfile segment bounds, processing
options, seeds, code hashes, package versions); a step is skipped only if the fingerprint matches **and** every recorded
output validates (FIF header loads with `preload=False` and has EEG channels; sidecar parses). A changed step recomputes
from scratch (the previous sidecar is never an input, so no drift); `recovery` always follows a recomputed upstream step.
Changing only `rpeak_recovery.*` reruns only recovery; `auto_qrs.*` reruns fit+recovery; preprocessing options/seeds rerun
preprocess+recovery; `qc_rule` reruns nothing (re-evaluation only). EDF identity is size + SHA-256, reusing the stored hash
while size and mtime_ns are unchanged (`--verify-hash` forces re-hashing). Determinism: explicit seeds from the options,
native threads pinned to 1, `PYTHONHASHSEED=0` in every worker; HEPPy sets `meas_date=1`, so saved FIFs are
byte-reproducible. Writes are atomic, stale `.part` files are cleaned at start, a run lock prevents concurrent writers, no
prompts, works from any cwd (HEPPy found from the repo root). A recording that raises is caught and recorded (sentinel +
exclusion `unhandled_exception`); a worker that dies (e.g. access violation) becomes `worker_crash`; neither stops the others
nor is retried with an unchanged fingerprint unless `--retry-failed`.

Observers (no behavioural change): HEPPy functions are wrapped by pass-through observers that capture PyPREP outcome per chunk
(`ok` / `ransac_fallback` / `skipped_toomanybad` / `skipped_no_montage` / `failed`), bad channels by detector, interpolated and
still-noisy channels, EEG counts before/after the montage, ICA size/decimation/ICLabel classes, and the ICA flatten-revert flag.
Verified on a TUEG clip: the stage's FIFs, ICA files and both R-peak sidecars are byte-identical to the legacy call sequence
(`OLD/run_heppy_preprocessing` + `OLD/fast_ecg_review`).

**Adoption** (`--adopt-dir LEGACY_HEPPY_OUTPUT`, read-only, in place, nothing copied or written there). A clip is adopted only if
(1) its row in `logs/headless_preprocessing_summary.csv` has `status == ok`; (2) legacy `options_used.json` equals the locked
options after normalisation (BOM/whitespace/key order/paths ignored; compared keys: filters hp/lp/line_freqs, ecg channel,
montage, pyprep, prep_ransac, asr, ica_iclabel, effective CFA mode, target_sfreq, spirometry/stim_keep, all
`preprocessing_tuning`, `adaptive_rr` rr_min/rr_max/rr_outlier_mad, `auto_qrs`, `rpeak_recovery`, and the PRNG seed - legacy
hard-coded 42; `epoch`, `hrv`, `export`, `bpm_smooth_win` are ignored); (3) both FIF headers load; (4) the R-peak sidecar
exists with >= 3 peaks. The sentinel records `adopted: true`, paths, sizes, mtimes (optional SHA-256 with `--adopt-hash`) and the
options verdict, and `heppy_summary.json` lists adopted clips and why others were refused. PyPREP degradation of adopted clips is
reconstructed from the legacy per-row log when present (`logs/headless_rows/*_<base>.log` captured the swallowed warnings);
otherwise it is `unknown_adopted` and reported under `not_evaluable`, never silently passed. When the log is unambiguous (one
`pyprep_fit` announcement per keyfile segment, no RANSAC fits) swallowed `TooManyBad`/`failed` warnings are attributed to
segments, so the segment-level rule applies to adopted recordings too; interpolation fractions of adopted clips are never
available (`not_evaluable: segment_interpolated_fraction`).

**Manifest** (`build_vg_manifest`) - same columns/file names as legacy. Every keyfile row ends in the manifest or in
`vg_microstate_exclusions.csv` with a code (`unsupported_segment`, `out_of_scope_condition`, `heppy_excluded` = recording-scope
HEPPy exclusion, `heppy_segment_excluded` = segment-scope HEPPy exclusion matched on (clip, segment, start, end),
`heppy_output_missing`, `segment_too_short`, `missing_database_row`). QC phase first, database (and thus group / seizure
labels) only afterwards. **Conditions:** only `II` and `preict` are emitted by default. The hierarchical VG fit pools GFP peaks
over every manifest row, so seizure-contaminated ictal/postictal/transictal rows would alter the global/participant
solutions, and the registered analysis (`recover_timestamped_transition_backfits.m` default conditions, the timestamped
dynamics script) uses only II and preictal; the rest is logged as `out_of_scope_condition` and `--conditions` widens the set
explicitly. Crops: sentinel keyed by (source FIF SHA-256, crop bounds, code, MNE version), validated by reloading
(sample count, EEG channels), written atomically. Conversion: unmodified VG converter in a staging dir, always with
`--montage` (default `standard_1020`), per-file sentinel (crop hash, montage, converter hash, library versions), atomic move;
scipy's wall-clock `.mat` header text is normalised so a rebuilt `.mat` is byte-identical.

**Fixture** - 14 distinct TUEG `01_tcp_ar` patients (seeded `random.Random(seed).sample` over sorted, header-scanned candidates;
selection never sees QC results): FS x5 (2 hypo, 2 hyper, 1 two-event hypo+hyper), ES x5, DD x4 (each FS + ES event). Single-event
patients use 3330-4500 s files, two-event patients 5850-9000 s files (>= 2400 s between ictal periods so each event keeps a full
1200 s preictal and no `transictal` segment forms). Each has one 600 s II interval at **120-720 s**, first ictal onset at 1950 s,
ictal 30-60 s. Why 14 patients and why II starts at 120 s: the first 8-patient fixture put II at 30-330 s and PyPREP skipped that
chunk (`RobustReference:TooManyBad`) in 3/8 recordings; a probe of 5 windows x 8 clips showed windows starting at 30 s fail at any
length (300/600/900 s: 3/8, 3/8, 2/8 skipped) while 300-s windows starting at 120 s pass 8/8 (TUEG openings are artifact-prone), so
II start is now a fixed design constant, and the cohort is larger than the analysis needs so enough participants survive the
segment rule. **All groups, semiologies and times are synthetic labels** (`FIXTURE_NOT_REAL_DATA.txt`, `fixture_manifest.json`
with `fixture: true`, the design rationale, TUEG header/first-MiB hashes).

## Pre-specified QC exclusion rule v2 (`qc_rule` in the options JSON, id `heppy_recording_exclusion_v2`; group-blind)

`fsms.heppy.qc.evaluate_rule(qc, rule)` (recording scope) and `evaluate_segments(qc, rule, segments)` (segment scope) take only
QC evidence and keyfile segment times: no label argument exists. Output: reason-coded `heppy_exclusions.csv`
(`base, stage, reason_code, evidence, scope, segment, patient_code, source_row, event_type, event_number, segment_number,
output_start_seconds, output_end_seconds`; segment columns empty for `scope=recording`), `heppy_summary.json` with CONSORT counts at
recording and segment level, per-recording `heppy_qc/<base>.json` (per-chunk status, clip-time bounds, bad channels by detector,
interpolated channels). Exclusion is a normal outcome (exit 0); exit 2 only for infrastructure errors.

| scope | code | criterion | rationale |
|---|---|---|---|
| recording | `too_few_eeg_channels` | EEG channels after montage < 15 | 19 scalp + 2 ear channels survive the montage; losing more than ~4 of 19 leaves too little to interpolate or to share a montage across participants for the VG common-channel set |
| recording | `no_ecg_channel`, `ecg_no_valid_window` | no ECG-like channel; < 1 valid 60-s R-R window (`rpeak_qc_score`) | the registered analysis is cardio-neural |
| recording | `unhandled_exception`, `worker_crash`, `missing_outputs` | any | no silent partial recordings |
| segment (II, preictal rows only) | `segment_pyprep_skipped_toomanybad` / `_skipped_no_montage` / `_failed` | ANY PyPREP chunk overlapping the segment (positive-length overlap) has that status | the legacy swallowed these and carried unreferenced data forward |
| segment (II, preictal rows only) | `segment_excess_interpolation` | an overlapping chunk interpolated > 0.30 of its EEG channels | investigator's pre-specified bound; beyond it interpolated (not measured) signal dominates the topography |

PyPREP chunks overlapping only ictal/postictal/transictal segments never exclude anything (`ransac_fallback` and the ICA
flatten-revert are only flagged). A participant left without a usable II segment is not handled here (downstream reference minimum).
Chunk bounds come from the clip's keyfile (HEPPy chunks PyPREP by exactly those rows); a chunk without bounds is conservatively
taken to cover the whole recording.

`ecg_qc_review.csv` keeps the legacy schema; `viable` is `yes` only with an ECG channel and >= 1 valid 60-s window, else `no`
with the reason in `notes`. EEG FIFs are produced regardless of ECG viability (legacy gated on viability only because it was a
manual-review artefact; the rule now excludes explicitly and visibly).

## Deliberate deviations from legacy

* Signal processing: none (byte-identical outputs verified on a clip). R-peak source labels in the final sidecar are now
  reproducible (legacy re-runs relabelled every peak `reviewed` and lost `is_gap_recovered`).
* The auto-fit result is kept as `<base>_rpeaks_autofit.csv`; `<base>_rpeaks_corrected.csv` is the final sidecar (legacy: same file
  overwritten by recovery; when recovery would lower the valid-window count the sidecar equals the auto-fit, as legacy).
* HEPPy's own "cached" check is bypassed (`redo=True`); only the fingerprint sentinel decides.
* PyPREP `TooManyBad`, RANSAC fallback, and ICA flatten-revert are recorded and rule-evaluated instead of silent.
* `snip_clips`: self-check no longer on every call; per-patient skip logic; atomic keyfile (unchanged content is not rewritten).

## Evidence from the TUEG fixture run (28 Oct 2026; 16 logical cores; `F:\FS_MS_fixture_tueg`)

* Runtimes: fixture build 17 s (45 s with a cold 3500-header scan); clips ~3 s; HEPPy 980 s for 14 recordings with 4 workers
  (166-362 s per recording: ECG fit ~45 s, preprocess ~110-250 s, recovery ~40 s); manifest crops + conversion 63 s.
* Result after QC: no recording-level exclusions; 5 of 33 in-scope segments excluded (all `segment_excess_interpolation`; no PyPREP skips
  once II starts at 120 s); manifest 28 rows (12 II + 16 preict); 12 participants have a usable II and >= 1 preict (FS 4 incl. the
  two-event one, ES 5, DD with both events 2; FIX_001 (FS) loses II and its preictal, FIX_006 (DD) loses its II and one
  event, FIX_007 (DD) keeps one event).
* Idempotency: a second full run (fixture -> clips -> HEPPy -> manifest) recomputed nothing (clips `skipped_unchanged`, HEPPy
  `recomputed=[]`, manifest crops and conversions all `unchanged`) in 0.8 + 5 + 4 s, and all 296 output files (sentinels included)
  were byte-identical; repeated after the tamper and kill tests below.
* Tamper (earlier 8-clip fixture, same engine): changing only `rpeak_recovery.iterations` reran only `recovery` (8/8 clips, manifest
  untouched); changing `ica_fit_hz` for one clip reran `preprocess`+`recovery` of that clip only and, downstream, only that
  participant's crops and `.mat` files; restoring the options recomputed again and every output (FIFs, sidecars, crops, `.mat`)
  returned to the original bytes. Changing `qc_rule` recomputes nothing (unit-tested).
* Kill: the stage was killed mid-PyPREP of one recording (FIFs already removed by the clean-slate step); the next run replanned
  `preprocess`+`recovery` of that recording only ("outputs invalid: ... missing"), removed the stale lock, and every output matched
  the pre-kill bytes; crops/conversions stayed `unchanged` because their fingerprint uses the FIF's content hash.
* Legacy equivalence (one TUEG clip, legacy call sequence from `OLD/` run on the same clip): FIFs (both variants), ICA files, auto-fit
  sidecar and recovery sidecar byte-identical to this stage's outputs.

## Limitations / open decisions

* PyPREP skips/over-interpolation concentrate in short or opening-of-recording chunks on TUEG; on real data the same rule may remove
  more II segments than preictal ones, which the downstream reference minimum then turns into participant exclusions.
* Adopted legacy recordings lack some evidence (`n_eeg_input`, interpolation fractions, sometimes PyPREP status or chunk
  attribution); the rule marks those `not_evaluable` rather than excluding or passing them silently.
* The VG converter's reference-position fallback pools positions over the files of one conversion batch; with
  `--montage standard_1020` every EEG channel gets a montage position, so batching does not matter here.
* Worker crashes are deterministic-by-assumption (not retried without `--retry-failed`).
* Not verified: any run against real study data (forbidden), MATLAB consumption of the generated `.mat` manifest, HEPPy runs on
  files with 50 Hz mains (only 60 Hz TUEG was exercised; the real-study options differ only in `line_freqs`).
