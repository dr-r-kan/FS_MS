# Validation report (28 October 2026)

What was actually run and demonstrated, and what was not. This report is **not locked**. Everything below used
synthetic data or Temple University Hospital EEG corpus (TUEG) recordings with **synthetic** diagnostic labels,
onsets and semiology. **No restricted study data was processed.** Scientific results from the fixture are meaningless
by construction.

## 1. Software tests

135 tests pass (`python -B -m unittest discover -s tests`); 7 opt-in slow tests (MATLAB hierarchy, TUEG end to end) are
skipped by default and pass when run with `FSMS_RUN_SLOW=1` (MATLAB-stage suite 15/15, end-to-end fixture suite 3/3). They cover: the restricted-data guard and the real-config refusal through the
actual CLI; ledger idempotency/invalidation/healing/failure-resume with stub stages; the plan lock (tamper detection,
registration gate); the lean autonomic stage reproducing the legacy implementation exactly; HEPPy sentinels, adoption,
exclusion rule and manifest building; MATLAB labelling and the two validation gates; and the statistics (below).

## 2. Statistical validity (simulation, `tests/test_inference_validity.py`)

* `JS − E0` is unbiased where plug-in JS is not (|mean| < 0.0035 across four reference-sparsity/weight-sharpness/window-size
  cells; raw JS biased upward by > 0.01) and does not move with window transition count (shift < 0.004 between 30 and 150
  transitions while raw log JS shifts > 0.15).
* With a 30 % lower FS final-minute transition rate and no true effect, a participant-level Welch test on raw log JS rejects
  > 15 % of the time; on calibrated excess JS it stays < 20 % in the small test (40 repetitions; the independent review's
  20 000-resample result was 0.046–0.061 vs 0.38).
* A genuine 30 % grammar mixture leaves clearly positive excess JS (t > 4).
* CR2 + Bell–McCaffrey keeps hypo–hyper type I error 0.01–0.075 at 9/3/0, 3/3/0 and 5/5/5 clusters per stratum.
* Hash-seeded, bit-identical reruns of every stochastic step.

## 3. End-to-end run on the TUEG fixture (`config/fixture_tueg.json`)

14 TUEG patients (FS 5, ES 5, DD 4; groups, semiology and times synthetic), chosen in a seeded, outcome-blind order. After
the pre-specified QC (0 recording-level exclusions; 5 of 33 in-scope segments excluded for >30 % interpolation), 13
participants / 28 records (12 II + 16 preictal) reached the microstate stage; the primary FS–ES contrast had 4 FS and
5 ES participants (one synthetic-FS patient lost both its interictal and preictal segments to QC; one preictal FS final
window lacked 30 action-supported transitions for the tensor outcomes).

| Stage | Time (first, cold) | Notes |
|---|---|---|
| fixture build + clips | ~20 s | real `snip_clips` code path |
| HEPPy (4 workers) | 980 s | 14 recordings, 166–362 s each |
| manifest crops + conversion | 63 s | |
| MATLAB hierarchy + exact transitions | 397 s | both validation gates pass on 440 windows / 28 records; 382 529 transitions |
| autonomic CSI/CPI | 55 s | |
| descriptive preparation + data-freeze lock | 75 s | |
| lock verification / confirmatory tests | 4 s / 5 s | |

Labelling on the fixture: assigned |r| 0.18 / 0.84 / 0.69 / 0.66 / 0.98 / 0.95; one state below 0.5, so
`labelling_quality = below_threshold` (a reported flag; outcomes do not depend on state names). Confirmatory structure:
2 primary and 3 of 6 secondary tests estimable; hypo–hyper (2 hyper participants) and within-DD (2 pairs) correctly
`not_estimable`; 16 descriptive cells carry no p-values; with synthetic labels the primary effects were ≈ 0
(tensor excess JS 0.001, p = 0.82).

## 4. Idempotency, healing, protection (all on the fixture)

* **Second full run:** all 8 stages `[SKIP]`; 642 output files, 0 changed, 0 added, 0 removed (SHA-256).
* **Healing:** deleting one converted EEG file and one transition cache → only `manifest` and `microstates` reran (39 s);
  downstream stages stayed current; every file identical except ledger/sentinel timestamps and the rebuilt MATLAB v7.3 cache
  (bytes differ by an embedded timestamp, **content hash identical**); data-freeze lock verification still passed.
* **HEPPy:** killing a worker mid-recording and rerunning recomputed only that recording, all outputs byte-identical to
  pre-kill; changing one option reran only the affected steps; restoring it restored the original bytes (agent B).
* **Overwrite protection:** forcing `stats_confirm` after results exist is refused.
* **Real config:** `run --config config/study.json` exits 2 with "Real study data is LOCKED" before any stage runs.

## 5. Not validated / limits

* Any run on real study data (deliberately not done). Paths in `config/study.json` are unverified.
* HEPPy with 50 Hz mains (only 60 Hz TUEG exercised; real options differ only in `line_freqs`); adoption of the real
  partial HEPPy output (tested only against a synthetic legacy tree).
* Whether the real clips, with their ictal/postictal chunks and EEG quality, pass the QC rule at an acceptable rate;
  whether MetaMaps C and E are separable in the real montage; whether the fitted maps are credible microstates.
* The review's simulation assumptions (see `STATISTICAL_ANALYSIS_PLAN.md` §9). The calibrated statistic is untested below
  30 transitions per window.
* The labelling flag is a coarse indicator: random maps reach |r| ≥ 0.5 in 37 % of 19-channel simulations.
* The TUEG fixture proves the machinery runs; it says nothing about FS versus ES.
