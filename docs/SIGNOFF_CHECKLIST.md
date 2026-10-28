# Investigator sign-off checklist (before `plan-lock` / registration)

This file is **not locked** and is not part of the registration. It lists every decision made while preparing the
pipeline that changes what the registered analysis does, so each can be approved, altered, or reverted *before* the
lock. Anything you change in a locked file means re-running `run_pipeline.py plan-lock` (allowed until you register).

## A. Outcome and inference changes (motivated by an independent simulation review; see SAP §2)

| # | Change from draft v2 | Why | Where to change |
|---|---|---|---|
| A1 | **Primary outcome 1 is null-calibrated excess JS, not log JS** (JS minus a per-window parametric-bootstrap null expectation) | Log plug-in JS depended on window count and weight sharpness: false-positive rate 0.38–0.87 at nominal 0.05 with *no* effect; calibrated: 0.046–0.061, power unchanged | `OUTCOMES`, `null_expected_js` |
| A2 | **Secondary family: 6 tests (Holm), 16 estimate-only cells** instead of Holm(22) | Holm(22): true FWER 0.029, power 2–4× lower than a focused family; outcomes are highly correlated (~4 effective tests) | `plan_family` |
| A3 | **Final-minute level estimands are descriptive only** | Participant-level SD 0.250 vs 0.167 for change; cross-sectional levels carry between-person variance | `plan_family` |
| A4 | **Hypo–hyper uses CR2 + Bell–McCaffrey df** (≥ 3 participants per semiology) instead of CR1 + t(G−1) | CR1 type I 0.07–0.09 at 8/4 and 9/3 clusters | `cluster_cr2`, `MIN_CLUSTERS_PER_SEMIOLOGY` |
| A5 | **Within-DD paired test needs ≥ 3 pairs** | A paired t on fewer is not informative | `MIN_DD_PAIRS` |
| A6 | **Exact-30 sensitivity removed**; replaced by Yuen trimmed, studentised permutation and pre-final window [−120, −60) sensitivities | Exact-30 removed count dependence only at a power cost (0.81 vs 0.96), did not fix weight sharpness | `primary_sensitivity` |

## B. Minimum-data rules (these change who/what is analysed)

| # | Rule | Draft v2 | Now | Effect unknown until the run — check the coverage table |
|---|---|---|---|---|
| B1 | Interictal reference length | ≥ 30 transitions | **≥ 300** | participants with < ≈ 3–5 min of usable interictal EEG drop out |
| B2 | Early windows per event | ≥ 1 | **≥ 5** | events with sparse early coverage drop out (single-window events had ≈ 1.7× variance) |
| B3 | Window transitions | ≥ 30 | ≥ 30 (unchanged) | |

If B1/B2 look too strict you may relax them *now* (they are named constants); you cannot tune them after seeing results.
Note the review suggested ≥ 300–500 for B1 and ≥ 5 for B2.

## C. Upstream rules that replace hand decisions

| # | Rule | Details |
|---|---|---|
| C1 | **Deterministic microstate labelling** replaces hand-typed labels | Exact optimal one-to-one assignment of the six fitted global maps to MetaMaps classes {A,B,C,E,F,G} (D excluded a priori) by polarity-invariant |r|. **Names are nominal — all registered outcomes are invariant to them — so low similarity (any |r| < 0.5) is a reported flag (`labelling_quality`), not a stop;** the run stops only on technical failures (e.g. no usable channel match). The previous hand labels (A,F,C,G,B,E) are *not* assumed: fitted-state order is read from the audit file. *Why changed:* the first full TUEG fixture run was halted by a single map at |r| = 0.18 although nothing in the inference depends on the names. |
| C2 | HEPPy exclusion rule `heppy_recording_exclusion_v2` | Group-blind; thresholds in `config/heppy_options.json` (`qc_rule`). **Recording level:** < 15 EEG channels, no ECG channel, no valid 60-s R-R window, crash/exception. **Segment level, II and preictal only:** an overlapping PyPREP chunk that skipped/failed, or interpolated > 30 % of channels, excludes that segment. Ictal/postictal chunks never exclude anything. Reason-coded, counted by scope. *Caution: the earlier partial real run produced only 50 preprocessed files for 204 clips, so the exclusion rate on real data may be large and cannot be known without processing it; the 30 % and 15-channel limits are your call before locking.* |
| C3 | GFP-peak feasibility: record ≥ 20 peaks, participant ≥ 60 peaks | Mechanical limits so the VB fit can run; excluded with reason code |
| C4 | Existing partial HEPPy output is adopted only if its options match the locked options | Otherwise reprocessed |

**What the |r| flag can and cannot tell you (MATLAB-stage author's simulation, 19 channels).** Six *random* maps,
after the pipeline's own spatial smoothing, reach "every assigned |r| ≥ 0.5" 37 % of the time (iid Gaussian maps 0 %,
true microstate maps plus noise 100 %), so the flag is a coarse quality indicator, not proof of valid microstates. On the
TUEG fixture a 3-patient fit assigned at |r| 0.77–0.98 while the final 14-patient fit had one map at 0.18. MetaMaps
classes **C and E correlate 0.91 on a 19-channel montage**, so no fit on such a montage can separate them. How separable
they are in your real montage is unknown until the run. Plan to inspect and report the global map figure and
`labelling_audit.json` after the run; whether the fitted maps are physiologically credible microstates is a scientific
judgement the pipeline cannot make for you.

**Concordance caveat (cannot be checked without touching real data):** the new rule may assign classes to fitted
states differently from your earlier hand labels, and class pairs that are nearly collinear on a small montage (the
review of the MATLAB stage flags MetaMaps C and E) can be unseparable. If the rule rejects, the run stops and you must
decide (as a documented deviation). You may want a one-off concordance check of the rule against your hand labels,
which requires touching real-study fit output — do it only after registration and report it.

## D. Housekeeping decisions

* The registration unlock is a **record the investigator creates** (`register --url`); it verifies the lock hash and URL
  syntax and cannot verify the OSF record itself.
* `config/study.json` real paths (e.g. `F:/FS_microstates_study/clips/snip_keyfile.csv`, `F:/FS_MS_analysis/prereg_v2_run`)
  were **not verified** against the real folder. A path mismatch discovered after registration needs a configuration
  amendment (re-lock and re-register) before the run.
* The vendored `external/VG-microstates` is a modified derivative of your private repo. The code is already in this
  public repository's history (since commit `1e06399`). Consider pushing the vendored tree to
  `dr-r-kan/VG-microstates` on a tagged branch so the registered code is citable.
* The repository `dr-r-kan/FS_MS` is **public**. Patient-level CSVs are gitignored; participant-level outputs are written
  only under `run_root`, outside the repository.
* Prior-access disclosure in `PREREGISTRATION.md` has a bracketed placeholder for what *you* have already seen.
