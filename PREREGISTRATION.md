# FS_MS registration draft — v3, 28 October 2026

**Status: analysis plan prepared for registration. This file is not a registration and does not validate the science.**
Register the final commit together with `prereg/PLAN_LOCK.json` (see `docs/RUNBOOK.md`). The earlier v1/v2 locks do not
apply to this code.

## Question and hypotheses

Do functional-seizure-only (FS) and epileptic-seizure-only (ES) participants differ in the change in
autonomic-gradient-conditioned microstate transition organisation approaching seizure onset?

There are **two primary hypotheses**, both two-sided: an FS–ES difference in the change (final minute minus early
preictal interval) of (1) **tensor excess JS** — the Jensen–Shannon divergence of the transition grammar from the
participant's interictal reference, calibrated against its own finite-sample null — and (2) **tensor log κ** — the
log baseline-oriented concentration. Holm correction across these two controls the primary family at 0.05. A
separate **secondary family of six tests** (ordinary-matrix excess JS and log κ, FS–ES; tensor excess JS and log κ
for FS hypo versus hyper and for within-DD FS versus ES) is Holm-corrected within itself and cannot rescue a negative
primary conclusion. The remaining 16 estimands (all final-minute levels, ordinary hypo–hyper and DD changes) are
reported as estimates with 95 % CIs and no hypothesis test. No study-wide familywise error claim is made. Full
definitions: `docs/STATISTICAL_ANALYSIS_PLAN.md`.

## Design and prior access

Observational analysis of existing EEG/ECG recordings. Data collection, clinical annotation, preprocessing of part of
the recordings, and earlier exploratory analyses (global, autonomic and transition-model analyses, with outputs stored
locally) predate this registration; this is **not** untouched prospective confirmation, a seizure-prediction
validation or a causal experiment. The responsible investigators must state here exactly which related results they
have already seen: *[investigators to complete before registering]*.

Disclosure of access during preparation of this plan (28 October 2026), made by the AI coding assistant used to
build the pipeline:

* The pipeline code, the tests and every statistical decision in this plan were developed and checked **only on
  synthetic data and on Temple University Hospital EEG corpus (TUEG) recordings with synthetic diagnostic labels,
  onsets and semiology**. No group test was run on study data.
* The restricted study folder was **not processed or analysed**. Before the instruction not to touch it, the assistant
  listed its top-level and HEPPy sub-folders, counted files (204 clips, 50 preprocessed EEG files, about 201 ECG
  R-peak files) and folder sizes, counted the rows of the EDF key file, patient key file and analysis database, read the
  first 3–5 rows of the EDF key file, the HEPPy ECG-QC review table, the HEPPy headless summary and the HEPPy
  preprocessing table (file paths, status and PyPREP error messages), the tail of the HEPPy fatal log, and the HEPPy
  options file. No EEG/ECG signal file was opened and no outcome variable was computed. Two sub-agents also glanced at
  repository-root files that hold patient metadata: one at the header and first row of the key files and the header of
  the analysis database, another at the first ~140 lines of the database-building script, which contains hard-coded
  patient identifiers with sex (only column names were used; nothing was copied).
* Changes to the outcome definition, multiplicity structure, minimum-data rules and cluster inference
  (`docs/STATISTICAL_ANALYSIS_PLAN.md` §2) were motivated by simulation studies on synthetic data and by reading the
  code, not by study outcomes.
* Partial HEPPy output for the study recordings exists from earlier runs. It is re-used only if its recorded options
  equal the locked options and the files validate; otherwise recordings are reprocessed (`docs/PIPELINE_RULES.md`).

## Inputs and sampling

The fixed set of clips, the clinical annotation database and the pipeline in this repository. Diagnostic group is FS,
ES or dual diagnosis (DD); event type and FS semiology (hypo/hyper) are separate variables. No participants or events
are added after inspecting results; **no outcome-based exclusion, imputation, optional stopping or observed-power
calculation.** Exact eligible counts are produced by the run (`tables/coverage_by_group.csv`, `heppy_summary.json`,
`exclusions.csv`); historical descriptive counts are not the sample size.

Upstream processing is fully specified in code and `docs/PIPELINE_RULES.md`: HEPPy preprocessing with a group-blind,
reason-coded exclusion rule; a deterministic **template-based microstate labelling rule** instead of hand labelling
(state names are nominal: the registered outcomes are invariant to them, so template similarity is reported as a quality
flag and only technical failures stop the run); GFP-peak feasibility thresholds; ECG validity rules for CSI/CPI. These
rules use no diagnostic labels or outcomes.

## Measurements and intervals

Six aligned microstates define a next-distinct-state jump chain excluding self transitions. Four soft action weights
describe increasing/decreasing CPI crossed with increasing/decreasing CSI at transition timestamps (gradients
interpolated only inside finite contiguous runs with no bridging of gaps > 5 s and no extrapolation; whole-record
scaling; retrospective). Weights sum to one; fractional weights are not independent observations. CSI/CPI are
autonomic proxies, not direct measures of sympathetic or vagal outflow.

Windows are 60 s. The **final measurement is exactly [−60, 0) s**; **early measurements** are complete windows wholly
inside [−1200, −600) s, requiring **≥ 5** eligible early windows per event. A window needs ≥ 30 transitions, a
participant ≥ 300 interictal reference transitions. Overlapping early windows are summaries, never independent units.
Interictal data define each participant's reference grammar (tensor prior strength 10, Jeffreys 0.5; the evaluated
interictal window is left out). JS enters inference **only after subtracting its per-window null expectation** (400
hash-seeded bootstrap draws conditional on the window's own sources and soft weights); raw log JS is descriptive.

## Analysis

Within each interval average eligible window outcomes per event, take final − early per event, then average events
per participant; every participant has equal weight. Independent FS-only versus ES-only participants: Welch t test
(DD participants are excluded from it). Within-DD: paired t test on participant means (≥ 3 pairs). Hypo versus hyper:
participant-clustered OLS on participant-by-semiology means with CR2 covariance and Bell–McCaffrey degrees of
freedom (≥ 3 participants per semiology). Report estimates, counts, statistics, df, raw and adjusted p, pointwise 95 %
CIs. Unavailable tests stay in their fixed family as p = 1 for adjustment and are shown as unavailable, never as
negative findings. No one-sided conversion after seeing a direction.

## Sensitivity, interpretation and reporting

Labelled sensitivities, never substituted for the registered tests: Yuen 20 %-trimmed Welch, studentised permutation
test, and the pre-final window [−120, −60). Coverage by group (events and participants at each eligibility step) is
reported with every result and differences in eligibility discussed, because exclusion is informative if transition
rate differs by group. Kappa boundary estimates, missingness, temporal support and participant distributions are
reported. Manifold plots, routes, entropy concentration and autonomic trajectories are descriptive. Seizure-onset
specificity cannot be inferred without matched non-seizure/pseudo-onset controls, which this design lacks.

Conclusions depend on ECG fiducial validity, microstate assignment quality, diagnosis/semiology adjudication, artefact
handling, medication/state confounding and representativeness, and require investigator review and independent
replication. This plan does not promise a publication or a positive result; inadequate numbers or measurement quality
yield a limited-feasibility conclusion, never evidence of no difference.

## Reproducibility and registration materials

Upload this document, `docs/STATISTICAL_ANALYSIS_PLAN.md`, `docs/PIPELINE_RULES.md`, `docs/RUNBOOK.md`,
`prereg/PLAN_LOCK.json` and `.sha256`, the commit identifier, and a code archive. The lock hashes all analysis code,
configurations, the plan, the pinned package set and the vendored HEPPy / VG-microstates trees, and contains no
participant data. Keep participant-level tables, identifying paths and restricted EEG/ECG in authorised storage. After
registration, `run_pipeline.py register --url <immutable URL>` records the registration; real study data cannot be read
before that record exists and any later change to a locked file blocks the run. The pipeline checks URL syntax and
hashes; it does not certify the OSF record. The [Centre for Open Science guidance](https://www.cos.io/initiatives/prereg)
stresses disclosure of prior access when registering existing-data analyses; this registration cannot undo prior
analytical access.
