# Statistical analysis plan v3

Companion to `PREREGISTRATION.md`. The executable definition is `analyse_timestamped_transition_dynamics.py`;
this plan is locked together with it (`prereg/PLAN_LOCK.json`). The 60-second v5 action tensor in
`FS_action_conditioned_transition_pipeline_v5/` is historical and is not this model.

## 1. Estimands and hypotheses

**Primary (two tests, Holm over 2, two-sided α = 0.05).** The FS − ES difference between independent
participants in the event-averaged **final-minus-early change** of

1. **tensor excess JS** (null-calibrated Jensen–Shannon divergence of the autonomic-gradient-conditioned
   transition grammar from the participant's interictal reference, §3), and
2. **tensor log κ** (log baseline-oriented concentration, §3).

**Secondary (one family, Holm over 6; supporting evidence only, it cannot rescue a negative primary):**
ordinary-matrix excess JS and ordinary log κ (change, FS − ES); tensor excess JS and tensor log κ change for
FS hypo versus hyper and for paired within-DD FS versus ES.

**Descriptive (16 cells; estimate with pointwise 95 % CI, no hypothesis test, no p-value reported):** all
final-minute *level* estimands and the remaining change estimands (ordinary hypo–hyper and DD).

Primary and secondary families are not pooled and no study-wide familywise error claim is made. Effects on
the log scale (κ) are multiplicative; excess JS is additive. There are no route-level, state-level or
time-bin-selection tests.

## 2. Why these outcome definitions (changes from draft v2, pre-registration)

The v2 draft used **log plug-in JS**, a **Holm(22)** secondary family, **CR1** cluster inference, a reference
minimum of 30 transitions and an "exact-30 transitions" sensitivity. An independent adversarial review with
synthetic data (running the repository's own functions; all numbers from simulation, assumptions listed in §9)
found:

* *Plug-in JS depends on the window transition count and on the sharpness of the soft action weights,
  independently of any grammar change.* With **no** true effect, a 30 % lower FS final-minute transition rate
  gave a Welch rejection rate of 0.38 (nominal 0.05), a 30 % loss of autonomic support 0.44, a doubled
  gradient SD 0.55 and a halved one 0.87. Ordinary-matrix JS and κ were not affected as strongly. Because
  transition rate and gradient magnitude are plausible group differences in exactly this biology, log JS cannot
  be a primary outcome. **Fix:** subtract a per-window null expectation (§3); with it the rejection rates were
  0.046–0.061 and power was unchanged (0.966 vs 0.963). The "exact-30" subsample removed the count dependence
  but cost power (0.81 vs 0.96) and did not remove the sharpness dependence, so it is dropped.
* *Holm(22) is wasteful:* the eight FS–ES outcomes have about four effective tests; true familywise error under
  the global null was 0.029 and per-test power was 2–4× lower than for a focused family. The secondary family is
  reduced to the six pre-specified supporting tests; everything else is reported with CIs only.
* *CR1 + t(G−1) is liberal with few/unbalanced clusters* (type I 0.07–0.09 at 8/4 and 9/3 clusters per
  semiology). **Fix:** CR2 with Bell–McCaffrey (Satterthwaite) degrees of freedom, and a minimum of three
  participants contributing to each semiology.
* *Final-level estimands have larger variance than change* (participant-level SD 0.250 vs 0.167 for tensor
  JS), so they are descriptive. A reference built from few transitions inflates JS (0.43 log units at 60 vs 1 500
  reference transitions) so the reference minimum is raised (§4). Events with a single early window have about 1.7×
  the variance of events with ten, so at least 5 early windows are required (§4).

The Welch participant-level t test itself was sound (type I 0.043–0.052 across unbalanced, heavy-tailed and
skewed scenarios) and stays primary; robust and permutation versions are labelled sensitivities (§6).

## 3. Measurement

For source *s* and destination *d ≠ s*, observed transitions exclude self transitions. Four soft action weights are
the products of two logistic soft signs (temperature 0.5) of the recording-scaled CPI and CSI **gradients** at each
transition timestamp (§5); they sum to one per transition and are fractional allocations, never independent
replicated counts.

* *Ordinary reference rows:* (count + 0.5)/(row total + 2.5) from the participant's interictal windows. *Tensor
  rows:* action counts + 10 × the ordinary reference probability + 0.5, row-normalised. Interictal windows are
  evaluated against a reference that leaves the evaluated window out; preictal windows never update the reference.
  There is no population borrowing.
* *JS:* outgoing-count-weighted mean of row-wise Jensen–Shannon divergences (natural logs) between the window's
  smoothed destination proportions and the reference.
* *Null-calibrated excess JS (the inferential JS outcome):* `E0` is the mean JS over 400 hash-seeded parametric-bootstrap
  replicates in which each transition's destination is drawn from `Σ_a w_ia · reference[a, s_i, ·]`, conditional on the
  window's **own** source sequence and soft weights. **excess JS = JS − E0.** Calibration check: for windows drawn
  exactly from the reference, E[JS − E0] is within ±0.0004 for n = 30–300 across reference sparsities (simulation);
  `tests/test_inference_validity.py` re-verifies this. Excess JS may be negative. Log JS is kept as a *descriptive*
  column only.
* *κ:* maximises the baseline-oriented multinomial likelihood with probabilities ∝ baseline^κ, bounds
  exp(−4) ≤ κ ≤ exp(4); a bracketed derivative root solves interior fits; an exactly uniform reference does not
  identify κ and gives an unavailable value. Boundary hits are recorded in the metrics table and reported.
  Log κ is nearly count-free and robust to weight sharpness (simulation).

## 4. Windows, intervals and eligibility (fixed before data are processed)

* 60-s windows. **Final measurement: exactly [−60, 0) s.** **Early measurement:** complete windows wholly inside
  [−1200, −600) s (10-s stride; overlapping windows are measurement summaries, never inferential units).
* A window needs ≥ 30 transitions (≥ 30 action-supported transitions for tensor outcomes). A participant needs an
  interictal reference of ≥ **300** transitions after leaving the evaluated window out.
* An event contributes to an outcome only if its final window is finite **and** it has ≥ **5** eligible early
  windows with finite values. Each outcome is judged separately, so denominators differ between tests; per-test
  counts are in the results table.
* **No imputation, optional stopping, outcome-based exclusion or observed-power calculation.**
* `tables/coverage_by_group.csv` is produced before any test: by group and outcome, events and participants in
  the index, with an evaluated window, with a final window, with ≥ 5 early windows, and complete. Counts only.
  Exclusion is *informative* if transition rate differs by group (simulated: 74 % vs 91 % final-window eligibility
  at a 30 % rate difference; up to 42 % attenuation if the rate falls with effect size). Coverage must be
  reported with every result and eligibility differences discussed; this plan does not adjust for them.

## 5. Autonomic support

CSI/CPI are the Candia-Rivera robust indices on 15-s windows, standardised to each participant's interictal
distribution. Levels and centred finite-difference gradients are interpolated only within finite contiguous runs: no
interpolation across explicit missing values or gaps > 5 s and no extrapolation. 5 s is a measurement-support rule,
not a physiological threshold. A zero finite gradient gives neutral weights; a missing gradient stays missing.
Gradients are scaled by each recording's median absolute transition-sampled gradient (SD fallback). Whole-record
scaling and centred differencing make this retrospective; no forecasting claim is made. These are autonomic
*proxies*, not measurements of sympathetic or vagal outflow.

## 6. Inference, aggregation and sensitivity analyses

* **Aggregation:** mean the eligible window outcomes within event and interval, take final − early per event, then
  average events within participant (or participant × semiology / event type). Every participant has equal weight;
  multiple events do not multiply the independent sample size.
* **FS versus ES (independent):** Welch t test on participant means; FS-only and ES-only participants only; DD
  participants never enter it; groups must not overlap; labels must be consistent within participant.
* **Within-DD FS versus ES:** paired t test on participant means; requires ≥ 3 complete pairs, otherwise unavailable.
* **FS hypo versus hyper:** OLS with intercept and hyper indicator on participant-by-semiology means with **CR2**
  participant-cluster covariance and Bell–McCaffrey degrees of freedom; the estimand is hyper − hypo, combining
  within- and between-person information (not a pure paired effect); requires ≥ 3 participants contributing to each
  semiology, otherwise unavailable.
* **Reporting:** estimates, group/cluster counts, statistics, df, raw and Holm-adjusted p (primary and secondary
  only), pointwise 95 % CIs (not simultaneous). Unavailable tests stay in their fixed family as p = 1 for the
  adjustment and are displayed as unavailable, never as negative findings. No one-sided conversion.
* **Sensitivity analyses** (`tables/primary_sensitivity_results.csv`, labelled, no multiplicity adjustment, never
  substituted for the registered tests): Yuen 20 %-trimmed Welch; studentised permutation test of the Welch statistic
  (20 000 hash-seeded permutations); the same contrast with the pre-final window [−120, −60) replacing [−60, 0)
  (guards against onset artefact in the last minute). Check that clinical and EEG onset are defined identically for FS
  and ES.
* No normality test selects an estimator. Inspect distributions, residuals, influential participants, missingness and
  group imbalance after the registered run; any additional check is a labelled deviation/exploratory result.
  Contrasts are unadjusted associations; confounding by medication, state, age and recording conditions must be
  discussed. With few DD or hypo/hyper participants, report the corresponding tests as underpowered (a paired t at
  n = 8 must reach |t| ≈ 3.64 at Holm's strictest step over six tests, versus 4.68 under the superseded
  Holm(22)).

## 7. Upstream processing rules (pre-specified, group-blind)

See `docs/PIPELINE_RULES.md`: HEPPy preprocessing and its exclusion rule, the deterministic microstate labelling rule
`template_prespecified_v2`, GFP-peak feasibility thresholds, ECG/CSI/CPI validity rules, and what stops the run.
Exclusions are reason-coded and counted by group; the same rule applies to every recording, applied before any
outcome is computed. Deterministic seeds and the pinned package set are recorded in the lock.

## 8. Exploratory material

Manifold plots, source/destination routes, entropy concentration and autonomic level trajectories are descriptive.
A within-preictal group difference is not a seizure-specific biomarker without matched non-seizure/pseudo-onset
controls, which this design does not supply.

## 9. Limits and what was not verified

* The review's simulation assumed 60 transitions/min with lognormal participant SD 0.45, Dirichlet concentration
  ≈ 1.5, an action perturbation of 0.3 and iid Gaussian gradients. Real transition rates, gradient distributions,
  between-group differences in count and sharpness, reference lengths, burstiness, autocorrelation and onset
  artefact were not simulated. The calibrated statistic was not validated below n = 30.
* A residual false-positive rate of about 0.07 remained when a strongly concentrated action shift is the true
  change (it is then partly a real grammar-by-action change).
* A noisy, nearly flat reference attenuates κ; boundary hits and reference informativeness are reported, not used to
  exclude.
* Scientific conclusions depend on ECG fiducial validity, microstate assignment quality, diagnosis and semiology
  adjudication, artefact handling, medication/state confounding and representativeness. This plan cannot promise a
  positive result; inadequate numbers or measurement quality give a limited-feasibility conclusion, never evidence of
  no difference.
