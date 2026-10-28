# FS microstate transition analysis — action-conditioned mechanistic extension

This package contains the revised global microstate-transition analysis and a new mechanistic extension that estimates an empirical four-action transition tensor:

\[
P(m_{t+1}=d\mid m_t=s, a_t),
\]

where the autonomic-action proxy \(a_t\) is represented by four CPI/CSI combinations:

1. CPI+ / CSI+
2. CPI+ / CSI−
3. CPI− / CSI+
4. CPI− / CSI−

The package is self-contained for the supplied 60-s transition-count and autonomic-window inputs. It does not refit the microstate maps.

## Main revisions

- **Destination-labelled transition structure:** source and destination identities remain explicit in the full \(6\times6\) object; the diagonal is excluded because the supplied counts describe next-distinct-state transitions.
- **Participant-cluster inference:** FS–ES resampling preserves participants who contribute both functional and epileptic events.
- **Common-reference pseudo-onsets:** genuine and matched pseudo windows are evaluated against the same participant II reference after the pseudo window is removed.
- **Stricter terminal pseudo matching:** pseudo-onsets are drawn from the terminal 1% of an II recording; covariate-adjusted and strict-caliper sensitivities are exported.
- **Four-action II tensor:** action-specific II transition rows are partially pooled toward each participant's action-pooled grammar and a leave-participant-out population action effect.
- **No pseudo-replication:** the primary “interpolation” is soft interpolation in CPI/CSI action space. The four fractional weights sum to one, so each observed transition count is used once.
- **Correct κ parameterization:** \(\kappa\) is fitted directly in \(\operatorname{softmax}(\kappa\log p)\); `action_log_kappa` is then \(\log\kappa\). This prevents double exponentiation.
- **Nested II validation:** the action prior is selected using other participants and evaluated once in each held-out participant.
- **Temporal checks:** the package contains a preceding-contiguous-window action sensitivity and a separate participant-held-out model asking whether current transitions predict the next autonomic quadrant beyond the current quadrant.
- **Publication output:** all main figures are supplied as 400-dpi PNG and editable vector PDF.

## Reproduce the mechanistic extension

From the package root:

```bash
python code/run_mechanistic_extension.py \
  --out-root rerun \
  --bootstrap-draws 2000 \
  --run-lagged
```

The wrapper uses the files under `input/` by default. Run the internal tests with:

```bash
python code/run_mechanistic_extension.py --self-check
python code/analyse_global_transition_exploration.py --self-check
```

## Primary outputs

- `action_tensor/ACTION_TENSOR_RESULTS.md` — numerical action-tensor results.
- `action_tensor/figures/figure_07_action_conditioned_transition_tensor.*` — main mechanistic figure.
- `action_tensor/figures/figure_08_action_tensor_validation_and_pseudo.*` — II validation and matched-pseudo analysis.
- `action_tensor/figures/figure_S7_interictal_action_transition_tensors.*` — four II transition tensors.
- `action_tensor/figures/figure_S8_action_timing_sensitivity.*` — same-window versus preceding-window action specification.
- `action_tensor/figures/figure_S9_caliper_pseudo_sensitivity.*` — strict-caliper sensitivity.
- `directionality/figures/figure_09_transition_to_next_action_directionality.*` — reverse temporal-prediction analysis.
- `MODEL_SPECIFICATION.md` — equations and estimands.
- `MECHANISTIC_EXTENSION_RESULTS.md` — concise interpretation.
- `PUBLICATION_AUDIT.json` and `SHA256SUMS.csv` — provenance and file hashes.

## Results in one paragraph

Interictally, action conditioning improved participant-held-out transition log score by 0.003910 per transition (95% interval 0.002715 to 0.005255), establishing that CPI/CSI action carries reproducible information about the next microstate destination. In the final minute, action-conditioned JS was 0.01523 in FS and 0.01288 in ES; the FS–ES log contrast was 0.167 (-0.014 to 0.348). The contrast was clearest at −2 to −1 min (0.171, 0.006 to 0.341). However, κ was below 1 in both FS (0.565) and ES (0.631), so the data do **not** support the proposed ES κ>1 pattern. The primary λ and genuine/pseudo estimates remained uncertain; a small strict-caliper subset showed a marked negative FS-specific λ difference-in-differences and is retained only as an exploratory sensitivity. Current transitions did not reliably predict the next autonomic quadrant beyond the current quadrant.

## Interpretation boundary

The tensor is an observation-level heuristic for an action-conditioned state-transition model. CPI/CSI are autonomic policy proxies, and EEG microstates are coarse-grained observations of neural activity. The analysis therefore tests whether autonomic state is associated with the next-distinct microstate distribution and whether that association changes near seizures. It does not identify a literal active-inference \(B\) matrix or prove causal direction.
