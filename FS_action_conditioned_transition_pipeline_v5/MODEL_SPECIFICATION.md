# Model specification

## 1. Observed transition object

For participant \(i\), recording/event \(e\), and 60-s window \(w\), the input contains the destination-labelled count matrix

\[
Y_{iew}(s,d), \qquad s,d\in\{1,\ldots,6\},\quad d\neq s.
\]

Consecutive repetitions were already collapsed upstream. The estimand is therefore a **next-distinct-state jump chain** rather than a complete discrete-time transition matrix: it models the destination conditional on leaving the current microstate, not dwell duration or self-persistence.

## 2. Four autonomic actions and soft interpolation

The four action corners are

\[
A=\{(+,+),(+,-),(-,+),(-,-)\},
\]

where the first sign refers to CPI and the second to CSI. With temperature \(T=0.5\),

\[
p_{CPI+}=\sigma(CPI/T),\qquad p_{CSI+}=\sigma(CSI/T).
\]

The four weights are

\[
\begin{aligned}
w_{++}&=p_{CPI+}p_{CSI+},\\
w_{+-}&=p_{CPI+}(1-p_{CSI+}),\\
w_{-+}&=(1-p_{CPI+})p_{CSI+},\\
w_{--}&=(1-p_{CPI+})(1-p_{CSI+}).
\end{aligned}
\]

They are nonnegative and sum exactly to one. Thus a window's transition counts are not copied into additional independent observations. Instead, the baseline transition probability at its continuous CPI/CSI coordinate is interpolated from four interpretable action corners.

The same-window CPI/CSI summary is primary. A prespecified sensitivity uses the immediately preceding contiguous 60-s CPI/CSI window.

## 3. Participant II action tensor

For each participant, weighted II counts are

\[
C_i(a,s,d)=\sum_{w\in II_i} w_{iwa}Y_{iw}(s,d).
\]

The action-pooled participant grammar is

\[
C_i^{pool}(s,d)=\sum_a C_i(a,s,d).
\]

Each participant's pooled grammar is shrunk toward a leave-participant-out population grammar. Each action row is then shrunk toward a hierarchy that combines the participant's pooled grammar with the leave-participant-out population action log-ratio. The action-row prior strength is selected from \(\{2,5,10,20,50\}\) using II leave-one-window-out log score, with participant-nested validation. The selected value in the supplied analysis is 10 transitions per action-by-source row.

For II evaluation, the current window's pooled and action-weighted counts are removed before constructing its reference. Preictal windows are compared only with II data.

## 4. Interpolated action-conditioned prediction

Let \(B_i(a,s,d)\) be the estimated II action tensor. For a target window with weights \(w_{iwa}\),

\[
\widetilde B_{iw}(s,d)=\sum_a w_{iwa}B_i(a,s,d).
\]

This is the empirical analogue of an action-conditioned transition likelihood. It is not asserted to be an identified latent-state active-inference \(B\) matrix.

## 5. Jensen–Shannon divergence

For each source row, the observed destination distribution is compared with \(\widetilde B_{iw}(s,\cdot)\). Row-wise JS divergences are weighted by the number of observed outgoing transitions:

\[
JS_{iw}=\frac{\sum_s n_{iws}\,JS\!\left(P_{iw}(\cdot\mid s),\widetilde B_{iw}(\cdot\mid s)\right)}{\sum_s n_{iws}}.
\]

Larger JS indicates a greater departure from the II transition grammar expected for the observed autonomic action.

## 6. Baseline-oriented κ

A single positive gain \(\kappa_{iw}\) is fitted across all six source rows:

\[
P_{\kappa}(d\mid s,a_{iw})
=\operatorname{softmax}_d\left[\kappa_{iw}\log\widetilde B_{iw}(s,d)\right].
\]

- \(\kappa=1\): II action-conditioned route preferences are retained.
- \(0<\kappa<1\): those route preferences are flattened.
- \(\kappa>1\): those route preferences are sharpened.

The code stores both `action_kappa = κ` and `action_log_kappa = log(κ)`.

## 7. Action-specificity λ

To distinguish general transition reorganization from loss of action dependence, define the participant's action-pooled II grammar \(B_i^{pool}\) and action log-ratio

\[
R_i(a,s,d)=\log B_i(a,s,d)-\log B_i^{pool}(s,d).
\]

The window-specific interpolated action effect is

\[
\widetilde R_{iw}(s,d)=\sum_a w_{iwa}R_i(a,s,d),
\]

and \(\lambda\) is fitted in

\[
P_{\lambda}(d\mid s,a_{iw})
=\operatorname{softmax}_d\left[\log B_i^{pool}(s,d)+\lambda_{iw}\widetilde R_{iw}(s,d)\right].
\]

- \(\lambda=1\): II action specificity is preserved.
- \(\lambda=0\): the action-specific deviation is removed.
- \(\lambda<0\): the II action relationship is reversed.

λ is more directly targeted to an action-to-transition mapping than κ, whereas κ measures gain on the complete action-conditioned row.

## 8. Predictive validation

For each II window, the action-conditioned and action-pooled models are scored using the multinomial log score per transition:

\[
G_{iw}=\frac{1}{N_{iw}}\left[\log p(Y_{iw}\mid\widetilde B_{iw})-\log p(Y_{iw}\mid B_i^{pool})\right].
\]

The action prior is selected using all other participants, and the held-out participant contributes one mean gain. A positive participant-bootstrap interval establishes that the action-conditioned tensor contains reproducible II information before any preictal interpretation is attempted.

## 9. Group and time inference

Window metrics are averaged first within event and time bin, then across events within participant. FS–ES intervals use participant-cluster bootstrap resampling, preserving the dependence created when a dual-diagnosis participant contributes to both event types.

## 10. Genuine/pseudo-onset comparison

Final genuine windows are matched within participant to terminal II pseudo windows. Primary action matching requires the same hard action quadrant. Both members of a pair are evaluated against exactly the same II tensor after removing the pseudo window. The estimand is

\[
[FS_{genuine}-FS_{pseudo}]-[ES_{genuine}-ES_{pseudo}].
\]

Covariate-adjusted and strict-caliper estimates are exported. The strict-caliper analysis is a sensitivity because it retains a much smaller matched subset.

## 11. Temporal-direction sensitivity

A separate participant-held-out multinomial model tests whether current transition features predict the **next** autonomic quadrant beyond the current quadrant. It is trained on II only. This prospective model addresses temporal prediction, but neither direction is causal without intervention or stronger identification assumptions.
