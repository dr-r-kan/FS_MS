# VG microstates study: results summary

## Executive summary

The analysis retained **1,313 recording entries from 200 participants** and selected a **six-state global microstate solution**. The global maps matched the canonical templates strongly (mean absolute correlation **0.948**; all **6/6** matches above the strong-match threshold). Both corrected backfit methods completed for every recording entry.

The primary topographic TANOVA found two state-specific pairwise results after max-statistic correction across the six states:

- **ES versus FS, state G:** dGFP/RMS = **0.0569**, corrected *p* = **0.0464**.
- **Postictal versus preictal, state G:** dGFP/RMS = **0.0090**, corrected *p* = **0.0230**.

The overall group, condition, and group-by-condition omnibus tests were not significant. The Bayesian analysis gave moderate support that the ES–FS state-G effect exceeded the prespecified practical threshold of 0.05, but indicated that the postictal–preictal effect, although permutation-significant, was much smaller than that threshold.

Channel-wise Bhattacharyya analysis localized secondary differences in selected states and channels, particularly for ES versus FS and for postictal versus preictal comparisons. The C/E analyses found condition-related changes in **mean GFP**, but no FDR-corrected effects on C/E coverage, no corrected preictal temporal gradients, and no corrected semiology effects.

## Dataset and analysis population

| Group | Recording entries | Participants |
|---|---:|---:|
| DD | 383 | 41 |
| ES | 452 | 63 |
| FS | 478 | 96 |
| **Total** | **1,313** | **200** |

| Condition | Recording entries | Participants |
|---|---:|---:|
| Interictal (II) | 265 | 192 |
| Postictal | 460 | 195 |
| Preictal | 455 | 194 |
| Transictal | 133 | 47 |

The complete four-condition factorial TANOVA retained only **45 participants**, chiefly because transictal data were available for 47 participants. Pairwise postictal–preictal testing retained **194 paired participants**, whereas the ES–FS complete-condition comparison retained **36 participants**. These different analysis populations should be kept in mind when comparing effect sizes and uncertainty.

## Six-state solution and template quality

Among candidate solutions K = 4–7, K = 6 had the highest selection score (**0.456**) and explained **84.2% GEV**. The selected states and canonical correlations were:

| State index | Canonical label | Absolute template correlation |
|---:|:---:|---:|
| 1 | D | 0.915 |
| 2 | F | 0.892 |
| 3 | G | 0.930 |
| 4 | C | 0.993 |
| 5 | B | 0.993 |
| 6 | A | 0.966 |

Group-level template correlations ranged from **0.855 to 0.900**, and condition-level correlations ranged from **0.900 to 0.928**; every group and condition solution retained six strong matches. Across 200 participant fits, mean template correlation was **0.895** (range **0.575–0.965**) with a mean of **5.98/6** strong matches. This supports a stable shared state structure while acknowledging weaker template correspondence in a small subset of participants.

## Corrected global backfit results

Both methods were available for all **1,313 recording entries**:

| Backfit method | Available entries | State occupancy/count missingness |
|---|---:|---:|
| Hard assignment | 1,313/1,313 | 0 |
| Gaussian mixture | 1,313/1,313 | 0 |

Mean GFP remains undefined when a state receives exactly zero weight; this accounts for 10 hard and 2 Gaussian state rows. Their occupancy, percentage present, occurrence count, and occurrence rate are correctly recorded as zero rather than as failed measurements.

Participant-condition mean occupancy across all groups was:

| State | Hard assignment | Gaussian mixture |
|:---:|---:|---:|
| A | 0.202 | 0.147 |
| B | 0.193 | 0.158 |
| C | 0.191 | 0.177 |
| D | 0.167 | 0.154 |
| F | 0.177 | 0.181 |
| G | 0.070 | 0.178 |

The methods answer related but different questions: hard occupancy is the proportion assigned exclusively to each state, whereas Gaussian occupancy distributes weight across states. Their absolute occupancies should therefore not be treated as interchangeable.

## Primary topographic TANOVA

TANOVA tests polarity-aligned scalp-field shape rather than amplitude. The primary *p* value, `p_max_state`, controls the six selected states within each tested effect using a permutation max statistic.

### Corrected findings

| Contrast | State | dGFP/RMS | Uncorrected *p* | Max-state *p* | Units |
|---|:---:|---:|---:|---:|---:|
| ES vs FS | G | 0.0569 | 0.0414 | **0.0464** | 36 |
| Postictal vs preictal | G | 0.0090 | 0.0230 | **0.0230** | 194 |

No state passed max-statistic correction for the overall group, overall condition, or group-by-condition interaction tests. No other group pair, condition pair, within-group condition effect, or condition-specific group effect passed this correction.

These two findings are isolated post-hoc contrasts rather than significant omnibus effects, and the output also contains many related contrasts. They should therefore be reported as focused pairwise evidence rather than as broad group or condition effects.

### Bayesian practical-significance check

For the ES–FS state-G contrast, the posterior mean effect was **0.0647** (95% interval **0.0337–0.1006**), with probability **0.794** that the effect exceeded the 0.05 ROPE and an evidence ratio of **3.84**. This is moderate evidence for a practically non-negligible topographic difference.

For the postictal–preictal state-G contrast, the posterior mean was **0.0101** (95% interval **0.0053–0.0159**) and the estimated probability of exceeding 0.05 was effectively zero. Thus, this contrast is statistically detectable in the large paired sample but small relative to the chosen practical threshold.

The strongest Bayesian result in the broader table was the state-G group effect at the transictal condition (posterior probability above the ROPE **0.992**), but only 47 participants contributed and its permutation max-state *p* was **0.0848**. It is suggestive, not conventionally confirmed.

## Channel-wise Bhattacharyya TANOVA

This secondary analysis compared channel-wise Gaussian distributions of polarity-aligned GFP-peak maps assigned to the fixed global states. Bayesian FDR at 0.05 selected **41 channel-state findings** across the tested contrasts.

Notable localized results were:

- **II minus postictal:** state B at T6 and CP1, and state F at T6 (3 findings).
- **Preictal minus postictal:** state C at Cz, CP1, FC1, and Fz (4 findings); the fitted mean differences were negative, indicating lower preictal than postictal values at these locations.
- **FS minus ES:** state C at CP2 and Pz and state F at O2 (3 findings).
- **DD minus FS:** states D and G at FC2 (2 findings).
- **Group-by-condition contrasts:** 28 findings involved ES versus FS across the six condition pairs, and one involved FS versus DD. No ES-versus-DD interaction channel survived Bayesian FDR.

The channel-level findings are useful for localization but should be interpreted alongside the topographic TANOVA. They test many channel/state/contrast combinations and are not substitutes for a significant global omnibus interaction.

## C/E group-condition analysis

The participant-template table retained **628 participant-condition rows from 200 participants**. The GEE analyses considered global state C and participant-level states C and E, with coverage and mean GFP as outcomes.

After Benjamini–Hochberg correction:

- **Condition affected mean GFP** for global C (*q* = **3.36×10⁻⁷**), participant C (*q* = **9.99×10⁻⁸**), and participant E (*q* = **3.22×10⁻⁴**).
- No corrected group main effect or group-by-condition interaction remained significant.
- No corrected condition, group, or interaction effect was found for state coverage.

For global state C, pooled mean GFP was approximately **2.92 µV in II**, **3.20 µV postictally**, **3.20 µV preictally**, and **3.85 µV transictally**. In the GEE parameterization, postictal and preictal mean GFP were each significantly higher than the II reference. Transictal estimates were larger descriptively but less precise because of the smaller sample.

Coverage was comparatively stable across conditions. For global state C it averaged approximately 0.197 in II, 0.185 postictally, 0.190 preictally, and 0.194 transictally, consistent with the nonsignificant coverage model.

## Preictal-gradient and semiology analyses

The preictal-gradient models included up to **3,491 participant-bin observations from 156 participants**. No time-to-ictus slope, seizure-type effect, hypo/hyper effect, or interaction survived FDR correction for global C, participant C, or participant E coverage or mean GFP. Several nominal effects were present, but the smallest corrected *q* was 0.178; these do not support a reliable preictal temporal gradient.

The revised semiology-by-condition analysis expresses each participant's preictal and postictal values as change from that participant's own interictal (II) baseline and compares **FS with ES**. It retained **151 participants** for global C, **145** for participant C, and **101** for participant E. No FS–ES group effect, preictal–postictal condition effect, or group-by-condition interaction survived FDR correction for baseline-adjusted coverage or mean GFP. The nominal participant-E coverage interaction (*p* = 0.0179) did not survive correction (*q* = 0.322).

## Overall interpretation

The results support a robust, canonical six-state microstate structure across the dataset. The clearest topographic evidence is concentrated in state G: an ES–FS difference of moderate practical size and a much smaller postictal–preictal difference detectable in the large paired sample. Channel-wise distributional tests add localized evidence, especially for ES–FS and postictal–preictal contrasts, but the absence of significant omnibus effects argues against describing the dataset as showing a general group, condition, or group-by-condition topographic shift.

The C/E analyses point to condition-dependent changes in map-associated GFP rather than changes in state coverage. These GFP findings did not translate into corrected preictal-gradient or semiology effects. Taken together, the current results are more consistent with selective changes in the strength and local distribution of particular microstate topographies than with wholesale changes in the six-state repertoire.

## Important limitations

1. The full factorial TANOVA used only 45 condition-complete participants; transictal inference is consequently less precise than II, preictal, or postictal inference.
2. The groups are unbalanced (41 DD, 63 ES, 96 FS participants), as are the condition counts.
3. The two corrected standard TANOVA findings are post-hoc pairwise contrasts without significant corresponding omnibus effects.
4. Bhattacharyya findings are channel-level secondary analyses. Bayesian FDR controls their selected set, but interpretation should emphasize spatial patterns rather than isolated electrodes.
5. Hard and Gaussian-mixture occupancies have different meanings and should not be directly compared as if they were estimates from the same assignment rule.
6. The results are observational and do not establish causal relationships between group, seizure phase, semiology, and microstate measures.

## Principal output files

- Global fit: `G:\FS_microstates_study\vg_microstates\hierarchical\global\global_model_comparison.csv`
- Fit summary: `G:\FS_microstates_study\vg_microstates\hierarchical\hierarchical_fit_summary.csv`
- Corrected backfit metrics: `G:\FS_microstates_study\vg_microstates\hierarchical\global_state_backfit_metrics.csv`
- Standard TANOVA: `G:\FS_microstates_study\vg_microstates\hierarchical\omnibus_analysis\tanova\tanova_results.csv`
- Bayesian TANOVA: `G:\FS_microstates_study\vg_microstates\hierarchical\omnibus_analysis\tanova\bayesian_tanova_results.csv`
- Bhattacharyya TANOVA: `G:\FS_microstates_study\vg_microstates\hierarchical\omnibus_analysis\bhattacharyya_tanova\bhattacharyya_tanova_results.csv`
- C/E group-condition analysis: `G:\FS_microstates_study\vg_microstates\hierarchical\ce_group_condition_analysis\ce_group_condition_gee_terms.csv`
- C/E preictal-gradient analysis: `G:\FS_microstates_study\vg_microstates\hierarchical\ce_group_condition_analysis\ce_preictal_gradient_gee_terms.csv`
- C/E semiology analysis: `G:\FS_microstates_study\vg_microstates\hierarchical\ce_group_condition_analysis\ce_semiology_gee_terms.csv`
