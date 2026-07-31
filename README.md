# FS_MS global-first analysis

The publication pipeline reuses the fitted K=6 microstates and starts with the
whole transition grammar. No microstate is privileged in the headline model.

The primary transition quantities are:

- global Jensen–Shannon divergence from each participant's interictal grammar;
- one global baseline-oriented concentration parameter, κ.

Interictal windows use a leave-that-window-out participant baseline. Preictal
windows use the participant's pooled interictal baseline. The fixed transition
threshold is 30; transition count is an eligibility/information quantity, not a
predictor.

After the global analysis, the pipeline:

- drills down across all six source microstates as BH-FDR-controlled exploratory work;
- tests exploratory CSI, CPI and semiology associations by source state;
- fits an auditable two-dimensional PCA manifold in JS, κ, CSI and CPI and plots
  FS and ES movement from II through five preictal bins;
- fits planned nested logistic models in the order JS → κ → CSI → CPI,
  evaluated with leave-one-participant-out predictions;
- retains the participant-matched genuine/pseudo difference-in-differences so a
  phenotype is not mislabeled as an onset-specific biomarker.

Run the complete reusable pipeline with:

```powershell
run_everything.cmd
```

The one interactive step is confirming the six global labels if the reusable
hierarchy has not already completed. Existing microstate fits, ECG caches and
completed upstream stages are reused.

To rebuild only the global-first publication analysis from retained data:

```powershell
.\.venv\Scripts\python.exe -B analyse_global_transition_exploration.py `
  --peak-windows-csv "C:\Users\Admin\OneDrive - NHS\FS_MS\balanced\data\directional_peak_backfit_windows.csv" `
  --common-windows-csv "C:\Users\Admin\OneDrive - NHS\FS_MS\balanced\data\autonomic_peak_microstate_common_windows.csv.gz" `
  --focused-dir "C:\Users\Admin\OneDrive - NHS\FS_MS\balanced" `
  --out-dir "C:\Users\Admin\OneDrive - NHS\FS_MS\publication_final" `
  --review-dir "C:\Users\Admin\OneDrive - NHS\FS_MS\for_review" `
  --overwrite
```

`publication_final` contains five 400-dpi/vector publication figures, exact
analysis frames, PCA loadings/scores, participant-held-out predictions,
exploratory source models, matched-pseudo balance, decisions and hashes.
`for_review` is a self-contained small review package with the single analysis
script, compressed reused inputs, all result tables and figures.

The prior confirmatory hierarchy and 3D displays are not part of the headline
pipeline: the attempted confirmatory fits failed validation, and the 2D
four-variable manifold is more complete and easier to interpret.
