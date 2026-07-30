# FS_MS focused analysis

The retained pipeline starts from the converted MAT manifest and runs the K=6
hierarchy, manual global labelling, participant backfit, and focused cardiac /
microstate analyses.

Run:

```powershell
run_everything.cmd
```

The only interactive step is entering and confirming the six global labels.
Completed stages are reused on rerun.

The current publication pipeline is deliberately non-duplicative. It retains
cohort/event context, heart rate, microstate coverage, autonomic levels and
postictal recovery. The hierarchical multinomial model replaces the legacy
C-origin transition models, raw-count sensitivities, route tests, pairwise
source-specificity tests, autonomic kitchen-sink associations and long
predictive-flow forests. The final publication stage combines only the retained
and replacement results.

To rebuild the curated report and publication figures from saved model outputs:

```powershell
.\.venv\Scripts\python.exe -B build_publication_fs_ms.py `
  --focused-dir "C:\Users\Admin\OneDrive - NHS\FS_MS\balanced" `
  --optimised-dir "C:\Users\Admin\OneDrive - NHS\FS_MS\balanced\optimised" `
  --out-dir "C:\Users\Admin\OneDrive - NHS\FS_MS\publication_final"
```

The final folder contains ten grouped figures as both 400-dpi PNG and vector PDF,
curated result tables, an explicit keep/replace ledger, and complete hashes.
Separate matched-axis 3D views show κ, J and CPI for FS and ES with event-bin
jitter and time-bin centroid paths.

The optimised hierarchy can still be run directly:

```powershell
.\.venv\Scripts\python.exe -B analyse_optimised_transition_model.py
```

It fits one hierarchical multinomial count model, derives the three primary
transition outcomes, tests the planned early/late contrasts, defines interictal
autonomic PCs, and compares five dynamic models by leave-one-event-out
prediction. `run_everything.cmd` runs the retained-only analysis, the optimised
model, then the publication builder. Override the final folder with
`-PublicationAnalysis "D:\path\to\output"`.

## Output directories

Defaults are the `$Results` and `$Analysis` parameters at the top of
`run_ace_manual_pipeline.ps1`. Override either without editing the file:

```powershell
run_everything.cmd -Analysis "C:\Users\Admin\OneDrive - NHS\FS_MS\focused_analysis_final"
```

`-Results` holds the large fitted hierarchy. `-Analysis` holds retained inputs,
logs, and reusable caches; `-OptimisedAnalysis` holds the replacement-model
outputs; and `-PublicationAnalysis` is the small curated report with the final
figures and tables. Keep `-Results` on `G:` unless you specifically want the
hierarchy synchronized by OneDrive.

Files retired from earlier analyses are recoverable under `OLD/`, which is
excluded from Git.
