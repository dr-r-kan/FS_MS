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

To tweak and regenerate every figure without rerunning ECG processing or any
Bayesian model, edit `plot_focused_fs_ms.py` and run:

```powershell
.\.venv\Scripts\python.exe -B plot_focused_fs_ms.py --analysis-dir "G:\FS_microstates_study\vg_microstates\focused_analysis_final"
```

The plotting script reads only saved CSV/NPZ files under `data\` and overwrites
the paper PNG files under `figures\`, then records the renderer and figure hashes
in `PLOT_AUDIT.json`. A full `run_everything.cmd` run invokes the same script as
its final `focused_plots` stage.

The focused run also performs the feedback-driven transition sensitivities:
raw C-exit count posteriors, fixed-30-exit and matched-60-second comparisons,
matched interictal pseudo-onsets, route/concentration and destination
decomposition, all-source specificity, C-reference threshold sensitivity,
preictal trajectories, and lagged autonomic models for C coverage, JS and κ.
Requested respiration, movement/EMG, CFA variants, alternate segmentations and
VR links require source exports that are not present and are listed explicitly
in `data\feedback_analysis_summary.json` rather than imputed.

The optimised hierarchy reuses those saved microstate/count exports and writes
only the new model outputs:

```powershell
.\.venv\Scripts\python.exe -B analyse_optimised_transition_model.py
```

Its default output is
`C:\Users\Admin\OneDrive - NHS\FS_MS\equalised and optimised`. It fits one
hierarchical multinomial count model, derives the three primary transition
outcomes, tests the planned early/late contrasts, defines interictal autonomic
PCs, and compares the five dynamic models by leave-one-event-out prediction.
`run_everything.cmd` invokes it automatically after the focused analysis stage;
override its output with `-OptimisedAnalysis "D:\path\to\output"`.

## Output directories

Defaults are the `$Results` and `$Analysis` parameters at the top of
`run_ace_manual_pipeline.ps1`. Override either without editing the file:

```powershell
run_everything.cmd -Analysis "C:\Users\Admin\OneDrive - NHS\FS_MS\focused_analysis_final"
```

`-Results` holds the large fitted hierarchy. `-Analysis` holds `RESULTS.md`,
figures, tables, logs, and analysis caches. Keep `-Results` on `G:` unless you
specifically want the hierarchy synchronized by OneDrive.

Files retired from earlier analyses are recoverable under `OLD/`, which is
excluded from Git.
