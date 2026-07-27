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
