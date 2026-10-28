# Legacy runners — do not use for the preregistered study

These scripts drove the earlier, exploratory analysis (`run_everything.cmd`,
`run_ace_manual_pipeline.ps1`) and the intermediate timestamped pipeline
(`run_timestamped_transition_pipeline.ps1`). They are kept for provenance only.

They are **unsafe** for the registered analysis because they:

- hard-code `G:\FS_microstates_study` paths and interactive, hand-typed microstate labels;
- force-rerun stages unconditionally and prepare outputs from the real data *before* registration;
- regenerate upstream inputs, so they cannot guarantee that a registered input is unchanged.

Use `run_pipeline.ps1` / `run_pipeline.py` (see `docs/RUNBOOK.md`). It is idempotent, has a
pre-specified deterministic microstate labelling rule, and refuses to read the real study
data until the plan is locked and registered.
