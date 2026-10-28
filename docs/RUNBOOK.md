# Runbook: set up, validate, preregister, execute

One entry point: `run_pipeline.ps1` (Windows) / `run_pipeline.py`. The legacy runners in `legacy/` are
**not** safe for the registered study.

The workflow is built around one rule: **the real study data (`F:\FS_microstates_study`) is not read, listed
or written until the plan is locked and registered.** Everything before registration — software tests, the
end-to-end validation run, the lock — uses only synthetic data and the Temple University EEG corpus (TUEG).

```
 (A) install + test        (B) validate end to end         (C) lock + register       (D) execute once
 synthetic unit tests  ->  TUEG fixture, MATLAB, HEPPy  ->  plan-lock, commit, OSF ->  run on real data
 no data                   synthetic labels only            no data read              unlocked by registration
```

## A. Install and test (no data)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\run_pipeline.ps1 preflight --deep
```

`preflight` checks the interpreter, every pinned package, the vendored external code hashes, MATLAB with
SPM and EEGLAB on the path (`--deep`), the plan lock and the registration record. It reads no study data.

## B. Validate the whole chain on TUEG (no real data)

```powershell
$env:FSMS_RUN_SLOW = '1'
.\run_pipeline.ps1 fixture                                   # builds F:\FS_MS_fixture_tueg from TUEG headers
.\run_pipeline.ps1 run --config config\fixture_tueg.json --confirm
.\run_pipeline.ps1 run --config config\fixture_tueg.json --confirm   # second run: every stage [SKIP]
```

The fixture uses real TUEG recordings with **synthetic** diagnostic groups, semiology and seizure times,
so its statistics mean nothing; it exists to prove the unmodified code runs from clips to confirmatory
tests and is idempotent. The fixture config is validated by `fsms.guard`: a path inside restricted
study storage is refused before any stage runs.

Stages (each skipped when its fingerprint is current): `clips_check → heppy → manifest → microstates →
autonomic → stats_prepare → stats_verify → stats_confirm (only with --confirm)`.

## C. Lock and register (no data read)

1. Review `PREREGISTRATION.md`, `docs/STATISTICAL_ANALYSIS_PLAN.md`, `config/study.json`, `config/heppy_options.json`.
   Fill in the prior-access disclosure honestly (see the section in `PREREGISTRATION.md`).
2. `.\run_pipeline.ps1 plan-lock` — writes `prereg/PLAN_LOCK.json` + `.sha256`: hashes of all code, configs,
   the analysis plan, the pinned package list and the vendored HEPPy/VG-microstates trees. No participant data.
3. Commit `prereg/` and everything else; re-run `plan-lock` if it warns about uncommitted changes.
   `.\run_pipeline.ps1 verify-lock` must pass.
4. Register on OSF: upload `PREREGISTRATION.md`, `docs/STATISTICAL_ANALYSIS_PLAN.md`, this runbook,
   `prereg/PLAN_LOCK.json`/`.sha256` and the commit id (and an archive of the repository).
5. `.\run_pipeline.ps1 register --url https://osf.io/<id>` — verifies the lock, checks the URL syntax and
   writes `prereg/REGISTRATION.json` (never overwritten). **This is what unlocks the real data.** The record
   is the investigator's attestation; nothing here certifies the OSF record itself.

After registration, any change to a locked file makes `verify-lock` fail and the real-data run refuse to start.
Need a change? Make it, record it as a deviation/amendment, and register the amendment.

## D. Execute on the real data (once)

```powershell
.\run_pipeline.ps1 preflight --config config\study.json
.\run_pipeline.ps1 run --config config\study.json            # through stats_verify: prepared + data-frozen, NO tests
.\run_pipeline.ps1 run --config config\study.json --confirm  # the registered confirmatory tests
```

* Outputs go to `run_root` (`F:\FS_MS_analysis\prereg_v2_run`); nothing is ever written into the study folder.
* Existing HEPPy output is adopted in place only if its recorded options match the locked options and the files
  validate; otherwise the recording is reprocessed. Adoption decisions are in `heppy_summary.json`.
* Interrupted? Re-run the same command; completed stages and completed recordings are skipped, a half-written
  file is never trusted (atomic writes, fingerprint sentinels).
* A change in any input or setting reruns that stage and everything after it. A stale preparation is
  **archived, not deleted**; existing confirmatory results are protected and never replaced.
* Paths in `config/study.json` could not be verified before registration; the first run reports any mismatch
  before processing. Fixing a path is a configuration amendment (re-lock/re-register before running).

## What stops the run (by design)

| Condition | Behaviour |
|---|---|
| plan not locked/registered, or any locked file changed | real data stays locked |
| technical labelling failure (no usable channel match, too few states) | `LABELLING_REJECTED.json`, run stops; investigator decides (deviation). Weak template similarity only sets a reported flag |
| recording fails the pre-specified QC rule | excluded with a reason code, listed in `heppy_exclusions.csv`; run continues |
| transition cache / backfit validation gate fails | `stats_prepare` refuses |
| existing confirmatory results | preserved; refuses to overwrite |
