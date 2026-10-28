# FS_MS — preregistration-ready microstate transition pipeline

Does the change in autonomic-gradient-conditioned EEG microstate transition organisation approaching seizure
onset differ between functional-seizure-only (FS) and epileptic-seizure-only (ES) participants?

**Background.** The work is predicated on an Active Inference account of dissociative symptoms. HEPPy preprocesses
the EEG/ECG, variational Bayesian (GMM) microstates are fitted, and the analysis then looks at the "out of C" flow
between microstates: how far the transition distribution has drifted from the participant's interictal baseline, and
how precise (concentrated) it is.

This repository runs the whole analysis from raw EDF clips to the registered confirmatory tests:

```
clips (EDF) → HEPPy EEG/ECG preprocessing → VG-microstates (K=6 VB-GMM, template labels, exact transitions)
            → CSI/CPI autonomic table → timestamped transition tensor → preregistered tests
```

* **Preregistered question and plan:** [`PREREGISTRATION.md`](PREREGISTRATION.md),
  [`docs/STATISTICAL_ANALYSIS_PLAN.md`](docs/STATISTICAL_ANALYSIS_PLAN.md)
* **How to run, lock and register:** [`docs/RUNBOOK.md`](docs/RUNBOOK.md)

## Quick start

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v   # synthetic tests, no data
.\run_pipeline.ps1 preflight --deep                               # environment, MATLAB/SPM/EEGLAB, locks
```

Validate the full chain on Temple University EEG (TUEG) recordings with synthetic labels — no real data:

```powershell
$env:FSMS_RUN_SLOW = '1'
.\run_pipeline.ps1 fixture
.\run_pipeline.ps1 run --config config\fixture_tueg.json --confirm
```

## Design guarantees

| Guarantee | How |
|---|---|
| Real data stays untouched until registration | `config/study.json` is `kind: real`; `fsms.lock` refuses every data-touching stage until `register` records the immutable registration against the verified plan lock. Fixture configs are rejected if any path is inside restricted study storage. |
| Idempotent | every stage records a fingerprint (parameters + code + upstream outputs) and its output hashes; a rerun with no change executes nothing and rewrites nothing; per-recording sentinels in HEPPy; atomic writes; stale preparations are archived, never deleted. |
| Reproducible | pinned environment (`requirements-lock.txt`, 44 packages), explicit seeds, vendored external code with tree hashes (`external/PROVENANCE.json`), line-ending-independent hashing. |
| No analyst degrees of freedom in preprocessing | HEPPy exclusion rule, microstate labelling rule (`template_prespecified_v2`) and ECG/autonomic QC thresholds are fixed in code/config, group-blind, reason-coded; nothing is hand-labelled. |
| Confirmatory tests only after registration | `stats_confirm` needs `--confirm`; verifies the data-freeze lock first; refuses to overwrite existing results. |

## Layout

| Path | Purpose |
|---|---|
| `run_pipeline.py`, `run_pipeline.ps1` | single entry point |
| `fsms/` | orchestration (`pipeline`, `ledger`, `lock`, `guard`, `config`, `provenance`), stage drivers (`heppy/`, `clips/`, `manifest/`, `matlab/`, `autonomic.py`), TUEG `fixture/` |
| `analyse_timestamped_transition_dynamics.py` | measurement and inference (prepare / verify-lock / confirmatory) |
| `config/` | study and fixture configuration, HEPPy options |
| `prereg/` | plan lock and (after registration) the registration record |
| `external/` | vendored HEPPy and VG-microstates with provenance |
| `tests/` | synthetic unit tests; opt-in slow end-to-end fixture test |
| `legacy/`, root `analyse_*.py`, `build_*.py` | earlier exploratory work, kept for provenance; not part of the registered pipeline |

External code: [HEPPy](https://github.com/dr-r-kan/HEPPy) and VG-microstates (private repository
`dr-r-kan/VG-microstates`; the vendored tree is a modified derivative, see `external/PROVENANCE.json`).
Requires Windows, Python 3.12.7, MATLAB R2025a with [SPM](https://github.com/spm/spm) and EEGLAB.
