"""Stage DAG and runner.

    clips_check -> heppy -> manifest -> microstates -> autonomic -> stats_prepare -> stats_verify -> [stats_confirm]

Every stage is idempotent: it records a fingerprint (config section + code hashes + upstream fingerprints
+ input listing) and its outputs in ``<run_root>/state/<stage>.done.json``; a rerun with nothing changed
executes nothing.  A changed stage reruns from scratch (never from half-written state) and invalidates
everything downstream.  ``stats_confirm`` only runs when explicitly requested and, for real data, only after
registration.  Nothing here ever reads the real study data unless ``fsms.lock`` says it is unlocked.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import psutil

from . import lock as planlock
from . import provenance
from .atomic import atomic_write_json, fingerprint, json_content_digest, read_json, sha256_file
from .config import Config
from .guard import assert_unrestricted
from .ledger import Ledger, Timer

STAGE_ORDER = ("clips_check", "heppy", "manifest", "microstates", "autonomic",
               "stats_prepare", "stats_verify", "stats_confirm")
PYTHON = sys.executable


class PipelineError(RuntimeError):
    pass


# ------------------------------------------------------------------ helpers
def code_hash(config: Config, *patterns: str) -> str:
    items = {}
    for pattern in patterns:
        for path in sorted(config.repo.glob(pattern)):
            if path.is_file() and "__pycache__" not in path.parts:
                items[path.relative_to(config.repo).as_posix()] = sha256_file(path)
    if not items:
        raise PipelineError(f"No code files match {patterns}")
    return fingerprint(items)


def file_listing(directory: Path, suffix: str) -> list[list]:
    """Name+size (no content hashing: clips can be tens of GB; per-clip sentinels hash what matters)."""
    return sorted([p.name, p.stat().st_size] for p in directory.glob(f"*{suffix}") if p.is_file())


def csv_column(path: Path, column: str) -> list[str]:
    import csv
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [row[column] for row in csv.DictReader(handle) if row.get(column)]


def existing(paths) -> list[Path]:
    return [Path(p) for p in paths if p and Path(p).is_file()]


def run_logged(command: list[str], log_path: Path, cwd: Path, env: dict | None = None) -> int:
    """Run a subprocess, mirroring combined output to the console and a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", **(env or {})}
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n===== {datetime.now(timezone.utc).isoformat()} :: {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")

        def pump():
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print("   | " + line.rstrip(), flush=True)

        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        code = process.wait()
        thread.join()
        log.write(f"===== exit {code}\n")
    return code


class RunLock:
    """One pipeline run per run_root; stale locks from dead processes are reclaimed."""

    def __init__(self, directory: Path):
        self.path = directory / ".run.lock"
        self.directory = directory

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(handle, json.dumps({"pid": os.getpid(), "created": time.time()}).encode())
                os.close(handle)
                return self
            except FileExistsError:
                try:
                    pid = json.loads(self.path.read_text()).get("pid")
                except (OSError, ValueError):
                    pid = None
                if pid and psutil.pid_exists(pid):
                    raise PipelineError(f"Another pipeline run (pid {pid}) is using this run_root")
                self.path.unlink(missing_ok=True)
        raise PipelineError("Could not acquire the run lock")

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)
        return False


@dataclass
class Stage:
    name: str
    fingerprint: Callable[["Context"], str]
    run: Callable[["Context"], None]
    outputs: Callable[["Context"], list[Path]]
    touches_data: bool = True
    optional: bool = False


@dataclass
class Context:
    config: Config
    ledger: Ledger
    fingerprints: dict = field(default_factory=dict)
    confirm: bool = False
    force_stage: str | None = None

    @property
    def log(self) -> Path:
        return self.config.log_dir / "pipeline.log"

    def python_module(self, module: str, *args: str, stage: str) -> int:
        return run_logged([PYTHON, "-B", "-m", module, *args], self.config.log_dir / f"{stage}.log", self.config.repo)


# ------------------------------------------------------------------ stage implementations
def _clips_fp(ctx: Context) -> str:
    c = ctx.config
    return fingerprint({"mode": c.get("clips.mode"), "clips": file_listing(c.clips_dir, ".edf"),
                        "database": sha256_file(c.database_csv), "keyfile": sha256_file(c.snip_keyfile)})


def _clips_run(ctx: Context) -> None:
    c = ctx.config
    if c.get("clips.mode") != "adopt":
        raise PipelineError("clips.mode 'build' is not enabled in this configuration: the real clips already "
                            "exist and the pipeline never rewrites them. Use fsms.clips.snip_clips directly.")
    missing = [str(p) for p in (c.clips_dir, c.database_csv, c.snip_keyfile) if not p.exists()]
    if missing:
        raise PipelineError("Missing inputs: " + ", ".join(missing))
    if not list(c.clips_dir.glob("*.edf")):
        raise PipelineError(f"No clip EDFs in {c.clips_dir}")
    atomic_write_json(c.state_dir / "clips_check.json", {"edf_clips": len(list(c.clips_dir.glob('*.edf')))})


def _heppy_options(c: Config) -> Path:
    return c.repo_file(c.get("heppy.options"))


def _heppy_fp(ctx: Context) -> str:
    c = ctx.config
    return fingerprint({"upstream": ctx.fingerprints["clips_check"], "options": sha256_file(_heppy_options(c)),
                        "code": code_hash(c, "fsms/heppy/*.py", "fsms/atomic.py"),
                        "vendored": provenance.tree_hash(c.repo / "external" / "HEPPy")[0],
                        "adopt": bool(c.get("heppy.adopt_existing", False)),
                        "workers_ignored_for_results": True})


def _heppy_run(ctx: Context) -> None:
    c = ctx.config
    args = ["--clips-dir", str(c.clips_dir), "--heppy-out", str(c.heppy_out),
            "--options", str(_heppy_options(c)), "--workers", str(c.get("heppy.workers", 1)),
            "--mode", c.kind]
    if c.get("heppy.adopt_existing", False) and c.adopt_heppy_dir:
        args += ["--adopt-dir", str(c.adopt_heppy_dir)]
    if ctx.config.kind == "fixture":
        assert_unrestricted(c.heppy_out, "HEPPy output")
    if ctx.python_module("fsms.heppy.run_heppy_stage", *args, stage="heppy") != 0:
        raise PipelineError("HEPPy stage failed (infrastructure error); see logs/heppy.log")


def _heppy_outputs(ctx: Context) -> list[Path]:
    """Summary files plus every per-recording product, so deleting one FIF/sidecar makes the stage rerun."""
    from .autonomic import sidecar_for
    review = ctx.config.heppy_out / "ecg_qc_review.csv"
    products: list[str] = []
    for column in ("raw_fif", "raw_fif_keep"):
        products += csv_column(review, column)
    products += [str(sidecar_for(p)) for p in csv_column(review, "raw_fif")]
    return [ctx.config.heppy_out / "heppy_summary.json", review, *existing(products)]


def _manifest_fp(ctx: Context) -> str:
    c = ctx.config
    return fingerprint({"upstream": ctx.fingerprints["heppy"], "manifest": c.get("manifest"),
                        "code": code_hash(c, "fsms/manifest/*.py"),
                        "converter": sha256_file(c.repo / "external/VG-microstates/_internal/python/_convert_fif_manifest_to_set.py"),
                        "database": sha256_file(c.database_csv), "keyfile": sha256_file(c.snip_keyfile),
                        "heppy_summary": sha256_file(c.heppy_out / "heppy_summary.json")})


def _manifest_run(ctx: Context) -> None:
    c = ctx.config
    args = ["--snip-keyfile", str(c.snip_keyfile), "--heppy-out", str(c.heppy_out),
            "--database", str(c.database_csv), "--out-dir", str(c.vg_dir),
            "--format", c.get("manifest.format", "mat"), "--montage", c.get("manifest.montage", "standard_1020"),
            "--min-duration-s", str(c.get("manifest.min_duration_s", 10)), "--mode", c.kind]
    if ctx.python_module("fsms.manifest.build_vg_manifest", *args, stage="manifest") != 0:
        raise PipelineError("Manifest stage failed; see logs/manifest.log")


def _manifest_outputs(ctx: Context) -> list[Path]:
    return [ctx.config.manifest_mat, *existing(csv_column(ctx.config.manifest_mat, "file_path"))]


def hierarchy_dir(c: Config) -> Path:
    params = fingerprint({"manifest": sha256_file(c.manifest_mat), "microstates": c.get("microstates"),
                          "vg": provenance.tree_hash(c.repo / "external" / "VG-microstates")[0],
                          "code": code_hash(c, "fsms/matlab/*.m", "run_vg_microstates.m",
                                            "recover_timestamped_transition_backfits.m",
                                            "export_directional_peak_windows.m")})
    return c.vg_dir / f"hierarchy_{params[:12]}"


def _micro_fp(ctx: Context) -> str:
    c = ctx.config
    return fingerprint({"upstream": ctx.fingerprints["manifest"], "dir": hierarchy_dir(c).name})


def _micro_run(ctx: Context) -> None:
    c = ctx.config
    out = hierarchy_dir(c)
    quote = lambda value: str(value).replace("'", "''")
    # The MATLAB-side restricted-path guard is lifted only for a real-study run, i.e. after real_data_gate() passed.
    real_mode = "true" if c.kind == "real" else "false"
    command = (f"addpath('{quote(c.repo)}'); addpath(genpath(fullfile('{quote(c.repo)}','fsms','matlab'))); "
               f"run_study_hierarchy('{quote(c.manifest_mat)}','{quote(out)}','real_study_mode',{real_mode});")
    matlab = shutil.which("matlab")
    if not matlab:
        raise PipelineError("matlab executable not found on PATH")
    if run_logged([matlab, "-batch", command], c.log_dir / "microstates.log", c.repo) != 0:
        raise PipelineError("MATLAB VG-microstates stage failed; see logs/microstates.log")


def _micro_outputs(ctx: Context) -> list[Path]:
    root = hierarchy_dir(ctx.config)
    out = root / "timestamped_backfits"
    glob = root / "hierarchy" / "global"
    return [out / "transition_index.csv", out / "recovery_audit.json", root / "STAGE_C_DONE.json",
            glob / "global_solution.mat", glob / "global_manual_labels.csv", glob / "labelling_audit.json",
            *existing(csv_column(out / "transition_index.csv", "cache_file"))]


def _auto_fp(ctx: Context) -> str:
    c = ctx.config
    return fingerprint({"upstream": ctx.fingerprints["manifest"], "code": code_hash(c, "fsms/autonomic.py"),
                        "heppy": ctx.fingerprints["heppy"]})


def _auto_run(ctx: Context) -> None:
    c = ctx.config
    if ctx.python_module("fsms.autonomic", "--manifest", str(c.manifest_mat), "--out-dir", str(c.autonomic_dir),
                         stage="autonomic") != 0:
        raise PipelineError("Autonomic stage failed; see logs/autonomic.log")


def _auto_outputs(ctx: Context) -> list[Path]:
    return [ctx.config.autonomic_native, ctx.config.autonomic_dir / "autonomic_audit.json"]


def _analysis_args(ctx: Context) -> list[str]:
    c = ctx.config
    out = hierarchy_dir(c) / "timestamped_backfits"
    return ["--transition-index", str(out / "transition_index.csv"), "--autonomic-native", str(c.autonomic_native),
            "--manifest", str(c.manifest_mat), "--out-dir", str(c.analysis_dir),
            "--plan-lock", str(planlock.LOCK_PATH)]


def _prepare_fp(ctx: Context) -> str:
    c = ctx.config
    index_dir = hierarchy_dir(c) / "timestamped_backfits"
    return fingerprint({"upstream": [ctx.fingerprints["microstates"], ctx.fingerprints["autonomic"]],
                        "analysis_code": sha256_file(c.repo / "analyse_timestamped_transition_dynamics.py"),
                        "analysis": c.get("analysis"),
                        "index": sha256_file(index_dir / "transition_index.csv"),
                        "audit": json_content_digest(index_dir / "recovery_audit.json"),
                        "autonomic": sha256_file(c.autonomic_native), "manifest": sha256_file(c.manifest_mat)})


def _archive_stale(ctx: Context, directory: Path) -> None:
    if not directory.exists() or not any(directory.iterdir()):
        return
    if (directory / "tables" / "confirmatory_results.csv").exists():
        raise PipelineError(f"{directory} already holds confirmatory results; they are preserved, never replaced. "
                            "Use a new run_root for a new analysis.")
    target = directory.parent / f"archive_prepared_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    shutil.move(str(directory), str(target))
    print(f"   archived stale preparation to {target}")


def _prepare_run(ctx: Context) -> None:
    _archive_stale(ctx, ctx.config.analysis_dir)
    ctx.config.analysis_dir.parent.mkdir(parents=True, exist_ok=True)
    code = run_logged([PYTHON, "-B", "analyse_timestamped_transition_dynamics.py", *_analysis_args(ctx), "--prepare"],
                      ctx.config.log_dir / "stats_prepare.log", ctx.config.repo)
    if code != 0:
        raise PipelineError("Descriptive preparation failed; see logs/stats_prepare.log")


def _prepare_outputs(ctx: Context) -> list[Path]:
    a = ctx.config.analysis_dir
    return [a / "PREREGISTRATION_LOCK.json", a / "PREREGISTRATION_LOCK.sha256", a / "ANALYSIS_AUDIT.json",
            a / "tables" / "timestamped_window_metrics.csv.gz"]


def _verify_fp(ctx: Context) -> str:
    return fingerprint({"prepare": ctx.fingerprints["stats_prepare"]})


def _verify_run(ctx: Context) -> None:
    code = run_logged([PYTHON, "-B", "analyse_timestamped_transition_dynamics.py", *_analysis_args(ctx),
                       "--verify-lock"], ctx.config.log_dir / "stats_verify.log", ctx.config.repo)
    if code != 0:
        raise PipelineError("Data-freeze lock verification failed; see logs/stats_verify.log")
    atomic_write_json(ctx.config.state_dir / "stats_verify.json", {"verified_utc": datetime.now(timezone.utc).isoformat()})


def registration_id(ctx: Context) -> str:
    c = ctx.config
    if c.kind == "fixture":
        return c.get("analysis.registration_id_placeholder")
    return read_json(planlock.REGISTRATION_PATH)["registration_url"]


def _confirm_fp(ctx: Context) -> str:
    return fingerprint({"verify": ctx.fingerprints["stats_verify"], "registration": registration_id(ctx)})


def _confirm_run(ctx: Context) -> None:
    code = run_logged([PYTHON, "-B", "analyse_timestamped_transition_dynamics.py", *_analysis_args(ctx),
                       "--run-confirmatory", "--preregistration-id", registration_id(ctx)],
                      ctx.config.log_dir / "stats_confirm.log", ctx.config.repo)
    if code != 0:
        raise PipelineError("Confirmatory run failed; inspect logs/stats_confirm.log and the partial outputs. "
                            "Do not delete evidence and rerun blindly (see docs/RUNBOOK.md).")


def _confirm_outputs(ctx: Context) -> list[Path]:
    a = ctx.config.analysis_dir
    return [a / "tables" / "confirmatory_results.csv", a / "CONFIRMATORY_EXECUTION.json"]


STAGES: dict[str, Stage] = {
    "clips_check": Stage("clips_check", _clips_fp, _clips_run,
                         lambda ctx: [ctx.config.state_dir / "clips_check.json"]),
    "heppy": Stage("heppy", _heppy_fp, _heppy_run, _heppy_outputs),
    "manifest": Stage("manifest", _manifest_fp, _manifest_run, _manifest_outputs),
    "microstates": Stage("microstates", _micro_fp, _micro_run, _micro_outputs),
    "autonomic": Stage("autonomic", _auto_fp, _auto_run, _auto_outputs),
    "stats_prepare": Stage("stats_prepare", _prepare_fp, _prepare_run, _prepare_outputs),
    "stats_verify": Stage("stats_verify", _verify_fp, _verify_run,
                          lambda ctx: [ctx.config.state_dir / "stats_verify.json"]),
    "stats_confirm": Stage("stats_confirm", _confirm_fp, _confirm_run, _confirm_outputs, optional=True),
}


# ------------------------------------------------------------------ gates and driver
def real_data_gate(config: Config) -> None:
    if config.kind != "real":
        return
    problems = planlock.real_data_unlock_problems()
    if problems:
        raise PipelineError("Real study data is LOCKED. Resolve before any data-touching stage:\n  - "
                            + "\n  - ".join(problems))


def run(config: Config, through: str = "stats_verify", confirm: bool = False, force: str | None = None) -> dict:
    if through not in STAGE_ORDER:
        raise PipelineError(f"Unknown stage {through!r}; choose from {STAGE_ORDER}")
    if through == "stats_confirm" or confirm:
        confirm, through = True, "stats_confirm"  # --confirm means: also run the registered confirmatory tests
    real_data_gate(config)
    config.assert_paths_allowed(config.data_root, config.run_root, purpose="fixture run")
    ledger = Ledger(config.state_dir)
    ctx = Context(config, ledger, confirm=confirm, force_stage=force)
    report: dict[str, str] = {}
    with RunLock(config.state_dir):
        for name in STAGE_ORDER[: STAGE_ORDER.index(through) + 1]:
            stage = STAGES[name]
            if stage.optional and not confirm:
                continue
            ctx.fingerprints[name] = stage.fingerprint(ctx)
            current, reason = ledger.verify(name, ctx.fingerprints[name])
            if current and force != name:
                print(f"[SKIP] {name} (current)")
                report[name] = "skipped"
                continue
            print(f"[RUN ] {name}  ({reason})")
            with Timer() as timer:
                stage.run(ctx)
            ctx.fingerprints[name] = stage.fingerprint(ctx)  # outputs now exist; fingerprint may include them
            ledger.mark_done(name, ctx.fingerprints[name], stage.outputs(ctx), seconds=timer.seconds)
            print(f"[DONE] {name}  ({timer.seconds:.0f}s)")
            report[name] = "ran"
    return report


def status(config: Config) -> dict[str, str]:
    ledger = Ledger(config.state_dir)
    result = {}
    for name in STAGE_ORDER:
        record = ledger.read(name)
        result[name] = "never run" if record is None else f"done {record['completed_utc']}"
    return result


def preflight(config: Config | None, deep: bool = False) -> list[tuple[str, bool, str]]:
    """Static environment checks; never reads study data."""
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    import platform
    add("python", platform.python_version().startswith("3.12"), platform.python_version())
    mismatches = planlock.runtime_mismatches()
    add("pinned packages", not mismatches, "; ".join(mismatches) or "match requirements-lock.txt")
    problems = provenance.verify()
    add("vendored externals", not problems, "; ".join(problems) or "match external/PROVENANCE.json")
    matlab = shutil.which("matlab")
    add("matlab on PATH", matlab, matlab or "not found")
    lock_problems = planlock.verify_lock()
    add("plan lock", not lock_problems, "; ".join(lock_problems[:3]) or "verified")
    add("registration record", planlock.REGISTRATION_PATH.exists(),
        "present" if planlock.REGISTRATION_PATH.exists() else "absent (real data stays locked)")
    if config is not None:
        add(f"config ({config.kind})", True, str(config.path))
        add("run_root outside restricted storage", not (config.kind == "real" and
                                                         str(config.run_root).lower().startswith(str(config.data_root).lower())),
            str(config.run_root))
        anchor = config.run_root
        while not anchor.exists() and anchor.parent != anchor:
            anchor = anchor.parent
        free = shutil.disk_usage(anchor).free / 1e9
        add("free disk (GB)", free > 100, f"{free:.0f}")
        if config.kind == "real":
            gate = planlock.real_data_unlock_problems()
            add("real-data gate", not gate, "open" if not gate else "LOCKED (expected before registration)")
    if deep and matlab:
        code = subprocess.run([matlab, "-batch", "assert(~isempty(which('spm_mix')) && ~isempty(which('pop_loadset')));"],
                              capture_output=True, text=True, timeout=600).returncode
        add("MATLAB SPM + EEGLAB on path", code == 0)
    return checks
