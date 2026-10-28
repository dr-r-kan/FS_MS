"""Data-free preregistration lock and the real-data unlock gate.

Workflow (see docs/RUNBOOK.md):

1. ``run_pipeline.py plan-lock``  hash code + configs + analysis plan + vendored externals (no data read).
2. Investigator commits ``prereg/`` and registers the commit and lock on OSF.
3. ``run_pipeline.py register --url https://osf.io/...``  records the immutable registration URL next to the
   lock hash; only then can a ``kind: real`` config read the restricted study data.
4. Every real run re-verifies the lock first, so any later change to code, configs, plan or externals stops it.

The lock contains no participant data, paths to participants, or data-derived numbers.
"""
from __future__ import annotations

import importlib.metadata
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from . import provenance
from .atomic import atomic_write_json, atomic_write_text, read_json, sha256_normalised
from .config import REPO

LOCK_SCHEMA = "fs-ms-plan-lock/v3"
PREREG_DIR = REPO / "prereg"
LOCK_PATH = PREREG_DIR / "PLAN_LOCK.json"
SIDECAR_PATH = PREREG_DIR / "PLAN_LOCK.sha256"
REGISTRATION_PATH = PREREG_DIR / "REGISTRATION.json"

# Files whose content defines the analysis.  Tests and operational docs are intentionally excluded.
LOCKED_PATTERNS = (
    "PREREGISTRATION.md",
    "docs/STATISTICAL_ANALYSIS_PLAN.md",
    "requirements-lock.txt",
    "run_pipeline.py",
    "run_pipeline.ps1",
    "analyse_timestamped_transition_dynamics.py",
    "microstate_labels.py",
    "*.m",
    "config/*.json",
    "fsms/**/*.py",
    "fsms/**/*.m",
    "external/PROVENANCE.json",
)
EXCLUDED_PARTS = {"__pycache__"}


class LockError(RuntimeError):
    pass


def locked_files() -> dict[str, str]:
    found: dict[str, str] = {}
    for pattern in LOCKED_PATTERNS:
        for path in sorted(REPO.glob(pattern)):
            if path.is_file() and not (set(path.relative_to(REPO).parts) & EXCLUDED_PARTS):
                found[path.relative_to(REPO).as_posix()] = sha256_normalised(path)
    return dict(sorted(found.items()))


def pinned_requirements() -> dict[str, str]:
    pins = {}
    path = REPO / "requirements-lock.txt"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;#]+)", line.strip())
            if match:
                pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    return pins


def installed_versions() -> dict[str, str]:
    versions = {}
    for name in pinned_requirements():
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT INSTALLED"
    return versions


def runtime_mismatches() -> list[str]:
    pins, installed = pinned_requirements(), installed_versions()
    return [f"{name}: pinned {pins[name]}, installed {installed[name]}"
            for name in pins if installed[name] != pins[name]]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              timeout=30, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_state() -> dict:
    dirty = [line for line in _git("status", "--porcelain").splitlines()
             if "prereg/" not in line.replace("\\", "/")]
    return {"head": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "uncommitted_paths_excluding_prereg": [line[3:] for line in dirty]}


def validate_registration_url(value: str) -> None:
    url = urlparse(value.strip())
    if (url.scheme != "https" or url.hostname not in {"osf.io", "doi.org"} or not url.path.strip("/")
            or any(word in value.lower() for word in ("replace", "placeholder", "example", "fixture", "synthetic"))):
        raise LockError("Supply the actual immutable registration HTTPS URL at osf.io or doi.org")


def create_lock() -> dict:
    if REGISTRATION_PATH.exists():
        raise LockError("A registration record exists; the lock is frozen. Create a new amendment instead of "
                        "relocking (delete nothing; document the amendment and register it).")
    problems = provenance.verify()
    if problems:
        raise LockError("External provenance is stale ('python -m fsms.provenance --write' after review): "
                        + "; ".join(problems))
    mismatches = runtime_mismatches()
    if mismatches:
        raise LockError("Installed packages differ from requirements-lock.txt: " + "; ".join(mismatches))
    lock = {
        "schema": LOCK_SCHEMA,
        "status": "LOCKED_PENDING_REGISTRATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "real_study_data_read_when_locking": False,
        "hash_algorithm": "SHA-256",
        "git": git_state(),
        "python": platform.python_version(),
        "pinned_packages": installed_versions(),
        "external_tree_sha256": {name: meta["tree_sha256"]
                                 for name, meta in read_json(provenance.PROVENANCE_PATH)["externals"].items()},
        "files": locked_files(),
        "unlock_condition": "run_pipeline.py register --url <immutable OSF registration> after committing prereg/",
    }
    atomic_write_json(LOCK_PATH, lock)
    atomic_write_text(SIDECAR_PATH, f"{sha256_normalised(LOCK_PATH)}  PLAN_LOCK.json\n")
    return lock


def verify_lock() -> list[str]:
    """Return problems (empty list = the working tree matches the lock exactly)."""
    if not LOCK_PATH.exists() or not SIDECAR_PATH.exists():
        return ["plan lock or sidecar missing (run: run_pipeline.py plan-lock)"]
    problems = []
    if SIDECAR_PATH.read_text(encoding="utf-8").split()[0].lower() != sha256_normalised(LOCK_PATH):
        problems.append("PLAN_LOCK.sha256 does not match PLAN_LOCK.json")
    lock = read_json(LOCK_PATH)
    if lock.get("schema") != LOCK_SCHEMA:
        problems.append(f"unexpected lock schema {lock.get('schema')!r}")
    current = locked_files()
    for name, digest in lock.get("files", {}).items():
        if name not in current:
            problems.append(f"locked file missing: {name}")
        elif current[name] != digest:
            problems.append(f"locked file changed: {name}")
    for name in current:
        if name not in lock.get("files", {}):
            problems.append(f"unlocked file now in lock scope: {name}")
    problems.extend(provenance.verify())
    recorded_trees = lock.get("external_tree_sha256", {})
    for name, meta in provenance.compute()["externals"].items():
        if recorded_trees.get(name) != meta["tree_sha256"]:
            problems.append(f"vendored {name} differs from the locked tree hash")
    problems.extend("package " + item for item in runtime_mismatches())
    if lock.get("pinned_packages") != installed_versions():
        problems.append("installed package versions differ from those recorded in the lock")
    return problems


def register(url: str) -> dict:
    validate_registration_url(url)
    problems = verify_lock()
    if problems:
        raise LockError("Cannot register: lock does not verify: " + "; ".join(problems))
    record = {"schema": "fs-ms-registration-record/v1", "registration_url": url.strip(),
              "plan_lock_sha256": sha256_normalised(LOCK_PATH),
              "recorded_utc": datetime.now(timezone.utc).isoformat(),
              "attestation": "recorded by the investigator; syntax and lock hash are checked, the immutable "
                             "status of the OSF record is not independently certified"}
    if REGISTRATION_PATH.exists():
        existing = read_json(REGISTRATION_PATH)
        if existing.get("registration_url") == record["registration_url"] and \
                existing.get("plan_lock_sha256") == record["plan_lock_sha256"]:
            return existing
        raise LockError("A different registration record already exists and is never overwritten")
    atomic_write_json(REGISTRATION_PATH, record)
    return record


def real_data_unlock_problems() -> list[str]:
    """Empty list = a kind:real config may read restricted data."""
    problems = list(verify_lock())
    if not REGISTRATION_PATH.exists():
        problems.append("no registration record: real study data stays locked until the plan is registered "
                        "(run_pipeline.py register --url ...)")
        return problems
    record = read_json(REGISTRATION_PATH)
    try:
        validate_registration_url(record.get("registration_url", ""))
    except LockError as error:
        problems.append(f"registration record invalid: {error}")
    if LOCK_PATH.exists() and record.get("plan_lock_sha256") != sha256_normalised(LOCK_PATH):
        problems.append("registration record refers to a different plan lock")
    return problems
