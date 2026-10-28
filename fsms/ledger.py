"""Per-stage completion ledger: the single mechanism that makes the pipeline idempotent.

A stage is *current* when a sentinel exists whose fingerprint equals the stage's present
fingerprint (parameters + code hashes + upstream fingerprints) and every recorded output still
exists with the recorded size (and sha256 for files below ``HASH_LIMIT``).  Anything else reruns
the stage from scratch; a stage never resumes from half-written state.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .atomic import atomic_write_json, read_json, sha256_file

HASH_LIMIT = 64 * 1024 * 1024  # hash outputs below this size; larger ones are verified by size


def _describe(path: Path) -> dict:
    size = path.stat().st_size
    return {"path": str(path), "bytes": size, "sha256": sha256_file(path) if size <= HASH_LIMIT else None}


class Ledger:
    def __init__(self, state_dir: str | Path):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, stage: str) -> Path:
        return self.dir / f"{stage}.done.json"

    def read(self, stage: str) -> dict | None:
        try:
            return read_json(self.path(stage))
        except (OSError, ValueError):
            return None

    def verify(self, stage: str, fingerprint: str) -> tuple[bool, str]:
        """Return (is_current, reason)."""
        record = self.read(stage)
        if record is None:
            return False, "no completion record"
        if record.get("fingerprint") != fingerprint:
            return False, "fingerprint changed (parameters, code, or upstream output differ)"
        for item in record.get("outputs", []):
            target = Path(item["path"])
            if not target.is_file():
                return False, f"output missing: {target}"
            if target.stat().st_size != item["bytes"]:
                return False, f"output size changed: {target}"
            if item.get("sha256") and sha256_file(target) != item["sha256"]:
                return False, f"output content changed: {target}"
        return True, "current"

    def mark_done(self, stage: str, fingerprint: str, outputs: Iterable[str | Path], extra: dict | None = None,
                  seconds: float | None = None) -> dict:
        described = []
        for output in outputs:
            target = Path(output)
            if not target.is_file():
                raise FileNotFoundError(f"Stage {stage!r} declared output that does not exist: {target}")
            described.append(_describe(target))
        record = {"stage": stage, "fingerprint": fingerprint, "outputs": described,
                  "completed_utc": datetime.now(timezone.utc).isoformat(),
                  "seconds": None if seconds is None else round(seconds, 2), **(extra or {})}
        atomic_write_json(self.path(stage), record)
        return record

    def invalidate(self, stage: str) -> None:
        self.path(stage).unlink(missing_ok=True)


class Timer:
    def __enter__(self):
        self.start = time.monotonic()
        return self

    def __exit__(self, *exc):
        self.seconds = time.monotonic() - self.start
        return False
