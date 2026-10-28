"""Atomic writes, hashing and stage fingerprints shared by every pipeline stage."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: str | os.PathLike[str], chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


TEXT_SUFFIXES = {".py", ".m", ".md", ".json", ".txt", ".csv", ".ps1", ".yml", ".yaml", ".cff", ".cmd",
                 ".cfg", ".toml", ".ini", ".sha256"}


def sha256_normalised(path: str | os.PathLike[str]) -> str:
    """SHA-256 that ignores CRLF vs LF for text files, so locks verify on any checkout/autocrlf setting."""
    if Path(path).suffix.lower() not in TEXT_SUFFIXES:
        return sha256_file(path)
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def json_content_digest(path: str | os.PathLike[str]) -> str:
    """SHA-256 of a JSON file's content ignoring top-level ``*_utc`` timestamps (and key order/whitespace).

    Lets a stage that is rebuilt later with identical content (new ``created_utc``) hash the same.
    """
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(value, dict):
        value = {k: v for k, v in value.items() if not k.endswith("_utc")}
    return fingerprint(value)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fingerprint(value: Any) -> str:
    """Stable SHA-256 of any JSON-serialisable value (key order and whitespace independent)."""
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    """Write ``data`` so that readers see either the old complete file or the new complete file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".part", dir=target.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def atomic_write_text(path: str | os.PathLike[str], text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def read_json(path: str | os.PathLike[str]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def remove_stale_parts(directory: str | os.PathLike[str]) -> int:
    """Delete leftover ``*.part`` temporaries from interrupted atomic writes; return the count."""
    removed = 0
    for stale in Path(directory).rglob("*.part"):
        try:
            stale.unlink()
            removed += 1
        except OSError:
            pass
    return removed
