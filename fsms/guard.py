"""Restricted-data guard.

The real FS/ES study data must not be read or written until the analysis plan is preregistered.
Every stage driver that is handed a path calls ``assert_unrestricted`` when running in fixture or
test mode, so a mistyped path cannot silently reach patient data.  There is deliberately no
environment-variable or flag bypass here: real-data mode is selected only by a study config whose
lock/registration record is verified by ``fsms.lock`` (see ``docs/RUNBOOK.md``).
"""
from __future__ import annotations

import os
from pathlib import Path

RESTRICTED_ROOTS: tuple[str, ...] = (
    r"F:\FS_microstates_study",
    r"G:\FS_microstates_study",
    r"D:\EEGData",
    r"D:\Chalfont",
    r"G:\Chalfont",
    r"E:\EEGs",
    r"F:\EEGs",
)


class RestrictedDataError(PermissionError):
    """Raised when a path resolves inside restricted real-study storage."""


def _lexical(path: str | os.PathLike[str]) -> str:
    """Case/separator-normalised absolute path computed WITHOUT touching the filesystem."""
    return os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(path))))


def _under(candidate: str, root: str) -> bool:
    return candidate == root or candidate.startswith(root.rstrip(os.sep) + os.sep)


def is_restricted(path: str | os.PathLike[str], roots: tuple[str, ...] = RESTRICTED_ROOTS) -> bool:
    """True if ``path`` lies in restricted storage.

    The lexical test runs first and never touches the disk, so asking about a restricted path does not
    stat it.  Only a lexically clean path is symlink-resolved (to catch a link into restricted storage).
    """
    lexical_roots = [_lexical(root) for root in roots]
    candidate = _lexical(path)
    if any(_under(candidate, root) for root in lexical_roots):
        return True
    if os.path.lexists(candidate):
        resolved = os.path.normcase(os.path.realpath(candidate))
        return any(_under(resolved, root) for root in lexical_roots)
    return False


def is_inside(path: str | os.PathLike[str], root: str | os.PathLike[str]) -> bool:
    """Lexical containment test (no filesystem access)."""
    return _under(_lexical(path), _lexical(root))


def assert_unrestricted(path: str | os.PathLike[str], purpose: str = "access",
                        roots: tuple[str, ...] = RESTRICTED_ROOTS) -> Path:
    """Return ``Path(path)`` or raise if it lies inside restricted study storage."""
    if is_restricted(path, roots):
        raise RestrictedDataError(
            f"Refusing {purpose} of {path!s}: it is inside restricted real-study storage and the "
            "analysis plan is not yet preregistered/unlocked for real data.")
    return Path(path)
