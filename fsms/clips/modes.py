"""Run-mode guard shared by every stage-B command line tool.

``fixture`` mode (the default) calls :func:`fsms.guard.assert_unrestricted` on every user-supplied
input and output path, so a mistyped path cannot reach restricted real-study storage.

``real`` mode is allowed only when the preregistration lock gate passes: it calls
``fsms.lock.real_data_unlock_problems()`` (lock verified AND an immutable registration record
present) and refuses on any problem.  There is no flag or environment-variable bypass; the only
other way in is for a Python caller to hand in its own ``real_data_verifier`` callable (the
orchestrator may do so after it has verified the lock itself).
"""
from __future__ import annotations

import os
from typing import Callable, Mapping

from fsms.guard import RestrictedDataError, assert_unrestricted

MODES = ("fixture", "real")


def lock_verifier() -> None:
    """Default verifier for ``mode='real'``: raise unless the study is registered and the lock holds."""
    try:
        from fsms import lock
    except Exception as exc:  # module absent or broken: stay locked
        raise RestrictedDataError(f"mode=real needs a working fsms.lock ({type(exc).__name__}: {exc})") from exc
    problems = lock.real_data_unlock_problems()
    if problems:
        raise RestrictedDataError("mode=real refused: " + "; ".join(problems))


def enforce_mode(mode: str, paths: Mapping[str, "str | os.PathLike[str] | None"],
                 real_data_verifier: Callable[[], None] | None = None) -> None:
    """Guard ``paths`` (purpose -> path) for ``mode``; raises :class:`RestrictedDataError`."""
    if mode == "fixture":
        for purpose, path in paths.items():
            if path:
                assert_unrestricted(path, purpose)
        return
    if mode == "real":
        (real_data_verifier or lock_verifier)()
        return
    raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
