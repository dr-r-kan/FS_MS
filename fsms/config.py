"""Study configuration: one JSON file defines a whole run (real study or fixture).

Layout of a run (everything the pipeline *writes* lives under ``run_root``; ``data_root`` is only read):

    <data_root>/clips/*.edf, clips/snip_keyfile.csv, analysis_database.csv      (inputs)
    <run_root>/state/<stage>.done.json                                          (idempotency ledger)
    <run_root>/heppy_output/ ...                                                (HEPPy stage)
    <run_root>/vg_microstates/ manifests, hierarchy_<fingerprint>/ ...          (VG-microstates stage)
    <run_root>/autonomic/autonomic_native_ii_standardised.csv.gz                (CSI/CPI native table)
    <run_root>/analysis/prepared/ ...                                           (descriptive measurements + data freeze)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .atomic import fingerprint
from .guard import RestrictedDataError, assert_unrestricted, is_inside, is_restricted

REPO = Path(__file__).resolve().parents[1]
SCHEMA = "fsms-config/v1"
KINDS = ("real", "fixture")
_MISSING = object()


class ConfigError(ValueError):
    pass


class Config:
    def __init__(self, raw: dict[str, Any], path: Path):
        self.raw = raw
        self.path = path
        self.repo = REPO
        self.kind: str = raw.get("kind", "")
        self.name: str = raw.get("name", "")
        self._validate()

    # ---- loading -------------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "Config":
        target = Path(path)
        if not target.is_absolute():
            target = (REPO / target) if (REPO / target).exists() else target.resolve()
        try:
            raw = json.loads(target.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as error:
            raise ConfigError(f"Cannot read config {target}: {error}") from error
        return cls(raw, target)

    def _validate(self) -> None:
        if self.raw.get("schema") != SCHEMA:
            raise ConfigError(f"Unsupported config schema {self.raw.get('schema')!r}; expected {SCHEMA!r}")
        if self.kind not in KINDS:
            raise ConfigError(f"config kind must be one of {KINDS}, got {self.kind!r}")
        for key in ("paths.data_root", "paths.run_root"):
            self.get(key)
        if self.kind == "fixture":
            for label in ("data_root", "run_root"):
                try:
                    assert_unrestricted(self.get(f"paths.{label}"), f"fixture {label}")
                except RestrictedDataError as error:
                    raise ConfigError(str(error)) from error
            adopt = self.get("paths.adopt_heppy_dir", None)
            if adopt:
                raise ConfigError("fixture configs may not adopt existing HEPPy output")
        else:
            if is_inside(self.run_root, self.data_root) or is_restricted(self.run_root):
                raise ConfigError("run_root must be outside the restricted real-study data tree: the "
                                  "pipeline never writes into real study storage")

    # ---- access --------------------------------------------------------------------------
    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _MISSING:
                    raise ConfigError(f"Missing required config key {dotted!r} in {self.path}")
                return default
            node = node[part]
        return node

    def _path(self, value: str, base: Path) -> Path:
        candidate = Path(os.path.expandvars(str(value)))
        return candidate if candidate.is_absolute() else (base / candidate)

    @property
    def data_root(self) -> Path:
        return self._path(self.get("paths.data_root"), self.repo)

    @property
    def run_root(self) -> Path:
        return self._path(self.get("paths.run_root"), self.repo)

    @property
    def adopt_heppy_dir(self) -> Path | None:
        value = self.get("paths.adopt_heppy_dir", None)
        return self._path(value, self.repo) if value else None

    def repo_file(self, relative: str) -> Path:
        return self._path(relative, self.repo)

    # ---- run layout ----------------------------------------------------------------------
    @property
    def state_dir(self) -> Path:
        return self.run_root / "state"

    @property
    def log_dir(self) -> Path:
        return self.run_root / "logs"

    @property
    def clips_dir(self) -> Path:
        return self._path(self.get("clips.dir", "clips"), self.data_root)

    @property
    def database_csv(self) -> Path:
        return self._path(self.get("clips.database", "analysis_database.csv"), self.data_root)

    @property
    def snip_keyfile(self) -> Path:
        return self._path(self.get("clips.snip_keyfile", "clips/snip_keyfile.csv"), self.data_root)

    @property
    def heppy_out(self) -> Path:
        return self.run_root / "heppy_output"

    @property
    def vg_dir(self) -> Path:
        return self.run_root / "vg_microstates"

    @property
    def manifest_mat(self) -> Path:
        return self.vg_dir / "vg_microstate_manifest_mat.csv"

    @property
    def autonomic_dir(self) -> Path:
        return self.run_root / "autonomic"

    @property
    def autonomic_native(self) -> Path:
        return self.autonomic_dir / "autonomic_native_ii_standardised.csv.gz"

    @property
    def analysis_dir(self) -> Path:
        return self.run_root / "analysis" / "prepared"

    @property
    def prereg_dir(self) -> Path:
        return self.repo / "prereg"

    def section_fingerprint(self, *sections: str) -> str:
        return fingerprint({name: self.raw.get(name) for name in sections})

    def assert_paths_allowed(self, *paths: Path, purpose: str = "access") -> None:
        """Fixture runs may never touch restricted storage; real runs must be unlocked (see fsms.lock)."""
        if self.kind == "fixture":
            for path in paths:
                assert_unrestricted(path, purpose)
