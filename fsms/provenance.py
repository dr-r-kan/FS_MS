"""Provenance and tamper detection for vendored external code (HEPPy, VG-microstates).

The vendored trees are modified derivatives of the investigator's GitHub repositories and are not
reproducible from any single upstream commit, so the content hash of the vendored tree is the
source of truth.  ``external/PROVENANCE.json`` records that hash plus what is known about the
upstream relationship; the plan lock includes it, and ``verify`` recomputes it.

    python -m fsms.provenance --write     # regenerate external/PROVENANCE.json (investigator action)
    python -m fsms.provenance --verify    # recompute hashes and compare
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from .atomic import atomic_write_json, read_json, sha256_normalised
from .config import REPO

PROVENANCE_PATH = REPO / "external" / "PROVENANCE.json"
EXCLUDED_DIRS = {"__pycache__", ".git", ".idea", ".ipynb_checkpoints", "outputs", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".part"}

EXTERNALS = {
    "HEPPy": {
        "path": "external/HEPPy",
        "upstream": "https://github.com/dr-r-kan/HEPPy",
        "upstream_reference": "main@758ef7c (2026-01-16); last upstream code change 2025-11-27",
        "relationship": "Diverged fork: _preprocessing.py and hep.py are substantially extended (atomic saves, "
                        "_both CFA mode, PyPREP fallbacks, ICLabel ONNX); ecg_peak_correction.py and heppy_stats.py "
                        "are local additions. Upstream has no functional change newer than this vendored copy.",
        "earlier_submodule_pin": "ClinicalAffectiveNeuroscienceLab/HEPPy@96d379013ebb38eafd4871b6b335a9dd8333964a",
    },
    "VG-microstates": {
        "path": "external/VG-microstates",
        "upstream": "https://github.com/dr-r-kan/VG-microstates (private)",
        "upstream_reference": "closest branch: spm-ms-import-without-cardiac@18a67a8 (2026-07-04); master@32aad2c",
        "relationship": "Modified derivative. Adds spm_mix_prior.m, fixed-reference inference, backfit changes and "
                        "recovery commands that exist on no upstream branch. The earlier submodule pin "
                        "(ClinicalAffectiveNeuroscienceLab/VG-microstates@1bba8fa776ee229635ff198b0421ff8efbea0a48) "
                        "is not reachable, so the exact base cannot be recovered; the vendored tree hash below governs.",
        "earlier_submodule_pin": "ClinicalAffectiveNeuroscienceLab/VG-microstates@1bba8fa776ee229635ff198b0421ff8efbea0a48",
    },
}


def tree_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in relative.parts) or path.suffix in EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    return files


def tree_hash(root: Path) -> tuple[str, int]:
    """SHA-256 over sorted ``relative_posix_path NUL file_sha256`` lines; returns (hash, file_count)."""
    digest = hashlib.sha256()
    files = tree_files(root)
    for path in files:
        digest.update(f"{path.relative_to(root).as_posix()}\0{sha256_normalised(path)}\n".encode("utf-8"))
    return digest.hexdigest(), len(files)


def compute() -> dict:
    record = {"schema": "fs-ms-external-provenance/v1", "externals": {}}
    for name, meta in EXTERNALS.items():
        digest, count = tree_hash(REPO / meta["path"])
        record["externals"][name] = {**meta, "tree_sha256": digest, "file_count": count}
    template = REPO / "external" / "VG-microstates" / "MetaMaps_2023_06.set"
    record["metamaps_template_sha256"] = sha256_normalised(template)
    return record


def verify() -> list[str]:
    """Return a list of problems (empty when the vendored code matches its recorded hash)."""
    if not PROVENANCE_PATH.exists():
        return [f"missing {PROVENANCE_PATH}"]
    recorded = read_json(PROVENANCE_PATH)
    current = compute()
    problems = []
    for name in EXTERNALS:
        if recorded["externals"].get(name, {}).get("tree_sha256") != current["externals"][name]["tree_sha256"]:
            problems.append(f"vendored {name} differs from external/PROVENANCE.json")
    if recorded.get("metamaps_template_sha256") != current["metamaps_template_sha256"]:
        problems.append("MetaMaps template differs from external/PROVENANCE.json")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.write:
        atomic_write_json(PROVENANCE_PATH, compute())
        print(f"wrote {PROVENANCE_PATH}")
        return 0
    problems = verify()
    for problem in problems:
        print("PROVENANCE PROBLEM:", problem, file=sys.stderr)
    print("external provenance verified" if not problems else "external provenance FAILED")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
