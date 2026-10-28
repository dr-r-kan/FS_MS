"""Fast synthetic tests of the HEPPy stage driver (no TUEG, no real HEPPy run, no MATLAB).

The real step implementations (ECG fit, HEPPy preprocessing, R-peak recovery) are replaced by a
stub with the same interface that writes tiny but genuine FIF files and R-peak sidecars, so the
sentinel / fingerprint / exclusion / adoption / guard logic is exercised end to end by the real
orchestrator code.  The real-data safety constraint is respected throughout: the guard test uses a
non-existent path under a restricted root and nothing is ever created there.
"""
from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from fsms.guard import RestrictedDataError
from fsms.heppy import options as opt
from fsms.heppy import preprocess as pre
from fsms.heppy import qc as qcmod
from fsms.heppy import sentinel as sn
from fsms.heppy import run_heppy_stage as stage

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = ROOT / "config" / "heppy_options_fixture.json"
RESTRICTED_PROBE = r"F:\FS_microstates_study\__never_created__\heppy_out"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def quietly(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def tree_hash(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def write_tiny_fif(path: Path, seed: int, n_eeg: int = 19, with_ecg: bool = True) -> None:
    import mne

    rng = np.random.default_rng(seed)
    names = [f"Ch{i}" for i in range(n_eeg)] + (["EKG1"] if with_ecg else [])
    types = ["eeg"] * n_eeg + (["ecg"] if with_ecg else [])
    info = mne.create_info(names, 100.0, types)
    raw = mne.io.RawArray(rng.standard_normal((len(names), 6000)) * 1e-5, info, verbose="ERROR")
    raw.set_meas_date(1)   # as HEPPy does: makes the saved bytes deterministic (no wall-clock file id)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(str(path), overwrite=True, verbose="ERROR")


class StubImpl:
    """Drop-in for ``worker.RealImpl``; records calls and can fail or degrade chosen recordings."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail: dict[str, str] = {}          # base -> step that raises
        self.n_eeg: dict[str, int] = {}         # base -> EEG channels after montage
        self.chunk_status: dict[str, dict[int, str]] = {}      # base -> {chunk index: PyPREP status}
        self.chunk_interpolated: dict[str, dict[int, int]] = {}  # base -> {chunk index: n interpolated of 20}
        self.no_ecg: set[str] = set()

    def run(self, step: str, job: dict, sentinel: dict) -> dict:
        from fsms.heppy import sentinel as sentinel_mod
        from fsms.heppy import worker as wk

        base = job["base"]
        self.calls.append((base, step))
        if self.fail.get(base) == step:
            raise RuntimeError(f"stub failure in {step}")
        heppy_out = Path(job["heppy_out"])
        p = wk.paths(job)
        if step == "ecg_fit":
            if base in self.no_ecg:
                return {"status": "no_ecg_channel", "qc": {}, "outputs": []}
            p["autofit"].parent.mkdir(parents=True, exist_ok=True)
            p["autofit"].write_text("rpeak_time_s\n1.0\n2.0\n3.0\n4.0\n", encoding="utf-8")
            score = {"peak_count": 4, "valid_60s_windows": 3, "total_60s_windows": 5}
            return {"status": "ok", "qc": {"channel": "EEG EKG1-REF", "sfreq": 100.0, "n_samples": 6000,
                                           "rpeak_score": score},
                    "outputs": [sentinel_mod.output_item("autofit_sidecar", p["autofit"], heppy_out, "sidecar")]}
        if step == "preprocess":
            seed = int(hashlib.sha256(base.encode()).hexdigest()[:6], 16)
            for key in ("raw_fif", "raw_fif_keep"):
                write_tiny_fif(p[key], seed, with_ecg=base not in self.no_ecg)
            chunk_status = self.chunk_status.get(base, {})
            leaves = []
            for index in range(len(SEGMENTS)):
                status = chunk_status.get(index, "ok")
                good = status in ("ok", "ransac_fallback")
                n_interp = self.chunk_interpolated.get(base, {}).get(index, 1)
                leaf = {"children": [], "fits": [{"ransac": False, "ok": good,
                                                  "error": None if good else "RobustReference:TooManyBad x"}],
                        "messages": [], "exception": None, "n_eeg": 20,
                        "bad_by_type": {"bad_by_deviation": [f"c{i}" for i in range(n_interp)]},
                        "interpolated": [f"c{i}" for i in range(n_interp)] if good else [], "still_noisy": []}
                if status == "ransac_fallback":
                    leaf["fits"].insert(0, {"ransac": True, "ok": False, "error": "Too many noisy channels"})
                leaves.append(leaf)
            summary = pre.annotate_chunk_bounds(
                pre.summarise_pyprep([{"children": leaves, "fits": [], "messages": [], "exception": None}], True),
                pre.segment_bounds_s(Path(job["edf_path"])))
            return {"status": "ok", "qc": {
                "sfreq": 100.0, "n_eeg_input": 25, "n_eeg_after_montage": self.n_eeg.get(base, 19),
                "n_eeg_output": 19, "pyprep": summary,
                "ica": {"enabled": True, "n_components": 18, "decim": 1}, "warnings": []},
                "outputs": [sentinel_mod.output_item("raw_fif", p["raw_fif"], heppy_out, "fif"),
                            sentinel_mod.output_item("raw_fif_keep", p["raw_fif_keep"], heppy_out, "fif")]}
        if step == "recovery":
            steps = sentinel.get("steps", {})
            if (steps.get("ecg_fit") or {}).get("status") != "ok":
                return {"status": "skipped_no_ecg", "qc": {}, "outputs": []}
            p["corrected"].write_text("rpeak_time_s,rpeak_source,is_gap_recovered\n1.0,reviewed,0\n", encoding="utf-8")
            score = {"peak_count": 5, "valid_60s_windows": 3, "total_60s_windows": 5}
            return {"status": "ok", "qc": {"channel": "EKG1", "final_score": score},
                    "outputs": [sentinel_mod.output_item("corrected_sidecar", p["corrected"], heppy_out, "sidecar")]}
        raise AssertionError(step)


SEGMENTS = [("II", 0, 300, "2"), ("preictal", 300, 1500, "3"), ("ictal", 1500, 1550, "3"), ("postictal", 1550, 2750, "3")]
KEY_FIELDS = ["patient_code", "output_edf", "segment", "output_start_seconds", "output_end_seconds", "source_row",
              "event_type", "event_number", "segment_number"]


def make_clips(root: Path, bases: list[str]) -> Path:
    clips = root / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    rows = []
    for base in bases:
        (clips / f"{base}.edf").write_bytes(b"EDF-PLACEHOLDER-" + base.encode() * 50)
        for name, a, b, source_row in SEGMENTS:
            rows.append({"patient_code": base.split("_")[0], "output_edf": f"{base}.edf", "segment": name,
                         "output_start_seconds": str(float(a)), "output_end_seconds": str(float(b)),
                         "source_row": source_row, "event_type": "II" if name == "II" else "Ict",
                         "event_number": "0" if name == "II" else "1", "segment_number": ""})
    with (clips / "snip_keyfile.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=KEY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return clips


class ModeTests(unittest.TestCase):
    def test_real_mode_is_gated_by_the_lock_and_fixture_mode_is_default(self) -> None:
        from fsms.clips import modes

        with mock.patch("fsms.lock.real_data_unlock_problems", return_value=["no registration record"]):
            with self.assertRaises(RestrictedDataError):
                modes.enforce_mode("real", {"x": "C:/tmp"})
        with mock.patch("fsms.lock.real_data_unlock_problems", return_value=[]):
            modes.enforce_mode("real", {"x": RESTRICTED_PROBE})            # unlocked: path guard not applied
        with self.assertRaises(RestrictedDataError):
            modes.enforce_mode("fixture", {"x": RESTRICTED_PROBE})
        with self.assertRaises(ValueError):
            modes.enforce_mode("nonsense", {})
        self.assertEqual(stage.parse_args(["--clips-dir", "a", "--heppy-out", "b", "--options", "c"]).mode, "fixture")


class OptionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.options = opt.load_options(OPTIONS)

    def test_locked_options_valid_and_real_matches_legacy_normalisation(self) -> None:
        self.assertEqual(opt.validate_options(self.options), [])
        real = opt.load_options(ROOT / "config" / "heppy_options.json")
        self.assertEqual(opt.validate_options(real), [])
        legacy = {"filters": {"hp": 0.5, "lp": 45.0, "line_freqs": [50.0, 100.0]}, "ecg": {"channel": "", "flip_on_load": False},
                  "montage": {"name": "standard_1020", "rename_to_1020": True}, "pyprep": True, "asr": False,
                  "ica_iclabel": True, "cfa_mode": "both", "remove_cfa_mode": "both",
                  "preprocessing_tuning": {"pyprep_channel_wise": False, "pyprep_max_chunk_size": 8,
                                           "pyprep_by_segment": True, "pyprep_safe_quantile_patch": True,
                                           "ica_fit_hz": 100.0, "iclabel_backend": "auto"},
                  "adaptive_rr": {"rr_min_bpm": 35.0, "rr_max_bpm": 140.0, "rr_outlier_mad": 3.5, "bpm_smooth_win": 5},
                  "epoch": {"tmin": -0.2}, "hrv": {"compute": True}}
        self.assertEqual(opt.diff_processing_options(real, legacy), [])
        self.assertEqual(opt.diff_processing_options(self.options, legacy), ["filters.line_freqs"])

    def test_key_order_and_irrelevant_keys_do_not_matter(self) -> None:
        shuffled = json.loads(json.dumps(self.options, sort_keys=True))
        shuffled["epoch"]["amp_reject_uv"] = 999.0           # downstream-only key
        shuffled["qc_rule"]["segment"]["max_interpolated_fraction"] = 0.5
        self.assertEqual(opt.options_fingerprint(shuffled), opt.options_fingerprint(self.options))
        changed = copy.deepcopy(self.options)
        changed["filters"]["hp"] = 1.0
        self.assertNotEqual(opt.options_fingerprint(changed), opt.options_fingerprint(self.options))
        self.assertEqual(opt.diff_processing_options(changed, self.options), ["filters.hp"])

    def test_bom_tolerant_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "o.json"
            path.write_bytes(b"\xef\xbb\xbf" + json.dumps(self.options).encode())
            self.assertEqual(opt.processing_options(opt.load_options(path)), opt.processing_options(self.options))

    def test_flip_or_missing_rule_rejected(self) -> None:
        bad = copy.deepcopy(self.options)
        bad["ecg"]["flip_on_load"] = True
        del bad["qc_rule"]
        self.assertGreaterEqual(len(opt.validate_options(bad)), 2)

    def test_step_fingerprints_invalidate_only_dependants(self) -> None:
        versions = opt.package_versions()
        edf = {"size": 10, "sha256": "a" * 64, "mtime_ns": 1}
        base = sn.step_fingerprints(edf, "kf", self.options, versions)

        def fps(**changes):
            options = copy.deepcopy(self.options)
            for dotted, value in changes.items():
                node = options
                *path, leaf = dotted.split("__")
                for key in path:
                    node = node[key]
                node[leaf] = value
            return sn.step_fingerprints(edf, "kf", options, versions)

        same = lambda x, y, *keys: all(x[k] == y[k] for k in keys)
        recov = fps(rpeak_recovery__iterations=3)
        self.assertTrue(same(base, recov, "ecg_fit", "preprocess"))
        self.assertNotEqual(base["recovery"], recov["recovery"])
        auto = fps(auto_qrs__alpha=0.1)
        self.assertTrue(same(base, auto, "preprocess"))
        self.assertNotEqual(base["ecg_fit"], auto["ecg_fit"])
        self.assertNotEqual(base["recovery"], auto["recovery"])
        tuning = fps(preprocessing_tuning__ica_fit_hz=50.0)
        self.assertTrue(same(base, tuning, "ecg_fit"))
        self.assertNotEqual(base["preprocess"], tuning["preprocess"])
        self.assertNotEqual(base["recovery"], tuning["recovery"])      # recovery reads the new FIF
        seed = fps(seeds__heppy_random_state=7)
        self.assertNotEqual(base["preprocess"], seed["preprocess"])
        self.assertTrue(same(base, seed, "ecg_fit"))
        rule = fps(qc_rule__segment__max_interpolated_fraction=0.9)
        self.assertEqual(base, rule)                                      # rule never invalidates outputs
        other_edf = sn.step_fingerprints({**edf, "sha256": "b" * 64}, "kf", self.options, versions)
        self.assertTrue(all(base[k] != other_edf[k] for k in base))
        other_keys = sn.step_fingerprints(edf, "kf2", self.options, versions)
        self.assertNotEqual(base["preprocess"], other_keys["preprocess"])
        self.assertEqual(base["ecg_fit"], other_keys["ecg_fit"])
        newer = sn.step_fingerprints(edf, "kf", self.options, {**versions, "onnxruntime": "99.0"})
        self.assertNotEqual(base["preprocess"], newer["preprocess"])       # ICLabel backend version
        self.assertEqual(base["ecg_fit"], newer["ecg_fit"])
        newer_mne = sn.step_fingerprints(edf, "kf", self.options, {**versions, "mne": "99.0"})
        self.assertTrue(all(base[k] != newer_mne[k] for k in ("ecg_fit", "preprocess", "recovery")))


class PyprepSummaryTests(unittest.TestCase):
    def leaf(self, fits, messages=(), n_eeg=19, interpolated=(), still=()):
        return {"children": [], "fits": fits, "messages": list(messages), "exception": None, "n_eeg": n_eeg,
                "bad_by_type": {"bad_by_deviation": list(interpolated)}, "interpolated": list(interpolated),
                "still_noisy": list(still)}

    def test_statuses_and_precedence(self) -> None:
        ok = self.leaf([{"ransac": False, "ok": True, "error": None}], interpolated=["a", "b"])
        ransac = self.leaf([{"ransac": True, "ok": False, "error": "Too many noisy channels"},
                            {"ransac": False, "ok": True, "error": None}])
        toomany = self.leaf([{"ransac": False, "ok": False, "error": "RobustReference:TooManyBad"}],
                            messages=["PyPREP skipped; too many bad EEG channels for robust reference: x"])
        nomont = self.leaf([], messages=["PyPREP skipped; no EEG channels have usable montage positions."])
        failed = self.leaf([{"ransac": False, "ok": False, "error": "RuntimeError: boom"}])
        self.assertEqual(pre.leaf_status(ok), "ok")
        self.assertEqual(pre.leaf_status(ransac), "ransac_fallback")
        self.assertEqual(pre.leaf_status(toomany), "skipped_toomanybad")
        self.assertEqual(pre.leaf_status(nomont), "skipped_no_montage")
        self.assertEqual(pre.leaf_status(failed), "failed")
        parent = {"children": [ok, ransac, toomany], "fits": [], "messages": [], "exception": None}
        summary = pre.summarise_pyprep([parent], True)
        self.assertEqual(summary["status"], "skipped_toomanybad")
        self.assertEqual(summary["n_chunks"], 3)
        self.assertEqual(summary["status_counts"], {"ok": 1, "ransac_fallback": 1, "skipped_toomanybad": 1})
        self.assertAlmostEqual(summary["interpolated_fraction_max"], 2 / 19)
        self.assertEqual(pre.summarise_pyprep([parent, failed], True)["status"], "failed")
        self.assertEqual(pre.summarise_pyprep([ok], False)["status"], "disabled")
        self.assertEqual(pre.summarise_pyprep([], True)["status"], "not_run")


class RuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = opt.load_options(OPTIONS)["qc_rule"]

    def good(self) -> dict:
        return {"errors": [], "outputs_ok": True, "n_eeg_after_montage": 19,
                "pyprep": {"status": "ok", "interpolated_fraction_max": 0.1},
                "ecg": {"found": True, "valid_60s_windows": 4, "total_60s_windows": 40, "final_peak_count": 99},
                "ica": {}}

    def codes(self, qc) -> list[str]:
        return [r["reason_code"] for r in qcmod.evaluate_rule(qc, self.rule)["reasons"]]

    def test_rule_is_version_two_and_structurally_valid(self) -> None:
        self.assertEqual(self.rule["rule_id"], "heppy_recording_exclusion_v2")
        self.assertEqual(qcmod.rule_problems(self.rule), [])
        self.assertEqual(self.rule["segment"]["scope_segments"], ["II", "preictal"])
        self.assertTrue(qcmod.rule_problems({"rule_id": "x"}))

    def test_good_recording_included(self) -> None:
        result = qcmod.evaluate_rule(self.good(), self.rule)
        self.assertFalse(result["excluded"])
        self.assertEqual(result["not_evaluable"], [])

    def test_each_recording_level_criterion(self) -> None:
        qc = self.good(); qc["n_eeg_after_montage"] = 14
        self.assertEqual(self.codes(qc), ["too_few_eeg_channels"])
        qc = self.good(); qc["n_eeg_after_montage"] = 15            # threshold is inclusive of 15
        self.assertEqual(self.codes(qc), [])
        qc = self.good(); qc["ecg"] = {"found": False}
        self.assertEqual(self.codes(qc), ["no_ecg_channel"])
        qc = self.good(); qc["ecg"]["valid_60s_windows"] = 0
        self.assertEqual(self.codes(qc), ["ecg_no_valid_window"])
        qc = self.good(); qc["errors"] = [{"step": "preprocess", "kind": "exception", "type": "ValueError", "message": "x"}]
        self.assertIn("unhandled_exception", self.codes(qc))
        qc = self.good(); qc["errors"] = [{"step": "preprocess", "kind": "crash", "type": "worker_crash", "message": "exit=1"}]
        self.assertIn("worker_crash", self.codes(qc))
        qc = self.good(); qc["outputs_ok"] = False
        self.assertEqual(self.codes(qc), ["missing_outputs"])

    def test_pyprep_is_never_a_recording_level_criterion(self) -> None:
        for status in ("skipped_toomanybad", "skipped_no_montage", "failed"):
            qc = self.good(); qc["pyprep"] = {"status": status, "interpolated_fraction_max": 0.9}
            self.assertEqual(self.codes(qc), [], status)
        qc = self.good(); qc["pyprep"] = {"status": "ransac_fallback"}
        result = qcmod.evaluate_rule(qc, self.rule)
        self.assertFalse(result["excluded"]); self.assertEqual(result["flags"], ["pyprep_ransac_fallback"])

    def test_multiple_reasons_sorted_by_precedence_and_group_blind(self) -> None:
        qc = self.good(); qc["n_eeg_after_montage"] = 3; qc["ecg"] = {"found": False}
        result = qcmod.evaluate_rule(qc, self.rule)
        self.assertEqual([r["reason_code"] for r in result["reasons"]], ["too_few_eeg_channels", "no_ecg_channel"])
        self.assertEqual(result["primary_reason"], "too_few_eeg_channels")
        import inspect
        self.assertEqual(list(inspect.signature(qcmod.evaluate_rule).parameters), ["qc", "rule"])
        self.assertEqual(list(inspect.signature(qcmod.evaluate_segments).parameters), ["qc", "rule", "segments"])

    def test_viable_column_by_rule(self) -> None:
        self.assertEqual(qcmod.ecg_viable(self.good(), self.rule)[0], "yes")
        qc = self.good(); qc["ecg"] = {"found": False}
        self.assertEqual(qcmod.ecg_viable(qc, self.rule)[0], "no")
        qc = self.good(); qc["ecg"]["valid_60s_windows"] = 0
        self.assertEqual(qcmod.ecg_viable(qc, self.rule)[0], "no")

    # ---- segment scope
    SEGS = [{"segment": name, "output_start_seconds": str(float(a)), "output_end_seconds": str(float(b))}
            for name, a, b, _r in SEGMENTS]

    def chunks(self, **status) -> dict:
        bounds = [(a, b) for _n, a, b, _r in SEGMENTS]
        chunks = [{"index": i, "status": status.get(f"c{i}", "ok"), "start_s": float(a), "end_s": float(b),
                   "interpolated_fraction": 0.1 if status.get(f"c{i}", "ok") == "ok" else None}
                  for i, (a, b) in enumerate(bounds)]
        for key, value in status.items():
            if key.startswith("frac"):
                chunks[int(key[4:])]["interpolated_fraction"] = value
        return {"status": "ok", "chunks": chunks}

    def evaluate(self, **status) -> dict[str, list[str]]:
        qc = self.good(); qc["pyprep"] = self.chunks(**status)
        out = qcmod.evaluate_segments(qc, self.rule, self.SEGS)
        return {r["segment"]["segment"]: [x["reason_code"] for x in r["reasons"]] for r in out["results"]}

    def test_segment_rule_only_covers_in_scope_segments(self) -> None:
        self.assertEqual(self.evaluate(), {"II": [], "preictal": []})
        self.assertEqual(self.evaluate(c0="skipped_toomanybad"),
                         {"II": ["segment_pyprep_skipped_toomanybad"], "preictal": []})
        self.assertEqual(self.evaluate(c1="failed")["preictal"], ["segment_pyprep_failed"])
        self.assertEqual(self.evaluate(c0="skipped_no_montage")["II"], ["segment_pyprep_skipped_no_montage"])
        # failures confined to ictal/postictal chunks never exclude anything
        self.assertEqual(self.evaluate(c2="skipped_toomanybad", c3="failed"), {"II": [], "preictal": []})
        self.assertEqual(self.evaluate(frac1=0.31)["preictal"], ["segment_excess_interpolation"])
        self.assertEqual(self.evaluate(frac1=0.30)["preictal"], [])
        self.assertEqual(self.evaluate(frac2=0.9), {"II": [], "preictal": []})
        flagged = self.evaluate(c1="ransac_fallback")
        self.assertEqual(flagged["preictal"], [])

    def test_overlap_not_adjacency_decides_and_unbounded_chunks_cover_everything(self) -> None:
        qc = self.good()
        qc["pyprep"] = {"status": "ok", "chunks": [{"index": 0, "status": "skipped_toomanybad", "start_s": 300.0,
                                                    "end_s": 301.0, "interpolated_fraction": None}]}
        out = qcmod.evaluate_segments(qc, self.rule, self.SEGS)["results"]
        self.assertEqual({r["segment"]["segment"]: r["usable"] for r in out}, {"II": True, "preictal": False})
        qc["pyprep"]["chunks"][0].update(start_s=300.0, end_s=300.0)       # touches II only at its end: no overlap
        out = qcmod.evaluate_segments(qc, self.rule, self.SEGS)["results"]
        self.assertTrue(all(r["usable"] for r in out))
        qc["pyprep"]["chunks"] = [{"index": 0, "status": "failed", "start_s": None, "end_s": None,
                                   "interpolated_fraction": None}]
        out = qcmod.evaluate_segments(qc, self.rule, self.SEGS)["results"]
        self.assertFalse(any(r["usable"] for r in out))                    # unbounded: conservative

    def test_missing_segment_evidence_is_reported_not_excluded(self) -> None:
        qc = self.good(); qc["adopted"] = True
        qc["pyprep"] = {"status": qcmod.ADOPTED_UNKNOWN}
        out = qcmod.evaluate_segments(qc, self.rule, self.SEGS)
        self.assertTrue(all(r["usable"] for r in out["results"]))
        self.assertIn("segment_pyprep_status", out["not_evaluable"])
        qc["pyprep"] = {"status": "skipped_toomanybad"}                    # degraded but not attributable
        self.assertIn("segment_pyprep_attribution", qcmod.evaluate_segments(qc, self.rule, self.SEGS)["not_evaluable"])


class StageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="fsms_heppy_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.clips = make_clips(self.tmp, ["P1_snips", "P2_snips", "P3_snips"])
        self.out = self.tmp / "heppy_output"
        self.impl = StubImpl()
        self.options_path = self.tmp / "options.json"
        shutil.copy(OPTIONS, self.options_path)

    def run_stage(self, **kwargs):
        kwargs.setdefault("log", lambda m: None)
        return quietly(stage.run_stage, self.clips, self.out, self.options_path, in_process=True, impl=self.impl, **kwargs)

    def outputs_hash(self, sentinels: bool = True) -> dict[str, str]:
        """Hashes of every output except run logs (and, optionally, sentinels: they carry compute timestamps)."""
        return {k: v for k, v in tree_hash(self.out).items()
                if not k.startswith("logs") and (sentinels or not k.startswith(".sentinels"))}

    def test_first_run_then_idempotent_rerun(self) -> None:
        first = self.run_stage()
        self.assertEqual(first["exit_code"], 0)
        self.assertEqual(sorted(first["actions"].values()), ["compute"] * 3)
        self.assertEqual(len(self.impl.calls), 9)
        summary = first["summary"]
        self.assertEqual(summary["consort"]["included"], 3)
        self.assertTrue((self.out / "heppy_exclusions.csv").read_text(encoding="utf-8").startswith("base,stage,reason_code,evidence"))
        for base in ("P1_snips", "P2_snips", "P3_snips"):
            self.assertTrue((self.out / "heppy_qc" / f"{base}.json").exists())
            self.assertTrue((self.out / ".sentinels" / f"{base}.json").exists())
        before = self.outputs_hash()
        calls = len(self.impl.calls)
        second = self.run_stage()
        self.assertEqual(sorted(second["actions"].values()), ["unchanged"] * 3)
        self.assertEqual(len(self.impl.calls), calls, "nothing may be recomputed")
        self.assertEqual(self.outputs_hash(), before, "outputs must be byte-identical")

    def test_review_csv_legacy_schema_and_viable_by_rule(self) -> None:
        self.impl.no_ecg.add("P2_snips")
        self.run_stage()
        rows = {r["base"]: r for r in read_rows(self.out / "ecg_qc_review.csv")}
        self.assertEqual(list(next(iter(rows.values()))), stage.REVIEW_COLS)
        self.assertEqual(rows["P1_snips"]["viable"], "yes")
        self.assertEqual(rows["P2_snips"]["viable"], "no")
        self.assertIn("no ECG channel", rows["P2_snips"]["notes"])
        self.assertTrue(rows["P2_snips"]["raw_fif"], "EEG FIFs are produced regardless of ECG viability")
        reasons = read_rows(self.out / "heppy_exclusions.csv")
        self.assertEqual([(r["base"], r["reason_code"]) for r in reasons], [("P2_snips", "no_ecg_channel")])

    def test_option_tamper_recomputes_only_affected_steps(self) -> None:
        self.run_stage()
        self.impl.calls.clear()
        tampered = json.loads(self.options_path.read_text())
        tampered["rpeak_recovery"]["iterations"] = 3
        self.options_path.write_text(json.dumps(tampered), encoding="utf-8")
        result = self.run_stage()
        self.assertEqual(sorted(result["actions"].values()), ["compute"] * 3)
        self.assertEqual({step for _b, step in self.impl.calls}, {"recovery"})
        self.impl.calls.clear()
        tampered["preprocessing_tuning"]["ica_fit_hz"] = 50.0
        self.options_path.write_text(json.dumps(tampered), encoding="utf-8")
        self.run_stage()
        self.assertEqual({step for _b, step in self.impl.calls}, {"preprocess", "recovery"})
        self.impl.calls.clear()
        tampered["qc_rule"]["recording"]["min_eeg_channels_after_montage"] = 25   # rule change: re-evaluate only
        self.options_path.write_text(json.dumps(tampered), encoding="utf-8")
        result = self.run_stage()
        self.assertEqual(self.impl.calls, [])
        self.assertEqual(result["summary"]["consort"]["excluded"], 3)

    def test_deleted_fif_and_stale_part_files_heal(self) -> None:
        self.run_stage()
        before_outputs = self.outputs_hash(sentinels=False)
        (self.out / "raw_fif" / "P2_snips_pp_raw.fif").unlink()
        stale = self.out / "raw_fif" / "P2_snips_pp.part-4242_raw.fif"
        stale.write_bytes(b"partial")
        (self.out / ".sentinels" / "P1_snips.json.deadbeef.part").write_bytes(b"x")
        self.impl.calls.clear()
        result = self.run_stage()
        self.assertEqual(result["actions"]["P2_snips"], "compute")
        self.assertEqual({(b, s) for b, s in self.impl.calls}, {("P2_snips", "preprocess"), ("P2_snips", "recovery")})
        self.assertFalse(stale.exists())
        self.assertFalse(list(self.out.rglob("*.part")))
        self.assertEqual(self.outputs_hash(sentinels=False), before_outputs)

    def test_truncated_fif_is_detected(self) -> None:
        self.run_stage()
        fif = self.out / "raw_fif" / "P1_snips_pp_raw_keepcfa.fif"
        data = fif.read_bytes()
        fif.write_bytes(data[: len(data) // 2])
        self.impl.calls.clear()
        self.run_stage()
        self.assertIn(("P1_snips", "preprocess"), self.impl.calls)

    def test_failure_in_one_recording_does_not_stop_others_and_is_not_retried(self) -> None:
        self.impl.fail["P2_snips"] = "preprocess"
        result = self.run_stage()
        self.assertEqual(result["exit_code"], 0)
        reasons = {(r["base"], r["reason_code"]) for r in read_rows(self.out / "heppy_exclusions.csv")}
        self.assertIn(("P2_snips", "unhandled_exception"), reasons)
        self.assertTrue(result["summary"]["recordings"]["P1_snips"]["included"])
        self.assertTrue(result["summary"]["recordings"]["P3_snips"]["included"])
        self.impl.calls.clear()
        self.run_stage()
        self.assertEqual([c for c in self.impl.calls if c[0] == "P2_snips"], [], "deterministic failure not retried")
        self.impl.fail.clear()
        self.run_stage(retry_failed=True)
        self.assertIn(("P2_snips", "preprocess"), self.impl.calls)
        self.assertEqual(self.run_stage()["summary"]["consort"]["excluded"], 0)

    def test_segment_level_pyprep_rule_end_to_end(self) -> None:
        self.impl.chunk_status["P1_snips"] = {0: "skipped_toomanybad"}            # II chunk: II segment excluded only
        self.impl.chunk_status["P2_snips"] = {2: "skipped_toomanybad", 3: "failed"}  # ictal/postictal only: nothing
        self.impl.chunk_interpolated["P3_snips"] = {1: 7}                          # 7/20 = 0.35 > 0.30: preictal excluded
        result = self.run_stage()
        summary = result["summary"]
        rows = read_rows(self.out / "heppy_exclusions.csv")
        self.assertEqual(list(rows[0]), stage.EXCLUSION_COLS)
        self.assertEqual([(r["base"], r["scope"], r["reason_code"], r["segment"]) for r in rows],
                         [("P1_snips", "segment", "segment_pyprep_skipped_toomanybad", "II"),
                          ("P3_snips", "segment", "segment_excess_interpolation", "preictal")])
        self.assertEqual((rows[0]["output_start_seconds"], rows[0]["output_end_seconds"], rows[0]["source_row"]),
                         ("0.0", "300.0", "2"))
        self.assertTrue(all(r["included"] for r in summary["recordings"].values()), "no recording-level exclusion")
        consort = summary["consort"]
        self.assertEqual(consort["recording_level"]["excluded"], 0)
        seg = consort["segment_level"]
        self.assertEqual(seg["in_scope_segments"], {"II": 3, "preictal": 3})
        self.assertEqual(seg["usable_in_scope_segments"], {"II": 2, "preictal": 2})
        self.assertEqual(seg["excluded_by_segment_rule"],
                         {"segment_excess_interpolation": 1, "segment_pyprep_skipped_toomanybad": 1})
        self.assertEqual(seg["recordings_with_a_usable_segment_of_each_scope_kind"], 1)   # only P2 has both
        self.assertEqual(summary["recordings"]["P1_snips"]["in_scope_segments"],
                         {"II": {"in_scope": 1, "usable": 0}, "preictal": {"in_scope": 1, "usable": 1}})
        first = (self.out / "heppy_exclusions.csv").read_bytes()
        self.run_stage()
        self.assertEqual((self.out / "heppy_exclusions.csv").read_bytes(), first)

    def test_recording_level_exclusion_subsumes_segments(self) -> None:
        self.impl.n_eeg["P2_snips"] = 10
        self.impl.chunk_status["P2_snips"] = {0: "failed"}
        result = self.run_stage()
        rows = read_rows(self.out / "heppy_exclusions.csv")
        self.assertEqual([(r["base"], r["scope"], r["reason_code"]) for r in rows], [("P2_snips", "recording", "too_few_eeg_channels")])
        self.assertEqual(rows[0]["segment"], "")
        seg = result["summary"]["consort"]["segment_level"]
        self.assertEqual(seg["excluded_with_recording"], {"II": 1, "preictal": 1})
        self.assertEqual(seg["usable_in_scope_segments"], {"II": 2, "preictal": 2})
        self.assertEqual(result["summary"]["flags"], {})
        self.impl.chunk_status["P3_snips"] = {1: "ransac_fallback"}
        self.assertEqual(self.run_stage(force=True)["summary"]["flags"], {"pyprep_ransac_fallback": ["P3_snips"]})

    def test_changed_clip_recomputes_only_that_recording(self) -> None:
        self.run_stage()
        self.impl.calls.clear()
        (self.clips / "P3_snips.edf").write_bytes(b"EDF-CHANGED" * 40)
        self.run_stage()
        self.assertEqual({b for b, _s in self.impl.calls}, {"P3_snips"})
        self.assertEqual({s for _b, s in self.impl.calls}, {"ecg_fit", "preprocess", "recovery"})

    def test_touched_but_identical_clip_is_not_recomputed(self) -> None:
        self.run_stage()
        self.impl.calls.clear()
        edf = self.clips / "P1_snips.edf"
        os.utime(edf, ns=(edf.stat().st_atime_ns, edf.stat().st_mtime_ns + 10**9))
        self.run_stage()
        self.assertEqual(self.impl.calls, [])

    def test_only_and_limit(self) -> None:
        result = self.run_stage(only=["P2_snips"])
        self.assertEqual(result["actions"], {"P2_snips": "compute"})
        self.assertEqual(result["summary"]["consort"]["not_run"], 2)
        result = self.run_stage(limit=2)
        self.assertEqual(sorted(result["actions"]), ["P1_snips", "P2_snips"])

    def test_missing_clip_is_infrastructure_error(self) -> None:
        (self.clips / "P3_snips.edf").unlink()
        result = self.run_stage()
        self.assertEqual(result["exit_code"], 2)
        self.assertEqual(result["actions"]["P3_snips"], "missing_edf")

    def test_guard_refuses_restricted_paths_without_touching_them(self) -> None:
        import fsms.guard as guard

        self.assertTrue(guard.is_restricted(RESTRICTED_PROBE))
        with self.assertRaises(RestrictedDataError):
            stage.run_stage(self.clips, RESTRICTED_PROBE, self.options_path, in_process=True, impl=self.impl,
                            log=lambda m: None)
        with self.assertRaises(RestrictedDataError):
            stage.run_stage(r"D:\EEGData\__never_created__", self.out, self.options_path, in_process=True,
                            impl=self.impl, log=lambda m: None)
        with self.assertRaises(RestrictedDataError):
            stage.run_stage(self.clips, self.out, self.options_path, adopt_dir=r"E:\EEGs\__never_created__",
                            in_process=True, impl=self.impl, log=lambda m: None)
        with mock.patch("fsms.lock.real_data_unlock_problems", return_value=["no registration record"]):
            with self.assertRaises(RestrictedDataError):   # real mode is gated by the preregistration lock
                stage.run_stage(self.clips, self.out, self.options_path, mode="real", in_process=True,
                                impl=self.impl, log=lambda m: None)
        self.assertEqual(self.impl.calls, [])
        self.assertFalse(self.out.exists())


def make_legacy_tree(root: Path, bases: list[str], options: dict | None = None) -> Path:
    legacy = root / "legacy_heppy_output"
    (legacy / "logs").mkdir(parents=True)
    legacy_options = options if options is not None else json.loads((ROOT / "config" / "heppy_options_fixture.json").read_text())
    legacy_options = {k: v for k, v in legacy_options.items() if k not in ("seeds", "qc_rule", "mains_hz", "auto_qrs",
                                                                         "rpeak_recovery", "prep_ransac", "target_sfreq",
                                                                         "schema_version", "_comment")}
    (legacy / "options_used.json").write_text("\ufeff" + json.dumps(legacy_options, indent=4), encoding="utf-8")
    rows = []
    for i, base in enumerate(bases):
        seed = 100 + i
        write_tiny_fif(legacy / "raw_fif" / f"{base}_pp_raw.fif", seed)
        write_tiny_fif(legacy / "raw_fif" / f"{base}_pp_raw_keepcfa.fif", seed)
        (legacy / "ecg_qc").mkdir(exist_ok=True)
        times = np.arange(1.0, 58.0, 0.8)
        (legacy / "ecg_qc" / f"{base}_rpeaks_corrected.csv").write_text(
            "rpeak_time_s,rpeak_source,is_gap_recovered\n" + "".join(f"{float(t)!r},reviewed,0\n" for t in times), encoding="utf-8")
        rows.append({"base": base, "edf_path": rf"G:\gone\{base}.edf", "status": "ok", "error": "",
                     "raw_fif": rf"G:\gone\{base}_pp_raw.fif", "raw_fif_keep": ""})
    with (legacy / "logs" / "headless_preprocessing_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return legacy


class AdoptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="fsms_adopt_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.bases = ["A_snips", "B_snips", "C_snips", "D_snips"]
        self.clips = make_clips(self.tmp, self.bases)
        self.out = self.tmp / "heppy_output"
        self.impl = StubImpl()
        self.options_path = self.tmp / "options.json"
        shutil.copy(OPTIONS, self.options_path)

    def run_stage(self, legacy: Path, **kwargs):
        return quietly(stage.run_stage, self.clips, self.out, self.options_path, adopt_dir=legacy, in_process=True,
                       impl=self.impl, log=lambda m: None, **kwargs)

    def test_compatible_tree_is_adopted_in_place_and_never_written(self) -> None:
        legacy = make_legacy_tree(self.tmp, self.bases)
        # break three of them in distinct ways
        summary = legacy / "logs" / "headless_preprocessing_summary.csv"
        rows = read_rows(summary)
        rows[1]["status"] = "error"
        with summary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (legacy / "ecg_qc" / "C_snips_rpeaks_corrected.csv").unlink()
        fif = legacy / "raw_fif" / "D_snips_pp_raw.fif"
        fif.write_bytes(fif.read_bytes()[:500])
        legacy_before = tree_hash(legacy)
        result = self.run_stage(legacy)
        self.assertEqual(tree_hash(legacy), legacy_before, "adopt dir must be read-only")
        self.assertEqual(result["actions"], {"A_snips": "adopted", "B_snips": "compute", "C_snips": "compute",
                                             "D_snips": "compute"})
        adoption = result["summary"]["adoption"]
        self.assertEqual(adoption["adopted"], ["A_snips"])
        self.assertTrue(adoption["options_compat"]["compatible"])
        self.assertIn("status is 'error'", adoption["not_adopted"]["B_snips"])
        self.assertIn("sidecar", adoption["not_adopted"]["C_snips"])
        self.assertIn("raw_fif", adoption["not_adopted"]["D_snips"])
        sentinel = json.loads((self.out / ".sentinels" / "A_snips.json").read_text())
        self.assertTrue(sentinel["adopted"])
        info = sentinel["adoption"]["files"]["raw_fif"]
        self.assertEqual(Path(info["path"]).parent, legacy / "raw_fif")        # in place: no copy
        self.assertFalse((self.out / "raw_fif" / "A_snips_pp_raw.fif").exists())
        self.assertTrue(sentinel["adoption"]["options_compat"]["compatible"])
        self.assertEqual(result["summary"]["recordings"]["A_snips"]["adopted"], True)
        self.assertIn("segment_pyprep_status", result["summary"]["not_evaluable"])   # honest about missing evidence
        review = {r["base"]: r for r in read_rows(self.out / "ecg_qc_review.csv")}
        self.assertTrue(review["A_snips"]["raw_fif"].startswith(str(legacy)))
        # re-run: adoption kept, nothing recomputed, legacy still untouched
        self.impl.calls.clear()
        again = self.run_stage(legacy)
        self.assertEqual(again["actions"]["A_snips"], "adopted_unchanged")
        self.assertNotIn("A_snips", {b for b, _ in self.impl.calls})
        self.assertEqual(tree_hash(legacy), legacy_before)

    def test_legacy_row_log_attributes_pyprep_failures_to_segments(self) -> None:
        legacy = make_legacy_tree(self.tmp, self.bases[:1])
        rows_dir = legacy / "logs" / "headless_rows"
        rows_dir.mkdir()
        fit = "HEPPY_STAGE\tpyprep_fit:r=0:channel_wise=0\trss_gib=0.5"
        warn = "PyPREP skipped; too many bad EEG channels for robust reference: RobustReference:TooManyBad"
        lines = [fit, fit, "2026-01-01 00:00:00,000 - root - WARNING - " + warn, warn, fit, fit]   # chunk 1 = preictal
        (rows_dir / "001_A_snips.log").write_text("\n".join(lines), encoding="utf-8")
        result = self.run_stage(legacy)
        self.assertEqual(result["actions"]["A_snips"], "adopted")
        rows = read_rows(self.out / "heppy_exclusions.csv")
        self.assertEqual([(r["base"], r["scope"], r["reason_code"], r["segment"]) for r in rows],
                         [("A_snips", "segment", "segment_pyprep_skipped_toomanybad", "preictal")])
        self.assertIn("segment_interpolated_fraction", result["summary"]["not_evaluable"])
        # an ambiguous log (RANSAC announces fits twice) is not attributed to segments
        (rows_dir / "001_A_snips.log").write_text("HEPPY_STAGE\tpyprep_fit:r=1:channel_wise=0\n" + warn, encoding="utf-8")
        shutil.rmtree(self.out)
        again = self.run_stage(legacy)
        self.assertIn("segment_pyprep_attribution", again["summary"]["not_evaluable"])
        self.assertEqual(read_rows(self.out / "heppy_exclusions.csv"), [])

    def test_incompatible_options_block_adoption(self) -> None:
        legacy = make_legacy_tree(self.tmp, self.bases[:1])
        changed = json.loads((legacy / "options_used.json").read_text(encoding="utf-8-sig"))
        changed["filters"]["hp"] = 1.0
        (legacy / "options_used.json").write_text(json.dumps(changed), encoding="utf-8")
        result = self.run_stage(legacy)
        self.assertEqual(result["actions"]["A_snips"], "compute")
        compat = result["summary"]["adoption"]["options_compat"]
        self.assertFalse(compat["compatible"])
        self.assertEqual(compat["differing_keys"], ["filters.hp"])

    def test_mains_mismatch_blocks_adoption_of_50hz_legacy_under_60hz_lock(self) -> None:
        real = json.loads((ROOT / "config" / "heppy_options.json").read_text())
        legacy = make_legacy_tree(self.tmp, self.bases[:1], options=real)
        result = self.run_stage(legacy)
        self.assertEqual(result["actions"]["A_snips"], "compute")
        self.assertEqual(result["summary"]["adoption"]["options_compat"]["differing_keys"], ["filters.line_freqs"])

    def test_adopt_dir_inside_output_rejected(self) -> None:
        with self.assertRaises(stage.InfrastructureError):
            stage.run_stage(self.clips, self.out, self.options_path, adopt_dir=self.out / "legacy", in_process=True,
                            impl=self.impl, log=lambda m: None)


if __name__ == "__main__":
    unittest.main()
