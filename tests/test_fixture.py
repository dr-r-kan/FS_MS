"""Tests for the synthetic TUEG fixture builder and the tracked clip stage it drives.

Fast tests use synthetic header rows and tiny hand-written EDFs (no TUEG).  The slow end-to-end test
(real TUEG headers, real HEPPy, VG converter; ~30-60 min) runs only when ``FSMS_RUN_SLOW=1`` and
``F:\\TUEG`` exists; it writes under a fresh temporary directory (or ``FSMS_FIXTURE_ROOT``), never
under any restricted real-study root.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fsms.clips import snip_clips
from fsms.fixture import make_tueg_fixture as fx
from fsms.guard import RestrictedDataError

ROOT = Path(__file__).resolve().parents[1]
RESTRICTED_PROBE = r"F:\FS_microstates_study\__never_created__\fixture"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fake_rows(n_patients: int = 60) -> list[dict]:
    """Synthetic header rows shaped like TUEG scans; durations span single/double pools."""
    labels = [f"EEG {name}-REF" for name in fx.STANDARD_19] + ["EEG EKG1-REF", "EEG A1-REF", "EEG A2-REF"]
    rows = []
    for i in range(n_patients):
        patient = f"{i:08d}"
        for j, duration in enumerate((1200.0, 3400.0 + 10 * i, 6000.0 + 30 * i, 20000.0)):
            rows.append({"path": rf"C:\fake\tueg\{patient}\s{j:03d}\{patient}_s{j:03d}_t000.edf", "patient": patient,
                         "duration_s": duration, "record_s": 1.0, "sfreq": 256.0, "labels": labels,
                         "samples": [256] * len(labels), "header_len": 256 + 256 * len(labels)})
    return rows


class ChooseTests(unittest.TestCase):
    def test_deterministic_and_composition(self) -> None:
        rows = fake_rows()
        a, b = fx.choose(rows, 20261028), fx.choose(rows, 20261028)
        self.assertEqual(json.dumps(a, sort_keys=True, default=str), json.dumps(b, sort_keys=True, default=str))
        other = fx.choose(rows, 7)
        self.assertNotEqual([p["row"]["patient"] for p in a["patients"]], [p["row"]["patient"] for p in other["patients"]])
        patients = a["patients"]
        self.assertEqual(len(patients), 14)
        self.assertEqual(len({p["row"]["patient"] for p in patients}), 14, "distinct TUEG patients")
        self.assertEqual([p["code"] for p in patients], [f"FIX_{i:03d}" for i in range(1, 15)])
        groups = [p["group"] for p in patients]
        self.assertEqual((groups.count("FS"), groups.count("ES"), groups.count("DD")), (5, 5, 4))
        roles = [p["role"] for p in patients]
        self.assertEqual(roles.count("FS_two_events"), 1)
        self.assertEqual(roles.count("DD"), 4)
        self.assertEqual(roles.count("ES"), 5)
        singles = [p for p in patients if p["role"].startswith("FS_single")]
        self.assertEqual(sorted(p["events"][0][1] for p in singles), ["hyper", "hyper", "hypo", "hypo"])
        for p in singles:
            self.assertEqual(len(p["events"]), 1)
            self.assertEqual(p["events"][0][0], "FS")
        two = next(p for p in patients if p["role"] == "FS_two_events")
        self.assertEqual(sorted(h for _s, h in two["events"]), ["hyper", "hypo"])
        self.assertEqual({s_ for s_, _h in two["events"]}, {"FS"})
        for p in patients:
            if p["role"] == "DD":
                self.assertEqual(sorted(s_ for s_, _h in p["events"]), ["ES", "FS"])
            if p["role"] == "ES":
                self.assertEqual(p["events"], [("ES", "")])
        for p in patients:
            n = len(p["events"])
            lo, hi = fx.DOUBLE_DURATION if n == 2 else fx.SINGLE_DURATION
            self.assertTrue(lo <= p["row"]["duration_s"] <= hi)
            previous_end = None
            for onset, end in p["ictal"]:
                self.assertGreaterEqual(onset, fx.FIRST_ONSET_S)
                self.assertTrue(30 <= end - onset <= 60)
                if previous_end is not None:
                    self.assertGreaterEqual(onset - previous_end, 2400, "no transictal merge")
                previous_end = end
            self.assertLessEqual(p["ictal"][-1][1] + fx.POSTICTAL_S, p["row"]["duration_s"])

    def test_candidate_filter(self) -> None:
        row = fake_rows(1)[1]
        self.assertTrue(fx.is_candidate(row))
        self.assertFalse(fx.is_candidate({**row, "sfreq": 512.0}))
        self.assertFalse(fx.is_candidate({**row, "labels": [l for l in row["labels"] if "EKG" not in l]}))
        self.assertFalse(fx.is_candidate({**row, "labels": [l for l in row["labels"] if "FP1" not in l]}))

    def test_database_rows_follow_real_schema_and_timeline(self) -> None:
        plan = fx.choose(fake_rows(), 1)
        rows = fx.database_rows(plan)
        self.assertEqual(set(rows[0]), set(fx.DB_COLUMNS))
        self.assertEqual(len({r["source_row"] for r in rows}), len(rows))
        by_patient: dict[str, list[dict]] = {}
        for r in rows:
            by_patient.setdefault(r["patient_code"], []).append(r)
        for code, items in by_patient.items():
            ii = [r for r in items if r["event_type"] == "II"]
            self.assertEqual(len(ii), 1)
            self.assertEqual(float(ii[0]["end_seconds"]) - float(ii[0]["start_seconds"]), fx.II_END_S - fx.II_START_S)
            self.assertGreaterEqual(float(ii[0]["start_seconds"]), 120.0, "II avoids the artifact-prone opening")
            self.assertLessEqual(float(ii[0]["end_seconds"]), fx.FIRST_ONSET_S - 1200)
            self.assertTrue(all("FIXTURE" in r["notes"] or "FIXTURE" in r["source"] for r in items))
        dd = [r for r in rows if r["patient_type"] == "DD" and r["event_type"] == "Ict"]
        self.assertEqual(sorted({r["seizure_type"] for r in dd}), ["ES", "FS"])
        text = fx.csv_text(rows).decode("utf-8")
        self.assertTrue(text.startswith(",".join(fx.DB_COLUMNS)))


class BuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="fsms_fixture_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.rows = fake_rows()
        self.fake_tueg = self.tmp / "tueg"
        self._orig_scan = fx.scan_headers
        fx.scan_headers = lambda root, max_headers, cache_path=None: self.rows
        self.addCleanup(setattr, fx, "scan_headers", self._orig_scan)

    def materialise(self, plan_rows: list[dict]) -> dict[str, str]:
        hashes = {}
        for row in plan_rows:
            path = Path(row["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(os.urandom(4096))
            hashes[str(path)] = sha(path)
        return hashes

    def test_build_writes_labelled_synthetic_study_and_leaves_sources_untouched(self) -> None:
        # redirect the fake TUEG paths into the temp dir
        for row in self.rows:
            row["path"] = str(self.tmp / "tueg" / Path(row["path"]).name)
        plan = fx.choose(self.rows, 20261028)
        hashes = self.materialise([p["row"] for p in plan["patients"]])
        work = self.tmp / "work"
        logs: list[str] = []
        fx.build(work, self.fake_tueg, 20261028, 100, run_clips=False, log=logs.append)
        self.assertTrue((work / "FIXTURE_NOT_REAL_DATA.txt").read_text().startswith("FIXTURE - NOT REAL DATA"))
        manifest = json.loads((work / "fixture_manifest.json").read_text())
        self.assertIs(manifest["fixture"], True)
        self.assertIs(manifest["not_real_data"], True)
        self.assertEqual(len(manifest["sources"]), 14)
        self.assertEqual(manifest["design"]["cohort"], fx.COHORT)
        for source in manifest["sources"]:
            self.assertEqual(len(source["header_sha256"]), 64)
            self.assertEqual(len(source["first_mib_sha256"]), 64)
        for path, digest in hashes.items():
            self.assertEqual(sha(Path(path)), digest, "TUEG sources unmodified")
        db_before = sha(work / "analysis_database.csv")
        fx.build(work, self.fake_tueg, 20261028, 100, run_clips=False, log=lambda m: None)
        self.assertEqual(sha(work / "analysis_database.csv"), db_before, "rebuild is deterministic")
        self.assertEqual(len(read_rows(work / "analysis_database.csv")), 33)

    def test_guard_refuses_restricted_work_root_and_tueg_root(self) -> None:
        with self.assertRaises(RestrictedDataError):
            fx.main(["--work-root", RESTRICTED_PROBE])
        with self.assertRaises(RestrictedDataError):
            fx.main(["--work-root", str(self.tmp / "w"), "--tueg-root", r"D:\EEGData\__never_created__"])
        with mock.patch("fsms.lock.real_data_unlock_problems", return_value=["locked"]):
            with self.assertRaises(RestrictedDataError):
                fx.main(["--work-root", str(self.tmp / "w"), "--mode", "real"])
        self.assertFalse((self.tmp / "w").exists())


class ClipStageTests(unittest.TestCase):
    """The tracked clip stage: idempotent, label-blind fingerprints, byte-identical to the legacy writer."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="fsms_clip_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.edf = self.tmp / "source.edf"
        snip_clips.write_fixture_edf(self.edf)
        self.db = self.tmp / "db.csv"
        self.fields = ["patient_code", "patient_id", "event_type", "event_number", "segment_number", "source_row",
                       "start_seconds", "end_seconds", "migrated_edf_path", "edf_path", "patient_type", "seizure_type"]
        self.write_db("FS")

    def write_db(self, group: str) -> None:
        rows = [("II", "0", "1", "2", "1", "3"), ("Ict", "1", "", "3", "5", "6")]
        with self.db.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            writer.writeheader()
            for et, en, sn, sr, a, b in rows:
                writer.writerow({"patient_code": "P_test", "patient_id": "1", "event_type": et, "event_number": en,
                                 "segment_number": sn, "source_row": sr, "start_seconds": a, "end_seconds": b,
                                 "migrated_edf_path": str(self.edf), "patient_type": group, "seizure_type": group})

    def run_clips(self, **kwargs) -> dict[str, str]:
        actions: dict[str, str] = {}
        snip_clips.write_snips(self.db, self.tmp / "out", self.tmp / "out" / "snip_keyfile.csv",
                               self.tmp / "out" / "snip_skipped.csv", actions=actions, **kwargs)
        return actions

    def test_idempotent_label_blind_and_self_healing(self) -> None:
        out = self.tmp / "out" / "P_test_snips.edf"
        self.assertEqual(self.run_clips(), {"P_test": "written"})
        before = (out.stat().st_mtime_ns, sha(out), sha(self.tmp / "out" / "snip_keyfile.csv"))
        self.assertEqual(self.run_clips(), {"P_test": "skipped_unchanged"})
        self.assertEqual((out.stat().st_mtime_ns, sha(out), sha(self.tmp / "out" / "snip_keyfile.csv")), before)
        self.write_db("ES")                                   # diagnostic labels are not part of the fingerprint
        self.assertEqual(self.run_clips(), {"P_test": "skipped_unchanged"})
        out.write_bytes(out.read_bytes()[:-9])               # truncated output heals
        self.assertEqual(self.run_clips(), {"P_test": "written"})
        self.assertEqual(sha(out), before[1])
        stale = self.tmp / "out" / ".P_test_snips.edf.part-77"
        stale.write_bytes(b"x")
        self.assertEqual(self.run_clips(), {"P_test": "skipped_unchanged"})
        self.assertFalse(stale.exists(), "stale temporaries removed")
        os.utime(self.edf, ns=(1, 2))                         # source mtime changed -> new fingerprint
        self.assertEqual(self.run_clips(), {"P_test": "written"})
        self.assertEqual(self.run_clips(force=True), {"P_test": "written"})
        self.assertEqual(sha(out), before[1])
        rows = read_rows(self.tmp / "out" / "snip_keyfile.csv")
        self.assertIn("preictal", {r["segment"] for r in rows})

    def test_changed_event_time_rewrites(self) -> None:
        self.run_clips()
        text = self.db.read_text(encoding="utf-8").replace(",5,6,", ",5,7,")
        self.db.write_text(text, encoding="utf-8")
        self.assertEqual(self.run_clips(), {"P_test": "written"})

    def test_cli_guard_and_root_independence(self) -> None:
        with self.assertRaises(RestrictedDataError):
            snip_clips.main(["--database", str(self.db), "--out-dir", RESTRICTED_PROBE])
        with self.assertRaises(RestrictedDataError):
            snip_clips.main(["--database", r"E:\EEGs\__never_created__\analysis_database.csv", "--out-dir", str(self.tmp / "o")])
        # runs from an unrelated cwd with explicit paths only
        cwd = self.tmp / "elsewhere"
        cwd.mkdir()
        done = subprocess.run([sys.executable, "-m", "fsms.clips.snip_clips", "--database", str(self.db),
                               "--out-dir", str(self.tmp / "cli_out")], cwd=cwd, capture_output=True, text=True,
                              env=dict(os.environ, PYTHONPATH=str(ROOT)))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertTrue((self.tmp / "cli_out" / "P_test_snips.edf").exists())
        self.assertIn("1 written", done.stdout)

    @unittest.skipUnless((ROOT / "OLD" / "snip_clips.py").exists(), "legacy OLD/snip_clips.py not present")
    def test_byte_identical_to_legacy_writer(self) -> None:
        spec = importlib.util.spec_from_file_location("legacy_snip_clips", ROOT / "OLD" / "snip_clips.py")
        legacy = importlib.util.module_from_spec(spec)
        sys.modules["legacy_snip_clips"] = legacy
        spec.loader.exec_module(legacy)
        legacy.write_snips(self.db, self.tmp / "old", self.tmp / "old_manifest.csv", self.tmp / "old_skipped.csv")
        self.run_clips()
        self.assertEqual((self.tmp / "old" / "P_test_snips.edf").read_bytes(), (self.tmp / "out" / "P_test_snips.edf").read_bytes())
        self.assertEqual((self.tmp / "old_manifest.csv").read_bytes(), (self.tmp / "out" / "snip_keyfile.csv").read_bytes())

    def test_self_check(self) -> None:
        snip_clips.self_check()


SLOW = os.environ.get("FSMS_RUN_SLOW") == "1" and Path(r"F:\TUEG").exists()


@unittest.skipUnless(SLOW, "set FSMS_RUN_SLOW=1 (and have F:\\TUEG) to run the end-to-end fixture pipeline")
class SlowEndToEnd(unittest.TestCase):
    """fixture -> clips -> HEPPy -> manifest, run twice; the second run must change nothing."""

    def run_module(self, *args: str) -> str:
        done = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True,
                              env=dict(os.environ, PYTHONPATH=str(ROOT)))
        self.assertIn(done.returncode, (0,), done.stdout[-2000:] + done.stderr[-2000:])
        return done.stdout

    def snapshot(self, root: Path) -> dict[str, str]:
        skip = {"logs", ".sentinels", ".jobs", ".convert_stage", "logs_driver"}
        return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
                if p.is_file() and not (set(p.relative_to(root).parts) & skip) and p.name not in (".stage.lock",)}

    def test_end_to_end_and_idempotent_rerun(self) -> None:
        work = Path(os.environ.get("FSMS_FIXTURE_ROOT") or tempfile.mkdtemp(prefix="fsms_e2e_"))
        from fsms.guard import assert_unrestricted

        assert_unrestricted(work, "slow end-to-end work root")
        options = str(ROOT / "config" / "heppy_options_fixture.json")

        def pipeline() -> str:
            out = self.run_module("fsms.fixture.make_tueg_fixture", "--work-root", str(work))
            out += self.run_module("fsms.heppy.run_heppy_stage", "--clips-dir", str(work / "clips"), "--heppy-out",
                                   str(work / "heppy_output"), "--options", options, "--workers", "3")
            out += self.run_module("fsms.manifest.build_vg_manifest", "--snip-keyfile",
                                   str(work / "clips" / "snip_keyfile.csv"), "--heppy-out", str(work / "heppy_output"),
                                   "--database", str(work / "analysis_database.csv"), "--out-dir",
                                   str(work / "vg_microstates"), "--format", "mat")
            return out

        pipeline()
        first = self.snapshot(work)
        second_log = pipeline()
        self.assertIn("recomputed=[]", second_log)
        self.assertEqual(self.snapshot(work), first)


if __name__ == "__main__":
    unittest.main()
