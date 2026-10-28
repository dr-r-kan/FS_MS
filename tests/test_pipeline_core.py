"""Core orchestration safety: restricted-data guard, ledger idempotency, config gating, lock gate."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fsms import atomic, guard, ledger, lock, pipeline  # noqa: E402
from fsms.config import Config, ConfigError  # noqa: E402


def write_config(root: Path, **overrides) -> Path:
    raw = {"schema": "fsms-config/v1", "name": "t", "kind": "fixture",
           "paths": {"data_root": str(root / "data"), "run_root": str(root / "run")}}
    raw.update(overrides)
    path = root / "cfg.json"
    path.write_text(json.dumps(raw))
    return path


class GuardTests(unittest.TestCase):
    def test_restricted_roots_refused_case_insensitively_and_nested(self):
        for path in (r"F:\FS_microstates_study", r"f:\fs_microstates_study\clips\x.edf",
                     r"D:\EEGData\anything", r"G:\Chalfont\a.edf", r"F:\EEGs\Anon NHNN"):
            self.assertTrue(guard.is_restricted(path), path)
            with self.assertRaises(guard.RestrictedDataError):
                guard.assert_unrestricted(path)

    def test_sibling_with_shared_prefix_is_not_restricted(self):
        self.assertFalse(guard.is_restricted(r"F:\FS_microstates_study_fixture\clips"))
        self.assertFalse(guard.is_restricted(r"F:\TUEG\v1.2.0\edf"))
        self.assertFalse(guard.is_restricted(r"F:\FS_MS_fixture_tueg"))

    def test_fixture_config_rejects_restricted_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = write_config(root, paths={"data_root": r"F:\FS_microstates_study", "run_root": str(root / "run")})
            with self.assertRaises(ConfigError):
                Config.load(path)
            path = write_config(root, paths={"data_root": str(root), "run_root": r"D:\EEGData\out"})
            with self.assertRaises(ConfigError):
                Config.load(path)
            path = write_config(root, paths={"data_root": str(root), "run_root": str(root / "r"),
                                             "adopt_heppy_dir": r"F:\FS_microstates_study\heppy_output"})
            with self.assertRaises(ConfigError):
                Config.load(path)

    def test_real_config_may_not_write_inside_data_root(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_config(Path(td), kind="real", paths={"data_root": r"F:\FS_microstates_study",
                                                              "run_root": r"F:\FS_microstates_study\out"})
            with self.assertRaises(ConfigError):
                Config.load(path)

    def test_shipped_configs_load_and_real_is_gated(self):
        study = Config.load(REPO / "config" / "study.json")
        fixture = Config.load(REPO / "config" / "fixture_tueg.json")
        self.assertEqual((study.kind, fixture.kind), ("real", "fixture"))
        self.assertTrue(guard.is_restricted(study.data_root))
        self.assertFalse(guard.is_restricted(study.run_root))
        with mock.patch.object(lock, "real_data_unlock_problems", return_value=["not registered"]):
            with self.assertRaisesRegex(pipeline.PipelineError, "LOCKED"):
                pipeline.run(study)

    def test_unregistered_real_run_never_touches_data(self):
        """Without a registration record the gate must fail before any stage function is called."""
        study = Config.load(REPO / "config" / "study.json")
        with mock.patch.object(lock, "REGISTRATION_PATH", Path(tempfile.gettempdir()) / "no_such_registration.json"):
            with mock.patch.object(pipeline, "STAGES", {}):  # any stage lookup would KeyError if reached
                with self.assertRaises(pipeline.PipelineError):
                    pipeline.run(study)


class LedgerTests(unittest.TestCase):
    def test_current_then_stale_on_fingerprint_output_change(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            led = ledger.Ledger(root / "state")
            out = root / "out.txt"
            out.write_text("hello")
            led.mark_done("s", "fp1", [out])
            self.assertEqual(led.verify("s", "fp1"), (True, "current"))
            self.assertFalse(led.verify("s", "fp2")[0])
            out.write_text("hellO")  # same size, different content (below hash limit)
            self.assertIn("content changed", led.verify("s", "fp1")[1])
            out.write_text("hellO!")
            self.assertIn("size changed", led.verify("s", "fp1")[1])
            out.unlink()
            self.assertIn("missing", led.verify("s", "fp1")[1])

    def test_mark_done_refuses_phantom_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                ledger.Ledger(Path(td)).mark_done("s", "fp", [Path(td) / "nope"])

    def test_atomic_write_leaves_no_partial_file_on_failure(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "a.json"
            atomic.atomic_write_json(target, {"a": 1})
            with self.assertRaises(TypeError):
                atomic.atomic_write_bytes(target, "not bytes")  # type: ignore[arg-type]
            self.assertEqual(json.loads(target.read_text()), {"a": 1})
            self.assertEqual(list(Path(td).glob("*.part")), [])

    def test_json_content_digest_ignores_utc_and_order(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td) / "a.json", Path(td) / "b.json"
            a.write_text(json.dumps({"x": [1, 2], "created_utc": "A", "y": 3}))
            b.write_text(json.dumps({"y": 3, "created_utc": "B", "x": [1, 2]}, indent=2))
            self.assertEqual(atomic.json_content_digest(a), atomic.json_content_digest(b))
            b.write_text(json.dumps({"y": 4, "created_utc": "B", "x": [1, 2]}))
            self.assertNotEqual(atomic.json_content_digest(a), atomic.json_content_digest(b))

    def test_fingerprint_is_order_independent(self):
        self.assertEqual(atomic.fingerprint({"a": 1, "b": [1, 2]}), atomic.fingerprint({"b": [1, 2], "a": 1}))
        self.assertNotEqual(atomic.fingerprint({"a": 1}), atomic.fingerprint({"a": 2}))

    def test_run_lock_blocks_second_run_and_reclaims_stale(self):
        with tempfile.TemporaryDirectory() as td:
            with pipeline.RunLock(Path(td)):
                with self.assertRaises(pipeline.PipelineError):
                    with pipeline.RunLock(Path(td)):
                        pass
            (Path(td) / ".run.lock").write_text(json.dumps({"pid": 99999999, "created": 0}))
            with pipeline.RunLock(Path(td)):  # stale pid reclaimed
                pass


class LockTests(unittest.TestCase):
    def test_registration_url_validation(self):
        for bad in ("", "http://osf.io/abcde", "https://example.com/x", "https://osf.io/",
                    "https://osf.io/REPLACE_ME", "https://osf.io/fixture-not-a-registration",
                    "https://osf.io/synthetic"):
            with self.assertRaises(lock.LockError):
                lock.validate_registration_url(bad)
        lock.validate_registration_url("https://osf.io/ab12c")
        lock.validate_registration_url("https://doi.org/10.17605/OSF.IO/AB12C")

    def test_unlock_requires_registration_record_matching_lock(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fake_lock = root / "PLAN_LOCK.json"
            fake_lock.write_text("{}")
            reg = root / "REGISTRATION.json"
            with mock.patch.object(lock, "verify_lock", return_value=[]), \
                    mock.patch.object(lock, "LOCK_PATH", fake_lock), mock.patch.object(lock, "REGISTRATION_PATH", reg):
                self.assertTrue(any("no registration record" in p for p in lock.real_data_unlock_problems()))
                reg.write_text(json.dumps({"registration_url": "https://osf.io/ab12c", "plan_lock_sha256": "0" * 64}))
                self.assertTrue(any("different plan lock" in p for p in lock.real_data_unlock_problems()))
                reg.write_text(json.dumps({"registration_url": "https://osf.io/ab12c",
                                           "plan_lock_sha256": atomic.sha256_file(fake_lock)}))
                self.assertEqual(lock.real_data_unlock_problems(), [])
                with mock.patch.object(lock, "verify_lock", return_value=["locked file changed: x"]):
                    self.assertTrue(lock.real_data_unlock_problems())

    def test_lock_detects_changed_file(self):
        """Create a lock over a temp 'repo' and show a one-byte edit is detected."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "PREREGISTRATION.md").write_text("plan")
            prereg = root / "prereg"
            with mock.patch.object(lock, "REPO", root), mock.patch.object(lock, "PREREG_DIR", prereg), \
                    mock.patch.object(lock, "LOCK_PATH", prereg / "PLAN_LOCK.json"), \
                    mock.patch.object(lock, "SIDECAR_PATH", prereg / "PLAN_LOCK.sha256"), \
                    mock.patch.object(lock, "REGISTRATION_PATH", prereg / "REGISTRATION.json"), \
                    mock.patch.object(lock.provenance, "verify", return_value=[]), \
                    mock.patch.object(lock.provenance, "PROVENANCE_PATH", root / "prov.json"), \
                    mock.patch.object(lock.provenance, "compute",
                                      return_value={"externals": {"X": {"tree_sha256": "t"}}}), \
                    mock.patch.object(lock, "runtime_mismatches", return_value=[]), \
                    mock.patch.object(lock, "installed_versions", return_value={}):
                (root / "prov.json").write_text(json.dumps({"externals": {"X": {"tree_sha256": "t"}}}))
                lock.create_lock()
                self.assertEqual(lock.verify_lock(), [])
                (root / "PREREGISTRATION.md").write_text("plan!")
                self.assertTrue(any("locked file changed: PREREGISTRATION.md" in p for p in lock.verify_lock()))
                (root / "PREREGISTRATION.md").write_text("plan")
                self.assertEqual(lock.verify_lock(), [])
                lock.register("https://osf.io/ab12c")
                with self.assertRaisesRegex(lock.LockError, "frozen"):
                    lock.create_lock()


if __name__ == "__main__":
    unittest.main()
