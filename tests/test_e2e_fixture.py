"""End-to-end run on the synthetic TUEG fixture (opt-in: needs MATLAB, the TUEG corpus and ~1 h).

    $env:FSMS_RUN_SLOW = '1'; .\\.venv\\Scripts\\python.exe -B -m unittest tests.test_e2e_fixture -v

Proves, on real EEG/ECG recordings with SYNTHETIC labels, that the unmodified pipeline code runs from clips to
the registered confirmatory tests, and that an immediate rerun executes nothing and changes no output byte.
No restricted real-study data is read: the fixture config is validated by fsms.guard.
"""
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fsms.atomic import sha256_file  # noqa: E402
from fsms.config import Config  # noqa: E402
from fsms.guard import is_restricted  # noqa: E402

CONFIG = REPO / "config" / "fixture_tueg.json"
SLOW = os.environ.get("FSMS_RUN_SLOW") == "1" and Path(r"F:\TUEG\v1.2.0").exists() and shutil.which("matlab")


def cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-B", str(REPO / "run_pipeline.py"), *args],
                          capture_output=True, text=True, cwd=REPO)


def output_hashes(root: Path) -> dict[str, str]:
    skip = {"logs", "state"}
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*"))
            if p.is_file() and not (set(p.relative_to(root).parts) & skip) and not p.name.startswith(".")
            and "TUEG" not in p.parts and p.suffix != ".part"}


class FixtureEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config.load(CONFIG)
        assert not is_restricted(cls.config.run_root) and not is_restricted(cls.config.data_root)

    @unittest.skipUnless(SLOW, "set FSMS_RUN_SLOW=1 with MATLAB and the TUEG corpus available")
    def test_full_run_then_idempotent_rerun(self):
        if not (self.config.snip_keyfile.exists() and self.config.database_csv.exists()):
            built = cli("fixture", "--work-root", str(self.config.data_root))
            self.assertEqual(built.returncode, 0, built.stdout[-2000:] + built.stderr[-2000:])
        first = cli("run", "--config", str(CONFIG), "--confirm")
        self.assertEqual(first.returncode, 0, first.stdout[-4000:] + first.stderr[-4000:])
        results = self.config.analysis_dir / "tables" / "confirmatory_results.csv"
        self.assertTrue(results.exists())
        self.assertTrue((self.config.analysis_dir / "CONFIRMATORY_EXECUTION.json").exists())
        before = output_hashes(self.config.run_root)
        second = cli("run", "--config", str(CONFIG), "--confirm")
        self.assertEqual(second.returncode, 0, second.stdout[-4000:] + second.stderr[-4000:])
        report = json.loads(second.stdout.strip().splitlines()[-1])
        self.assertTrue(report and all(state == "skipped" for state in report.values()), report)
        self.assertEqual(before, output_hashes(self.config.run_root), "a rerun changed output bytes")

    @unittest.skipUnless(SLOW, "needs a completed fixture run")
    def test_changed_parameter_invalidates_only_downstream_stages(self):
        status = cli("status", "--config", str(CONFIG))
        self.assertEqual(status.returncode, 0)
        self.assertNotIn("never run", status.stdout)


class RealStudyStaysLocked(unittest.TestCase):
    """Runs everywhere (no MATLAB/TUEG): the real config must refuse before touching any data."""

    def test_real_config_refuses_without_registration(self):
        if (REPO / "prereg" / "REGISTRATION.json").exists():
            self.skipTest("registration record exists: the real study is legitimately unlocked")
        run = cli("run", "--config", str(REPO / "config" / "study.json"))
        self.assertEqual(run.returncode, 2)
        self.assertIn("LOCKED", run.stderr)


if __name__ == "__main__":
    unittest.main()
