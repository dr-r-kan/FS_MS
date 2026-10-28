"""Idempotency/invalidation/recovery semantics of the stage runner, using stub stages (no data, no MATLAB)."""
import collections
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fsms import pipeline  # noqa: E402
from fsms.atomic import fingerprint  # noqa: E402
from fsms.config import Config  # noqa: E402


class StubChain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        path = root / "cfg.json"
        path.write_text(json.dumps({"schema": "fsms-config/v1", "name": "stub", "kind": "fixture",
                                    "paths": {"data_root": str(root / "data"), "run_root": str(root / "run")}}))
        self.config = Config.load(path)
        self.runs = collections.Counter()
        self.params = {"a": 1, "b": 1, "c": 1}
        self.fail = set()
        order = ("a", "b", "c")

        def make(name, upstream):
            def fp(ctx):
                return fingerprint({"n": name, "p": self.params[name],
                                    "up": [ctx.fingerprints[u] for u in upstream]})

            def run(ctx):
                if name in self.fail:
                    raise pipeline.PipelineError(f"stub {name} failed")
                self.runs[name] += 1
                out = ctx.config.run_root / f"{name}.txt"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(f"{name}:{self.params[name]}")  # deterministic content

            return pipeline.Stage(name, fp, run, lambda ctx, n=name: [ctx.config.run_root / f"{n}.txt"])

        stages = {"a": make("a", []), "b": make("b", ["a"]), "c": make("c", ["b"])}
        patches = [mock.patch.object(pipeline, "STAGES", stages), mock.patch.object(pipeline, "STAGE_ORDER", order)]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.addCleanup(self.tmp.cleanup)

    def go(self, **kwargs):
        return pipeline.run(self.config, through="c", **kwargs)

    def test_second_run_executes_nothing(self):
        self.assertEqual(self.go(), {"a": "ran", "b": "ran", "c": "ran"})
        self.assertEqual(self.go(), {"a": "skipped", "b": "skipped", "c": "skipped"})
        self.assertEqual(dict(self.runs), {"a": 1, "b": 1, "c": 1})

    def test_changed_upstream_parameter_reruns_it_and_everything_after(self):
        self.go()
        self.params["a"] = 2
        self.assertEqual(self.go(), {"a": "ran", "b": "ran", "c": "ran"})

    def test_changed_downstream_parameter_reruns_only_that_stage(self):
        self.go()
        self.params["c"] = 2
        self.assertEqual(self.go(), {"a": "skipped", "b": "skipped", "c": "ran"})

    def test_deleted_or_corrupted_output_heals_on_next_run(self):
        self.go()
        (self.config.run_root / "b.txt").unlink()
        report = self.go()
        self.assertEqual((report["a"], report["b"]), ("skipped", "ran"))
        (self.config.run_root / "a.txt").write_text("tampered")
        self.assertEqual(self.go()["a"], "ran")
        self.assertEqual((self.config.run_root / "a.txt").read_text(), "a:1")

    def test_failure_leaves_no_completion_record_and_resumes_at_failed_stage(self):
        self.fail.add("b")
        with self.assertRaises(pipeline.PipelineError):
            self.go()
        ledger_dir = self.config.state_dir
        self.assertTrue((ledger_dir / "a.done.json").exists())
        self.assertFalse((ledger_dir / "b.done.json").exists())
        self.fail.clear()
        self.assertEqual(self.go(), {"a": "skipped", "b": "ran", "c": "ran"})

    def test_force_reruns_exactly_one_stage(self):
        self.go()
        self.assertEqual(self.go(force="b"), {"a": "skipped", "b": "ran", "c": "skipped"})

    def test_lock_released_after_failure(self):
        self.fail.add("a")
        with self.assertRaises(pipeline.PipelineError):
            self.go()
        self.assertFalse((self.config.state_dir / ".run.lock").exists())

    def test_unknown_stage_rejected(self):
        with self.assertRaises(pipeline.PipelineError):
            pipeline.run(self.config, through="nope")


if __name__ == "__main__":
    unittest.main()
