"""Single entry point for the FS_MS preregistered pipeline.

    python run_pipeline.py preflight [--config CFG] [--deep]   environment/lock checks, reads no study data
    python run_pipeline.py plan-lock                           hash code+config+plan (no data) -> prereg/PLAN_LOCK.json
    python run_pipeline.py verify-lock                         verify the working tree against the lock
    python run_pipeline.py register --url https://osf.io/...   record the immutable registration; unlocks real data
    python run_pipeline.py run --config CFG [--through STAGE] [--confirm]
    python run_pipeline.py status --config CFG
    python run_pipeline.py fixture [--work-root DIR]           build the synthetic TUEG test study

Always run with the project's virtual environment (``run_pipeline.ps1`` does this).  Reruns are idempotent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fsms import lock as planlock  # noqa: E402
from fsms import pipeline  # noqa: E402
from fsms.config import Config, ConfigError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight"); p.add_argument("--config"); p.add_argument("--deep", action="store_true")
    sub.add_parser("plan-lock")
    sub.add_parser("verify-lock")
    p = sub.add_parser("register"); p.add_argument("--url", required=True)
    p = sub.add_parser("run"); p.add_argument("--config", required=True)
    p.add_argument("--through", default="stats_verify", choices=pipeline.STAGE_ORDER)
    p.add_argument("--confirm", action="store_true",
                   help="also run the registered confirmatory tests (real data: only after registration)")
    p.add_argument("--force", choices=pipeline.STAGE_ORDER, help="rerun this stage even if current")
    p = sub.add_parser("status"); p.add_argument("--config", required=True)
    p = sub.add_parser("fixture"); p.add_argument("--work-root", default="F:/FS_MS_fixture_tueg")
    args = parser.parse_args(argv)

    try:
        if args.command == "preflight":
            config = Config.load(args.config) if args.config else None
            checks = pipeline.preflight(config, deep=args.deep)
            for name, ok, detail in checks:
                print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")
            critical = {"python", "pinned packages", "vendored externals", "matlab on PATH"}
            return 0 if all(ok for name, ok, _ in checks if name in critical) else 1
        if args.command == "plan-lock":
            lock = planlock.create_lock()
            print(f"locked {len(lock['files'])} files; git head {lock['git']['head'][:12]}; wrote {planlock.LOCK_PATH}")
            if lock["git"]["uncommitted_paths_excluding_prereg"]:
                print("WARNING: uncommitted changes exist; commit them, then re-run plan-lock before registering.")
            return 0
        if args.command == "verify-lock":
            problems = planlock.verify_lock()
            for problem in problems:
                print("LOCK PROBLEM:", problem, file=sys.stderr)
            print("plan lock verified" if not problems else "plan lock FAILED")
            return 1 if problems else 0
        if args.command == "register":
            print(json.dumps(planlock.register(args.url), indent=2))
            return 0
        if args.command == "status":
            for name, state in pipeline.status(Config.load(args.config)).items():
                print(f"{name:14s} {state}")
            return 0
        if args.command == "fixture":
            from fsms.pipeline import run_logged, PYTHON
            return run_logged([PYTHON, "-B", "-m", "fsms.fixture.make_tueg_fixture", "--work-root", args.work_root],
                              Path(args.work_root) / "fixture_build.log", Path(__file__).resolve().parent)
        report = pipeline.run(Config.load(args.config), through=args.through, confirm=args.confirm, force=args.force)
        print(json.dumps(report))
        return 0
    except (pipeline.PipelineError, planlock.LockError, ConfigError, PermissionError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
