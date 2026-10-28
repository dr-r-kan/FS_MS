"""Idempotent HEPPy stage: clips -> R-peak fit + EEG preprocessing + R-peak recovery -> QC/exclusions.

    python -m fsms.heppy.run_heppy_stage --clips-dir CLIPS --heppy-out OUT --options OPTIONS.json
        [--workers N] [--limit N] [--only BASE ...] [--adopt-dir LEGACY_HEPPY_OUTPUT]
        [--retry-failed] [--force] [--verify-hash] [--adopt-hash] [--mode fixture]

Per clip the stage fits ECG R-peaks, preprocesses the EEG (both cardiac variants) and recovers
missed R-peaks (legacy algorithms; see ``docs/STAGE_B_NOTES.md``), guarded by per-step
fingerprint sentinels in ``<heppy-out>/.sentinels``.  A pre-specified, group-blind rule
(``qc_rule`` v2 in the options JSON) is then applied to every recording's QC record, yielding
``heppy_exclusions.csv`` (base, stage, reason_code, evidence, scope = recording|segment, plus the
segment identifiers of segment-scope rows), ``heppy_summary.json`` (CONSORT counts at recording and
segment level), ``heppy_qc/<base>.json`` and a legacy-schema ``ecg_qc_review.csv``.  Recording scope:
too few EEG channels, no ECG, no valid ECG window, exception, crash, missing outputs.  Segment scope
(II and preictal rows of the snip keyfile only): an overlapping PyPREP chunk skipped/failed or
interpolated > 30 % of its channels.

Exit status: 0 whenever the run completed - exclusions are a normal outcome; 2 only for
infrastructure errors (unreadable options, missing clip EDF, lock held, HEPPy missing, ...).
The stage never sees diagnostic group labels: it reads the clip EDFs, ``snip_keyfile.csv`` (segment
times only) and the options JSON.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, deque
from pathlib import Path

from fsms.atomic import atomic_write_bytes, atomic_write_json, fingerprint
from fsms.clips.modes import MODES, enforce_mode
from fsms.guard import RestrictedDataError
from fsms.heppy import options as opt
from fsms.heppy import qc as qcmod
from fsms.heppy import sentinel as sn
from fsms.heppy import worker as wk

REVIEW_COLS = ["base", "edf_path", "raw_fif", "raw_fif_keep", "qc_png", "viable",
               "true_rr_s", "true_hr_bpm", "beats_per_gap", "notes", "flip"]
EXCLUSION_COLS = ["base", "stage", "reason_code", "evidence", "scope", "segment", "patient_code", "source_row",
                  "event_type", "event_number", "segment_number", "output_start_seconds", "output_end_seconds"]
SEGMENT_ID_COLS = ("segment", "patient_code", "source_row", "event_type", "event_number", "segment_number",
                   "output_start_seconds", "output_end_seconds")
_PART = re.compile(r"(\.part-\d+|\.part$)")


class InfrastructureError(RuntimeError):
    """Run-level failure (not a recording-level exclusion): exit status 2."""


# ----------------------------------------------------------------------------- small utilities

def csv_bytes(rows: list[dict], fieldnames: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_if_changed(path: Path, data: bytes) -> bool:
    """Atomic write that leaves byte-identical files untouched; returns True when written."""
    try:
        if Path(path).read_bytes() == data:
            return False
    except OSError:
        pass
    atomic_write_bytes(path, data)
    return True


def write_json_if_changed(path: Path, value) -> bool:
    text = json.dumps(wk.jsonable(value), indent=2, sort_keys=True) + "\n"
    return write_if_changed(path, text.encode("utf-8"))


def clean_stale_parts(root: Path) -> int:
    removed = 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        if path.is_file() and _PART.search(path.name):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return removed


class RunLock:
    """Exclusive per-output-directory lock so two runs cannot interleave sentinel writes."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.held = False

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._stale():
                    self.path.unlink(missing_ok=True)
                    continue
                raise InfrastructureError(f"another run holds {self.path} (delete it only if no run is active)")
            os.write(handle, str(os.getpid()).encode())
            os.close(handle)
            self.held = True
            return self
        raise InfrastructureError(f"cannot acquire {self.path}")

    def _stale(self) -> bool:
        try:
            pid = int(self.path.read_text().strip())
        except (OSError, ValueError):
            return True
        try:
            import psutil

            return not psutil.pid_exists(pid)
        except Exception:
            return False

    def __exit__(self, *exc) -> None:
        if self.held:
            self.path.unlink(missing_ok=True)


def read_keyfile(clips_dir: Path) -> dict[str, list[dict]]:
    """Group snip-keyfile rows by output EDF stem (empty dict when no keyfile)."""
    path = clips_dir / "snip_keyfile.csv"
    if not path.exists():
        return {}
    out: dict[str, list[dict]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            stem = Path(str(row.get("output_edf", "")).replace("\\", "/")).stem
            if stem:
                out.setdefault(stem, []).append(row)
    return out


def discover_bases(clips_dir: Path, keyed: dict[str, list[dict]]) -> list[str]:
    stems = {p.stem for p in clips_dir.glob("*.edf") if not p.name.startswith(".")}
    return sorted(stems | set(keyed))


# ----------------------------------------------------------------------------- planning

def adoption_fingerprint(record: dict, options: dict) -> str:
    files = {role: [info["size"], info["mtime_ns"], info.get("sha256")] for role, info in record["files"].items()}
    return fingerprint({"adopted": True, "files": files, "options": opt.options_fingerprint(options),
                        "compat": record["options_compat"].get("legacy_options_fingerprint")})


def adopted_sentinel(base: str, record: dict, options: dict, edf_id: dict | None) -> dict:
    eff = opt.processing_options(options)
    return {
        "schema": sn.SENTINEL_SCHEMA, "base": base, "adopted": True, "adoption": record, "edf": edf_id,
        "options_fingerprint": opt.options_fingerprint(options), "seeds": eff["seeds"],
        "versions": {"note": "legacy outputs adopted in place; package versions of the original run were not recorded"},
        "step_fingerprints": None, "fingerprint": adoption_fingerprint(record, options), "steps": {},
    }


def plan_recording(base: str, ctx: dict) -> dict:
    """Decide what to do for one clip.  Returns {'action', 'todo', 'job', 'sentinel', 'reasons'...}."""
    heppy_out: Path = ctx["heppy_out"]
    options = ctx["options"]
    spath = sn.sentinel_path(heppy_out, base)
    previous = sn.read_sentinel(spath)
    edf = ctx["clips_dir"] / f"{base}.edf"
    adoption = ctx.get("adoption")
    ecg_settings = opt.processing_options(options)["ecg_settings"]
    out = {"base": base, "todo": [], "reasons": {}, "action": None, "note": ""}

    # 1. an existing, still-valid adoption is kept
    if previous and previous.get("adopted"):
        if adoption is not None and adoption.revalidate(previous["adoption"]) is None \
                and previous.get("options_fingerprint") == opt.options_fingerprint(options):
            out["action"] = "adopted_unchanged"
            return out
        previous = None  # no longer valid: fall through to a normal (re)computation or re-adoption

    edf_id = None
    if edf.exists():
        edf_id = sn.edf_identity(edf, (previous or {}).get("edf"), ctx["verify_hash"])

    # 2. a current non-adopted sentinel wins over adoption (never flip a computed recording)
    fps = None
    if edf_id is not None:
        keyfile_fp = sn.keyfile_fingerprint(ctx["keyed"].get(base, []))
        fps = sn.step_fingerprints(edf_id, keyfile_fp, options, ctx["versions"])
        todo, reasons = sn.plan_steps(previous, fps, heppy_out, ctx["retry_failed"], ctx["force"])
        crash = (previous or {}).get("crash")
        if (todo and crash and crash.get("fp") == fps["overall"] and not ctx["retry_failed"] and not ctx["force"]):
            out.update(action="crashed_unchanged", note="worker crashed with unchanged fingerprint (use --retry-failed)")
            return out
        if not todo:
            skeleton = sn.fresh_sentinel(base, edf_id, keyfile_fp, fps, options, ctx["versions"], previous)
            if previous is None or skeleton["edf"] != previous.get("edf"):
                # identity cache refreshed (e.g. touched file, same content): record, do not recompute
                merged = dict(previous or skeleton)
                merged["edf"] = edf_id
                sn.write_sentinel(spath, merged)
            out.update(action="unchanged", reasons=reasons)
            return out
    else:
        todo, reasons, keyfile_fp = None, {}, None

    # 3. try adoption for anything that would otherwise be computed
    if adoption is not None:
        record, why = adoption.try_adopt(base, ecg_settings, ctx["keyed"].get(base, []))
        if record is not None:
            sentinel = adopted_sentinel(base, record, options, edf_id)
            sn.write_sentinel(spath, sentinel)
            out.update(action="adopted")
            return out
        out["adoption_refused"] = why

    if edf_id is None:
        out.update(action="missing_edf", note=f"clip EDF missing: {edf}")
        return out

    skeleton = sn.fresh_sentinel(base, edf_id, keyfile_fp, fps, options, ctx["versions"], previous)
    job = {
        "base": base, "edf_path": str(edf), "heppy_out": str(heppy_out), "options": options,
        "todo": todo, "step_fps": {k: v for k, v in fps.items() if k != "overall"}, "sentinel": skeleton,
        "result_path": str(heppy_out / ".jobs" / f"{base}.result.json"),
    }
    out.update(action="compute", todo=todo, reasons=reasons, job=job)
    return out


# ----------------------------------------------------------------------------- execution

def last_stage(log_text: str) -> str:
    for line in reversed(log_text.splitlines()):
        if line.startswith("HEPPY_STAGE\t"):
            return line.split("\t", 1)[1].strip().split("\t")[0]
    return "unknown"


def describe_exit(code: int) -> str:
    unsigned = code & 0xFFFFFFFF
    names = {0xC0000005: "access_violation", 0xC0000017: "no_memory",
             0xC0000409: "stack_buffer_overrun", 0xC0000374: "heap_corruption"}
    return f"{code}/0x{unsigned:08X}" + (f" ({names[unsigned]})" if unsigned in names else "")


def record_crash(heppy_out: Path, plan: dict, exit_code: int | None, log_path: Path) -> None:
    """A worker died without a result: persist the crash against this fingerprint."""
    base = plan["base"]
    spath = sn.sentinel_path(heppy_out, base)
    current = sn.read_sentinel(spath) or plan["job"]["sentinel"]
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    current["crash"] = {"exit_code": describe_exit(exit_code) if exit_code is not None else "none",
                        "last_stage": last_stage(text), "fp": plan["job"]["sentinel"]["fingerprint"]}
    sn.write_sentinel(spath, current)


def execute(plans: list[dict], ctx: dict, log) -> dict[str, dict]:
    """Run all ``compute`` plans, ``workers`` at a time.  Returns base -> run record."""
    heppy_out: Path = ctx["heppy_out"]
    jobs = [p for p in plans if p["action"] == "compute"]
    records: dict[str, dict] = {}
    if not jobs:
        return records
    (heppy_out / ".jobs").mkdir(parents=True, exist_ok=True)
    (heppy_out / "logs" / "rows").mkdir(parents=True, exist_ok=True)

    if ctx.get("in_process"):
        for index, plan in enumerate(jobs, start=1):
            log(f"[{index}/{len(jobs)}] {plan['base']}: running {plan['todo']} (in-process)")
            started = time.time()
            wk.run_job(plan["job"], ctx.get("impl") or wk.RealImpl)
            records[plan["base"]] = {"steps": plan["todo"], "elapsed_s": round(time.time() - started, 1), "exit": 0}
        return records

    env = wk.pinned_environment(ctx["options"])
    pending = deque(jobs)
    running: dict[str, tuple] = {}
    total = len(jobs)
    done = 0
    workers = max(1, int(ctx["workers"]))
    log(f"Running {total} recording(s) with {workers} worker process(es); per-row logs in {heppy_out / 'logs' / 'rows'}")
    while pending or running:
        while pending and len(running) < workers:
            plan = pending.popleft()
            base = plan["base"]
            job_path = heppy_out / ".jobs" / f"{base}.job.json"
            atomic_write_json(job_path, plan["job"])
            Path(plan["job"]["result_path"]).unlink(missing_ok=True)
            log_path = heppy_out / "logs" / "rows" / f"{base}.log"
            handle = log_path.open("w", encoding="utf-8", errors="replace")
            proc = subprocess.Popen(
                [sys.executable, "-m", "fsms.heppy.worker", str(job_path)], cwd=str(opt.repo_root()),
                env=env, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            running[base] = (proc, handle, time.time(), plan, log_path)
            log(f"  start {base}: {plan['todo']}")
        for base in list(running):
            proc, handle, started, plan, log_path = running[base]
            code = proc.poll()
            if code is None:
                continue
            handle.close()
            del running[base]
            done += 1
            result_path = Path(plan["job"]["result_path"])
            ok = code == 0 and result_path.exists()
            if not ok:
                record_crash(heppy_out, plan, code, log_path)
            records[base] = {"steps": plan["todo"], "elapsed_s": round(time.time() - started, 1),
                             "exit": code, "result": "ok" if ok else "crash"}
            log(f"[{done}/{total}] {base}: {'done' if ok else 'CRASH exit=' + describe_exit(code)} "
                f"in {records[base]['elapsed_s']}s")
            result_path.unlink(missing_ok=True)
            (heppy_out / ".jobs" / f"{base}.job.json").unlink(missing_ok=True)
        if running:
            time.sleep(0.5)
    return records


# ----------------------------------------------------------------------------- reporting

def review_row(base: str, qc: dict | None, sentinel: dict | None, rule: dict, ctx: dict) -> dict:
    heppy_out: Path = ctx["heppy_out"]
    row = {k: "" for k in REVIEW_COLS}
    row.update(base=base, edf_path=str(ctx["clips_dir"] / f"{base}.edf"), flip="False")
    if qc is None:
        row["viable"] = ""
        row["notes"] = "not run"
        return row
    outs = qc.get("outputs") or {}

    def full(rel):
        if not rel:
            return ""
        path = Path(rel)
        return str(path if path.is_absolute() else heppy_out / path)

    row["raw_fif"] = full(outs.get("raw_fif")) if qc.get("outputs_ok") else ""
    row["raw_fif_keep"] = full(outs.get("raw_fif_keep")) if qc.get("outputs_ok") else ""
    row["qc_png"] = full(outs.get("autofit_png"))
    viable, reason = qcmod.ecg_viable(qc, rule)
    row["viable"] = viable
    row["notes"] = ("automatic QRS fit; " + reason) if viable == "yes" else f"no: {reason}"
    return row


def finalize(bases: list[str], ctx: dict, actions: dict[str, str], infra: list[str],
             adoption_refused: dict[str, str], runs: dict[str, dict], log) -> dict:
    """Derive QC JSONs, apply the rule, write exclusions/summary/review CSVs; return the summary."""
    heppy_out: Path = ctx["heppy_out"]
    options = ctx["options"]
    rule = options["qc_rule"]
    qcs: dict[str, dict | None] = {}
    sentinels: dict[str, dict | None] = {}
    for base in bases:
        sentinel = sn.read_sentinel(sn.sentinel_path(heppy_out, base))
        sentinels[base] = sentinel
        if sentinel is None:
            qcs[base] = None
            continue
        qc = qcmod.assemble_qc(base, sentinel)
        qcs[base] = qc
        write_json_if_changed(heppy_out / "heppy_qc" / f"{base}.json", qc)

    exclusion_rows: list[dict] = []
    evaluations: dict[str, dict] = {}
    segment_evals: dict[str, dict] = {}
    order = {code: i for i, code in enumerate(rule.get("reason_precedence", []))}
    for base in bases:
        qc = qcs[base]
        if qc is None:
            continue
        evaluation = qcmod.evaluate_rule(qc, rule)
        evaluations[base] = evaluation
        for reason in evaluation["reasons"]:
            exclusion_rows.append({"base": base, "scope": "recording", **reason})
        if not evaluation["excluded"]:   # a recording-level exclusion already covers every segment
            seg = qcmod.evaluate_segments(qc, rule, ctx["keyed"].get(base, []))
            segment_evals[base] = seg
            for result in seg["results"]:
                for reason in result["reasons"]:
                    exclusion_rows.append({"base": base, "scope": "segment", **reason,
                                           **{k: result["segment"].get(k, "") for k in SEGMENT_ID_COLS}})
    exclusion_rows.sort(key=lambda r: (r["base"], r["scope"] != "recording", order.get(r["reason_code"], 99),
                                       str(r.get("output_start_seconds", "")), r["stage"]))
    write_if_changed(heppy_out / "heppy_exclusions.csv", csv_bytes(exclusion_rows, EXCLUSION_COLS))
    review = [review_row(b, qcs[b], sentinels[b], rule, ctx) for b in bases]
    write_if_changed(heppy_out / "ecg_qc_review.csv", csv_bytes(review, REVIEW_COLS))

    # legacy-shaped aggregates so legacy readers keep working
    summary_rows = []
    for b in bases:
        qc = qcs[b]
        if qc is None:
            continue
        status = "ok" if qc.get("outputs_ok") and not qc.get("errors") else (
            "skipped_no_eeg" if any("No EEG channels remain" in e.get("message", "") for e in qc.get("errors", []))
            else "error")
        outs = qc.get("outputs") or {}
        summary_rows.append({
            "base": b, "edf_path": str(ctx["clips_dir"] / f"{b}.edf"), "status": status,
            "error": "; ".join(f"{e['step']}:{e.get('type', '')}" for e in qc.get("errors", [])),
            "raw_fif": review_row(b, qc, None, rule, ctx)["raw_fif"],
            "raw_fif_keep": review_row(b, qc, None, rule, ctx)["raw_fif_keep"]})
    write_if_changed(heppy_out / "logs" / "headless_preprocessing_summary.csv",
                     csv_bytes(summary_rows, ["base", "edf_path", "status", "error", "raw_fif", "raw_fif_keep"]))
    auto_rows, recovery_rows = [], []
    for b in bases:
        qc = qcs[b]
        if qc is None or qc.get("adopted"):
            continue
        ecg = qc.get("ecg") or {}
        if ecg.get("autofit"):
            auto_rows.append({"base": b, **ecg["autofit"]})
        if ecg.get("recovery"):
            recovery_rows.append({"base": b, **ecg["recovery"]})
    if auto_rows:
        fields = list(dict.fromkeys(k for r in auto_rows for k in r))
        write_if_changed(heppy_out / "ecg_qc" / "auto_qrs_report.csv", csv_bytes(auto_rows, fields))
    if recovery_rows:
        flat = [{k: (json.dumps(v, sort_keys=True) if isinstance(v, dict) else v) for k, v in r.items()}
                for r in recovery_rows]
        fields = list(dict.fromkeys(k for r in flat for k in r))
        write_if_changed(heppy_out / "ecg_qc" / "missed_rpeak_recovery_report.csv", csv_bytes(flat, fields))
    # legacy tools (fast_ecg_review, hep.py) look for heppy_output/options_used.json
    write_json_if_changed(heppy_out / "options_used.json", options)

    assessed = [b for b in bases if qcs[b] is not None]
    excluded = [b for b in assessed if evaluations[b]["excluded"]]
    primary = Counter(evaluations[b]["primary_reason"] for b in excluded)
    any_reason = Counter(r["reason_code"] for b in excluded for r in evaluations[b]["reasons"])
    adopted = sorted(b for b in assessed if qcs[b].get("adopted"))
    flags: dict[str, list[str]] = {}
    gaps: dict[str, list[str]] = {}
    for b in assessed:
        for flag in evaluations[b]["flags"]:
            flags.setdefault(flag, []).append(b)
        for gap in evaluations[b]["not_evaluable"] + segment_evals.get(b, {}).get("not_evaluable", []):
            gaps.setdefault(gap, []).append(b)
    scope_names = list((rule.get("segment") or {}).get("scope_segments", []))
    in_scope_total = {name: 0 for name in scope_names}
    usable = {name: 0 for name in scope_names}
    seg_reason_counts: Counter = Counter()
    per_recording: dict[str, dict] = {}
    for b in assessed:
        rows = [r for r in ctx["keyed"].get(b, []) if str(r.get("segment", "")).strip() in scope_names]
        counts = {name: {"in_scope": 0, "usable": 0} for name in scope_names}
        for r in rows:
            counts[str(r["segment"]).strip()]["in_scope"] += 1
        if evaluations[b]["excluded"]:
            for name in scope_names:
                in_scope_total[name] += counts[name]["in_scope"]
        else:
            for result in segment_evals[b]["results"]:
                name = str(result["segment"]["segment"]).strip()
                in_scope_total[name] += 1
                if result["usable"]:
                    counts[name]["usable"] += 1
                    usable[name] += 1
                for reason in result["reasons"]:
                    seg_reason_counts[reason["reason_code"]] += 1
        per_recording[b] = counts
    both = [b for b in assessed if not evaluations[b]["excluded"]
            and all(per_recording[b][n]["usable"] >= 1 for n in scope_names)]
    current_fp = opt.options_fingerprint(options)
    stale = sorted(b for b in assessed if not qcs[b].get("adopted") and qcs[b].get("options_fingerprint") != current_fp)
    summary = {
        "schema": "heppy_summary/1",
        "options_fingerprint": current_fp,
        "recordings_computed_under_other_options": stale,
        "qc_rule": rule,
        "qc_rule_fingerprint": opt.qc_rule_fingerprint(options),
        "versions": ctx["versions"],
        "consort": {
            "clips_discovered": len(bases),
            "not_run": len(bases) - len(assessed),
            "assessed": len(assessed),
            "excluded": len(excluded),
            "included": len(assessed) - len(excluded),
            "excluded_by_primary_reason": dict(sorted(primary.items())),
            "excluded_by_any_reason": dict(sorted(any_reason.items())),
            "recording_level": {
                "assessed": len(assessed), "excluded": len(excluded), "included": len(assessed) - len(excluded),
                "excluded_by_primary_reason": dict(sorted(primary.items())),
                "excluded_by_any_reason": dict(sorted(any_reason.items())),
            },
            "segment_level": {
                "scope_segments": scope_names,
                "in_scope_segments": dict(in_scope_total),
                "excluded_with_recording": {n: sum(per_recording[b][n]["in_scope"] for b in excluded) for n in scope_names},
                "excluded_by_segment_rule": dict(sorted(seg_reason_counts.items())),
                "usable_in_scope_segments": dict(usable),
                "recordings_with_a_usable_segment_of_each_scope_kind": len(both),
            },
        },
        "flags": {k: sorted(v) for k, v in sorted(flags.items())},
        "not_evaluable": {k: sorted(v) for k, v in sorted(gaps.items())},
        "adoption": {
            "adopt_dir": str(ctx["adopt_dir"]) if ctx.get("adopt_dir") else None,
            "options_compat": ctx["adoption"].compat if ctx.get("adoption") else None,
            "n_adopted": len(adopted),
            "adopted": adopted,
            "not_adopted": dict(sorted(adoption_refused.items())),
        },
        "infrastructure_errors": sorted(infra),
        "recordings": {
            b: {"assessed": qcs[b] is not None,
                "included": (qcs[b] is not None and not evaluations[b]["excluded"]),
                "adopted": bool(qcs[b] and qcs[b].get("adopted")),
                "reasons": [r["reason_code"] for r in evaluations[b]["reasons"]] if qcs[b] is not None else [],
                "pyprep_status": (qcs[b] or {}).get("pyprep", {}).get("status") if qcs[b] else None,
                "n_eeg_after_montage": (qcs[b] or {}).get("n_eeg_after_montage") if qcs[b] else None,
                "ecg_viable": qcmod.ecg_viable(qcs[b], rule)[0] if qcs[b] else None,
                "in_scope_segments": per_recording.get(b),
                "fingerprint": (qcs[b] or {}).get("fingerprint") if qcs[b] else None}
            for b in bases},
    }
    write_json_if_changed(heppy_out / "heppy_summary.json", summary)
    return summary


# ----------------------------------------------------------------------------- entry points

def run_stage(clips_dir, heppy_out, options_path, workers: int = 1, limit: int | None = None,
              only: list[str] | None = None, adopt_dir=None, retry_failed: bool = False, force: bool = False,
              verify_hash: bool = False, adopt_hash: bool = False, mode: str = "fixture",
              real_data_verifier=None, in_process: bool = False, impl=None, log=print) -> dict:
    """Run the stage; returns {'exit_code', 'summary', 'actions', ...}.  Raises nothing for exclusions."""
    clips_dir, heppy_out = Path(clips_dir).resolve(), Path(heppy_out).resolve()
    adopt = Path(adopt_dir).resolve() if adopt_dir else None
    enforce_mode(mode, {"clips directory (input)": clips_dir, "HEPPy output directory (output)": heppy_out,
                        "options file (input)": options_path, "adopt directory (read-only input)": adopt},
                 real_data_verifier)
    infra: list[str] = []
    if adopt is not None:
        for a, b in ((adopt, heppy_out), (heppy_out, adopt)):
            if a == b or b in a.parents:
                raise InfrastructureError("--adopt-dir must be outside --heppy-out (adoption never writes there)")
    options = opt.load_options(options_path)
    problems = opt.validate_options(options)
    if problems:
        raise InfrastructureError("options not acceptable: " + "; ".join(problems))
    if not clips_dir.is_dir():
        raise InfrastructureError(f"clips directory not found: {clips_dir}")
    opt.ensure_heppy_on_path()
    started = time.time()
    heppy_out.mkdir(parents=True, exist_ok=True)

    with RunLock(heppy_out / ".stage.lock"):
        removed = clean_stale_parts(heppy_out)
        if removed:
            log(f"Removed {removed} stale temporary file(s) from interrupted runs")
        keyed = read_keyfile(clips_dir)
        bases = discover_bases(clips_dir, keyed)
        selected = bases
        if only:
            wanted = {Path(item).stem for item in only}
            selected = [b for b in bases if b in wanted]
            missing = sorted(wanted - set(bases))
            if missing:
                infra.append("--only names not found: " + ", ".join(missing))
        if limit:
            selected = selected[: int(limit)]
        versions = opt.package_versions()
        ctx = {"clips_dir": clips_dir, "heppy_out": heppy_out, "options": options, "keyed": keyed,
               "versions": versions, "verify_hash": verify_hash, "retry_failed": retry_failed, "force": force,
               "workers": workers, "adopt_dir": adopt, "in_process": in_process, "impl": impl,
               "adoption": None}
        if adopt is not None:
            from fsms.heppy.adopt import AdoptionContext

            ctx["adoption"] = AdoptionContext(adopt, options, hash_outputs=adopt_hash)
            log(f"Adoption: legacy options {'COMPATIBLE' if ctx['adoption'].compat['compatible'] else 'NOT compatible'}"
                f" ({ctx['adoption'].compat['reason']})")

        log(f"{len(bases)} clip(s) discovered, {len(selected)} selected")
        plans = []
        for index, base in enumerate(selected, start=1):
            plan = plan_recording(base, ctx)
            plans.append(plan)
            detail = ""
            if plan["action"] == "compute":
                detail = " steps=" + ",".join(plan["todo"]) + " (" + "; ".join(
                    f"{s}: {plan['reasons'][s]}" for s in plan["todo"]) + ")"
            elif plan.get("note"):
                detail = " " + plan["note"]
            log(f"[plan {index}/{len(selected)}] {base}: {plan['action']}{detail}")
            if plan["action"] == "missing_edf":
                infra.append(plan["note"])
        runs = execute(plans, ctx, log)
        actions = {p["base"]: p["action"] for p in plans}
        refused = {p["base"]: p["adoption_refused"] for p in plans if p.get("adoption_refused")}
        summary = finalize(bases, ctx, actions, infra, refused, runs, log)

        run_record = {
            "started_unix": started, "elapsed_s": round(time.time() - started, 1), "workers": workers,
            "actions": actions, "runs": runs,
            "stage_reasons": {p["base"]: p["reasons"] for p in plans if p["action"] == "compute"},
        }
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(started))
        atomic_write_json(heppy_out / "logs" / f"run_{stamp}.json", wk.jsonable(run_record))
        recomputed = sorted(b for b, a in actions.items() if a == "compute")
        log(f"Stage actions: recomputed={recomputed}; "
            f"unchanged={sorted(b for b, a in actions.items() if a in ('unchanged', 'adopted_unchanged'))}; "
            f"adopted={sorted(b for b, a in actions.items() if a == 'adopted')}")
        consort = summary["consort"]
        segment_level = consort["segment_level"]
        log(f"CONSORT recording-level: discovered={consort['clips_discovered']} assessed={consort['assessed']} "
            f"excluded={consort['excluded']} included={consort['included']} "
            f"reasons={consort['excluded_by_primary_reason']}")
        log(f"CONSORT segment-level: in_scope={segment_level['in_scope_segments']} usable={segment_level['usable_in_scope_segments']} "
            f"excluded_with_recording={segment_level['excluded_with_recording']} "
            f"by_segment_rule={segment_level['excluded_by_segment_rule']}")
    return {"exit_code": 2 if infra else 0, "summary": summary, "actions": actions, "runs": runs,
            "infrastructure_errors": infra}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips-dir", type=Path, required=True, help="Directory with <patient>_snips.edf + snip_keyfile.csv")
    parser.add_argument("--heppy-out", type=Path, required=True, help="Output directory (heppy_output layout)")
    parser.add_argument("--options", type=Path, required=True, help="Locked options JSON (config/heppy_options*.json)")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent recording processes")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N clips (sorted)")
    parser.add_argument("--only", nargs="+", default=None, metavar="BASE", help="Process only these clip bases")
    parser.add_argument("--adopt-dir", type=Path, default=None,
                        help="Legacy heppy_output tree to adopt in place (read-only; see docs/STAGE_B_NOTES.md)")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed/crashed recordings with unchanged fingerprints")
    parser.add_argument("--force", action="store_true", help="Recompute every step of every selected clip")
    parser.add_argument("--verify-hash", action="store_true", help="Re-hash clip EDFs even if size+mtime match the sentinel")
    parser.add_argument("--adopt-hash", action="store_true", help="Also record sha256 of adopted legacy outputs")
    parser.add_argument("--mode", choices=MODES, default="fixture", help="fixture (default): refuse restricted paths")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_stage(args.clips_dir, args.heppy_out, args.options, workers=args.workers, limit=args.limit,
                           only=args.only, adopt_dir=args.adopt_dir, retry_failed=args.retry_failed,
                           force=args.force, verify_hash=args.verify_hash, adopt_hash=args.adopt_hash,
                           mode=args.mode, log=lambda m: print(m, flush=True))
    except (InfrastructureError, RestrictedDataError) as exc:
        print(f"INFRASTRUCTURE ERROR: {exc}", file=sys.stderr)
        return 2
    for message in result["infrastructure_errors"]:
        print(f"INFRASTRUCTURE ERROR: {message}", file=sys.stderr)
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
