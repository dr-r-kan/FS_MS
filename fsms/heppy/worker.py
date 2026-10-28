"""Per-recording worker: runs the needed steps for ONE clip in its own process.

    python -m fsms.heppy.worker JOB.json

Steps (legacy order and algorithms): ``ecg_fit`` (automatic R-peak fit on the clip EDF) ->
``preprocess`` (HEPPy ``_both``: cardiac-removed and cardiac-kept FIFs) -> ``recovery`` (missed
R-peak recovery on the preprocessed FIF's ECG channel, as the legacy stage 5 did).  After every
step the sentinel is rewritten atomically, so a kill between steps loses only the running step.
Exceptions are caught and stored in the sentinel; they never abort other recordings.

The worker is process-isolated for the same reason as in the legacy driver: native crashes
(Windows access violations in ICA/ONNX) must not take the whole run down.
"""
from __future__ import annotations

import faulthandler
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

from fsms.atomic import atomic_write_json
from fsms.heppy import options as opt
from fsms.heppy import sentinel as sn

THREAD_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TBB_NUM_THREADS", "NUMBA_NUM_THREADS",
)


def pinned_environment(options: dict, base_env: dict | None = None) -> dict:
    """Environment for a deterministic worker: threads pinned, hash seed fixed, no GUI/home writes."""
    env = dict(os.environ if base_env is None else base_env)
    seeds = options.get("seeds") or {}
    threads = str(int(seeds.get("native_threads", 1)))
    for var in THREAD_VARS:
        env[var] = threads
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    env["PYTHONHASHSEED"] = str(seeds.get("pythonhashseed", "0"))
    env["HEPPY_NATIVE_THREADS"] = threads
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUTF8"] = "1"
    env.setdefault("MNE_DONTWRITE_HOME", "true")
    temp = Path(tempfile.gettempdir())
    env.setdefault("_MNE_FAKE_HOME_DIR", str(temp / "mne_home"))
    env.setdefault("MPLCONFIGDIR", str(temp / "mpl_cache"))
    env.setdefault("NUMBA_CACHE_DIR", str(temp / "numba_cache"))
    root = str(opt.repo_root())
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def jsonable(value):
    """Plain-Python, strict-JSON copy of ``value`` (numpy scalars cast, non-finite floats -> None)."""
    import math

    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if np is not None and isinstance(value, np.generic):
        value = value.item()
    if np is not None and isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def paths(job: dict) -> dict[str, Path]:
    heppy_out = Path(job["heppy_out"])
    base = job["base"]
    qc_dir = heppy_out / "ecg_qc"
    raw_dir = heppy_out / "raw_fif"
    return {
        "autofit": qc_dir / f"{base}_rpeaks_autofit.csv",
        "corrected": qc_dir / f"{base}_rpeaks_corrected.csv",
        "autofit_png": qc_dir / "auto_qrs_plots" / f"{base}_auto_qrs_qc.png",
        "recovery_png": qc_dir / "recovery_qc_plots" / f"{base}_recovered_qrs_qc.png",
        "raw_fif": raw_dir / f"{base}_pp_raw.fif",
        "raw_fif_keep": raw_dir / f"{base}_pp_raw_keepcfa.fif",
        "ica_remove": raw_dir / f"{base}_ica.fif",
        "ica_keep": raw_dir / f"{base}_keepcfa_ica.fif",
    }


class RealImpl:
    """The real step implementations (tests substitute stubs with the same three methods)."""

    @staticmethod
    def run(step: str, job: dict, sentinel: dict) -> dict:
        return getattr(RealImpl, step)(job, sentinel)

    @staticmethod
    def ecg_fit(job: dict, sentinel: dict) -> dict:
        opt.ensure_heppy_on_path()
        from fsms.heppy import ecg_qrs

        eff = opt.processing_options(job["options"])
        p = paths(job)
        heppy_out = Path(job["heppy_out"])
        for key in ("autofit", "corrected", "autofit_png", "recovery_png"):
            p[key].unlink(missing_ok=True)
        try:
            ecg = ecg_qrs.load_ecg(Path(job["edf_path"]), eff["ecg_channel"],
                                   eff["filters"]["hp"], eff["filters"]["lp"])
        except ecg_qrs.NoEcgChannel as exc:
            return {"status": "no_ecg_channel", "qc": {"message": str(exc)}, "outputs": []}
        a = eff["auto_qrs"]
        settings = dict(eff["ecg_settings"], channel=eff["ecg_channel"])
        rpeaks, stats = ecg_qrs.auto_fit(ecg["signal"], ecg["sfreq"], settings, a["alpha"], a["iterations"],
                                         a["min_corr"], a["corr_search_s"])
        ecg_qrs.save_corrected_atomic(p["autofit"], rpeaks, ecg["sfreq"])
        qc = dict(stats, channel=ecg["channel"], sfreq=ecg["sfreq"], n_samples=len(ecg["signal"]),
                  source=ecg["source"],
                  rpeak_score=ecg_qrs.rpeak_qc_score(rpeaks, ecg["sfreq"], len(ecg["signal"]), settings))
        outputs = [sn.output_item("autofit_sidecar", p["autofit"], heppy_out, "sidecar")]
        try:
            ecg_qrs.save_auto_qc_plot(
                p["autofit_png"], ecg["signal"], ecg["sfreq"], rpeaks, a["alpha"],
                f"{job['base']} | final peaks={len(rpeaks)} | dropped={stats['dropped_after_refit']} "
                f"({stats['drop_percent']:.2f}%)")
            outputs.append(sn.output_item("autofit_png", p["autofit_png"], heppy_out, "file"))
        except Exception as exc:  # a missing PNG must not fail the recording
            qc["plot_error"] = f"{type(exc).__name__}: {exc}"
        return {"status": "ok", "qc": qc, "outputs": outputs}

    @staticmethod
    def preprocess(job: dict, sentinel: dict) -> dict:
        from fsms.heppy import preprocess

        heppy_out = Path(job["heppy_out"])
        eff = opt.processing_options(job["options"])
        fragment = preprocess.run_preprocess(job["options"], job["base"], Path(job["edf_path"]), heppy_out,
                                             eff["seeds"]["heppy_random_state"])
        p = paths(job)
        outputs = [sn.output_item("raw_fif", p["raw_fif"], heppy_out, "fif"),
                   sn.output_item("raw_fif_keep", p["raw_fif_keep"], heppy_out, "fif")]
        for role in ("ica_remove", "ica_keep"):
            if p[role].exists():
                outputs.append(sn.output_item(role, p[role], heppy_out, "file"))
        return {"status": "ok", "qc": fragment, "outputs": outputs}

    @staticmethod
    def recovery(job: dict, sentinel: dict) -> dict:
        steps = sentinel.get("steps") or {}
        ecg_status = (steps.get("ecg_fit") or {}).get("status")
        if ecg_status == "no_ecg_channel":
            return {"status": "skipped_no_ecg", "qc": {}, "outputs": []}
        if ecg_status != "ok":
            return {"status": "skipped_ecg_fit_failed", "qc": {}, "outputs": []}
        if (steps.get("preprocess") or {}).get("status") != "ok":
            return {"status": "skipped_no_fif", "qc": {}, "outputs": []}
        opt.ensure_heppy_on_path()
        from fsms.heppy import ecg_qrs

        eff = opt.processing_options(job["options"])
        p = paths(job)
        heppy_out = Path(job["heppy_out"])
        for key in ("corrected", "recovery_png"):
            p[key].unlink(missing_ok=True)
        ecg = ecg_qrs.load_ecg(p["raw_fif"], eff["ecg_channel"], eff["filters"]["hp"], eff["filters"]["lp"])
        settings = dict(eff["ecg_settings"], channel=eff["ecg_channel"])
        import numpy as np

        saved = ecg_qrs.read_sidecar_peaks(p["autofit"], ecg["sfreq"], len(ecg["signal"]))
        saved = np.asarray([] if saved is None else saved, dtype=int)
        result = ecg_qrs.recover(ecg["signal"], ecg["sfreq"], saved, settings, eff["rpeak_recovery"],
                                 eff["auto_qrs"]["alpha"], eff["auto_qrs"]["iterations"])
        qc = {
            "channel": ecg["channel"], "sfreq": ecg["sfreq"], "n_samples": len(ecg["signal"]),
            **result["stats"],
            "saved_before": int(len(saved)), "full_record_detected": int(len(result["detected"])),
            "final": int(len(result["recovered"])),
            "before_qc": result["before_qc"], "after_qc": result["after_qc"],
        }
        outputs = []
        if result["declined"]:
            # Legacy raised here and left the auto-fit sidecar in place; we materialise that explicitly.
            ecg_qrs.save_corrected_atomic(p["corrected"], saved, ecg["sfreq"])
            qc["declined_reason"] = (
                f"full-record recovery reduced valid 60-s windows "
                f"({result['before_qc']['valid_60s_windows']} to {result['after_qc']['valid_60s_windows']})")
            outputs.append(sn.output_item("corrected_sidecar", p["corrected"], heppy_out, "sidecar"))
            return {"status": "declined", "qc": qc, "outputs": outputs}
        counts = ecg_qrs.write_recovery_sidecar(p["corrected"], result["recovered"], ecg["sfreq"],
                                                result["detected"], saved)
        qc.update({f"final_source_{k}": v for k, v in counts.items()})
        qc["final_score"] = result["after_qc"]
        outputs.append(sn.output_item("corrected_sidecar", p["corrected"], heppy_out, "sidecar"))
        try:
            ecg_qrs.save_auto_qc_plot(
                p["recovery_png"], ecg["signal"], ecg["sfreq"], result["recovered"], eff["auto_qrs"]["alpha"],
                f"{job['base']} | full-record detector plus gap recovery | peaks={len(result['recovered'])}")
            outputs.append(sn.output_item("recovery_png", p["recovery_png"], heppy_out, "file"))
        except Exception as exc:
            qc["plot_error"] = f"{type(exc).__name__}: {exc}"
        return {"status": "ok", "qc": qc, "outputs": outputs}


def run_job(job: dict, impl=RealImpl) -> dict:
    """Run ``job['todo']`` steps; update the sentinel after each; return a result record."""
    heppy_out = Path(job["heppy_out"])
    base = job["base"]
    sentinel = job["sentinel"]
    sentinel.setdefault("steps", {})
    path = sn.sentinel_path(heppy_out, base)
    ran: list[str] = []
    for step in opt.STEPS:
        if step not in job["todo"]:
            continue
        started = time.time()
        print(f"HEPPY_STEP\t{base}\t{step}\tbegin", flush=True)
        try:
            res = impl.run(step, job, sentinel)
        except Exception as exc:
            res = {"status": "failed", "qc": {}, "outputs": [],
                   "error": {"kind": "exception", "type": type(exc).__name__, "message": str(exc),
                             "traceback": traceback.format_exc()[-3000:]}}
            print(f"HEPPY_STEP\t{base}\t{step}\tFAILED\t{type(exc).__name__}: {exc}", flush=True)
        sentinel["steps"][step] = jsonable({
            "fp": job["step_fps"][step],
            "status": res["status"],
            "qc": res.get("qc", {}),
            "outputs": res.get("outputs", []),
            "error": res.get("error"),
            "elapsed_s": round(time.time() - started, 2),
            "computed_utc": sn.utc_now(),
        })
        sentinel.pop("crash", None)
        sn.write_sentinel(path, sentinel)
        ran.append(step)
        print(f"HEPPY_STEP\t{base}\t{step}\t{res['status']}\t{sentinel['steps'][step]['elapsed_s']}s", flush=True)
    return {"base": base, "ran": ran, "ok": True}


def _disable_windows_error_dialogs() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m fsms.heppy.worker JOB.json", file=sys.stderr)
        return 2
    _disable_windows_error_dialogs()
    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        pass
    for var, value in pinned_environment(json.loads(Path(argv[0]).read_text(encoding="utf-8"))["options"]).items():
        if var in THREAD_VARS or var in ("PYTHONHASHSEED",):
            os.environ[var] = value
    job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    result = run_job(job)
    atomic_write_json(job["result_path"], result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
