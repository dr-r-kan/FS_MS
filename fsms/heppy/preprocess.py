"""Run vendored HEPPy EEG preprocessing on one clip and record what it silently did.

HEPPy's signal processing is untouched.  The module only (a) binds the runtime configuration the
way ``OLD/run_heppy_preprocessing.runtime_config`` did and (b) wraps a handful of HEPPy functions
with *transparent observers* (the wrapper calls the original with the original arguments and
returns its result unchanged) to capture facts HEPPy otherwise swallows:

* ``standardise_and_montage``     -> EEG channel counts before/after the montage step;
* ``run_pyprep`` (+ a ``PrepPipeline`` subclass and a root-logger handler) -> per-chunk PyPREP
  outcome: ``ok`` / ``ransac_fallback`` / ``skipped_toomanybad`` / ``skipped_no_montage`` /
  ``failed``, bad channels by detector type, interpolated and still-noisy channel names;
* ``_fit_ica_and_labels`` / ``_fit_asr_ica_iclabel_once`` -> ICA size, decimation, ICLabel classes,
  exclusions per cardiac variant;
* ``_apply_ica_excludes`` -> whether the "ICA flattened EEG, keeping pre-ICA data" fallback fired.

The wrapped names are looked up as module globals by HEPPy at call time, so patching the module
attribute is sufficient and reversible (``Collector.uninstall``).
"""
from __future__ import annotations

import logging
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from fsms.heppy import options as opt

PYPREP_STATUS_ORDER = ("failed", "skipped_toomanybad", "skipped_no_montage", "ransac_fallback", "ok")
BAD_TYPES = ("bad_by_nan", "bad_by_flat", "bad_by_deviation", "bad_by_hf_noise", "bad_by_correlation",
             "bad_by_SNR", "bad_by_dropout", "bad_by_psd", "bad_by_ransac", "bad_by_manual")


def runtime_config(options: dict, heppy_output: Path) -> SimpleNamespace:
    """HEPPy runtime configuration (mirror of the legacy ``runtime_config``)."""
    eff = opt.processing_options(options)
    tuning = eff["tuning"]
    cutoff = eff["asr_cutoff"]
    return SimpleNamespace(
        output_root=str(heppy_output),
        target_sfreq=eff["target_sfreq"],
        use_asr=cutoff,
        asr_cutoff=cutoff,
        use_pyprep=eff["pyprep"],
        remove_cfa_mode=eff["cfa_mode"],
        remove_cfa=eff["cfa_mode"] != "keep",
        log_file=str(heppy_output / "logs" / "preprocessing.tsv"),
        ref_chs="eeg",
        reref_chs="eeg",
        high_pass=eff["filters"]["hp"],
        low_pass=eff["filters"]["lp"],
        prep_ransac=eff["prep_ransac"],
        line_freqs=tuple(eff["filters"]["line_freqs"]),
        montage_name=eff["montage"]["name"],
        rename_to_1020=eff["montage"]["rename_to_1020"],
        use_ica=eff["ica_iclabel"],
        spirometry_channel=eff["spirometry_channel"],
        pyprep_channel_wise=tuning["pyprep_channel_wise"],
        pyprep_max_chunk_size=tuning["pyprep_max_chunk_size"],
        pyprep_by_segment=tuning["pyprep_by_segment"],
        pyprep_safe_quantile_patch=tuning["pyprep_safe_quantile_patch"],
        ica_fit_hz=tuning["ica_fit_hz"],
        iclabel_backend=tuning["iclabel_backend"],
    )


# --------------------------------------------------------------------------- classification

def leaf_status(record: dict) -> str:
    """Outcome of one PyPREP chunk from the fit attempts and warnings seen while it ran."""
    messages = record.get("messages", [])
    fits = record.get("fits", [])
    if any("no EEG channels have usable montage positions" in m for m in messages):
        return "skipped_no_montage"
    if any(f["ok"] for f in fits):
        if fits and not fits[0]["ok"]:
            return "ransac_fallback"
        return "ok"
    if any("RobustReference:TooManyBad" in (f.get("error") or "") for f in fits) or any(
            "too many bad EEG channels for robust reference" in m for m in messages):
        return "skipped_toomanybad"
    return "failed"


def _leaves(records: list[dict]) -> list[dict]:
    out: list[dict] = []
    for record in records:
        children = record.get("children", [])
        out.extend(_leaves(children))
        if record.get("fits") or record.get("exception") or not children:
            out.append(record)
    return out


def summarise_pyprep(records: list[dict], enabled: bool) -> dict:
    """Recording-level PyPREP summary from the recorded call tree (pure; unit-tested)."""
    if not enabled:
        return {"status": "disabled", "n_chunks": 0}
    leaves = _leaves(records)
    if not leaves:
        return {"status": "not_run", "n_chunks": 0}
    chunks = []
    union_bad: dict[str, set] = {name: set() for name in BAD_TYPES}
    union_interp: set[str] = set()
    for index, leaf in enumerate(leaves):
        status = leaf_status(leaf)
        interpolated = sorted(set(leaf.get("interpolated", [])) | set(leaf.get("still_noisy", [])))
        n_eeg = int(leaf.get("n_eeg") or 0)
        fraction = (len(interpolated) / n_eeg) if (n_eeg and status in ("ok", "ransac_fallback")) else None
        for name, channels in (leaf.get("bad_by_type") or {}).items():
            union_bad.setdefault(name, set()).update(channels)
        if status in ("ok", "ransac_fallback"):
            union_interp.update(interpolated)
        chunks.append({
            "index": index, "n_eeg": n_eeg, "status": status,
            "n_interpolated": len(interpolated) if status in ("ok", "ransac_fallback") else None,
            "interpolated_fraction": fraction,
            "interpolated": interpolated if status in ("ok", "ransac_fallback") else [],
            "bad_by_type_counts": {k: len(v) for k, v in (leaf.get("bad_by_type") or {}).items() if v},
            "error": leaf.get("exception") or next((f["error"] for f in leaf.get("fits", []) if f.get("error")), None),
        })
    counts = Counter(chunk["status"] for chunk in chunks)
    status = next(s for s in PYPREP_STATUS_ORDER if counts.get(s))
    fractions = [c["interpolated_fraction"] for c in chunks if c["interpolated_fraction"] is not None]
    return {
        "status": status,
        "n_chunks": len(chunks),
        "status_counts": dict(sorted(counts.items())),
        "bad_channels_by_type": {k: sorted(v) for k, v in sorted(union_bad.items()) if v},
        "n_interpolated_union": len(union_interp),
        "interpolated_union": sorted(union_interp),
        "interpolated_fraction_max": max(fractions) if fractions else None,
        "interpolated_fraction_mean": (sum(fractions) / len(fractions)) if fractions else None,
        "chunks": chunks,
    }


def annotate_chunk_bounds(summary: dict, bounds: list[tuple[float, float]]) -> dict:
    """Attach ``start_s``/``end_s`` to each PyPREP chunk when chunks and keyfile segments correspond."""
    chunks = summary.get("chunks") or []
    if chunks and len(chunks) == len(bounds):
        for chunk, (start, end) in zip(chunks, bounds):
            chunk["start_s"], chunk["end_s"] = start, end
        summary["chunks_have_bounds"] = True
    else:
        summary["chunks_have_bounds"] = False
    return summary


def summarise_ica(collector: "Collector", enabled: bool) -> dict:
    if not enabled:
        return {"enabled": False}
    out: dict = {"enabled": True, "n_components": None, "decim": collector.ica_decim}
    if collector.ica_labels is not None:
        out["n_components"] = len(collector.ica_labels)
        out["iclabel_classes"] = dict(sorted(Counter(collector.ica_labels).items()))
    for variant in ("remove", "keep"):
        info = collector.ica_variants.get(variant)
        if info is not None:
            out[f"excluded_{variant}"] = dict(sorted(Counter(info["label_map"].values()).items()))
            out[f"n_excluded_{variant}"] = len(info["bads"])
    flat = collector.flatten
    out["flatten_reverted_remove"] = bool(flat[0]) if len(flat) > 0 else None
    out["flatten_reverted_keep"] = bool(flat[1]) if len(flat) > 1 else None
    out["flatten_reverted_any"] = any(flat)
    return out


# --------------------------------------------------------------------------- observers

class _CaptureHandler(logging.Handler):
    def __init__(self, collector: "Collector") -> None:
        super().__init__(level=logging.WARNING)
        self.collector = collector

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            return
        self.collector.messages.append(message)
        if self.collector.stack:
            self.collector.stack[-1]["messages"].append(message)


class Collector:
    """Gathers observations while ``preprocess_edf_both`` runs."""

    def __init__(self) -> None:
        self.n_eeg_input: int | None = None
        self.n_eeg_after_montage: int | None = None
        self.pyprep_records: list[dict] = []
        self.stack: list[dict] = []
        self.ica_labels: list[str] | None = None
        self.ica_decim: int | None = None
        self.ica_variants: dict[str, dict] = {}
        self.flatten: list[bool] = []
        self.messages: list[str] = []
        self._patched: list[tuple[object, str, object]] = []
        self._handlers: list[logging.Handler] = []

    # -- patch plumbing
    def _patch(self, owner, name: str, replacement) -> None:
        self._patched.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def uninstall(self) -> None:
        for owner, name, original in reversed(self._patched):
            setattr(owner, name, original)
        self._patched.clear()
        root = logging.getLogger()
        for handler in self._handlers:
            root.removeHandler(handler)
        self._handlers.clear()

    def install(self, pp) -> "Collector":
        import mne

        collector = self
        root = logging.getLogger()
        capture = _CaptureHandler(self)
        root.addHandler(capture)
        self._handlers.append(capture)
        if len(root.handlers) == 1:  # nothing else prints warnings (lastResort is now bypassed): keep them visible
            console = logging.StreamHandler(sys.stderr)
            console.setLevel(logging.WARNING)
            root.addHandler(console)
            self._handlers.append(console)

        orig_montage = pp.standardise_and_montage

        def standardise_and_montage(raw):
            collector.n_eeg_input = sum(1 for t in raw.get_channel_types() if t == "eeg")
            result = orig_montage(raw)
            collector.n_eeg_after_montage = sum(1 for t in result.get_channel_types() if t == "eeg")
            return result

        self._patch(pp, "standardise_and_montage", standardise_and_montage)

        orig_pyprep = pp.run_pyprep

        def run_pyprep(raw, *args, **kwargs):
            record = {"children": [], "fits": [], "messages": [], "exception": None,
                      "n_eeg": len(mne.pick_types(raw.info, eeg=True)),
                      "bad_by_type": {}, "interpolated": [], "still_noisy": []}
            parent = collector.stack[-1] if collector.stack else None
            collector.stack.append(record)
            try:
                return orig_pyprep(raw, *args, **kwargs)
            except Exception as exc:
                record["exception"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                collector.stack.pop()
                (parent["children"] if parent is not None else collector.pyprep_records).append(record)

        self._patch(pp, "run_pyprep", run_pyprep)

        pyprep_module = getattr(pp, "pyprep", None)
        if pyprep_module is not None:
            Base = pyprep_module.PrepPipeline

            class RecordingPrepPipeline(Base):  # no __init__ override: inspect.signature is unchanged
                def fit(self):  # noqa: D401
                    try:
                        result = super().fit()
                    except Exception as exc:
                        collector._record_fit(self, ok=False, error=f"{type(exc).__name__}: {exc}")
                        raise
                    collector._record_fit(self, ok=True, error=None)
                    return result

            RecordingPrepPipeline.__name__ = Base.__name__
            RecordingPrepPipeline.__qualname__ = Base.__qualname__
            self._patch(pyprep_module, "PrepPipeline", RecordingPrepPipeline)

        orig_fit_ica = pp._fit_ica_and_labels

        def _fit_ica_and_labels(raw, asr_thresh, random_seed, n_comp):
            result = orig_fit_ica(raw, asr_thresh, random_seed, n_comp)
            raw_eeg, _ica, labels, _eog = result
            collector.ica_labels = [str(label) for label in labels]
            try:
                collector.ica_decim = int(pp._ica_decimation(raw_eeg))
            except Exception:
                collector.ica_decim = None
            return result

        self._patch(pp, "_fit_ica_and_labels", _fit_ica_and_labels)

        orig_once = pp._fit_asr_ica_iclabel_once

        def _fit_asr_ica_iclabel_once(*args, **kwargs):
            result = orig_once(*args, **kwargs)
            (_raw, _ica, _labels, bads_remove, bads_keep, map_remove, map_keep, _eog) = result
            collector.ica_variants["remove"] = {"bads": list(bads_remove), "label_map": dict(map_remove)}
            collector.ica_variants["keep"] = {"bads": list(bads_keep), "label_map": dict(map_keep)}
            return result

        self._patch(pp, "_fit_asr_ica_iclabel_once", _fit_asr_ica_iclabel_once)

        orig_apply = pp._apply_ica_excludes

        def _apply_ica_excludes(inst, fitted_ica, exclude):
            requested = list(exclude)
            result = orig_apply(inst, fitted_ica, exclude)
            collector.flatten.append(bool(requested) and not list(fitted_ica.exclude))
            return result

        self._patch(pp, "_apply_ica_excludes", _apply_ica_excludes)
        return self

    def _record_fit(self, prep, ok: bool, error: str | None) -> None:
        if not self.stack:
            return
        record = self.stack[-1]
        params = getattr(prep, "ransac_settings", {}) or {}
        record["fits"].append({"ransac": bool(params.get("ransac")), "ok": ok, "error": error})
        if ok:
            before = getattr(prep, "noisy_channels_before_interpolation", None) or {}
            record["bad_by_type"] = {k: sorted(map(str, v)) for k, v in before.items()
                                     if isinstance(v, (list, tuple, set)) and k != "bad_all"}
            record["interpolated"] = sorted(map(str, getattr(prep, "interpolated_channels", []) or []))
            record["still_noisy"] = sorted(map(str, getattr(prep, "still_noisy_channels", []) or []))


# --------------------------------------------------------------------------- driver

def output_paths(base: str, heppy_out: Path) -> dict[str, Path]:
    raw_dir = heppy_out / "raw_fif"
    return {
        "raw_fif": raw_dir / f"{base}_pp_raw.fif",
        "raw_fif_keep": raw_dir / f"{base}_pp_raw_keepcfa.fif",
        "ica_remove": raw_dir / f"{base}_ica.fif",
        "ica_keep": raw_dir / f"{base}_keepcfa_ica.fif",
    }


def segment_bounds_s(edf_path: Path) -> list[tuple[float, float]]:
    """Output-time bounds (s) of this clip's segments from the neighbouring snip keyfile, sorted.

    HEPPy's ``_snip_sample_bounds_for_input`` chunks PyPREP by exactly these rows, in this order,
    so chunk *i* of the QC record covers ``bounds[i]`` whenever the counts agree.
    """
    import csv

    path = Path(edf_path).with_name("snip_keyfile.csv")
    bounds: list[tuple[float, float]] = []
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if Path(str(row.get("output_edf", "")).replace("\\", "/")).name != Path(edf_path).name:
                    continue
                try:
                    bounds.append((float(row["output_start_seconds"]), float(row["output_end_seconds"])))
                except (KeyError, ValueError):
                    continue
    except OSError:
        return []
    return sorted(bounds)


def run_preprocess(options: dict, base: str, edf_path: Path, heppy_out: Path, seed: int) -> dict:
    """Preprocess one clip (cardiac-removed + cardiac-kept FIFs); returns the QC fragment."""
    opt.ensure_heppy_on_path()
    import _preprocessing as pp  # type: ignore  # vendored HEPPy

    eff = opt.processing_options(options)
    outs = output_paths(base, heppy_out)
    for path in outs.values():  # recompute from scratch: never reuse partial/old outputs
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
    tsv = heppy_out / "logs" / "preprocessing" / f"{base}.tsv"
    tsv.parent.mkdir(parents=True, exist_ok=True)
    tsv.unlink(missing_ok=True)

    pp.set_runtime_config(runtime_config(options, heppy_out))
    collector = Collector().install(pp)
    try:
        pp.preprocess_edf_both(
            str(edf_path),
            output_path_remove=str(outs["raw_fif"]),
            output_path_keep=str(outs["raw_fif_keep"]),
            redo=True,
            flip_ecg=False,
            ecg_channel=eff["ecg_channel"] or None,
            stim_keep=eff["stim_keep"] or None,
            spirometry_channel=eff["spirometry_channel"] or None,
            spirometry_dir=None,
            logging_path=str(tsv),
            random_seed=int(seed),
        )
    finally:
        collector.uninstall()
    for key in ("raw_fif", "raw_fif_keep"):
        if not outs[key].exists():
            raise RuntimeError(f"HEPPy finished without writing {outs[key].name}")

    import mne

    header = mne.io.read_raw_fif(str(outs["raw_fif"]), preload=False, verbose="ERROR")
    eeg = mne.pick_types(header.info, eeg=True)
    ecg = mne.pick_types(header.info, ecg=True)
    fragment = {
        "sfreq": float(header.info["sfreq"]),
        "n_times": int(header.n_times),
        "n_eeg_input": collector.n_eeg_input,
        "n_eeg_after_montage": collector.n_eeg_after_montage,
        "n_eeg_output": int(len(eeg)),
        "eeg_channels_output": [header.ch_names[i] for i in eeg],
        "ecg_channels_output": [header.ch_names[i] for i in ecg],
        "pyprep": annotate_chunk_bounds(summarise_pyprep(collector.pyprep_records, eff["pyprep"]),
                                        segment_bounds_s(Path(edf_path))),
        "ica": summarise_ica(collector, eff["ica_iclabel"]),
        "warnings": sorted(set(collector.messages))[:50],
        "seed": int(seed),
    }
    header.close()
    return fragment
