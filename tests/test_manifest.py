"""Fast synthetic tests of the VG-microstates manifest builder (tiny RawArray FIFs; no TUEG, no MATLAB).

Real study data is never touched: the guard test uses a non-existent path under a restricted root.
The conversion tests run the unmodified VG converter in a subprocess on tiny FIFs (a few seconds).
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from fsms.guard import RestrictedDataError
from fsms.manifest import build_vg_manifest as vg

EEG_NAMES = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
             "Fz", "Cz", "Pz", "A1", "A2"]
RESTRICTED_PROBE = r"G:\FS_microstates_study\__never_created__\vg"
SNIP_FIELDS = ["patient_code", "patient_id", "output_edf", "segment", "output_start_seconds", "output_end_seconds",
               "source", "source_sheet", "source_row", "event_type", "event_number", "segment_number",
               "source_edf_path", "source_migrated_edf_path"]
DB_FIELDS = ["patient_code", "source", "source_sheet", "source_row", "event_type", "event_number", "segment_number",
             "patient_type", "patient_type_raw", "seizure_type", "hypo_hyper", "event_classification"]
RESTRICTED_CSV_NAMES = ("analysis_database.csv", "edf_keyfile.csv", "patient_keyfile.csv")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_fif(path: Path, seed: int, duration_s: float = 1000.0, sfreq: float = 100.0, montage: bool = False) -> None:
    import mne

    rng = np.random.default_rng(seed)
    names = EEG_NAMES + ["EKG1"]
    info = mne.create_info(names, sfreq, ["eeg"] * len(EEG_NAMES) + ["ecg"])
    data = rng.standard_normal((len(names), int(duration_s * sfreq))) * 1e-5
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_meas_date(1)   # as HEPPy does: saved bytes are deterministic
    if montage:
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), match_case=False, on_missing="ignore")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(str(path), overwrite=True, verbose="ERROR")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def snip(patient: str, segment: str, start: float, end: float, row: str, event: str, seg_no: str = "") -> dict:
    return {"patient_code": patient, "patient_id": "id-" + patient, "output_edf": f"{patient}_snips.edf",
            "segment": segment, "output_start_seconds": str(start), "output_end_seconds": str(end),
            "source": "SYN", "source_sheet": "s", "source_row": row,
            "event_type": "II" if segment == "II" else "Ict", "event_number": event, "segment_number": seg_no,
            "source_edf_path": "x.edf", "source_migrated_edf_path": "x.edf"}


class ManifestBase(unittest.TestCase):
    patients = ("P1", "P2", "P3", "P4")

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="fsms_vg_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.heppy = self.tmp / "heppy_output"
        self.out = self.tmp / "vg"
        for i, patient in enumerate(self.patients):
            write_fif(self.heppy / "raw_fif" / f"{patient}_snips_pp_raw.fif", seed=i)
        # P1: good; P2: excluded by the HEPPy QC rule; P3: no HEPPy FIF path ("pending"); P4: good plus a
        # 5 s preictal sliver that is too short once cropped.  Each patient has II, preictal, ictal (5 s),
        # postictal; P1 additionally has an unsupported segment name.
        rows = []
        for patient, offset in (("P1", 2), ("P2", 10), ("P3", 20), ("P4", 30)):
            rows += [snip(patient, "II", 0, 300, str(offset), "0", "1"),
                     snip(patient, "preictal", 300, 900, str(offset + 1), "1"),
                     snip(patient, "ictal", 900, 905, str(offset + 1), "1"),
                     snip(patient, "postictal", 905, 1100, str(offset + 1), "1")]
        rows.append(snip("P1", "weirdsegment", 0, 5, "99", "1"))
        rows.append(snip("P4", "preictal", 995, 1000, "31", "1"))     # 5 s: too short once cropped
        self.snip_rows = rows
        self.keyfile = self.tmp / "clips" / "snip_keyfile.csv"
        write_csv(self.keyfile, rows, SNIP_FIELDS)
        review = [{"base": f"{p}_snips", "raw_fif": str(self.heppy / "raw_fif" / f"{p}_snips_pp_raw.fif"),
                   "raw_fif_keep": "", "viable": "yes"} for p in self.patients]
        review[2]["raw_fif"] = ""                                     # P3 has no FIF path at all
        write_csv(self.heppy / "ecg_qc_review.csv", review, ["base", "raw_fif", "raw_fif_keep", "viable"])
        self.exclusion_fields = ["base", "stage", "reason_code", "evidence", "scope", "segment",
                                 "output_start_seconds", "output_end_seconds"]
        write_csv(self.heppy / "heppy_exclusions.csv",
                  [{"base": "P2_snips", "stage": "ecg", "reason_code": "no_ecg_channel", "evidence": "x", "scope": "recording"},
                   {"base": "P4_snips", "stage": "pyprep", "reason_code": "segment_pyprep_skipped_toomanybad",
                    "evidence": "chunk2@300-900s", "scope": "segment", "segment": "preictal",
                    "output_start_seconds": "300", "output_end_seconds": "900"}],
                  self.exclusion_fields)
        self.database = self.tmp / "db.csv"
        self.write_database({"P1": "FS", "P2": "ES", "P3": "FS", "P4": "DD"}, skip={"P3"})

    def write_database(self, groups: dict[str, str], skip: frozenset | set = frozenset()) -> None:
        db = []
        for patient, offset in (("P1", 2), ("P2", 10), ("P3", 20), ("P4", 30)):
            if patient in skip:
                continue
            group = groups[patient]
            db.append({"patient_code": patient, "source": "SYN", "source_sheet": "s", "source_row": str(offset),
                       "event_type": "II", "event_number": "0", "segment_number": "1", "patient_type": group,
                       "patient_type_raw": group, "seizure_type": "", "hypo_hyper": "", "event_classification": "ii"})
            for r in (offset + 1,):
                db.append({"patient_code": patient, "source": "SYN", "source_sheet": "s", "source_row": str(r),
                           "event_type": "Ict", "event_number": "1", "segment_number": "", "patient_type": group,
                           "patient_type_raw": group, "seizure_type": "FS" if group == "FS" else "ES",
                           "hypo_hyper": "hypo" if group == "FS" else "", "event_classification": "ict"})
        write_csv(self.database, db, DB_FIELDS)

    def run_builder(self, **kwargs):
        kwargs.setdefault("fmt", "fif")
        kwargs.setdefault("log", lambda m: None)
        return vg.run(self.keyfile, self.heppy, self.database, self.out, **kwargs)

    def outputs_hash(self) -> dict[str, str]:
        return {str(p.relative_to(self.out)): sha(p) for p in sorted(self.out.rglob("*")) if p.is_file()
                and ".sentinels" not in p.parts}


class ReasonCodeTests(ManifestBase):
    def test_every_row_is_in_manifest_or_exclusions_with_reason_codes(self) -> None:
        result = self.run_builder()
        rows = result["rows"]
        excluded = result["excluded"]
        self.assertEqual(len(rows) + len(excluded), len(self.snip_rows), "no silent drops")
        manifest = read_rows(self.out / "vg_microstate_manifest_fif.csv")
        self.assertEqual(list(manifest[0]), vg.MANIFEST_COLUMNS)
        self.assertEqual({r["condition"] for r in manifest}, {"II", "preict"})
        self.assertEqual({(r["participant"], r["condition"]) for r in manifest},
                         {("P1", "II"), ("P1", "preict"), ("P4", "II")})   # P4's preictal is segment-excluded
        codes = {}
        for e in excluded:
            codes.setdefault(e["reason_code"], []).append((e["patient_code"], e["segment"]))
        self.assertEqual(set(codes), {"unsupported_segment", "out_of_scope_condition", "heppy_excluded",
                                      "heppy_segment_excluded", "heppy_output_missing", "segment_too_short"})
        self.assertEqual(codes["heppy_segment_excluded"], [("P4", "preictal")])
        segment_row = next(e for e in excluded if e["reason_code"] == "heppy_segment_excluded")
        self.assertIn("segment_pyprep_skipped_toomanybad", segment_row["evidence"])
        self.assertEqual((segment_row["crop_start_seconds"], segment_row["crop_end_seconds"]), ("300", "900"))
        self.assertEqual(codes["unsupported_segment"], [("P1", "weirdsegment")])
        self.assertEqual(sorted(codes["heppy_excluded"]), [("P2", "II"), ("P2", "preictal")])
        self.assertEqual(sorted(codes["heppy_output_missing"]), [("P3", "II"), ("P3", "preictal")])
        self.assertEqual(codes["segment_too_short"], [("P4", "preictal")])
        self.assertTrue(all(e["reason_code"] != "out_of_scope_condition" or e["segment"] in ("ictal", "postictal")
                            for e in excluded))
        written = read_rows(self.out / "vg_microstate_exclusions.csv")
        self.assertEqual(len(written), len(excluded))
        self.assertEqual(list(written[0]), vg.EXCLUSION_COLUMNS)
        summary = json.loads((self.out / "vg_manifest_summary.json").read_text())
        self.assertEqual(sum(summary["excluded_by_reason"].values()), len(excluded))

    def test_missing_database_row_is_reason_coded_after_qc(self) -> None:
        self.write_database({"P1": "FS", "P2": "ES", "P3": "FS", "P4": "DD"}, skip={"P1"})
        result = self.run_builder()
        missing = [e for e in result["excluded"] if e["reason_code"] == "missing_database_row"]
        self.assertEqual({(e["patient_code"], e["segment"]) for e in missing}, {("P1", "II"), ("P1", "preictal")})
        self.assertEqual({r["participant"] for r in result["rows"]}, {"P4"})

    def test_labels_cannot_influence_which_segments_survive(self) -> None:
        import inspect

        self.assertNotIn("database", inspect.signature(vg.plan_qc_phase).parameters)
        a = self.run_builder()
        survivors_a = [r["file_path"] for r in a["rows"]]
        groups_a = {r["participant"]: r["group"] for r in a["rows"]}
        self.write_database({"P1": "ES", "P2": "FS", "P3": "DD", "P4": "FS"}, skip={"P3"})
        b = self.run_builder()
        self.assertEqual([r["file_path"] for r in b["rows"]], survivors_a)
        self.assertEqual({r["participant"]: r["group"] for r in b["rows"]}, {"P1": "ES", "P4": "FS"})
        self.assertEqual(groups_a, {"P1": "FS", "P4": "DD"})
        # excluded patient with no DB row is still reported as heppy_excluded, not missing_database_row
        db = [r for r in read_rows(self.database) if r["patient_code"] != "P2"]
        write_csv(self.database, db, DB_FIELDS)
        c = self.run_builder()
        self.assertTrue(any(e["reason_code"] == "heppy_excluded" and e["patient_code"] == "P2" for e in c["excluded"]))
        self.assertFalse(any(e["reason_code"] == "missing_database_row" and e["patient_code"] == "P2" for e in c["excluded"]))

    def test_conditions_can_be_widened_explicitly(self) -> None:
        result = self.run_builder(conditions=("II", "preict", "ictal"), min_duration_s=3.0)
        self.assertIn("ictal", {r["condition"] for r in result["rows"]})
        result = self.run_builder(conditions=("II", "preict", "ictal"), min_duration_s=10.0)
        self.assertTrue(any(e["reason_code"] == "segment_too_short" and e["segment"] == "ictal" for e in result["excluded"]))
        with self.assertRaises(vg.InfrastructureError):
            self.run_builder(conditions=("II", "nonsense"))

    def test_duplicate_stems_get_crop_suffix_like_legacy(self) -> None:
        rows = [dict(r, segment_number="2") if (r["patient_code"], r["segment"]) == ("P1", "preictal") else r
                for r in self.snip_rows]
        rows.append(snip("P1", "preictal", 600, 900, "3", "1", "2"))   # same patient/condition/row/event/segment
        write_csv(self.keyfile, rows, SNIP_FIELDS)
        result = self.run_builder()
        names = [Path(r["file_path"]).name for r in result["rows"] if r["participant"] == "P1" and r["condition"] == "preict"]
        self.assertEqual(len(names), 2)
        self.assertTrue(any("_crop600-900" in n for n in names))

    def test_first_generation_exclusion_file_without_scope_column_is_recording_level(self) -> None:
        write_csv(self.heppy / "heppy_exclusions.csv",
                  [{"base": "P4_snips", "stage": "pyprep", "reason_code": "pyprep_failed", "evidence": "old format"}],
                  ["base", "stage", "reason_code", "evidence"])
        result = self.run_builder()
        self.assertEqual({e["patient_code"] for e in result["excluded"] if e["reason_code"] == "heppy_excluded"}, {"P4"})
        self.assertFalse(any(e["reason_code"] == "heppy_segment_excluded" for e in result["excluded"]))

    def test_missing_heppy_outputs_is_infrastructure_error(self) -> None:
        (self.heppy / "heppy_exclusions.csv").unlink()
        with self.assertRaises(vg.InfrastructureError):
            self.run_builder()

    def test_guard_refuses_restricted_output_dir(self) -> None:
        with self.assertRaises(RestrictedDataError):
            vg.run(self.keyfile, self.heppy, self.database, RESTRICTED_PROBE, log=lambda m: None)
        with self.assertRaises(RestrictedDataError):
            vg.run(self.keyfile, r"D:\Chalfont\__never_created__", self.database, self.out, log=lambda m: None)
        with self.assertRaises(RestrictedDataError):
            vg.run(self.keyfile, self.heppy, r"F:\EEGs\__never_created__\analysis_database.csv", self.out, log=lambda m: None)
        with mock.patch("fsms.lock.real_data_unlock_problems", return_value=["locked"]):
            with self.assertRaises(RestrictedDataError):
                vg.run(self.keyfile, self.heppy, self.database, self.out, mode="real", log=lambda m: None)
        self.assertFalse(self.out.exists())


class CropIdempotencyTests(ManifestBase):
    def test_second_run_recomputes_nothing_and_changes_no_bytes(self) -> None:
        first = self.run_builder()
        self.assertTrue(first["crop_actions"] and set(first["crop_actions"].values()) == {"written"})
        before = self.outputs_hash()
        mtimes = {p: p.stat().st_mtime_ns for p in (self.out / "fif_segments").glob("*.fif")}
        second = self.run_builder()
        self.assertEqual(set(second["crop_actions"].values()), {"unchanged"})
        self.assertEqual(self.outputs_hash(), before)
        self.assertEqual({p: p.stat().st_mtime_ns for p in (self.out / "fif_segments").glob("*.fif")}, mtimes)
        self.assertFalse(list(self.out.rglob("*.part*")))

    def test_corrupt_crop_heals_to_identical_bytes(self) -> None:
        self.run_builder()
        before = self.outputs_hash()
        victim = sorted((self.out / "fif_segments").glob("*.fif"))[0]
        data = victim.read_bytes()
        victim.write_bytes(data[: len(data) // 3])
        stale = victim.with_name(f".{victim.stem}.part-1.fif")
        stale.write_bytes(b"partial")
        result = self.run_builder()
        self.assertEqual(list(result["crop_actions"].values()).count("written"), 1)
        self.assertEqual(result["crop_actions"][str(victim)], "written")
        self.assertFalse(stale.exists())
        self.assertEqual(self.outputs_hash(), before)

    def test_changed_source_or_bounds_recompute(self) -> None:
        self.run_builder()
        write_fif(self.heppy / "raw_fif" / "P1_snips_pp_raw.fif", seed=99)         # P1 FIF replaced
        result = self.run_builder()
        actions = {Path(k).name.split("_")[0]: v for k, v in result["crop_actions"].items()}
        self.assertEqual(actions["P1"], "written")
        self.assertEqual(actions["P4"], "unchanged")
        rows = [dict(r) for r in self.snip_rows]
        for r in rows:
            if r["patient_code"] == "P1" and r["segment"] == "preictal":
                r["output_end_seconds"] = "800"
        write_csv(self.keyfile, rows, SNIP_FIELDS)
        result = self.run_builder()
        changed = [Path(k).name for k, v in result["crop_actions"].items() if v == "written"]
        self.assertEqual(len(changed), 1)
        self.assertTrue(changed[0].startswith("P1_preict"))

    def test_crop_has_expected_length_and_channels(self) -> None:
        import mne

        result = self.run_builder()
        row = next(r for r in result["rows"] if r["participant"] == "P1" and r["condition"] == "preict")
        raw = mne.io.read_raw_fif(row["file_path"], preload=False, verbose="ERROR")
        self.assertEqual(raw.n_times, 600 * 100)
        self.assertEqual(len(mne.pick_types(raw.info, eeg=True)), len(EEG_NAMES))


class ConversionTests(ManifestBase):
    def test_mat_conversion_passes_montage_and_is_idempotent(self) -> None:
        import scipy.io as sio

        # FIFs carry NO channel positions; positions must come from the --montage the builder passes
        result = self.run_builder(fmt="mat", montage="standard_1020", conditions=("II",))
        self.assertEqual(set(result["convert_actions"].values()), {"written"})
        manifest = read_rows(self.out / "vg_microstate_manifest_mat.csv")
        self.assertEqual(list(manifest[0]), vg.MANIFEST_COLUMNS)
        self.assertTrue(all(r["file_path"].endswith(".mat") for r in manifest))
        data = sio.loadmat(manifest[0]["file_path"])
        pos = np.asarray(data["pos"], dtype=float)
        self.assertEqual(pos.shape, (len(EEG_NAMES), 3))
        self.assertTrue(np.all(np.isfinite(pos)), "montage must supply channel positions")
        self.assertEqual(data["data"].shape[0], len(EEG_NAMES))
        before = self.outputs_hash()
        again = self.run_builder(fmt="mat", montage="standard_1020", conditions=("II",))
        self.assertEqual(set(again["convert_actions"].values()), {"unchanged"})
        self.assertEqual(self.outputs_hash(), before)
        # delete one .mat and truncate another: only those two are rebuilt
        mats = sorted((self.out / "converted_mat").glob("*.mat"))
        self.assertEqual(len(mats), 2)
        mats[0].unlink()
        mats[1].write_bytes(mats[1].read_bytes()[:200])
        healed = self.run_builder(fmt="mat", montage="standard_1020", conditions=("II",))
        self.assertEqual(list(healed["convert_actions"].values()).count("written"), 2)
        self.assertEqual(set(healed["crop_actions"].values()), {"unchanged"})
        self.assertEqual(self.outputs_hash(), before)
        self.assertFalse((self.out / ".convert_stage").exists(), "staging area removed")

    def test_changed_montage_invalidates_conversion_only(self) -> None:
        self.run_builder(fmt="mat", montage="standard_1020", conditions=("II",))
        result = self.run_builder(fmt="mat", montage="standard_1005", conditions=("II",))
        self.assertEqual(set(result["crop_actions"].values()), {"unchanged"})
        self.assertEqual(set(result["convert_actions"].values()), {"written"})


if __name__ == "__main__":
    unittest.main()
