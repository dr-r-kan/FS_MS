"""The lean autonomic stage must reproduce the legacy implementation exactly (synthetic data only)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fsms import autonomic  # noqa: E402


def synthetic_rpeaks(duration_s: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    times, t = [], 0.4
    while t < duration_s:
        times.append(t)
        rr = 0.85 + 0.06 * np.sin(2 * np.pi * t / 90) + 0.03 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.012)
        t += max(rr, 0.45)
    return np.asarray(times)


def build_tree(root: Path, participants: int = 2):
    heppy = root / "heppy_output"
    (heppy / "raw_fif").mkdir(parents=True)
    (heppy / "ecg_qc").mkdir()
    rows = []
    for i in range(participants):
        base = f"P_{i:03d}_snips"
        fif = heppy / "raw_fif" / f"{base}_pp_raw.fif"
        fif.write_bytes(b"")
        peaks = synthetic_rpeaks(2200, seed=i)
        recovered = np.zeros(len(peaks), dtype=int)
        recovered[::97] = 1
        pd.DataFrame({"rpeak_time_s": peaks, "rpeak_source": "reviewed", "is_gap_recovered": recovered}).to_csv(
            heppy / "ecg_qc" / f"{base}_rpeaks_corrected.csv", index=False)
        for condition, start, stop in (("II", 0.0, 500.0), ("preict", 600.0, 2000.0)):
            rows.append({"participant": f"P{i}", "condition": condition, "group": "FS",
                         "file_path": str(root / f"P{i}_{condition}.mat"), "source_raw_fif": str(fif),
                         "crop_start_seconds": start, "crop_end_seconds": stop})
    return heppy, pd.DataFrame(rows)


class AutonomicTests(unittest.TestCase):
    def test_sidecar_resolution_from_fif_location(self):
        path = autonomic.sidecar_for(r"X:\run\heppy_output\raw_fif\P_001_snips_pp_raw_keepcfa.fif")
        self.assertEqual(path.name, "P_001_snips_rpeaks_corrected.csv")
        self.assertEqual(path.parent.name, "ecg_qc")

    def test_missing_sidecar_is_reason_coded_not_silent(self):
        with tempfile.TemporaryDirectory() as td:
            heppy, manifest = build_tree(Path(td))
            (heppy / "ecg_qc" / "P_001_snips_rpeaks_corrected.csv").unlink()
            native, qc = autonomic.build_native(manifest)
            missing = qc[qc.participant.eq("P1")]
            self.assertTrue((~missing.autonomic_valid).all())
            self.assertTrue((missing.reason == "no_corrected_rpeaks").all())
            self.assertEqual(set(native.participant), {"P0"})

    def test_all_invalid_raises_with_reasons(self):
        with tempfile.TemporaryDirectory() as td:
            heppy, manifest = build_tree(Path(td), participants=1)
            (heppy / "ecg_qc" / "P_000_snips_rpeaks_corrected.csv").unlink()
            with self.assertRaisesRegex(ValueError, "no_corrected_rpeaks"):
                autonomic.build_native(manifest)

    def test_matches_legacy_implementation_exactly(self):
        try:
            import analyse_focused_fs_ms as legacy
        except Exception as error:  # legacy stack unavailable (e.g. CI): nothing to compare against
            self.skipTest(f"legacy module not importable: {error!r}")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            heppy, manifest = build_tree(root)
            data_dir = root / "legacy_data"
            data_dir.mkdir()
            old_native, _, _ = legacy.read_ecg(manifest, heppy / "ecg_qc", heppy / "ecg_qc", data_dir, reuse=False)
            old = legacy.standardise_autonomic_to_ii(old_native)
            new_native, qc = autonomic.build_native(manifest)
            new = autonomic.standardise_autonomic_to_ii(new_native)
            self.assertTrue(qc.autonomic_valid.all())
            key = ["participant", "file_path", "event_time_s"]
            pd.testing.assert_frame_equal(old.sort_values(key).reset_index(drop=True),
                                          new.sort_values(key).reset_index(drop=True), check_exact=False,
                                          rtol=0, atol=1e-12)
            self.assertGreater(new.csi_z.notna().sum(), 100)

    def test_run_writes_table_qc_and_audit_atomically(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, manifest = build_tree(root)
            manifest_csv = root / "manifest.csv"
            manifest.to_csv(manifest_csv, index=False)
            audit = autonomic.run(manifest_csv, root / "out")
            self.assertEqual(audit["records_excluded"], 0)
            self.assertTrue((root / "out" / "autonomic_native_ii_standardised.csv.gz").exists())
            self.assertFalse(list((root / "out").glob("*.part")))
            table = pd.read_csv(root / "out" / "autonomic_native_ii_standardised.csv.gz")
            self.assertTrue({"file_path", "event_time_s", "csi_z", "cpi_z"} <= set(table.columns))


if __name__ == "__main__":
    unittest.main()
