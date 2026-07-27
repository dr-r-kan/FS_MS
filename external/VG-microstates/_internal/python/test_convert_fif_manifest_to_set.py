import importlib.util
import os
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba_cache"))
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(Path(tempfile.gettempdir()) / "pycache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_cache"))

MODULE_PATH = Path(__file__).with_name("_convert_fif_manifest_to_set.py")
spec = importlib.util.spec_from_file_location("convert_fif_manifest_to_set", MODULE_PATH)
convert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert)


def test_heppy_front_back_landmarks_define_anterior_axis():
    labels = ["C18", "A25"]
    ras = np.array([[0.0, 0.093, 0.056], [0.0, -0.086, 0.0]])
    rotated = np.array([[0.093, 0.0, 0.056], [-0.086, 0.0, 0.0]])

    assert convert.heppy_front_back_landmarks_ok(labels, ras)
    assert not convert.heppy_front_back_landmarks_ok(labels, rotated)


def test_mne_export_keeps_ras_axis():
    c18 = np.array([0.0, 0.093, 0.056])
    assert np.allclose(convert.mne_xyz_to_pipeline_xyz(c18), c18)


if __name__ == "__main__":
    test_heppy_front_back_landmarks_define_anterior_axis()
    test_mne_export_keeps_ras_axis()
