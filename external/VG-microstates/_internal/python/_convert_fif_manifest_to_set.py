#!/usr/bin/env python3
"""Convert FIF manifest rows to EEGLAB .set plus chanloc sidecars."""

import argparse
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba_cache"))
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(Path(tempfile.gettempdir()) / "pycache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_cache"))

import mne
import numpy as np
import scipy.io as sio


FILE_COLUMNS = ("file_path", "file path", "file", "path", "filename")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest_csv")
    p.add_argument("output_dir")
    p.add_argument("--output-manifest", default="")
    p.add_argument("--file-column", default="")
    p.add_argument("--input-dir", default="", help="Directory containing files when manifest paths are stale")
    p.add_argument("--montage", default="", help="MNE builtin montage name or montage file to apply before export")
    p.add_argument("--format", choices=("set", "mat"), default="set", help="Output file format")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    manifest = Path(args.manifest_csv).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve() if args.input_dir else manifest.parent
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = Path(args.output_manifest).expanduser().resolve() if args.output_manifest else out_dir / "converted_manifest.csv"

    with manifest.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not rows or not fieldnames:
        raise SystemExit(f"No rows found in {manifest}")

    file_col = args.file_column or first_file_column(fieldnames)
    if not file_col:
        raise SystemExit(f"No file column found. Tried: {', '.join(FILE_COLUMNS)}")

    reference_positions = build_reference_positions(rows, file_col, input_dir, args.montage)
    print(f"reference montage positions: {len(reference_positions)}")

    for row in rows:
        src = resolve_input_path(row[file_col], input_dir)
        stem = src.stem
        out_path = out_dir / f"{stem}.{args.format}"
        rebuild_stale = False
        if out_path.exists() and not args.overwrite:
            if not export_needs_orientation_rebuild(out_path, args.format):
                row[file_col] = str(out_path)
                print(f"{src.name} -> {out_path.name} (exists)")
                continue
            rebuild_stale = True
            print(f"{src.name} -> {out_path.name} (stale montage; rebuilding)")
        raw = mne.io.read_raw_fif(src, preload=True, verbose="ERROR")
        raw.pick("eeg")
        if args.montage:
            raw.set_montage(load_montage(args.montage), on_missing="warn")

        if args.format == "set":
            raw.export(out_path, fmt="eeglab", overwrite=args.overwrite or rebuild_stale, verbose="ERROR")
            finite = write_chanloc_sidecars(raw, out_dir / stem, reference_positions)
        else:
            finite = write_mat_eeg(raw, out_path, reference_positions)
        row[file_col] = str(out_path)
        print(f"{src.name} -> {out_path.name} ({len(raw.ch_names)} EEG channels, {finite} finite positions)")

    with out_manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"manifest -> {out_manifest}")


def first_file_column(fieldnames):
    by_norm = {norm_name(x): x for x in fieldnames}
    for name in FILE_COLUMNS:
        hit = by_norm.get(norm_name(name))
        if hit:
            return hit
    return ""


def norm_name(name):
    return "".join(ch for ch in name.lower() if ch.isalnum())


def resolve_input_path(value, manifest_dir):
    p = Path(value).expanduser()
    if p.exists():
        return p.resolve()
    local = manifest_dir / Path(str(value).replace("\\", "/")).name
    if local.exists():
        return local.resolve()
    raise FileNotFoundError(value)


def load_montage(value):
    p = Path(value).expanduser()
    if p.exists():
        return mne.channels.read_custom_montage(p)
    return mne.channels.make_standard_montage(value)


def build_reference_positions(rows, file_col, input_dir, montage_name):
    ref = {}
    for row in rows:
        src = resolve_input_path(row[file_col], input_dir)
        raw = mne.io.read_raw_fif(src, preload=False, verbose="ERROR")
        raw.pick("eeg")
        if montage_name:
            raw.set_montage(load_montage(montage_name), on_missing="warn")
        xyz = channel_positions(raw)
        for label, pos in zip(raw.ch_names, xyz):
            key = normalise_channel_label(label)
            if key and key not in ref and usable_position(pos):
                ref[key] = pos
    return ref


def channel_positions(raw, reference_positions=None):
    reference_positions = reference_positions or {}
    montage_pos = {}
    montage = raw.get_montage()
    if montage is not None:
        montage_pos = montage.get_positions().get("ch_pos", {})
    montage_pos_by_key = {normalise_channel_label(k): v for k, v in montage_pos.items()}
    xyz = np.full((len(raw.ch_names), 3), np.nan)
    for i, (name, ch) in enumerate(zip(raw.ch_names, raw.info["chs"])):
        key = normalise_channel_label(name)
        candidates = (
            (montage_pos.get(name), True),
            (montage_pos_by_key.get(key), True),
            (ch.get("loc", np.full(12, np.nan))[:3], True),
            (reference_positions.get(key), False),
        )
        for pos, from_mne in candidates:
            if usable_position(pos):
                pos = np.asarray(pos, dtype=float)[:3]
                xyz[i] = mne_xyz_to_pipeline_xyz(pos) if from_mne else pos
                break
    return xyz


def mne_xyz_to_pipeline_xyz(pos):
    """Keep MNE/FIF RAS coords; MATLAB builds topoplot polar fields later."""
    return np.asarray(pos, dtype=float)[:3].copy()


def export_needs_orientation_rebuild(out_path, fmt):
    try:
        if fmt == "set":
            out_path = out_path.with_name(out_path.stem + "_chanlocs.mat")
        labels, xyz = load_exported_labels_pos(out_path)
        ok = heppy_front_back_landmarks_ok(labels, xyz)
        return ok is False
    except Exception:
        return True


def load_exported_labels_pos(path):
    if not path.exists():
        return [], np.empty((0, 3))
    S = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    if "labels" in S and "pos" in S:
        labels = [str(x) for x in np.ravel(S["labels"])]
        xyz = np.asarray(S["pos"], dtype=float)
        if xyz.ndim == 2 and xyz.shape[0] == 3 and xyz.shape[1] != 3:
            xyz = xyz.T
        return labels, xyz
    if "chanlocs" in S:
        chanlocs = np.ravel(S["chanlocs"])
        labels = [str(getattr(c, "labels", "")) for c in chanlocs]
        xyz = np.array(
            [[float(getattr(c, "X", np.nan)), float(getattr(c, "Y", np.nan)), float(getattr(c, "Z", np.nan))] for c in chanlocs],
            dtype=float,
        )
        return labels, xyz
    return [], np.empty((0, 3))


def heppy_front_back_landmarks_ok(labels, xyz):
    by_key = {normalise_channel_label(label): i for i, label in enumerate(labels)}
    front = by_key.get("c18")
    back = by_key.get("a25")
    if front is None or back is None or xyz.shape[0] <= max(front, back):
        return None
    delta = np.asarray(xyz[front], dtype=float)[:3] - np.asarray(xyz[back], dtype=float)[:3]
    if not np.all(np.isfinite(delta)):
        return False
    return bool(delta[1] > 0 and abs(delta[1]) > abs(delta[0]))


def usable_position(pos):
    if pos is None:
        return False
    pos = np.asarray(pos, dtype=float)
    return pos.size >= 3 and np.all(np.isfinite(pos[:3])) and np.linalg.norm(pos[:3]) > 0


def normalise_channel_label(label):
    s = "".join(ch for ch in str(label).strip().lower() if ch.isalnum())
    for prefix in ("eeg", "channel", "chan"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    if s.isdigit():
        s = s.lstrip("0") or "0"
    return s


def matlab_chanlocs(labels, xyz):
    chanlocs = np.empty((1, len(labels)), dtype=[("labels", "O"), ("X", "O"), ("Y", "O"), ("Z", "O")])
    for i, label in enumerate(labels):
        chanlocs[0, i] = (label, float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]))
    return chanlocs


def write_mat_eeg(raw, out_path, reference_positions=None):
    labels = raw.ch_names
    xyz = channel_positions(raw, reference_positions)
    chanlocs = matlab_chanlocs(labels, xyz)
    data = raw.get_data().astype(np.float32, copy=False)
    sio.savemat(
        out_path,
        {
            "EEG": {"data": data, "srate": float(raw.info["sfreq"]), "chanlocs": chanlocs},
            "data": data,
            "sfreq": float(raw.info["sfreq"]),
            "srate": float(raw.info["sfreq"]),
            "labels": np.array(labels, dtype=object),
            "pos": xyz,
            "chanlocs": chanlocs,
        },
    )
    return int(np.sum(np.all(np.isfinite(xyz), axis=1)))


def write_chanloc_sidecars(raw, stem, reference_positions=None):
    labels = raw.ch_names
    xyz = channel_positions(raw, reference_positions)
    chanlocs = matlab_chanlocs(labels, xyz)

    sio.savemat(str(stem) + "_chanlocs.mat", {"chanlocs": chanlocs, "labels": np.array(labels, dtype=object), "pos": xyz})
    with open(str(stem) + "_electrodes.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["name", "x", "y", "z"])
        for label, pos in zip(labels, xyz):
            w.writerow([label, *pos])
    return int(np.sum(np.all(np.isfinite(xyz), axis=1)))


if __name__ == "__main__":
    main()
