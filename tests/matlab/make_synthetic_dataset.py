"""Synthetic EEG fixture with KNOWN microstate labels. No real or restricted data is read or written.

Writes, under ``--out`` (which must NOT be inside restricted real-study storage; checked with
``fsms.guard.assert_unrestricted``):

* ``eeg/<participant>_<condition>.mat``  in the FIF->MAT converter schema (EEG.data float32 channels x time,
  EEG.srate, EEG.chanlocs with MNE-RAS head coordinates, data, sfreq, srate, labels, pos, chanlocs);
* ``synthetic_manifest.csv``             participant, condition, group, file_path, seizure_type, hypo_hyper,
  source_raw_fif, crop_start_seconds, crop_end_seconds;
* ``truth/truth.json`` + ``truth/<participant>_<condition>_segments.csv``  the generating class map per
  participant and the generating class of every sample (so the correct labelling is known).

The data are noisy mixtures of six known topographies (the MetaMaps_2023_06 classes A,B,C,E,F,G sampled at
the synthetic electrodes, plus a small participant-specific perturbation) that switch every 60-120 ms with a
half-sine GFP envelope and random polarity, over white noise and a slow drift.  With ``--maps random`` the
class maps are unrelated random topographies instead, which the pre-specified labelling must REJECT.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from fsms.guard import assert_unrestricted  # noqa: E402

CLASSES = ("A", "B", "C", "E", "F", "G")
CH19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4",
        "T6", "O1", "O2"]
EXTRA_LABELS = ["Oz", "Fpz"]            # variant 'labels17': 21 scalp channels, >=16 template label matches
TEMPLATE_ALIAS = {"T3": "T7/T3", "T4": "T8/T4", "T5": "P7/T5", "T6": "P8/T6"}
TEMPLATE = REPO / "external" / "VG-microstates" / "MetaMaps_2023_06.set"


def load_template():
    S = loadmat(str(TEMPLATE), squeeze_me=True, struct_as_record=False)
    rec = S["msinfo"].MSMaps[6]
    maps = np.asarray(rec.Maps, dtype=float)
    labels = [str(x) for x in np.ravel(rec.Labels)]
    channels = [str(c.labels) for c in S["chanlocs"]]
    return maps, labels, channels


def template_maps_for(channels, classes=CLASSES):
    maps, labels, template_channels = load_template()
    index = {name: i for i, name in enumerate(template_channels)}
    columns = []
    for name in channels:
        key = TEMPLATE_ALIAS.get(name, name)
        if key not in index:
            raise KeyError(f"Channel {name} is not an electrode of the MetaMaps template")
        columns.append(index[key])
    out = []
    for cls in classes:
        row = maps[labels.index(cls), columns]
        row = row - row.mean()
        out.append(row / np.linalg.norm(row))
    return np.array(out)


def montage_positions(channels):
    import mne
    montage = mne.channels.make_standard_montage("standard_1020")
    positions = montage.get_positions()["ch_pos"]
    return np.array([positions[name] for name in channels], dtype=float)


def push_apart(maps, max_correlation, iterations=2000, step=0.5):
    """Nudge maps apart until every pair has |r| <= max_correlation (keeps each close to its original).

    The MetaMaps classes C and E are almost collinear (|r| ~ 0.9) on a 19-channel montage, so a clustering
    fit cannot separate them; the synthetic ground truth must be identifiable, otherwise a mismatch would
    reflect the fit and not the labelling rule.
    """
    maps = np.array(maps, dtype=float)
    k = len(maps)
    for _ in range(iterations):
        r = maps @ maps.T
        if np.abs(r - np.eye(k)).max() <= max_correlation + 1e-12:
            break
        gradient = np.zeros_like(maps)
        for i in range(k):
            for j in range(k):
                if i != j and abs(r[i, j]) > max_correlation:
                    gradient[i] -= np.sign(r[i, j]) * maps[j] * (abs(r[i, j]) - max_correlation)
        maps += step * gradient
        maps -= maps.mean(axis=1, keepdims=True)
        maps /= np.linalg.norm(maps, axis=1, keepdims=True)
    else:
        raise RuntimeError("could not decorrelate the generating maps")
    return maps


def random_maps(rng, n_channels, count=6):
    maps = rng.normal(size=(count, n_channels))
    maps -= maps.mean(axis=1, keepdims=True)
    return maps / np.linalg.norm(maps, axis=1, keepdims=True)


def simulate_record(rng, maps, n_channels_total, n_samples, sfreq, noise_sigma, scale):
    n_classes, n_signal = maps.shape
    starts, ends, classes = [], [], []
    t, previous = 0, -1
    lo, hi = int(0.060 * sfreq), int(0.120 * sfreq)
    signal = np.zeros((n_signal, n_samples))
    while t < n_samples:
        length = int(rng.integers(lo, hi + 1))
        choices = [k for k in range(n_classes) if k != previous]
        cls = int(rng.choice(choices))
        end = min(n_samples, t + length)
        window = np.sin(np.pi * (np.arange(end - t) + 0.5) / length)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        amplitude = float(rng.lognormal(0.0, 0.2))
        signal[:, t:end] = sign * amplitude * np.outer(maps[cls], window)
        starts.append(t)
        ends.append(end)
        classes.append(cls)
        previous = cls
        t = end
    noise = rng.normal(scale=noise_sigma, size=(n_signal, n_samples))
    drift = np.cumsum(rng.normal(size=(n_signal, n_samples)), axis=1)
    drift = 5.0 * noise_sigma * drift / (np.abs(drift).max(axis=1, keepdims=True) + 1e-12)
    data = scale * (signal + noise + drift)
    extra = n_channels_total - n_signal
    if extra:
        data = np.vstack([data, scale * rng.normal(scale=noise_sigma, size=(extra, n_samples))])
    return data.astype(np.float32), np.array(starts), np.array(ends), np.array(classes)


def matlab_chanlocs(labels, xyz):
    chanlocs = np.empty((1, len(labels)), dtype=[("labels", "O"), ("X", "O"), ("Y", "O"), ("Z", "O")])
    for i, label in enumerate(labels):
        chanlocs[0, i] = (label, float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]))
    return chanlocs


def write_mat(path, data, sfreq, labels, xyz):
    chanlocs = matlab_chanlocs(labels, xyz)
    savemat(path, {
        "EEG": {"data": data, "srate": float(sfreq), "chanlocs": chanlocs},
        "data": data, "sfreq": float(sfreq), "srate": float(sfreq),
        "labels": np.array(labels, dtype=object), "pos": xyz, "chanlocs": chanlocs,
    })


def make_dataset(out_dir, seed=20260101, n_participants=12, sfreq=256.0, maps_mode="template",
                 channels_mode="ten20_19", noise_sigma=0.15, durations=None, no_positions=False,
                 edge_cases=False, max_class_correlation=0.55):
    out_dir = assert_unrestricted(out_dir, "synthetic fixture generation")
    durations = durations or {"II": 180.0, "preict": 150.0, "postict": 90.0}
    rng = np.random.default_rng(seed)
    eeg_dir, truth_dir = out_dir / "eeg", out_dir / "truth"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    signal_channels = CH19 + (EXTRA_LABELS if channels_mode == "labels17" else [])
    retention = None
    if maps_mode == "template":
        template_sampled = template_maps_for(signal_channels)
        base_maps = push_apart(template_sampled, max_class_correlation)
        retention = np.diag(base_maps @ template_sampled.T).tolist()
    elif maps_mode == "random":
        base_maps = random_maps(np.random.default_rng(seed + 1), len(signal_channels))
    else:
        raise ValueError(maps_mode)

    groups = ["FS"] * 5 + ["ES"] * 5 + ["DD"] * max(0, n_participants - 10)
    groups = (groups + ["FS"] * n_participants)[:n_participants]
    specs = []
    for i in range(n_participants):
        has_ear = channels_mode == "ten20_19" and i >= n_participants - 4
        per_condition = {k: v for k, v in durations.items() if not (k == "postict" and i % 4)}
        specs.append((f"SYN{i + 1:02d}", groups[i], per_condition, has_ear, i))
    if edge_cases:
        # SYN13: a single record far too short -> excluded (RECORD_TOO_FEW_GFP_PEAKS)
        # SYN14: two short records that individually pass but together fall below the participant minimum
        #        -> excluded (PARTICIPANT_TOO_FEW_GFP_PEAKS)
        # SYN15: 45-s records: enough peaks but shorter than one 60-s window -> retained, no exported windows
        base = n_participants
        specs.append((f"SYN{base + 1:02d}", "FS", {"II": 1.0}, False, base))
        specs.append((f"SYN{base + 2:02d}", "ES", {"II": 2.5, "preict": 2.5}, False, base + 1))
        specs.append((f"SYN{base + 3:02d}", "FS", {"II": 45.0, "preict": 45.0}, False, base + 2))

    rows, truth = [], {"classes": list(CLASSES), "channels": signal_channels, "seed": seed,
                       "maps_mode": maps_mode, "channels_mode": channels_mode, "sfreq": sfreq,
                       "edge_cases": bool(edge_cases), "max_class_correlation": max_class_correlation,
                       "template_retention": retention,
                       "base_maps": base_maps.tolist(), "participants": {}, "records": {},
                       "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for participant, group, per_condition, has_ear, i in specs:
        perturbed = base_maps + 0.25 * rng.normal(size=base_maps.shape) / np.sqrt(base_maps.shape[1])
        perturbed -= perturbed.mean(axis=1, keepdims=True)
        perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
        truth["participants"][participant] = perturbed.tolist()
        labels = signal_channels + (["A1", "A2"] if has_ear else [])
        if no_positions:
            xyz = np.full((len(labels), 3), np.nan)
        else:
            xyz = montage_positions(labels)
        hypo = "hypo" if i % 2 else "hyper"
        for condition, duration in per_condition.items():
            n_samples = int(round(duration * sfreq))
            data, starts, ends, classes = simulate_record(rng, perturbed, len(labels), n_samples, sfreq,
                                                         noise_sigma, scale=3.5e-5)
            path = eeg_dir / f"{participant}_{condition}.mat"
            write_mat(path, data, sfreq, labels, xyz)
            stem = f"{participant}_{condition}"
            with (truth_dir / f"{stem}_segments.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["start_sample", "end_sample_exclusive", "class"])
                for s_, e_, c_ in zip(starts, ends, classes):
                    writer.writerow([int(s_), int(e_), CLASSES[int(c_)]])
            crop_start = {"II": 100.0, "preict": 1000.0, "postict": 1200.0}[condition]
            rows.append({
                "participant": participant, "condition": condition, "group": group,
                "file_path": str(path), "seizure_type": "focal" if i % 3 else "generalised",
                "hypo_hyper": hypo if group != "ES" else "", "source_raw_fif": f"synthetic_{stem}.fif",
                "crop_start_seconds": crop_start, "crop_end_seconds": crop_start + duration,
            })
            truth["records"][stem] = {"segments_csv": f"{stem}_segments.csv", "n_samples": n_samples,
                                      "participant": participant, "condition": condition,
                                      "n_channels": len(labels)}
    manifest = out_dir / "synthetic_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (truth_dir / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--participants", type=int, default=12)
    parser.add_argument("--maps", choices=("template", "random"), default="template")
    parser.add_argument("--channels", choices=("ten20_19", "labels17"), default="ten20_19")
    parser.add_argument("--noise-sigma", type=float, default=0.15)
    parser.add_argument("--no-positions", action="store_true")
    parser.add_argument("--max-class-correlation", type=float, default=0.55)
    parser.add_argument("--edge-cases", action="store_true",
                        help="append SYN(n+1..n+3): too-short record, too-little-data participant, no-window records")
    args = parser.parse_args()
    manifest = make_dataset(args.out, args.seed, args.participants, maps_mode=args.maps,
                            channels_mode=args.channels, noise_sigma=args.noise_sigma,
                            no_positions=args.no_positions, edge_cases=args.edge_cases,
                            max_class_correlation=args.max_class_correlation)
    print(manifest)


if __name__ == "__main__":
    main()
