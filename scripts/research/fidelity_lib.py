"""Small offline fMRI helpers; run --selftest in the research venv."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "docs/validation/ws11_prereg.md"
N_VERTICES = 20484


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def outside_repo(path):
    path = Path(path).expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("Research data, predictions and outputs must be outside the repository")
    return path


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def read_ascii_gifti(path):
    root = ET.parse(path).getroot()
    arrays = list(root.iter("DataArray"))
    if len(arrays) != 1 or arrays[0].get("Encoding") != "ASCII" or arrays[0].get("Dim0") != "163842":
        raise ValueError(f"Expected one ASCII 163842-vertex GIFTI: {path}")
    data = np.array(arrays[0].findtext("Data", "").split(), dtype=np.float32)
    if data.shape != (163842,) or not np.isfinite(data).all():
        raise ValueError(f"Invalid GIFTI values: {path}")
    return data


def fsavg5(paths_lr):
    if len(paths_lr) != 2:
        raise ValueError("Supply left then right surface")
    return np.concatenate([read_ascii_gifti(p)[:10242] for p in paths_lr])


def colcorr(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or len(a) < 2 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Correlation needs equal finite arrays and >=2 samples")
    a, b = a - a.mean(0), b - b.mean(0)
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    return np.divide((a * b).sum(0), den, out=np.zeros_like(den), where=den > 0)


def repeat_ceiling(repeats):
    """(runs, subjects, time, quantities): single-presentation group reliability."""
    repeats = np.asarray(repeats)
    if repeats.ndim != 4 or min(repeats.shape[:2]) < 1 or len(repeats) < 2:
        raise ValueError("Need >=2 repeats with the same subjects and time samples")
    groups = repeats.mean(1)
    rel = np.mean([colcorr(groups[a], groups[b]) for a, b in itertools.combinations(range(len(groups)), 2)], axis=0)
    return np.sqrt(np.clip(rel, 0, 1)), rel


def circ_null(pred, meas, n=1000, min_shift=20, seed=0):
    if n < 1 or min_shift < 1 or len(meas) <= 2 * min_shift:
        raise ValueError("Too few samples for the registered circular-shift null")
    r = colcorr(pred, meas)
    shifts = np.random.default_rng(seed).integers(min_shift, len(meas) - min_shift + 1, n)
    counts = sum(colcorr(np.roll(pred, int(k), 0), meas) >= r for k in shifts)
    return r, (1 + counts) / (1 + n)


def synthetic_to_real(row_t, onsets):
    onsets = np.asarray(onsets, dtype=float)
    if onsets.ndim != 1 or len(onsets) < 2 or not np.isfinite(onsets).all() or (np.diff(onsets) < 0).any():
        raise ValueError("Need finite, ordered real word onsets")
    return np.interp(row_t, np.arange(len(onsets)) * .40, onsets)


def load_atlas(annotations=None):
    """K2 loader wins; scratch 7N annotations are the pre-merge fallback."""
    import sys
    sys.path.insert(0, str(ROOT))
    try:
        from tribe_service.roi_features import load_roi_assets
    except ModuleNotFoundError as exc:
        if exc.name != "tribe_service.roi_features":
            raise
        assets = None
    else:
        assets = load_roi_assets()
    if assets is not None:
        return assets[0], assets[1], {"source": "ws07", "atlas_sha256": assets[4]}
    if not annotations or len(annotations) != 2:
        raise ValueError("WS07 assets unavailable: supply --annotations LEFT RIGHT")
    from nibabel.freesurfer.io import read_annot
    labels, names = [], ["medial_wall"]
    for hemi, path in enumerate(annotations):
        raw, _, parcel_names = read_annot(outside_repo(path))
        if raw.shape != (10242,) or len(parcel_names) != 201:
            raise ValueError("Expected Schaefer-400 fsaverage5 annotations")
        labels.append(np.where(raw > 0, raw + hemi * 200, 0))
        names.extend(name.decode() for name in parcel_names[1:])
    labels = np.concatenate(labels)
    if not np.array_equal(np.unique(labels), np.arange(401)):
        raise ValueError("Expected 400 unique parcel labels and medial wall")
    return labels, np.array(names), {"source": "scratch_schaefer400_7n",
                                   "annotation_sha256": [sha256(p) for p in annotations]}


def selftest():
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "bad.gii"
        path.write_text('<GIFTI><DataArray Encoding="ASCII" Dim0="1"><Data>0</Data></DataArray></GIFTI>')
        try:
            read_ascii_gifti(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Wrong GIFTI shape was accepted")
    rng = np.random.default_rng(17)
    signal = rng.normal(size=(1000, 2))
    assert np.allclose(colcorr(signal, signal), 1)
    _, p = circ_null(signal, signal + rng.normal(scale=.1, size=signal.shape))
    assert (p < .05).all(), p
    _, p = circ_null(rng.normal(size=signal.shape), rng.normal(size=signal.shape))
    assert (p > .05).all(), p
    common = rng.normal(size=(20000, 1))
    repeats = common + rng.normal(size=(12, 3, 20000, 1))
    ceiling, rel = repeat_ceiling(repeats)
    assert abs(rel[0] - .75) < .05, rel  # group noise variance 1/3, signal variance 1
    assert np.allclose(ceiling ** 2, rel)
    print("fidelity_lib: 4 self-checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", required=True)
    parser.parse_args()
    selftest()
