"""Additional Alexander-Bloch-style parcel robustness nulls; NumPy statistics only.

Prepare once with the offline research venv (nilearn/nibabel); the resulting NPZ
is consumed without those packages. No spin result changes a primary verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def nearest(centroids, rotation):
    return np.argmax(centroids @ (centroids @ rotation).T, axis=1)


def rotations(centroids, hemispheres, n=1000, seed=0):
    centroids, hemispheres = np.asarray(centroids, float), np.asarray(hemispheres)
    if (centroids.shape != (len(hemispheres), 3) or set(hemispheres) != {0, 1}
            or not np.isfinite(centroids).all() or n < 1
            or not np.allclose(np.linalg.norm(centroids, axis=1), 1)):
        raise ValueError("Need unit sphere centroids, two hemispheres and positive spin count")
    rng, reflection = np.random.default_rng(seed), np.diag([-1., 1., 1.])
    indices = [np.flatnonzero(hemispheres == h) for h in (0, 1)]
    result = np.empty((n, len(centroids)), dtype=np.int32)
    for i in range(n):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q *= np.sign(np.diag(r))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        for h, rotation in enumerate((q, reflection @ q @ reflection)):
            ids = indices[h]
            result[i, ids] = ids[nearest(centroids[ids], rotation)]
    return result


def prepare(output):
    output = Path(output).resolve()
    if output.suffix != ".npz" or output == ROOT or ROOT in output.parents or output.exists():
        raise ValueError("Use a new .npz output outside the repository")
    from nilearn.datasets import fetch_surf_fsaverage
    import nibabel as nib
    from tribe_service.roi_features import load_roi_assets

    assets = load_roi_assets()
    if assets is None:
        raise ValueError("WS07 atlas required")
    labels = assets[0]
    fs = fetch_surf_fsaverage("fsaverage5", data_dir=str(output.parent / "nilearn_data"))
    paths = [Path(fs[f"sphere_{h}"]) for h in ("left", "right")]
    spheres = [np.asarray(nib.load(p).agg_data()[0], float) for p in paths]
    if any(s.shape != (10242, 3) for s in spheres):
        raise ValueError("Expected fsaverage5 sphere coordinates")
    sphere = np.concatenate(spheres)
    centroids = np.array([sphere[labels == p].mean(0) for p in range(1, 401)])
    centroids /= np.linalg.norm(centroids, axis=1)[:, None]
    hemispheres = np.repeat([0, 1], 200)
    if np.any(labels[:10242] > 200) or np.any((labels[10242:] > 0) & (labels[10242:] <= 200)):
        raise ValueError("Atlas hemisphere ordering differs from Schaefer-400")
    assignments = rotations(centroids, hemispheres)
    metadata = {"method": "Haar SO(3), reflected RH rotation, within-hemisphere nearest rotated centroid",
                "n_spins": 1000, "seed": 0, "atlas_sha256": assets[4],
                "sphere_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                "sphere_source": "nilearn.datasets.fetch_surf_fsaverage('fsaverage5')",
                "duplicate_fraction_mean": float(np.mean([1 - len(np.unique(a)) / 400 for a in assignments])),
                "scope": "additional robustness only; nearest assignment permits repeated parcels"}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, assignments=assignments, metadata=json.dumps(metadata))
    return metadata


def load(path, atlas_sha):
    with np.load(path, allow_pickle=False) as data:
        assignments, metadata = data["assignments"], json.loads(str(data["metadata"]))
    if (metadata.get("atlas_sha256") != atlas_sha or metadata.get("n_spins") != 1000
            or metadata.get("seed") != 0 or assignments.shape != (1000, 400)
            or assignments.dtype.kind not in "iu" or np.any(assignments < 0) or np.any(assignments >= 400)
            or np.any(assignments[:, :200] >= 200) or np.any(assignments[:, 200:] < 200)):
        raise ValueError("Spin cache does not match the atlas / fixed 1000-spin seed-0 protocol")
    return assignments, metadata


def roi_weights(labels, vertices):
    # shortcut: partial locations use whole parcels; use vertex spins for subparcel inference.
    weights = np.bincount(np.asarray(labels)[vertices], minlength=401)[1:].astype(float)
    if weights.shape != (400,) or not weights.sum():
        raise ValueError("ROI must intersect cortical parcels")
    return weights / weights.sum()


def _dz_columns(values):
    mean, sd = values.mean(0), values.std(0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, mean / sd, np.where(mean == 0, 0., np.copysign(np.inf, mean)))


def roi_null(deltas, weights, assignments, alternative):
    deltas, weights = np.asarray(deltas, float), np.asarray(weights, float)
    if (deltas.ndim != 2 or deltas.shape[0] < 2 or deltas.shape[1] != len(weights)
            or not np.isfinite(deltas).all() or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1)
            or alternative not in {"greater", "less"}):
        raise ValueError("Need paired parcel deltas, normalized ROI weights and a direction")
    observed = float(_dz_columns(deltas @ weights))
    null = _dz_columns(np.einsum("nsp,p->ns", deltas[:, assignments], weights))
    direction = 1 if alternative == "greater" else -1
    return {"parcel_d_z": observed if np.isfinite(observed) else None,
            "spin_p": float((1 + np.count_nonzero(direction * null >= direction * observed - 1e-12)) / (1 + len(null))),
            "n_spins": len(null), "alternative": alternative,
            "statistic": "d_z of vertex-count-weighted intersecting parcels; partial locations are approximated"}


def rankdata(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 else 0.0


def correlation_null(prediction, target, labels, masks, assignments):
    prediction, target, labels = np.asarray(prediction), np.asarray(target), np.asarray(labels)
    if (prediction.shape != labels.shape or target.shape != labels.shape
            or not np.isfinite(prediction).all() or not np.isfinite(target[labels > 0]).all()):
        raise ValueError("Finite aligned cortical maps required")
    counts = np.bincount(labels, minlength=401)[1:]
    pm = np.bincount(labels, weights=prediction, minlength=401)[1:] / np.maximum(counts, 1)
    result = {}
    for key, mask in masks.items():
        mask = np.asarray(mask, bool) & (labels > 0)
        n = np.bincount(labels[mask], minlength=401)[1:]
        keep = n > 0
        if keep.sum() < 3:
            raise ValueError("Need at least three target parcels")
        tm = np.bincount(labels[mask], weights=target[mask], minlength=401)[1:] / np.maximum(n, 1)
        row = {"n_parcels": int(keep.sum())}
        for kind, transform in (("pearson", np.asarray), ("spearman", rankdata)):
            fixed = transform(tm[keep])
            observed = _corr(transform(pm[keep]), fixed)
            null = [_corr(transform(pm[spin][keep]), fixed) for spin in assignments]
            row.update({f"{kind}_r": observed,
                        f"{kind}_spin_p": float((1 + np.count_nonzero(np.asarray(null) >= observed - 1e-12)) / (1 + len(null)))})
        result[key] = row
    return result


def suite_null(rows, sidecar, assets, assignments):
    from scripts.research.run_contrasts import paired_rows
    result = []
    for contrast in sidecar["contrasts"]:
        pairs = [(a, b) for a, b in paired_rows(rows, contrast)
                 if a.get("parcels") is not None and b.get("parcels") is not None]
        if len(pairs) < sidecar["minimum_pairs"]:
            continue
        deltas = np.array([np.subtract(a["parcels"], b["parcels"]) for a, b in pairs])
        for roi in contrast["rois"]:
            vertices = assets[2].get(roi, assets[3].get(roi))
            if roi == "insula_right":
                vertices = assets[2]["insula"][assets[2]["insula"] >= 10242]
            if vertices is not None:
                result.append({"contrast": contrast["name"], "roi": roi,
                    **roi_null(deltas, roi_weights(assets[0], vertices), assignments, contrast["alternative"])})
    return result


def selftest():
    rng = np.random.default_rng(7)
    left = rng.normal(size=(200, 3)); left /= np.linalg.norm(left, axis=1)[:, None]
    points = np.concatenate((left, left * [-1, 1, 1]))
    hemispheres = np.repeat([0, 1], 200)
    spins = rotations(points, hemispheres)
    assert np.array_equal(spins, rotations(points, hemispheres))
    assert np.array_equal(nearest(left, np.eye(3)), np.arange(200))
    assert np.array_equal(spins[:, :200] + 200, spins[:, 200:])
    smooth = points[:, 2]
    maps = smooth + rng.normal(0, .2, (20, 400))
    weights = np.zeros(400); weights[np.argmax(smooth)] = 1
    assert roi_null(maps, weights, spins, "greater")["spin_p"] < .05
    result = correlation_null(smooth, smooth, np.arange(1, 401), {"all": np.ones(400, bool)}, spins)
    assert np.isclose(result["all"]["spearman_r"], 1)
    assert result["all"]["spearman_spin_p"] <= 2 / 1001
    assert np.array_equal(rankdata([4, 1, 1]), [3, 1.5, 1.5])
    print("parcel_spin selftest: 7 checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--prepare", type=Path, help="New NPZ output, outside repo")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.prepare:
        print(json.dumps(prepare(args.prepare), indent=2))
    else:
        parser.error("Choose --prepare or --selftest")


if __name__ == "__main__":
    main()
