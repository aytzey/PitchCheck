"""Offline WS07 atlas builder. Requires nibabel==5.4.2, nilearn==0.14.1, numpy.

Derived from the planning proto3.py; CSV is the sole curated coordinate source.
Downloads only pinned CBIG annotations/license and nilearn's fsaverage5 meshes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import nibabel as nib
import nilearn
import numpy as np
from nilearn import datasets, surface

ROOT = Path(__file__).resolve().parents[1]
CBIG_SHA = "634f676630929a71297852d01dd92a287103e861"
BASE = f"https://raw.githubusercontent.com/ThomasYeoLab/CBIG/{CBIG_SHA}/"
ANNOT = "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage5/label/"
M305_TO_152 = np.array([[0.9975, -0.0073, 0.0176, -0.0429], [0.0146, 1.0009, -0.0024, 1.5496],
                        [-0.0130, -0.0093, 0.9971, 1.1840], [0, 0, 0, 1]])
OTHER_TO_TAL = np.array([[0.9357, 0.0029, -0.0072, -1.0423], [-0.0065, 0.9396, -0.0726, -1.3940],
                         [0.0103, 0.0752, 0.8967, 3.6475], [0, 0, 0, 1]])
ALLOWED = {
    "vmpfc_frontopolar": {"Default", "Limbic"}, "mofc": {"Default", "Limbic"},
    "dmpfc": {"Default"}, "insula": {"SalVentAttn"}, "r_ifg_ai": {"Default", "Limbic", "SalVentAttn"},
    "pcc_precuneus": {"Default", "Cont"}, "r_tpj": {"Cont", "Default", "TempPar"},
    "lateral_temporal": {"TempPar", "Default", "Limbic"},
    "l_ifg": {"Default", "Cont", "TempPar", "SalVentAttn"},
}
FLAGS = {
    "vmpfc_frontopolar": ["dropout_risk"], "mofc": ["dropout_risk", "image_analogy"],
    "dmpfc": ["between_subject_source", "social_content_confound"], "insula": ["image_price_analogy"],
    "pcc_precuneus": ["mixed_sources"], "r_tpj": ["box_centre_only"],
    "lateral_temporal": ["language_confound"], "l_ifg": ["opposite_constructs"],
}
FORBIDDEN = "self value reward trust engage social mentaliz persua attitude preference purchase salience conflict emotion affect motivation resistance doubt skeptic anger counterargu".split()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(cache: Path, output: Path, ui_output: Path):
    cache.mkdir(parents=True, exist_ok=True)
    downloads = {}

    def fetch(relative):
        dest = cache / Path(relative).name
        if not dest.exists():
            with urlopen(BASE + relative, timeout=60) as response:
                dest.write_bytes(response.read())
        downloads[relative] = {"url": BASE + relative, "sha256": sha(dest.read_bytes())}
        return dest

    def annotation(networks):
        arrays, names = [], ["MEDIAL_WALL"]
        for hemi, offset in (("lh", 0), ("rh", 200)):
            arr, _, raw_names = nib.freesurfer.read_annot(fetch(ANNOT + f"{hemi}.Schaefer2018_400Parcels_{networks}Networks_order.annot"))
            assert arr.shape == (10242,)
            arrays.append(np.where(arr > 0, arr + offset, 0).astype(np.int16))
            names.extend(name.decode().replace(f"{networks}Networks_", "") for name in raw_names[1:])
        return np.concatenate(arrays), np.array(names)

    labels, names = annotation(17)
    labels7, names7 = annotation(7)
    assert np.array_equal(np.unique(labels), np.arange(401))
    assert int((labels[:10242] == 0).sum()) == 870 and int((labels[10242:] == 0).sum()) == 873
    assert (labels[:10242] <= 200).all() and ((labels[10242:] == 0) | (labels[10242:] > 200)).all()
    network_ids = {name: i for i, name in enumerate(("MW", "Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"))}
    network7 = np.zeros(401, dtype=np.int8)
    for parcel in range(1, 401):
        match = np.unique(labels7[labels == parcel])
        assert match.size == 1 and (labels7 == match[0]).sum() == (labels == parcel).sum()
        network7[parcel] = network_ids[names7[match[0]].split("_")[1]]
    fs = datasets.fetch_surf_fsaverage("fsaverage5", data_dir=str(cache / "nilearn_data"))
    coords = lambda kind: np.vstack([surface.load_surf_mesh(fs[f"{kind}_{h}"]).coordinates for h in ("left", "right")])
    mid = 0.5 * (coords("pial") + coords("white"))
    to305, fromtal = np.linalg.inv(M305_TO_152), np.linalg.inv(OTHER_TO_TAL)
    stems = np.array(["MW" if not i else name.split("_")[1].rstrip("ABC") for i, name in enumerate(names)])
    rois = {key: set() for key in ALLOWED}
    locations, rows = {}, []
    csv_path = ROOT / "scripts/roi_peaks.csv"
    for row in csv.DictReader(csv_path.open()):
        log = {**row, "assignments": []}
        rows.append(log)
        if not row["x"]:
            log["reason"] = "non_coordinate_record"
            continue
        p = np.array([float(row[k]) for k in ("x", "y", "z")])
        hemis = (0, 1) if row["hemi_rule"] in {"midline", "mirror"} else (int(p[0] > 0),)
        for hemi in hemis:
            q = p.copy()
            if row["hemi_rule"] in {"midline", "mirror"}:
                q[0] = abs(q[0]) * (-1 if hemi == 0 else 1)
            q152 = (fromtal @ np.r_[q, 1])[:3] if row["space"] == "Talairach_AFNI" else q
            off = hemi * 10242
            assignments = []
            for tag, point in (("aff", (to305 @ np.r_[q152, 1])[:3]), ("noaff", q152)):
                distance = np.linalg.norm(mid[off:off + 10242] - point, axis=1)
                vertex = int(distance.argmin()) + off
                parcel = int(labels[vertex])
                record = {"method": tag, "hemi": "LR"[hemi], "distance_mm": round(float(distance[vertex - off]), 5),
                          "nearest_vertex": vertex, "nearest_parcel": str(names[parcel]), "parcel": str(names[parcel])}
                assignments.append(record)
                if row["role"] == "location":
                    if tag == "aff":
                        record["reason"] = "dropped_far_from_surface" if distance.min() > 6 else "accepted"
                        idx = np.flatnonzero((distance <= 10) & (labels[off:off + 10242] > 0)) + off
                        for key in row["roi"].split(";"):
                            locations.setdefault(key, set()).update(idx.tolist() if distance.min() <= 6 else [])
                    continue
                if parcel == 0:
                    record["reason"] = "medial_wall"
                elif distance.min() > 8:
                    record["reason"] = "dropped_far_from_surface"
                elif row["role"] == "primary":
                    if stems[parcel] not in ALLOWED[row["roi"]]:
                        record["network_mismatch"] = True
                        allowed = np.isin(stems[labels[off:off + 10242]], sorted(ALLOWED[row["roi"]])) & (labels[off:off + 10242] > 0)
                        vertex = int(np.where(allowed, distance, np.inf).argmin()) + off
                        parcel = int(labels[vertex])
                        record.update(parcel=str(names[parcel]), distance_mm=round(float(distance[vertex - off]), 5))
                        if distance[vertex - off] > 8:
                            record["reason"] = "network_mismatch_no_nearby_allowed_vertex"
                    if "reason" not in record:
                        record["reason"] = "accepted"
                        rois[row["roi"]].add(parcel)
                else:
                    record["reason"] = row["role"]
            unstable = assignments[0]["parcel"] != assignments[1]["parcel"]
            for record in assignments:
                record["unstable_assignment"] = unstable
            log["assignments"].extend(assignments)

    arrays = {"labels": labels, "parcel_names": names, "network7": network7}
    provenance = {"cbig_sha": CBIG_SHA, "downloads": downloads, "numpy": np.__version__, "nilearn": nilearn.__version__,
                  "nibabel": nib.__version__, "mni305_to_mni152": M305_TO_152.tolist(), "icbm_other2tal": OTHER_TO_TAL.tolist(),
                  "mni152_to_mni305": to305.tolist(), "talairach_afni_to_mni152": fromtal.tolist(),
                  "coordinate_method": "nilearn fsaverage5 midthickness + MNI152→305; APPROXIMATION of TRIBE's non-public NLin2009cAsym hybrid meshes",
                  "csv_sha256": sha(csv_path.read_bytes()), "peaks": rows, "rois": {}, "locations": {}}
    centroids, used = {}, set()
    for key, parcels in rois.items():
        idx = np.flatnonzero(np.isin(labels, sorted(parcels)))
        assert idx.size >= 30 and not used.intersection(idx.tolist()), key
        used.update(idx.tolist())
        arrays[f"roi_{key}"] = idx.astype(np.int16)
        flags = list(FLAGS.get(key, []))
        if any(a["unstable_assignment"] for row in rows if row["roi"] == key and row["role"] == "primary" for a in row["assignments"]):
            flags.append("unstable_assignment")
        provenance["rois"][key] = {"parcels": names[sorted(parcels)].tolist(), "n_vertices": int(idx.size),
                                   "left": int((idx < 10242).sum()), "right": int((idx >= 10242).sum()), "flags": flags,
                                   "network_composition": {str(n): int((stems[labels[idx]] == n).sum()) for n in sorted(set(stems[labels[idx]]))}}
        # Actual parcel-union centroids projected onto the desktop's top-down schematic.
        centroids[key] = [[round(float(v), 4) for v in mid[part].mean(axis=0)]
                          for part in (idx[idx < 10242], idx[idx >= 10242]) if part.size]
    for key, selected in sorted(locations.items()):
        idx = np.array(sorted(selected), dtype=np.int16)
        ids, counts = np.unique(labels[idx], return_counts=True)
        provenance["locations"][key] = {"n_vertices": int(idx.size), "left": int((idx < 10242).sum()), "right": int((idx >= 10242).sum()),
                                        "dominant_parcels": {str(names[i]): int(n) for i, n in sorted(zip(ids, counts), key=lambda pair: (-pair[1], pair[0]))},
                                        "peaks": [r for r in rows if r["role"] == "location" and key in r["roi"].split(";")],
                                        "reason": "accepted" if idx.size >= 20 else "fewer_than_20_vertices"}
        if idx.size >= 20:
            arrays[f"loc_{key}"] = idx
    assert all(not any(word in key for word in FORBIDDEN) for key in (*rois, *locations))
    license_text = fetch("LICENSE.md").read_text()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "roi_atlas_v1.npz", **arrays)
    provenance["npz_sha256"] = sha((output / "roi_atlas_v1.npz").read_bytes())
    (output / "roi_provenance_v1.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    (output / "LICENSE-Schaefer2018.txt").write_text(license_text + "\nvertex selections computed with nilearn fsaverage5 meshes; no mesh data redistributed\n")
    ui_output.write_text(json.dumps(centroids, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"sha256": provenance["npz_sha256"], "rois": {k: v["n_vertices"] for k, v in provenance["rois"].items()},
                      "locations": {k: v["n_vertices"] for k, v in provenance["locations"].items()}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "tribe_service/roi_assets")
    parser.add_argument("--ui-output", type=Path, default=ROOT / "shared/roi-centroids.json")
    args = parser.parse_args()
    build(args.cache_dir, args.output, args.ui_output)
