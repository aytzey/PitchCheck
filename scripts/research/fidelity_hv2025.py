"""Offline HV2025 report and zero-weight TRIBE-space template; no inference here."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import re

import numpy as np

try:
    from .fidelity_lib import PREREG, N_VERTICES, fsavg5, load_atlas, outside_repo, sha256, write_json
    from .tribe_predict import check_registration, load_manifest
except ImportError:
    from fidelity_lib import PREREG, N_VERTICES, fsavg5, load_atlas, outside_repo, sha256, write_json
    from tribe_predict import check_registration, load_manifest

STIMULI_SETS = {"original_en", "original_es", "category_faithful", "tone_pairs"}
TEMPLATE = "hv2025_dogmatic_minus_suggestive_tribe_fsaverage5"
SCOPE = ("Similarity to a measured dogmatic-minus-suggestive group reading contrast "
         "(n=33 map; health messages; language likely Spanish). Not a mental-state measure.")


def masks_for(target, labels, names):
    valid = (labels > 0) & (target != 0) & np.isfinite(target)
    excluded = np.array([bool(re.search(r"_(Vis[^_]*|SomMot[^_]*)_", str(name))) for name in names])
    masks = {"valid": valid, "valid_minus_Vis_SomMot": valid & ~excluded[labels],
             "abs_T_gt_3.1": valid & (np.abs(target) > 3.1)}
    if any(mask.sum() < 3 for mask in masks.values()):
        raise ValueError("Insufficient target coverage for registered masks")
    return masks


def welch_t(a, b):
    return (a.mean(0) - b.mean(0)) / np.sqrt(np.maximum(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b), 1e-20))


def length_adjusted_t(maps, is_d, words):
    design = np.column_stack((np.ones(len(maps)), is_d, words))
    if np.linalg.matrix_rank(design) < 3:
        return None
    coef = np.linalg.lstsq(design, maps, rcond=None)[0]
    variance = np.square(maps - design @ coef).sum(0) / (len(maps) - 3)
    return coef[1] / np.sqrt(np.maximum(variance * np.linalg.inv(design.T @ design)[1, 1], 1e-20))


def correlations(pred, target, masks):
    from scipy.stats import rankdata
    result = {}
    for name, mask in masks.items():
        p, t = pred[mask].astype(float), target[mask].astype(float)
        def corr(a, b):
            a, b = a - a.mean(), b - b.mean()
            return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-20))
        result[name] = {"n": int(mask.sum()), "spearman_r": corr(rankdata(p), rankdata(t)), "pearson_r": corr(p, t)}
    return result


def anchored_template(pred, target, labels):
    counts = np.bincount(labels, minlength=401)
    pm = np.bincount(labels, weights=pred, minlength=401) / np.maximum(counts, 1)
    tm = np.bincount(labels, weights=target, minlength=401) / np.maximum(counts, 1)
    keep = (np.sign(pm) == np.sign(tm)) & (np.abs(tm) > 1)
    keep[0] = False
    return np.where(keep[labels], pred, 0)


def auc(is_d, scores):
    from scipy.stats import rankdata
    n = int(is_d.sum())
    return float((rankdata(scores)[is_d].sum() - n * (n + 1) / 2) / (n * (~is_d).sum()))


def loo_auc(maps, is_d, target, labels):
    scores = []
    for i, item in enumerate(maps):
        train = np.arange(len(maps)) != i
        t = welch_t(maps[train & is_d], maps[train & ~is_d])
        template = anchored_template(t, target, labels)
        scores.append(float(item @ template / max(np.linalg.norm(template), 1e-20)))
    return auc(is_d, scores)


def sign_agreement(pred, target, mask):
    return float((np.sign(pred[mask]) == np.sign(target[mask])).mean())


def label_nulls(is_d, pair_ids=None, n=10000, seed=0):
    if pair_ids is None:
        rng = np.random.default_rng(seed)
        for _ in range(n):
            yield rng.permutation(is_d)
    else:
        pairs = sorted(set(pair_ids))
        positions = [np.flatnonzero(np.array(pair_ids) == key) for key in pairs]
        for flips in itertools.product((False, True), repeat=len(pairs)):
            labels = is_d.copy()
            for flip, pos in zip(flips, positions):
                if flip:
                    labels[pos] = ~labels[pos]
            yield labels


def verdict(stimuli_set, r, p, sign_p, control_r):
    if r > 0 and p < .05 and sign_p < .05 and r > control_r:
        return "agree"
    return "uninformative" if stimuli_set == "category_faithful" else "no_agreement"


def validate_items(manifest):
    if manifest.get("stimuli_set") not in STIMULI_SETS:
        raise ValueError("Unknown preregistered stimulus set")
    items = manifest["items"]
    if Counter(row["condition"] for row in items) != {"D": 15, "S": 15, "long": 15, "short": 15}:
        raise ValueError("Need 15 D, 15 S and 15 long/short control items")
    for row in items:
        lo, hi = {"long": (25, 30), "short": (10, 14)}.get(row["condition"], (10, 30))
        if not lo <= row["n_words"] <= hi:
            raise ValueError("Word count outside preregistered bounds")
    ds = [row for row in items if row["condition"] in {"D", "S"}]
    if len({row["sha256"] for row in items}) != len(items):
        raise ValueError("Stimuli must be unique")
    if manifest["stimuli_set"] == "tone_pairs":
        groups = {}
        for row in ds:
            pair = row.get("pair_id")
            if not isinstance(pair, str) or not pair:
                raise ValueError("Tone-pair IDs are required")
            groups.setdefault(pair, []).append(row)
        if len(groups) != 15 or any(len(g) != 2 or {r["condition"] for r in g} != {"D", "S"}
                                    or g[0]["n_words"] != g[1]["n_words"] for g in groups.values()):
            raise ValueError("Need 15 word-matched D/S pairs")
    return ds


def analyze(manifest, maps, target, labels, names, *, n_permutations=10000):
    """Small permutation counts are for synthetic tests only; CLI fixes the registration."""
    ds = validate_items(manifest)
    ds_idx = [i for i, row in enumerate(manifest["items"]) if row["condition"] in {"D", "S"}]
    x = maps[ds_idx]
    is_d = np.array([r["condition"] == "D" for r in ds])
    pairs = [r["pair_id"] for r in ds] if manifest["stimuli_set"] == "tone_pairs" else None
    masks = masks_for(target, labels, names)
    t = welch_t(x[is_d], x[~is_d])
    observed = correlations(t, target, masks)
    mean_difference = correlations(x[is_d].mean(0) - x[~is_d].mean(0), target, masks)
    adjusted = length_adjusted_t(x, is_d, np.array([r["n_words"] for r in ds]))
    long = maps[[r["condition"] == "long" for r in manifest["items"]]]
    short = maps[[r["condition"] == "short" for r in manifest["items"]]]
    control = correlations(welch_t(long, short), target, masks)
    primary = "valid_minus_Vis_SomMot"
    r = observed[primary]["spearman_r"]
    agreement = sign_agreement(t, target, masks["abs_T_gt_3.1"])
    observed_auc = loo_auc(x, is_d, target, labels)
    counts = np.zeros(3, dtype=int)
    total = 0
    # Rebuild both the contrast and every LOO fold under the same label null.
    for perm in label_nulls(is_d, pairs, n_permutations):
        null_t = welch_t(x[perm], x[~perm])
        counts[0] += correlations(null_t, target, {primary: masks[primary]})[primary]["spearman_r"] >= r
        counts[1] += sign_agreement(null_t, target, masks["abs_T_gt_3.1"]) >= agreement
        counts[2] += loo_auc(x, perm, target, labels) >= observed_auc
        total += 1
    # Exact enumeration includes the observed assignment; Monte Carlo uses +1.
    ps = counts / total if pairs is not None else (counts + 1) / (total + 1)
    result = {"analysis_status": "complete", "prereg_sha256": sha256(PREREG),
              "stimuli_set": manifest["stimuli_set"], "metrics": observed,
              "raw_mean_difference": mean_difference,
              "length_adjusted": correlations(adjusted, target, masks) if adjusted is not None else {"status": "non_identifiable"},
              "long_minus_short": control, "sign_agreement": agreement,
              "p": float(ps[0]), "sign_p": float(ps[1]), "null_assignments": total, "seed": 0,
              "null_kind": "exact_paired_sign_flip" if pairs is not None else "random_balanced_labels",
              "loo_auc": {"auc": observed_auc, "p": float(ps[2])},
              "verdict": verdict(manifest["stimuli_set"], r, ps[0], ps[1], control[primary]["spearman_r"]),
              "template_written": bool(r > 0), "scope": SCOPE}
    if manifest["stimuli_set"] == "category_faithful":
        result["interpretation"] = "Conceptual replication; a fail is uninformative about TRIBE fidelity."
    return result, anchored_template(t, target, labels).astype(np.float16) if r > 0 else None


def prepare_target(surfaces, inverse, annotations=None):
    target, other = fsavg5(surfaces), fsavg5(inverse)
    if not np.allclose(target, -other, atol=1e-6, rtol=1e-6):
        raise ValueError("NeuroVault 886851 is not the inverse of 886850")
    labels, names, atlas_info = load_atlas(annotations)
    masks = masks_for(target, labels, names)
    info = {"analysis_status": "target_prepared_predictions_not_run", "prereg_sha256": sha256(PREREG),
            "image_id": 886850, "inverse_image_id": 886851, "collection": 17520,
            "licence": "CC0 (NeuroVault FAQ 13)", "n": {"neurovault": 33, "paper": 37, "osf_script": 35},
            "inverse_sign_check": True, "medial_wall_vertices": int((labels == 0).sum()),
            "expected_medial_wall_vertices": 1743, "expected_valid_vertices": 17934,
            "masks": {key: int(mask.sum()) for key, mask in masks.items()}, "atlas": atlas_info,
            "surface_sha256": {Path(p).name: sha256(p) for p in (*surfaces, *inverse)},
            "target_min": float(target.min()), "target_max": float(target.max()),
            "verdict": "pending_gpu", "scope": SCOPE}
    return target, labels, names, info


def render_report(result):
    body = ["# WS11 11.1 — HV2025 offline fidelity", "", f"Status: {result['analysis_status']}",
            f"Verdict: {result['verdict']}", f"Pre-registration SHA-256: {result['prereg_sha256']}", "",
            "No score or selection weight is assigned to this contrast.", "", result["scope"], ""]
    if "metrics" in result:
        body += ["| Mask | N | Spearman r | Pearson r |", "|---|---:|---:|---:|"]
        body += [f"| {key} | {v['n']} | {v['spearman_r']:.6f} | {v['pearson_r']:.6f} |" for key, v in result["metrics"].items()]
    body += ["", "Full metrics, nulls, hashes, length control and provenance are in the adjacent JSON.", ""]
    return "\n".join(body)


def selftest():
    rng = np.random.default_rng(11)
    labels = np.tile(np.arange(401), 2)
    target = np.linspace(-5, 5, len(labels))
    names = np.array(["medial_wall"] + [f"7Networks_LH_Default_PFC_{i}" for i in range(1, 401)])
    names[1] = "7Networks_LH_Vis_1"
    names[2] = "7Networks_LH_SomMot_1"
    masks = masks_for(target, labels, names)
    assert not masks["valid_minus_Vis_SomMot"][np.isin(labels, [0, 1, 2])].any()
    conditions = ["D"] * 15 + ["S"] * 15 + ["long"] * 15 + ["short"] * 15
    manifest = {"stimuli_set": "category_faithful", "items": [
        {"id": f"item{i}", "condition": condition, "sha256": f"{i:064x}",
         "n_words": (25 if condition == "long" else 12 if condition == "short" else 15 + i % 10)}
        for i, condition in enumerate(conditions)]}
    maps = rng.normal(size=(60, len(labels)))
    maps[:15] += target
    maps[15:30] -= target
    result, template = analyze(manifest, maps, target, labels, names, n_permutations=19)
    assert result["metrics"]["valid_minus_Vis_SomMot"]["spearman_r"] > .9
    assert result["null_assignments"] == 19 and result["loo_auc"]["auc"] > .95
    assert template.dtype == np.float16 and np.all(template[labels == 0] == 0)
    negative, template = analyze(manifest, -maps, target, labels, names, n_permutations=9)
    assert template is None and negative["verdict"] == "uninformative"
    labels_d = np.array([True, False, True, False])
    nulls = list(label_nulls(labels_d, ["a", "a", "b", "b"]))
    assert len(nulls) == 4 and all(n[:2].sum() == n[2:].sum() == 1 for n in nulls)
    assert len(list(label_nulls(labels_d, n=10000))) == 10000
    assert verdict("original_en", .2, .01, .01, .1) == "agree"
    assert verdict("original_en", .2, .01, .01, .3) == "no_agreement"
    assert verdict("category_faithful", -.2, .9, .9, .1) == "uninformative"
    assert np.isfinite(welch_t(np.ones((3, 2)), np.zeros((3, 2)))).all()
    assert length_adjusted_t(maps[:30], np.arange(30) < 15, np.ones(30)) is None
    print("fidelity_hv2025: 9 self-checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--surfaces", type=Path, nargs=2)
    parser.add_argument("--inverse", type=Path, nargs=2)
    parser.add_argument("--annotations", type=Path, nargs=2)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if not (args.surfaces and args.inverse and args.output):
        parser.error("--surfaces LEFT RIGHT, --inverse LEFT RIGHT and --output are required")
    output = outside_repo(args.output)
    surfaces = [outside_repo(p) for p in args.surfaces]
    inverse = [outside_repo(p) for p in args.inverse]
    target, labels, names, result = prepare_target(surfaces, inverse, args.annotations)
    output.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        if not args.lock:
            parser.error("--manifest requires --lock")
        manifest = load_manifest(args.manifest)
        check_registration(manifest, args.lock)
        if manifest.get("prereg_sha256") != sha256(PREREG) or manifest.get("stimulus_lock_sha256") != sha256(args.lock):
            raise ValueError("Predictions do not match the registered analysis")
        provenance = manifest.get("provenance", {})
        if (provenance.get("text_model") != "meta-llama/Llama-3.2-3B"
                or provenance.get("timing") != "pitchcheck_direct_0.40s"
                or any(not provenance.get(k) for k in ("tribe_hf_sha", "text_hf_sha", "tribev2_commit", "image_digest", "pitchcheck_sha", "packages"))):
            raise ValueError("HV analysis requires product-path Llama predictions with full provenance")
        maps = []
        for row in manifest["items"]:
            path = outside_repo(row["prediction_file"])
            if sha256(path) != row["prediction_sha256"]:
                raise ValueError("Prediction hash mismatch")
            pred = np.load(path, allow_pickle=False)
            if pred.ndim != 2 or pred.shape[0] < 1 or pred.shape[1] != N_VERTICES or not np.isfinite(pred).all():
                raise ValueError("Invalid signed vertex prediction matrix")
            maps.append(pred.mean(0, dtype=np.float64))
        analysis, template = analyze(manifest, np.array(maps), target, labels, names)
        result.update(analysis, provenance=provenance, stimulus_lock_sha256=sha256(args.lock))
        if template is not None:
            if not np.isfinite(template).all():
                raise ValueError("Template exceeds float16 range")
            np.save(output / f"{TEMPLATE}.npy", template)
            sidecar = {k: result[k] for k in ("image_id", "collection", "licence", "n", "prereg_sha256", "verdict", "loo_auc", "scope")}
            sidecar.update(stimuli_set=manifest["stimuli_set"], spearman_r=result["metrics"]["valid_minus_Vis_SomMot"]["spearman_r"], p=result["p"])
            write_json(output / f"{TEMPLATE}.json", sidecar)
        else:
            # Do not let an older positive-result template survive a rerun with r <= 0.
            for suffix in ("npy", "json"):
                (output / f"{TEMPLATE}.{suffix}").unlink(missing_ok=True)
    result["date"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "report.json", result)
    (output / "report.md").write_text(render_report(result))
    print(f"{result['verdict']}: {output / 'report.md'}")


if __name__ == "__main__":
    main()
