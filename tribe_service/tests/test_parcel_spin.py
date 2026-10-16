import copy
import json

import numpy as np
import pytest

from scripts.research import parcel_spin as spin, run_contrasts as harness, fidelity_hv2025 as hv
from tribe_service.tests.fixtures.ws08_synthetic import contrasts, reference


def test_spatial_null_is_deterministic_hemisphere_safe_and_report_only(tmp_path):
    spin.selftest()
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(400, 3))
    centers /= np.linalg.norm(centers, axis=1)[:, None]
    indices = spin.rotations(centers, np.repeat([0, 1], 200))
    assert (indices[:, :200] < 200).all() and (indices[:, 200:] >= 200).all()
    rows, sidecar = contrasts()
    original = copy.deepcopy(rows)
    primary = harness.analyze_suite(rows, sidecar, reference(), True, B=50)
    labels = np.arange(1, 401)
    assets = (labels, [], {"target": np.array([0, 1])}, {"other": np.array([201])}, "atlas")
    extra = spin.suite_null(rows, sidecar, assets, indices)
    assert len(extra) == 2 and all(1 / 1001 <= r["spin_p"] <= 1 for r in extra)
    assert rows == original
    assert harness.analyze_suite(rows, sidecar, reference(), True, B=50) == primary
    path = tmp_path / "spin.npz"
    np.savez(path, assignments=indices, metadata=json.dumps({"atlas_sha256": "atlas", "n_spins": 1000, "seed": 0}))
    assert np.array_equal(spin.load(path, "atlas")[0], indices)
    with pytest.raises(ValueError, match="does not match"):
        spin.load(path, "wrong")
    with pytest.raises(ValueError, match="new .npz"):
        spin.prepare(path)
    with pytest.raises(ValueError, match="new .npz"):
        spin.prepare(tmp_path / "missing-extension")
    with pytest.raises(ValueError, match="Finite aligned"):
        spin.correlation_null(np.zeros(2), np.zeros(3), labels, {}, indices)
    conditions = ["D"] * 15 + ["S"] * 15 + ["long"] * 15 + ["short"] * 15
    manifest = {"stimuli_set": "category_faithful", "items": [
        {"id": str(i), "condition": condition, "sha256": f"{i:064x}",
         "n_words": 25 if condition == "long" else 12 if condition == "short" else 15 + i % 10}
        for i, condition in enumerate(conditions)]}
    target = np.linspace(-5, 5, 400)
    maps = rng.normal(size=(60, 400)) - (np.array(conditions) == "D")[:, None] * target
    names = np.array(["medial_wall"] + [f"17Networks_Default_{i}" for i in range(400)])
    baseline, template = hv.analyze(manifest, maps, target, labels, names, n_permutations=4)
    extra, spin_template = hv.analyze(manifest, maps, target, labels, names, n_permutations=4, spins=indices)
    assert template is spin_template is None and extra["verdict"] == "uninformative"
    assert extra.pop("parcel_spin")["maps"]["D-S Welch T"]["valid"]["n_parcels"] == 400
    assert extra == baseline
