import hashlib
import json

import numpy as np
import pytest

from tribe_service import engine, roi_features as roi

LOCATION_KEYS = {"mpfc_falk2010", "falk2010jocn_core", "tp_l", "tp_r", "ipl_ifg_how_gt_why_vezich2017_l",
                 "pifg_vezich2017", "ripl_vezich2017", "dmpfc_chua2011", "precuneus_chua2011",
                 "rmpfc_chua2009", "vmpfc_chua2009", "l_angular_chua2011", "ifg47_l_chua2011"}


@pytest.fixture(autouse=True)
def clear_caches():
    roi.load_roi_assets.cache_clear()
    roi.roi_calibration.cache_clear()
    yield
    roi.load_roi_assets.cache_clear()
    roi.roi_calibration.cache_clear()


def test_committed_asset_integrity_and_provenance():
    assets = roi.load_roi_assets()
    assert assets is not None and len(assets) == 5
    labels, names, rois, locations, sha = assets
    assert labels.shape == (20484,) and np.issubdtype(labels.dtype, np.integer)
    assert np.array_equal(np.unique(labels), np.arange(401))
    assert (labels[:10242] <= 200).all() and ((labels[10242:] == 0) | (labels[10242:] > 200)).all()
    assert (labels == 0).sum() == 1743
    assert set(rois) == {"vmpfc_frontopolar", "mofc", "dmpfc", "insula", "r_ifg_ai", "pcc_precuneus", "r_tpj", "lateral_temporal", "l_ifg"}
    assert set(locations) <= LOCATION_KEYS
    assert {"dmpfc_chua2011", "precuneus_chua2011", "tp_l", "tp_r"} <= set(locations)
    used = set()
    for key, idx in rois.items():
        assert idx.size >= 30 and not used.intersection(idx.tolist())
        used.update(idx.tolist())
    for idx in locations.values():
        assert idx.size >= 20
    for arr in (labels, names, *rois.values(), *locations.values()):
        assert not arr.flags.writeable
    for idx in (*rois.values(), *locations.values()):
        assert np.unique(idx).size == idx.size and idx.min() >= 0 and idx.max() < 20484 and (labels[idx] > 0).all()
    assert set(locations["tp_l"]) <= set(locations["falk2010jocn_core"])
    forbidden = "self value reward trust engage social mentaliz persua attitude preference purchase salience conflict emotion affect motivation resistance doubt skeptic anger counterargu".split()
    assert not any(word in key for key in (*rois, *locations) for word in forbidden)
    assert "average-subject" in roi.CAVEAT and "not this recipient" in roi.CAVEAT
    p = json.loads((roi.ROI_ASSET_DIR / "roi_provenance_v1.json").read_text())
    assert hashlib.sha256(roi.ASSET.read_bytes()).hexdigest() == p["npz_sha256"]
    assert sha == p["npz_sha256"][:16]
    with np.load(roi.ASSET, allow_pickle=False) as data:
        assert data["network7"].dtype == np.int8 and set(data["network7"]) == set(range(8))
        assert all(data[key].dtype != object for key in data.files)
    for key, idx in rois.items():
        assert p["rois"][key]["n_vertices"] == idx.size
    for row in p["peaks"]:
        if row["role"] in {"excluded", "null_record", "benchmark"}:
            assert all(a.get("reason") != "accepted" for a in row["assignments"])
    cooper = next(row for row in p["peaks"] if row["label"] == "unpublished PCC centroid")
    assert all(a["reason"] == "medial_wall" for a in cooper["assignments"])


@pytest.mark.parametrize("key", sorted(roi.ROI_KEYS))
def test_signed_geometry_and_affine_invariance(key):
    labels, _, rois, _, _ = roi.load_roi_assets()
    m = np.zeros((4, 20484))
    m[:, rois[key]] = 1
    p = rois[key].size / (labels > 0).sum()
    summary = roi.roi_summary(m)
    assert summary["rois"][key]["minus_cortex"] == pytest.approx(1 - p, abs=1e-5)
    z = summary["rois"][key]["z"]
    assert z == pytest.approx((1 - p) / np.sqrt(p * (1 - p)), abs=1e-4)
    assert roi.roi_summary(-m)["rois"][key]["z"] == -z
    assert roi.roi_summary(3 * m + 2)["rois"][key]["z"] == z


@pytest.mark.parametrize("m", [np.ones((5, 20)), np.ones(20484), np.ones((0, 20484)),
                              np.ones((4, 20484)), np.full((2, 20484), np.nan), np.full((2, 20484), np.inf)])
def test_invalid_or_degenerate_predictions_disable_overlay(m):
    assert roi.roi_summary(m) is None


def test_missing_and_corrupt_asset_disable_overlay(monkeypatch, tmp_path, caplog):
    asset = tmp_path / "missing.npz"
    monkeypatch.setattr(roi, "ASSET", asset)
    assert roi.load_roi_assets() is None
    asset.write_bytes(b"corrupt")
    roi.load_roi_assets.cache_clear()
    assert roi.load_roi_assets() is None
    assert "ROI atlas unavailable" in caplog.text


@pytest.mark.parametrize("key,broken", [("labels", np.ones(4, dtype=int)), ("roi_insula", np.array([-1])),
                                       ("roi_dmpfc", np.ones(40)), ("network7", np.full(401, 99))])
def test_malformed_arrays_fail_closed(monkeypatch, tmp_path, key, broken):
    with np.load(roi.ASSET, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays[key] = broken
    asset = tmp_path / "bad.npz"
    np.savez_compressed(asset, **arrays)
    monkeypatch.setattr(roi, "ASSET", asset)
    assert roi.load_roi_assets() is None


def test_native_python_overlays_and_json_types_match(monkeypatch):
    m = np.random.default_rng(7).normal(size=(6, 20484)).astype(np.float32)
    summary = roi.roi_summary(m)
    json.dumps(summary, allow_nan=False)
    expected = {"mean_signed", "minus_cortex", "z", "n_vertices"}
    assert set(next(iter(summary["rois"].values()))) == set(next(iter(summary["locations"].values()))) == expected
    for row in (*summary["rois"].values(), *summary["locations"].values()):
        assert type(row["n_vertices"]) is int
        assert all(type(row[k]) is float for k in ("z", "mean_signed", "minus_cortex"))
    fallback = engine.analyze_predictions(m)[1]["roi_summary"]
    assert fallback == engine.summarize_fmri_output(m)["roi_summary"] == summary
    monkeypatch.setattr(engine.native_core, "prediction_analysis", lambda *args: ({}, {}, {}))
    monkeypatch.setattr(engine.native_core, "summarize_fmri_output", lambda *args: {})
    assert engine.analyze_predictions(m)[1]["roi_summary"] == engine.summarize_fmri_output(m)["roi_summary"] == summary


def test_multiplier_hooks_share_roi_and_location_keys(monkeypatch):
    sha = roi.load_roi_assets()[4]
    monkeypatch.setattr(engine, "roi_reference", lambda encoder: {"atlas_sha256": sha, "checks": {
        "insula": {"weight_multiplier": 0.5}, "dmpfc_chua2011": {"weight_multiplier": 0.4}}}, raising=False)
    monkeypatch.setattr(roi, "fidelity_factor", lambda key: (0.5, 0.5), raising=False)
    multipliers = roi.roi_multipliers("test")
    assert multipliers["insula"] == 0.25 and multipliers["dmpfc_chua2011"] == 0.2
    monkeypatch.setenv("PITCHCHECK_ROI_INSILICO_TUNING", "0")
    assert roi.roi_multipliers("test")["insula"] == 0.5


def test_encoder_unit_calibration_and_reference_precedence(monkeypatch, tmp_path):
    from tribe_service.persuasion_features import ROI_PRIOR, roi_prior_from_summary
    monkeypatch.setattr(roi, "ROI_ASSET_DIR", tmp_path)
    data = {"encoder": "a/b", "atlas_sha256": roi.load_roi_assets()[4], "provenance": "unit_calibration:test",
            "centers": {k: 1.0 for k in ROI_PRIOR}, "ROI_Z_SCALE": 2.0}
    (tmp_path / "roi_unit_calibration_a__b.json").write_text(json.dumps(data))
    summary = {"rois": {k: {"z": 1.0} for k in ROI_PRIOR}}
    prior = roi_prior_from_summary(summary, encoder="a/b")
    assert prior["score"] == 50 and prior["ROI_Z_SCALE"] == 2 and prior["unit_calibration_provenance"] == "unit_calibration:test"
    assert roi_prior_from_summary(summary, encoder="different")["unit_calibration_provenance"].startswith("provisional_")
    roi.roi_calibration.cache_clear()
    monkeypatch.setattr(engine, "roi_reference", lambda encoder: {"atlas_sha256": data["atlas_sha256"], "unit_calibration": {**data, "ROI_Z_SCALE": 3}}, raising=False)
    assert roi.roi_calibration("a/b")["ROI_Z_SCALE"] == 3


@pytest.mark.parametrize('broken', [[], {'centers': []}, {'centers': {'insula': float('nan')}, 'ROI_Z_SCALE': 1, 'provenance': 'unit_calibration:x'},
                                    {'centers': {}, 'ROI_Z_SCALE': 0, 'provenance': 'unit_calibration:x'}])
def test_invalid_unit_artifacts_keep_provisional_defaults(monkeypatch, tmp_path, broken):
    monkeypatch.setattr(roi, 'ROI_ASSET_DIR', tmp_path)
    (tmp_path / 'roi_unit_calibration_test.json').write_text(json.dumps(broken))
    assert roi.roi_calibration('test') == {}
