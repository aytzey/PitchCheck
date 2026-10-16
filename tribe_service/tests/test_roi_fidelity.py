"""WS11 trust boundary, no-op compatibility and the reserved WS07 multiplier hook."""
import copy
import json
import math
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from tribe_service import roi_fidelity as fidelity

ROOT = Path(__file__).resolve().parents[2]
ROI_KEYS = {"vmpfc_frontopolar", "mofc", "dmpfc", "insula", "r_ifg_ai", "pcc_precuneus", "r_tpj", "lateral_temporal", "l_ifg"}
LOCATION_KEYS = {"mpfc_falk2010", "falk2010jocn_core", "tp_l", "tp_r", "ipl_ifg_how_gt_why_vezich2017_l",
                 "dmpfc_chua2011", "precuneus_chua2011", "rmpfc_chua2009", "vmpfc_chua2009", "l_angular_chua2011",
                 "ifg47_l_chua2011", "pifg_vezich2017", "ripl_vezich2017"}


@pytest.fixture
def atlas():
    return (np.zeros(20484, dtype=np.int16), np.array(["medial_wall"]),
            dict.fromkeys(ROI_KEYS, np.arange(30)), dict.fromkeys(LOCATION_KEYS, np.arange(20)), "a" * 16)


def row(r_over_ceiling=.06, status="weak"):
    return {"status": status, "r": .03, "ceiling": .5, "r_over_ceiling": r_over_ceiling,
            "ci": [0., .2], "q": .2, "r_over_ceiling_real_timing": .1}


@pytest.fixture
def asset():
    return {"schema": 1, "validated": False,
            "threshold": {"status": "unvalidated_prior", "r_over_ceiling": .3, "floor_factor": .25},
            "atlas_sha256": "a" * 64, "prereg_sha256": "b" * 64,
            "provenance": {"text_model": fidelity.TEXT_MODEL, "timing": "pitchcheck_direct_0.40s",
                           "tribe_hf_sha": "c" * 40, "tribev2_commit": "d" * 40,
                           "pitchcheck_sha": "e" * 40, "date": "2026-10-16"},
            "source": {"id": "lebel2023_val", "modality": "english_story_listening", "stories": sorted(fidelity.VAL_STORIES),
                       "subjects_in_training": True, "stimuli_in_training": False},
            "rois": {"mofc": row()}, "locations": {"mpfc_falk2010": row(), "dmpfc_chua2011": row(), "precuneus_chua2011": row()},
            "networks": {}, "hv2025": {}}


@pytest.fixture(autouse=True)
def isolated_asset(tmp_path, monkeypatch):
    monkeypatch.setattr(fidelity, "ROI_ASSET_DIR", tmp_path)
    monkeypatch.delenv("PITCHCHECK_ROI_FIDELITY_SCALING", raising=False)
    fidelity.roi_fidelity.cache_clear()
    yield tmp_path
    fidelity.roi_fidelity.cache_clear()


def ship(directory, asset, atlas, monkeypatch):
    (directory / "roi_fidelity.json").write_text(json.dumps(asset))
    monkeypatch.setattr(fidelity, "load_roi_assets", lambda: atlas)


def test_no_asset_and_no_ws07_is_noop():
    assert fidelity.fidelity_factor("mofc") == (1., 1.)
    assert fidelity.external_fidelity("mofc")["status"] == "not_benchmarked"
    assert not fidelity.fidelity_robustness({})["asset_present"]


def test_asset_without_atlas_is_noop(isolated_asset, asset, monkeypatch):
    ship(isolated_asset, asset, None, monkeypatch)
    assert fidelity.fidelity_factor("mofc") == (1., 1.)


def test_measured_asset_selects_v24(isolated_asset, asset, atlas, monkeypatch):
    from tribe_service.llm_layer import _augment_persuasion_evidence
    ship(isolated_asset, asset, atlas, monkeypatch)
    evidence = _augment_persuasion_evidence("A short test pitch", "A test recipient", "email", None, None)
    assert evidence["methodology_version"] == "neural_only_v2.4"


def test_annotations_survive_report_schema():
    from tribe_service.engine import analyze_predictions
    from tribe_service.schemas import FmriOutput
    _, fmri, _ = analyze_predictions(np.ones((5, 20484), dtype=np.float32))
    fmri["roi_summary"] = {"rois": {"mofc": {"z": 1.}}, "locations": {"mpfc_falk2010": {"z": 2.}}}
    fidelity.annotate_fidelity(fmri)
    report = FmriOutput(**fmri).model_dump()
    for section, key in (("rois", "mofc"), ("locations", "mpfc_falk2010")):
        info = report["roi_summary"][section][key]["external_fidelity"]
        assert info["status"] == "not_benchmarked" and info["factor"] == 1.


@pytest.mark.parametrize("ratio,status,expected", [(.06, "weak", .25), (.15, "weak", .5), (.45, "ok", 1.),
                                                  (-.4, "weak", .25), (.06, "unmeasurable", 1.)])
def test_factors(isolated_asset, asset, atlas, monkeypatch, ratio, status, expected):
    asset["rois"]["mofc"] = row(ratio, status)
    ship(isolated_asset, asset, atlas, monkeypatch)
    assert fidelity.fidelity_factor("mofc") == (expected, expected)


@pytest.mark.parametrize("mode", ["off", "shadow", "garbage"])
def test_modes_keep_would_be(isolated_asset, asset, atlas, monkeypatch, mode):
    ship(isolated_asset, asset, atlas, monkeypatch)
    monkeypatch.setenv("PITCHCHECK_ROI_FIDELITY_SCALING", mode)
    assert fidelity.fidelity_factor("mpfc_falk2010") == (1., .25)
    assert fidelity.external_fidelity("mpfc_falk2010")["would_be"] == .25


@pytest.mark.parametrize("raw", ["{", "[]", "null", '{"schema":1}', '"text"'])
def test_bad_json_does_not_crash(isolated_asset, raw, caplog):
    (isolated_asset / "roi_fidelity.json").write_text(raw)
    assert fidelity.fidelity_factor("mofc") == (1., 1.)
    assert "ignoring" in caplog.text


@pytest.mark.parametrize("bad", [{"status": "unknown"}, {"r_over_ceiling": math.nan},
                                {"r_over_ceiling": True}, {"ceiling": math.inf}, {"ci": [2., 1.]}, {"q": -1},
                                {"status": []}, {"r_over_ceiling": 10 ** 1000}])
def test_invalid_row_ignored(isolated_asset, asset, atlas, monkeypatch, bad, caplog):
    asset["rois"]["mofc"].update(bad)
    ship(isolated_asset, asset, atlas, monkeypatch)
    assert fidelity.fidelity_factor("mofc") == (1., 1.)
    assert "ignoring" in caplog.text.lower() or "invalid fidelity row" in caplog.text.lower()


def test_unknown_key_ignored_and_strict_schema_rejects(isolated_asset, asset, atlas, monkeypatch, caplog):
    asset["rois"]["unknown"] = row()
    with pytest.raises(ValueError):
        fidelity.validate_fidelity_asset(asset, atlas)
    ship(isolated_asset, asset, atlas, monkeypatch)
    assert fidelity.fidelity_factor("unknown") == (1., 1.)
    assert fidelity.fidelity_factor("mofc") == (.25, .25)
    assert "unknown" in caplog.text


@pytest.mark.parametrize("mutation", ["validated", "atlas", "threshold", "source", "provenance"])
def test_metadata_rejected(isolated_asset, asset, atlas, monkeypatch, mutation):
    if mutation == "validated":
        asset["validated"] = True
    elif mutation == "atlas":
        asset["atlas_sha256"] = "f" * 64
    elif mutation == "threshold":
        asset["threshold"]["floor_factor"] = 0
    elif mutation == "source":
        asset["source"]["stimuli_in_training"] = True
    else:
        asset["provenance"]["timing"] = "other"
    ship(isolated_asset, asset, atlas, monkeypatch)
    with pytest.raises(ValueError):
        fidelity.validate_fidelity_asset(asset, atlas)
    assert fidelity.fidelity_factor("mofc") == (1., 1.)


def test_schema_and_optional_real_shipped_asset(asset, atlas):
    assert set(fidelity.validate_fidelity_asset(asset, atlas)) == {"mofc", "mpfc_falk2010", "dmpfc_chua2011", "precuneus_chua2011"}
    path = ROOT / "tribe_service/roi_assets/roi_fidelity.json"
    if path.exists():
        real_atlas = fidelity.load_roi_assets()
        assert real_atlas is not None, "Shipped fidelity requires WS07 atlas"
        fidelity.validate_fidelity_asset(json.loads(path.read_text()), real_atlas)
    template = ROOT / "tribe_service/roi_assets/hv2025_dogmatic_minus_suggestive_tribe_fsaverage5.npy"
    if template.exists():
        values = np.load(template, allow_pickle=False)
        assert values.shape == (20484,) and values.dtype == np.float16 and np.isfinite(values).all()
        sidecar = json.loads(template.with_suffix(".json").read_text())
        assert set(sidecar) == {"image_id", "collection", "licence", "n", "stimuli_set", "spearman_r", "p", "verdict", "loo_auc", "scope", "prereg_sha256"}
        assert not re.search(r"reactance|anger|conflict|doubt", json.dumps(sidecar), re.I)


def test_ws07_reserved_hook_consumes_locations_without_renormalizing(isolated_asset, asset, atlas, monkeypatch):
    """Exercise the same-module hook WS07 actually reserves for WS09/12 consumers."""
    ship(isolated_asset, asset, atlas, monkeypatch)
    module = ModuleType("tribe_service.roi_features")
    module.load_roi_assets = lambda: atlas
    # WS07 looks up fidelity_factor in its own globals, then multiplies WS08.
    exec("def roi_multipliers(encoder):\n    f = globals().get('fidelity_factor')\n    return {k: (f(k)[0] if f else 1.) for k in ('mofc', 'mpfc_falk2010', 'dmpfc_chua2011', 'precuneus_chua2011')}\n", module.__dict__)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    fmri = {"text_feature_compatible": False, "roi_summary": {"rois": {"mofc": {"z": 1.}},
                                                            "locations": {"mpfc_falk2010": {"z": 2.}}}}
    fidelity.annotate_fidelity(fmri)
    m = module.roi_multipliers("Hermes")
    assert .5 * m["mpfc_falk2010"] == .125  # WS09 term retains its separate encoder multiplier.
    assert 1.5 * (m["dmpfc_chua2011"] + m["precuneus_chua2011"]) / 2 == .375
    assert sum(.2 * m[k] * 2 for k in m) / sum(.2 for _ in m) == .5  # K4 denominator stays fixed.
    assert fmri["roi_summary"]["locations"]["mpfc_falk2010"]["external_fidelity"]["factor"] == .25
    assert fidelity.fidelity_robustness(fmri)["encoder_match"] is False
    assert fidelity.fidelity_robustness(fmri)["asset_sha256"]
    assert "0.03" not in fidelity.fidelity_prompt(fmri)  # No raw r/ROI values enter a prompt.


def test_no_asset_report_preserves_pin(monkeypatch):
    from tribe_service import llm_layer as llm
    from tribe_service.tests.test_llm_layer import (SAMPLE_MESSAGE, SAMPLE_PERSONA, SAMPLE_PLATFORM,
                                                 SAMPLE_NEURAL_SIGNALS, SAMPLE_RAW_FEATURES)
    monkeypatch.setattr(llm, "_call_openrouter", lambda *a, **k: None)
    args = (SAMPLE_MESSAGE, SAMPLE_PERSONA, SAMPLE_PLATFORM, SAMPLE_NEURAL_SIGNALS, SAMPLE_RAW_FEATURES)
    on = llm.interpret_persuasion(*args)
    monkeypatch.setenv("PITCHCHECK_ROI_FIDELITY_SCALING", "off")
    off = llm.interpret_persuasion(*args)
    assert on["persuasion_score"] == off["persuasion_score"]
    a, b = copy.deepcopy(on), copy.deepcopy(off)
    a["robustness"].pop("roi_fidelity")
    b["robustness"].pop("roi_fidelity")
    assert a == b
    assert on["persuasion_evidence"]["methodology_version"] == "neural_only_v2.3"
    assert round(llm.neuro_axis_score_from_axes(llm.neuro_axes_from_analysis(SAMPLE_NEURAL_SIGNALS)), 2) == 66.30
    pin = llm._calibrate_result({"persuasion_score": 78}, neural_signals=SAMPLE_NEURAL_SIGNALS,
                                persuasion_evidence={"warnings": []}, llm_used=True)
    assert pin["persuasion_score"] == 73


def test_real_word_times_and_product_cache_path():
    import pandas as pd
    from scripts.research.tribe_predict import predict, timed_events
    events = pd.DataFrame({"type": ["Word", "Word", "Text"], "start": [0., .4, 0.],
                           "duration": [.32, .32, .72], "stop": [.32, .72, .72]})
    times = [[0., .5], [1., .2]]
    adjusted = timed_events(events, times)
    assert adjusted["start"].tolist() == [0., 1., 0.]
    assert adjusted["stop"].tolist() == [.5, 1.2, 1.2]
    assert events["start"].tolist() == [0., .4, 0.]
    with pytest.raises(ValueError):
        timed_events(events, [[1., .1]])
    def score(text, *, retry_index):
        assert retry_index == 0
        assert fake.build_direct_text_events(text)["start"].tolist() == [0., 1., 0.]
        return np.ones((2, 20484)), {}
    fake = SimpleNamespace(build_direct_text_events=lambda text: events, _score_text_once=score,
                           get_model=lambda: SimpleNamespace(data=SimpleNamespace(text_feature=SimpleNamespace(infra=SimpleNamespace(cache_dict={})))) )
    assert predict(fake, "two words", times).shape == (2, 20484)


def test_registration_rejects_changed_protocol(monkeypatch):
    from scripts.research.tribe_predict import committed_file
    import subprocess
    protocol = ROOT / "docs/validation/ws11_prereg.md"
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: b"old protocol\n")
    with pytest.raises(ValueError, match="Commit registration"):
        committed_file(protocol)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: protocol.read_bytes())
    committed_file(protocol)
