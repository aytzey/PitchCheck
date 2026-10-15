"""WS12: deterministic controls, signed contrasts and bounded literature priors (CPU only)."""
import json
import os

os.environ["TRIBE_ALLOW_MOCK"] = "1"

import numpy as np
import pytest

from tribe_service import llm_layer as llm, persuasion_features as pf, roi_features
from tribe_service import specificity as sp
from tribe_service.research_synthesis import build_tribe_synthesis

MESSAGE = "You smoke 20 cigarettes a day, mostly when talking on the phone."
PERSONA = "You smoke 20 cigarettes a day, mostly when talking on the phone."
SPANS = [
    {"span": "20 cigarettes a day", "category": "a lot of cigarettes", "persona_fact": "20 cigarettes a day"},
    {"span": "when talking on the phone", "category": "during daily routines", "persona_fact": "when talking on the phone"},
]


def contrast(**kwargs):
    return {"available": True, "grounded": True, "medial_z": 3.0, "direction": "higher",
            "encoder": "meta-llama/Llama-3.2-3B", "text_feature_compatible": True,
            "replaced": [{**SPANS[0], "grounded": True}], "top_parcels": [], **kwargs}


def test_chua_pair_and_ungrounded_control():
    twin = sp.build_specificity_twin(MESSAGE, PERSONA, SPANS)
    assert twin["available"] and twin["grounded"]
    assert twin["twin"] == "You smoke a lot of cigarettes, mostly during daily routines."
    assert twin["n_words"] == [12, 10]
    assert not sp.build_specificity_twin(MESSAGE, "Engineering manager", SPANS[:1])["grounded"]


@pytest.mark.parametrize("rows,reason", [
    ([{**SPANS[0], "span": "not in this text"}], "span_not_verbatim"),
    ([{**SPANS[0], "category": "30 cigarettes a day"}], "category_adds_detail"),
    ([{**SPANS[0], "category": "using HubSpot every day"}], "category_adds_detail"),
    ([{**SPANS[0], "category": "HubSpot"}], "category_adds_detail"),
    ([{**SPANS[0], "category": "may smoke many cigarettes"}], "certainty_changed"),
    ([{"span": "You smoke 20 cigarettes a day", "category": "They smoke a lot of cigarettes", "persona_fact": ""}], "category_adds_detail"),
    ([{"span": "You smoke 20 cigarettes a day", "category": "they smoke a lot of cigarettes", "persona_fact": ""}], "person_changed"),
    ([{**SPANS[0], "category": "cigarettes"}], "length_mismatch"),
    ([{"span": "mostly", "category": "often", "persona_fact": ""}], "no_specific_detail_removed"),
    ([{"span": "You", "category": "you", "persona_fact": "You"}], "no_specific_detail_removed"),
    (SPANS[:1] * 2, "overlapping_spans"),
    (SPANS * 3, "too_many_spans"),
    ([None], "invalid_spans"),
    ([{**SPANS[0], "category": ""}], "invalid_spans"),
    ([], "no_specific_detail"),
])
def test_twin_rejections(rows, reason):
    assert sp.build_specificity_twin(MESSAGE, "smoke 20 cigarettes a day when talking on the phone", rows) == {"available": False, "reason": reason}


@pytest.mark.parametrize("message,reason", [("A short message", "too_short"), ("word " * 251, "too_long"),
    ("Bu mesaj sizin için özellikle hazırlandı ve günlük çalışma planınıza uygun bir çözüm sunar.", "non_english_unsupported")])
def test_preflight_skips_without_llm(monkeypatch, message, reason):
    monkeypatch.setattr(llm, "_openrouter_enabled", lambda: True)
    monkeypatch.setattr(llm, "_post_refine_chat", lambda *a, **k: pytest.fail("LLM must not run"))
    assert sp.make_specificity_twin(message, PERSONA)["reason"] == reason


def test_twin_retry_and_llm_errors(monkeypatch):
    monkeypatch.setattr(llm, "_openrouter_enabled", lambda: True)
    responses = iter([json.dumps({"spans": [None]}), json.dumps({"spans": SPANS})])
    prompts = []
    def post(system, user, model, temperature):
        prompts.append(user)
        return next(responses)
    monkeypatch.setattr(llm, "_post_refine_chat", post)
    assert sp.make_specificity_twin(MESSAGE, PERSONA)["available"]
    assert len(prompts) == 2 and "invalid_spans" in prompts[1]
    assert sp.make_specificity_twin(MESSAGE, PERSONA)["reason"] == "llm_failed"
    monkeypatch.setattr(llm, "_openrouter_enabled", lambda: False)
    assert sp.make_specificity_twin(MESSAGE, PERSONA)["reason"] == "llm_unavailable"


def test_signed_contrast_uniform_mock_and_missing_assets(monkeypatch):
    assets = roi_features.load_roi_assets()
    assert assets is not None and set(sp.SPECIFICITY_ROI_KEYS) <= assets[3].keys()
    labels, _, _, locations, _ = assets
    a = np.full((5, 20484), -1.0)
    b = a.copy()
    for key in sp.SPECIFICITY_ROI_KEYS:
        b[:, locations[key]] = -1.2
    twin = sp.build_specificity_twin(MESSAGE, PERSONA, SPANS)
    monkeypatch.setattr(sp, "score_text", lambda text: b)
    result = sp.measure_specificity(MESSAGE, twin, a)
    assert result["available"] and result["medial_z"] > 0
    assert result["rois"]["dmpfc_chua2011"]["z"] > 0
    assert result["rois"]["l_angular_chua2011"]["role"] == "lateral_comparison_not_self_related"
    assert result["control_text"] == twin["twin"] and "twin" not in result
    assert len(result["top_parcels"]) == 3 and "self_value_axis_delta" not in result
    monkeypatch.setattr(sp, "score_text", lambda text: a - .2)
    assert sp.measure_specificity(MESSAGE, twin, a)["reason"] == "uniform_change"
    assert sp.measure_specificity(MESSAGE, twin, np.ones((5, 20)))["reason"] == "roi_unavailable"
    monkeypatch.setattr(roi_features, "load_roi_assets", lambda: None)
    assert sp.measure_specificity(MESSAGE, twin, a)["reason"] == "roi_unavailable"


@pytest.mark.parametrize("z,grounded,expected", [(0.9, True, 0), (3, True, 3), (10, True, 3), (-2, True, -1.5), (3, False, 0)])
def test_prior_dead_zone_and_cap(monkeypatch, z, grounded, expected):
    assert pf.specificity_prior_points(contrast(medial_z=z, grounded=grounded)) == expected
    monkeypatch.setattr(pf, "SPECIFICITY_SCORE_WEIGHT", 0)
    assert pf.specificity_prior_points(contrast(medial_z=z)) == 0


@pytest.mark.parametrize("multipliers,factor", [({}, 1), ({"dmpfc_chua2011": .25}, .625),
    ({"dmpfc_chua2011": 0, "precuneus_chua2011": 0}, 0)])
def test_calibration_multiplier_shadow_and_neural_only(monkeypatch, multipliers, factor):
    from tribe_service.tests.test_llm_layer import SAMPLE_MESSAGE, SAMPLE_PERSONA, SAMPLE_NEURAL_SIGNALS, SAMPLE_RAW_FEATURES
    monkeypatch.setattr(llm, "_call_openrouter", lambda *a, **k: None)
    monkeypatch.setattr(roi_features, "roi_multipliers", lambda encoder: multipliers)
    base = llm.interpret_persuasion(SAMPLE_MESSAGE, SAMPLE_PERSONA, "email", SAMPLE_NEURAL_SIGNALS, SAMPLE_RAW_FEATURES)
    result = llm.interpret_persuasion(SAMPLE_MESSAGE, SAMPLE_PERSONA, "email", SAMPLE_NEURAL_SIGNALS, SAMPLE_RAW_FEATURES,
                                    specificity_contrast=contrast())
    r = result["robustness"]
    assert r["specificity_contrast"]["prior_points"] == 3 * factor
    assert round(r["score_without_specificity_prior"], 1) == base["robustness"]["final_score"]
    assert result["persuasion_score"] == round(r["score_without_specificity_prior"] + 3 * factor)
    assert r["evidence_score"] == base["robustness"]["evidence_score"]
    assert r["score_terms"]["specificity_prior"]["value"] == 3 * factor


def test_none_parity_and_safe_prompt():
    from tribe_service.tests.test_llm_layer import SAMPLE_MESSAGE, SAMPLE_PERSONA, SAMPLE_NEURAL_SIGNALS
    args = (SAMPLE_MESSAGE, SAMPLE_PERSONA, "email", SAMPLE_NEURAL_SIGNALS)
    assert llm._build_user_prompt(*args) == llm._build_user_prompt(*args, specificity_contrast=None)
    prompt = llm._build_user_prompt(*args, specificity_contrast=contrast(control_text="DO_NOT_SEND_CONTROL", rois={"DO_NOT_SEND_ROI": {"z": 9001}}))
    assert "## Specificity Contrast" in prompt and "Do not change persuasion_score or context_fit" in prompt
    assert "DO_NOT_SEND_CONTROL" not in prompt and "DO_NOT_SEND_ROI" not in prompt


@pytest.mark.parametrize("z,grounded,level,expected", [(3, True, None, 3), (-3, True, None, -3),
    (3, False, None, 0), (3, True, "generic", 0), (3, True, "category", 0), (3, True, "recipient_specific", 3)])
def test_llm_path_and_ws06_soft_gate(z, grounded, level, expected):
    from tribe_service.tests.test_llm_layer import SAMPLE_NEURAL_SIGNALS
    context = {k: {"score": 75} for k in llm.CONTEXT_FIT_KEYS}
    if level is not None:
        context["personalization_level"] = level
    result = llm._calibrate_result({"persuasion_score": 75, "context_fit": context}, neural_signals=SAMPLE_NEURAL_SIGNALS,
                                   persuasion_evidence={}, llm_used=True, specificity_contrast=contrast(medial_z=z, grounded=grounded))
    r = result["robustness"]
    assert r["specificity_contrast"]["prior_points"] == expected
    assert result["persuasion_score"] == round(r["score_without_specificity_prior"] + expected)
    assert result["context_fit"] == context


def test_score_weight_zero_keeps_shadow_without_guardrail(monkeypatch):
    from tribe_service.tests.test_llm_layer import SAMPLE_NEURAL_SIGNALS
    monkeypatch.setattr(pf, "SPECIFICITY_SCORE_WEIGHT", 0)
    result = llm._calibrate_result({}, neural_signals=SAMPLE_NEURAL_SIGNALS, persuasion_evidence={},
                                   llm_used=False, specificity_contrast=contrast())
    r = result["robustness"]
    assert r["specificity_contrast"]["available"] and r["specificity_contrast"]["prior_points"] == 0
    assert result["persuasion_score"] == round(r["score_without_specificity_prior"])
    assert "specificity_contrast_literature_prior_applied" not in r["guardrails_applied"]


@pytest.mark.parametrize("z,grounded,key", [(2, True, "specificity_strength"), (-2, True, "specificity_flat"), (2, False, "specificity_unverified")])
def test_synthesis_rules_and_canonical_sources(z, grounded, key):
    result = build_tribe_synthesis("", {}, specificity_contrast=contrast(medial_z=z, grounded=grounded))
    item = next(i for i in result["items"] if i["key"] == key)
    assert item["axis"] == "cortical_roi" and item["roi"] == "specificity_contrast"
    assert set(item["source_keys"]) <= {s["key"] for s in pf.RESEARCH_SOURCES}


@pytest.mark.parametrize("jev", [True, False])
def test_refine_retention_penalty_and_payload_isolation(monkeypatch, jev):
    from tribe_service.tests.test_ws09 import measurement
    rows = [measurement(), measurement("c1"), measurement("c2")]
    rows[0]["message"] = MESSAGE
    rows[1]["message"] = "You smoke 20 cigarettes a day, so consider the benefits of quitting."
    rows[2]["message"] = "You smoke a lot of cigarettes, so consider the benefits of quitting."
    reviews = [{"id": r["id"], "supported": True, "intent_preserved": True, "recipient_respected": True,
                "voice_preserved": True, "context_fit": {k: 40 if r["id"] == "original" else 75 for k in llm.CONTEXT_FIT_KEYS}} for r in rows]
    payloads = []
    def post(system, user, *args, **kwargs):
        payloads.append(user)
        return json.dumps({"evaluations": reviews})
    monkeypatch.setattr(llm, "_post_refine_chat", post)
    def jev_reviews(message, persona, platform, measurements, strategy, answers):
        payloads.append(json.dumps([llm._jev_response_measurement(r) for r in measurements]))
        return reviews, "test", {"choice": "c2"}
    monkeypatch.setattr(llm, "_jev_refinement_reviews", jev_reviews)
    result = {"model": "test", **({"decision_strategy": {"choices": {}, "instructions": []}} if jev else {})}
    c = contrast(control_text="SECRET_INTERNAL_CONTROL")
    guide = llm.select_tribe_refinement(MESSAGE, PERSONA, "general", [], result, rows, specificity=c)["tribe_guidance"]
    assert guide["selected_id"] == "c1"
    assert guide["evaluations"][1]["specificity_retention"]["dropped"] == []
    dropped = guide["evaluations"][2]
    assert dropped["specificity_retention"]["dropped"] == [SPANS[0]["span"]]
    assert guide["evaluations"][1]["selection_score"] - dropped["selection_score"] == 2
    assert "SECRET_INTERNAL_CONTROL" not in json.dumps(payloads)
    assert "control_text" not in json.dumps(guide["specificity"])
    monkeypatch.setattr(llm, "SPECIFICITY_REFINE_PENALTY", 0)
    disabled = llm.select_tribe_refinement(MESSAGE, PERSONA, "general", [], result, rows, specificity=c)
    assert "specificity_retention" not in disabled["tribe_guidance"]["evaluations"][1]
