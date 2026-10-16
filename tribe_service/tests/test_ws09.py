"""WS09: report-only signed locations, bounded selection, and categorical hypotheses."""
import copy
import json
import math
import re

import pytest

from tribe_service import app, engine, llm_layer as llm, roi_features as roi
from tribe_service.research_synthesis import build_tribe_synthesis

ENCODER = "meta-llama/Llama-3.2-3B"
KEY, GUARD = llm.ROI_PRIOR_KEY, llm.ROI_ROUTE_GUARD_KEY
FLOORS = {KEY: 0.1, GUARD: 0.1, "floor_source": "default_prior"}


def measurement(id="original", delta=0):
    return {"id": id, "message": "Could we review the deployment workflow together?", "model_id": "tribe",
            "mode": "model", "text_feature_model": ENCODER, "text_feature_compatible": True,
            "voxel_count": 20484, "segments": 4, "temporal_trace": [1, 1, 1, 1],
            "temporal_trace_basis": "synthetic_word_order", "quality_weight": 1, "neural_score": 50,
            "response_features": {"focus_ratio": 0.3}, "roi_features": {KEY: -0.5 + delta, GUARD: 0},
            "roi_status": "ok", "roi_deltas": {KEY: delta, GUARD: 0}}


@pytest.mark.parametrize("delta,expected,reason", [
    (0, 0, "below_noise_floor"), (0.1, 0, "below_noise_floor"), (-0.1, 0, "below_noise_floor"),
    (0.15, 0.25, "applied"), (0.2, 0.5, "applied"), (2, 0.5, "applied"), (-2, -0.5, "applied"),
])
def test_dead_zone_and_signed_saturation(delta, expected, reason):
    term = llm._roi_prior_term(measurement("c1", delta), measurement(), FLOORS)
    assert term["roi_term"] == pytest.approx(expected)
    assert term["roi_reason"] == reason


def test_guards_quality_encoder_language_and_multipliers(monkeypatch):
    base, row = measurement(), measurement("c1", 0.2)
    assert llm._roi_prior_term(base, base, FLOORS)["roi_reason"] == "baseline"
    for status in ("runtime_not_comparable", "roi_unavailable"):
        assert llm._roi_prior_term({**row, "roi_status": status}, base, FLOORS)["roi_reason"] == status
    assert llm._roi_prior_term(row, base, {})["roi_reason"] == "no_noise_floor_for_runtime"
    assert llm._roi_prior_term({**row, "roi_deltas": {KEY: 0.2, GUARD: -0.2}}, base, FLOORS)["roi_reason"] == "why_how_route_trade"
    assert llm._roi_prior_term({**row, "quality_weight": 0.4}, base, FLOORS)["roi_term"] == 0.2
    mismatch = {**base, "text_feature_compatible": False}
    effect = llm._roi_prior_term(row, mismatch, FLOORS)
    assert effect["roi_term"] == 0.25 and effect["roi_reason"] == "applied_encoder_mismatch"
    turkish = llm._roi_prior_term(row, base, {**FLOORS, "non_english": True})
    assert turkish["roi_term"] == 0.25 and turkish["roi_reason"].endswith(";non_english_default_floor")
    monkeypatch.setattr(roi, "roi_multipliers", lambda encoder: {KEY: 0.25})
    assert llm._roi_prior_term(row, base, FLOORS)["roi_term"] == 0.125
    monkeypatch.setattr(roi, "roi_multipliers", lambda encoder: {KEY: 0})
    assert llm._roi_prior_term(row, base, FLOORS)["roi_reason"] == "insilico_reversed"
    monkeypatch.setattr(llm, "ROI_ENCODER_MISMATCH_FACTOR", 0)
    assert llm._roi_prior_term(row, mismatch, FLOORS)["roi_reason"] == "encoder_mismatch_disabled"
    monkeypatch.setattr(llm, "ROI_PRIOR_POINTS", 0)
    assert llm._roi_prior_term(row, base, FLOORS) == {"roi_term": 0, "roi_reason": "disabled"}


def select(monkeypatch, *, jev=True, gap=0, trace=0, all_ineligible=False, turkish=False, mock=False, floor_values=None):
    rows = [measurement(), measurement("c1", 0.2), measurement("c2", -0.2), measurement("c3", 20)]
    if mock:
        for row in rows:
            row.update(mode="mock", roi_features={})
    reviews = [{"id": row["id"], "supported": not all_ineligible and row["id"] != "c3",
                "intent_preserved": True, "recipient_respected": True, "voice_preserved": True,
                "context_fit": {facet: 25 if row["id"] == "original" else 75 for facet in llm.CONTEXT_FIT_KEYS}}
               for row in rows]
    # Smallest facet weight is .15: an 8 1/3-point facet change gives a 1.25-point semantic step.
    reviews[2]["context_fit"]["channel_fit"] += gap / 0.15
    monkeypatch.setattr(roi, "roi_noise_floors", lambda encoder, **kwargs: dict(FLOORS if floor_values is None else floor_values))
    monkeypatch.setattr(llm, "_jev_refinement_reviews", lambda *args: (reviews, "test", {"choice": "c2"}))
    monkeypatch.setattr(llm, "_post_refine_chat", lambda *args, **kwargs: json.dumps({"evaluations": reviews}))
    if trace:
        monkeypatch.setattr(llm, "_refine_response_effect", lambda *args: {"available": True, "score_adjustment": trace})
    result = {"model": "test"}
    if jev:
        result["decision_strategy"] = {"choices": {}, "instructions": []}
    return llm.select_tribe_refinement(rows[0]["message"], "Bir arkadaş" if turkish else "Engineering manager",
                                       "general", [], result, rows)["tribe_guidance"]


def test_jev_tie_counterfactual_and_semantic_step(monkeypatch):
    guidance = select(monkeypatch)
    assert guidance["selected_id"] == "c1"
    policy = guidance["selection_policy"]
    assert policy["roi_counterfactual_id"] == "c2" and policy["roi_changed_selection"]
    assert policy["basis"] == "context_first_experimental_trace_tiebreak"
    assert policy["roi_prior"]["floor_source"] == "default_prior"
    assert guidance["roi_readout"]["used_in_selection"]
    assert guidance["roi_readout"]["direction"] == "higher"
    assert guidance["evaluations"][1]["empirical_effect"]["available"] is False
    assert not guidance["evaluations"][3]["eligible"]
    assert select(monkeypatch, gap=1.25)["selected_id"] == "c2"


@pytest.mark.parametrize("trace", [-3, 3])
def test_total_adjustment_is_bounded(monkeypatch, trace):
    guidance = select(monkeypatch, trace=trace)
    for row in guidance["evaluations"]:
        assert abs(row["selection_score"] - row["semantic_score"]) <= 3


def test_turkish_default_floor_and_mock_and_ineligible_paths(monkeypatch):
    guidance = select(monkeypatch, turkish=True)
    row = guidance["evaluations"][1]
    assert row["empirical_effect"]["roi_term"] == 0.25
    assert row["empirical_effect"]["roi_reason"].endswith(";non_english_default_floor")
    assert row["selection_score_without_roi"] == 75
    mock = select(monkeypatch, mock=True)
    assert all(row["roi_status"] == "roi_unavailable" and row["empirical_effect"]["roi_term"] == 0 for row in mock["evaluations"])
    assert "roi_readout" not in mock
    empty = select(monkeypatch, all_ineligible=True)
    assert empty["selected_id"] == empty["selection_policy"]["roi_counterfactual_id"] == "original"
    assert empty["roi_readout"]["direction"] == "no_reliable_difference"
    assert select(monkeypatch, jev=False, all_ineligible=True)["roi_shadow_selected_id"] == "original"


def test_measured_turkish_floor_restores_language_weight_and_keeps_shadow(monkeypatch):
    floors = {**FLOORS, "floor_source": "paraphrase_noise_floor_tr"}
    guidance = select(monkeypatch, turkish=True, floor_values=floors)
    assert guidance["selection_policy"]["roi_prior"]["provenance"].endswith(";measured_turkish_paraphrase_floor")
    row = guidance["evaluations"][1]
    assert row["empirical_effect"]["roi_term"] == 0.5
    assert row["empirical_effect"]["roi_reason"] == "applied;measured_turkish_paraphrase_floor"
    assert row["empirical_effect"]["roi_provenance"].endswith(";measured_turkish_paraphrase_floor")
    assert row["selection_score_without_roi"] == 75
    base = {**measurement(), "text_feature_compatible": False}
    term = llm._roi_prior_term(measurement("c1", 0.2), base, {**floors, "non_english": True})
    assert term["roi_term"] == 0.25  # K8 remains independent of the language baseline.
    assert term["roi_provenance"].endswith(";encoder_mismatch;measured_turkish_paraphrase_floor")


@pytest.mark.parametrize("fault", [None, "language", "encoder", "atlas", "noise", "missing", "rollback"])
def test_turkish_reference_loader_and_default_fallback(monkeypatch, tmp_path, fault):
    folder = tmp_path / "roi_assets"
    folder.mkdir()
    data = reference(quantities={f"roi:{k}": reference()["quantities"]["evidence_score"] for k in (KEY, GUARD)})
    data["quantities"]["evidence_score"] = reference()["quantities"]["evidence_score"]
    data["language"] = "tr"
    default_bytes = (roi.ROI_ASSET_DIR / "roi_default_floors.json").read_bytes()
    (folder / "roi_default_floors.json").write_bytes(default_bytes)
    if fault in {"language", "encoder"}:
        data[fault] = "wrong"
    if fault == "atlas":
        data["atlas_sha256"] = "wrong"
    if fault == "noise":
        data["quantities"][f"roi:{KEY}"]["noise"]["min_detectable_delta"] = -1
    if fault != "missing":
        (folder / f"roi_reference_{ENCODER.replace('/', '__')}_tr.json").write_text(json.dumps(data))
    monkeypatch.setattr(engine, "__file__", str(tmp_path / "engine.py"))
    monkeypatch.setattr(roi, "ROI_ASSET_DIR", folder)
    monkeypatch.setenv("PITCHCHECK_ROI_INSILICO_TUNING", "0" if fault == "rollback" else "1")
    floors = roi.roi_noise_floors(ENCODER, language="tr")
    assert floors["floor_source"] == ("paraphrase_noise_floor_tr" if fault is None else "default_prior")
    assert engine.roi_reference(ENCODER, "../tr") is None
    assert roi.roi_noise_floors(ENCODER, language="tr", default_only=True)["floor_source"] == "default_prior"


@pytest.mark.parametrize("encoder", [ENCODER, "NousResearch/Hermes-3-Llama-3.2-3B"])
def test_shipped_turkish_measurement_is_consumed_without_replacing_english_units(monkeypatch, encoder):
    monkeypatch.setenv("PITCHCHECK_ROI_INSILICO_TUNING", "1")
    data = engine.roi_reference(encoder, "tr")
    assert data["language"] == "tr" and data["atlas_sha256"] == roi.load_roi_assets()[4]
    floors = roi.roi_noise_floors(encoder, language="tr")
    assert floors["floor_source"] == "paraphrase_noise_floor_tr"
    for key in (KEY, GUARD):
        quantity = data["quantities"][f"roi:{key}"]
        assert quantity["n_paraphrase_families"] == 30 and quantity["n_reference"] == 150
        noise = quantity["noise"]
        assert floors[key] == noise["min_detectable_delta"] == pytest.approx(1.96 * math.sqrt(2) * noise["sigma_within"])
    assert roi.roi_calibration(encoder)["n_cases"] == 300
    monkeypatch.setenv("PITCHCHECK_ROI_INSILICO_TUNING", "0")
    assert roi.roi_noise_floors(encoder, language="tr")["floor_source"] == "default_prior"


def test_disabled_golden_parity_and_legacy_shadow(monkeypatch):
    legacy = select(monkeypatch, jev=False)
    assert legacy["selected_id"] == "c1" and legacy["roi_shadow_selected_id"] == "c1"
    monkeypatch.setattr(llm, "ROI_PRIOR_POINTS", 0)
    disabled_legacy = select(monkeypatch, jev=False)
    assert [(r["id"], r["selection_score"]) for r in legacy["evaluations"]] == [(r["id"], r["selection_score"]) for r in disabled_legacy["evaluations"]]
    disabled = select(monkeypatch)
    assert disabled["selected_id"] == "c2"
    assert not disabled["selection_policy"]["roi_changed_selection"]
    assert all(r["selection_score"] == r["selection_score_without_roi"] for r in disabled["evaluations"])


def test_adapter_uses_signed_locations_only_and_no_template_without_asset():
    fmri = {"roi_summary": {"rois": {KEY: {"minus_cortex": 99}}, "locations": {
        KEY: {"minus_cortex": -0.1234567}, GUARD: {"minus_cortex": True}, "falk2010jocn_core": {"minus_cortex": math.inf}}}}
    assert app._roi_features(fmri) == {KEY: -0.123457}
    assert app._roi_features({}) == app._roi_features({"roi_summary": None}) == {}
    assert "tmpl_hv2025_d_minus_s_r" not in app._roi_features(fmri)


def test_legacy_roi_swap_does_not_change_selection(monkeypatch):
    rows = [measurement(), measurement("c1", -0.2), measurement("c2", 0.2)]
    reviews = [{"id": row["id"], "supported": True, "intent_preserved": True, "recipient_respected": True,
                "voice_preserved": True, "context_fit": {k: 25 if row["id"] == "original" else 75 for k in llm.CONTEXT_FIT_KEYS}}
               for row in rows]
    monkeypatch.setattr(roi, "roi_noise_floors", lambda *args, **kwargs: dict(FLOORS))
    monkeypatch.setattr(llm, "_post_refine_chat", lambda *args, **kwargs: json.dumps({"evaluations": reviews}))
    def run():
        return llm.select_tribe_refinement(rows[0]["message"], "Manager", "general", [], {"model": "test"}, rows)["tribe_guidance"]
    before = run()
    assert before["selected_id"] == "c1" and before["roi_shadow_selected_id"] == "c2"
    rows[1]["roi_features"], rows[2]["roi_features"] = rows[2]["roi_features"], rows[1]["roi_features"]
    after = run()
    assert after["selected_id"] == before["selected_id"]
    assert after["evaluations"][1]["roi_deltas"][KEY] == -before["evaluations"][1]["roi_deltas"][KEY]
    rows[1]["text_feature_model"] = "other"
    assert run()["evaluations"][1]["roi_status"] == "runtime_not_comparable"
    assert "roi_readout" not in run()


def test_legacy_writer_cannot_receive_numeric_roi_advice():
    prompt = llm._build_refine_prompt("Could we review the deployment workflow together?", "Manager", "general",
                                      [json.dumps({"roi_features": {KEY: -0.123456}, "roi_deltas": {KEY: 0.654321}})])
    assert "roi_" not in prompt and "-0.123456" not in prompt and "0.654321" not in prompt


def test_reviewer_state_never_sees_roi(monkeypatch):
    def decisions(state, questions):
        payload = json.dumps(state)
        assert '"roi_' not in payload and "roi_features" not in payload and "-0.123456" not in payload
        return {"model": "test", "answers": {name: {"noul": 1} if question["type"] == "noul" else
                {"score": 3} if question["type"] == "score" else {"choice": "c1"} for name, question in questions.items()}}
    monkeypatch.setattr(llm, "_post_jev_decisions", decisions)
    row = measurement("c1")
    row.update(roi_features={KEY: -0.123456}, roi_prior_score=95.123456)
    llm._jev_refinement_reviews(row["message"], "Manager", "general", [row], {"creative_brief": {}, "choices": {}, "instructions": []}, [])


@pytest.fixture(autouse=True)
def clear_reference_cache():
    engine.roi_reference.cache_clear()
    yield
    engine.roi_reference.cache_clear()


def reference(**kwargs):
    from tribe_service.tests.fixtures.ws08_synthetic import reference as stage0_reference
    data = stage0_reference(ENCODER)
    data["atlas_sha256"] = roi.load_roi_assets()[4]
    for key, row in kwargs.pop("quantities", {}).items():
        quantity = copy.deepcopy(data["quantities"]["evidence_score"])
        quantity["noise"].update(row["noise"])
        data["quantities"][key] = quantity
    return {**data, **kwargs}


def test_committed_floors_and_reference_precedence(monkeypatch, tmp_path):
    for encoder in (ENCODER, "NousResearch/Hermes-3-Llama-3.2-3B"):
        floors = roi.roi_noise_floors(encoder, default_only=True)
        assert floors["floor_source"] == "default_prior" and floors[KEY] > 0 and floors[GUARD] > 0
        assert "retest_not_preserved" in floors["provenance"]
    assert roi.roi_noise_floors("unknown") == {}
    monkeypatch.setattr(engine, "__file__", str(tmp_path / "engine.py"))
    folder = tmp_path / "roi_assets"
    folder.mkdir()
    monkeypatch.setattr(roi, "ROI_ASSET_DIR", folder)
    path = folder / f"roi_reference_{ENCODER.replace('/', '__')}.json"
    data = reference(quantities={f"roi:{k}": {"noise": {"min_detectable_delta": 0.3}} for k in (KEY, GUARD)})
    path.write_text(json.dumps(data))
    engine.roi_reference.cache_clear()
    assert roi.roi_noise_floors(ENCODER)[KEY] == 0.3
    assert roi.roi_noise_floors(ENCODER)["floor_source"] == "paraphrase_noise_floor"
    assert roi.roi_noise_floors(ENCODER, default_only=True) == {}
    for key in ("encoder", "atlas_sha256", "event_version", "text_input_mode"):
        path.write_text(json.dumps({**data, key: "mismatch"}))
        engine.roi_reference.cache_clear()
        assert roi.roi_noise_floors(ENCODER) == {}


@pytest.mark.parametrize("broken", [[], None, {"encoders": []}, {"encoders": {ENCODER: {"floors": []}}}])
def test_malformed_default_floors_are_unavailable(monkeypatch, tmp_path, broken):
    monkeypatch.setattr(roi, "ROI_ASSET_DIR", tmp_path)
    (tmp_path / "roi_default_floors.json").write_text(json.dumps(broken))
    assert roi.roi_noise_floors(ENCODER, default_only=True) == {}


@pytest.mark.parametrize("quantity", [None, [], {"noise": []}, {"noise": {"min_detectable_delta": True}},
                                       {"noise": {"min_detectable_delta": -1}}, {"length_model": []}])
def test_malformed_reference_quantities_fail_safe(monkeypatch, tmp_path, quantity):
    monkeypatch.setattr(roi, "ROI_ASSET_DIR", tmp_path)
    monkeypatch.setattr(roi, "_reference", lambda encoder: {"quantities": {f"roi:{k}": quantity for k in (KEY, GUARD, "falk2010jocn_core")}})
    assert roi.roi_noise_floors(ENCODER) == {}
    assert roi.roi_reference_p25(ENCODER, 50, "email") is None


def hypothesis_fmri(z=-0.67):
    return {"text_feature_model": ENCODER, "roi_summary": {"locations": {
        "falk2010jocn_core": {"minus_cortex": -0.2, "z": z}, "tp_l": {"minus_cortex": -0.1}, "tp_r": {"minus_cortex": -0.3}}}}


def test_hypothesis_provisional_signed_guard_order_and_rollback(monkeypatch):
    monkeypatch.setattr(roi, "roi_reference_p25", lambda *args: None)
    axes = {k: {"score": v} for k, v in (("self_value", 75), ("processing_fluency", 40))}
    fmri = hypothesis_fmri()
    items = build_tribe_synthesis("A short draft", axes, fmri)["items"]
    assert [i["kind"] for i in items] == ["gap", "hypothesis", "strength"]
    item = items[1]
    assert item["axis"] is None and item["experimental"] and item["reference"] == "default_prior_within_draft"
    assert not re.search("value|self|mentaliz|perspective|reactance", item["key"])
    assert "provisional reference" in item["observation"] and "not this recipient" in item["finding"]
    from tribe_service.persuasion_features import RESEARCH_SOURCES
    assert set(item["source_keys"]) <= {source["key"] for source in RESEARCH_SOURCES}
    assert build_tribe_synthesis("A draft", {}, {})["items"] == build_tribe_synthesis("A draft", {}, hypothesis_fmri(-0.669))["items"]
    for value in (0.1, math.nan, True, None):
        bad = copy.deepcopy(fmri)
        bad["roi_summary"]["locations"]["tp_r"]["minus_cortex"] = value
        assert not any(i["kind"] == "hypothesis" for i in build_tribe_synthesis("draft", {}, bad)["items"])
    monkeypatch.setenv("PITCHCHECK_ROI_HYPOTHESES", "0")
    assert not any(i["kind"] == "hypothesis" for i in build_tribe_synthesis("draft", axes, fmri)["items"])


@pytest.mark.parametrize("words,bucket", [(60, "le60"), (61, "61_150"), (150, "61_150"), (151, "151_240"), (240, "151_240")])
def test_reference_length_buckets_and_channel_pooling(monkeypatch, words, bucket):
    pooled = {"length_buckets": {bucket: {"p25": -0.1}}, "channels": {"email": {"n": 49, "length_buckets": {bucket: {"p25": -0.3}}}}}
    monkeypatch.setattr(roi, "_reference", lambda encoder: {"quantities": {"roi:falk2010jocn_core": pooled}})
    def hypotheses():
        return [i for i in build_tribe_synthesis("word " * words, {}, hypothesis_fmri(z=99), platform="email")["items"] if i["kind"] == "hypothesis"]
    assert hypotheses()[0]["reference"] == "length_matched_reference"
    pooled["channels"]["email"]["n"] = 50
    assert hypotheses() == []


def test_stage0_length_model_p25(monkeypatch):
    monkeypatch.setattr(roi, "_reference", lambda encoder: {"quantities": {"roi:falk2010jocn_core": {
        "length_model": {"a": 0.1, "b_log_words": 0.02}, "resid_q": {"p25": -0.3}}}})
    assert roi.roi_reference_p25(ENCODER, 100, "general") == pytest.approx(0.1 + 0.02 * math.log(100) - 0.3)
