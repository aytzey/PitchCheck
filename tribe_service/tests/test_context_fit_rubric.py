"""WS05 prompt parity, rollback and selection checks; no network or GPU."""
import importlib
import json
from unittest.mock import patch

import pytest

from tribe_service import llm_layer as llm
from tribe_service.tests.test_llm_layer import (
    SAMPLE_MESSAGE, SAMPLE_PERSONA, SAMPLE_PLATFORM, SAMPLE_NEURAL_SIGNALS,
    SAMPLE_RAW_FEATURES, SAMPLE_SYNTHETIC_FMRI_SUMMARY, VALID_LLM_RESPONSE,
)


def test_shared_rubric_and_respect_rules_reach_all_three_judges(monkeypatch):
    evaluator = llm._build_user_prompt(SAMPLE_MESSAGE, SAMPLE_PERSONA, SAMPLE_PLATFORM,
                                      SAMPLE_NEURAL_SIGNALS)
    critic = llm._build_refine_critic_prompt(SAMPLE_MESSAGE, SAMPLE_PERSONA, SAMPLE_PLATFORM, [], [])
    captured = {}

    def decisions(state, questions):
        captured.update(questions)
        return {"model": "typesafe/jev-test", "answers": {
            name: {"noul": 1} if question["type"] == "noul" else
            {"score": 3} if question["type"] == "score" else {"choice": "original", "confidence": .9}
            for name, question in questions.items()}}

    monkeypatch.setattr(llm, "_post_jev_decisions", decisions)
    reviews, _, _ = llm._jev_refinement_reviews(SAMPLE_MESSAGE, SAMPLE_PERSONA, SAMPLE_PLATFORM,
        [{"id": "original", "message": SAMPLE_MESSAGE}], {"creative_brief": {}, "choices": {}, "instructions": []}, [])
    assert set(llm.CONTEXT_FIT_NOTES) == set(llm.CONTEXT_FIT_KEYS)
    for key in llm.CONTEXT_FIT_KEYS:
        note = llm.CONTEXT_FIT_NOTES[key]
        assert note in evaluator and note in critic
        question = captured[f"original_{key}"]
        assert note in question["instructions"] and len(question["criteria"]) == 5
        assert reviews[0]["context_fit"][key] == 75
    criteria = captured["original_recipient_respected"]["criteria"]
    assert llm.RECIPIENT_RESPECT_FAIL in critic and llm.RECIPIENT_RESPECT_FAIL in criteria["false"]
    assert llm.RECIPIENT_RESPECT_PASS in critic and llm.RECIPIENT_RESPECT_PASS in criteria["true"]
    for phrase in ("even if copied from the original or followed by a question", "falling behind"):
        assert phrase in criteria["false"]
    for phrase in ("See pricing", "comparable peers", "ad-copy or landing-page channels only"):
        assert phrase in criteria["true"]
    assert "Generic:" in captured["original_persona_pain_alignment"]["criteria"][1]
    assert "Weak/generic" in captured["original_cta_ease"]["criteria"][1]
    # Long instructions live outside the JSON placeholders.
    schema = evaluator.split("Return JSON with this exact shape:")[1]
    assert all(note not in schema for note in llm.CONTEXT_FIT_NOTES.values())
    assert "TRIBE never sees the persona" in critic
    assert "region-labelled response fields" in critic


def test_protocol_and_refine_writer_alignment():
    for phrase in ("7. Value and threat", "never infer these from job title", "neither frame is a default",
                   "appeal basis", "never set a default frame or penalty", "compliments do not substitute for value"):
        assert phrase in llm.SYSTEM_PROMPT
    evaluator = llm._build_user_prompt(SAMPLE_MESSAGE, SAMPLE_PERSONA, SAMPLE_PLATFORM, SAMPLE_NEURAL_SIGNALS)
    assert "value/threat" in evaluator and "RECIPIENT-SPECIFIC" in evaluator
    assert "wherever it appears" in llm.CONTEXT_FIT_NOTES["cta_ease"]
    writer = llm._build_refine_prompt(SAMPLE_MESSAGE, SAMPLE_PERSONA, SAMPLE_PLATFORM, [])
    for phrase in ("category-level paraphrase", "never invent one", "what happens after yes",
                   "outcome-led, insight-led and proof-led", "recipient-specific detail",
                   "comparable peers do is allowed", "no durations, days, dates or counts",
                   "For skeptical readers", "two of the three"):
        assert phrase in writer
    assert "Make the opener persona-specific" not in writer


def test_clarification_api_keeps_rubric_version(monkeypatch):
    from tribe_service.tests.test_app import client
    monkeypatch.setattr(llm, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_post_refine_chat", lambda *args, **kwargs: json.dumps({
        "needs_clarification": True, "questions": [{"question": "Which proof is supplied?"}]}))
    response = client.post("/refine", json={"message": SAMPLE_MESSAGE, "persona": SAMPLE_PERSONA})
    assert response.status_code == 200
    assert response.json()["needs_clarification"] is True
    assert response.json()["tribe_guidance"]["context_fit_rubric_version"] == "v2"


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_rollback_and_versions_in_every_score_and_refine_path(monkeypatch, version):
    try:
        with monkeypatch.context() as context:
            context.setenv("PITCHCHECK_CONTEXT_FIT_RUBRIC", version)
            importlib.reload(llm)
            assert llm.CONTEXT_FIT_RUBRIC_VERSION == version
            writer = llm._build_refine_prompt(SAMPLE_MESSAGE, SAMPLE_PERSONA, SAMPLE_PLATFORM, [])
            critic = llm._build_refine_critic_prompt(SAMPLE_MESSAGE, SAMPLE_PERSONA, SAMPLE_PLATFORM, [], [])
            evaluator = llm._build_user_prompt(SAMPLE_MESSAGE, SAMPLE_PERSONA, SAMPLE_PLATFORM, SAMPLE_NEURAL_SIGNALS)
            if version == "v1":
                assert "how easy is it to say yes: effort, commitment, social risk" in evaluator
                assert "no guilt or pressure" in critic
                assert "gain frame or an avoided-loss frame" in llm.SYSTEM_PROMPT
                assert "7. Value and threat" not in llm.SYSTEM_PROMPT
                assert "category-level paraphrase" not in writer
                assert llm._JEV_STRATEGY_OPTIONS["objection"]["trust"] == "Credibility concern: use existing evidence, never fabricate proof."
            for raw in (None, dict(VALID_LLM_RESPONSE), {"persuasion_score": 78, "breakdown": None}):
                with patch.object(llm, "_call_openrouter", return_value=raw):
                    result = llm.interpret_persuasion(SAMPLE_MESSAGE, SAMPLE_PERSONA, SAMPLE_PLATFORM,
                        SAMPLE_NEURAL_SIGNALS, SAMPLE_RAW_FEATURES, SAMPLE_SYNTHETIC_FMRI_SUMMARY)
                assert result["robustness"]["context_fit_rubric_version"] == version
                assert result["robustness"]["prompt_doctrine_version"] == "doctrine_v2"
            context.setattr(llm, "OPENROUTER_API_KEY", "test-key")
            for body in ({"needs_clarification": True, "questions": [{"question": "What proof is supplied?"}]},
                         {"candidates": ["A fresh invitation for you?", "Would you like to join me?", "Shall we go together?"]}):
                with patch.object(llm, "_post_refine_chat", return_value=json.dumps(body)):
                    result = llm.refine_pitch_message(SAMPLE_MESSAGE, SAMPLE_PERSONA, SAMPLE_PLATFORM)
                assert result["tribe_guidance"]["context_fit_rubric_version"] == version
    finally:
        # Reload after monkeypatch restores the env; do not leak a v1 module to other tests.
        importlib.reload(llm)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_jev_writer_gets_guardrail_descriptions_only_in_v2(monkeypatch, version):
    try:
        with monkeypatch.context() as context:
            context.setenv("PITCHCHECK_CONTEXT_FIT_RUBRIC", version)
            importlib.reload(llm)
            context.setattr(llm, "OPENROUTER_API_KEY", "test-key")
            def decisions(state, questions):
                choices = {"relationship": "business", "objection": "trust", "angle": "outcome",
                           "tone": "professional", "repair": "proof", "move": "concrete_value", "repair_target": "whole"}
                return {"model": "typesafe/jev-test", "answers": {
                    name: {"choice": choices[name], "confidence": .9} for name in questions}}
            context.setattr(llm, "_post_jev_decisions", decisions)
            strategy = llm.plan_tribe_refinement(SAMPLE_MESSAGE, SAMPLE_PERSONA, SAMPLE_PLATFORM,
                                                {"id": "original", "message": SAMPLE_MESSAGE})
            with patch.object(llm, "_post_refine_chat", return_value=json.dumps({
                "needs_clarification": True, "questions": [{"question": "What proof is supplied?"}]})) as post:
                result = llm.refine_pitch_message(SAMPLE_MESSAGE, SAMPLE_PERSONA, SAMPLE_PLATFORM, decision_strategy=strategy)
            prompt = post.call_args.args[1]
            assert ("no invented feelings" in prompt) == (version == "v2")
            assert ("off-topic status" in prompt) == (version == "v2")
            assert ('"barrier":' in prompt) == (version == "v2")
            assert "neural_score" not in prompt and "temporal_trace" not in prompt
            assert "experience" not in llm._JEV_STRATEGY_OPTIONS["angle"]
            assert result["tribe_guidance"]["context_fit_rubric_version"] == version
    finally:
        importlib.reload(llm)


def test_disrespectful_original_allows_a_lower_semantic_safe_candidate(monkeypatch):
    messages = ["You must review this proposal with me.", "Would you review this proposal with me?",
                "Can we review this proposal together?", "Shall we review this proposal together?"]
    measurements = [{"id": name, "message": message, "neural_score": 50, "quality_weight": 1,
                     "mode": "model", "model_id": "test"}
                    for name, message in zip(("original", "c1", "c2", "c3"), messages)]
    reviews = [{"id": row["id"], "supported": True, "intent_preserved": True,
                "recipient_respected": row["id"] == "c1", "voice_preserved": True,
                "context_fit": {key: 90 if row["id"] == "original" else 60 for key in llm.CONTEXT_FIT_KEYS},
                "issues": []} for row in measurements]
    monkeypatch.setattr(llm, "_post_refine_chat", lambda *args, **kwargs: json.dumps({"evaluations": reviews}))
    result = llm.select_tribe_refinement(messages[0], "A colleague", "email", [],
                                        {"model": "test"}, measurements)
    proof = result["tribe_guidance"]
    assert proof["selected_id"] == "c1"
    assert proof["evaluations"][1]["semantic_score"] < proof["evaluations"][0]["semantic_score"]
    assert proof["context_fit_rubric_version"] == "v2"
