import pytest
from tribe_service.schemas import (
    PitchRefineRequest,
    PitchRefineResponse,
    PitchScoreRequest,
    PitchScoreReport,
    BreakdownSection,
    NeuralSignal,
    RewriteSuggestion,
    MAX_MESSAGE_CHARS,
    MAX_PERSONA_CHARS,
    PLATFORM_VALUES,
)

def test_valid_request():
    req = PitchScoreRequest(message="Hello world, this is a pitch", persona="CTO at startup")
    assert req.platform == "general"

def test_request_short_message():
    with pytest.raises(Exception):
        PitchScoreRequest(message="Hi", persona="CTO at startup")

def test_request_short_persona():
    with pytest.raises(Exception):
        PitchScoreRequest(message="This is a valid pitch message", persona="CTO")

def test_request_strips_before_length_validation():
    with pytest.raises(Exception):
        PitchScoreRequest(message="          ", persona="CTO at startup")

def test_request_invalid_platform_defaults():
    req = PitchScoreRequest(message="Valid message here", persona="Valid persona", platform="invalid")
    assert req.platform == "general"

def test_request_normalizes_platform_case():
    req = PitchScoreRequest(message="Valid message here", persona="Valid persona", platform="LinkedIn")
    assert req.platform == "linkedin"

def test_request_accepts_long_text_and_model_alias():
    req = PitchScoreRequest(
        message="A" * 6000,
        persona="Technical buyer " * 120,
        openRouterModel="openai/gpt-5.4",
    )
    assert req.open_router_model == "openai/gpt-5.4"

def test_request_rejects_oversized_message_and_persona():
    with pytest.raises(Exception):
        PitchScoreRequest(message="A" * (MAX_MESSAGE_CHARS + 1), persona="Valid persona")
    with pytest.raises(Exception):
        PitchScoreRequest(message="Valid message here", persona="A" * (MAX_PERSONA_CHARS + 1))

def test_request_drops_invalid_model_alias():
    req = PitchScoreRequest(
        message="Valid message here",
        persona="Valid persona",
        openRouterModel="bad model\ninjected",
    )
    assert req.open_router_model is None

def test_refine_request_accepts_suggestions_and_model_alias():
    req = PitchRefineRequest(
        message="Valid message here",
        persona="Valid persona",
        platform="Email",
        suggestions=["Make the CTA easier"],
        openRouterModel="anthropic/claude-sonnet-4.6",
    )
    assert req.platform == "email"
    assert req.suggestions == ["Make the CTA easier"]
    assert req.open_router_model == "anthropic/claude-sonnet-4.6"

def test_refine_response_defaults_to_llm_methodology():
    response = PitchRefineResponse(refined_message="Better pitch", model="test-model")
    assert response.methodology == "llm_semantic_refine_no_tribe_rescore"

def test_refine_response_can_hold_questions_without_rewrite():
    response = PitchRefineResponse(
        model="test-model",
        needs_clarification=True,
        questions=[{
            "id": "proof",
            "label": "Proof",
            "question": "Which proof can we mention?",
            "why": "Avoids invented claims.",
        }],
    )
    assert response.refined_message is None
    assert response.needs_clarification is True
    assert response.questions[0].id == "proof"

def test_valid_report():
    report = PitchScoreReport(
        persuasion_score=75.0,
        verdict="Strong pitch for technical audience",
        narrative="The pitch effectively addresses pain points.",
        breakdown=[
            BreakdownSection(key="clarity", label="Clarity", score=80.0, explanation="Clear messaging")
        ],
        neural_signals=[
            NeuralSignal(key="attention_capture", label="Attention Capture", score=70.0, direction="up")
        ],
        strengths=["Clear value proposition"],
        risks=["Missing social proof"],
        rewrite_suggestions=[
            RewriteSuggestion(title="Opener", before="Hi", after="Hi [Name]", why="Personalization")
        ],
        persona_summary="Technical CTO at early-stage startup",
    )
    assert report.persuasion_score == 75.0
    assert len(report.breakdown) == 1

def test_report_score_out_of_range():
    with pytest.raises(Exception):
        PitchScoreReport(
            persuasion_score=150.0,
            verdict="x", narrative="x",
            breakdown=[], neural_signals=[],
            strengths=[], risks=[], rewrite_suggestions=[],
            persona_summary="x",
        )

def test_report_roundtrip():
    report = PitchScoreReport(
        persuasion_score=50.0,
        verdict="Average", narrative="Needs work.",
        breakdown=[BreakdownSection(key="clarity", label="Clarity", score=50.0, explanation="OK")],
        neural_signals=[NeuralSignal(key="memorability", label="Memorability", score=60.0)],
        strengths=["Good opener"], risks=["Weak close"],
        rewrite_suggestions=[RewriteSuggestion(title="Close", before="Thanks", after="Let's schedule a call", why="CTA")],
        persona_summary="Marketing manager",
    )
    data = report.model_dump()
    restored = PitchScoreReport(**data)
    assert restored.persuasion_score == 50.0
    assert restored.breakdown[0].key == "clarity"
