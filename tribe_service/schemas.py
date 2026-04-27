from __future__ import annotations
from datetime import datetime, timezone
import os
import re
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

PLATFORM_VALUES = ("email", "linkedin", "cold-call-script", "landing-page", "ad-copy", "general")
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._:/@+-]{1,160}$")


def _env_int(name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


MAX_MESSAGE_CHARS = _env_int("PITCHCHECK_MAX_MESSAGE_CHARS", 30_000, 10)
MAX_PERSONA_CHARS = _env_int("PITCHCHECK_MAX_PERSONA_CHARS", 5_000, 5)


class PitchScoreRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    message: str = Field(..., min_length=10)
    persona: str = Field(..., min_length=5)
    platform: str = Field(default="general")
    open_router_model: str | None = Field(default=None, alias="openRouterModel")

    @field_validator("message", "persona", mode="before")
    @classmethod
    def strip_text_fields(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v

    @field_validator("message")
    @classmethod
    def validate_message_length(cls, v: str) -> str:
        if len(v) > MAX_MESSAGE_CHARS:
            raise ValueError(f"message must be at most {MAX_MESSAGE_CHARS} characters.")
        return v

    @field_validator("persona")
    @classmethod
    def validate_persona_length(cls, v: str) -> str:
        if len(v) > MAX_PERSONA_CHARS:
            raise ValueError(f"persona must be at most {MAX_PERSONA_CHARS} characters.")
        return v

    @field_validator("platform")
    @classmethod
    def validate_platform(cls, v: str) -> str:
        value = v.strip().lower() if isinstance(v, str) else "general"
        if value not in PLATFORM_VALUES:
            return "general"
        return value

    @field_validator("open_router_model")
    @classmethod
    def validate_open_router_model(cls, v: str | None) -> str | None:
        if v is None:
            return None
        value = v.strip()
        if not value:
            return None
        if not _MODEL_ID_RE.match(value):
            return None
        return value


class PitchRefineRequest(PitchScoreRequest):
    suggestions: list[str] = Field(default_factory=list, max_length=12)
    clarification_answers: list["PitchRefineClarificationAnswer"] = Field(
        default_factory=list,
        max_length=6,
        alias="clarificationAnswers",
    )


class PitchRefineClarificationAnswer(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default="", max_length=80)
    question: str = Field(default="", max_length=500)
    answer: str = Field(..., min_length=1, max_length=1000)


class AuthLoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)

class AuthChangePasswordRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    current_password: str = Field(..., min_length=1, alias="currentPassword")
    new_username: str = Field(..., min_length=1, alias="newUsername")
    new_password: str = Field(..., min_length=1, alias="newPassword")

class BreakdownSection(BaseModel):
    key: str
    label: str
    score: float = Field(..., ge=0, le=100)
    explanation: str

class NeuralSignal(BaseModel):
    key: str
    label: str
    score: float = Field(..., ge=0, le=100)
    direction: Literal["up", "down", "neutral"] = "neutral"

class RewriteSuggestion(BaseModel):
    title: str
    before: str
    after: str
    why: str

class FmriOutput(BaseModel):
    """fMRI summary from TRIBE — temporal trace and top voxel data."""
    segments: int
    voxel_count: int
    global_mean_abs: float
    global_peak_abs: float
    temporal_trace: list[float]     # per-segment mean activation
    temporal_peaks: list[float]     # per-segment peak activation
    top_voxel_indices: list[int]    # top 6 most-activated voxel indices
    top_voxel_values: list[float]   # their mean activation values
    response_kind: str = "tribe_predicted_fmri_analogue"
    prediction_subject_basis: str = "average_subject"
    cortical_mesh: str = "fsaverage5"
    hemodynamic_lag_seconds: float = 5.0
    temporal_trace_basis: Literal["real_time_seconds", "synthetic_word_order"] = "real_time_seconds"
    temporal_segment_label: str = "second"
    temporal_trace_note: str = ""

class PitchScoreReport(BaseModel):
    persuasion_score: float = Field(..., ge=0, le=100)
    verdict: str
    narrative: str
    breakdown: list[BreakdownSection]
    neural_signals: list[NeuralSignal]
    strengths: list[str]
    risks: list[str]
    rewrite_suggestions: list[RewriteSuggestion]
    persona_summary: str
    fmri_output: FmriOutput | None = None
    persuasion_evidence: dict[str, Any] | None = None
    robustness: dict[str, Any] | None = None
    platform: str = "general"
    scored_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PitchRefineQuestion(BaseModel):
    id: str = Field(..., min_length=1, max_length=80)
    label: str = Field(default="", max_length=80)
    question: str = Field(..., min_length=1, max_length=500)
    why: str = Field(default="", max_length=500)


class PitchRefineResponse(BaseModel):
    model: str
    refined_message: str | None = None
    needs_clarification: bool = False
    questions: list[PitchRefineQuestion] = Field(default_factory=list, max_length=3)
    safety_notes: list[str] = Field(default_factory=list, max_length=5)
    persuasion_profile: dict[str, Any] | None = None
    methodology: str = "llm_semantic_refine_no_tribe_rescore"
