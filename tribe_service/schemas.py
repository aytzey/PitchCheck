from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

PLATFORM_VALUES = ("email", "linkedin", "cold-call-script", "landing-page", "ad-copy", "general")

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

    @field_validator("platform")
    @classmethod
    def validate_platform(cls, v: str) -> str:
        if v not in PLATFORM_VALUES:
            return "general"
        return v

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
