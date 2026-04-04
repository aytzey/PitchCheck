from __future__ import annotations
from datetime import datetime, timezone
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

PLATFORM_VALUES = ("email", "linkedin", "cold-call-script", "landing-page", "ad-copy", "general")

class PitchScoreRequest(BaseModel):
    message: str = Field(..., min_length=10)
    persona: str = Field(..., min_length=5)
    platform: str = Field(default="general")

    @field_validator("platform")
    @classmethod
    def validate_platform(cls, v: str) -> str:
        if v not in PLATFORM_VALUES:
            return "general"
        return v

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
    platform: str = "general"
    scored_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
