"""PitchScore FastAPI application."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from tribe_service.schemas import (
    PitchScoreRequest,
    PitchScoreReport,
    BreakdownSection,
    FmriOutput,
    NeuralSignal,
    RewriteSuggestion,
)
from tribe_service.engine import (
    score_text,
    extract_features,
    derive_persuasion_signals,
    summarize_fmri_output,
    is_model_loaded,
    PERSUASION_SIGNAL_LABELS,
    TRIBE_DEVICE,
    TRIBE_MODEL_ID,
)
from tribe_service.llm_layer import (
    interpret_persuasion,
    OPENROUTER_ENABLED,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="PitchScore TRIBE Service", docs_url="/docs", redoc_url=None)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://pitch.machinity.ai",
        "https://www.pitch.machinity.ai",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": "pitchscore-tribe",
        "model_id": TRIBE_MODEL_ID,
        "device": TRIBE_DEVICE,
        "model_loaded": is_model_loaded(),
        "openrouter_enabled": OPENROUTER_ENABLED,
    }


@app.post("/score")
async def score_pitch(request: PitchScoreRequest):
    """Score a sales pitch for persuasion effectiveness against a target persona."""
    try:
        # 1. Run TRIBE text scoring
        predictions = score_text(request.message)

        # 2. Extract raw features + fMRI summary
        raw_features = extract_features(predictions)
        fmri_data = summarize_fmri_output(predictions)

        # 3. Derive persuasion signals
        neural_signals = derive_persuasion_signals(raw_features)

        # 4. LLM interpretation (or fallback) — include fMRI temporal trace
        llm_result = interpret_persuasion(
            message=request.message,
            persona=request.persona,
            platform=request.platform,
            neural_signals=neural_signals,
            raw_features=raw_features,
            fmri_summary=fmri_data,
        )

        # 5. Assemble PitchScoreReport
        breakdown = [
            BreakdownSection(
                key=b["key"],
                label=b["label"],
                score=max(0, min(100, float(b["score"]))),
                explanation=b.get("explanation", ""),
            )
            for b in llm_result.get("breakdown", [])
        ]

        neural_signal_list = [
            NeuralSignal(
                key=k,
                label=PERSUASION_SIGNAL_LABELS.get(k, k),
                score=round(v, 1),
                direction="up" if v >= 60 else "down" if v < 40 else "neutral",
            )
            for k, v in neural_signals.items()
        ]

        rewrite_suggestions = [
            RewriteSuggestion(
                title=r.get("title", ""),
                before=r.get("before", ""),
                after=r.get("after", ""),
                why=r.get("why", ""),
            )
            for r in llm_result.get("rewrite_suggestions", [])
        ]

        report = PitchScoreReport(
            persuasion_score=max(0, min(100, float(llm_result.get("persuasion_score", 50)))),
            verdict=llm_result.get("verdict", "Analysis complete"),
            narrative=llm_result.get("narrative", ""),
            breakdown=breakdown,
            neural_signals=neural_signal_list,
            strengths=llm_result.get("strengths", [])[:3],
            risks=llm_result.get("risks", [])[:3],
            rewrite_suggestions=rewrite_suggestions,
            persona_summary=llm_result.get("persona_summary", request.persona),
            fmri_output=FmriOutput(**fmri_data),
            platform=request.platform,
            scored_at=datetime.now(timezone.utc).isoformat(),
        )

        return {"report": report.model_dump()}

    except Exception as exc:
        LOGGER.exception("Scoring failed")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(exc)}")


@app.on_event("startup")
async def startup():
    LOGGER.info(
        "PitchScore TRIBE service starting — model=%s device=%s openrouter=%s",
        TRIBE_MODEL_ID, TRIBE_DEVICE, OPENROUTER_ENABLED,
    )
