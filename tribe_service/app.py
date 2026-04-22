"""PitchScore FastAPI application."""
from __future__ import annotations

import logging
import os
import asyncio
import contextlib
import time
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from tribe_service.schemas import (
    AuthChangePasswordRequest,
    AuthLoginRequest,
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
    runtime_config,
    unload_model,
    PERSUASION_SIGNAL_LABELS,
    TRIBE_DEVICE,
    TRIBE_MODEL_ID,
    TRIBE_TEXT_INPUT_MODE,
)
from tribe_service.llm_layer import (
    interpret_persuasion,
    OPENROUTER_ENABLED,
)
from tribe_service.auth import (
    AUTH_STORE,
    AuthConfigurationError,
    InvalidCredentialUpdateError,
    InvalidCredentialsError,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
TRIBE_SCORE_TIMEOUT_SECONDS = float(os.getenv("TRIBE_SCORE_TIMEOUT_SECONDS", "900"))
TRIBE_MAX_SCORE_CONCURRENCY = max(1, int(os.getenv("TRIBE_MAX_SCORE_CONCURRENCY", "1")))
TRIBE_IDLE_UNLOAD_SECONDS = max(0.0, float(os.getenv("TRIBE_IDLE_UNLOAD_SECONDS", "600")))

app = FastAPI(title="PitchScore TRIBE Service", docs_url="/docs", redoc_url=None)
_score_lock = asyncio.Semaphore(TRIBE_MAX_SCORE_CONCURRENCY)
_bearer = HTTPBearer(auto_error=False)
_pipeline_lock = asyncio.Lock()
_active_scores = 0
_last_runtime_activity = time.monotonic()
_idle_unload_task: asyncio.Task | None = None

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


def _auth_error(error: Exception) -> HTTPException:
    if isinstance(error, AuthConfigurationError):
        return HTTPException(status_code=503, detail=str(error))
    if isinstance(error, InvalidCredentialUpdateError):
        return HTTPException(status_code=400, detail=str(error))
    if isinstance(error, InvalidCredentialsError):
        return HTTPException(status_code=401, detail=str(error))
    return HTTPException(status_code=500, detail="Authentication failed.")


def _token_from_credentials(credentials: HTTPAuthorizationCredentials | None) -> str | None:
    return credentials.credentials if credentials else None


def _idle_for_seconds() -> float:
    return max(0.0, time.monotonic() - _last_runtime_activity)


async def _begin_runtime_activity() -> None:
    global _active_scores, _last_runtime_activity
    async with _pipeline_lock:
        _active_scores += 1
        _last_runtime_activity = time.monotonic()


async def _finish_runtime_activity() -> None:
    global _active_scores, _last_runtime_activity
    async with _pipeline_lock:
        _active_scores = max(0, _active_scores - 1)
        _last_runtime_activity = time.monotonic()


async def _pipeline_status() -> dict:
    async with _pipeline_lock:
        return {
            "model_loaded": is_model_loaded(),
            "active_scores": _active_scores,
            "idle_for_seconds": round(_idle_for_seconds(), 3),
            "idle_unload_seconds": TRIBE_IDLE_UNLOAD_SECONDS,
        }


async def _unload_pipeline(reason: str) -> dict:
    async with _pipeline_lock:
        if _active_scores > 0:
            return {
                "ok": False,
                "unloaded": False,
                "reason": "score_in_progress",
                "active_scores": _active_scores,
                "model_loaded": is_model_loaded(),
            }
        was_loaded = is_model_loaded()
        if was_loaded:
            await run_in_threadpool(unload_model)
        return {
            "ok": True,
            "unloaded": was_loaded,
            "reason": reason,
            "active_scores": _active_scores,
            "model_loaded": is_model_loaded(),
        }


async def _idle_unload_loop() -> None:
    if TRIBE_IDLE_UNLOAD_SECONDS <= 0:
        return
    interval = min(60.0, max(5.0, TRIBE_IDLE_UNLOAD_SECONDS / 4))
    while True:
        await asyncio.sleep(interval)
        if not is_model_loaded():
            continue
        async with _pipeline_lock:
            should_unload = _active_scores == 0 and _idle_for_seconds() >= TRIBE_IDLE_UNLOAD_SECONDS
        if should_unload:
            LOGGER.info("Unloading idle TRIBE pipeline after %.1fs", _idle_for_seconds())
            await _unload_pipeline("idle_timeout")


async def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    try:
        return AUTH_STORE.verify_token(_token_from_credentials(credentials))
    except Exception as error:
        raise _auth_error(error) from error


@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": "pitchscore-tribe",
        "auth": AUTH_STORE.status(),
        "model_id": TRIBE_MODEL_ID,
        "device": TRIBE_DEVICE,
        "model_loaded": is_model_loaded(),
        "pipeline": await _pipeline_status(),
        "runtime": runtime_config(),
        "max_score_concurrency": TRIBE_MAX_SCORE_CONCURRENCY,
        "openrouter_enabled": OPENROUTER_ENABLED,
    }


@app.post("/auth/login")
async def auth_login(request: AuthLoginRequest):
    try:
        return AUTH_STORE.login(request.username, request.password)
    except Exception as error:
        raise _auth_error(error) from error


@app.post("/auth/change-password")
async def auth_change_password(
    request: AuthChangePasswordRequest,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
):
    try:
        return AUTH_STORE.change_credentials(
            token=_token_from_credentials(credentials),
            current_password=request.current_password,
            new_username=request.new_username,
            new_password=request.new_password,
        )
    except Exception as error:
        raise _auth_error(error) from error


@app.post("/score")
async def score_pitch(request: PitchScoreRequest, _: str = Depends(require_auth)):
    """Score a sales pitch for persuasion effectiveness against a target persona."""
    await _begin_runtime_activity()
    try:
        # 1. Run TRIBE text scoring
        async with _score_lock:
            predictions = await asyncio.wait_for(
                run_in_threadpool(score_text, request.message),
                timeout=TRIBE_SCORE_TIMEOUT_SECONDS,
            )

        # 2. Extract raw features + fMRI summary
        raw_features = extract_features(predictions)
        fmri_data = summarize_fmri_output(
            predictions,
            text_input_mode=TRIBE_TEXT_INPUT_MODE,
        )

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
            openrouter_model=request.open_router_model,
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
            persuasion_evidence=llm_result.get("persuasion_evidence"),
            robustness=llm_result.get("robustness"),
            platform=request.platform,
            scored_at=datetime.now(timezone.utc).isoformat(),
        )

        return {"report": report.model_dump()}

    except asyncio.TimeoutError:
        LOGGER.exception("Scoring timed out")
        raise HTTPException(
            status_code=504,
            detail="Scoring timed out while running TRIBE. Try a shorter message or reconnect the runtime.",
        )
    except Exception:
        LOGGER.exception("Scoring failed")
        raise HTTPException(
            status_code=500,
            detail="Scoring failed while running TRIBE. Check service logs for the internal error code.",
        )
    finally:
        await _finish_runtime_activity()


@app.post("/runtime/unload")
async def unload_runtime(_: str = Depends(require_auth)):
    return await _unload_pipeline("requested")


@app.on_event("startup")
async def startup():
    global _idle_unload_task
    LOGGER.info(
        "PitchScore TRIBE service starting — model=%s device=%s openrouter=%s auth=%s idle_unload=%ss",
        TRIBE_MODEL_ID,
        TRIBE_DEVICE,
        OPENROUTER_ENABLED,
        AUTH_STORE.status(),
        TRIBE_IDLE_UNLOAD_SECONDS,
    )
    if TRIBE_IDLE_UNLOAD_SECONDS > 0:
        _idle_unload_task = asyncio.create_task(_idle_unload_loop())


@app.on_event("shutdown")
async def shutdown():
    if _idle_unload_task:
        _idle_unload_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _idle_unload_task
    await _unload_pipeline("shutdown")
