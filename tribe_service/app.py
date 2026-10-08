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
from starlette.responses import JSONResponse

from tribe_service.schemas import (
    AuthChangePasswordRequest,
    AuthLoginRequest,
    PitchRefineRequest,
    PitchRefineResponse,
    PitchScoreRequest,
    PitchScoreReport,
    BreakdownSection,
    FmriOutput,
    NeuralSignal,
    RewriteSuggestion,
    TopMove,
)
from tribe_service.engine import (
    score_text,
    analyze_predictions,
    is_model_loaded,
    is_text_model_loaded,
    load_runtime_models,
    runtime_config,
    unload_model,
    PERSUASION_SIGNAL_LABELS,
    TRIBE_DEVICE,
    TRIBE_MODEL_ID,
    TRIBE_TEXT_INPUT_MODE,
    TRIBE_ALLOW_MOCK,
)
from tribe_service.llm_layer import (
    interpret_persuasion,
    refine_pitch_message,
    select_tribe_refinement,
    _augment_persuasion_evidence,
    _openrouter_enabled,
    plan_tribe_refinement,
    OPENROUTER_ENABLED,
)
from tribe_service.persuasion_features import calibration_quality_weight, evidence_score_from_analysis, neuro_axes_from_analysis
from tribe_service.auth import (
    AUTH_STORE,
    AuthConfigurationError,
    InvalidCredentialUpdateError,
    InvalidCredentialsError,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _env_int(name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _env_float(name: str, default: float, minimum: float) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except ValueError:
        return default


TRIBE_SCORE_TIMEOUT_SECONDS = _env_float("TRIBE_SCORE_TIMEOUT_SECONDS", 900.0, 1.0)
TRIBE_SCORE_QUEUE_TIMEOUT_SECONDS = _env_float("TRIBE_SCORE_QUEUE_TIMEOUT_SECONDS", 30.0, 0.1)
TRIBE_MAX_SCORE_CONCURRENCY = _env_int("TRIBE_MAX_SCORE_CONCURRENCY", 1, 1)
TRIBE_MAX_LLM_CONCURRENCY = _env_int("TRIBE_MAX_LLM_CONCURRENCY", 2, 1)
TRIBE_LLM_TIMEOUT_SECONDS = _env_float("TRIBE_LLM_TIMEOUT_SECONDS", 180.0, 1.0)
TRIBE_IDLE_UNLOAD_SECONDS = _env_float("TRIBE_IDLE_UNLOAD_SECONDS", 600.0, 0.0)
MAX_REQUEST_BODY_BYTES = _env_int("PITCHCHECK_MAX_REQUEST_BODY_BYTES", 128 * 1024, 1024)

_score_lock = asyncio.Semaphore(TRIBE_MAX_SCORE_CONCURRENCY)
_llm_lock = asyncio.Semaphore(TRIBE_MAX_LLM_CONCURRENCY)
_worker_tasks: set[asyncio.Task] = set()
_bearer = HTTPBearer(auto_error=False)
_pipeline_lock = asyncio.Lock()
_active_scores = 0
_last_runtime_activity = time.monotonic()
_idle_unload_task: asyncio.Task | None = None


class ScoreQueueTimeoutError(TimeoutError):
    """Raised when a score request cannot enter the bounded queue in time."""


class ScoreRunTimeoutError(TimeoutError):
    """Raised when TRIBE keeps running past the client-facing timeout."""


class RequestBodyLimitMiddleware:
    """Bound JSON bodies before FastAPI buffers or validates them, including chunked requests."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        body = bytearray()
        try:
            async with asyncio.timeout(15):
                while True:
                    event = await receive()
                    if event["type"] == "http.disconnect":
                        return
                    chunk = event.get("body", b"")
                    if len(body) + len(chunk) > MAX_REQUEST_BODY_BYTES:
                        return await JSONResponse({"detail": "Request body too large."}, 413)(scope, receive, send)
                    body.extend(chunk)
                    if not event.get("more_body", False):
                        break
        except TimeoutError:
            return await JSONResponse({"detail": "Request body timed out."}, 408)(scope, receive, send)

        delivered = False

        async def buffered_receive():
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, buffered_receive, send)


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    global _idle_unload_task
    LOGGER.info(
        "PitchScore TRIBE service starting - model=%s device=%s openrouter=%s auth=%s idle_unload=%ss",
        TRIBE_MODEL_ID,
        TRIBE_DEVICE,
        OPENROUTER_ENABLED,
        AUTH_STORE.status(),
        TRIBE_IDLE_UNLOAD_SECONDS,
    )
    if TRIBE_IDLE_UNLOAD_SECONDS > 0:
        _idle_unload_task = asyncio.create_task(_idle_unload_loop())
    try:
        yield
    finally:
        if _idle_unload_task:
            _idle_unload_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await _idle_unload_task
            _idle_unload_task = None
        if _worker_tasks:
            await asyncio.gather(*tuple(_worker_tasks), return_exceptions=True)
        await _unload_pipeline("shutdown")


app = FastAPI(
    title="PitchScore TRIBE Service",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)
app.add_middleware(RequestBodyLimitMiddleware)

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
            "text_model_loaded": is_text_model_loaded(),
            "active_scores": _active_scores,
            "idle_for_seconds": round(_idle_for_seconds(), 3),
            "idle_unload_seconds": TRIBE_IDLE_UNLOAD_SECONDS,
        }


async def _unload_pipeline(reason: str) -> dict:
    # The worker keeps the pipeline lock even if the phone disconnects mid-unload.
    task = asyncio.create_task(_perform_unload_pipeline(reason))
    _worker_tasks.add(task)
    task.add_done_callback(_worker_tasks.discard)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        task.add_done_callback(_log_detached_worker_result)
        raise


async def _perform_unload_pipeline(reason: str) -> dict:
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


def _log_detached_worker_result(task: asyncio.Task) -> None:
    try:
        exception = task.exception()
    except asyncio.CancelledError:
        exception = None
    if exception is not None:
        LOGGER.warning(
            "Detached worker finished with %s",
            exception.__class__.__name__,
        )


async def _run_with_backpressure(function, *args, lock, timeout, track_runtime=False, **kwargs):
    try:
        await asyncio.wait_for(
            lock.acquire(),
            timeout=TRIBE_SCORE_QUEUE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise ScoreQueueTimeoutError from exc

    async def run_worker():
        runtime_started = False
        try:
            if track_runtime:
                await _begin_runtime_activity()
                runtime_started = True
            return await run_in_threadpool(function, *args, **kwargs)
        finally:
            if runtime_started:
                await _finish_runtime_activity()
            lock.release()

    # The worker owns its slot; HTTP timeout/cancellation cannot stop a running thread.
    score_task = asyncio.create_task(run_worker())
    _worker_tasks.add(score_task)
    score_task.add_done_callback(_worker_tasks.discard)
    try:
        return await asyncio.wait_for(
            asyncio.shield(score_task),
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        score_task.add_done_callback(_log_detached_worker_result)
        raise ScoreRunTimeoutError from exc
    except asyncio.CancelledError:
        score_task.add_done_callback(_log_detached_worker_result)
        raise


async def _score_text_with_backpressure(message: str):
    return await _run_with_backpressure(
        score_text, message, lock=_score_lock, timeout=TRIBE_SCORE_TIMEOUT_SECONDS,
        track_runtime=True,
    )


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


def require_auth(
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
        "score_queue_timeout_seconds": TRIBE_SCORE_QUEUE_TIMEOUT_SECONDS,
        "openrouter_enabled": OPENROUTER_ENABLED,
    }


@app.post("/auth/login")
def auth_login(request: AuthLoginRequest):
    try:
        return AUTH_STORE.login(request.username, request.password)
    except Exception as error:
        raise _auth_error(error) from error


@app.post("/auth/change-password")
def auth_change_password(
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


@app.post("/auth/logout")
def auth_logout(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    _: str = Depends(require_auth),
):
    AUTH_STORE.logout(_token_from_credentials(credentials))
    return {"ok": True}


@app.post("/score")
async def score_pitch(request: PitchScoreRequest, _: str = Depends(require_auth)):
    """Score a sales pitch for persuasion effectiveness against a target persona."""
    try:
        # 1. Run TRIBE text scoring
        predictions = await _score_text_with_backpressure(request.message)

        # 2. Extract raw features, fMRI summary, and neural signals.
        raw_features, fmri_data, neural_signals = await run_in_threadpool(
            analyze_predictions, predictions,
            text_input_mode=TRIBE_TEXT_INPUT_MODE,
        )

        # 3. LLM interpretation or deterministic neural-only report.
        llm_result = await _run_with_backpressure(
            interpret_persuasion,
            lock=_llm_lock, timeout=TRIBE_LLM_TIMEOUT_SECONDS,
            message=request.message,
            persona=request.persona,
            platform=request.platform,
            neural_signals=neural_signals,
            raw_features=raw_features,
            fmri_summary=fmri_data,
            openrouter_model=request.open_router_model,
        )

        # 4. Assemble PitchScoreReport
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

        top_moves = [
            TopMove(
                priority=min(3, max(1, int(m.get("priority", i + 1)))),
                title=str(m.get("title", "")),
                do=str(m.get("do", "")),
                because=str(m.get("because", "")),
                principle=str(m.get("principle", "")),
            )
            for i, m in enumerate(llm_result.get("top_moves", [])[:3])
            if isinstance(m, dict) and m.get("title") and m.get("do")
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
            top_moves=top_moves,
            context_fit=llm_result.get("context_fit"),
            fmri_output=FmriOutput(**fmri_data),
            persuasion_evidence=llm_result.get("persuasion_evidence"),
            robustness=llm_result.get("robustness"),
            platform=request.platform,
            scored_at=datetime.now(timezone.utc).isoformat(),
        )

        return {"report": report.model_dump()}

    except ScoreQueueTimeoutError:
        LOGGER.warning("Scoring queue timed out")
        raise HTTPException(
            status_code=429,
            headers={"Retry-After": "5"},
            detail=(
                "Scoring service is busy. Try again shortly or increase "
                "TRIBE_SCORE_QUEUE_TIMEOUT_SECONDS."
            ),
        )
    except ScoreRunTimeoutError:
        LOGGER.warning("Scoring timed out; the running worker retains its slot until completion")
        raise HTTPException(
            status_code=504,
            detail=(
                "Scoring timed out while processing the request. "
                "Try again shortly, use a shorter message, or reconnect the runtime."
            ),
        )
    except Exception as exc:
        LOGGER.error("Scoring failed (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail="Scoring failed while running TRIBE. Check service logs for the internal error code.",
        )


def _measure_refine_candidates(message: str, candidates: list[str], persona: str, platform: str,
                               baseline: dict | None = None) -> list[dict]:
    measurements = [baseline] if baseline is not None else []
    texts = candidates if baseline is not None else [message, *candidates]
    for index, text in enumerate(texts, start=1 if baseline is not None else 0):
        predictions = score_text(text)
        raw, fmri, signals = analyze_predictions(predictions, text_input_mode=TRIBE_TEXT_INPUT_MODE)
        evidence = _augment_persuasion_evidence(text, persona, platform, raw, fmri)
        measurements.append({
            "id": "original" if index == 0 else f"c{index}", "message": text,
            "model_id": TRIBE_MODEL_ID, "mode": "mock" if TRIBE_ALLOW_MOCK else "model",
            "neural_score": round(evidence_score_from_analysis(signals, evidence), 3),
            "quality_weight": calibration_quality_weight(evidence),
            "neural_signals": signals, "neuro_axes": neuro_axes_from_analysis(signals),
            "voxel_count": fmri["voxel_count"], "segments": fmri["segments"],
            "temporal_trace": fmri.get("temporal_trace", []),
            "temporal_trace_basis": fmri.get("temporal_trace_basis", "unknown"),
            "text_feature_model": fmri.get("text_feature_model"),
            "expected_text_feature_model": fmri.get("expected_text_feature_model"),
            "text_feature_compatible": fmri.get("text_feature_compatible") is True,
            "response_features": {key: raw[key] for key in (
                "global_mean_abs", "global_peak_abs", "focus_ratio", "spatial_spread", "sustain_ratio", "arc_ratio",
            ) if key in raw},
        })
    return measurements


@app.post("/refine")
async def refine_pitch(request: PitchRefineRequest, _: str = Depends(require_auth)):
    """Generate drafts, measure each with TRIBE, then select a context-validated measured draft."""
    try:
        baseline = strategy = None
        if request.jev_strategy:
            if not _openrouter_enabled(request.open_router_model):
                raise RuntimeError("OpenRouter API key is missing.")
            baseline = (await _run_with_backpressure(
                _measure_refine_candidates, request.message, [], request.persona, request.platform,
                lock=_score_lock, timeout=TRIBE_SCORE_TIMEOUT_SECONDS, track_runtime=True,
            ))[0]
            strategy = await _run_with_backpressure(
                plan_tribe_refinement, request.message, request.persona, request.platform, baseline,
                [item.model_dump() for item in request.clarification_answers],
                lock=_llm_lock, timeout=TRIBE_LLM_TIMEOUT_SECONDS,
            )
        result = await _run_with_backpressure(
            refine_pitch_message,
            lock=_llm_lock, timeout=TRIBE_LLM_TIMEOUT_SECONDS,
            message=request.message,
            persona=request.persona,
            platform=request.platform,
            suggestions=request.suggestions,
            clarification_answers=[item.model_dump() for item in request.clarification_answers],
            clarification_round=request.clarification_round,
            force_rewrite=request.force_rewrite,
            openrouter_model=request.open_router_model,
            **({"decision_strategy": strategy} if strategy is not None else {}),
        )
        if not result.get("needs_clarification"):
            measurements = await _run_with_backpressure(
                _measure_refine_candidates, request.message, result["candidates"], request.persona, request.platform,
                baseline,
                lock=_score_lock, timeout=TRIBE_SCORE_TIMEOUT_SECONDS, track_runtime=True,
            )
            result = await _run_with_backpressure(
                select_tribe_refinement, request.message, request.persona, request.platform,
                request.suggestions, result, measurements,
                [item.model_dump() for item in request.clarification_answers],
                lock=_llm_lock, timeout=TRIBE_LLM_TIMEOUT_SECONDS,
            )
        return PitchRefineResponse(**result).model_dump()
    except ScoreQueueTimeoutError as exc:
        raise HTTPException(status_code=429, detail="Refine service is busy. Try again shortly.",
                            headers={"Retry-After": "5"}) from exc
    except ScoreRunTimeoutError as exc:
        raise HTTPException(status_code=504, detail="Refine service timed out.") from exc
    except RuntimeError as exc:
        missing_key = "API key is missing" in str(exc)
        status_code = 503 if missing_key else 502
        detail = "OpenRouter API key is missing." if missing_key else "Refine candidates could not be measured and validated."
        LOGGER.warning("Refine failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except Exception as exc:
        LOGGER.error("Refine failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Refine failed while evaluating candidates.")


@app.post("/runtime/load")
async def load_runtime(_: str = Depends(require_auth)):
    try:
        await _run_with_backpressure(
            load_runtime_models, lock=_score_lock, timeout=TRIBE_SCORE_TIMEOUT_SECONDS,
            track_runtime=True,
        )
    except ScoreQueueTimeoutError as exc:
        raise HTTPException(status_code=429, detail="Runtime is busy. Try again shortly.") from exc
    except ScoreRunTimeoutError as exc:
        raise HTTPException(status_code=504, detail="Model loading is still running. Check runtime status.") from exc
    except Exception as exc:
        LOGGER.error("Runtime load failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Model could not be loaded within the available resources.") from exc
    return {"ok": True, "pipeline": await _pipeline_status(), "runtime": runtime_config()}


@app.post("/runtime/unload")
async def unload_runtime(_: str = Depends(require_auth)):
    return await _unload_pipeline("requested")
