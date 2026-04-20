"""TRIBE scoring engine — model loading, text scoring, and feature extraction."""
from __future__ import annotations

import hashlib
import gc
import importlib.util
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TRIBE_MODEL_ID = os.getenv("TRIBE_MODEL_ID", "facebook/tribev2")
TRIBE_DEVICE = os.getenv("TRIBE_DEVICE", "auto").strip().lower()
TRIBE_TEXT_DEVICE = os.getenv(
    "TRIBE_TEXT_DEVICE",
    "auto" if TRIBE_DEVICE in {"auto", "cuda"} else TRIBE_DEVICE,
).strip().lower()
TRIBE_TEXT_INPUT_MODE = os.getenv("TRIBE_TEXT_INPUT_MODE", "direct").strip().lower()
TRIBE_DIRECT_TEXT_LANGUAGE = os.getenv("TRIBE_DIRECT_TEXT_LANGUAGE", "english")
TRIBE_CACHE_DIR = Path(os.getenv("TRIBE_CACHE_DIR", "/models")).resolve()
TRIBE_TEXT_MODEL = os.getenv("TRIBE_TEXT_MODEL", "NousResearch/Hermes-3-Llama-3.2-3B")
TRIBE_TEXT_BATCH_SIZE = os.getenv("TRIBE_TEXT_BATCH_SIZE", "auto").strip().lower()
TRIBE_TEXT_CUDA_MIN_TOTAL_GB = float(os.getenv("TRIBE_TEXT_CUDA_MIN_TOTAL_GB", "7"))
TRIBE_TEXT_CUDA_MIN_FREE_GB = float(os.getenv("TRIBE_TEXT_CUDA_MIN_FREE_GB", "5"))
TRIBE_OOM_FALLBACK_TEXT_DEVICE = os.getenv("TRIBE_OOM_FALLBACK_TEXT_DEVICE", "cpu").strip().lower()
if TRIBE_OOM_FALLBACK_TEXT_DEVICE in {"0", "false", "none", "off"}:
    TRIBE_OOM_FALLBACK_TEXT_DEVICE = ""
TRIBE_OOM_FALLBACK_TEXT_BATCH_SIZE = max(1, int(os.getenv("TRIBE_OOM_FALLBACK_TEXT_BATCH_SIZE", "1")))
TRIBE_UNLOAD_TEXT_MODEL_AFTER_SCORE = os.getenv("TRIBE_UNLOAD_TEXT_MODEL_AFTER_SCORE", "0") == "1"
TRIBE_ACCELERATE_MAX_GPU_MEMORY_GB = os.getenv("TRIBE_ACCELERATE_MAX_GPU_MEMORY_GB", "auto").strip().lower()
TRIBE_ACCELERATE_MAX_CPU_MEMORY_GB = os.getenv("TRIBE_ACCELERATE_MAX_CPU_MEMORY_GB", "32").strip()
TRIBE_ACCELERATE_OFFLOAD_FOLDER = Path(
    os.getenv("TRIBE_ACCELERATE_OFFLOAD_FOLDER", str(TRIBE_CACHE_DIR / "offload"))
).resolve()
TRIBE_ALLOW_MOCK = os.getenv("TRIBE_ALLOW_MOCK", "0") == "1"
LOGGER = logging.getLogger(__name__)

# ── Helpers ──


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if abs(denominator) > 1e-9 else default


def band_score(value: float, lo: float, hi: float) -> float:
    """Map value in [lo, hi] to [0, 100]. Clamps to range."""
    if abs(hi - lo) < 1e-9:
        return 50.0
    return clamp((value - lo) / (hi - lo) * 100.0)


def weighted_signal(
    scores: list[tuple[float, float]],
    floor: float = 8.0,
    ceiling: float = 92.0,
) -> float:
    """Weighted average of (score, weight) pairs → compressed to [floor, ceiling]."""
    total_weight = sum(w for _, w in scores)
    if total_weight < 1e-9:
        return 50.0
    raw = sum(s * w for s, w in scores) / total_weight
    # Compress to avoid extreme scores that erode trust
    compressed = floor + (ceiling - floor) * (clamp(raw) / 100.0)
    return clamp(compressed)


# ── Mock model for testing ──


class _MockModel:
    """Deterministic mock that returns predictable features."""

    def get_events_dataframe(self, **kwargs: Any) -> Any:
        return {"mock": True}

    def predict(self, events: Any) -> np.ndarray:
        # Return a (5, 20) matrix with deterministic values
        rng = np.random.RandomState(42)
        return rng.rand(5, 20).astype(np.float32)


# ── WhisperX runtime patch ──
# tribev2 runs whisperx via `uvx whisperx` subprocess which can trigger
# click circular import errors. This patches the transcript extraction
# to run whisperx directly via subprocess with proper env isolation.

WHISPERX_CUDA_COMPUTE_TYPE = os.getenv("WHISPERX_CUDA_COMPUTE_TYPE", "float16")
WHISPERX_CPU_COMPUTE_TYPE = os.getenv("WHISPERX_CPU_COMPUTE_TYPE", "float32")
WHISPERX_CUDA_BATCH_SIZE = max(1, int(os.getenv("WHISPERX_CUDA_BATCH_SIZE", "16")))
WHISPERX_CPU_BATCH_SIZE = max(1, int(os.getenv("WHISPERX_CPU_BATCH_SIZE", "4")))


def _resolve_device() -> str:
    device = TRIBE_DEVICE
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _cuda_memory_info_gb() -> tuple[float, float] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        gib = 1024**3
        return free_bytes / gib, total_bytes / gib
    except Exception:
        return None


def _accelerate_available() -> bool:
    return importlib.util.find_spec("accelerate") is not None


def _resolve_requested_text_device(requested_device: str, model_device: str) -> str:
    device = requested_device
    if device == "auto":
        import torch
        if model_device == "cuda" and torch.cuda.is_available():
            memory = _cuda_memory_info_gb()
            if memory is None:
                device = "cuda"
            else:
                free_gb, total_gb = memory
                if total_gb >= TRIBE_TEXT_CUDA_MIN_TOTAL_GB and free_gb >= TRIBE_TEXT_CUDA_MIN_FREE_GB:
                    device = "cuda"
                else:
                    LOGGER.warning(
                        "Text model auto device chose CPU: CUDA memory free=%.2f GiB total=%.2f GiB "
                        "(minimum free=%.2f, total=%.2f)",
                        free_gb,
                        total_gb,
                        TRIBE_TEXT_CUDA_MIN_FREE_GB,
                        TRIBE_TEXT_CUDA_MIN_TOTAL_GB,
                    )
                    device = "cpu"
        else:
            device = model_device
    if device == "cuda" and model_device != "cuda":
        return model_device
    if device == "accelerate" and not _accelerate_available():
        LOGGER.warning("TRIBE_TEXT_DEVICE=accelerate requested, but accelerate is not installed; using CPU")
        return "cpu"
    return device


def _resolve_text_device(model_device: str) -> str:
    return _resolve_requested_text_device(TRIBE_TEXT_DEVICE, model_device)


def _resolve_text_batch_size(text_device: str) -> int:
    if TRIBE_TEXT_BATCH_SIZE and TRIBE_TEXT_BATCH_SIZE != "auto":
        try:
            return max(1, int(TRIBE_TEXT_BATCH_SIZE))
        except ValueError:
            LOGGER.warning("Invalid TRIBE_TEXT_BATCH_SIZE=%r; using auto", TRIBE_TEXT_BATCH_SIZE)

    if text_device in {"cuda", "accelerate"}:
        memory = _cuda_memory_info_gb()
        if memory is not None:
            _, total_gb = memory
            return 1 if total_gb < 10 else 4
        return 1
    return 4


def _accelerate_max_memory() -> dict[Any, str] | None:
    value = TRIBE_ACCELERATE_MAX_GPU_MEMORY_GB
    if value in {"", "0", "false", "none", "off"}:
        return None

    gpu_limit_gb: float | None
    if value == "auto":
        memory = _cuda_memory_info_gb()
        if memory is None:
            gpu_limit_gb = None
        else:
            _, total_gb = memory
            gpu_limit_gb = max(1.0, total_gb - 2.0)
    else:
        try:
            gpu_limit_gb = max(1.0, float(value))
        except ValueError:
            LOGGER.warning("Invalid TRIBE_ACCELERATE_MAX_GPU_MEMORY_GB=%r; disabling max_memory", value)
            gpu_limit_gb = None

    max_memory: dict[Any, str] = {}
    if gpu_limit_gb is not None:
        max_memory[0] = f"{gpu_limit_gb:.0f}GiB"

    try:
        cpu_limit_gb = max(1, int(float(TRIBE_ACCELERATE_MAX_CPU_MEMORY_GB)))
        max_memory["cpu"] = f"{cpu_limit_gb}GiB"
    except ValueError:
        LOGGER.warning("Invalid TRIBE_ACCELERATE_MAX_CPU_MEMORY_GB=%r", TRIBE_ACCELERATE_MAX_CPU_MEMORY_GB)

    return max_memory or None


def _release_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _reset_cuda_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _cuda_memory_metrics_gb() -> dict[str, float] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_gb, total_gb = _cuda_memory_info_gb() or (0.0, 0.0)
        gib = 1024**3
        return {
            "free": round(free_gb, 3),
            "total": round(total_gb, 3),
            "allocated": round(torch.cuda.memory_allocated() / gib, 3),
            "reserved": round(torch.cuda.memory_reserved() / gib, 3),
            "peak_allocated": round(torch.cuda.max_memory_allocated() / gib, 3),
            "peak_reserved": round(torch.cuda.max_memory_reserved() / gib, 3),
        }
    except Exception:
        return None


def _patch_whisperx_runtime() -> None:
    """Monkey-patch tribev2's whisperx extraction to avoid uvx subprocess issues."""
    try:
        import pandas as pd
        import tribev2.eventstransforms as eventstransforms
    except Exception:
        return

    transform = eventstransforms.ExtractWordsFromAudio
    if getattr(transform, "_pitchscore_patched", False):
        return

    def _patched_get_transcript(wav_filename: Path, language: str) -> "pd.DataFrame":
        lang_codes = {"english": "en", "french": "fr", "spanish": "es", "dutch": "nl", "chinese": "zh"}
        if language not in lang_codes:
            raise ValueError(f"Language {language} not supported")

        device = _resolve_device()
        compute_type = WHISPERX_CUDA_COMPUTE_TYPE if device == "cuda" else WHISPERX_CPU_COMPUTE_TYPE
        batch_size = WHISPERX_CUDA_BATCH_SIZE if device == "cuda" else WHISPERX_CPU_BATCH_SIZE

        with tempfile.TemporaryDirectory() as output_dir:
            cmd = [
                "uvx", "whisperx",
                str(wav_filename),
                "--model", "large-v3",
                "--language", lang_codes[language],
                "--device", device,
                "--compute_type", compute_type,
                "--batch_size", str(batch_size),
                "--output_dir", output_dir,
                "--output_format", "json",
            ]
            if language == "english":
                cmd.extend(["--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H"])
            env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                # Retry with float32 on CPU if compute type fails
                if device == "cpu" and compute_type != "float32" and "float16" in result.stderr.lower():
                    LOGGER.warning("whisperx CPU %s failed, retrying float32", compute_type)
                    cmd[cmd.index(compute_type)] = "float32"
                    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
                    if result.returncode != 0:
                        raise RuntimeError(f"whisperx failed:\n{result.stderr}")
                else:
                    raise RuntimeError(f"whisperx failed:\n{result.stderr}")

            json_path = Path(output_dir) / f"{wav_filename.stem}.json"
            transcript = json.loads(json_path.read_text())

        words = []
        for i, seg in enumerate(transcript.get("segments", [])):
            sentence = seg.get("text", "").replace('"', "")
            for w in seg.get("words", []):
                if "start" not in w:
                    continue
                words.append({
                    "text": w["word"].replace('"', ""),
                    "start": w["start"],
                    "duration": w["end"] - w["start"],
                    "sequence_id": i,
                    "sentence": sentence,
                })
        return pd.DataFrame(words)

    transform._get_transcript_from_audio = staticmethod(_patched_get_transcript)
    transform._pitchscore_patched = True
    LOGGER.info("WhisperX runtime patch applied")


def _patch_neuralset_hf_text_runtime() -> None:
    """Patch neuralset's HF loader so Accelerate gets explicit memory/offload limits."""
    try:
        import itertools

        import torch
        from neuralset.extractors.text import HuggingFaceText, part_reversal
    except Exception:
        return

    if getattr(HuggingFaceText, "_pitchscore_accelerate_patched", False):
        return

    def _load_model(self: Any, **kwargs: Any) -> Any:
        from transformers import AutoModel as Model

        if "t5" in self.model_name or "bert" in self.model_name:
            from transformers import AutoModelForTextEncoding as Model
        elif "Phi-3" in self.model_name:
            from transformers import AutoModelForCausalLM as Model
        elif "Llama-3.2-11B-Vision" in self.model_name:
            from transformers import MllamaForConditionalGeneration as Model

        if self.device == "accelerate":
            TRIBE_ACCELERATE_OFFLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
            kwargs.setdefault("device_map", "auto")
            kwargs.setdefault("torch_dtype", torch.float16)
            kwargs.setdefault("low_cpu_mem_usage", True)
            kwargs.setdefault("offload_folder", str(TRIBE_ACCELERATE_OFFLOAD_FOLDER))
            kwargs.setdefault("offload_state_dict", True)
            max_memory = _accelerate_max_memory()
            if max_memory is not None:
                kwargs.setdefault("max_memory", max_memory)

        model = Model.from_pretrained(self.model_name, **kwargs)
        if not self.pretrained:
            rawmodel = Model.from_config(model.config)
            with torch.no_grad():
                for p1, p2 in itertools.zip_longest(model.parameters(), rawmodel.parameters()):
                    p1.data = p2.to(p1)
        elif self.pretrained == "part-reversal":
            with torch.no_grad():
                for param in model.parameters():
                    part_reversal(param)
        if self.device != "accelerate":
            model.to(self.device)
        model.eval()
        return model

    HuggingFaceText._load_model = _load_model
    HuggingFaceText._pitchscore_accelerate_patched = True
    LOGGER.info("neuralset HuggingFaceText Accelerate/offload patch applied")


# ── Model singleton ──

_model: Any = None
_model_lock = threading.Lock()
_runtime_text_device_override: str | None = None
_runtime_text_batch_size_override: int | None = None
_loaded_runtime_config: dict[str, Any] = {}
_last_score_metrics: dict[str, Any] = {}
_last_score_lock = threading.Lock()

DIRECT_TEXT_TIMING_BASIS = "synthetic_word_order"
REAL_TIME_TIMING_BASIS = "real_time_seconds"


def _round_seconds(value: float) -> float:
    return round(value, 3)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_last_score_metrics(metrics: dict[str, Any]) -> None:
    global _last_score_metrics
    with _last_score_lock:
        _last_score_metrics = _sanitize_last_score_metrics(metrics)


def last_score_metrics() -> dict[str, Any]:
    with _last_score_lock:
        return dict(_last_score_metrics)


def _parse_oom_fallback_devices() -> list[str]:
    devices: list[str] = []
    for raw in TRIBE_OOM_FALLBACK_TEXT_DEVICE.split(","):
        device = raw.strip().lower()
        if not device or device in {"0", "false", "none", "off"}:
            continue
        if device not in devices:
            devices.append(device)
    return devices


def _error_code_for_exception(exc: BaseException) -> str:
    if _is_cuda_oom(exc):
        return "cuda_oom"
    name = exc.__class__.__name__
    code = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return code or "score_error"


def _sanitize_failed_attempt(attempt: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "retry_index",
        "failed_at",
        "seconds",
        "cuda_oom",
        "error_type",
        "error_code",
        "runtime",
        "cuda_memory_gb",
    }
    return {key: value for key, value in attempt.items() if key in allowed_keys}


def _sanitize_last_score_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(metrics)
    sanitized.pop("error", None)
    sanitized.pop("error_message", None)
    if "failed_attempts" in sanitized:
        sanitized["failed_attempts"] = [
            _sanitize_failed_attempt(attempt)
            for attempt in sanitized.get("failed_attempts", [])
            if isinstance(attempt, dict)
        ]
    return sanitized


def _timing_basis_for_input_mode(text_input_mode: str | None = None) -> str:
    mode = (text_input_mode or TRIBE_TEXT_INPUT_MODE).strip().lower()
    return DIRECT_TEXT_TIMING_BASIS if mode == "direct" else REAL_TIME_TIMING_BASIS


def _timing_metadata_for_input_mode(text_input_mode: str | None = None) -> dict[str, Any]:
    basis = _timing_basis_for_input_mode(text_input_mode)
    if basis == DIRECT_TEXT_TIMING_BASIS:
        return {
            "temporal_trace_basis": DIRECT_TEXT_TIMING_BASIS,
            "temporal_segment_label": "synthetic word-order segment",
            "temporal_trace_note": (
                "Direct text mode skips TTS/WhisperX. Segment order follows the pitch text; "
                "segment positions are not real elapsed seconds."
            ),
        }
    return {
        "temporal_trace_basis": REAL_TIME_TIMING_BASIS,
        "temporal_segment_label": "second",
        "temporal_trace_note": (
            "Audio/TTS timing path with speech-event alignment; segment positions are time-based."
        ),
    }


def _load_model() -> Any:
    global _model, _loaded_runtime_config
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        if TRIBE_ALLOW_MOCK:
            LOGGER.info("TRIBE_ALLOW_MOCK=1 — using mock model")
            _model = _MockModel()
            _loaded_runtime_config = {
                "device": "mock",
                "text_device": "mock",
                "text_batch_size": 0,
                "text_input_mode": TRIBE_TEXT_INPUT_MODE,
                "text_model": "mock",
                "cache_dir": str(TRIBE_CACHE_DIR),
                "accelerate_available": _accelerate_available(),
                "accelerate_max_memory": None,
                "accelerate_offload_folder": None,
            }
            return _model
        # Real model loading
        try:
            _patch_whisperx_runtime()
            _patch_neuralset_hf_text_runtime()
            from tribev2.demo_utils import TribeModel

            device = TRIBE_DEVICE
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            text_device = _resolve_requested_text_device(
                _runtime_text_device_override or TRIBE_TEXT_DEVICE,
                device,
            )
            text_batch_size = _runtime_text_batch_size_override or _resolve_text_batch_size(text_device)
            LOGGER.info(
                "Loading TRIBE model %s on %s with text features on %s batch_size=%s",
                TRIBE_MODEL_ID,
                device,
                text_device,
                text_batch_size,
            )
            _model = TribeModel.from_pretrained(
                TRIBE_MODEL_ID,
                cache_folder=str(TRIBE_CACHE_DIR),
                device=device,
                config_update={
                    "data.text_feature.model_name": TRIBE_TEXT_MODEL,
                    "data.text_feature.device": text_device,
                    "data.text_feature.batch_size": text_batch_size,
                    "data.audio_feature.device": device,
                    "data.image_feature.image.device": device,
                    "data.video_feature.image.device": device,
                },
            )
            _loaded_runtime_config = {
                "device": device,
                "text_device": text_device,
                "text_batch_size": text_batch_size,
                "text_input_mode": TRIBE_TEXT_INPUT_MODE,
                "text_model": TRIBE_TEXT_MODEL,
                "cache_dir": str(TRIBE_CACHE_DIR),
                "accelerate_available": _accelerate_available(),
                "accelerate_max_memory": _accelerate_max_memory() if text_device == "accelerate" else None,
                "accelerate_offload_folder": (
                    str(TRIBE_ACCELERATE_OFFLOAD_FOLDER) if text_device == "accelerate" else None
                ),
            }
            LOGGER.info("TRIBE model loaded successfully")
            return _model
        except Exception as exc:
            LOGGER.error("Failed to load TRIBE model: %s", exc)
            raise


def get_model() -> Any:
    return _load_model()


def is_model_loaded() -> bool:
    return _model is not None


def runtime_config() -> dict[str, Any]:
    return {
        "configured_device": TRIBE_DEVICE,
        "configured_text_device": TRIBE_TEXT_DEVICE,
        "configured_oom_fallback_text_devices": _parse_oom_fallback_devices(),
        "loaded": dict(_loaded_runtime_config),
        "model_loaded": is_model_loaded(),
        "last_score": last_score_metrics(),
    }


def unload_model(*, text_device: str | None = None, text_batch_size: int | None = None) -> None:
    global _model, _runtime_text_device_override, _runtime_text_batch_size_override, _loaded_runtime_config
    with _model_lock:
        old_model = _model
        _model = None
        _runtime_text_device_override = text_device
        _runtime_text_batch_size_override = text_batch_size
        _loaded_runtime_config = {}
    del old_model
    _release_cuda_cache()


def unload_text_model(model: Any) -> None:
    text_feature = getattr(getattr(model, "data", None), "text_feature", None)
    if text_feature is None:
        return
    for attr in ("_model", "_tokenizer"):
        if hasattr(text_feature, attr):
            try:
                delattr(text_feature, attr)
            except Exception:
                pass
    _release_cuda_cache()


def _is_cuda_oom(exc: BaseException) -> bool:
    if exc.__class__.__name__ == "OutOfMemoryError":
        return True
    message = str(exc).lower()
    return "cuda out of memory" in message or "cublas_status_alloc_failed" in message


# ── Text scoring ──


def write_text_asset(text: str) -> Path:
    """Write text to a temporary file for TRIBE processing."""
    h = hashlib.sha256(text.encode()).hexdigest()[:12]
    tmp = Path(tempfile.gettempdir()) / f"pitchscore_{h}.txt"
    tmp.write_text(text, encoding="utf-8")
    return tmp


_WORD_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)


def build_direct_text_events(text: str) -> pd.DataFrame:
    """Build TRIBE Word/Text events directly from text.

    TRIBE's public text path synthesizes TTS audio, then transcribes it back
    through WhisperX. For already-written pitch text that is both slow and
    fragile, so we create the word-level events that the text extractor needs.
    """
    from neuralset.events.transforms import AddContextToWords, RemoveMissing
    from neuralset.events.utils import standardize_events

    matches = list(_WORD_RE.finditer(text))
    if not matches:
        raise ValueError("Text contains no words to score")

    word_duration = 0.32
    word_spacing = 0.40
    rows = []
    for index, match in enumerate(matches):
        rows.append(
            {
                "type": "Word",
                "text": match.group(0),
                "start": index * word_spacing,
                "duration": word_duration,
                "timeline": "default",
                "subject": "default",
                "language": TRIBE_DIRECT_TEXT_LANGUAGE,
                "sentence": text,
                "sentence_char": match.start(),
            }
        )

    events = standardize_events(pd.DataFrame(rows))
    text_event = {
        "type": "Text",
        "text": text,
        "start": 0.0,
        "duration": float(events["stop"].max()),
        "timeline": "default",
        "subject": "default",
        "language": TRIBE_DIRECT_TEXT_LANGUAGE,
        "context": "",
    }
    events = pd.concat([events, pd.DataFrame([text_event])], ignore_index=True)
    events = standardize_events(events)
    events = AddContextToWords(
        sentence_only=False,
        max_context_len=1024,
        split_field="",
    )(events)
    events = RemoveMissing()(events)
    events = standardize_events(events)
    events.attrs.update(_timing_metadata_for_input_mode("direct"))
    events.attrs["synthetic_word_duration_seconds"] = word_duration
    events.attrs["synthetic_word_spacing_seconds"] = word_spacing
    return events


def _score_text_once(message: str, *, retry_index: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Run TRIBE text scoring and return raw prediction matrix."""
    total_start = time.perf_counter()
    events: pd.DataFrame | None = None
    _reset_cuda_peak_memory()
    load_start = time.perf_counter()
    model = get_model()
    load_seconds = time.perf_counter() - load_start
    events_start = time.perf_counter()
    if isinstance(model, _MockModel):
        events = {"mock": True}
    elif TRIBE_TEXT_INPUT_MODE == "direct":
        events = build_direct_text_events(message)
    else:
        text_path = write_text_asset(message)
        try:
            events = model.get_events_dataframe(text_path=str(text_path))
        finally:
            try:
                text_path.unlink(missing_ok=True)
            except Exception:
                pass
    events_seconds = time.perf_counter() - events_start
    event_count = int(len(events))
    word_count = int((events["type"] == "Word").sum()) if "type" in events else len(_WORD_RE.findall(message))
    timing_metadata = _timing_metadata_for_input_mode(TRIBE_TEXT_INPUT_MODE)
    if isinstance(events, pd.DataFrame):
        timing_metadata.update(
            {
                key: events.attrs[key]
                for key in (
                    "temporal_trace_basis",
                    "temporal_segment_label",
                    "temporal_trace_note",
                    "synthetic_word_duration_seconds",
                    "synthetic_word_spacing_seconds",
                )
                if key in events.attrs
            }
        )
    try:
        predict_start = time.perf_counter()
        result = model.predict(events)
        predict_seconds = time.perf_counter() - predict_start
        # model.predict returns (predictions, segments) tuple for real model
        predictions = result[0] if isinstance(result, tuple) else result
        predictions = np.asarray(predictions, dtype=np.float32)
        metrics = {
            "ok": True,
            "scored_at": _now_iso(),
            "retry_index": retry_index,
            "fallback_used": retry_index > 0,
            "input_chars": len(message),
            "word_count": word_count,
            "event_count": event_count,
            "segments": int(predictions.shape[0]) if predictions.ndim >= 1 else 0,
            "voxel_count": int(predictions.shape[1]) if predictions.ndim >= 2 else 1,
            **timing_metadata,
            "model_load_seconds": _round_seconds(load_seconds),
            "event_build_seconds": _round_seconds(events_seconds),
            "predict_seconds": _round_seconds(predict_seconds),
            "total_seconds": _round_seconds(time.perf_counter() - total_start),
            "runtime": dict(_loaded_runtime_config),
            "cuda_memory_gb": _cuda_memory_metrics_gb(),
        }
        return predictions, metrics
    finally:
        if TRIBE_UNLOAD_TEXT_MODEL_AFTER_SCORE:
            unload_text_model(model)
        else:
            _release_cuda_cache()


def score_text(message: str) -> np.ndarray:
    """Run TRIBE text scoring with limited-VRAM recovery."""
    fallback_devices = _parse_oom_fallback_devices()
    failed_attempts: list[dict[str, Any]] = []
    retry_index = 0

    while True:
        attempt_start = time.perf_counter()
        try:
            predictions, metrics = _score_text_once(message, retry_index=retry_index)
            metrics["failed_attempts"] = failed_attempts
            _set_last_score_metrics(metrics)
            LOGGER.info("TRIBE score metrics: %s", json.dumps(metrics, sort_keys=True))
            return predictions
        except Exception as exc:
            cuda_oom = _is_cuda_oom(exc)
            failed_attempt = {
                "retry_index": retry_index,
                "failed_at": _now_iso(),
                "seconds": _round_seconds(time.perf_counter() - attempt_start),
                "cuda_oom": cuda_oom,
                "error_type": exc.__class__.__name__,
                "error_code": _error_code_for_exception(exc),
                "runtime": dict(_loaded_runtime_config),
                "cuda_memory_gb": _cuda_memory_metrics_gb(),
            }
            failed_attempts.append(failed_attempt)

            next_device = None
            loaded_text_device = _loaded_runtime_config.get("text_device")
            while fallback_devices:
                candidate = fallback_devices.pop(0)
                if candidate != loaded_text_device:
                    next_device = candidate
                    break

            if not cuda_oom or next_device is None:
                _set_last_score_metrics(
                    {
                        "ok": False,
                        "scored_at": _now_iso(),
                        "input_chars": len(message),
                        "word_count": len(_WORD_RE.findall(message)),
                        "failed_attempts": failed_attempts,
                    }
                )
                raise

            retry_index += 1
            LOGGER.warning(
                "CUDA OOM while scoring; unloading TRIBE and retrying with text features on %s",
                next_device,
                exc_info=True,
            )
            unload_model(
                text_device=next_device,
                text_batch_size=TRIBE_OOM_FALLBACK_TEXT_BATCH_SIZE,
            )


# ── Feature extraction ──

FEATURE_KEYS = [
    "global_mean_abs",
    "global_peak_abs",
    "temporal_std",
    "early_mean",
    "late_mean",
    "max_temporal_delta",
    "spatial_spread",
    "focus_ratio",
    "sustain_ratio",
    "arc_ratio",
]


def _coerce_prediction_matrix(predictions: np.ndarray) -> np.ndarray:
    """Return a finite 2-D prediction matrix or raise a sanitized error.

    TRIBE should return ``(segments, voxels)``, but tests, mocks, or upstream
    failures can produce 1-D, empty, NaN, or infinite arrays.  Normalizing here
    keeps all downstream feature math deterministic and prevents malformed model
    output from leaking noisy scores into the persuasion layer.
    """
    arr = np.asarray(predictions, dtype=np.float32)
    if arr.ndim == 0:
        raise ValueError("Prediction matrix must contain at least one segment")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("Prediction matrix is empty")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def extract_features(predictions: np.ndarray) -> dict[str, float]:
    """Extract 10 raw features from TRIBE prediction matrix.

    Aligned with isthisviral's extract_feature_vector: uses quartile splits,
    ratio-based focus/spatial/arc, and fraction-based sustain/spread.
    """
    predictions = _coerce_prediction_matrix(predictions)
    abs_preds = np.abs(predictions)
    n_segments = abs_preds.shape[0]
    n_voxels = abs_preds.shape[1] if abs_preds.ndim > 1 else 1

    global_mean_abs = float(abs_preds.mean())
    global_peak_abs = float(abs_preds.max())

    # Temporal analysis — quartile splits (not halves)
    temporal_means = abs_preds.mean(axis=1)
    temporal_std = float(temporal_means.std()) if n_segments > 1 else 0.0

    q1 = max(1, n_segments // 4)
    early_mean = float(temporal_means[:q1].mean())
    late_slice = temporal_means[-q1:] if q1 < n_segments else temporal_means
    late_mean = float(late_slice.mean())

    # Max consecutive segment change
    if n_segments > 1:
        max_temporal_delta = float(np.abs(np.diff(temporal_means)).max())
    else:
        max_temporal_delta = 0.0

    # Spatial spread: fraction of voxels above mean (isthisviral formula)
    spatial_means = abs_preds.mean(axis=0)
    spatial_spread = float((spatial_means > spatial_means.mean()).mean()) if n_voxels > 1 else 0.0

    # Focus ratio: top-10% voxel mean / global mean (isthisviral formula)
    sorted_spatial = np.sort(spatial_means)[::-1]
    top_k = max(1, n_voxels // 10)
    focus_ratio = safe_ratio(float(sorted_spatial[:top_k].mean()), global_mean_abs, 1.0)

    # Sustain ratio: fraction of segments above mean activation
    sustain_ratio = float((temporal_means >= temporal_means.mean()).mean()) if n_segments > 1 else 0.5

    # Arc ratio: (max - min) / mean of temporal trace (isthisviral formula)
    if n_segments > 1:
        arc_ratio = safe_ratio(
            float(temporal_means.max() - temporal_means.min()),
            float(temporal_means.mean()),
            0.0,
        )
    else:
        arc_ratio = 0.0

    return {
        "global_mean_abs": global_mean_abs,
        "global_peak_abs": global_peak_abs,
        "temporal_std": temporal_std,
        "early_mean": early_mean,
        "late_mean": late_mean,
        "max_temporal_delta": max_temporal_delta,
        "spatial_spread": spatial_spread,
        "focus_ratio": focus_ratio,
        "sustain_ratio": sustain_ratio,
        "arc_ratio": arc_ratio,
    }


# ── fMRI Summary Output ──

def summarize_fmri_output(
    predictions: np.ndarray,
    *,
    text_input_mode: str | None = None,
) -> dict[str, Any]:
    """Extract fMRI summary for frontend visualization.

    Returns temporal trace, peaks, and top voxel data — same pattern
    as isthisviral's summarize_fmri_output.
    """
    predictions = _coerce_prediction_matrix(predictions)
    abs_preds = np.abs(predictions)
    n_segments = abs_preds.shape[0]
    n_voxels = abs_preds.shape[1] if abs_preds.ndim > 1 else 1

    # Per-segment mean activation (temporal engagement trace)
    temporal_trace = abs_preds.mean(axis=1).tolist()

    # Per-segment peak activation
    temporal_peaks = abs_preds.max(axis=1).tolist()

    # Top 6 most-activated voxels (by mean across all segments)
    spatial_means = abs_preds.mean(axis=0)
    top_n = min(6, n_voxels)
    top_indices = np.argsort(spatial_means)[::-1][:top_n]
    top_voxel_indices = top_indices.tolist()
    top_voxel_values = spatial_means[top_indices].tolist()

    return {
        "segments": n_segments,
        "voxel_count": n_voxels,
        "global_mean_abs": float(abs_preds.mean()),
        "global_peak_abs": float(abs_preds.max()),
        "temporal_trace": [round(v, 4) for v in temporal_trace],
        "temporal_peaks": [round(v, 4) for v in temporal_peaks],
        "top_voxel_indices": top_voxel_indices,
        "top_voxel_values": [round(v, 4) for v in top_voxel_values],
        **_timing_metadata_for_input_mode(text_input_mode),
    }


# ── Persuasion signal derivation ──

PERSUASION_SIGNAL_KEYS = [
    "emotional_engagement",
    "personal_relevance",
    "social_proof_potential",
    "memorability",
    "attention_capture",
    "cognitive_friction",
]

PERSUASION_SIGNAL_LABELS = {
    "emotional_engagement": "Affective Value Salience",
    "personal_relevance": "Self-Value Relevance",
    "social_proof_potential": "Social Cognition / Sharing",
    "memorability": "Encoding Potential",
    "attention_capture": "Early Attention Salience",
    "cognitive_friction": "Cognitive Friction",
}


def derive_persuasion_signals(raw_features: dict[str, float]) -> dict[str, float]:
    """Map raw TRIBE features into 6 persuasion-relevant neural signals (0-100).

    Uses ratio-normalized inputs (divided by global_mean_abs) like isthisviral,
    making scores robust to overall activation magnitude differences.
    Band ranges tuned to ratio-normalized values from real TRIBE outputs.
    """
    gma = max(raw_features.get("global_mean_abs", 0.01), 1e-9)

    # Ratio-normalize all features (isthisviral pattern)
    peak_r = raw_features.get("global_peak_abs", 0.0) / gma
    ts_r = raw_features.get("temporal_std", 0.0) / gma
    early_r = raw_features.get("early_mean", 0.0) / gma
    late_r = raw_features.get("late_mean", 0.0) / gma
    delta_r = raw_features.get("max_temporal_delta", 0.0) / gma
    ss = raw_features.get("spatial_spread", 0.0)       # already a fraction (0-1)
    fr = raw_features.get("focus_ratio", 1.0)           # already a ratio
    sr = raw_features.get("sustain_ratio", 0.5)         # already a fraction (0-1)
    ar = raw_features.get("arc_ratio", 0.0)             # range/mean ratio

    # Affective value salience:
    # High peak intensity + temporal variation = predicted reward/affect response.
    emotional_engagement = weighted_signal([
        (band_score(peak_r, 5.0, 12.0), 0.35),    # peak/mean ratio
        (band_score(ts_r, 0.05, 0.5), 0.25),      # temporal variability
        (band_score(ar, 0.1, 0.8), 0.20),          # engagement arc
        (band_score(delta_r, 0.1, 1.0), 0.20),     # max shift (emotional moments)
    ])

    # Self-value relevance:
    # Sustained activation + focused spatial pattern = predicted self/value integration.
    personal_relevance = weighted_signal([
        (band_score(sr, 0.4, 0.75), 0.35),         # sustained above-mean segments
        (band_score(fr, 2.0, 5.0), 0.30),          # top-voxel focus ratio
        (band_score(late_r, 0.8, 1.3), 0.20),      # late engagement (reflection)
        (band_score(peak_r, 6.0, 10.0), 0.15),     # peak depth
    ])

    # Social cognition / sharing potential:
    # Strong peaks + temporal dynamics = predicted social-cognitive engagement.
    # This does not mean the message has social proof; text evidence handles that.
    social_proof_potential = weighted_signal([
        (band_score(peak_r, 6.5, 11.0), 0.35),     # sharp peaks
        (band_score(delta_r, 0.15, 0.8), 0.25),    # transition moments
        (band_score(ts_r, 0.08, 0.4), 0.20),       # temporal richness
        (band_score(ss, 0.25, 0.42), 0.20),         # spatial breadth
    ])

    # Encoding potential:
    # Engagement arc + sustained + peak = predicted memory/encoding strength.
    memorability = weighted_signal([
        (band_score(ar, 0.15, 0.65), 0.30),        # dynamic range of trace
        (band_score(sr, 0.45, 0.75), 0.25),        # sustained activation
        (band_score(peak_r, 6.0, 10.0), 0.25),     # peak moments
        (band_score(fr, 2.5, 4.5), 0.20),          # focused encoding
    ])

    # Early attention salience:
    # Early activation + peak + broad spatial response = attention grab.
    attention_capture = weighted_signal([
        (band_score(early_r, 0.85, 1.25), 0.35),   # early engagement (opener)
        (band_score(peak_r, 6.0, 11.0), 0.30),     # peak salience
        (band_score(ss, 0.25, 0.42), 0.20),         # spatial breadth
        (band_score(delta_r, 0.1, 0.7), 0.15),     # onset surprise
    ])

    # Cognitive Friction (dlPFC load analogue):
    # INVERSE — high friction = bad. Low sustain + low focus + narrow spread = confusion
    cognitive_friction = weighted_signal(
        [
            (100 - band_score(sr, 0.35, 0.7), 0.35),   # low sustain = lost attention
            (100 - band_score(fr, 1.8, 4.0), 0.30),    # low focus = scattered processing
            (100 - band_score(ss, 0.22, 0.40), 0.20),   # narrow spread = shallow
            (band_score(ts_r, 0.3, 0.6), 0.15),        # high temporal noise = confusion
        ],
        floor=4.0,
        ceiling=84.0,  # friction uses tighter ceiling (isthisviral pattern)
    )

    return {
        "emotional_engagement": emotional_engagement,
        "personal_relevance": personal_relevance,
        "social_proof_potential": social_proof_potential,
        "memorability": memorability,
        "attention_capture": attention_capture,
        "cognitive_friction": cognitive_friction,
    }
