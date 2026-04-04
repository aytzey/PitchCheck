"""TRIBE scoring engine — model loading, text scoring, and feature extraction."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

TRIBE_MODEL_ID = os.getenv("TRIBE_MODEL_ID", "facebook/tribev2")
TRIBE_DEVICE = os.getenv("TRIBE_DEVICE", "auto")
TRIBE_CACHE_DIR = Path(os.getenv("TRIBE_CACHE_DIR", "/models")).resolve()
TRIBE_TEXT_MODEL = os.getenv("TRIBE_TEXT_MODEL", "NousResearch/Hermes-3-Llama-3.2-3B")
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


# ── Model singleton ──

_model: Any = None
_model_lock = threading.Lock()


def _load_model() -> Any:
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        if TRIBE_ALLOW_MOCK:
            LOGGER.info("TRIBE_ALLOW_MOCK=1 — using mock model")
            _model = _MockModel()
            return _model
        # Real model loading
        try:
            _patch_whisperx_runtime()
            from tribev2.demo_utils import TribeModel

            device = TRIBE_DEVICE
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            LOGGER.info("Loading TRIBE model %s on %s", TRIBE_MODEL_ID, device)
            _model = TribeModel.from_pretrained(
                TRIBE_MODEL_ID,
                cache_folder=str(TRIBE_CACHE_DIR),
                device=device,
                config_update={
                    "data.text_feature.model_name": TRIBE_TEXT_MODEL,
                    "data.text_feature.device": device,
                    "data.audio_feature.device": device,
                    "data.image_feature.image.device": device,
                    "data.video_feature.image.device": device,
                },
            )
            LOGGER.info("TRIBE model loaded successfully")
            return _model
        except Exception as exc:
            LOGGER.error("Failed to load TRIBE model: %s", exc)
            raise


def get_model() -> Any:
    return _load_model()


def is_model_loaded() -> bool:
    return _model is not None


# ── Text scoring ──


def write_text_asset(text: str) -> Path:
    """Write text to a temporary file for TRIBE processing."""
    h = hashlib.sha256(text.encode()).hexdigest()[:12]
    tmp = Path(tempfile.gettempdir()) / f"pitchscore_{h}.txt"
    tmp.write_text(text, encoding="utf-8")
    return tmp


def score_text(message: str) -> np.ndarray:
    """Run TRIBE text scoring and return raw prediction matrix."""
    model = get_model()
    text_path = write_text_asset(message)
    try:
        events = model.get_events_dataframe(text_path=str(text_path))
        result = model.predict(events)
        # model.predict returns (predictions, segments) tuple for real model
        predictions = result[0] if isinstance(result, tuple) else result
        return np.asarray(predictions, dtype=np.float32)
    finally:
        try:
            text_path.unlink(missing_ok=True)
        except Exception:
            pass


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


def extract_features(predictions: np.ndarray) -> dict[str, float]:
    """Extract 10 raw features from TRIBE prediction matrix.

    Aligned with isthisviral's extract_feature_vector: uses quartile splits,
    ratio-based focus/spatial/arc, and fraction-based sustain/spread.
    """
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

def summarize_fmri_output(predictions: np.ndarray) -> dict[str, Any]:
    """Extract fMRI summary for frontend visualization.

    Returns temporal trace, peaks, and top voxel data — same pattern
    as isthisviral's summarize_fmri_output.
    """
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
    "emotional_engagement": "Emotional Engagement",
    "personal_relevance": "Personal Relevance",
    "social_proof_potential": "Social Proof Potential",
    "memorability": "Memorability",
    "attention_capture": "Attention Capture",
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

    # Emotional Engagement (MPFC activation analogue):
    # High peak intensity + temporal variation = emotional processing
    emotional_engagement = weighted_signal([
        (band_score(peak_r, 5.0, 12.0), 0.35),    # peak/mean ratio
        (band_score(ts_r, 0.05, 0.5), 0.25),      # temporal variability
        (band_score(ar, 0.1, 0.8), 0.20),          # engagement arc
        (band_score(delta_r, 0.1, 1.0), 0.20),     # max shift (emotional moments)
    ])

    # Personal Relevance (self-referential processing analogue):
    # Sustained activation + focused spatial pattern = deep processing
    personal_relevance = weighted_signal([
        (band_score(sr, 0.4, 0.75), 0.35),         # sustained above-mean segments
        (band_score(fr, 2.0, 5.0), 0.30),          # top-voxel focus ratio
        (band_score(late_r, 0.8, 1.3), 0.20),      # late engagement (reflection)
        (band_score(peak_r, 6.0, 10.0), 0.15),     # peak depth
    ])

    # Social Proof Potential (TPJ/mentalizing analogue):
    # Strong peaks + temporal dynamics = social-cognitive engagement
    social_proof_potential = weighted_signal([
        (band_score(peak_r, 6.5, 11.0), 0.35),     # sharp peaks
        (band_score(delta_r, 0.15, 0.8), 0.25),    # transition moments
        (band_score(ts_r, 0.08, 0.4), 0.20),       # temporal richness
        (band_score(ss, 0.25, 0.42), 0.20),         # spatial breadth
    ])

    # Memorability (temporal pole / hippocampal analogue):
    # Engagement arc + sustained + peak = encoding strength
    memorability = weighted_signal([
        (band_score(ar, 0.15, 0.65), 0.30),        # dynamic range of trace
        (band_score(sr, 0.45, 0.75), 0.25),        # sustained activation
        (band_score(peak_r, 6.0, 10.0), 0.25),     # peak moments
        (band_score(fr, 2.5, 4.5), 0.20),          # focused encoding
    ])

    # Attention Capture (salience network analogue):
    # Early activation + peak + broad spatial response = attention grab
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
