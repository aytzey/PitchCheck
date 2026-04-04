"""TRIBE scoring engine — model loading, text scoring, and feature extraction."""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

TRIBE_MODEL_ID = os.getenv("TRIBE_MODEL_ID", "facebook/tribev2")
TRIBE_DEVICE = os.getenv("TRIBE_DEVICE", "auto")
TRIBE_CACHE_DIR = Path(os.getenv("TRIBE_CACHE_DIR", "/models")).resolve()
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


def weighted_signal(scores: list[tuple[float, float]]) -> float:
    """Weighted average of (score, weight) pairs -> 0-100."""
    total_weight = sum(w for _, w in scores)
    if total_weight < 1e-9:
        return 50.0
    return clamp(sum(s * w for s, w in scores) / total_weight)


# ── Mock model for testing ──


class _MockModel:
    """Deterministic mock that returns predictable features."""

    def get_events_dataframe(self, **kwargs: Any) -> Any:
        return {"mock": True}

    def predict(self, events: Any) -> np.ndarray:
        # Return a (5, 20) matrix with deterministic values
        rng = np.random.RandomState(42)
        return rng.rand(5, 20).astype(np.float32)


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
            from tribe_service.patch_tribev2_whisperx import patch_tribe_whisperx_runtime
            patch_tribe_whisperx_runtime()
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
    """Extract 10 raw features from TRIBE prediction matrix."""
    # predictions shape: (segments, voxels) or (time, features)
    abs_preds = np.abs(predictions)

    global_mean_abs = float(abs_preds.mean())
    global_peak_abs = float(abs_preds.max())

    # Temporal analysis (along axis 0 = time/segments)
    temporal_means = abs_preds.mean(axis=1)
    temporal_std = float(temporal_means.std()) if len(temporal_means) > 1 else 0.0

    n_segments = len(temporal_means)
    half = max(1, n_segments // 2)
    early_mean = float(temporal_means[:half].mean())
    late_slice = temporal_means[half:]
    late_mean = float(late_slice.mean()) if len(late_slice) > 0 else early_mean

    # Max change between consecutive segments
    if n_segments > 1:
        deltas = np.abs(np.diff(temporal_means))
        max_temporal_delta = float(deltas.max())
    else:
        max_temporal_delta = 0.0

    # Spatial analysis (along axis 1 = voxels/features)
    spatial_means = abs_preds.mean(axis=0)
    spatial_spread = float(spatial_means.std()) if len(spatial_means) > 1 else 0.0

    # Focus: how concentrated is activation in top voxels
    sorted_spatial = np.sort(spatial_means)[::-1]
    top_k = max(1, len(sorted_spatial) // 5)
    total = float(spatial_means.sum())
    focus_ratio = safe_ratio(float(sorted_spatial[:top_k].sum()), total, 0.5)

    # Sustain: ratio of segments above median activation
    median_val = float(np.median(temporal_means))
    sustain_ratio = safe_ratio(
        float(np.sum(temporal_means >= median_val)),
        float(n_segments),
        0.5,
    )

    # Arc: late vs early ratio (does engagement build?)
    arc_ratio = safe_ratio(late_mean, early_mean + 1e-9, 1.0)

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
    """Map raw TRIBE features into 6 persuasion-relevant neural signals (0-100)."""
    gma = raw_features.get("global_mean_abs", 0.0)
    gpa = raw_features.get("global_peak_abs", 0.0)
    ts = raw_features.get("temporal_std", 0.0)
    em = raw_features.get("early_mean", 0.0)
    lm = raw_features.get("late_mean", 0.0)
    mtd = raw_features.get("max_temporal_delta", 0.0)
    ss = raw_features.get("spatial_spread", 0.0)
    fr = raw_features.get("focus_ratio", 0.5)
    sr = raw_features.get("sustain_ratio", 0.5)
    ar = raw_features.get("arc_ratio", 1.0)

    # Emotional Engagement: high overall activation + peak intensity
    emotional_engagement = weighted_signal([
        (band_score(gma, 0.05, 0.5), 0.4),
        (band_score(gpa, 0.1, 1.0), 0.3),
        (band_score(ts, 0.01, 0.2), 0.3),
    ])

    # Personal Relevance: sustained attention + spatial focus
    personal_relevance = weighted_signal([
        (band_score(sr, 0.3, 0.8), 0.4),
        (band_score(fr, 0.15, 0.5), 0.3),
        (band_score(gma, 0.05, 0.4), 0.3),
    ])

    # Social Proof Potential: emotional peaks + temporal dynamics
    social_proof_potential = weighted_signal([
        (band_score(gpa, 0.15, 0.8), 0.4),
        (band_score(mtd, 0.02, 0.3), 0.3),
        (band_score(ts, 0.02, 0.15), 0.3),
    ])

    # Memorability: arc (builds over time) + peak + sustain
    memorability = weighted_signal([
        (band_score(ar, 0.8, 1.5), 0.4),
        (band_score(gpa, 0.1, 0.8), 0.3),
        (band_score(sr, 0.4, 0.8), 0.3),
    ])

    # Attention Capture: early activation + peak + spatial spread
    attention_capture = weighted_signal([
        (band_score(em, 0.05, 0.5), 0.4),
        (band_score(gpa, 0.1, 0.8), 0.3),
        (band_score(ss, 0.01, 0.15), 0.3),
    ])

    # Cognitive Friction: INVERSE — high is BAD. Low spatial spread + low sustain = confusion
    friction_raw = weighted_signal([
        (100 - band_score(sr, 0.3, 0.7), 0.4),
        (100 - band_score(fr, 0.15, 0.45), 0.3),
        (band_score(ss, 0.08, 0.2), 0.3),
    ])
    cognitive_friction = clamp(friction_raw)

    return {
        "emotional_engagement": emotional_engagement,
        "personal_relevance": personal_relevance,
        "social_proof_potential": social_proof_potential,
        "memorability": memorability,
        "attention_capture": attention_capture,
        "cognitive_friction": cognitive_friction,
    }
