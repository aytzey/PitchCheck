"""Optional Rust acceleration layer for TRIBE post-processing.

The Python implementation remains the source of truth and fallback path.  When
the `_pitchcheck_core` PyO3 module is installed, this module routes dense numeric
post-processing through Rust while preserving the same public Python shapes.
"""
from __future__ import annotations

import importlib
import logging
import os
from types import ModuleType
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

_NATIVE_CORE: ModuleType | None = None
_NATIVE_CORE_LOADED = False
NATIVE_UNAVAILABLE = object()
_TRUTHY = {"1", "true", "on", "yes"}
_FALSY = {"0", "false", "off", "no", "none"}


def _env_flag(name: str, default: str) -> bool:
    value = os.getenv(name, default).strip().lower()
    if value in _FALSY:
        return False
    if value in _TRUTHY:
        return True
    return default.strip().lower() in _TRUTHY


def module() -> ModuleType | None:
    global _NATIVE_CORE, _NATIVE_CORE_LOADED
    if not _env_flag("PITCHCHECK_RUST_CORE", "1"):
        return None
    if _NATIVE_CORE_LOADED:
        return _NATIVE_CORE

    _NATIVE_CORE_LOADED = True
    for module_name in ("tribe_service._pitchcheck_core", "_pitchcheck_core"):
        try:
            _NATIVE_CORE = importlib.import_module(module_name)
            LOGGER.info("Loaded PitchCheck Rust core: %s", module_name)
            return _NATIVE_CORE
        except Exception as exc:
            LOGGER.debug("PitchCheck Rust core import failed for %s: %s", module_name, exc)
    return None


def available() -> bool:
    return module() is not None


def numeric_enabled() -> bool:
    """Return whether dense numeric post-processing should use Rust.

    The NumPy fallback remains available for development and for any deployment
    where the optional PyO3 module is not installed.
    """
    return _env_flag("PITCHCHECK_RUST_NUMERIC", "1") and available()


def coerce_prediction_matrix(predictions: Any) -> np.ndarray:
    arr = np.asarray(predictions, dtype=np.float32)
    if arr.ndim == 0:
        raise ValueError("Prediction matrix must contain at least one segment")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("Prediction matrix is empty")
    arr = np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(arr, dtype=np.float32)


def extract_features(predictions: np.ndarray) -> dict[str, float] | None:
    if not numeric_enabled():
        return None
    core = module()
    if core is None:
        return None
    return dict(core.extract_features(predictions))


def summarize_fmri_output(
    predictions: np.ndarray,
    text_input_mode: str,
) -> dict[str, Any] | None:
    if not numeric_enabled():
        return None
    core = module()
    if core is None:
        return None
    return dict(core.summarize_fmri_output(predictions, text_input_mode))


def derive_persuasion_signals(raw_features: dict[str, float]) -> dict[str, float] | None:
    if not numeric_enabled():
        return None
    core = module()
    if core is None:
        return None
    return dict(core.derive_persuasion_signals(raw_features))


def prediction_analysis(
    predictions: np.ndarray,
    text_input_mode: str,
) -> tuple[dict[str, float], dict[str, Any], dict[str, float]] | None:
    if not numeric_enabled():
        return None
    core = module()
    if core is None:
        return None
    result = dict(core.prediction_analysis(predictions, text_input_mode))
    return (
        dict(result["raw_features"]),
        dict(result["fmri_summary"]),
        dict(result["neural_signals"]),
    )


def extract_balanced_json_object(content: str) -> str | None | object:
    core = module()
    if core is None:
        return NATIVE_UNAVAILABLE
    return core.extract_balanced_json_object(content)
