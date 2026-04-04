import os
import numpy as np
import pytest

os.environ["TRIBE_ALLOW_MOCK"] = "1"

from tribe_service.engine import (
    band_score,
    clamp,
    derive_persuasion_signals,
    extract_features,
    score_text,
    safe_ratio,
    weighted_signal,
    FEATURE_KEYS,
    PERSUASION_SIGNAL_KEYS,
)


class TestHelpers:
    def test_clamp_within_range(self):
        assert clamp(50.0) == 50.0

    def test_clamp_below(self):
        assert clamp(-10.0) == 0.0

    def test_clamp_above(self):
        assert clamp(150.0) == 100.0

    def test_band_score_boundaries(self):
        assert band_score(0.0, 0.0, 1.0) == 0.0
        assert band_score(0.5, 0.0, 1.0) == 50.0
        assert band_score(1.0, 0.0, 1.0) == 100.0

    def test_band_score_clamps(self):
        assert band_score(-1.0, 0.0, 1.0) == 0.0
        assert band_score(2.0, 0.0, 1.0) == 100.0

    def test_band_score_equal_range(self):
        assert band_score(5.0, 5.0, 5.0) == 50.0

    def test_safe_ratio_normal(self):
        assert safe_ratio(10.0, 5.0) == 2.0

    def test_safe_ratio_zero_denom(self):
        assert safe_ratio(10.0, 0.0, -1.0) == -1.0

    def test_weighted_signal(self):
        result = weighted_signal([(100.0, 1.0), (0.0, 1.0)])
        assert abs(result - 50.0) < 1e-6


class TestScoreText:
    def test_returns_ndarray(self):
        result = score_text("This is a test pitch for a product launch")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2

    def test_returns_float32(self):
        result = score_text("Another test pitch message here")
        assert result.dtype == np.float32


class TestExtractFeatures:
    def test_returns_all_keys(self):
        preds = np.random.RandomState(42).rand(5, 20).astype(np.float32)
        features = extract_features(preds)
        assert set(features.keys()) == set(FEATURE_KEYS)
        assert len(features) == 10

    def test_all_values_are_floats(self):
        preds = np.random.RandomState(42).rand(5, 20).astype(np.float32)
        features = extract_features(preds)
        for key, val in features.items():
            assert isinstance(val, float), f"{key} is not float: {type(val)}"

    def test_single_segment(self):
        preds = np.random.RandomState(42).rand(1, 20).astype(np.float32)
        features = extract_features(preds)
        assert len(features) == 10
        assert features["temporal_std"] == 0.0


class TestDerivePersuasionSignals:
    def test_returns_all_keys(self):
        preds = np.random.RandomState(42).rand(5, 20).astype(np.float32)
        raw = extract_features(preds)
        signals = derive_persuasion_signals(raw)
        assert set(signals.keys()) == set(PERSUASION_SIGNAL_KEYS)
        assert len(signals) == 6

    def test_all_values_in_range(self):
        preds = np.random.RandomState(42).rand(5, 20).astype(np.float32)
        raw = extract_features(preds)
        signals = derive_persuasion_signals(raw)
        for key, val in signals.items():
            assert isinstance(val, float), f"{key} is not float"
            assert 0.0 <= val <= 100.0, f"{key}={val} out of [0,100]"

    def test_empty_features_doesnt_crash(self):
        signals = derive_persuasion_signals({})
        assert len(signals) == 6
        for val in signals.values():
            assert 0.0 <= val <= 100.0
