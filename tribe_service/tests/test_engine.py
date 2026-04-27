import os
import numpy as np
import pytest

os.environ["TRIBE_ALLOW_MOCK"] = "1"

from tribe_service.engine import (
    _MockModel,
    analyze_predictions,
    band_score,
    clamp,
    derive_persuasion_signals,
    extract_features,
    last_score_metrics,
    runtime_config,
    score_text,
    safe_ratio,
    summarize_fmri_output,
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

    def test_repeated_message_uses_prediction_cache(self):
        message = "Unique cache test pitch with concrete proof and Tuesday CTA"

        first = score_text(message)
        first_metrics = last_score_metrics()
        second = score_text(message)
        second_metrics = last_score_metrics()

        assert np.array_equal(first, second)
        assert first_metrics["cache_hit"] is False
        assert second_metrics["cache_hit"] is True
        assert runtime_config()["prediction_cache_entries"] >= 1

    def test_failed_score_metrics_do_not_expose_exception_text(self, monkeypatch):
        class FailingModel(_MockModel):
            def predict(self, events):
                raise RuntimeError("secret customer pitch phrase")

        monkeypatch.setattr("tribe_service.engine.get_model", lambda: FailingModel())

        with pytest.raises(RuntimeError):
            score_text("This customer pitch contains sensitive launch copy")

        metrics = last_score_metrics()
        assert metrics["ok"] is False
        assert "secret customer pitch phrase" not in repr(metrics)
        assert "sensitive launch copy" not in repr(metrics)
        assert metrics["failed_attempts"][0]["error_type"] == "RuntimeError"
        assert metrics["failed_attempts"][0]["error_code"] == "runtime_error"
        assert "error" not in metrics["failed_attempts"][0]


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

    def test_summarize_fmri_output_labels_direct_trace_as_synthetic(self):
        preds = np.random.RandomState(42).rand(5, 20).astype(np.float32)
        summary = summarize_fmri_output(preds, text_input_mode="direct")

        assert summary["temporal_trace_basis"] == "synthetic_word_order"
        assert summary["temporal_segment_label"] == "synthetic word-order segment"
        assert "not real elapsed seconds" in summary["temporal_trace_note"]
        assert summary["response_kind"] == "tribe_predicted_fmri_analogue"
        assert summary["prediction_subject_basis"] == "average_subject"
        assert summary["cortical_mesh"] == "fsaverage5"
        assert summary["hemodynamic_lag_seconds"] == 5.0

    def test_summarize_fmri_output_labels_tts_trace_as_real_time(self):
        preds = np.random.RandomState(42).rand(5, 20).astype(np.float32)
        summary = summarize_fmri_output(preds, text_input_mode="tts")

        assert summary["temporal_trace_basis"] == "real_time_seconds"
        assert summary["temporal_segment_label"] == "second"


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

    def test_malformed_raw_features_do_not_create_nan_signals(self):
        signals = derive_persuasion_signals({
            "global_mean_abs": float("nan"),
            "global_peak_abs": float("inf"),
            "temporal_std": -1.0,
            "early_mean": None,
            "late_mean": "bad",
            "max_temporal_delta": float("-inf"),
            "spatial_spread": 4.0,
            "focus_ratio": float("nan"),
            "sustain_ratio": -3.0,
            "arc_ratio": float("inf"),
        })

        assert set(signals.keys()) == set(PERSUASION_SIGNAL_KEYS)
        for val in signals.values():
            assert np.isfinite(val)
            assert 0.0 <= val <= 100.0

    def test_extract_features_sanitizes_nan_and_1d_predictions(self):
        features = extract_features(np.array([1.0, np.nan, np.inf], dtype=np.float32))

        assert set(features.keys()) == set(FEATURE_KEYS)
        assert features["global_mean_abs"] >= 0.0
        assert features["temporal_std"] == 0.0

    def test_analyze_predictions_matches_separate_post_processing(self):
        preds = np.random.RandomState(7).rand(6, 30).astype(np.float32)

        raw_features, fmri_summary, neural_signals = analyze_predictions(
            preds,
            text_input_mode="direct",
        )

        assert raw_features == extract_features(preds)
        assert fmri_summary == summarize_fmri_output(preds, text_input_mode="direct")
        assert neural_signals == derive_persuasion_signals(raw_features)
