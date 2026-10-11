import os
import sys
from types import ModuleType, SimpleNamespace

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
    _patch_exca_no_value_alias,
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
    @pytest.mark.parametrize("prediction_fails", [False, True])
    def test_feature_cleanup_reloads_recreated_memmap_files(self, monkeypatch, tmp_path, prediction_fails):
        cachedict = pytest.importorskip("exca.cachedict")
        from tribe_service import engine
        cache = cachedict.CacheDict(tmp_path, cache_type="MemmapArrayFile", permissions=0o700)
        with cache.write():
            cache["first"] = np.full((2, 3), 11, dtype=np.float32)
            cache["second"] = np.full((2, 3), 22, dtype=np.float32)
        for key, value in [("first", 11), ("second", 22)]:
            np.testing.assert_array_equal(cache[key], np.full((2, 3), value))
        model = engine._MockModel()
        model.data = SimpleNamespace(text_feature=SimpleNamespace(infra=SimpleNamespace(cache_dict=cache)))
        monkeypatch.setattr(engine, "get_model", lambda: model)
        if prediction_fails:
            def fail(_):
                raise RuntimeError("predict failed")
            monkeypatch.setattr(model, "predict", fail)
            with pytest.raises(RuntimeError, match="predict failed"):
                engine._score_text_once("A valid cleanup failure test", retry_index=0)
        else:
            engine._score_text_once("A valid cleanup success test", retry_index=0)
        assert not cache
        # exca reshuffles extraction: new offsets must never read the old, unlinked inode.
        # A new writer also requires fresh JSONL reader offsets, not just fresh array maps.
        writer = cachedict.CacheDict(tmp_path, cache_type="MemmapArrayFile", permissions=0o700)
        with writer.write():
            writer["second"] = np.full((2, 3), 22, dtype=np.float32)
            writer["first"] = np.full((2, 3), 11, dtype=np.float32)
        for key, value in [("first", 11), ("second", 22)]:
            np.testing.assert_array_equal(cache[key], np.full((2, 3), value))
        assert cache.permissions == 0o700 and cache.cache_type == "MemmapArrayFile"

    def test_feature_cache_is_cleared_on_success_and_failure(self, monkeypatch):
        from tribe_service import engine
        cache = {"customer-context": np.ones((2, 3))}
        model = engine._MockModel()
        model.data = SimpleNamespace(text_feature=SimpleNamespace(infra=SimpleNamespace(cache_dict=cache)))
        monkeypatch.setattr(engine, "get_model", lambda: model)
        engine._score_text_once("A valid feature-cache cleanup test", retry_index=0)
        assert not cache
        cache["customer-context"] = np.ones((2, 3))
        def fail(_):
            raise RuntimeError("predict failed")
        monkeypatch.setattr(model, "predict", fail)
        with pytest.raises(RuntimeError, match="predict failed"):
            engine._score_text_once("A valid feature-cache failure test", retry_index=0)
        assert not cache

    @pytest.mark.parametrize("padding_side", ["left", "right"])
    def test_target_token_copy_uses_mask_with_distinct_pad_eos_ids(self, monkeypatch, padding_side):
        torch = pytest.importorskip("torch")
        text = pytest.importorskip("neuralset.extractors.text")
        from tribe_service import engine
        original = text.HuggingFaceText._get_data.fget.method
        monkeypatch.setattr(text.HuggingFaceText._get_data.fget, "method", original)
        monkeypatch.setattr(text.HuggingFaceText, "_load_model", text.HuggingFaceText._load_model)
        monkeypatch.setattr(text.HuggingFaceText, "_pitchscore_accelerate_patched", False, raising=False)
        monkeypatch.setenv("HF_HUB_OFFLINE", "0")
        engine._patch_neuralset_hf_text_runtime()
        optimized = text.HuggingFaceText._get_data.fget.method

        class Inputs(dict):
            def to(self, _):
                return self

        class Tokenizer:
            def encode(self, value, **_):
                return [128039 if word == "<eos>" else i + 1 for i, word in enumerate(value.split())]
            def __call__(self, values, **_):
                tokens = [self.encode(value) for value in values]
                width = max(map(len, tokens))
                padded, masks = [], []
                for row in tokens:
                    pads = [0] * (width - len(row))
                    padded.append(pads + row if padding_side == "left" else row + pads)
                    masks.append([int(token != 0) for token in padded[-1]])
                return Inputs(input_ids=torch.tensor(padded), attention_mask=torch.tensor(masks))

        class Model:
            def __call__(self, input_ids, **_):
                states = tuple(input_ids[..., None].float().repeat(1, 1, 3) + layer for layer in range(4))
                return type("Outputs", (dict,), {"hidden_states": states})(hidden_states=states)

        feature = SimpleNamespace(batch_size=2, device="cpu", contextualized=True, tokenizer=Tokenizer(),
                                  model=Model(), _pad_id=128039, cache_all_layers=True, cache_n_layers=None,
                                  _aggregate_tokens=lambda value: value.float().mean(dim=1),
                                  _aggregate_layers=lambda value: value.mean(axis=0))
        events = [SimpleNamespace(text="two words", context="a prefix two words"),
                  SimpleNamespace(text="short", context="short")]
        for contextualized in [True, False]:
            feature.contextualized = contextualized
            feature.batch_size = 1  # No padding: upstream gives the target-token reference.
            expected = list(original(feature, events))
            feature.batch_size = 2
            actual = list(optimized(feature, events))
            for before, after in zip(expected, actual, strict=True):
                np.testing.assert_array_equal(before, after)
        eos_events = [SimpleNamespace(text="<eos>", context=""), SimpleNamespace(text="three real tokens", context="")]
        eos_state = list(optimized(feature, eos_events))[0]
        np.testing.assert_array_equal(eos_state, np.array([[128039 + layer] * 3 for layer in range(4)]))
    def test_offline_text_model_check_uses_only_existing_cached_config(self, monkeypatch):
        from tribe_service import engine
        text, hub = ModuleType("neuralset.extractors.text"), ModuleType("huggingface_hub")

        class TextFeature:
            _REPOS = []
            _get_data = SimpleNamespace(fget=SimpleNamespace(method=None))

        text.HuggingFaceText, text.part_reversal = TextFeature, lambda _: None
        hub.try_to_load_from_cache = lambda *args: "/cached/config.json"
        monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
        monkeypatch.setitem(sys.modules, "neuralset.extractors.text", text)
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        engine._patch_neuralset_hf_text_runtime()
        assert TextFeature._REPOS == [engine.TRIBE_TEXT_MODEL]
        del TextFeature._pitchscore_accelerate_patched
        TextFeature._REPOS = []
        hub.try_to_load_from_cache = lambda *args: None
        engine._patch_neuralset_hf_text_runtime()
        assert TextFeature._REPOS == []

    def test_wrapped_cuda_oom_is_recovered_by_cpu_fallback(self, monkeypatch):
        from tribe_service import engine
        attempts = []

        def predict(message, *, retry_index):
            attempts.append(retry_index)
            if retry_index == 0:
                try:
                    raise RuntimeError("CUDA out of memory")
                except RuntimeError as cause:
                    raise RuntimeError("Model loading went wrong") from cause
            return np.ones((2, 3), dtype=np.float32), {"ok": True}

        monkeypatch.setattr(engine, "TRIBE_OOM_FALLBACK_TEXT_DEVICE", "cpu")
        monkeypatch.setattr(engine, "_score_text_once", predict)
        monkeypatch.setattr(engine, "unload_model", lambda **kwargs: None)
        result = engine.score_text("Wrapped OOM recovery regression check")
        assert result.shape == (2, 3)
        assert attempts == [0, 1]

    def test_text_model_survives_between_pitches_but_unload_option_is_respected(self, monkeypatch):
        from tribe_service import engine

        class TextFeature:
            _model = object()

        main = ModuleType("tribev2.main")
        text = ModuleType("neuralset.extractors.text")
        text.HuggingFaceText = TextFeature
        main._free_extractor_model = lambda extractor: delattr(extractor, "_model")
        monkeypatch.setitem(sys.modules, "tribev2.main", main)
        monkeypatch.setitem(sys.modules, "neuralset.extractors.text", text)
        monkeypatch.setattr(engine, "TRIBE_UNLOAD_TEXT_MODEL_AFTER_SCORE", False)
        engine._patch_tribe_text_model_lifetime()
        feature = TextFeature()
        feature._model = object()
        main._free_extractor_model(feature)
        assert hasattr(feature, "_model")
        monkeypatch.setattr(engine, "TRIBE_UNLOAD_TEXT_MODEL_AFTER_SCORE", True)
        main._free_extractor_model(feature)
        assert "_model" not in vars(feature)

    def test_patch_exca_no_value_alias_restores_legacy_path(self, monkeypatch):
        class SentinelNoValue:
            pass

        exca = ModuleType("exca")
        steps = ModuleType("exca.steps")
        base = ModuleType("exca.steps.base")
        identity = ModuleType("exca.steps.identity")
        identity.NoValue = SentinelNoValue
        exca.steps = steps
        steps.base = base
        steps.identity = identity

        monkeypatch.setitem(sys.modules, "exca", exca)
        monkeypatch.setitem(sys.modules, "exca.steps", steps)
        monkeypatch.setitem(sys.modules, "exca.steps.base", base)
        monkeypatch.setitem(sys.modules, "exca.steps.identity", identity)

        _patch_exca_no_value_alias()

        assert base.NoValue is SentinelNoValue

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
