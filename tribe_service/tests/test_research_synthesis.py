"""Tests for the deterministic TRIBE × research synthesis layer."""
from __future__ import annotations

from tribe_service.research_synthesis import (
    build_tribe_synthesis,
    localize_pitch_segments,
    synthesize_research_findings,
)


def _axes(self_value=50, reward=50, social=50, encoding=50, fluency=50):
    return {
        "self_value": {"score": self_value},
        "reward_affect": {"score": reward},
        "social_sharing": {"score": social},
        "encoding_attention": {"score": encoding},
        "processing_fluency": {"score": fluency},
    }


def _fmri(trace):
    return {"temporal_trace": trace}


class TestAxisFindings:
    def test_weak_self_value_yields_falk_gap_first(self):
        synthesis = synthesize_research_findings(
            _axes(self_value=38, fluency=42, encoding=40),
        )

        keys = [item["key"] for item in synthesis["items"]]
        assert "self_value_gap" in keys
        assert "fluency_gap" in keys
        assert "encoding_gap" in keys
        # Gaps are ordered by severity: weakest axis first.
        assert keys[0] == "self_value_gap"
        falk = next(item for item in synthesis["items"] if item["key"] == "self_value_gap")
        assert "Falk" in falk["citation"]
        assert falk["kind"] == "gap"
        assert falk["lever"]

    def test_strong_self_value_is_a_protected_strength(self):
        synthesis = synthesize_research_findings(_axes(self_value=78))

        strength = next(item for item in synthesis["items"] if item["key"] == "self_value_strength")
        assert strength["kind"] == "strength"
        assert "Scholz" in strength["citation"]

    def test_reward_dominance_sets_route_hint(self):
        synthesis = synthesize_research_findings(_axes(reward=72, social=50))

        assert synthesis["route_hint"] == "reward_led"
        assert any(item["key"] == "reward_route_dominant" for item in synthesis["items"])

    def test_social_dominance_sets_route_hint(self):
        synthesis = synthesize_research_findings(_axes(reward=48, social=68))

        assert synthesis["route_hint"] == "social_led"
        item = next(item for item in synthesis["items"] if item["key"] == "social_route_dominant")
        assert "Baek" in item["citation"] or "Scholz" in item["citation"]

    def test_balanced_axes_have_balanced_route(self):
        synthesis = synthesize_research_findings(_axes())

        assert synthesis["route_hint"] == "balanced"

    def test_items_are_capped_and_safe_on_garbage_axes(self):
        synthesis = synthesize_research_findings({"self_value": {"score": "broken"}})

        assert isinstance(synthesis["items"], list)
        assert len(synthesis["items"]) <= 5
        assert synthesis["temporal_archetype"] is None


class TestTemporalArchetypes:
    def test_flat_trace(self):
        synthesis = synthesize_research_findings(_axes(), _fmri([0.30, 0.31, 0.30, 0.31, 0.30]))
        assert synthesis["temporal_archetype"]["key"] == "flat_trace"
        assert "Chan" in synthesis["temporal_archetype"]["citation"]

    def test_strong_open_fade(self):
        synthesis = synthesize_research_findings(_axes(), _fmri([0.9, 0.6, 0.4, 0.3, 0.2]))
        assert synthesis["temporal_archetype"]["key"] == "strong_open_fade"

    def test_late_peak(self):
        synthesis = synthesize_research_findings(_axes(), _fmri([0.2, 0.3, 0.35, 0.4, 0.9]))
        assert synthesis["temporal_archetype"]["key"] == "late_peak"

    def test_buried_lede(self):
        synthesis = synthesize_research_findings(_axes(), _fmri([0.2, 0.3, 0.9, 0.5, 0.45, 0.5]))
        assert synthesis["temporal_archetype"]["key"] == "buried_lede"

    def test_short_or_missing_trace_returns_none(self):
        assert synthesize_research_findings(_axes(), _fmri([0.5, 0.6]))["temporal_archetype"] is None
        assert synthesize_research_findings(_axes(), None)["temporal_archetype"] is None
        assert synthesize_research_findings(_axes(), _fmri(["x", 1]))["temporal_archetype"] is None


class TestRawFeatureGrounding:
    def test_low_sustain_ratio_adds_grounded_gap(self):
        synthesis = synthesize_research_findings(
            _axes(), None, {"sustain_ratio": 0.25},
        )
        gap = next(item for item in synthesis["items"] if item["key"] == "low_sustain")
        assert "25%" in gap["observation"]
        assert "Chan" in gap["citation"]

    def test_healthy_sustain_ratio_adds_no_gap(self):
        synthesis = synthesize_research_findings(_axes(), None, {"sustain_ratio": 0.7})
        assert all(item["key"] != "low_sustain" for item in synthesis["items"])


class TestSegmentLocalization:
    MESSAGE = (
        "Hey Jordan saw your migration post congrats on the launch "
        "I built a tool that turns your traces into dashboards in ten minutes "
        "a few teams use it in production already "
        "would you be open to a quick look next week"
    )

    def test_localizes_peak_weak_and_cliff_to_text(self):
        # Trace peaks early, then falls off a cliff into the closing segment.
        trace = [0.9, 0.8, 0.4, 0.35, 0.2]
        loc = localize_pitch_segments(self.MESSAGE, _fmri(trace))

        assert loc is not None
        assert loc["peak"]["segment"] == 1
        assert loc["weakest"]["segment"] == 5
        assert loc["peak"]["text"]
        assert loc["weakest"]["text"]
        # Opener is the strongest quartile here, close is the weakest.
        assert loc["opener_strength_percentile"] >= loc["closer_strength_percentile"]
        assert loc["attention_cliff"] is not None
        assert loc["attention_cliff"]["to"]["text"]

    def test_no_cliff_when_trace_is_smooth(self):
        loc = localize_pitch_segments(self.MESSAGE, _fmri([0.5, 0.51, 0.49, 0.5, 0.5]))
        assert loc is not None
        assert loc["attention_cliff"] is None

    def test_short_or_missing_trace_returns_none(self):
        assert localize_pitch_segments(self.MESSAGE, _fmri([0.5, 0.6])) is None
        assert localize_pitch_segments(self.MESSAGE, None) is None
        assert localize_pitch_segments("", _fmri([0.5, 0.6, 0.7])) is None

    def test_build_tribe_synthesis_combines_both(self):
        combined = build_tribe_synthesis(
            self.MESSAGE, _axes(self_value=38), _fmri([0.9, 0.6, 0.4, 0.3, 0.2]), {"sustain_ratio": 0.3},
        )
        assert "items" in combined
        assert combined["localization"] is not None
        assert combined["temporal_archetype"] is not None
        assert any(item["key"] == "low_sustain" for item in combined["items"])
