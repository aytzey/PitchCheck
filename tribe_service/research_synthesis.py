"""Deterministic synthesis of TRIBE outputs with published persuasion research.

This module links the TRIBE-predicted response geometry of a specific pitch to
the findings that make it actionable: which neural axes are weak or dominant,
what the temporal trace shape implies, and which published lever applies. The
output is citation-anchored, computed without any LLM, and feeds three places:
the evaluator prompt (pre-digested evidence), the report robustness payload
(user-visible deep dive), and the refine brief (rewrite levers).
"""
from __future__ import annotations

import math
from typing import Any


def _safe_score(axes: dict[str, dict[str, Any]], key: str, default: float = 50.0) -> float:
    try:
        value = float(axes.get(key, {}).get("score", default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _trace_values(fmri_summary: dict[str, Any] | None) -> list[float]:
    if not isinstance(fmri_summary, dict):
        return []
    raw = fmri_summary.get("temporal_trace") or []
    values: list[float] = []
    for item in raw:
        try:
            value = float(item)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(value):
            return []
        values.append(value)
    return values


def _segment_excerpts(message: str, n_segments: int, max_chars: int = 130) -> list[str]:
    """Map each trace segment to the proportional word span it covers.

    TRIBE direct-text mode spaces words uniformly, so segment k corresponds to
    the k-th proportional slice of the word sequence. This is what lets a
    purely numeric trace point at an actual sentence.
    """
    words = message.split()
    if n_segments <= 0 or not words:
        return []
    total = len(words)
    excerpts: list[str] = []
    for index in range(n_segments):
        start = (index * total) // n_segments
        stop = max(start + 1, ((index + 1) * total) // n_segments)
        excerpt = " ".join(words[start:stop]).strip()
        if len(excerpt) > max_chars:
            excerpt = excerpt[: max_chars - 1].rstrip() + "…"
        excerpts.append(excerpt)
    return excerpts


def _percentile_rank(values: list[float], target: float) -> float:
    if not values:
        return 0.0
    below = sum(1 for value in values if value < target)
    return round(100.0 * below / len(values), 1)


def localize_pitch_segments(
    message: str,
    fmri_summary: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Pinpoint, from the TRIBE trace, exactly which spans of the pitch are the
    opener, the peak, the weakest moment, the biggest attention drop, and the
    close — each tied to the real text. This removes the guesswork the LLM was
    previously asked to do from a raw segment list.
    """
    trace = _trace_values(fmri_summary)
    n = len(trace)
    if n < 3:
        return None
    excerpts = _segment_excerpts(message, n)
    if len(excerpts) != n:
        return None

    def at(index: int) -> dict[str, Any]:
        return {
            "segment": index + 1,
            "of": n,
            "position_pct": round(100.0 * index / max(n - 1, 1)),
            "value": round(trace[index], 4),
            "percentile": _percentile_rank(trace, trace[index]),
            "text": excerpts[index],
        }

    peak_idx = max(range(n), key=lambda i: trace[i])
    weak_idx = min(range(n), key=lambda i: trace[i])

    # Largest consecutive drop = the "attention cliff": where predicted
    # engagement falls off the hardest between adjacent segments.
    cliff = None
    if n >= 2:
        drops = [(trace[i] - trace[i + 1], i) for i in range(n - 1)]
        max_drop, drop_idx = max(drops, key=lambda item: item[0])
        mean = sum(trace) / n
        if max_drop > 0 and mean > 1e-12 and (max_drop / mean) >= 0.20:
            cliff = {
                "drop": round(max_drop, 4),
                "drop_ratio": round(max_drop / mean, 3),
                "from": at(drop_idx),
                "to": at(drop_idx + 1),
            }

    quartile = max(1, n // 4)
    opener_strength = _percentile_rank(trace, sum(trace[:quartile]) / quartile)
    closer_strength = _percentile_rank(trace, sum(trace[-quartile:]) / quartile)

    return {
        "opener": at(0),
        "opener_strength_percentile": opener_strength,
        "closer_strength_percentile": closer_strength,
        "peak": at(peak_idx),
        "weakest": at(weak_idx),
        "attention_cliff": cliff,
        "basis": "deterministic_tribe_trace_localization_v1",
    }


def _classify_temporal_archetype(trace: list[float]) -> dict[str, Any] | None:
    """Classify the engagement-trace shape and tie it to the temporal-dynamics
    and serial-position literature (Chan et al. 2024 used temporal neural
    dynamics to predict ad response)."""
    n = len(trace)
    if n < 3:
        return None
    mean = sum(trace) / n
    if mean <= 1e-12:
        return None
    spread = (max(trace) - min(trace)) / mean
    quartile = max(1, n // 4)
    early = sum(trace[:quartile]) / quartile
    late = sum(trace[-quartile:]) / quartile
    peak_pos = trace.index(max(trace)) / max(n - 1, 1)

    if spread < 0.15:
        return {
            "key": "flat_trace",
            "label": "Flat engagement",
            "implication": (
                "No part of the pitch produces a salient predicted-response moment; "
                "temporal dynamics carry predictive signal for message response."
            ),
            "lever": "Create one concrete, vivid spike: a specific number, a named outcome, or a sharp contrast.",
            "citation": "Chan et al. 2024",
            "source_keys": ["chan_2024_ad_liking"],
        }
    if peak_pos <= 0.25 and late < mean:
        return {
            "key": "strong_open_fade",
            "label": "Strong open, fading close",
            "implication": (
                "Predicted engagement peaks early and decays into the close, so the CTA "
                "lands on the weakest moment of the message."
            ),
            "lever": "Tighten the middle and rebuild the close: put a concrete reason to act next to the CTA.",
            "citation": "Chan et al. 2024",
            "source_keys": ["chan_2024_ad_liking"],
        }
    if peak_pos >= 0.75:
        return {
            "key": "late_peak",
            "label": "Slow burn, late peak",
            "implication": (
                "The strongest predicted response arrives near the end, after many readers "
                "on this channel have already stopped reading."
            ),
            "lever": "Move the late strong moment into the opener; delete the warm-up it replaces.",
            "citation": "Chan et al. 2024",
            "source_keys": ["chan_2024_ad_liking"],
        }
    if early < mean:
        return {
            "key": "buried_lede",
            "label": "Buried lede",
            "implication": (
                "The opener underperforms the middle of the pitch: the most engaging material "
                "is buried where skimming readers may never reach it."
            ),
            "lever": "Promote the mid-pitch peak material into the first sentence.",
            "citation": "Chan et al. 2024",
            "source_keys": ["chan_2024_ad_liking"],
        }
    return {
        "key": "sustained",
        "label": "Sustained engagement",
        "implication": "Predicted engagement holds across the pitch with a usable opener.",
        "lever": "Preserve the structure; spend edits on proof credibility and CTA ease instead of re-ordering.",
        "citation": "Chan et al. 2024",
        "source_keys": ["chan_2024_ad_liking"],
    }


def _raw_feature(raw_features: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(raw_features, dict) or key not in raw_features:
        return None
    try:
        value = float(raw_features[key])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def synthesize_research_findings(
    neuro_axes: dict[str, dict[str, Any]],
    fmri_summary: dict[str, Any] | None = None,
    raw_features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map this pitch's TRIBE axis geometry, raw features, and trace shape onto
    published findings.

    Returns citation-anchored items ordered by importance (gaps before
    strengths), a temporal archetype, and a neural route hint.
    """
    self_value = _safe_score(neuro_axes, "self_value")
    reward = _safe_score(neuro_axes, "reward_affect")
    social = _safe_score(neuro_axes, "social_sharing")
    encoding = _safe_score(neuro_axes, "encoding_attention")
    fluency = _safe_score(neuro_axes, "processing_fluency")

    gaps: list[dict[str, Any]] = []
    strengths: list[dict[str, Any]] = []

    # Raw-feature grounding: sustain_ratio is the fraction of segments above the
    # pitch's own mean response — a direct TRIBE measure of how much of the
    # message holds predicted engagement.
    sustain_ratio = _raw_feature(raw_features, "sustain_ratio")
    if sustain_ratio is not None and sustain_ratio < 0.4:
        gaps.append({
            "key": "low_sustain",
            "kind": "gap",
            "axis": "encoding_attention",
            "observation": (
                f"Predicted engagement stays above the pitch's own average for only "
                f"{sustain_ratio * 100:.0f}% of its length."
            ),
            "finding": "Sustained attention and encoding responses track later recall and message effectiveness.",
            "citation": "Chan et al. 2024; Scholz, Chan & Falk 2025",
            "source_keys": ["chan_2024_ad_liking", "scholz_chan_falk_2025_mega_analysis"],
            "lever": "Cut the below-average stretches; every sentence should pull its weight or be deleted.",
        })

    if self_value <= 45:
        gaps.append({
            "key": "self_value_gap",
            "kind": "gap",
            "axis": "self_value",
            "observation": f"TRIBE self/value response is weak ({self_value:.0f}/100).",
            "finding": (
                "Neural self/value responses to a message predict message-consistent behavior "
                "change better than self-report, across a 16-study mega-analysis."
            ),
            "citation": "Falk et al. 2010, 2016; Scholz, Chan & Falk 2025",
            "source_keys": ["falk_2010_persuasion_change", "scholz_chan_falk_2025_mega_analysis"],
            "lever": "Reframe the opener and the core benefit inside the reader's own goals and identity — this is the single best-evidenced lever for action.",
        })
    elif self_value >= 65:
        strengths.append({
            "key": "self_value_strength",
            "kind": "strength",
            "axis": "self_value",
            "observation": f"TRIBE self/value response is strong ({self_value:.0f}/100).",
            "finding": "Strong self/value engagement is the best neural predictor of message-driven behavior change.",
            "citation": "Falk et al. 2010, 2016; Scholz, Chan & Falk 2025",
            "source_keys": ["falk_2010_persuasion_change", "scholz_chan_falk_2025_mega_analysis"],
            "lever": "Protect the self-relevant framing in any rewrite; it is carrying the message.",
        })

    if fluency <= 50:
        gaps.append({
            "key": "fluency_gap",
            "kind": "gap",
            "axis": "processing_fluency",
            "observation": f"Predicted processing fluency is low ({fluency:.0f}/100).",
            "finding": "Harder-to-process messages are judged less true, less likable, and riskier.",
            "citation": "Alter & Oppenheimer 2009",
            "source_keys": ["alter_oppenheimer_2009_fluency"],
            "lever": "Shorten sentences, cut clauses and jargon, and reduce the message to one idea before touching anything else stylistic.",
        })

    if encoding <= 45:
        gaps.append({
            "key": "encoding_gap",
            "kind": "gap",
            "axis": "encoding_attention",
            "observation": f"Predicted encoding/attention is weak ({encoding:.0f}/100).",
            "finding": "Attention and encoding responses track later recall and message effectiveness.",
            "citation": "Chan et al. 2024; Scholz, Chan & Falk 2025",
            "source_keys": ["chan_2024_ad_liking", "scholz_chan_falk_2025_mega_analysis"],
            "lever": "Add one memorable concrete anchor (number, image, contrast) and lead with the strongest moment.",
        })

    # Route dominance: reward vs social-cognition systems contribute dissociable
    # persuasion evidence (Cohen et al. 2024) — match the rewrite strategy to
    # whichever system this pitch actually engages.
    route_hint = "balanced"
    if reward - social >= 12:
        route_hint = "reward_led"
        strengths.append({
            "key": "reward_route_dominant",
            "kind": "pattern",
            "axis": "reward_affect",
            "observation": f"Reward/value response ({reward:.0f}) clearly exceeds social-cognition response ({social:.0f}).",
            "finding": "Reward and mentalizing systems contribute dissociable persuasion evidence; messages work best when they lean into the system they actually engage.",
            "citation": "Cohen et al. 2024",
            "source_keys": ["cohen_2024_reward_mentalizing"],
            "lever": "Lean into concrete personal value and outcomes; do not force testimonials or social angles that the message does not support.",
        })
    elif social - reward >= 12:
        route_hint = "social_led"
        strengths.append({
            "key": "social_route_dominant",
            "kind": "pattern",
            "axis": "social_sharing",
            "observation": f"Social-cognition response ({social:.0f}) clearly exceeds reward/value response ({reward:.0f}).",
            "finding": "Self-, social-, and value-related neural signals track sharing and social transmission across cultures.",
            "citation": "Baek, Scholz et al. 2017; Chan, Scholz et al. 2023; Cohen et al. 2024",
            "source_keys": ["baek_scholz_2017_sharing", "chan_scholz_2023_cross_cultural_sharing", "cohen_2024_reward_mentalizing"],
            "lever": "Lean into the narrative/peer angle: similar-other proof and a story the reader could retell in one sentence.",
        })

    # Severity order: biggest gaps first, then patterns/strengths.
    gaps.sort(key=lambda item: _safe_score(neuro_axes, str(item.get("axis", "")), 50.0))
    items = [*gaps, *strengths][:5]

    return {
        "items": items,
        "temporal_archetype": _classify_temporal_archetype(_trace_values(fmri_summary)),
        "route_hint": route_hint,
        "basis": "deterministic_tribe_x_research_synthesis_v1",
    }


def build_tribe_synthesis(
    message: str,
    neuro_axes: dict[str, dict[str, Any]],
    fmri_summary: dict[str, Any] | None = None,
    raw_features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full deterministic TRIBE read: research findings + segment localization.

    This is the single entry point the LLM layer and report share, so the
    same TRIBE-grounded analysis appears in the prompt, the report, and the
    refine brief.
    """
    synthesis = synthesize_research_findings(neuro_axes, fmri_summary, raw_features)
    synthesis["localization"] = localize_pitch_segments(message, fmri_summary)
    return synthesis
