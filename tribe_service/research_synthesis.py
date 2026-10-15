"""Deterministic synthesis of TRIBE outputs with published persuasion research.

This module links the TRIBE-predicted response geometry of a specific pitch to
the findings that make it actionable: which neural axes are weak or dominant,
what the temporal trace shape implies, and which published lever applies. The
output is citation-anchored, computed without any LLM, and feeds the evaluator
prompt and the report; only the desktop Rust fallback refine reads it as a brief.
"""
from __future__ import annotations

import math
import os
from typing import Any

from tribe_service.engine import _WORD_RE, direct_text_segment_words


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


def _context_segments_from_env() -> int:
    try:
        return max(0, min(3, int(os.getenv("PITCHCHECK_LOCALIZATION_CONTEXT_SEGMENTS", "1"))))
    except ValueError:
        return 1


# Literature prior (unvalidated, tunable; 0 disables): BOLD rises ~2 s and peaks 4-6 s after onset
# (Knutson et al. 2007), so even after TRIBE's 5-s offset row i blends ~3 s before / ~2 s after.
# ponytail: symmetric radius; retune once a span-edit probe measures TRIBE's real spread.
LOCALIZATION_CONTEXT_SEGMENTS = _context_segments_from_env()


def _segment_excerpts(
    message: str,
    n_segments: int,
    max_chars: int = 130,
    trace_basis: str | None = None,
    context: int = 0,
) -> list[str]:
    """Text presented during each 1-s trace segment, optionally widened by `context` segments per side.

    No lag shift: TRIBE rows are already lag-compensated (5-s target offset). Uses TRIBE's own word
    placement when the trace came from the direct schedule, else a proportional slice over the SAME
    _WORD_RE tokens. The response at segment i still blends neighbouring seconds, so this is the
    input at that moment, not the sole cause of the response.
    """
    spans = [match.span() for match in _WORD_RE.finditer(message)]
    if n_segments <= 0 or not spans:
        return []
    groups = direct_text_segment_words(message, n_segments, trace_basis)
    if groups is None:
        total = len(spans)
        groups = [
            spans[(i * total) // n_segments:max((i * total) // n_segments + 1, ((i + 1) * total) // n_segments)]
            for i in range(n_segments)
        ]
    excerpts: list[str] = []
    for index in range(n_segments):
        window = [group for group in groups[max(0, index - context):index + context + 1] if group]
        if not window:  # more segments than words in the fallback
            excerpts.append("")
            continue
        text = " ".join(message[window[0][0][0]:window[-1][-1][1]].split())
        excerpts.append(text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…")
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
    basis = fmri_summary.get("temporal_trace_basis") if isinstance(fmri_summary, dict) else None
    excerpts = _segment_excerpts(message, n, trace_basis=basis)
    if len(excerpts) != n:
        return None
    contexts = _segment_excerpts(
        message, n, max_chars=260, trace_basis=basis, context=LOCALIZATION_CONTEXT_SEGMENTS
    )

    def at(index: int) -> dict[str, Any]:
        return {
            "segment": index + 1,
            "of": n,
            "position_pct": round(100.0 * index / max(n - 1, 1)),
            "value": round(trace[index], 4),
            "percentile": _percentile_rank(trace, trace[index]),
            "text": excerpts[index],
            "context_text": contexts[index],
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
        "basis": "deterministic_tribe_trace_localization_v2",
        "excerpt_mapping": (
            "tribe_word_placement"
            if direct_text_segment_words(message, n, basis) is not None
            else "proportional_fallback"
        ),
        "context_segments": LOCALIZATION_CONTEXT_SEGMENTS,
        "context_basis": "literature_prior_hrf_dispersion",
    }


ARCHETYPE_CITATION = "PitchCheck heuristic inspired by Chan et al. 2024 (video ads)"
# Shared by low_sustain and encoding_gap: a memorability prior, not a persuasion verdict.
MEMORABILITY_FINDING = (
    "Moment-to-moment responses carried information about video-ad liking (Chan et al. 2024); whether this "
    "text proxy tracks recall is untested, and better memory did not predict quitting (Chua et al. 2011), so "
    "treat this as a memorability gap, not proof the pitch fails to persuade."
)
MEMORABILITY_SOURCES = ["chan_2024_ad_liking", "chua_2011_tailored_self_related"]


def _classify_temporal_archetype(trace: list[float]) -> dict[str, Any] | None:
    """Classify the predicted-response trace shape (a PitchCheck heuristic inspired by
    Chan et al. 2024, who used temporal neural dynamics to predict video-ad liking)."""
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
            "label": "Flat predicted response",
            "implication": (
                "No part of the pitch produces a salient predicted-response moment; "
                "temporal dynamics carry predictive signal for message response."
            ),
            "lever": "Create one concrete, vivid spike: a specific number, a named outcome, or a sharp contrast.",
            "citation": ARCHETYPE_CITATION,
            "source_keys": ["chan_2024_ad_liking"],
        }
    if peak_pos <= 0.25 and late < mean:
        return {
            "key": "strong_open_fade",
            "label": "Strong open, fading close",
            "implication": (
                "Predicted response peaks early and decays into the close, so the CTA "
                "sits in a low-response stretch."
            ),
            "lever": "Tighten the middle and rebuild the close: put a concrete reason to act next to the CTA.",
            "citation": ARCHETYPE_CITATION,
            "source_keys": ["chan_2024_ad_liking"],
        }
    if peak_pos >= 0.75:
        return {
            "key": "late_peak",
            "label": "Slow burn, late peak",
            "implication": "The highest predicted response comes near the end; check whether that point belongs earlier.",
            "lever": "Move the late strong moment into the opener; delete the warm-up it replaces.",
            "citation": ARCHETYPE_CITATION,
            "source_keys": ["chan_2024_ad_liking"],
        }
    if early < mean:
        return {
            "key": "buried_lede",
            "label": "Buried lede",
            "implication": (
                "The opener's predicted response is below the pitch average while a later span peaks; "
                "check whether that span belongs in the opener."
            ),
            "lever": "Promote the mid-pitch peak material into the first sentence.",
            "citation": ARCHETYPE_CITATION,
            "source_keys": ["chan_2024_ad_liking"],
        }
    return {
        "key": "sustained",
        "label": "Sustained predicted response",
        "implication": "Predicted response holds across the pitch with a usable opener.",
        "lever": "Preserve the structure; spend edits on proof credibility and CTA ease instead of re-ordering.",
        "citation": ARCHETYPE_CITATION,
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
                f"Predicted response stays at or above the pitch's own average for only "
                f"{sustain_ratio * 100:.0f}% of its length."
            ),
            "finding": MEMORABILITY_FINDING,
            "citation": "Chan et al. 2024; Chua et al. 2011",
            "source_keys": MEMORABILITY_SOURCES,
            "lever": "Check the below-average stretches first; cut or tighten the ones that are also semantically weak.",
        })

    if self_value <= 45:
        gaps.append({
            "key": "self_value_gap",
            "kind": "gap",
            "axis": "self_value",
            "observation": (
                f"TRIBE self/value proxy is low ({self_value:.0f}/100); computed from the text alone "
                "for an average reader, identical for every persona. This is not a self-relevance measurement; "
                "independently of it, check whether the opener uses the persona's own specifics."
            ),
            "finding": (
                "In small health-message fMRI studies, medial-prefrontal responses during exposure predicted which "
                "people later changed behavior beyond self-reported intentions (Falk 2010, N=20; Falk 2011, N=28; "
                "Cooper 2015, N=42; Vezich 2017, gain-framed messages; mainly in non-users), and aggregated "
                "responses forecast campaign response (Falk 2016). Links depended on content and audience; used "
                "here as a literature prior."
            ),
            "citation": "Falk et al. 2010, 2011, 2016; Cooper et al. 2015; Vezich et al. 2017; Chua et al. 2009",
            "source_keys": [
                "falk_2010_persuasion_change",
                "falk_2011_smoking_replication",
                "falk_2016_campaign_success",
                "cooper_2015_self_value_mpfc",
                "vezich_2017_framing_action",
                "chua_2009_tailoring",
            ],
            "lever": (
                "Tie the opener and core benefit to a goal, routine or fact the persona actually states, and say "
                "what the next step is worth toward it. You-language, the reader's name or flattery is not "
                "personalization; never invent details."
            ),
        })
    elif self_value >= 65:
        strengths.append({
            "key": "self_value_strength",
            "kind": "strength",
            "axis": "self_value",
            "observation": f"TRIBE self/value response is strong ({self_value:.0f}/100).",
            "finding": (
                "Medial-prefrontal self/value responses were repeatedly linked to later behavior change in several "
                "small health-message samples, mostly from one research group with overlapping samples (Falk 2010, "
                "2011, 2016; Cooper 2015); a literature prior, not a forecast for this reader."
            ),
            "citation": "Falk et al. 2010, 2011, 2016; Cooper et al. 2015",
            "source_keys": [
                "falk_2010_persuasion_change",
                "falk_2011_smoking_replication",
                "falk_2016_campaign_success",
                "cooper_2015_self_value_mpfc",
            ],
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
            "lever": "Shorten sentences and cut clauses and jargon first; keep every fact and mechanism a detail-seeking reader needs, each in its own short sentence.",
        })

    if encoding <= 45:
        gaps.append({
            "key": "encoding_gap",
            "kind": "gap",
            "axis": "encoding_attention",
            "observation": f"Predicted encoding/attention is weak ({encoding:.0f}/100).",
            "finding": MEMORABILITY_FINDING,
            "citation": "Chan et al. 2024; Chua et al. 2011",
            "source_keys": MEMORABILITY_SOURCES,
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
            "finding": (
                "Reward- and mentalizing-related neural synchrony carried separable persuasion evidence by narrative "
                "type (Cohen et al. 2024); here the split is a magnitude analogy over whole-brain response shape, not "
                "a measured reward-versus-mentalizing contrast."
            ),
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
            "finding": (
                "Mentalizing-network responses (dmPFC/pSTS/temporal pole) were higher for text arguments people rated "
                "persuasive in US and Korean samples (Falk, Rameson et al. 2010); related self/social/value signals "
                "tracked sharing in scanned groups (Baek, Scholz et al. 2017; Chan, Scholz et al. 2023)."
            ),
            "citation": "Falk, Rameson et al. 2010 (JoCN); Baek, Scholz et al. 2017; Chan, Scholz et al. 2023; Cohen et al. 2024",
            "source_keys": [
                "falk_2010_jocn_mentalizing_network",
                "baek_scholz_2017_sharing",
                "chan_scholz_2023_cross_cultural_sharing",
                "cohen_2024_reward_mentalizing",
            ],
            "lever": (
                "Lean into the narrative angle: one concrete example the reader could retell. Add peer evidence only "
                "if a true peer fact exists; this signal does not show whether peer proof is missing or would help."
            ),
        })

    # Severity order: biggest gaps first, then patterns/strengths.
    gaps.sort(key=lambda item: _safe_score(neuro_axes, str(item.get("axis", "")), 50.0))
    items = [*gaps, *strengths][:5]

    return {
        "items": items,
        "temporal_archetype": _classify_temporal_archetype(_trace_values(fmri_summary)),
        "route_hint": route_hint,
        "basis": "deterministic_tribe_x_research_synthesis_v2",
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
