"""Neural-only persuasion calibration for PitchCheck.

This module intentionally does not score words, patterns, CTAs, proof points, or
other text heuristics.  The persuasion prior is derived from TRIBE-predicted
fMRI response geometry only.  Text and persona are passed to the LLM as
untrusted context for semantic explanation and rewrite generation, not as a
deterministic scoring audit.
"""
from __future__ import annotations

import math
from typing import Any


NEURO_AXIS_WEIGHTS = {
    "self_value": 0.28,
    "reward_affect": 0.24,
    "social_sharing": 0.18,
    "encoding_attention": 0.15,
    "processing_fluency": 0.15,
}

CALIBRATION_METHODOLOGY_VERSION = "neural_only_v2.2"

RESEARCH_SOURCES = [
    {
        "key": "falk_2010_persuasion_change",
        "citation": "Falk et al. 2010",
        "title": "Predicting persuasion-induced behavior change from the brain",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3027351/",
        "finding": "MPFC response during persuasive messages predicted later message-consistent behavior change.",
    },
    {
        "key": "falk_2012_neural_focus_group",
        "citation": "Falk et al. 2012",
        "title": "From neural responses to population behavior",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3725133/",
        "finding": "Small-sample neural responses predicted population-level media effects.",
    },
    {
        "key": "venkatraman_2015_ad_success",
        "citation": "Venkatraman et al. 2015",
        "title": "Predicting advertising success beyond traditional measures",
        "url": "https://doi.org/10.1509/jmr.13.0593",
        "finding": "Neurophysiological measures improved market-response prediction beyond traditional measures.",
    },
    {
        "key": "baek_scholz_2017_sharing",
        "citation": "Baek, Scholz et al. 2017",
        "title": "The value of sharing information",
        "url": "https://doi.org/10.1177/0956797617695073",
        "finding": "Value, self-related, and social-cognition ROIs tracked selection and sharing preferences.",
    },
    {
        "key": "chan_2024_ad_liking",
        "citation": "Chan et al. 2024",
        "title": "Neural signals of video advertisement liking",
        "url": "https://doi.org/10.1177/00222437231194319",
        "finding": "Affective and social-cognition responses predicted ad liking with informative temporal dynamics.",
    },
    {
        "key": "chan_scholz_2023_cross_cultural_sharing",
        "citation": "Chan, Scholz et al. 2023",
        "title": "Neural signals predict information sharing across cultures",
        "url": "https://doi.org/10.1073/pnas.2313175120",
        "finding": "Self-, social-, and value-related neural signals generalized across cultures better than self-report alone.",
    },
    {
        "key": "cohen_2024_reward_mentalizing",
        "citation": "Cohen et al. 2024",
        "title": "Reward and mentalizing circuits separately predict persuasiveness",
        "url": "https://doi.org/10.1038/s41598-024-62341-3",
        "finding": "Reward and mentalizing synchrony contributed dissociable persuasion evidence by narrative type.",
    },
    {
        "key": "scholz_chan_falk_2025_mega_analysis",
        "citation": "Scholz, Chan & Falk et al. 2025",
        "title": "Brain activity explains message effectiveness: A mega-analysis of 16 neuroimaging studies",
        "url": "https://doi.org/10.1093/pnasnexus/pgaf287",
        "finding": "Reward and social-processing responses tracked message effectiveness across 16 fMRI datasets.",
    },
    {
        "key": "cao_reimann_2020_triangulation",
        "citation": "Cao & Reimann 2020",
        "title": "Data triangulation in consumer neuroscience",
        "url": "https://doi.org/10.3389/fpsyg.2020.550204",
        "finding": "Consumer-neuroscience interpretation should explicitly mitigate reverse-inference risk.",
    },
    {
        "key": "tribe_v2_foundation_model",
        "citation": "d'Ascoli et al. 2026",
        "title": "A foundation model of vision, audition, and language for in-silico neuroscience",
        "url": "https://github.com/facebookresearch/tribev2",
        "finding": "TRIBE v2 predicts average-subject fMRI response analogues on the fsaverage5 cortical mesh.",
    },
]

RESEARCH_BASIS = [
    f"{source['citation']}: {source['finding']}"
    for source in RESEARCH_SOURCES
]

NEURO_AXIS_META = {
    "self_value": {
        "label": "Self-value fit",
        "analogue": "mPFC/vmPFC/PCC self- and value-processing analogue",
        "caveat": "Predicted response analogue; not a measured region-level fMRI claim.",
        "source_keys": [
            "falk_2010_persuasion_change",
            "falk_2012_neural_focus_group",
            "baek_scholz_2017_sharing",
            "chan_scholz_2023_cross_cultural_sharing",
        ],
    },
    "reward_affect": {
        "label": "Reward/affect motivation",
        "analogue": "Ventral-striatum/OFC/affective valuation analogue",
        "caveat": "Motivational salience is inferred from TRIBE response geometry.",
        "source_keys": [
            "venkatraman_2015_ad_success",
            "chan_2024_ad_liking",
            "cohen_2024_reward_mentalizing",
            "scholz_chan_falk_2025_mega_analysis",
        ],
    },
    "social_sharing": {
        "label": "Social cognition/sharing",
        "analogue": "TPJ/dmPFC/default-network social-cognition analogue",
        "caveat": "This is neural social-cognition potential, not proof that social proof exists in the text.",
        "source_keys": [
            "baek_scholz_2017_sharing",
            "chan_scholz_2023_cross_cultural_sharing",
            "chan_2024_ad_liking",
            "cohen_2024_reward_mentalizing",
            "scholz_chan_falk_2025_mega_analysis",
        ],
    },
    "encoding_attention": {
        "label": "Encoding and attention",
        "analogue": "Memory/attention/salience analogue",
        "caveat": "This estimates encoding potential, not actual recall.",
        "source_keys": [
            "chan_2024_ad_liking",
            "scholz_chan_falk_2025_mega_analysis",
            "tribe_v2_foundation_model",
        ],
    },
    "processing_fluency": {
        "label": "Processing fluency",
        "analogue": "Inverse cognitive-control/friction analogue",
        "caveat": "High fluency supports comprehension; it does not guarantee persuasion.",
        "source_keys": ["chan_2024_ad_liking", "cao_reimann_2020_triangulation"],
    },
}

SCIENTIFIC_CAVEATS = [
    "TRIBE returns predicted neural-response analogues, not measured fMRI from this recipient.",
    "TRIBE predictions are average-subject model outputs, not individualized recipient measurements.",
    "Brain-region labels are interpretive anchors; reverse inference remains limited.",
    "The score is a neural prior from in-silico fMRI geometry, not a validated recipient-level outcome probability.",
]


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return lo
    if math.isnan(numeric) or math.isinf(numeric):
        return lo
    return max(lo, min(hi, numeric))


def _round(value: float, digits: int = 1) -> float:
    return round(clamp(value), digits)


def _signal(neural_signals: dict[str, float], key: str, default: float = 50.0) -> float:
    try:
        return clamp(float(neural_signals.get(key, default)))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


QUALITY_WARNING_CAPS = {
    "near_zero_prediction_response": 0.35,
    "low_voxel_count_prediction": 0.55,
    "low_temporal_resolution": 0.65,
    "flat_temporal_trace": 0.72,
}


def calibration_quality_weight(evidence: dict[str, Any] | None = None) -> float:
    """Return how strongly neural evidence should pull away from neutral.

    Scientific caveats such as average-subject prediction and synthetic word
    order are exposed to users, but they are not score penalties by themselves.
    This weight only shrinks the score when the prediction matrix is too small,
    flat, near-zero, or otherwise weak as evidence.
    """
    if not isinstance(evidence, dict):
        return 1.0

    diagnostics = evidence.get("calibration_quality")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    warning_values = evidence.get("warnings") or []
    if not isinstance(warning_values, (list, tuple, set)):
        warning_values = [warning_values]
    diagnostic_warning_values = diagnostics.get("warnings") or []
    if not isinstance(diagnostic_warning_values, (list, tuple, set)):
        diagnostic_warning_values = [diagnostic_warning_values]
    warnings = {
        str(warning)
        for warning in warning_values
        if warning
    }
    warnings.update(
        str(warning)
        for warning in diagnostic_warning_values
        if warning
    )

    weight = 1.0
    for warning, cap in QUALITY_WARNING_CAPS.items():
        if warning in warnings:
            weight = min(weight, cap)

    segments = _safe_int(diagnostics.get("segments"), 0)
    if segments == 1:
        weight = min(weight, 0.55)
    elif segments == 2:
        weight = min(weight, 0.65)

    voxel_count = _safe_int(diagnostics.get("voxel_count"), 0)
    if 0 < voxel_count < 1000:
        weight = min(weight, 0.55)

    global_mean_abs = _safe_float(diagnostics.get("global_mean_abs"), 0.0)
    global_peak_abs = _safe_float(diagnostics.get("global_peak_abs"), 0.0)
    if (
        ("global_mean_abs" in diagnostics or "global_peak_abs" in diagnostics)
        and (global_mean_abs <= 1e-7 or global_peak_abs <= 1e-7)
    ):
        weight = min(weight, QUALITY_WARNING_CAPS["near_zero_prediction_response"])

    temporal_std_ratio = _safe_float(diagnostics.get("temporal_std_ratio"), 1.0)
    arc_ratio = _safe_float(diagnostics.get("arc_ratio"), 1.0)
    if segments > 2 and temporal_std_ratio < 0.02 and arc_ratio < 0.05:
        weight = min(weight, QUALITY_WARNING_CAPS["flat_temporal_trace"])

    return clamp(weight, 0.25, 1.0)


def quality_adjusted_score(score: float, evidence: dict[str, Any] | None = None) -> float:
    """Shrink low-quality neural evidence toward neutral instead of overclaiming."""
    weight = calibration_quality_weight(evidence)
    return clamp(50.0 + (clamp(score) - 50.0) * weight)


def analyze_persuasion_text(message: str, persona: str, platform: str = "general") -> dict[str, Any]:
    """Return compatibility metadata without text scoring.

    The previous implementation used keyword and regex heuristics.  That path is
    deliberately disabled so the prompt and final calibration are not biased by
    brittle text features or prompt-injection-like surface forms.
    """
    platform_key = (platform or "general").strip().lower()
    return {
        "overall_text_score": 50.0,
        "feature_scores": {},
        "detected_strategies": [],
        "missing_elements": [],
        "warnings": [],
        "prompt_injection_risk": 0.0,
        "readability": {},
        "matched_persona_terms": [],
        "strategy_counts": {},
        "platform": platform_key,
        "input_metadata": {
            "message_character_count": len(message or ""),
            "persona_character_count": len(persona or ""),
        },
        "methodology_version": CALIBRATION_METHODOLOGY_VERSION,
        "methodology": "text_heuristics_removed_neural_only_calibration",
        "research_basis": RESEARCH_BASIS,
        "research_sources": research_sources(),
    }


def neural_score_from_signals(neural_signals: dict[str, float]) -> float:
    """Convert six TRIBE-derived signal scores to a conservative neural prior."""
    ee = _signal(neural_signals, "emotional_engagement")
    pr = _signal(neural_signals, "personal_relevance")
    sp = _signal(neural_signals, "social_proof_potential")
    mem = _signal(neural_signals, "memorability")
    ac = _signal(neural_signals, "attention_capture")
    cf = _signal(neural_signals, "cognitive_friction")
    return clamp(
        pr * 0.25
        + ee * 0.20
        + mem * 0.18
        + ac * 0.15
        + sp * 0.12
        + (100.0 - cf) * 0.10
    )


def neuro_axes_from_analysis(neural_signals: dict[str, float], evidence: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Build neuro-persuasive axes using TRIBE-derived neural signals only."""
    del evidence
    ee = _signal(neural_signals, "emotional_engagement")
    pr = _signal(neural_signals, "personal_relevance")
    sp = _signal(neural_signals, "social_proof_potential")
    mem = _signal(neural_signals, "memorability")
    ac = _signal(neural_signals, "attention_capture")
    cf = _signal(neural_signals, "cognitive_friction")
    inv_cf = 100.0 - cf

    raw_scores = {
        "self_value": clamp(pr * 0.55 + ee * 0.18 + mem * 0.15 + inv_cf * 0.12),
        "reward_affect": clamp(ee * 0.48 + pr * 0.20 + ac * 0.18 + mem * 0.14),
        "social_sharing": clamp(sp * 0.50 + pr * 0.18 + ee * 0.14 + mem * 0.10 + ac * 0.08),
        "encoding_attention": clamp(mem * 0.42 + ac * 0.34 + ee * 0.14 + inv_cf * 0.10),
        "processing_fluency": clamp(inv_cf * 0.62 + mem * 0.14 + pr * 0.14 + ac * 0.10),
    }

    evidence_lines = {
        "self_value": [
            f"personal_relevance={pr:.0f}",
            f"emotional_engagement={ee:.0f}",
            f"memorability={mem:.0f}",
        ],
        "reward_affect": [
            f"emotional_engagement={ee:.0f}",
            f"personal_relevance={pr:.0f}",
            f"attention_capture={ac:.0f}",
        ],
        "social_sharing": [
            f"social_cognition_proxy={sp:.0f}",
            f"personal_relevance={pr:.0f}",
            f"emotional_engagement={ee:.0f}",
        ],
        "encoding_attention": [
            f"memorability={mem:.0f}",
            f"attention_capture={ac:.0f}",
            f"emotional_engagement={ee:.0f}",
        ],
        "processing_fluency": [
            f"inverse_cognitive_friction={inv_cf:.0f}",
            f"memorability={mem:.0f}",
            f"personal_relevance={pr:.0f}",
        ],
    }

    return {
        key: {
            "label": NEURO_AXIS_META[key]["label"],
            "score": _round(score),
            "weight": NEURO_AXIS_WEIGHTS[key],
            "contribution": round((score - 50.0) * NEURO_AXIS_WEIGHTS[key], 1),
            "analogue": NEURO_AXIS_META[key]["analogue"],
            "evidence": evidence_lines[key],
            "caveat": NEURO_AXIS_META[key]["caveat"],
            "source_keys": NEURO_AXIS_META[key]["source_keys"],
        }
        for key, score in raw_scores.items()
    }


def neuro_axis_score_from_axes(axes: dict[str, dict[str, Any]]) -> float:
    score = 0.0
    total_weight = 0.0
    for key, weight in NEURO_AXIS_WEIGHTS.items():
        item = axes.get(key, {})
        score += clamp(_safe_float(item.get("score", 50.0), 50.0)) * weight
        total_weight += weight
    return clamp(score / max(total_weight, 1e-9))


def evidence_score_from_analysis(neural_signals: dict[str, float], evidence: dict[str, Any] | None = None) -> float:
    axis_score = neuro_axis_score_from_axes(neuro_axes_from_analysis(neural_signals, evidence))
    return quality_adjusted_score(axis_score, evidence)


def calibration_confidence(neural_score: float, text_score: float, evidence: dict[str, Any]) -> float:
    """Estimate confidence from neural signal strength and prediction quality.

    ``text_score`` is accepted for backward compatibility and ignored.
    """
    del text_score
    distance_from_mid = abs(clamp(neural_score) - 50.0) / 50.0
    confidence = 0.62 + distance_from_mid * 0.22
    quality_weight = calibration_quality_weight(evidence)
    if quality_weight < 0.99:
        confidence = min(confidence, 0.45 + quality_weight * 0.38)
    return max(0.35, min(0.90, confidence))


def confidence_reasons(
    neural_score: float,
    text_score: float,
    evidence: dict[str, Any],
    axes: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    del text_score
    reasons = [
        "text_heuristic_audit_disabled",
        "tribe_predicted_fmri_primary",
        "reverse_inference_caveat_applied",
    ]
    quality_weight = calibration_quality_weight(evidence)
    if quality_weight < 0.75:
        reasons.append("low_prediction_quality_shrunk_to_neutral")
    elif quality_weight < 0.99:
        reasons.append("prediction_quality_caveat_applied")

    warnings = evidence.get("warnings", []) if isinstance(evidence, dict) else []
    if not isinstance(warnings, (list, tuple)):
        warnings = [warnings] if warnings else []
    for warning in warnings[:3]:
        reasons.append(f"warning_{warning}")

    if axes:
        strongest = max(axes.items(), key=lambda item: float(item[1].get("score", 50.0)))[0]
        weakest = min(axes.items(), key=lambda item: float(item[1].get("score", 50.0)))[0]
        reasons.append(f"strongest_neural_axis_{strongest}")
        reasons.append(f"weakest_neural_axis_{weakest}")
    if neural_score >= 70:
        reasons.append("high_neural_persuasion_prior")
    elif neural_score <= 40:
        reasons.append("low_neural_persuasion_prior")
    else:
        reasons.append("moderate_neural_persuasion_prior")
    return sorted(set(reasons))


def scientific_caveats() -> list[str]:
    return SCIENTIFIC_CAVEATS[:]


def research_sources() -> list[dict[str, str]]:
    return [dict(source) for source in RESEARCH_SOURCES]
