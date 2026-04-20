"""Deterministic persuasion and robustness features for PitchCheck.

The TRIBE model gives us predicted neural-response structure, while this module
adds a content-level audit grounded in persuasion and neuroforecasting research:
value/self relevance, reward/affect, social cognition, encoding/attention,
processing fluency, credibility, actionability, and prompt integrity.  The
functions are intentionally dependency-free so they can run in mock mode, unit
tests, and constrained GPU containers.
"""
from __future__ import annotations

import math
import re
from typing import Any

_WORD_RE = re.compile(r"[\wÀ-ÖØ-öø-ÿ]+(?:[’'_\-][\wÀ-ÖØ-öø-ÿ]+)?%?", re.UNICODE)
_NUMBER_RE = re.compile(r"(?:\b\d+(?:[.,]\d+)?\s?(?:%|x|k|m|bn|gb|mb|ms|s|min|dk|gün|hafta|ay|yıl)?\b|\b\d+[+]\b)", re.I | re.U)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")

STOPWORDS = {
    # English
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "i", "in", "is", "it", "its", "of", "on", "or", "our",
    "that", "the", "their", "this", "to", "we", "with", "you", "your", "they",
    "them", "will", "would", "can", "could", "should", "about", "into", "than",
    # Turkish
    "ve", "veya", "ile", "için", "bir", "bu", "şu", "o", "de", "da", "mi",
    "mı", "mu", "mü", "ne", "nasıl", "neden", "çok", "daha", "en", "olan",
    "olarak", "sizin", "senin", "biz", "siz", "ben", "onlar", "gibi", "ama",
}

PLATFORM_WORD_BANDS: dict[str, tuple[int, int, int]] = {
    # min ideal, max ideal, hard-ish char limit used for fit warnings
    "email": (55, 210, 2000),
    "linkedin": (25, 110, 1000),
    "cold-call-script": (45, 150, 800),
    "landing-page": (18, 140, 1200),
    "ad-copy": (4, 45, 300),
    "general": (20, 320, 3000),
}

PATTERNS: dict[str, list[str]] = {
    "social_proof": [
        r"\btrusted by\b", r"\bused by\b", r"\b\d+[+]?\s*(customers|teams|companies|founders|users)\b",
        r"\b(teams|companies|customers|founders) like\b",
        r"\bcustomer(s)?\b", r"\bclient(s)?\b", r"\bcase stud(y|ies)\b", r"\btestimonial(s)?\b",
        r"\b(customer|user) review(s)?\b", r"\b\d+[+]?\s*review(s)?\b", r"\breviewed by\b",
        r"\brated\b", r"\breferenc(e|es)\b", r"\bpilot(s)?\b",
        r"\bmüşteri\b", r"\bekip\b", r"\breferans\b", r"\bvaka çalışması\b", r"\bkanıt\b",
    ],
    "authority": [
        r"\bSOC\s?2\b", r"\bISO\s?27001\b", r"\bGartner\b", r"\bForrester\b", r"\bpeer[- ]reviewed\b",
        r"\bresearch\b", r"\bbenchmark(ed|s)?\b", r"\bcertified\b", r"\baudit(ed)?\b",
        r"\bcompliance\b", r"\buzman\b", r"\baraştırma\b", r"\bsertifikalı\b", r"\bdenetim\b",
    ],
    "scarcity_urgency": [
        r"\btoday\b", r"\bthis week\b", r"\bnow\b", r"\bbefore\b", r"\bdeadline\b", r"\blimited\b",
        r"\bevery (day|week|month)\b", r"\bleaving .* on the table\b", r"\blosing\b", r"\bcost of inaction\b",
        r"\bhemen\b", r"\bbugün\b", r"\bbu hafta\b", r"\bson tarih\b", r"\bsınırlı\b", r"\bkaybed\w*\b",
    ],
    "loss_framing": [
        r"\bwaste(d|s|ful)?\b", r"\bleak(s|ing)?\b", r"\bmiss(ing)?\b", r"\bchurn\b", r"\brisk\b",
        r"\bcost(s|ing)?\b", r"\bmanual\b", r"\bdelay(s|ed)?\b", r"\bkayıp\b", r"\brisk\b", r"\bmaliyet\b",
    ],
    "reciprocity": [
        r"\bfree\b", r"\bno[- ]cost\b", r"\bno obligation\b", r"\bhappy to\b", r"\bshare\b", r"\bgive\b",
        r"\btemplate\b", r"\baudit\b", r"\btrial\b", r"\bpreview\b", r"\bücretsiz\b", r"\bpaylaş\w*\b",
        r"\bdeneme\b", r"\bhediye\b",
    ],
    "risk_reversal": [
        r"\brisk[- ]free\b", r"\bguarantee(d)?\b", r"\bcancel anytime\b", r"\bif (it|this) (isn'?t|is not)\b",
        r"\byou('?ve| have) lost nothing\b", r"\bno credit card\b", r"\bno commitment\b", r"\brisk yok\b",
        r"\brisksiz\b", r"\bgaranti\b", r"\btaahhüt yok\b",
    ],
    "cta": [
        r"\b(can|could|would) (we|you)\b", r"\bbook\b", r"\bschedule\b", r"\breply\b", r"\bcall\b", r"\bdemo\b",
        r"\btry\b", r"\bstart\b", r"\brun\b", r"\bcheck\b", r"\bopen to\b", r"\bgörüş\w*\b", r"\brandevu\b",
        r"\byanıt\w*\b", r"\bcevap\w*\b", r"\bdeney\w*\b", r"\bbaşla\w*\b",
    ],
    "specific_time": [
        r"\b\d{1,2}[:.]\d{2}\b", r"\b(mon|tue|wed|thu|fri|sat|sun)(day)?\b",
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b\d+\s?[- ]?(min|minute|minutes|dk|dakika)\b",
        r"\bnext (week|month|monday|tuesday|wednesday|thursday|friday)\b", r"\bpazartesi\b", r"\bsalı\b",
        r"\bçarşamba\b", r"\bperşembe\b", r"\bcuma\b",
    ],
    "outcome": [
        r"\breduce(s|d)?\b", r"\bincrease(s|d)?\b", r"\blift\b", r"\bsave(s|d)?\b", r"\bfaster\b",
        r"\bcut\b", r"\bimprove(s|d)?\b", r"\bconvert(s|ed)?\b", r"\brevenue\b", r"\bpipeline\b",
        r"\bazalt\w*\b", r"\bartır\w*\b", r"\btasarruf\b", r"\bhızlı\b", r"\bgelir\b", r"\bdönüşüm\b",
    ],
    "pain": [
        r"\bpain\b", r"\bstruggle(s|d)?\b", r"\bstretched thin\b", r"\bignored\b", r"\bbottleneck\b",
        r"\bmanual\b", r"\btedious\b", r"\bexpensive\b", r"\bproblem\b", r"\bsorun\b", r"\bzorl\w*\b",
        r"\byorucu\b", r"\bpahalı\b", r"\bdarboğaz\b",
    ],
    "contrast": [
        r"\bnot because\b", r"\binstead\b", r"\bwithout\b", r"\bbut\b", r"\brather than\b",
        r"\bama\b", r"\bfakat\b", r"\byerine\b", r"\bolmadan\b",
    ],
    "reasoning": [
        r"\bbecause\b", r"\bbased on\b", r"\bdata\b", r"\bevidence\b", r"\bmeasured\b",
        r"\bbenchmark(ed|s)?\b", r"\bcompared with\b", r"\btherefore\b", r"\bso that\b",
        r"\bçünkü\b", r"\bveri\b", r"\bkanıt\b", r"\bölç\w*\b", r"\bkarşılaştır\w*\b",
    ],
    "gain_framing": [
        r"\bgain\b", r"\bwin\b", r"\bunlock\b", r"\bgrow(th)?\b", r"\bmore\b",
        r"\bprotect\b", r"\brecover\b", r"\baccelerate\b", r"\bkazan\w*\b",
        r"\bbüyü\w*\b", r"\bkoru\w*\b", r"\bhızlan\w*\b",
    ],
}

VAGUE_TERMS = {
    "innovative", "powerful", "seamless", "world-class", "best-in-class", "cutting-edge",
    "revolutionary", "game-changing", "robust", "scalable", "solution", "platform",
    "leverage", "synergy", "transform", "optimize", "amazing", "great", "awesome",
    "yenilikçi", "harika", "mükemmel", "güçlü", "kusursuz", "çözüm", "platform",
}

SECOND_PERSON_TERMS = {"you", "your", "yours", "sen", "seni", "sana", "senin", "siz", "sizi", "size", "sizin"}
ROLE_TERMS = {
    "cto", "ceo", "cfo", "founder", "engineer", "developer", "vp", "director", "head",
    "manager", "marketer", "sales", "revenue", "product", "security", "finance",
    "kurucu", "mühendis", "direktör", "müdür", "pazarlama", "satış", "gelir", "ürün",
}

PROMPT_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|above|prior) instructions",
    r"disregard (all )?(previous|above|prior) instructions",
    r"system prompt", r"developer message", r"you are now", r"act as", r"jailbreak",
    r"return (only )?json", r"persuasion_score", r"score\s*(=|:)\s*100", r"give (me )?(a )?100",
    r"override", r"do not follow", r"reveal .*prompt", r"assistant:", r"system:", r"<\|system\|>",
    r"önceki talimat", r"talimatları yok say", r"sistem prompt", r"100 puan", r"json döndür",
]

NEURO_AXIS_WEIGHTS = {
    "self_value": 0.30,
    "reward_affect": 0.20,
    "social_sharing": 0.15,
    "encoding_attention": 0.17,
    "processing_fluency": 0.18,
}

NEURO_AXIS_META = {
    "self_value": {
        "label": "Self-value fit",
        "analogue": "mPFC/vmPFC/PCC self- and value-processing analogue",
        "caveat": "Predicted response analogue; not a measured region-level fMRI claim.",
    },
    "reward_affect": {
        "label": "Reward/affect motivation",
        "analogue": "Ventral-striatum/OFC/affective valuation analogue",
        "caveat": "Motivational salience is inferred from TRIBE geometry plus reward/outcome language.",
    },
    "social_sharing": {
        "label": "Social cognition/sharing",
        "analogue": "TPJ/dmPFC/default-network social-cognition analogue",
        "caveat": "Social proof is only credited when explicit text evidence is present.",
    },
    "encoding_attention": {
        "label": "Encoding and attention",
        "analogue": "Memory/attention/salience analogue",
        "caveat": "This estimates encoding potential, not actual recall.",
    },
    "processing_fluency": {
        "label": "Processing fluency",
        "analogue": "Inverse cognitive-control/friction analogue",
        "caveat": "High fluency supports comprehension; it does not guarantee persuasion.",
    },
}

SCIENTIFIC_CAVEATS = [
    "TRIBE returns predicted neural-response analogues, not measured fMRI from this recipient.",
    "Brain-region labels are interpretive anchors and must be triangulated with text, persona, and channel evidence.",
    "Reverse inference is limited by confidence, evidence agreement, and guardrail caps.",
]

RESEARCH_BASIS = [
    "Falk et al. 2011/2016: self- and value-related MPFC signals can predict behavior change and campaign response above self-report",
    "Venkatraman et al. 2015: fMRI, especially ventral striatum, predicts market-level advertising response beyond traditional measures",
    "Scholz et al. 2017 and Cohen et al. 2024: self/social cognition and reward systems predict sharing or persuasiveness depending on narrative type",
    "Chan et al. 2023: early emotion/memory and social-affective dynamics improve ad-liking prediction",
    "Cao & Reimann 2020: consumer neuroscience needs triangulation to reduce reverse-inference risk",
]


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    if math.isnan(value) or math.isinf(value):
        return lo
    return max(lo, min(hi, value))


def _round(value: float, digits: int = 1) -> float:
    return round(clamp(float(value)), digits)


def _normalise_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _tokens(text: str) -> list[str]:
    return [token.lower().strip("_-") for token in _WORD_RE.findall(text or "") if token.strip("_-")]


def _content_tokens(text: str) -> list[str]:
    return [token for token in _tokens(text) if len(token) >= 3 and token not in STOPWORDS]


def _sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split((text or "").strip()) if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _count_matches(text: str, patterns: list[str]) -> int:
    return sum(len(re.findall(pattern, text, flags=re.I | re.U)) for pattern in patterns)


def _strategy_counts(message: str) -> dict[str, int]:
    return {key: _count_matches(message, patterns) for key, patterns in PATTERNS.items()}


def _persona_overlap(message: str, persona: str) -> tuple[float, list[str]]:
    persona_terms = set(_content_tokens(persona))
    message_terms = set(_content_tokens(message))
    if not persona_terms:
        return 0.0, []
    matched = sorted(persona_terms & message_terms)
    # Saturate after the first handful of highly diagnostic overlaps.
    denominator = min(max(len(persona_terms), 1), 12)
    return clamp(len(matched) / denominator * 100.0), matched[:12]


def _platform_fit_score(platform: str, word_count: int, char_count: int) -> tuple[float, list[str]]:
    min_words, max_words, char_limit = PLATFORM_WORD_BANDS.get(platform, PLATFORM_WORD_BANDS["general"])
    warnings: list[str] = []
    if word_count < min_words:
        score = 72.0 - (min_words - word_count) * 1.25
        warnings.append("message_may_be_too_short_for_channel")
    elif word_count <= max_words:
        score = 90.0
    else:
        score = 90.0 - (word_count - max_words) * 0.65
        warnings.append("message_may_be_too_long_for_channel")
    if char_count > char_limit:
        score -= 22.0 + min(20.0, (char_count - char_limit) / max(char_limit, 1) * 35.0)
        warnings.append("message_exceeds_platform_character_guideline")
    return clamp(score, 12.0, 96.0), warnings


def _clarity_score(word_count: int, sentence_count: int, avg_sentence_words: float, vague_count: int, unique_ratio: float) -> float:
    score = 86.0
    if avg_sentence_words < 5:
        score -= 10.0
    elif avg_sentence_words > 24:
        score -= min(32.0, (avg_sentence_words - 24.0) * 1.8)
    if word_count > 260:
        score -= min(22.0, (word_count - 260) * 0.08)
    if vague_count:
        score -= min(24.0, vague_count * 4.5)
    if unique_ratio < 0.42 and word_count > 35:
        score -= 8.0
    if sentence_count <= 1 and word_count > 45:
        score -= 12.0
    return clamp(score, 10.0, 96.0)


def _prompt_injection_risk(message: str, persona: str) -> float:
    combined = f"{message}\n{persona}"
    hits = _count_matches(combined, PROMPT_INJECTION_PATTERNS)
    risk = hits * 18.0
    lower = combined.lower()
    if "persuasion_score" in lower and re.search(r"\b(100|99|perfect)\b", lower):
        risk += 32.0
    if "```" in combined and re.search(r"\b(system|assistant|developer|user)\b\s*:", lower):
        risk += 18.0
    if re.search(r"\b(ignore|disregard|override)\b", lower) and re.search(r"\b(score|json|instructions?)\b", lower):
        risk += 24.0
    return clamp(risk)


def neural_score_from_signals(neural_signals: dict[str, float]) -> float:
    """Convert six TRIBE-derived signal scores to a conservative neural prior."""
    ee = neural_signals.get("emotional_engagement", 50.0)
    pr = neural_signals.get("personal_relevance", 50.0)
    sp = neural_signals.get("social_proof_potential", 50.0)
    mem = neural_signals.get("memorability", 50.0)
    ac = neural_signals.get("attention_capture", 50.0)
    cf = neural_signals.get("cognitive_friction", 50.0)
    return clamp(
        pr * 0.24
        + ee * 0.18
        + mem * 0.16
        + ac * 0.14
        + sp * 0.12
        + (100.0 - cf) * 0.16
    )


def _feature(evidence: dict[str, Any], key: str, default: float = 50.0) -> float:
    scores = evidence.get("feature_scores", {})
    if not isinstance(scores, dict):
        return default
    try:
        return float(scores.get(key, default))
    except (TypeError, ValueError):
        return default


def neuro_axes_from_analysis(neural_signals: dict[str, float], evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build evidence-weighted neuro-persuasive axes from TRIBE and text audit data.

    The axes intentionally avoid one-to-one reverse inference.  A neural proxy can
    support an axis, but final axis strength also needs observable message
    evidence such as value, proof, audience fit, or clarity.
    """
    ee = neural_signals.get("emotional_engagement", 50.0)
    pr = neural_signals.get("personal_relevance", 50.0)
    sp = neural_signals.get("social_proof_potential", 50.0)
    mem = neural_signals.get("memorability", 50.0)
    ac = neural_signals.get("attention_capture", 50.0)
    cf = neural_signals.get("cognitive_friction", 50.0)
    counts = evidence.get("strategy_counts", {}) if isinstance(evidence.get("strategy_counts"), dict) else {}

    social_hits = float(counts.get("social_proof", 0.0) or 0.0)
    credibility = _feature(evidence, "credibility")
    audience_fit = _feature(evidence, "audience_fit")
    concreteness = _feature(evidence, "concreteness")
    cta_strength = _feature(evidence, "cta_strength")
    clarity = _feature(evidence, "clarity")
    platform_fit = _feature(evidence, "platform_fit")
    value_proposition = _feature(evidence, "value_proposition")
    argument_quality = _feature(evidence, "argument_quality")
    reward_gain = _feature(evidence, "reward_gain")
    memorability_text = _feature(evidence, "memorability")
    emotional_language = _feature(evidence, "emotional_language")
    urgency = _feature(evidence, "urgency")
    risk_reversal = _feature(evidence, "risk_reversal")
    social_proof_text = _feature(evidence, "social_proof_text")

    self_value = clamp(
        pr * 0.34
        + value_proposition * 0.23
        + audience_fit * 0.21
        + concreteness * 0.12
        + ee * 0.10
    )
    reward_affect = clamp(
        ee * 0.28
        + reward_gain * 0.24
        + emotional_language * 0.16
        + ac * 0.13
        + risk_reversal * 0.10
        + urgency * 0.09
    )
    social_sharing = clamp(
        sp * 0.28
        + social_proof_text * 0.24
        + credibility * 0.20
        + audience_fit * 0.14
        + argument_quality * 0.08
        + pr * 0.06
    )
    social_axis_unsupported_by_text = social_hits <= 0 and credibility < 65.0
    if social_axis_unsupported_by_text:
        social_sharing = min(social_sharing, 68.0)

    encoding_attention = clamp(
        mem * 0.28
        + ac * 0.20
        + memorability_text * 0.18
        + concreteness * 0.16
        + clarity * 0.12
        + platform_fit * 0.06
    )
    processing_fluency = clamp(
        (100.0 - cf) * 0.38
        + clarity * 0.28
        + argument_quality * 0.13
        + cta_strength * 0.13
        + platform_fit * 0.08
    )

    raw_scores = {
        "self_value": self_value,
        "reward_affect": reward_affect,
        "social_sharing": social_sharing,
        "encoding_attention": encoding_attention,
        "processing_fluency": processing_fluency,
    }

    evidence_lines = {
        "self_value": [
            f"personal_relevance={pr:.0f}",
            f"value_proposition={value_proposition:.0f}",
            f"audience_fit={audience_fit:.0f}",
        ],
        "reward_affect": [
            f"emotional_engagement={ee:.0f}",
            f"reward_gain={reward_gain:.0f}",
            f"emotional_language={emotional_language:.0f}",
        ],
        "social_sharing": [
            f"social_cognition_proxy={sp:.0f}",
            f"social_proof_text={social_proof_text:.0f}",
            f"credibility={credibility:.0f}",
        ],
        "encoding_attention": [
            f"memorability={mem:.0f}",
            f"attention_capture={ac:.0f}",
            f"concreteness={concreteness:.0f}",
        ],
        "processing_fluency": [
            f"inverse_cognitive_friction={100.0 - cf:.0f}",
            f"clarity={clarity:.0f}",
            f"argument_quality={argument_quality:.0f}",
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
            "unsupported_by_text": key == "social_sharing" and social_axis_unsupported_by_text,
        }
        for key, score in raw_scores.items()
    }


def neuro_axis_score_from_axes(axes: dict[str, dict[str, Any]]) -> float:
    score = 0.0
    total_weight = 0.0
    for key, weight in NEURO_AXIS_WEIGHTS.items():
        item = axes.get(key, {})
        score += float(item.get("score", 50.0)) * weight
        total_weight += weight
    return clamp(score / max(total_weight, 1e-9))


def evidence_score_from_analysis(neural_signals: dict[str, float], evidence: dict[str, Any]) -> float:
    axes = neuro_axes_from_analysis(neural_signals, evidence)
    neural_score = neuro_axis_score_from_axes(axes)
    text_score = float(evidence.get("overall_text_score", 50.0))
    injection_risk = float(evidence.get("prompt_injection_risk", 0.0))
    feature_scores = evidence.get("feature_scores", {}) if isinstance(evidence.get("feature_scores"), dict) else {}
    credibility = float(feature_scores.get("credibility", 50.0) or 50.0)
    concreteness = float(feature_scores.get("concreteness", 50.0) or 50.0)
    cta_strength = float(feature_scores.get("cta_strength", 50.0) or 50.0)
    platform = evidence.get("platform", "general")

    score = neural_score * 0.62 + text_score * 0.38
    if text_score < 45.0 and neural_score > text_score + 25.0:
        score = min(score, text_score + 22.0)
    if credibility < 45.0 and concreteness < 50.0:
        score = min(score, 68.0)
    if cta_strength < 45.0 and platform in {"email", "linkedin", "cold-call-script"}:
        score = min(score, 74.0)
    if injection_risk >= 35.0:
        score -= min(22.0, injection_risk * 0.22)
    return clamp(score)


def calibration_confidence(neural_score: float, text_score: float, evidence: dict[str, Any]) -> float:
    """Estimate confidence from agreement and input quality (0-1)."""
    agreement = 1.0 - min(abs(neural_score - text_score), 60.0) / 60.0
    readability = evidence.get("readability", {}) if isinstance(evidence.get("readability"), dict) else {}
    word_count = float(readability.get("word_count", 0.0) or 0.0)
    length_quality = 1.0 if 25 <= word_count <= 220 else 0.65 if 12 <= word_count <= 360 else 0.35
    injection_penalty = float(evidence.get("prompt_injection_risk", 0.0)) / 140.0
    warning_penalty = min(len(evidence.get("warnings", []) or []) * 0.035, 0.16)
    confidence = 0.42 + agreement * 0.30 + length_quality * 0.20 - injection_penalty - warning_penalty
    return max(0.30, min(0.94, confidence))


def confidence_reasons(
    neural_score: float,
    text_score: float,
    evidence: dict[str, Any],
    axes: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    reasons: list[str] = []
    delta = abs(neural_score - text_score)
    if delta <= 10:
        reasons.append("neural_text_agreement_high")
    elif delta <= 22:
        reasons.append("neural_text_agreement_moderate")
    else:
        reasons.append("neural_text_disagreement_requires_calibration")

    readability = evidence.get("readability", {}) if isinstance(evidence.get("readability"), dict) else {}
    word_count = float(readability.get("word_count", 0.0) or 0.0)
    if 25 <= word_count <= 220:
        reasons.append("message_length_supports_stable_read")
    elif word_count < 25:
        reasons.append("short_message_limits_confidence")
    else:
        reasons.append("long_message_adds_processing_variance")

    feature_scores = evidence.get("feature_scores", {}) if isinstance(evidence.get("feature_scores"), dict) else {}
    if float(feature_scores.get("credibility", 50.0) or 50.0) < 55:
        reasons.append("weak_credibility_evidence")
    if float(feature_scores.get("audience_fit", 50.0) or 50.0) < 55:
        reasons.append("weak_persona_specific_evidence")
    if float(evidence.get("prompt_injection_risk", 0.0) or 0.0) >= 35:
        reasons.append("prompt_integrity_guardrail_active")
    if axes and axes.get("social_sharing", {}).get("unsupported_by_text"):
        reasons.append("social_axis_has_no_explicit_social_proof")
    reasons.append("reverse_inference_caveat_applied")
    return sorted(set(reasons))


def scientific_caveats() -> list[str]:
    return SCIENTIFIC_CAVEATS[:]


def analyze_persuasion_text(message: str, persona: str, platform: str = "general") -> dict[str, Any]:
    """Return deterministic content-level persuasion evidence and guardrail signals."""
    message_clean = _normalise_ws(message)
    persona_clean = _normalise_ws(persona)
    platform_key = (platform or "general").strip().lower()
    tokens = _tokens(message_clean)
    content_tokens = [token for token in tokens if token not in STOPWORDS]
    word_count = len(tokens)
    char_count = len(message_clean)
    sentences = _sentences(message_clean)
    sentence_count = len(sentences)
    avg_sentence_words = word_count / max(sentence_count, 1)
    unique_ratio = len(set(content_tokens)) / max(len(content_tokens), 1)
    number_count = len(_NUMBER_RE.findall(message_clean))
    vague_count = sum(1 for token in tokens if token in VAGUE_TERMS)
    second_person_count = sum(1 for token in tokens if token in SECOND_PERSON_TERMS)
    persona_overlap, matched_persona_terms = _persona_overlap(message_clean, persona_clean)
    counts = _strategy_counts(message_clean)
    platform_fit, platform_warnings = _platform_fit_score(platform_key, word_count, char_count)
    injection_risk = _prompt_injection_risk(message_clean, persona_clean)

    role_match = 1 if set(_content_tokens(persona_clean)) & ROLE_TERMS & set(tokens) else 0
    specific_time = min(counts["specific_time"], 2)
    cta_hits = min(counts["cta"], 2)
    social_hits = min(counts["social_proof"], 3)
    authority_hits = min(counts["authority"], 2)
    urgency_hits = min(counts["scarcity_urgency"], 3)
    loss_hits = min(counts["loss_framing"], 3)
    reciprocity_hits = min(counts["reciprocity"], 3)
    risk_reversal_hits = min(counts["risk_reversal"], 2)
    outcome_hits = min(counts["outcome"], 4)
    pain_hits = min(counts["pain"], 3)
    contrast_hits = min(counts["contrast"], 3)
    reasoning_hits = min(counts["reasoning"], 4)
    gain_hits = min(counts["gain_framing"], 4)

    concreteness = clamp(30.0 + min(number_count, 4) * 12.0 + outcome_hits * 6.0 + specific_time * 6.0 - vague_count * 3.5)
    audience_fit = clamp(24.0 + persona_overlap * 0.46 + min(second_person_count, 5) * 4.5 + role_match * 10.0)
    credibility = clamp(22.0 + social_hits * 16.0 + authority_hits * 14.0 + min(number_count, 4) * 6.5 + risk_reversal_hits * 5.0 - vague_count * 2.0)
    cta_strength = clamp(22.0 + cta_hits * 25.0 + specific_time * 18.0 + risk_reversal_hits * 7.0 + (8.0 if "?" in message_clean else 0.0))
    urgency = clamp(28.0 + urgency_hits * 14.0 + loss_hits * 11.0 + cta_hits * 5.0)
    reciprocity = clamp(22.0 + reciprocity_hits * 20.0 + risk_reversal_hits * 8.0)
    risk_reversal = clamp(20.0 + risk_reversal_hits * 28.0 + reciprocity_hits * 6.0)
    emotional_language = clamp(30.0 + pain_hits * 11.0 + outcome_hits * 8.0 + loss_hits * 5.0 + min(second_person_count, 4) * 3.0)
    memorability_text = clamp(28.0 + min(number_count, 3) * 10.0 + contrast_hits * 8.0 + outcome_hits * 6.0 + social_hits * 5.0)
    clarity = _clarity_score(word_count, sentence_count, avg_sentence_words, vague_count, unique_ratio)
    value_proposition = clamp(
        24.0
        + outcome_hits * 12.0
        + pain_hits * 6.0
        + contrast_hits * 5.0
        + min(second_person_count, 4) * 4.0
        + min(number_count, 3) * 6.0
        - vague_count * 2.0
    )
    argument_quality = clamp(
        26.0
        + reasoning_hits * 11.0
        + authority_hits * 10.0
        + social_hits * 8.0
        + min(number_count, 4) * 7.0
        + contrast_hits * 5.0
        + max(0.0, clarity - 55.0) * 0.18
        - vague_count * 3.0
    )
    reward_gain = clamp(
        25.0
        + gain_hits * 14.0
        + outcome_hits * 9.0
        + reciprocity_hits * 7.0
        + risk_reversal_hits * 5.0
        + urgency_hits * 4.0
    )
    social_proof_text = clamp(
        18.0
        + social_hits * 24.0
        + authority_hits * 8.0
        + min(number_count, 2) * 6.0
        + risk_reversal_hits * 4.0
    )

    feature_scores = {
        "concreteness": concreteness,
        "value_proposition": value_proposition,
        "argument_quality": argument_quality,
        "reward_gain": reward_gain,
        "audience_fit": audience_fit,
        "credibility": credibility,
        "social_proof_text": social_proof_text,
        "cta_strength": cta_strength,
        "urgency": urgency,
        "reciprocity": reciprocity,
        "risk_reversal": risk_reversal,
        "emotional_language": emotional_language,
        "memorability": memorability_text,
        "clarity": clarity,
        "platform_fit": platform_fit,
        "prompt_integrity": 100.0 - injection_risk,
    }

    overall = (
        value_proposition * 0.15
        + audience_fit * 0.14
        + argument_quality * 0.13
        + credibility * 0.12
        + cta_strength * 0.11
        + concreteness * 0.10
        + clarity * 0.09
        + reward_gain * 0.06
        + memorability_text * 0.05
        + platform_fit * 0.03
        + risk_reversal * 0.02
    )
    if injection_risk >= 35:
        overall -= min(18.0, injection_risk * 0.18)

    detected_strategies = [
        name
        for name, count in counts.items()
        if count > 0 and name not in {"specific_time", "loss_framing", "outcome", "pain", "contrast", "reasoning", "gain_framing"}
    ]
    if number_count:
        detected_strategies.append("concreteness")
    if persona_overlap >= 35:
        detected_strategies.append("personalization")
    if loss_hits:
        detected_strategies.append("loss_framing")
    if outcome_hits:
        detected_strategies.append("outcome_framing")
    if reasoning_hits:
        detected_strategies.append("argument_quality")
    if gain_hits:
        detected_strategies.append("gain_framing")
    detected_strategies = sorted(set(detected_strategies))

    warnings = list(platform_warnings)
    if injection_risk >= 35:
        warnings.append("possible_prompt_injection_or_score_gaming")
    if word_count < 12:
        warnings.append("message_has_limited_context_for_stable_scoring")
    if cta_strength < 52 and platform_key not in {"ad-copy", "landing-page"}:
        warnings.append("missing_or_weak_call_to_action")
    if credibility < 52:
        warnings.append("missing_specific_proof_or_source_credibility")
    if audience_fit < 45:
        warnings.append("persona_connection_is_weak")
    if clarity < 55:
        warnings.append("clarity_or_processing_fluency_risk")
    if value_proposition < 52:
        warnings.append("value_proposition_is_underdeveloped")
    if argument_quality < 52 and word_count > 45:
        warnings.append("argument_quality_or_reasoning_is_weak")

    missing_elements = []
    if concreteness < 55:
        missing_elements.append("concrete_metric_or_specific_outcome")
    if credibility < 55:
        missing_elements.append("credible_proof_point")
    if cta_strength < 55:
        missing_elements.append("specific_low_friction_cta")
    if audience_fit < 55:
        missing_elements.append("persona_specific_context")
    if risk_reversal < 45 and platform_key in {"email", "linkedin", "cold-call-script"}:
        missing_elements.append("risk_reversal_or_reciprocity")
    if value_proposition < 55:
        missing_elements.append("clear_value_proposition")
    if argument_quality < 55 and word_count > 45:
        missing_elements.append("argument_quality_or_reasoning")

    readability = {
        "word_count": word_count,
        "character_count": char_count,
        "sentence_count": sentence_count,
        "avg_sentence_words": round(avg_sentence_words, 1),
        "unique_content_ratio": round(unique_ratio, 2),
        "number_count": number_count,
        "vague_term_count": vague_count,
        "second_person_count": second_person_count,
    }

    return {
        "overall_text_score": _round(overall),
        "feature_scores": {key: _round(value) for key, value in feature_scores.items()},
        "detected_strategies": detected_strategies,
        "missing_elements": missing_elements,
        "warnings": sorted(set(warnings)),
        "prompt_injection_risk": _round(injection_risk),
        "readability": readability,
        "matched_persona_terms": matched_persona_terms,
        "strategy_counts": counts,
        "platform": platform_key,
        "research_basis": RESEARCH_BASIS,
    }
