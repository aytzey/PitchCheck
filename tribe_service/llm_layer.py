"""LLM persuasion interpretation layer via OpenRouter."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini").strip()
OPENROUTER_API_BASE_URL = os.getenv(
    "OPENROUTER_API_BASE_URL", "https://openrouter.ai/api/v1"
).rstrip("/")
OPENROUTER_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "20"))
OPENROUTER_ENABLED = bool(OPENROUTER_API_KEY and OPENROUTER_MODEL)

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert sales psychologist and neuroscience-informed persuasion "
    "analyst. You analyze neural brain-response signals to assess how persuasive "
    "a sales message would be for a specific target persona. Always return strict JSON."
)


def _build_user_prompt(
    message: str,
    persona: str,
    platform: str,
    neural_signals: dict[str, float],
) -> str:
    signals_text = "\n".join(
        f"  - {k}: {v:.1f}/100" for k, v in neural_signals.items()
    )
    return f"""## Pitch Message
{message}

## Target Persona
{persona}

## Platform
{platform}

## Neural Brain-Response Signals (from TRIBE neuroscience model)
{signals_text}

## Instructions
Analyze this pitch for the target persona using the neural signals as evidence. Return JSON:
{{
  "persuasion_score": <0-100 int>,
  "verdict": "<one-line verdict>",
  "narrative": "<2-3 sentence expert analysis>",
  "persona_summary": "<your understanding of this persona's needs and psychology>",
  "breakdown": [
    {{"key": "emotional_resonance", "label": "Emotional Resonance", "score": <0-100>, "explanation": "<for this persona>"}},
    {{"key": "clarity", "label": "Clarity", "score": <0-100>, "explanation": "<for this persona>"}},
    {{"key": "urgency", "label": "Urgency", "score": <0-100>, "explanation": "<for this persona>"}},
    {{"key": "credibility", "label": "Credibility", "score": <0-100>, "explanation": "<for this persona>"}},
    {{"key": "personalization_fit", "label": "Personalization Fit", "score": <0-100>, "explanation": "<for this persona>"}}
  ],
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "risks": ["<risk 1>", "<risk 2>", "<risk 3>"],
  "rewrite_suggestions": [
    {{"title": "<what to improve>", "before": "<original snippet>", "after": "<improved version>", "why": "<reason>"}}
  ]
}}"""


def _call_openrouter(user_prompt: str) -> dict[str, Any] | None:
    """Call OpenRouter and return parsed JSON, or None on failure."""
    if not OPENROUTER_ENABLED:
        return None
    try:
        response = httpx.post(
            f"{OPENROUTER_API_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://pitch.machinity.ai",
                "X-Title": "PitchScore",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
                "response_format": {"type": "json_object"},
            },
            timeout=OPENROUTER_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        content = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        return json.loads(content)
    except Exception as exc:
        LOGGER.warning("OpenRouter call failed: %s", exc)
        return None


def _generate_fallback(
    message: str,
    persona: str,
    platform: str,
    neural_signals: dict[str, float],
) -> dict[str, Any]:
    """Generate a deterministic fallback report from neural signals alone."""
    ee = neural_signals.get("emotional_engagement", 50.0)
    pr = neural_signals.get("personal_relevance", 50.0)
    sp = neural_signals.get("social_proof_potential", 50.0)
    mem = neural_signals.get("memorability", 50.0)
    ac = neural_signals.get("attention_capture", 50.0)
    cf = neural_signals.get("cognitive_friction", 50.0)

    # Simple weighted average for overall score
    persuasion_score = int(
        round(
            ee * 0.20
            + pr * 0.20
            + ac * 0.20
            + mem * 0.15
            + sp * 0.10
            + (100 - cf) * 0.15
        )
    )
    persuasion_score = max(0, min(100, persuasion_score))

    if persuasion_score >= 70:
        verdict = "Strong persuasion potential"
    elif persuasion_score >= 40:
        verdict = "Moderate persuasion potential"
    else:
        verdict = "Weak persuasion potential — consider reworking"

    if persuasion_score >= 70:
        strength_word = "strong"
    elif persuasion_score >= 40:
        strength_word = "moderate"
    else:
        strength_word = "weak"

    attention_note = (
        "captures attention effectively"
        if ac >= 60
        else "may struggle to capture attention"
    )

    strengths = [
        s
        for s in [
            "Strong emotional engagement" if ee >= 60 else None,
            "Good attention capture" if ac >= 60 else None,
            "Memorable messaging" if mem >= 60 else None,
            "Clear communication" if cf <= 40 else None,
            "Strong personal relevance" if pr >= 60 else None,
        ]
        if s
    ][:3] or ["Neural signals are within normal range"]

    risks = [
        r
        for r in [
            "Low emotional engagement" if ee < 40 else None,
            "High cognitive friction" if cf >= 60 else None,
            "Weak attention capture" if ac < 40 else None,
            "Low memorability" if mem < 40 else None,
            "Weak personal relevance" if pr < 40 else None,
        ]
        if r
    ][:3] or ["No major risks detected from neural signals"]

    snippet = message[:50] + "..." if len(message) > 50 else message

    return {
        "persuasion_score": persuasion_score,
        "verdict": verdict,
        "narrative": (
            f"Neural analysis indicates {strength_word} engagement patterns "
            f"for the described persona. The pitch {attention_note}."
        ),
        "persona_summary": f"Target persona: {persona}",
        "breakdown": [
            {
                "key": "emotional_resonance",
                "label": "Emotional Resonance",
                "score": int(round(ee)),
                "explanation": "Based on neural emotional engagement signals.",
            },
            {
                "key": "clarity",
                "label": "Clarity",
                "score": int(round(100 - cf)),
                "explanation": (
                    "Inverse of cognitive friction — lower friction means "
                    "clearer messaging."
                ),
            },
            {
                "key": "urgency",
                "label": "Urgency",
                "score": int(round((ac + ee) / 2)),
                "explanation": (
                    "Derived from attention capture and emotional activation."
                ),
            },
            {
                "key": "credibility",
                "label": "Credibility",
                "score": int(round(pr)),
                "explanation": "Based on personal relevance neural signals.",
            },
            {
                "key": "personalization_fit",
                "label": "Personalization Fit",
                "score": int(round((pr + sp) / 2)),
                "explanation": (
                    "Combined personal relevance and social proof potential."
                ),
            },
        ],
        "strengths": strengths,
        "risks": risks,
        "rewrite_suggestions": [
            {
                "title": "Enhance opener",
                "before": snippet,
                "after": "[Consider a more attention-grabbing opener]",
                "why": (
                    "Neural attention capture signals suggest room for improvement."
                ),
            }
        ],
    }


def interpret_persuasion(
    message: str,
    persona: str,
    platform: str,
    neural_signals: dict[str, float],
    raw_features: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Interpret TRIBE neural signals into a persuasion report for the target persona."""
    user_prompt = _build_user_prompt(message, persona, platform, neural_signals)

    llm_result = _call_openrouter(user_prompt)

    if llm_result and isinstance(llm_result, dict) and "persuasion_score" in llm_result:
        # Validate and clamp
        try:
            score = int(llm_result["persuasion_score"])
            llm_result["persuasion_score"] = max(0, min(100, score))
            if "breakdown" not in llm_result or not isinstance(
                llm_result["breakdown"], list
            ):
                raise ValueError("Missing breakdown")
            if "strengths" not in llm_result or not isinstance(
                llm_result["strengths"], list
            ):
                raise ValueError("Missing strengths")
            return llm_result
        except (ValueError, TypeError, KeyError) as exc:
            LOGGER.warning(
                "LLM result validation failed: %s — using fallback", exc
            )

    return _generate_fallback(message, persona, platform, neural_signals)
