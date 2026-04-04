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

SYSTEM_PROMPT = """You are an expert sales psychologist and neuroscience-informed persuasion analyst inside PitchScore.

You analyze neural brain-response signals from TRIBE v2 — a neuroscience model (facebook/tribev2) that predicts fMRI-like brain responses to stimuli. The model maps text through audio → speech events → predicted cortical vertex activations on a 20,484-vertex brain mesh.

Key neuroscience mappings you use:
- emotional_engagement → medial prefrontal cortex (MPFC) activation analogue → value signal processing
- personal_relevance → self-referential processing → sustained focused activation
- social_proof_potential → temporoparietal junction (TPJ) / mentalizing network → social-cognitive engagement
- memorability → temporal pole / hippocampal analogue → encoding strength via engagement arc
- attention_capture → salience network → early onset + peak intensity
- cognitive_friction → dorsolateral prefrontal cortex (dlPFC) load → inverse of processing fluency

You also receive the temporal engagement trace — a per-second brain activation timeline showing how neural response builds, peaks, or drops across the pitch. Use this to identify which parts of the pitch are strongest and weakest.

Always return ONLY valid JSON — no markdown, no commentary."""


def _build_user_prompt(
    message: str,
    persona: str,
    platform: str,
    neural_signals: dict[str, float],
    fmri_summary: dict | None = None,
) -> str:
    signals_text = "\n".join(
        f"  - {k}: {v:.1f}/100" for k, v in neural_signals.items()
    )

    # Build temporal trace section if fMRI data available
    temporal_section = ""
    if fmri_summary and fmri_summary.get("temporal_trace"):
        trace = fmri_summary["temporal_trace"]
        n = len(trace)
        peak_idx = trace.index(max(trace)) if trace else 0
        peak_pct = round(peak_idx / max(n - 1, 1) * 100)
        temporal_section = f"""

## Temporal Engagement Trace (per-second brain activation)
{n} segments analyzed on {fmri_summary.get('voxel_count', 0):,} cortical vertices
Trace: {', '.join(f'{v:.3f}' for v in trace)}
Peak activation at segment {peak_idx + 1}/{n} ({peak_pct}% through the pitch)
Global mean: {fmri_summary.get('global_mean_abs', 0):.4f}, Global peak: {fmri_summary.get('global_peak_abs', 0):.4f}

Use this trace to identify which PARTS of the pitch generate the strongest/weakest brain response.
Early segments = opener, middle = body, late = close/CTA."""

    return f"""## Pitch Message
{message}

## Target Persona
{persona}

## Platform
{platform}

## Neural Brain-Response Signals (from TRIBE v2 neuroscience model)
{signals_text}{temporal_section}

## Instructions
Analyze this pitch for the target persona. Use the neural signals AND the temporal engagement trace as evidence. Return JSON:
{{
  "persuasion_score": <0-100 int>,
  "verdict": "<one-line verdict referencing the persona>",
  "narrative": "<2-3 sentence expert analysis referencing specific neural evidence and temporal patterns>",
  "persona_summary": "<your psychological profile of this persona — their decision drivers, biases, and communication preferences>",
  "breakdown": [
    {{"key": "emotional_resonance", "label": "Emotional Resonance", "score": <0-100>, "explanation": "<reference neural signals and temporal pattern for this persona>"}},
    {{"key": "clarity", "label": "Clarity", "score": <0-100>, "explanation": "<reference cognitive friction and temporal consistency>"}},
    {{"key": "urgency", "label": "Urgency", "score": <0-100>, "explanation": "<reference attention capture and temporal peaks near CTA>"}},
    {{"key": "credibility", "label": "Credibility", "score": <0-100>, "explanation": "<reference social proof potential and sustained engagement>"}},
    {{"key": "personalization_fit", "label": "Personalization Fit", "score": <0-100>, "explanation": "<reference personal relevance signal for this specific persona>"}}
  ],
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "risks": ["<risk 1>", "<risk 2>", "<risk 3>"],
  "rewrite_suggestions": [
    {{"title": "<what to improve>", "before": "<original snippet from the pitch>", "after": "<improved version tailored to the persona>", "why": "<reason citing neural evidence>"}}
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
            "Low emotional engagement — pitch may feel flat to this persona" if ee < 40 else None,
            "High cognitive friction — message may be hard to follow or overly complex" if cf >= 60 else None,
            "Weak attention capture — opener may not grab this persona's interest" if ac < 40 else None,
            "Low memorability — key points may not stick after reading" if mem < 40 else None,
            "Weak personal relevance — pitch may feel generic to this audience" if pr < 40 else None,
            "Social proof could be stronger — add concrete results or references" if sp < 50 else None,
            "Personalization gap — consider tailoring language to persona's role/industry" if pr < 55 and sp < 55 else None,
        ]
        if r
    ][:3] or ["Neural signals look balanced — consider A/B testing variations"]

    # Extract first sentence for rewrite suggestion
    first_sentence = message.split(".")[0].strip() if "." in message else message[:60]
    word_count = len(message.split())
    snippet = message[:50] + "..." if len(message) > 50 else message

    # Build smarter rewrite suggestions
    rewrite_suggestions = []
    if ac < 70:
        rewrite_suggestions.append({
            "title": "Strengthen the opening hook",
            "before": first_sentence[:80],
            "after": f"Lead with a specific pain point or surprising stat relevant to your persona",
            "why": f"Attention capture scored {ac:.0f}/100 — a stronger hook could improve first-impression engagement.",
        })
    if pr < 60:
        rewrite_suggestions.append({
            "title": "Increase persona specificity",
            "before": snippet,
            "after": "Reference their specific role, company stage, or recent achievement",
            "why": f"Personal relevance scored {pr:.0f}/100 — more targeted language would resonate better.",
        })
    if mem < 60:
        rewrite_suggestions.append({
            "title": "Add a memorable anchor",
            "before": "General value proposition",
            "after": "Include a concrete number, analogy, or visual comparison",
            "why": f"Memorability scored {mem:.0f}/100 — a vivid anchor helps the pitch stick.",
        })
    if not rewrite_suggestions:
        rewrite_suggestions.append({
            "title": "Test a variation",
            "before": first_sentence[:80],
            "after": "Try rephrasing with more urgency or a different angle",
            "why": "Scores are solid — A/B testing could reveal further optimization opportunities.",
        })

    # Build persona summary (don't just echo input)
    persona_lower = persona.lower()
    role_hint = "decision-maker" if any(w in persona_lower for w in ["cto", "ceo", "vp", "director", "head", "founder"]) else "stakeholder"
    tech_hint = "technical" if any(w in persona_lower for w in ["engineer", "developer", "technical", "tech"]) else "business-oriented"

    return {
        "persuasion_score": persuasion_score,
        "verdict": verdict,
        "narrative": (
            f"Neural analysis indicates {strength_word} engagement patterns "
            f"for the described persona. The pitch {attention_note}. "
            f"{'Emotional resonance is a key strength.' if ee >= 60 else 'Consider adding more emotional hooks.'} "
            f"Message clarity {'is high' if cf <= 40 else 'could be improved'} ({word_count} words)."
        ),
        "persona_summary": f"{role_hint.title()}, {tech_hint} profile — {persona[:120]}",
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
        "rewrite_suggestions": rewrite_suggestions,
    }


def interpret_persuasion(
    message: str,
    persona: str,
    platform: str,
    neural_signals: dict[str, float],
    raw_features: dict[str, float] | None = None,
    fmri_summary: dict | None = None,
) -> dict[str, Any]:
    """Interpret TRIBE neural signals into a persuasion report for the target persona."""
    user_prompt = _build_user_prompt(
        message, persona, platform, neural_signals, fmri_summary=fmri_summary,
    )

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
