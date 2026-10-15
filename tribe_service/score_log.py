"""Opt-in, numbers-only score log for later outcome joins (off unless path AND key are set).

Never writes message, persona or any LLM prose; texts are reduced to keyed HMAC ids so
outcomes (reply, meeting, rating) can be joined later by pitch_id.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import threading
import unicodedata

LOG_SCHEMA = 1
_LOCK = threading.Lock()  # ponytail: process lock; uvicorn runs one worker. Use fcntl.flock if workers > 1.
LOGGER = logging.getLogger(__name__)
ROI_KEYS = {
    "vmpfc_frontopolar", "mofc", "dmpfc", "insula", "r_ifg_ai", "pcc_precuneus", "r_tpj", "lateral_temporal", "l_ifg",
    "mpfc_falk2010", "falk2010jocn_core", "tp_l", "tp_r", "ipl_ifg_how_gt_why_vezich2017_l",
    "dmpfc_chua2011", "precuneus_chua2011", "rmpfc_chua2009", "vmpfc_chua2009", "l_angular_chua2011", "ifg47_l_chua2011",
    "pifg_vezich2017", "ripl_vezich2017",
}
PRIOR_FLAGS = ("PITCHCHECK_PERSONA_READ_PRIORS", "PITCHCHECK_ROI_PRIOR_WEIGHT", "PITCHCHECK_ROI_PRIOR_POINTS",
               "PITCHCHECK_SPECIFICITY_CONTRAST", "PITCHCHECK_SPECIFICITY_SCORE_WEIGHT", "PITCHCHECK_LITERATURE_PRIOR_BUDGET")


def monitoring_fields(robustness: dict, fmri: dict, context_fit=None) -> dict:
    """Whitelist optional WS06/07/12 scalars; never copy arbitrary nested reports."""
    def numbers(mapping, keys):
        return {key: value for key in keys
                if isinstance((value := mapping.get(key)), (int, float))
                and not isinstance(value, bool) and math.isfinite(value)}

    terms = {}
    for name in ("roi_prior", "persona_read_prior", "specificity_prior"):
        info = (robustness.get("score_terms") or {}).get(name) or {}
        terms[name] = numbers(info, ("value", "weight", "budget_scale"))
        if info.get("provenance") in {"literature_prior_unvalidated", "literature_prior_unvalidated;encoder_mismatch"}:
            terms[name]["provenance"] = info["provenance"]
    roi = {}
    for section in ("rois", "locations"):
        roi[section] = {
            key: numbers(values, ("mean_signed", "minus_cortex", "z", "n_vertices"))
            for key, values in ((fmri.get("roi_summary") or {}).get(section) or {}).items()
            if isinstance(values, dict) and key in ROI_KEYS
        }
    result = {"roi": roi if any(roi.values()) else None, "score_terms": terms if any(terms.values()) else None,
              "final_score_without_literature_priors": robustness.get("final_score_without_literature_priors")}
    if not isinstance(result["final_score_without_literature_priors"], (int, float)) or not math.isfinite(result["final_score_without_literature_priors"]):
        result["final_score_without_literature_priors"] = None
    inputs = robustness.get("prior_inputs") or {}
    result["prior_inputs"] = numbers(inputs, ("evidence_without_roi", "facet_without_boost", "tailoring_points", "specificity_points")) or None
    if result["prior_inputs"] is not None:
        if inputs.get("facet_without_boost") is None:
            result["prior_inputs"]["facet_without_boost"] = None
        result["prior_inputs"]["terms"] = terms
    result["literature_prior_budget"] = numbers(robustness.get("literature_prior_budget") or {}, ("budget", "delta", "budget_scale"))
    flags = robustness.get("active_flags") or {}
    result["active_flags"] = {key: value for key in PRIOR_FLAGS if isinstance((value := flags.get(key)), (bool, int, float)) and math.isfinite(value)}
    for key, allowed in (("recipient_stance", {"receptive", "neutral", "skeptical"}),
                         ("personalization_level", {"generic", "category", "recipient_specific"})):
        value = (context_fit or {}).get(key, robustness.get(key))
        result[key] = value if isinstance(value, str) and value in allowed else None
    return result


def _key() -> bytes | None:
    key = os.getenv("PITCHCHECK_SCORE_LOG_KEY", "")
    return key.encode() if len(key) >= 32 else None  # refuse unkeyed (plain sha256) ids


def enabled() -> bool:
    return bool(os.getenv("PITCHCHECK_SCORE_LOG_PATH")) and _key() is not None


def pitch_id(text: str) -> str | None:
    key = _key()
    norm = " ".join(unicodedata.normalize("NFC", text).split())
    return hmac.new(key, norm.encode(), hashlib.sha256).hexdigest() if key else None


def append(record: dict) -> None:
    path = os.getenv("PITCHCHECK_SCORE_LOG_PATH")
    if not path or _key() is None:
        return
    try:
        line = json.dumps({"log_schema": LOG_SCHEMA, **record}, separators=(",", ":"), allow_nan=False)
        with _LOCK, open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except (OSError, TypeError, ValueError) as exc:
        LOGGER.warning("Score log write failed (%s)", type(exc).__name__)  # never fails /score
