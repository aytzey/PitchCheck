"""Optional WS11 scaling; stdlib only until WS07's shared atlas is available."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib
import json
import logging
import math
import os
from pathlib import Path
import re

LOGGER = logging.getLogger(__name__)
ROI_ASSET_DIR = Path(__file__).with_name("roi_assets")
FIDELITY_THRESHOLD, FIDELITY_FLOOR = .3, .25  # unvalidated priors
STATUSES = {"ok", "weak", "unmeasurable"}
TEXT_MODEL = "meta-llama/Llama-3.2-3B"
VAL_STORIES = {"birthofanation", "exorcism", "food", "golfclubbing", "jugglingandjesus",
               "lifeanddeathontheoregontrail", "thepostmanalwayscalls"}


def _roi_module():
    try:
        return importlib.import_module("tribe_service.roi_features")
    except ModuleNotFoundError as exc:
        if exc.name != "tribe_service.roi_features":
            raise
        return None


def load_roi_assets():
    module = _roi_module()
    return module.load_roi_assets() if module else None


def _finite(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _row_valid(row):
    return (isinstance(row, dict) and isinstance(row.get("status"), str) and row["status"] in STATUSES
            and all(_finite(row.get(k)) for k in ("r", "ceiling", "r_over_ceiling", "q"))
            and -1 <= row["r"] <= 1 and 0 <= row["ceiling"] <= 1 and 0 <= row["q"] <= 1
            and isinstance(row.get("ci"), list) and len(row["ci"]) == 2
            and all(_finite(v) for v in row["ci"]) and row["ci"][0] <= row["ci"][1]
            and ("r_over_ceiling_real_timing" not in row or _finite(row["r_over_ceiling_real_timing"])))


def validate_fidelity_asset(data, assets, *, strict=True):
    """Validate the shipped-file trust boundary; strict mode is also the asset CI check."""
    if not isinstance(data, dict) or assets is None:
        raise ValueError("Fidelity requires the WS07 atlas")
    threshold, source, provenance = (data.get(k) for k in ("threshold", "source", "provenance"))
    if not (type(data.get("schema")) is int and data["schema"] == 1 and data.get("validated") is False
            and isinstance(threshold, dict) and threshold.get("status") == "unvalidated_prior"
            and threshold.get("r_over_ceiling") == FIDELITY_THRESHOLD
            and threshold.get("floor_factor") == FIDELITY_FLOOR
            and isinstance(data.get("atlas_sha256"), str)
            and re.fullmatch(r"(?:[0-9a-f]{16}|[0-9a-f]{64})", data["atlas_sha256"])
            and data["atlas_sha256"][:16] == assets[4][:16]
            and re.fullmatch(r"[0-9a-f]{64}", str(data.get("prereg_sha256", "")))
            and isinstance(source, dict) and source.get("id") == "lebel2023_val"
            and source.get("modality") == "english_story_listening"
            and source.get("subjects_in_training") is True and source.get("stimuli_in_training") is False
            and isinstance(source.get("stories"), list) and source["stories"]
            and all(isinstance(s, str) and s in VAL_STORIES for s in source["stories"])
            and len(source["stories"]) == len(set(source["stories"]))
            and isinstance(provenance, dict) and provenance.get("text_model") == TEXT_MODEL
            and provenance.get("timing") == "pitchcheck_direct_0.40s"
            and all(isinstance(provenance.get(k), str) and provenance[k]
                    for k in ("tribe_hf_sha", "tribev2_commit", "pitchcheck_sha", "date"))):
        raise ValueError("Invalid or incompatible fidelity metadata")
    if re.search(r"reactance|anger|conflict|doubt", json.dumps(data), re.I):
        raise ValueError("Process labels are prohibited in fidelity assets")
    # Reject non-finite numbers anywhere, including optional network/HV summaries.
    def finite_tree(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite fidelity value")
        if isinstance(value, dict):
            for item in value.values():
                finite_tree(item)
        elif isinstance(value, list):
            for item in value:
                finite_tree(item)
    finite_tree({k: v for k, v in data.items() if k not in {"rois", "locations"}})
    rows = {}
    for section, allowed in (("rois", assets[2]), ("locations", assets[3])):
        values = data.get(section)
        if not isinstance(values, dict):
            raise ValueError(f"Invalid fidelity section: {section}")
        for key, row in values.items():
            try:
                finite_tree(row)
                valid = key in allowed and _row_valid(row)
            except ValueError:
                valid = False
            if not valid:
                if strict:
                    raise ValueError(f"Invalid fidelity row: {section}.{key}")
                LOGGER.warning("Ignoring invalid fidelity row: %s.%s", section, key)
                continue
            rows[key] = row
    return rows


@lru_cache(maxsize=1)
def roi_fidelity():
    try:
        raw = (ROI_ASSET_DIR / "roi_fidelity.json").read_bytes()
    except FileNotFoundError:
        return {}
    except OSError:
        LOGGER.warning("roi_fidelity.json unreadable; ignoring")
        return {}
    try:
        data = json.loads(raw)
        rows = validate_fidelity_asset(data, load_roi_assets(), strict=False)
    except (ValueError, TypeError, KeyError):
        LOGGER.warning("roi_fidelity.json invalid or atlas unavailable; ignoring")
        return {}
    return {"rois": rows, "asset_sha256": hashlib.sha256(raw).hexdigest()}


def fidelity_mode():
    mode = os.getenv("PITCHCHECK_ROI_FIDELITY_SCALING", "on").strip().lower()
    return mode if mode in {"on", "shadow", "off"} else "off"


def fidelity_factor(key):
    """Return (applied, would_be); missing/unmeasurable rows never downweight."""
    row = roi_fidelity().get("rois", {}).get(key)
    factor = (min(1.0, max(FIDELITY_FLOOR, row["r_over_ceiling"] / FIDELITY_THRESHOLD))
              if row and row["status"] != "unmeasurable" else 1.0)
    return (factor if fidelity_mode() == "on" else 1.0), factor


def external_fidelity(key):
    row = roi_fidelity().get("rois", {}).get(key, {})
    factor, would_be = fidelity_factor(key)
    return {"status": row.get("status", "not_benchmarked"), "r_over_ceiling": row.get("r_over_ceiling"),
            "factor": factor, "would_be": would_be, "source": "lebel2023_val",
            "modality": "english_story_listening", "validated": False}


def annotate_fidelity(fmri):
    """Bind WS07's reserved hook before its prior/WS09/WS12 consumers run."""
    module = _roi_module()
    if module is not None:
        module.fidelity_factor = fidelity_factor  # WS07 roi_multipliers reads this same-module hook.
    summary = (fmri or {}).get("roi_summary")
    if isinstance(summary, dict):
        for section in ("rois", "locations"):
            for key, row in (summary.get(section) or {}).items():
                if isinstance(row, dict):
                    row["external_fidelity"] = external_fidelity(key)


def fidelity_robustness(fmri):
    asset = roi_fidelity()
    return {"mode": fidelity_mode(), "asset_present": bool(asset.get("rois")),
            "asset_sha256": asset.get("asset_sha256"),
            "encoder_match": (fmri or {}).get("text_feature_compatible") is True}


def fidelity_prompt(fmri):
    if not roi_fidelity():  # no measured asset: keep the evaluator prompt unchanged
        return ""
    assets = load_roi_assets()
    summary = (fmri or {}).get("roi_summary")
    if assets is None or not isinstance(summary, dict):
        return ""
    lines = []
    for section, keys in (("rois", assets[2]), ("locations", assets[3])):
        for key in keys:
            if key in (summary.get(section) or {}):
                info = external_fidelity(key)
                lines.append(f"{key} | external fidelity: {info['status']} (weight x{info['factor']:.2f})")
    return ("\nCortical literature prior, predicted average-reader response (unvalidated):\n" + "\n".join(lines)) if lines else ""
