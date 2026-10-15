"""Signed cortical overlays; numpy and stdlib only, no runtime downloads."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)
ROI_ASSET_DIR = Path(__file__).with_name("roi_assets")
ASSET = ROI_ASSET_DIR / "roi_atlas_v1.npz"
N_VERTICES = 20484
ROI_KEYS = {"vmpfc_frontopolar", "mofc", "dmpfc", "insula", "r_ifg_ai", "pcc_precuneus", "r_tpj", "lateral_temporal", "l_ifg"}
CAVEAT = ("Signed average-subject cortical response analogue (TRIBE z-scored-BOLD units relative to the "
          "model's per-vertex training mean), expressed relative to labelled cortex (medial wall excluded). "
          "Not a contrast versus rest or another message; not comparable to published effect sizes; "
          "not this recipient; not a mental-state readout.")


@lru_cache(maxsize=1)
def load_roi_assets():
    """The shared (labels, parcel_names, rois, locations, sha) loader; None disables overlays."""
    try:
        raw = ASSET.read_bytes()
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            labels, names, network7 = z["labels"], z["parcel_names"], z["network7"]
            groups = [{k[4:]: z[k] for k in z.files if k.startswith(prefix)} for prefix in ("roi_", "loc_")]
        if not (labels.shape == (N_VERTICES,) and np.issubdtype(labels.dtype, np.integer)
                and np.array_equal(np.unique(labels), np.arange(401)) and names.shape == (401,) and names.dtype.kind == "U"
                and network7.shape == (401,) and np.issubdtype(network7.dtype, np.integer)
                and ((0 <= network7) & (network7 <= 7)).all()):
            raise ValueError("invalid atlas arrays")
        if not ((labels[:10242] <= 200).all() and ((labels[10242:] == 0) | (labels[10242:] > 200)).all()
                and (labels == 0).sum() == 1743 and set(groups[0]) == ROI_KEYS):
            raise ValueError("invalid hemispheres, medial wall or ROI keys")
        for group, minimum in zip(groups, (30, 20)):
            for idx in group.values():
                if not (idx.ndim == 1 and np.issubdtype(idx.dtype, np.integer) and idx.size >= minimum
                        and idx.min() >= 0 and idx.max() < N_VERTICES and np.unique(idx).size == idx.size and (labels[idx] > 0).all()):
                    raise ValueError("invalid ROI indices")
        union = np.concatenate(list(groups[0].values()))
        if np.unique(union).size != union.size:
            raise ValueError("overlapping parcel unions")
    except Exception as exc:
        LOGGER.warning("ROI atlas unavailable or failed validation: %s", exc)
        return None
    rois, locations = [{k: idx.astype(np.intp) for k, idx in g.items()} for g in groups]
    for arr in (labels, names, *rois.values(), *locations.values()):
        arr.flags.writeable = False
    return labels, names, rois, locations, hashlib.sha256(raw).hexdigest()[:16]


def _stats(mm, idx, mu, sd):
    value = float(mm[idx].mean())
    return {"mean_signed": round(value, 5), "minus_cortex": round(value - mu, 5),
            "z": round((value - mu) / sd, 4), "n_vertices": int(idx.size)}


def roi_summary(matrix):
    assets, m = load_roi_assets(), np.asarray(matrix)
    if assets is None or m.ndim != 2 or m.shape[0] == 0 or m.shape[1] != N_VERTICES:
        return None
    labels, _, rois, locations, sha = assets
    mm = m.mean(axis=0, dtype=np.float64)  # Signed: abs would erase deactivation.
    mu = float(mm[labels > 0].mean())
    if not math.isfinite(mu):
        return None
    sd = float(mm[labels > 0].std())
    if not math.isfinite(sd) or sd < 1e-12:
        return None
    return {"atlas": "schaefer2018_400_17n_fsaverage5", "basis": "signed_parcel_union_minus_cortex_v1",
            "roi_set_version": "roi_set_v1", "location_set_version": "roi_locations_v1", "asset_sha256": sha,
            "subject_basis": "average_subject", "caveat": CAVEAT,
            "cortex_mean_signed": round(mu, 5), "cortex_sd_signed": round(sd, 5),
            "rois": {k: _stats(mm, i, mu, sd) for k, i in rois.items()},
            "locations": {k: _stats(mm, i, mu, sd) for k, i in locations.items()}}


def _reference(encoder):
    from tribe_service.engine import roi_reference  # Delayed: engine imports this overlay.
    data = roi_reference(encoder)
    assets = load_roi_assets()
    return data if isinstance(data, dict) and assets and str(data.get("atlas_sha256", ""))[:16] == assets[4] else {}


def roi_noise_floors(encoder, *, default_only=False):
    """Signed minus-cortex units; Stage 0 wins over the provisional benchmark artifact."""
    keys = ("mpfc_falk2010", "ipl_ifg_how_gt_why_vezich2017_l")
    def positive(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
    try:
        quantities = _reference(encoder).get("quantities", {}) if not default_only else {}
        floors = {key: quantities.get(f"roi:{key}", {}).get("noise", {}).get("min_detectable_delta") for key in keys}
    except (AttributeError, TypeError):
        floors = {}
    if floors and all(positive(value) for value in floors.values()):
        return {**floors, "floor_source": "paraphrase_noise_floor"}
    try:
        from tribe_service import engine
        data = json.loads((ROI_ASSET_DIR / "roi_default_floors.json").read_text())
        assets = load_roi_assets()
        row = data["encoders"][encoder]
        floors = row["floors"]
        if (data.get("schema_version") == 1 and assets and str(data.get("atlas_sha256", ""))[:16] == assets[4]
                and data.get("event_version") == engine.DIRECT_TEXT_EVENT_VERSION
                and data.get("text_input_mode") == engine.TRIBE_TEXT_INPUT_MODE
                and all(positive(floors.get(key)) for key in keys)):
            return {**{key: floors[key] for key in keys}, "floor_source": "default_prior",
                    "provenance": row["provenance"]}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return {}


def roi_reference_p25(encoder, words, platform):
    """Pool channels below n=50; consume Stage-0 signed location quantities only."""
    try:
        quantity = _reference(encoder).get("quantities", {}).get("roi:falk2010jocn_core", {})
        channel = quantity.get("channels", {}).get(platform, {})
        if channel.get("n", 0) >= 50:
            quantity = channel
        bucket = "le60" if words <= 60 else "61_150" if words <= 150 else "151_240"
        value = quantity.get("length_buckets", {}).get(bucket, {}).get("p25") if words <= 240 else None
        if value is None and "length_model" in quantity:
            model, residual = quantity["length_model"], quantity.get("resid_q", {}).get("p25")
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                   for v in (model.get("a"), model.get("b_log_words"), residual)):
                value = model["a"] + model["b_log_words"] * math.log(max(1, words)) + residual
    except (AttributeError, TypeError, ValueError):
        return None
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def roi_multipliers(encoder):
    reference = _reference(encoder)
    checks = reference.get("checks", {}) if os.getenv("PITCHCHECK_ROI_INSILICO_TUNING", "1").lower() not in {"0", "false", "off", "no"} else {}
    assets = load_roi_assets()
    fidelity = globals().get("fidelity_factor")  # WS11 supplies the same-module hook.
    if assets is None or not (checks or fidelity):
        return {}
    def bounded(value):
        return min(1.0, max(0.0, float(value))) if isinstance(value, (int, float)) and math.isfinite(value) else 1.0
    checks = checks if isinstance(checks, dict) else {}
    return {key: bounded((checks[key] if isinstance(checks.get(key), dict) else {}).get("weight_multiplier", 1.0)) * bounded(fidelity(key)[0] if fidelity else 1.0)
            for key in (*assets[2], *assets[3])}


@lru_cache(maxsize=4)
def roi_calibration(encoder):
    """Stage-0 reference wins; otherwise use the encoder's benchmark unit artifact."""
    def valid(data):
        if not isinstance(data, dict) or not isinstance(data.get("centers"), dict):
            return False
        scale = data.get("ROI_Z_SCALE")
        return (isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 1e-12
                and ROI_KEYS - {"r_tpj", "lateral_temporal", "l_ifg"} <= data["centers"].keys()
                and all(k in ROI_KEYS and isinstance(v, (int, float)) and math.isfinite(v) for k, v in data["centers"].items())
                and str(data.get("provenance", "")).startswith("unit_calibration:"))
    reference = _reference(encoder)
    if valid(reference.get("unit_calibration")):
        return reference["unit_calibration"]
    try:
        slug = (encoder or "").replace("/", "__")
        data = json.loads((ROI_ASSET_DIR / f"roi_unit_calibration_{slug}.json").read_text())
        assets = load_roi_assets()
        if (valid(data) and assets and data.get("encoder") == encoder and str(data.get("atlas_sha256", ""))[:16] == assets[4]):
            return data
    except (OSError, ValueError, TypeError):
        pass
    return {}
