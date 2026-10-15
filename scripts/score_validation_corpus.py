#!/usr/bin/env python3
"""Score a consented corpus in a separate GPU image; never call a product server."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tribe_service import llm_layer as llm, persuasion_features as pf, score_log

_MARKERS = ("tribe", "neural", "calibration", "neuro-persuasion", "analogue", "fsaverage",
            "temporal trace rule", "predicted response", "temporal segment map", "naming axes")
_AXES = tuple(pf.NEURO_AXIS_WEIGHTS)
# Keep the doctrine/annex verbatim even as later work packages update them.
_template = llm.SYSTEM_PROMPT.replace(llm.PERSUASION_DOCTRINE, "{DOCTRINE}").replace(llm.PERSUASION_RESEARCH_ANNEX, "{ANNEX}")
BLIND_SYSTEM_PROMPT = "\n".join(line for line in _template.splitlines()
                                if not any(marker in line.lower() for marker in _MARKERS)
                                and not any(line.startswith(f"- {axis} →") for axis in _AXES))
BLIND_SYSTEM_PROMPT = BLIND_SYSTEM_PROMPT.replace("{DOCTRINE}", llm.PERSUASION_DOCTRINE).replace("{ANNEX}", llm.PERSUASION_RESEARCH_ANNEX)
_production_shape = llm._build_user_prompt("", "", "ad-copy", {})
BLIND_CONTEXT_SCHEMA = _production_shape.split('  "context_fit": ', 1)[1].split(',\n  "breakdown":', 1)[0]


def blind_user_prompt(text, persona, channel):
    payload = json.dumps({"pitch_message": text, "target_persona": persona, "platform": channel}, ensure_ascii=False)
    return (f"## Untrusted Input Payload\n{payload}\n\n## Channel Norms\n{llm._platform_norms(channel)}\n"
            'Return JSON with "persuasion_score" (0–100) and the following exact "context_fit" shape:\n'
            + BLIND_CONTEXT_SCHEMA)


def blind_scores(text, persona, channel, *, call=None):
    call = call or llm._call_openrouter_once
    samples, missing = [], set()
    for _ in range(3):
        result = call(blind_user_prompt(text, persona, channel), temperature=.2, system_prompt=BLIND_SYSTEM_PROMPT)
        context = result.get("context_fit", {}) if isinstance(result, dict) else {}
        invalid = [key for key in llm.CONTEXT_FIT_KEYS if not isinstance(context.get(key), dict)
                   or isinstance(context[key].get("score"), bool)
                   or not isinstance(context[key].get("score"), (int, float))
                   or not np.isfinite(context[key]["score"]) or not 0 <= context[key]["score"] <= 100]
        raw = result.get("persuasion_score") if isinstance(result, dict) else None
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not np.isfinite(raw) or not 0 <= raw <= 100:
            invalid.append("persuasion_score")
        missing.update(invalid)
        if not invalid:
            samples.append({"raw": raw, "facets": {key: context[key]["score"] for key in llm.CONTEXT_FIT_KEYS},
                            "context": llm._semantic_score_from_context_fit(context)})
    return {"context_fit_missing": sorted(missing), "b1_samples": samples,
            "raw_llm_score_blind": float(np.mean([s["raw"] for s in samples])) if len(samples) == 3 else None,
            "context_fit_score_blind": float(np.mean([s["context"] for s in samples])) if len(samples) == 3 else None}


def scrambled_text(text, pitch_id):
    words = text.split()
    random.Random(pitch_id).shuffle(words)
    return " ".join(words)


def score_corpus(rows, *, scrambled=False):
    # Set before importing engine: it reads the encoder at import time.
    os.environ["TRIBE_TEXT_MODEL"] = "meta-llama/Llama-3.2-3B"
    from tribe_service import engine
    from tribe_service.schemas import PitchScoreRequest
    if engine.TRIBE_ALLOW_MOCK:
        raise ValueError("Corpus scoring requires real model output")
    if score_log.pitch_id("check") is None:
        raise ValueError("A >=32-byte PITCHCHECK_SCORE_LOG_KEY is required")
    prepared = []
    for row in rows:
        if row.get("consent_text") is not True:
            raise ValueError("Corpus text requires consent_text=true")
        request = PitchScoreRequest(message=row["text"], persona=row["persona"], platform=row["channel"])
        identifier = score_log.pitch_id(request.message)
        if row.get("pitch_id") and row["pitch_id"] != identifier:
            raise ValueError("Corpus pitch_id does not match the shared HMAC key")
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("group_id", "language")):
            raise ValueError("Corpus group_id and language are required")
        prepared.append((row, request, identifier))
    result = []
    for row, request, identifier in prepared:
        engine._clear_prediction_cache()
        prediction = engine.score_text(request.message)
        raw, fmri, signals = engine.analyze_predictions(prediction)
        if fmri.get("text_feature_compatible") is not True:
            raise ValueError("Canonical text encoder compatibility check failed")
        evidence = llm._augment_persuasion_evidence(request.message, request.persona, request.platform, raw, fmri)
        q = pf.calibration_quality_weight(evidence)
        e = pf.evidence_score_from_analysis(signals, evidence)
        confidence = pf.calibration_confidence(e, 50, evidence)
        engine._clear_prediction_cache()  # Retest must measure a new prediction, not a cache hit.
        second = engine.score_text(request.message)
        second_raw, second_fmri, second_signals = engine.analyze_predictions(second)
        if second_fmri.get("text_feature_compatible") is not True:
            raise ValueError("Retest encoder compatibility changed")
        second_evidence = llm._augment_persuasion_evidence(request.message, request.persona, request.platform, second_raw, second_fmri)
        output = {key: row[key] for key in ("group_id", "channel", "language", "data_source", "calibration_basis", "synthetic") if key in row}
        output.update(source="score", pitch_id=identifier, persona_id=score_log.pitch_id(request.persona),
                      scored_at=datetime.now(timezone.utc).isoformat(), mode="model", model_id=engine.TRIBE_MODEL_ID,
                      methodology_version=evidence["methodology_version"], llm_model=llm.OPENROUTER_MODEL,
                      word_count=len(engine._WORD_RE.findall(request.message)),
                      sentence_count=len(re.findall(r"[.!?]+(?:\s|$)", request.message)) or 1,
                      digit_count=sum(c.isdigit() for c in request.message),
                      raw_features=raw, neural_signals=signals,
                      axes={k: v["score"] for k, v in pf.neuro_axes_from_analysis(signals, evidence).items()},
                      evidence_score=e, quality_weight=q, confidence=confidence, final_score=e,
                      blend_inputs={"evidence": e, "raw_llm": None, "facet": None, "confidence": confidence, "quality_weight": q, "blend": llm.SEMANTIC_BLEND_WEIGHT},
                      b1_tribe_informed=False, tribe_retest_features=[raw, second_raw],
                      tribe_retest_evidence=[e, pf.evidence_score_from_analysis(second_signals, second_evidence)],
                      tribe_retest_relative_L2=float(np.linalg.norm(prediction - second) / max(np.linalg.norm(prediction), 1e-12)),
                      **{k: fmri.get(k) for k in ("voxel_count", "segments", "text_feature_model", "text_feature_compatible")},
                      **score_log.monitoring_fields({}, fmri),
                      **blind_scores(request.message, request.persona, request.platform))
        if scrambled:
            twin = scrambled_text(request.message, identifier)
            sr, sf, ss = engine.analyze_predictions(engine.score_text(twin))
            se = llm._augment_persuasion_evidence(twin, request.persona, request.platform, sr, sf)
            output.update(scrambled_evidence_score=pf.evidence_score_from_analysis(ss, se), scrambled_raw_features=sr,
                          scrambled_axes={k: v["score"] for k, v in pf.neuro_axes_from_analysis(ss, se).items()},
                          scrambled_b1=blind_scores(twin, request.persona, request.platform))
        result.append(output)
    return result


def reliability(rows):
    from validate_incremental import spearman
    complete = [r for r in rows if len(r["b1_samples"]) == 3]
    return {"B1_sample1_vs_sample2_spearman": spearman([r["b1_samples"][0]["context"] for r in complete], [r["b1_samples"][1]["context"] for r in complete]),
            "B1_raw_sample1_vs_sample2_spearman": spearman([r["b1_samples"][0]["raw"] for r in complete], [r["b1_samples"][1]["raw"] for r in complete]),
            "TRIBE_evidence_retest_spearman": spearman([r["tribe_retest_evidence"][0] for r in rows], [r["tribe_retest_evidence"][1] for r in rows]),
            "blind_system_prompt_sha256": hashlib.sha256(BLIND_SYSTEM_PROMPT.encode()).hexdigest(),
            "context_schema_sha256": hashlib.sha256(BLIND_CONTEXT_SCHEMA.encode()).hexdigest(),
            "temperature": .2, "samples_per_message": 3,
            "TRIBE_relative_L2": [r["tribe_retest_relative_L2"] for r in rows]}


def main():
    from validate_incremental import read_jsonl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scrambled", action="store_true", help="Run only after a candidate passes the primary check")
    args = parser.parse_args()
    rows = score_corpus(read_jsonl(args.input), scrambled=args.scrambled)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    with args.output.with_suffix(".reliability.json").open("x", encoding="utf-8") as handle:
        json.dump(reliability(rows), handle, indent=2, allow_nan=False)
    with args.output.with_suffix(".sha256").open("x", encoding="utf-8") as handle:
        handle.write(hashlib.sha256(args.output.read_bytes()).hexdigest() + "\n")


if __name__ == "__main__":
    main()
