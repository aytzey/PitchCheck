"""Own synthetic CI arms; never a real outcome dataset or power estimate."""
import numpy as np


def make(seed=10, *, effect=1.5, groups=40, arms=4):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(groups), arms)
    n = len(g)
    b1 = rng.normal(size=(n, 2))
    logwc = np.log(rng.integers(8, 120, n))
    candidate = .8 * b1[:, :1] + .6 * rng.normal(size=(n, 1))
    linear = rng.normal(size=groups)[g] + .8 * b1[:, 0] + .4 * b1[:, 1] + effect * candidate[:, 0]
    sends = np.full(n, 300)
    k = rng.binomial(sends, 1 / (1 + np.exp(-(-3 + .5 * linear))))
    y = np.log((k + .5) / (sends - k + .5))
    w = 1 / (1 / (k + .5) + 1 / (sends - k + .5))
    return np.column_stack([logwc, b1]), candidate, y, w, g


def records(seed=10, *, groups=40, arms=4):
    from tribe_service.llm_layer import CONTEXT_FIT_WEIGHTS, _blend_final
    from tribe_service.persuasion_features import NEURO_AXIS_WEIGHTS
    base, candidate, y, _, g = make(seed, groups=groups, arms=arms)
    result = []
    for i in range(len(g)):
        e = float(np.clip(50 + 10 * candidate[i, 0], 0, 100))
        raw, facet = [float(np.clip(50 + 10 * v, 0, 100)) for v in base[i, 1:]]
        inputs = {"evidence": e, "raw_llm": raw, "facet": facet, "confidence": .7, "quality_weight": 1., "blend": .55}
        row = {"source": "score", "pitch_id": f"synthetic-{i}", "persona_id": "synthetic-persona",
               "group_id": f"g{g[i]:03}", "channel": "email", "platform": "email", "outcome_family": "reply",
               "outcome": float(1 / (1 + np.exp(-y[i]))), "n_sends": 300,
               "positives": int(round(300 / (1 + np.exp(-y[i])))),
               "mode": "model", "voxel_count": 20484, "text_feature_model": "synthetic-encoder",
               "methodology_version": "synthetic-v1", "llm_model": "synthetic-LLM", "language": "en",
               "word_count": int(round(np.exp(base[i, 0]))), "sentence_count": 3, "digit_count": 0, "segments": 10,
               "evidence_score": e, "raw_llm_score": raw, "context_fit_score": facet,
               "raw_llm_score_blind": raw, "context_fit_score_blind": facet,
               "context_fit_missing": [], "synthetic": True, "data_source": "synthetic",
               "calibration_basis": "synthetic_not_validation", "scored_at": f"2026-01-01T{int(g[i]) // 60:02}:{int(g[i]) % 60:02}:00+00:00",
               "confidence": .7, "quality_weight": 1., "blend_inputs": inputs,
               "final_score": round(_blend_final(**inputs)[0], 1),
               "facets": dict.fromkeys(CONTEXT_FIT_WEIGHTS, facet), "axes": dict.fromkeys(NEURO_AXIS_WEIGHTS, e)}
        result.append(row)
    return result


def comparison_records(seed=10, *, groups=50):
    """Full numeric log schema, with artificial future priors and real shared math."""
    from tribe_service import llm_layer as llm, persuasion_features as pf
    rows = records(seed, groups=groups)
    rng = np.random.default_rng(seed)
    signal_keys = ("emotional_engagement", "personal_relevance", "social_proof_potential",
                   "memorability", "attention_capture", "cognitive_friction")
    raw = dict(zip(("global_mean_abs", "global_peak_abs", "temporal_std", "early_mean", "late_mean",
                    "max_temporal_delta", "spatial_spread", "focus_ratio", "sustain_ratio", "arc_ratio"),
                   (.25, .6, .08, .2, .3, .12, .07, .35, .6, 1.2)))
    for i, row in enumerate(rows):
        signals = dict(zip(signal_keys, rng.uniform(30, 80, 6).tolist()))
        diagnostics = llm._response_quality_diagnostics(raw, row)
        evidence = {"calibration_quality": diagnostics}
        base = pf.evidence_score_from_analysis(signals, evidence)
        e = pf.clamp(base + 3 * np.sin(i))
        inputs = {**row["blend_inputs"], "evidence": e,
                  "confidence": pf.calibration_confidence(e, 50, evidence)}
        priors = {"evidence_without_roi": base, "facet_without_boost": inputs["facet"],
                  "tailoring_points": float(2 * np.cos(i)), "specificity_points": float(1.5 * np.sin(i / 2)),
                  "terms": {k: {"weight": w, "provenance": "literature_prior_unvalidated"}
                            for k, w in (("roi_prior", .15), ("persona_read_prior", 3.), ("specificity_prior", 1.5))}}
        final, _, _, _, report = llm._blend_final(**inputs, priors=priors)
        row.update(blend_inputs=inputs, prior_inputs=priors, evidence_score=e, final_score=round(final, 1),
                   confidence=inputs["confidence"], raw_features=dict(raw), neural_signals=signals,
                   axes={k: v["score"] for k, v in pf.neuro_axes_from_analysis(signals).items()},
                   active_flags={"PITCHCHECK_ROI_PRIOR_WEIGHT": .15}, **report)
    return rows
