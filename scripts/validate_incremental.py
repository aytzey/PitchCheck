#!/usr/bin/env python3
"""Offline WS10 monitoring. Reports recommendations; never edits product settings."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tribe_service import persuasion_features as pf
from tribe_service.llm_layer import CONTEXT_FIT_WEIGHTS, SEMANTIC_BLEND_WEIGHT, _blend_final, _response_quality_diagnostics

PME_SCOPE = "PME is a self-report outcome; it tests perceived effectiveness only. A TRIBE null on PME is not evidence against behavioural validity, and a TRIBE positive is not evidence for reply rates."
CONDITIONED_B1 = "TRIBE-conditioned B1: blend-weight test, not incremental validity"
REPORT_TEMPLATE = "# WS10 predictive association\n\nDecision: {decision}\n\n{scope}\n\n```json\n{details}\n```\n"
BINOMIAL = {"reply", "meeting", "acceptance", "click", "binomial"}
SCORE_KEYS = {"evidence_score", "neural_score", "raw_llm_score", "llm_score", "final_score",
              "context_fit_score", "context_fit_score_blind", "raw_llm_score_blind"}
STRATA = ("text_feature_model", "methodology_version", "llm_model", "language", "channel", "outcome_family")


def finite(value, name, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if low is not None and value < low or high is not None and value > high:
        raise ValueError(f"{name} outside allowed range")
    return value


def validate_row(row, *, outcomes=True):
    required = ("pitch_id", "group_id", "channel", "outcome_family", "outcome") if outcomes else ("pitch_id",)
    if not isinstance(row, dict) or any(key not in row for key in required):
        raise ValueError(f"Missing required fields: {required}")
    if "text" in row and row.get("consent_text") is not True:
        raise ValueError("Text requires explicit consent_text=true")
    if "text" in row and not isinstance(row["text"], str):
        raise ValueError("Text must be a string")
    for key in required:
        if key != "outcome" and (not isinstance(row[key], str) or not row[key].strip()):
            raise ValueError(f"{key} must be a nonempty string")
    for key in SCORE_KEYS & row.keys():
        if row[key] is not None:
            finite(row[key], key, 0, 100)
    for key in ("facets", "axes", "neural_signals"):
        for value in (row.get(key) or {}).values():
            finite(value, key, 0, 100)
    for key, value in (row.get("raw_features") or {}).items():
        finite(value, key)
    for key in ("confidence", "quality_weight", "semantic_blend_weight"):
        if row.get(key) is not None:
            finite(row[key], key, 0, 1)
    for key in ("word_count", "sentence_count", "segments"):
        if key in row:
            finite(row[key], key, 1)
    if row.get("source") == "refine_candidate" and outcomes:
        raise ValueError("Refine candidate rows cannot enter the score B1 analysis")
    if outcomes:
        if row["outcome_family"] not in BINOMIAL | {"pme"}:
            raise ValueError("Unknown outcome_family")
        finite(row["outcome"], "outcome")
        if row["outcome_family"] in BINOMIAL:
            n, k = row.get("n_sends"), row.get("positives")
            if type(n) is not int or type(k) is not int or n < 1 or not 0 <= k <= n:
                raise ValueError("Require integer n_sends>=1 and 0<=positives<=n_sends")
        elif not 1 <= row["outcome"] <= 4:
            raise ValueError("PME mean must be in [1,4]")
    return row


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
                if not isinstance(row, dict):
                    raise ValueError("Each JSONL line must be an object")
                rows.append(row)
    return rows


def deduplicate_scores(rows):
    """Do not merge across provenance strata or persona; latest TRIBE, mean LLM."""
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["pitch_id"], row.get("persona_id"), *(row.get(k) for k in STRATA))].append(row)
    result = []
    for repeats in buckets.values():
        row = dict(max(repeats, key=lambda r: r.get("scored_at", "")))
        for key in ("raw_llm_score", "context_fit_score", "raw_llm_score_blind", "context_fit_score_blind"):
            values = [r[key] for r in repeats if r.get(key) is not None]
            if values:
                row[key] = float(np.mean(values))
        for field in ("facets",):
            keys = set().union(*(r.get(field, {}) for r in repeats))
            row[field] = {key: float(np.mean([r[field][key] for r in repeats if key in r.get(field, {})])) for key in keys}
        result.append(row)
    return result


def join_outcomes(scores, outcomes):
    index = defaultdict(list)
    for row in deduplicate_scores(scores):
        index[row["pitch_id"]].append(row)
    arms = defaultdict(list)
    for row in outcomes:
        validate_row(row)
        arms[(row["pitch_id"], row["group_id"], row.get("arm"), row["channel"], row["outcome_family"], row.get("edited_after_score", False))].append(row)
    joined = []
    for repeats in arms.values():
        arm = dict(repeats[-1])
        if arm["outcome_family"] in BINOMIAL:
            arm["n_sends"] = sum(r["n_sends"] for r in repeats)
            arm["positives"] = sum(r["positives"] for r in repeats)
            arm["outcome"] = arm["positives"] / arm["n_sends"]
        else:
            counts = [r.get("n_raters", 1) for r in repeats]
            arm["outcome"] = float(np.average([r["outcome"] for r in repeats], weights=counts))
            arm["n_raters"] = sum(counts)
        matches = index.get(arm["pitch_id"], [])
        if arm.get("persona_id"):
            matches = [r for r in matches if r.get("persona_id") == arm["persona_id"]]
        if not matches:
            raise ValueError("Outcome has no matching score")
        personas = {r.get("persona_id") for r in matches}
        if len(personas) > 1:
            raise ValueError("Ambiguous pitch_id join; include persona_id in outcomes")
        for score in matches:
            joined.append({**score, **arm})
    return joined


def join_blind(scores, blind_rows):
    keys = ("pitch_id", "persona_id", "text_feature_model", "methodology_version", "llm_model")
    index = {}
    for row in deduplicate_scores(blind_rows):
        key = tuple(row.get(k) for k in keys)
        if key in index:
            raise ValueError("Ambiguous blind corpus join")
        index[key] = row
    result = []
    fields = ("context_fit_score_blind", "raw_llm_score_blind", "context_fit_missing", "language",
              "b1_samples", "scrambled_b1", "scrambled_evidence_score", "tribe_retest_features", "tribe_retest_relative_L2")
    for row in scores:
        blind = index.get(tuple(row.get(k) for k in keys))
        if blind is None:
            raise ValueError("Missing matching blind B1; encoder/methodology/model/persona must match")
        result.append({**row, **{k: blind[k] for k in fields if k in blind}})
    return result


def eligible_strata(rows, *, allow_conditioned=False):
    exclusions = Counter()
    strata = defaultdict(list)
    for original in rows:
        row = dict(original)
        row.setdefault("channel", row.get("platform"))
        if row.get("source", "score") != "score":
            exclusions["refine_candidate"] += 1
            continue
        validate_row(row)
        reason = None
        if row.get("source", "score") != "score":
            reason = "refine_candidate"
        elif row.get("mode") != "model" or row.get("voxel_count") != 20484:
            reason = "mock_or_wrong_vertex_count"
        elif any(not row.get(k) for k in ("text_feature_model", "methodology_version", "llm_model", "language")):
            reason = "missing_provenance"
        elif row.get("edited_after_score") is True:
            reason = "edited_after_score_sensitivity"
        elif row["outcome_family"] == "reply" and (row["channel"] == "general" or row.get("personal_invitation")):
            reason = "general_or_personal_reply"
        elif row.get("context_fit_missing"):
            reason = "missing_blind_facets"
        elif not allow_conditioned and any(row.get(k) is None for k in ("context_fit_score_blind", "raw_llm_score_blind")):
            reason = "missing_blind_B1"
        if reason:
            exclusions[reason] += 1
        else:
            strata[tuple(row.get(k) for k in STRATA)].append(row)
    for key, records in strata.items():
        arms = defaultdict(list)
        for row in records:
            arms[(row["pitch_id"], row["group_id"], row.get("arm"), row.get("persona_id"))].append(row)
        strata[key] = [deduplicate_scores(repeats)[0] for repeats in arms.values()]
    return strata, dict(exclusions)


def demean(a, groups, weights=None):
    a = np.asarray(a, dtype=float)
    weights = np.ones(len(a)) if weights is None else weights
    _, inverse = np.unique(groups, return_inverse=True)
    matrix = a.reshape(len(a), -1)
    sums = np.zeros((int(inverse.max()) + 1, matrix.shape[1]))
    np.add.at(sums, inverse, matrix * weights[:, None])
    means = sums / np.bincount(inverse, weights=weights)[:, None]
    return (matrix - means[inverse]).reshape(a.shape)


def folds_for(groups, k=5, seed=10):
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("Need at least two groups for group CV")
    unique = np.random.default_rng(seed).permutation(unique)
    mapping = {g: i % min(k, len(unique)) for i, g in enumerate(unique)}
    return np.array([mapping[g] for g in groups])


def pairs(groups, folds=None):
    folds = np.zeros(len(groups)) if folds is None else folds
    result = []
    for group in np.unique(groups):
        for fold in np.unique(folds[groups == group]):
            result.extend(itertools.combinations(np.flatnonzero((groups == group) & (folds == fold)), 2))
    return np.array(result, dtype=int).reshape(-1, 2)


def pair_hits(scores, y, pair_index):
    i, j = pair_index.T
    truth = np.sign(y[i] - y[j])
    prediction = np.sign(scores[i] - scores[j])
    return (prediction == truth).astype(float) + (prediction == 0) * .5, truth != 0


def concordance(scores, y, pair_index):
    hits, valid = pair_hits(scores, y, pair_index)
    if not valid.any():
        raise ValueError("No comparable within-group outcome pairs")
    return float(hits[valid].mean())


def oof(X, y, weights, groups, folds, lam=1.0):
    X = np.asarray(X, float).reshape(len(y), -1)
    prediction = np.zeros(len(y))
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        x = demean(X[train], groups[train], weights[train])
        target = demean(y[train], groups[train], weights[train])
        scale = np.sqrt(np.average(x * x, axis=0, weights=weights[train]))
        scale[scale < 1e-8] = 1.0
        x = x / scale
        w = weights[train]
        coef = np.linalg.solve(x.T @ (w[:, None] * x) + lam * np.eye(X.shape[1]), x.T @ (w * target))
        prediction[test] = (demean(X[test], groups[test], weights[test]) / scale) @ coef
    return prediction


def bootstrap_delta(old, new, y, groups, pair_index, B=2000, seed=10):
    a, valid = pair_hits(old, y, pair_index)
    b, _ = pair_hits(new, y, pair_index)
    unique = np.unique(groups)
    sums, counts = [], []
    for group in unique:
        mask = (groups[pair_index[:, 0]] == group) & valid
        sums.append(np.sum((b - a)[mask]))
        counts.append(np.sum(mask))
    draws = np.random.default_rng(seed).integers(0, len(unique), size=(B, len(unique)))
    numer = np.array(sums)[draws].sum(axis=1)
    denom = np.array(counts)[draws].sum(axis=1)
    samples = numer[denom > 0] / denom[denom > 0]
    if not len(samples):
        raise ValueError("No bootstrap pairs")
    return [float(v) for v in np.quantile(samples, [.025, .975])]


def perm_test(base, candidate, y, weights, groups, folds, pair_index, *, B=1000, seed=10):
    candidate = np.asarray(candidate).reshape(len(y), -1)
    old = oof(base, y, weights, groups, folds)
    new = oof(np.column_stack([base, candidate]), y, weights, groups, folds)
    observed = concordance(new, y, pair_index) - concordance(old, y, pair_index)
    bd, cd = demean(base, groups, weights), demean(candidate, groups, weights)
    coef = np.linalg.lstsq(bd * np.sqrt(weights[:, None]), cd * np.sqrt(weights[:, None]), rcond=None)[0]
    fitted = base @ coef
    residual = candidate - fitted
    blocks = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    within = np.median([len(b) for b in blocks]) >= 3
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(B):
        idx = np.arange(len(y))
        if within:
            for block in blocks:
                idx[block] = rng.permutation(block)
        else:
            idx = rng.permutation(idx)
        # Refit the complete baseline and candidate pipeline for every permutation.
        pold = oof(base, y, weights, groups, folds)
        pnew = oof(np.column_stack([base, fitted + residual[idx]]), y, weights, groups, folds)
        null = concordance(pnew, y, pair_index) - concordance(pold, y, pair_index)
        exceed += null >= observed - 1e-12
    return {"delta": observed, "p": (1 + exceed) / (B + 1),
            "ci": bootstrap_delta(old, new, y, groups, pair_index, seed=seed),
            "base_C_w": concordance(old, y, pair_index), "candidate_C_w": concordance(new, y, pair_index),
            "permutation": "within_group_residual" if within else "global_residual"}, old, new


def holm(pvalues):
    order = np.argsort(pvalues)
    result = np.empty(len(order))
    maximum = 0.0
    for rank, i in enumerate(order):
        maximum = max(maximum, (len(order) - rank) * pvalues[i])
        result[i] = min(1.0, maximum)
    return result.tolist()


def targets(rows):
    if rows[0]["outcome_family"] == "pme":
        return np.array([r["outcome"] for r in rows]), np.ones(len(rows))
    n, k = (np.array([r[key] for r in rows]) for key in ("n_sends", "positives"))
    return np.log((k + .5) / (n - k + .5)), 1 / (1 / (k + .5) + 1 / (n - k + .5))


def powered(rows):
    return (len(rows) >= 80 and len({r["group_id"] for r in rows}) >= 10
            and all(r["outcome_family"] == "pme" or r["n_sends"] >= 100 for r in rows))


def decision(rows, delta, ci, p_holm=1.0, temporal=False, *, candidate=None, flag=None):
    if not powered(rows):
        return "underpowered"
    trusted = all(r.get("data_source") in {"randomized_campaign", "upworthy", "pme_panel"}
                  and not r.get("synthetic", False) for r in rows)
    if not trusted:
        return "keep"
    if flag and ci[1] < 0:
        return f"rollback_alarm:{flag}"
    if candidate and delta > 0 and ci[0] > 0 and p_holm < .05 and temporal:
        return f"tune:{candidate}"
    return "keep"


def blocks(rows, conditioned=False):
    channels = sorted({r["channel"] for r in rows})
    b0 = []
    for r in rows:
        wc = finite(r.get("word_count"), "word_count", 1)
        segments = finite(r.get("segments"), "segments", 1)
        b0.append([math.log(wc), wc / finite(r.get("sentence_count"), "sentence_count", 1),
                   finite(r.get("digit_count"), "digit_count", 0),
                   *[float(np.digitize(segments, [3, 9, 25, 65]) == i) for i in range(1, 5)],
                   *[float(r["channel"] == c) for c in channels[1:]]])
    keys = ("context_fit_score", "raw_llm_score") if conditioned else ("context_fit_score_blind", "raw_llm_score_blind")
    b1 = np.array([[finite(r.get(k), k, 0, 100) for k in keys] for r in rows])
    b2 = np.array([[finite(r.get("evidence_score"), "evidence_score", 0, 100)] for r in rows])
    result = {"B0": np.array(b0), "B1": b1, "B2": b2}
    for field, name in (("axes", "axes"), ("raw_features", "raw_features")):
        keys = sorted(set.intersection(*(set(r.get(field, {})) for r in rows)))
        if keys:
            result[name] = np.array([[r[field][k] for k in keys] for r in rows])
    roi_keys = sorted(set.intersection(*({f"{s}.{k}" for s, values in (r.get("roi") or {}).items()
                                        for k, v in values.items() if v.get("z") is not None} for r in rows)))
    if roi_keys:
        result["B3"] = np.array([[finite(r["roi"][s][k]["z"], "roi.z") for s, k in (v.split(".", 1) for v in roi_keys)] for r in rows])
    return result


def ols_nested(base, candidate, y, groups):
    y = demean(y, groups)
    a = demean(base, groups)
    b = np.column_stack([a, demean(candidate, groups)])
    ss0 = float(np.sum((y - a @ np.linalg.lstsq(a, y, rcond=None)[0]) ** 2))
    ss1 = float(np.sum((y - b @ np.linalg.lstsq(b, y, rcond=None)[0]) ** 2))
    rank0, rank1 = np.linalg.matrix_rank(a), np.linalg.matrix_rank(b)
    added = int(rank1 - rank0)
    df = len(y) - len(np.unique(groups)) - int(rank1)
    total = float(y @ y)
    return {"delta_R2": (ss0 - ss1) / total if total else None,
            "F": ((ss0 - ss1) / added) / (ss1 / df) if added > 0 and df > 0 and ss1 > 0 else None,
            "df_added": added, "df_residual": df, "rank_deficient": int(rank1) < b.shape[1],
            "perfect_fit": ss1 < 1e-12}


def increment(rows, *, B=1000, seed=10, conditioned=False):
    if len({r["group_id"] for r in rows}) < 2:
        return {"decision": "underpowered", "tests": [], "reason": "Need at least two groups for CV"}
    b = blocks(rows, conditioned)
    y, w = targets(rows)
    g = np.array([r["group_id"] for r in rows])
    folds = folds_for(g, seed=seed)
    P = pairs(g, folds)
    if not len(P) or not np.any(y[P[:, 0]] != y[P[:, 1]]):
        return {"decision": "underpowered", "tests": [], "reason": "No comparable within-group pairs"}
    base = np.column_stack([b["B0"], b["B1"]])
    tests = [("B2|B0+B1", base, b["B2"], False),
             ("B1|B0+B2", np.column_stack([b["B0"], b["B2"]]), b["B1"], True)]
    for name in ("axes", "raw_features", "B3"):
        if name in b:
            tests.append((f"{name}|B0+B1+B2", np.column_stack([base, b["B2"]]), b[name], True))
    if "B3" in b:
        tests.append(("B2|B0+B1+B3", np.column_stack([base, b["B3"]]), b["B2"], True))
    if all(r.get("scrambled_evidence_score") is not None for r in rows):
        tests.append(("B2|B0+B1+scrambled", np.column_stack([base, [r["scrambled_evidence_score"] for r in rows]]), b["B2"], True))
    if all((r.get("scrambled_b1") or {}).get("context_fit_score_blind") is not None
           and r["scrambled_b1"].get("raw_llm_score_blind") is not None for r in rows):
        twin_b1 = np.array([[r["scrambled_b1"][key] for key in ("context_fit_score_blind", "raw_llm_score_blind")] for r in rows])
        tests.append(("B1|B0+scrambled_B1", np.column_stack([b["B0"], twin_b1]), b["B1"], True))
    results = []
    for name, baseline, candidate, exploratory in tests:
        result, old, new = perm_test(baseline, candidate, y, w, g, folds, P, B=B, seed=seed)
        result.update(name=name, exploratory=exploratory, ols=ols_nested(baseline, candidate, y, g))
        if rows[0]["outcome_family"] == "pme":
            yd = demean(y, g)
            result["OOF_R2"] = 1 - float(np.sum((yd - new) ** 2) / np.sum(yd ** 2)) if np.any(yd) else None
        results.append(result)
    for result, p in zip(results, holm([r["p"] for r in results])):
        result["p_holm"] = p
    return {"decision": "keep" if powered(rows) else "underpowered", "tests": results,
            "B1": CONDITIONED_B1 if conditioned else "blind B1",
            "diagnostics": {"boundary_arms": sum(r.get("positives") in {0, r.get("n_sends")} for r in rows) if rows[0]["outcome_family"] != "pme" else 0,
                            "solver": "closed-form ridge; no iterative convergence"}}


def rankdata(values):
    values = np.asarray(values)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ranks = np.cumsum(counts) - (counts - 1) / 2
    return ranks[inverse]


def spearman(a, b):
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def replay(row, *, blend=None, facet_weights=None, axis_weights=None, roi_factor=1.0, priors_on=True, disabled_flag=None):
    inputs = row.get("blend_inputs")
    if inputs is None:
        # ponytail: pre-WS10B logs are rounded; parity fails rather than inventing missing precision.
        inputs = {"evidence": row["evidence_score"], "raw_llm": row.get("raw_llm_score"),
                  "facet": row.get("context_fit_score"), "confidence": row["confidence"],
                  "quality_weight": row["quality_weight"], "blend": row.get("base_semantic_blend_weight", SEMANTIC_BLEND_WEIGHT)}
    inputs = dict(inputs)
    for key in ("evidence", "raw_llm", "facet"):
        if inputs.get(key) is not None:
            finite(inputs[key], key, 0, 100)
    for key in ("confidence", "quality_weight", "blend"):
        finite(inputs[key], key, 0, 1)
    priors = row.get("prior_inputs")
    if priors is not None:
        priors = dict(priors)
        for key in ("evidence_without_roi", "facet_without_boost", "tailoring_points", "specificity_points"):
            if priors.get(key) is not None:
                finite(priors[key], key, 0 if key in {"evidence_without_roi", "facet_without_boost"} else None,
                       100 if key in {"evidence_without_roi", "facet_without_boost"} else None)
    if row.get("neural_signals") and row.get("raw_features"):
        diagnostics = _response_quality_diagnostics(row["raw_features"], row)
        evidence = {"warnings": row.get("warnings", []), "calibration_quality": diagnostics}
        rebuilt = pf.evidence_score_from_analysis(row["neural_signals"], evidence)
        if priors is None:
            inputs["evidence"] = rebuilt
        elif abs(rebuilt - priors["evidence_without_roi"]) > .05:
            raise ValueError("Persuasion feature replay differs from logged pre-ROI evidence")
    if disabled_flag:
        if priors is None:
            raise ValueError("Flag counterfactual requires logged prior_inputs")
        if disabled_flag == "PITCHCHECK_ROI_PRIOR_WEIGHT":
            inputs["evidence"] = priors["evidence_without_roi"]
        elif disabled_flag == "PITCHCHECK_PERSONA_READ_PRIORS":
            inputs["facet"] = priors.get("facet_without_boost", inputs["facet"])
            priors["tailoring_points"] = 0.0
        elif disabled_flag in {"PITCHCHECK_SPECIFICITY_CONTRAST", "PITCHCHECK_SPECIFICITY_SCORE_WEIGHT"}:
            priors["specificity_points"] = 0.0
        elif disabled_flag == "PITCHCHECK_LITERATURE_PRIOR_BUDGET":
            priors_on = False
        else:
            raise ValueError("Unknown flag counterfactual")
    if not priors_on and priors is not None:
        inputs["evidence"] = priors["evidence_without_roi"]
        inputs["facet"] = priors.get("facet_without_boost", inputs["facet"])
        priors = None
    elif not priors_on and row.get("score_terms") and row.get("final_score_without_literature_priors") is not None:
        raise ValueError("Prior replay requires prior_inputs; overlapping terms cannot be subtracted")
    if blend is not None:
        inputs["blend"] = blend
    if facet_weights is not None:
        if set(facet_weights) - row.get("facets", {}).keys():
            raise ValueError("Missing facets for sweep")
        inputs["facet"] = sum(row["facets"][k] * v for k, v in facet_weights.items())
    if axis_weights is not None:
        if set(axis_weights) - row.get("axes", {}).keys():
            raise ValueError("Missing axes for sweep")
        axis = sum(row["axes"][k] * v for k, v in axis_weights.items())
        old_roi = inputs["evidence"] - priors["evidence_without_roi"] if priors is not None else 0
        axis_evidence = pf.clamp(50 + (axis - 50) * inputs["quality_weight"])
        inputs["evidence"] = pf.clamp(axis_evidence + old_roi)
        if priors is not None:
            priors["evidence_without_roi"] = axis_evidence
        inputs["confidence"] = max(.35, min(.90, .62 + abs(inputs["evidence"] - 50) / 50 * .22,
                                                   .45 + inputs["quality_weight"] * .38 if inputs["quality_weight"] < .99 else .90))
    if roi_factor != 1.0:
        if priors is None or "evidence_without_roi" not in priors:
            raise ValueError("ROI sweep requires logged prior_inputs")
        inputs["evidence"] = priors["evidence_without_roi"] + (inputs["evidence"] - priors["evidence_without_roi"]) * roi_factor
    if priors is not None and (row.get("literature_prior_budget") or {}).get("budget") is not None:
        priors["budget"] = finite(row["literature_prior_budget"]["budget"], "budget", 0)
    return _blend_final(**inputs, priors=priors)


def compare(rows, *, outcomes=True):
    scores = [r for r in rows if r.get("source", "score") == "score"]
    refine = [r for r in rows if r.get("source") == "refine_candidate" and r.get("selected")]
    active, without, bands = [], [], []
    for row in scores:
        validate_row(row, outcomes=False)
        value, _, _, guards, _ = replay(row)
        if abs(value - finite(row.get("final_score"), "final_score", 0, 100)) > .05 + 1e-10:
            raise ValueError("Active score parity error > 0.05; use matching code and full-precision inputs")
        active.append(value)
        without.append(replay(row, priors_on=False)[0])
        bands.append("llm_score_clamped_to_neural_band" in guards)
    if not scores:
        raise ValueError("No score rows for parity comparison")
    result = {"decision": "keep", "n": len(scores), "parity": "pass", "spearman_rho": spearman(active, without),
              "mean_absolute_delta": float(np.mean(np.abs(np.array(active) - without))),
              "band_crossing_rate": float(np.mean(bands)),
              "refine_winner_change_rate": float(np.mean([r.get("selected_id", r.get("id")) != r["selected_id_semantic_only"] for r in refine])) if refine else None}
    if outcomes and all("outcome" in r and "group_id" in r for r in scores):
        for row in scores:
            validate_row(row)
        y, _ = targets(scores)
        g = np.array([r["group_id"] for r in scores])
        try:
            result.update(active_C_w=concordance(np.array(active), y, pairs(g)), without_C_w=concordance(np.array(without), y, pairs(g)))
        except ValueError:
            result.update(active_C_w=None, without_C_w=None, outcome_metric_reason="No comparable within-group outcome pairs")
    return result


def describe(rows):
    columns = [(field, k) for field in ("neural_signals", "axes")
               for k in sorted(set.intersection(*(set(r.get(field, {})) for r in rows)))]
    if not columns:
        raise ValueError("No shared signal/axis columns")
    X = np.array([[finite(r[f][k], k, 0, 100) for f, k in columns] for r in rows])
    corr = np.zeros((len(columns), len(columns)))
    vif = []
    for i in range(len(columns)):
        for j in range(len(columns)):
            corr[i, j] = np.corrcoef(X[:, i], X[:, j])[0, 1] if np.ptp(X[:, i]) and np.ptp(X[:, j]) else 0
        target = X[:, i] - X[:, i].mean()
        rest = np.delete(X - X.mean(axis=0), i, axis=1)
        residual = target - rest @ np.linalg.lstsq(rest, target, rcond=None)[0]
        r2 = 1 - residual @ residual / (target @ target) if target @ target else 1
        vif.append(float(1 / (1 - r2)) if r2 < 1 - 1e-10 else None)
    # Use the production axis formula for the memorability cross-loading ablation.
    ablated = []
    for r in rows:
        if "memorability" in r.get("neural_signals", {}):
            original = pf.neuro_axes_from_analysis(r["neural_signals"])
            changed = pf.neuro_axes_from_analysis({**r["neural_signals"], "memorability": 50.0})
            ablated.append({k: changed[k]["score"] - original[k]["score"] for k in original})
    return {"decision": "keep", "columns": [f"{f}.{k}" for f, k in columns], "correlations": corr.tolist(),
            "VIF": vif, "VIF_null_means": "constant or collinear", "memorability_ablation_mean_abs_axis_delta": {
                k: float(np.mean([abs(r[k]) for r in ablated])) for k in ablated[0]} if ablated else {},
            "prior_distribution": compare(rows), "reward_affect_vs_emotional_engagement": spearman(
                [r.get("axes", {}).get("reward_affect", 50) for r in rows],
                [r.get("neural_signals", {}).get("emotional_engagement", 50) for r in rows])}


def formula_test(old, new, y, groups, *, B=1000, seed=10):
    P = pairs(groups, folds_for(groups, seed=seed))
    a, valid = pair_hits(old, y, P)
    b, _ = pair_hits(new, y, P)
    changes, counts = [], []
    for group in np.unique(groups):
        mask = (groups[P[:, 0]] == group) & valid
        changes.append(float(np.sum((b - a)[mask])))
        counts.append(int(mask.sum()))
    delta = concordance(new, y, P) - concordance(old, y, P)
    signs = np.random.default_rng(seed).choice([-1, 1], size=(B, len(changes)))
    null = signs @ np.array(changes) / max(1, sum(counts))
    return {"delta": delta, "ci": bootstrap_delta(old, new, y, groups, P, seed=seed),
            "p": float((1 + np.sum(np.abs(null) >= abs(delta) - 1e-12)) / (B + 1))}


def simplex(keys):
    # ponytail: coarse .10 one-coordinate moves; expand only with a new pre-registration.
    current = CONTEXT_FIT_WEIGHTS if set(keys) == set(CONTEXT_FIT_WEIGHTS) else pf.NEURO_AXIS_WEIGHTS
    yield dict(current)
    for a, b in itertools.permutations(keys, 2):
        if current[b] >= .10:
            yield {**current, a: current[a] + .10, b: current[b] - .10}


def temporal_split(rows):
    if len(rows) < 2:
        raise ValueError("Temporal evaluation requires at least two arms")
    for row in rows:
        try:
            value = datetime.fromisoformat(row["scored_at"].replace("Z", "+00:00"))
            if value.tzinfo is None:
                raise ValueError("Timezone required")
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError("Temporal evaluation requires timezone-aware scored_at") from exc
    times = [datetime.fromisoformat(r["scored_at"].replace("Z", "+00:00")).timestamp() for r in rows]
    order = np.argsort(times, kind="stable")
    boundary = times[order[max(1, int(len(rows) * .75))]]
    test = np.array(times) >= boundary
    held_groups = {r["group_id"] for r, mask in zip(rows, test) if mask}
    train = np.array([not mask and r["group_id"] not in held_groups for r, mask in zip(rows, test)])
    return train, test


def sweep(rows, *, B=1000, seed=10):
    compare(rows, outcomes=False)  # Check parity without evaluating holdout outcomes.
    train, test = temporal_split(rows)
    if train.sum() < 2 or len({r["group_id"] for r, use in zip(rows, train) if use}) < 2:
        return {"decision": "underpowered", "candidates": [], "reason": "No group-disjoint temporal training slice"}
    y, _ = targets(rows)
    g = np.array([r["group_id"] for r in rows])
    old = np.array([replay(r)[0] for r in rows])
    candidates = [("SEMANTIC_BLEND_WEIGHT", round(v, 2), {"blend": round(v, 2)}) for v in np.arange(.35, .86, .1)]
    for name, weights, keyword in (("CONTEXT_FIT_WEIGHTS", CONTEXT_FIT_WEIGHTS, "facet_weights"),
                                   ("NEURO_AXIS_WEIGHTS", pf.NEURO_AXIS_WEIGHTS, "axis_weights")):
        field = "facets" if keyword == "facet_weights" else "axes"
        if all(set(weights) <= set(r.get(field, {})) for r in rows):
            candidates.extend((name, v, {keyword: v}) for v in simplex(list(weights)))
    unavailable = []
    if all(r.get("prior_inputs") and "evidence_without_roi" in r["prior_inputs"] for r in rows):
        weights = {(r.get("active_flags") or {}).get("PITCHCHECK_ROI_PRIOR_WEIGHT", .15) for r in rows}
        if len(weights) != 1 or next(iter(weights)) <= 0:
            raise ValueError("ROI weight sweep needs a common positive logged prior weight")
        weight = next(iter(weights))
        candidates.extend(("PITCHCHECK_ROI_PRIOR_WEIGHT", factor * weight, {"roi_factor": factor}) for factor in (0, .5, 1, 2))
    else:
        unavailable.append("PITCHCHECK_ROI_PRIOR_WEIGHT: missing ROI replay inputs")
    # Selection-point sweeps require candidate-set outcomes, not /score outcomes.
    unavailable.append("PITCHCHECK_ROI_PRIOR_POINTS: supply --refine-input with candidate outcomes for selection sweep")
    results, predictions = [], []
    for name, value, kwargs in candidates:
        new = np.array([replay(r, **kwargs)[0] for r in rows])
        result = formula_test(old[train], new[train], y[train], g[train], B=B, seed=seed)
        result.update(constant=name, value=value, CV_C_w=concordance(new[train], y[train], pairs(g[train], folds_for(g[train], seed=seed))))
        results.append(result)
        predictions.append(new)
    adjusted = holm([r["p"] for r in results])
    for result, p in zip(results, adjusted):
        result["p_holm"] = p
    best = max(range(len(results)), key=lambda i: results[i]["CV_C_w"])
    winner = results[best]
    try:
        holdout = concordance(predictions[best][test], y[test], pairs(g[test])) - concordance(old[test], y[test], pairs(g[test]))
    except ValueError:
        holdout = None
    train_rows = [r for r, use in zip(rows, train) if use]
    winner_name = f"{winner['constant']}={json.dumps(winner['value'], sort_keys=True)}"
    verdict = decision(train_rows, winner["delta"], winner["ci"], winner["p_holm"], holdout is not None and holdout > 0, candidate=winner_name)
    return {"decision": verdict, "candidates": results, "winner": winner_name, "temporal_delta": holdout,
            "n_train": int(train.sum()), "n_holdout": int(test.sum()), "unavailable": unavailable,
            "selection": "fixed formula group CV on first 75%; holdout evaluated only for selected candidate"}


def rollback_alarm(rows, flag, *, B=1000, seed=10):
    compare(rows)
    if not powered(rows):
        return {"decision": "underpowered", "reason": "underpowered — monitoring only"}
    active = np.array([replay(r)[0] for r in rows])
    without = np.array([replay(r, disabled_flag=flag)[0] for r in rows])
    y, _ = targets(rows)
    groups = np.array([r["group_id"] for r in rows])
    result = formula_test(without, active, y, groups, B=B, seed=seed)
    result["decision"] = decision(rows, result["delta"], result["ci"], flag=flag)
    return result


def randomized_arm_alarm(rows, flag, *, B=1000, seed=10):
    for row in rows:
        validate_row({**row, "source": "selection_outcome"})
        if row.get("data_source") != "randomized_campaign" or row.get("arm") not in {"with_neural", "semantic_only"}:
            raise ValueError("Require randomized with_neural/semantic_only arms")
        if row.get("edited_after_score"):
            raise ValueError("Edited arms belong in sensitivity analysis")
    if len({tuple(r.get(k) for k in STRATA) for r in rows}) != 1:
        raise ValueError("Do not pool randomized-arm provenance strata")
    arm_sets = defaultdict(list)
    for row in rows:
        arm_sets[row["group_id"]].append(row["arm"])
    if any(sorted(arms) != ["semantic_only", "with_neural"] for arms in arm_sets.values()):
        raise ValueError("Aggregate to one row per randomized policy arm and group")
    if not powered(rows):
        return {"decision": "underpowered", "reason": "underpowered — monitoring only"}
    y, _ = targets(rows)
    active = np.array([float(r["arm"] == "with_neural") for r in rows])
    result = formula_test(1 - active, active, y, np.array([r["group_id"] for r in rows]), B=B, seed=seed)
    result["decision"] = decision(rows, result["delta"], result["ci"], flag=flag)
    result["scope"] = "Randomized selection policy arms; no score/B1 rows are fitted"
    return result


def sensitivity_reports(rows, *, B=1000, seed=10, conditioned=False):
    reports = []
    variants = []
    if any(r.get("edited_after_score") for r in rows):
        variants.append(("include_edited_arms", [{**r, "edited_after_score": False} for r in rows]))
    if any(r.get("data_source") == "upworthy" and r.get("winner") for r in rows):
        variants.append(("exclude_upworthy_winners", [r for r in rows if not (r.get("data_source") == "upworthy" and r.get("winner"))]))
    primary, _ = eligible_strata(rows, allow_conditioned=conditioned)
    trimmed = []
    removed = 0
    for records in primary.values():
        y, _ = targets(records)
        residual = demean(y, np.array([r["group_id"] for r in records]))
        sd = np.std(residual)
        keep = np.abs(residual) <= 2.5 * sd if sd else np.ones(len(records), bool)
        trimmed.extend(r for r, use in zip(records, keep) if use)
        removed += int((~keep).sum())
    if removed:
        variants.append(("exclude_within_group_outcome_abs_z_gt_2.5", trimmed))
    for name, records in variants:
        strata, exclusions = eligible_strata(records, allow_conditioned=conditioned)
        for key, rs in strata.items():
            report = increment(rs, B=B, seed=seed, conditioned=conditioned)
            report.update(decision="descriptive_sensitivity_only", sensitivity=name, stratum=dict(zip(STRATA, key)),
                          exclusions=exclusions, n_arms=len(rs))
            reports.append(report)
    return reports


def selection_sweep(rows, *, B=1000, seed=10):
    """WS09 candidate ranking, kept completely separate from score/blind-B1 rows."""
    for row in rows:
        validate_row({**row, "source": "selection_outcome"})
        for key in ("selection_score_without_roi", "roi_selection_unit", "roi_prior_points", "selection_score"):
            finite(row.get(key), key)
        if row.get("data_source") != "randomized_campaign":
            raise ValueError("Selection sweep requires randomized candidate outcomes")
        if abs(row["selection_score_without_roi"] + row["roi_selection_unit"] * row["roi_prior_points"] - row["selection_score"]) > .005:
            raise ValueError("Selection score parity error")
    if len({r["roi_prior_points"] for r in rows}) != 1:
        raise ValueError("Do not pool selection-point versions")
    if len({tuple(r.get(k) for k in STRATA) for r in rows}) != 1:
        raise ValueError("Do not pool selection candidate provenance strata")
    prior = rows[0]["roi_prior_points"]
    old = np.array([r["selection_score"] for r in rows])
    y, _ = targets(rows)
    groups = np.array([r["group_id"] for r in rows])
    train, test = temporal_split(rows)
    results, predictions = [], []
    for factor in (0, .5, 1, 2):
        points = prior * factor
        new = np.array([r["selection_score_without_roi"] + r["roi_selection_unit"] * points for r in rows])
        result = formula_test(old[train], new[train], y[train], groups[train], B=B, seed=seed)
        result.update(constant="PITCHCHECK_ROI_PRIOR_POINTS", value=points,
                      CV_C_w=concordance(new[train], y[train], pairs(groups[train])))
        results.append(result)
        predictions.append(new)
    for result, p in zip(results, holm([r["p"] for r in results])):
        result["p_holm"] = p
    best = max(range(len(results)), key=lambda i: results[i]["CV_C_w"])
    winner = results[best]
    try:
        holdout = concordance(predictions[best][test], y[test], pairs(groups[test])) - concordance(old[test], y[test], pairs(groups[test]))
    except ValueError:
        holdout = None
    name = f"PITCHCHECK_ROI_PRIOR_POINTS={json.dumps(winner['value'])}"
    return {"decision": decision([r for r, use in zip(rows, train) if use], winner["delta"], winner["ci"], winner["p_holm"], holdout is not None and holdout > 0, candidate=name),
            "winner": name, "candidates": results, "temporal_delta": holdout,
            "scope": "predictive association in randomized candidate rankings; does not establish policy response uplift"}


def write_report(report, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    markdown = REPORT_TEMPLATE.format(decision=report.get("decision", "see strata"), scope=report.get("scope", "predictive association only"), details=payload)
    paths = [output.with_suffix(".json"), output.with_suffix(".md")]
    if any(path.exists() for path in [*paths, output.with_suffix(".sha256")]):
        raise FileExistsError("Report already exists; use a new output name")
    for path, content in zip(paths, (payload, markdown)):
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content)
    with output.with_suffix(".sha256").open("x", encoding="utf-8") as handle:
        handle.write("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("describe", "compare", "increment", "sweep", "tune", "rollback_alarm"))
    parser.add_argument("input", type=Path)
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument("--blind-input", type=Path, help="Match separate blind B1 reads without replacing product scores")
    parser.add_argument("--refine-input", type=Path, help="Separate candidate rows joined to randomized outcomes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--conditioned-b1", action="store_true")
    parser.add_argument("--flag", default="PITCHCHECK_LITERATURE_PRIOR_BUDGET")
    parser.add_argument("--randomized-arms", action="store_true", help="Alarm from randomized policy outcomes, separate from score replay")
    args = parser.parse_args()
    if args.permutations < 199:
        parser.error("Use at least 199 permutations (1,000 for real analyses)")
    rows = read_jsonl(args.input)
    for row in rows:
        validate_row(row, outcomes=False)
    source_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if args.blind_input:
        rows = join_blind([r for r in rows if r.get("source", "score") == "score"], read_jsonl(args.blind_input))
    if args.outcomes:
        rows = join_outcomes([r for r in rows if r.get("source", "score") == "score"], read_jsonl(args.outcomes))
    if args.command in {"increment", "sweep", "tune", "rollback_alarm"} and args.permutations < 1000 and any(not r.get("synthetic") for r in rows):
        parser.error("Real analyses require at least 1,000 permutations; 199 is for synthetic CI")
    if args.randomized_arms:
        if args.command != "rollback_alarm":
            parser.error("--randomized-arms applies only to rollback_alarm")
        reports = [randomized_arm_alarm(rows, args.flag, B=args.permutations, seed=args.seed)]
        exclusions = {}
    elif args.command in {"compare", "describe"}:
        # Outcome comparisons must still respect all provenance strata.
        strata = defaultdict(list)
        for row in rows:
            row.setdefault("channel", row.get("platform"))
            strata[tuple(row.get(k) for k in STRATA)].append(row)
        reports = [{"stratum": dict(zip(STRATA, key)), **(compare(rs) if args.command == "compare" else describe(rs))}
                   for key, rs in strata.items() if any(r.get("source", "score") == "score" for r in rs)]
        exclusions = {}
    else:
        strata, exclusions = eligible_strata(rows, allow_conditioned=args.conditioned_b1 or args.command != "increment")
        reports = []
        for key, rs in strata.items():
            if args.command == "increment":
                result = increment(rs, B=args.permutations, seed=args.seed, conditioned=args.conditioned_b1)
            elif args.command in {"sweep", "tune"}:
                result = sweep(rs, B=args.permutations, seed=args.seed)
            else:
                result = rollback_alarm(rs, args.flag, B=args.permutations, seed=args.seed)
            reports.append({"stratum": dict(zip(STRATA, key)), "n_arms": len(rs), **result})
        # Correct every stratum/candidate together, not just each individual run.
        candidates = [r for report in reports for r in report.get("tests", report.get("candidates", []))]
        for result, p in zip(candidates, holm([r["p"] for r in candidates])):
            result["p_holm"] = p
        for report in reports:
            if report["decision"].startswith("tune:"):
                chosen = next(r for r in report["candidates"] if f"{r['constant']}={json.dumps(r['value'], sort_keys=True)}" == report["winner"])
                if chosen["p_holm"] >= .05:
                    report["decision"] = "keep"
    if args.refine_input:
        if args.command not in {"sweep", "tune"}:
            parser.error("--refine-input applies to sweep/tune only")
        selection = selection_sweep(read_jsonl(args.refine_input), B=args.permutations, seed=args.seed)
        reports.append({"track": "selection", **selection})
        candidates = [r for report in reports for r in report.get("candidates", [])]
        for result, p in zip(candidates, holm([r["p"] for r in candidates])):
            result["p_holm"] = p
        for report in reports:
            if report["decision"].startswith("tune:"):
                chosen = next(r for r in report["candidates"] if f"{r['constant']}={json.dumps(r['value'], sort_keys=True)}" == report["winner"])
                if chosen["p_holm"] >= .05:
                    report["decision"] = "keep"
    verdicts = {r["decision"] for r in reports}
    report = {"dataset_id": args.dataset_id, "input_sha256": source_hash, "command": args.command,
              "decision": next(iter(verdicts)) if len(verdicts) == 1 else "see_strata" if verdicts else "underpowered",
              "underpowered_message": "underpowered — monitoring only" if not reports or "underpowered" in verdicts else None,
              "protocol_sha256": hashlib.sha256((ROOT / "docs/validation-protocol.md").read_bytes()).hexdigest(),
              "outcomes_sha256": hashlib.sha256(args.outcomes.read_bytes()).hexdigest() if args.outcomes else None,
              "blind_sha256": hashlib.sha256(args.blind_input.read_bytes()).hexdigest() if args.blind_input else None,
              "refine_sha256": hashlib.sha256(args.refine_input.read_bytes()).hexdigest() if args.refine_input else None,
              "seed": args.seed, "permutations": args.permutations, "bootstrap": 2000,
              "synthetic": bool(rows) and all(r.get("synthetic") for r in rows),
              "implementation_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                        for path in (Path(__file__), ROOT / "tribe_service/llm_layer.py", ROOT / "tribe_service/persuasion_features.py")},
              "exclusions": exclusions, "strata": reports,
              "scope": PME_SCOPE if any(r.get("outcome_family") == "pme" for r in rows) else "predictive association only; production data are descriptive monitoring",
              "calibration_basis": sorted({r.get("calibration_basis") or "unvalidated_monitoring" for r in rows}),
              "recommended_calibration_basis": f"fitted:{args.dataset_id}" if any(d.startswith("tune:") for d in verdicts) else None}
    if args.command == "increment":
        report["sensitivities"] = sensitivity_reports(rows, B=args.permutations, seed=args.seed, conditioned=args.conditioned_b1)
    selected = [r for r in rows if r.get("source") == "refine_candidate" and r.get("selected")]
    report["refine_monitoring"] = {"n_selected": len(selected), "winner_change_rate": float(np.mean([
        r.get("selected_id", r.get("id")) != r.get("selected_id_semantic_only") for r in selected])) if selected else None}
    write_report(report, args.output)


if __name__ == "__main__":
    main()
