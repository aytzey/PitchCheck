"""WS08: offline scoring (GPU) and analysis (CPU); never calls an LLM or a server.

Run only after fixture review and committing the sidecars, in a clean checkout:
  git merge-base --is-ancestor f6b5cd9 HEAD && test -z "$(git status --porcelain)"
Use a maintenance window: stop the live tribe container first, or use Vast >=24 GB.
  docker compose -f deploy/pitchserver.compose.yml stop tribe
  docker run --rm --gpus all --cpus 4 --memory 12g \
    -e TRIBE_ALLOW_MOCK=0 -e TRIBE_PREDICTION_CACHE_SIZE=0 \
    -e TRIBE_TEXT_INPUT_MODE=direct -e TRIBE_TEXT_MODEL=meta-llama/Llama-3.2-3B \
    -e TRIBE_CUDA_MEMORY_LIMIT_GB=8 -e HF_HUB_OFFLINE=1 \
    -v "$PWD:/app:ro" -v /models:/models -v /audit/insilico:/audit/insilico \
    "$IMG" python scripts/research/run_contrasts.py score --family stage0 \
    --git-sha "$(git rev-parse HEAD)" --out /audit/insilico/<run_id>
  docker compose -f deploy/pitchserver.compose.yml start tribe
  python scripts/research/run_contrasts.py analyze --run /audit/insilico/<run_id>
Use a normal checkout with accessible .git metadata inside the image. Match the
production tribev2 revision and model mounts; accept the Llama license beforehand.
Use a separate process with TRIBE_TEXT_MODEL=NousResearch/Hermes-3-Llama-3.2-3B
for Hermes Stage 0 and S2. Run S2 -> S4 -> S1 -> S3 for Llama only.
See docs/insilico/README.md for reference output and positive-control arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research import stats

FIXTURES = Path(__file__).with_name("fixtures")
POLICY = {"pass": 1.0, "pass_broad": .75, "null": .5, "reversed": 0.0,
          "not_run": 1.0, "uninformative": 1.0, "low_stability": .5}
LABEL = "relative to a synthetic reference set (n≈300), not real pitches"


def sha256(item):
    return hashlib.sha256(item.read_bytes()).hexdigest()


def write_json(item, value):
    item.parent.mkdir(parents=True, exist_ok=True)
    item.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def check_diversity(rows):
    """Reject cross-pair replication; matched conditions and derived controls are intentional."""
    from tribe_service.engine import _WORD_RE
    sentences, texts = {}, []
    for row in rows:
        if row.get("role") in {"structure", "retest"}:
            continue
        pair = row["pair_id"]
        for sentence in re.split(r"[.!?]+(?=\s|$)", row["text"]):
            normalized = tuple(_WORD_RE.findall(sentence.casefold()))
            if not normalized:
                continue
            other = sentences.setdefault(normalized, pair)
            if other != pair:
                raise ValueError(f"Fixture diversity: repeated sentence in {other} and {pair}")
        words = _WORD_RE.findall(row["text"].casefold())
        grams = {tuple(words[i:i + 8]) for i in range(len(words) - 7)}
        texts.append((pair, row["condition"], grams))
    for i, (pair, condition, grams) in enumerate(texts):
        for other, other_condition, other_grams in texts[:i]:
            if pair == other or not grams or not other_grams:
                continue
            overlap = len(grams & other_grams) / min(len(grams), len(other_grams))
            if overlap >= .10:
                raise ValueError(f"Fixture diversity: 8-gram overlap {overlap:.3f} >= .10 "
                                 f"in {other}/{other_condition} and {pair}/{condition}")


def load_fixtures(family, fixtures=FIXTURES):
    from tribe_service.engine import _WORD_RE
    sidecar_path = fixtures / f"{family}.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    item = fixtures / sidecar["stimuli_file"]
    if item.parent.resolve() != fixtures.resolve() or sha256(item) != sidecar["stimuli_sha256"]:
        raise ValueError("Fixture hash/path mismatch")
    if sidecar["schema_version"] != 1 or sidecar["family"] != family:
        raise ValueError("Invalid pre-registration")
    rows = [json.loads(line) for line in item.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if all(row.get(k) == v for k, v in sidecar.get("select", {}).items())]
    if not rows:
        raise ValueError("Empty fixture")
    groups, seen = {}, set()
    for row in rows:
        identity = (row["pair_id"], row["condition"])
        if identity in seen or not isinstance(row["text"], str):
            raise ValueError("Duplicate condition or invalid text")
        seen.add(identity)
        count = len(_WORD_RE.findall(row["text"]))
        if not 1 <= count <= 240:
            raise ValueError("Each text must contain 1–240 words")
        groups.setdefault(row["pair_id"], []).append(count)
    for counts in groups.values():
        if max(counts) > min(counts) * 1.1 or (sidecar.get("minimal_pairs") and max(counts) - min(counts) > 2):
            raise ValueError("Unmatched within-pair lengths")
    check_diversity(rows)
    for contrast in sidecar.get("contrasts", []):
        selected = [r for r in rows if all(r.get(k) == v for k, v in contrast.get("select", {}).items())]
        counts = [sum(r["condition"] == c for r in selected) for condition in contrast["conditions"]
                  for c in (condition if isinstance(condition, list) else [condition])]
        if min(counts) < sidecar["minimum_pairs"]:
            raise ValueError("Insufficient pre-registered pairs")
    return sidecar, rows, [sidecar_path, item]


def committed_preregistration(paths, repo, git_sha):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    if git("rev-parse", "HEAD") != git_sha or git("status", "--porcelain"):
        raise ValueError("Scoring requires the supplied revision and a clean checkout")
    relative = [str(p.resolve().relative_to(repo.resolve())) for p in paths]
    git("ls-files", "--error-unmatch", "--", *relative)
    return git("log", "-1", "--format=%H", "--", *relative)


def check_runtime(engine):
    if engine.TRIBE_ALLOW_MOCK or engine.TRIBE_PREDICTION_CACHE_SIZE != 0 or engine.TRIBE_TEXT_INPUT_MODE != "direct":
        raise ValueError("Requires mock OFF, prediction cache 0, direct input")


def check_prediction(text, pred):
    from tribe_service import engine
    n = len(engine._WORD_RE.findall(text))
    expected = math.ceil(engine.WORD_SPACING * (n - 1) + engine.WORD_DURATION)
    if pred.ndim != 2 or pred.shape != (expected, 20484) or not np.isfinite(pred).all():
        raise ValueError(f"Invalid prediction shape/values: {pred.shape}, expected {(expected, 20484)}")


def secondary_window(row, pred, assets):
    from tribe_service import engine
    if "cost_word_index" not in row:
        return None
    matches = list(engine._WORD_RE.finditer(row["text"]))
    index = row["cost_word_index"]
    if not isinstance(index, int) or not 0 <= index < len(matches):
        raise ValueError("Invalid cost sentence word index")
    spans = engine.direct_text_segment_words(row["text"], len(pred), engine.DIRECT_TEXT_TIMING_BASIS)
    if spans is None:
        raise ValueError("Direct-text segment mapping unavailable")
    target = matches[index].span()
    segment = next(i for i, group in enumerate(spans) if target in group)
    start, stop = max(0, segment - 1), len(pred)
    start = min(start, max(0, stop - 4))
    if stop - start < 4:
        return {"status": "not_run", "reason": "fewer_than_four_segments"}
    roi, _ = roi_readout(pred[start:stop], assets)
    return {"start_segment": start, "stop_segment": stop, "quantities": roi,
            "basis": "descriptive_cost_sentence_to_end; bidirectional_encoder; expanded_one_segment"}


def load_assets():
    try:
        roi_features = importlib.import_module("tribe_service.roi_features")
    except ModuleNotFoundError as exc:
        if exc.name != "tribe_service.roi_features":
            raise
        return None
    return roi_features.load_roi_assets()


def roi_readout(pred, assets):
    """Use the single WS07 loader; index arrays include both K2 ROI sets."""
    if assets is None:
        return {}, None
    from tribe_service import roi_features
    summary = roi_features.roi_summary(pred)
    if summary is None:
        return {}, None
    labels, _, _, _, _ = assets
    mean_map = pred.mean(0, dtype=np.float64)
    cortex_mean = mean_map[labels > 0].mean()
    counts = np.bincount(labels, minlength=401)[1:]
    parcels = np.bincount(labels, weights=mean_map, minlength=401)[1:] / np.maximum(counts, 1) - cortex_mean
    quantities = {f"roi:{key}": row["minus_cortex"] for group in ("rois", "locations")
                  for key, row in summary[group].items()}
    right_insula = np.asarray(assets[2].get("insula", []))
    right_insula = right_insula[right_insula >= 10242]
    if len(right_insula):
        quantities["roi:insula_right"] = float(mean_map[right_insula].mean() - cortex_mean)
    return quantities, parcels.tolist()


def measure(text, pred, channel, assets, compatible=None):
    from tribe_service import app, engine, native_core
    from tribe_service.llm_layer import _refine_response_shape
    from tribe_service.persuasion_features import neural_score_from_signals
    measured = app._measure_prediction(text, pred, "", channel)
    q = {"evidence_score": measured["neural_score"], "quality_weight": measured["quality_weight"],
         "neural_score_from_signals": neural_score_from_signals(measured["neural_signals"]),
         **{f"signal:{k}": v for k, v in measured["neural_signals"].items()},
         **{f"axis:{k}": v["score"] for k, v in measured["neuro_axes"].items()}}
    for window, values in _refine_response_shape(measured).get("windows", {}).items():
        q.update({f"response_shape:{window}:{k}": v for k, v in values.items() if k != "bucket_segments"})
    roi, parcels = roi_readout(pred, assets)
    q.update(roi)
    from tribe_service import persuasion_features
    raw, fmri, signals = engine.analyze_predictions(pred)
    evidence = app._augment_persuasion_evidence(text, "", channel, raw, fmri)
    evidence.pop("roi_prior", None)
    q["evidence_score"] = round(persuasion_features.evidence_score_from_analysis(signals, evidence), 3)
    if assets and hasattr(persuasion_features, "roi_prior_from_summary"):
        from tribe_service import roi_features
        prior = persuasion_features.roi_prior_from_summary(
            roi_features.roi_summary(pred), measured["text_feature_compatible"] if compatible is None else compatible,
            roi_features.roi_multipliers(measured["text_feature_model"]), encoder=measured["text_feature_model"], word_count=len(engine._WORD_RE.findall(text)))
        if prior:
            q["evidence_score_with_roi"] = round(persuasion_features.evidence_score_from_analysis(signals, {**evidence, "roi_prior": prior}), 3)
    words = engine._WORD_RE.findall(text)
    lexical = {"words": len(words), "sentences": max(1, len(re.findall(r"[.!?]+", text))),
               "mean_word_length": float(np.mean([len(w) for w in words])),
               "global_mean_abs": measured["response_features"].get("global_mean_abs", float(np.abs(pred).mean())),
               "motor_verbs": sum(w.lower() in {"apply", "spread", "rub", "press", "click", "open", "select", "move"} for w in words),
               "body_words": sum(w.lower() in {"skin", "hand", "hands", "face", "arm", "arms", "body"} for w in words)}
    # Compare key sets with Python without changing the process-wide numeric backend flag.
    python_keys = [set(engine._extract_features_from_matrix(pred)),
                   set(engine._summarize_fmri_output_from_matrix(pred, text_input_mode="direct")),
                   set(engine._derive_persuasion_signals_from_features(raw))]
    parity = [set(raw) == python_keys[0], set(fmri) == python_keys[1], set(signals) == python_keys[2]]
    return {"quantities": q, "parcels": parcels, "lexical": lexical,
            "voxel_count": measured["voxel_count"], "segments": measured["segments"],
            "backend_key_parity": parity, "rust_numeric_enabled": native_core.numeric_enabled(),
            "text_feature_model": measured["text_feature_model"],
            "text_feature_compatible": measured["text_feature_compatible"]}


def provenance(engine, paths, git_sha, prereg_commit):
    from huggingface_hub import try_to_load_from_cache
    try:
        direct_url = json.loads(importlib.metadata.distribution("tribev2").read_text("direct_url.json") or "{}")
    except importlib.metadata.PackageNotFoundError:
        direct_url = {}
    config = try_to_load_from_cache(engine.TRIBE_TEXT_MODEL, "config.json")
    if not isinstance(config, str):
        raise ValueError("Encoder snapshot must be cached offline")
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    if not commit:
        raise ValueError("Installed tribev2 must identify its source commit")
    return {"encoder": engine.TRIBE_TEXT_MODEL, "text_input_mode": engine.TRIBE_TEXT_INPUT_MODE,
            "event_version": engine.DIRECT_TEXT_EVENT_VERSION, "word_duration": engine.WORD_DURATION,
            "word_spacing": engine.WORD_SPACING, "language": engine.TRIBE_DIRECT_TEXT_LANGUAGE,
            "tribev2_commit": commit, "tribev2_patch": "patch_tribev2_whisperx",
            "hf_snapshot": Path(config).parent.name, "runtime": engine.runtime_config(),
            "prediction_dtype": "float32", "storage_dtype": "float16",
            "fixture_sha256": {p.name: sha256(p) for p in paths}, "git_sha": git_sha,
            "preregistration_commit": prereg_commit, "image_id": os.getenv("PITCHCHECK_IMAGE_ID", "unrecorded"),
            "harness_sha256": {p.name: sha256(p) for p in (Path(__file__), Path(stats.__file__))}}


def score(args):
    from tribe_service import engine
    check_runtime(engine)
    sidecar, rows, paths = load_fixtures(args.family)
    if engine.TRIBE_TEXT_MODEL not in sidecar["encoders"]:
        raise ValueError("Encoder is not pre-registered for this family")
    commit = committed_preregistration(paths, args.repo, args.git_sha)
    review = sidecar["manipulation_check"]
    if (review.get("blind_classifier") != "passed" or not review.get("classifier_model")
            or review.get("owner_sample_fraction", 0) < .2 or review.get("owner_review") != "passed"):
        raise ValueError("Complete and commit blind classification and the owner's 20% review before scoring")
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    if args.out.exists():
        raise ValueError("Use a new run directory; existing runs are never overwritten")
    engine.load_runtime_models()
    assets = load_assets()
    prov = provenance(engine, paths, args.git_sha, commit)
    prov.update(atlas_sha256=assets[4] if assets else None,
                text_feature_compatible=engine.runtime_config()["text_feature_compatible"])
    args.out.mkdir(parents=True)
    metrics = {"family": args.family, "date": datetime.now(timezone.utc).isoformat(),
               "provenance": prov, "sidecar": sidecar, "excluded_rows": 0, "complete": False}
    write_json(args.out / "metrics.json", metrics)
    try:
        with (args.out / "rows.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                pred = engine.score_text(row["text"])
                check_prediction(row["text"], pred)
                runtime = engine.last_score_metrics()
                excluded = bool(runtime.get("fallback_used") or runtime.get("retry_index", 0) > 0)
                metrics["excluded_rows"] += int(excluded)
                key = hashlib.sha256(json.dumps([row["text"], prov], sort_keys=True).encode()).hexdigest()
                # Retest rows deliberately bypass reuse of the on-disk text cache.
                item = f"{key}-{row['condition']}.npy"
                stored = pred.astype(np.float16)
                if not np.isfinite(stored).all():
                    raise ValueError("Prediction exceeds float16 storage range")
                np.save(args.out / item, stored)
                measured = measure(row["text"], pred, row.get("channel", "general"), assets)
                stream.write(json.dumps({**row, **measured, "prediction_file": item,
                                         "prediction_sha256": sha256(args.out / item),
                                         "secondary_window": secondary_window(row, pred, assets),
                                         "runtime": runtime, "excluded": excluded}, allow_nan=False) + "\n")
                stream.flush()
        metrics["complete"] = True
    finally:
        write_json(args.out / "metrics.json", metrics)
        engine.unload_model()


def noise_summary(x, seed=0, B=2000):
    icc, sigma = stats.icc1(x)
    rng = np.random.default_rng(seed)
    draws = [stats.icc1(x[rng.integers(len(x), size=len(x))])[0] for _ in range(B)]
    return {"sigma_within": sigma, "icc1": icc, "icc1_ci95": np.quantile(draws, [.025, .975]).tolist(),
            "min_detectable_delta": float(1.96 * np.sqrt(2) * sigma)}


def reference_quantity(rows, key, B=2000):
    reference = [r for r in rows if r.get("role") in {"paraphrase", "independent"} and key in r["quantities"]]
    families = {}
    for row in reference:
        if row.get("role") == "paraphrase":
            families.setdefault(row["pair_id"], {})[row["condition"]] = row["quantities"][key]
    x = np.array([[v[f"p{i}"] for i in range(5)] for v in families.values() if len(v) == 5])
    if len(x) < 2 or len(reference) < 3:
        return None
    y = np.array([r["quantities"][key] for r in reference])
    log_words = np.log([r["lexical"]["words"] for r in reference])
    design = np.column_stack([np.ones(len(y)), log_words])
    a, b = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ [a, b]
    total = ((y - y.mean()) ** 2).sum()
    def r2(X):
        return float(1 - np.sum(stats.residualize(y, X) ** 2) / total) if total else 0.0
    confound = np.array([[np.log(r["lexical"]["words"]), r["lexical"]["sentences"],
                         r["lexical"]["mean_word_length"], r["lexical"]["global_mean_abs"]] for r in reference])
    noise = noise_summary(x, B=B)
    stability = "stable" if noise["icc1"] >= .6 else "moderate" if noise["icc1"] >= .4 else "low_stability"
    return {"n_reference": len(reference), "n_paraphrase_families": len(x), "noise": noise,
            "length_model": {"a": float(a), "b_log_words": float(b)}, "resid_sd": float(residual.std(ddof=1)),
            "resid_q": dict(zip(("p05", "p25", "p50", "p75", "p95"), np.quantile(residual, [.05, .25, .5, .75, .95]).tolist())),
            "confound_r2": {**{name: r2(confound[:, i]) for i, name in enumerate(("log_words", "sentences", "mean_word_length", "global_mean_abs"))}, "joint": r2(confound)},
            "flags": [stability] + (["length_confounded"] if r2(log_words) > .5 else [])}


def status(delta, p_holm, reverse_p_holm, mdd, rank, alternative):
    direction = 1 if alternative == "greater" else -1
    if direction * np.mean(delta) < 0 and reverse_p_holm < .05:
        return "reversed"
    if direction * np.mean(delta) > 0 and p_holm < .05 and abs(np.median(delta)) > mdd:
        return "pass" if rank >= .90 else "pass_broad"
    return "null"


def multiplier(check_status, flags=(), informative=True):
    if not informative:
        return 1.0
    return POLICY[check_status] * (POLICY["low_stability"] if "low_stability" in flags else 1.0)


def paired_rows(rows, contrast):
    grouped = {}
    for row in rows:
        if all(row.get(k) == v for k, v in contrast.get("select", {}).items()):
            grouped.setdefault(row["pair_id"], {})[row["condition"]] = row
    left, right = contrast["conditions"]
    def condition(group, names):
        names = names if isinstance(names, list) else [names]
        if not all(name in group for name in names):
            return None
        selected = [group[name] for name in names]
        first = selected[0]
        if len(selected) == 1:
            return first
        return {**first, "quantities": {k: float(np.mean([r["quantities"][k] for r in selected]))
                                        for k in set.intersection(*(set(r["quantities"]) for r in selected))},
                "parcels": np.mean([r["parcels"] for r in selected], axis=0).tolist() if all(r.get("parcels") is not None for r in selected) else None,
                "lexical": {k: float(np.mean([r["lexical"][k] for r in selected])) for k in first["lexical"]}}
    result = [(condition(g, left), condition(g, right)) for g in grouped.values()]
    return [(a, b) for a, b in result if a is not None and b is not None]


def analyze_suite(rows, sidecar, reference, positive_control, B=10_000):
    tests = []
    for contrast in sidecar["contrasts"]:
        pairs = paired_rows(rows, contrast)
        for roi in contrast["rois"]:
            key = f"roi:{roi}"
            complete = [(a, b) for a, b in pairs if key in a["quantities"] and key in b["quantities"]
                        and a.get("parcels") is not None and b.get("parcels") is not None]
            if len(complete) < sidecar["minimum_pairs"]:
                tests.append({"roi": roi, "contrast": contrast["name"], "status": "not_run",
                              "primary": contrast.get("primary", True),
                              "reason": "missing_roi_or_complete_pairs", "n": len(complete), "weight_multiplier": 1.0})
                continue
            d = np.array([a["quantities"][key] - b["quantities"][key] for a, b in complete])
            parcels = np.array([np.subtract(a["parcels"], b["parcels"]) for a, b in complete])
            effect = stats.dz(d)
            entry = {"roi": roi, "contrast": contrast["name"], "n": len(d), "alternative": contrast["alternative"],
                     "primary": contrast.get("primary", True), "holm_group": contrast.get("holm_group", "primary"),
                     "d_z": effect if np.isfinite(effect) else None, "mean_delta": float(d.mean()),
                     "median_delta": float(np.median(d)), "ci95": stats.boot_ci(d),
                     "p": stats.signflip_p(d, contrast["alternative"], B=B),
                     "reverse_p": stats.signflip_p(d, "two-sided", B=B),
                     "parcel_rank": stats.parcel_rank([stats.dz(v) for v in parcels.T], effect, contrast["alternative"])}
            lexical = np.array([[a["lexical"][k] - b["lexical"][k] for k in ("motor_verbs", "body_words")] for a, b in complete])
            lexical = lexical[:, np.any(lexical != 0, axis=0)]
            design = np.column_stack([np.ones(len(d)), lexical])
            beta = np.linalg.lstsq(design, d, rcond=None)[0]
            adjusted = d - lexical @ beta[1:]
            identifiable = np.linalg.matrix_rank(design) == design.shape[1]
            entry["lexical_residualized"] = {"identifiable": bool(identifiable),
                "adjusted_delta": float(adjusted.mean()) if identifiable else None,
                "ci95": stats.boot_ci(adjusted) if identifiable else None,
                "p": stats.signflip_p(adjusted, contrast["alternative"], B=B) if identifiable else None,
                "note": "Descriptive covariate adjustment; primary inference uses the pre-registered raw paired difference."}
            tests.append(entry)
    computed = [t for t in tests if "p" in t]
    for group in {t["holm_group"] for t in computed}:
        members = [t for t in computed if t["holm_group"] == group]
        for t, p, reverse in zip(members, stats.holm([t["p"] for t in members]), stats.holm([t["reverse_p"] for t in members])):
            t.update(p_holm=p, reverse_p_holm=reverse)
    if computed:
        for t in computed:
            quantity = (reference or {}).get("quantities", {}).get(f"roi:{t['roi']}")
            review = sidecar.get("manipulation_check", {})
            informative = (review.get("blind_classifier") == "passed" and review.get("owner_review") == "passed"
                           and quantity is not None and (sidecar["family"] == "S2" or positive_control is True))
            if not informative:
                t.update(status="uninformative", reason="manipulation_or_reference_or_S2_positive_control", weight_multiplier=1.0)
                continue
            # Status is computed from complete paired deltas, never from a dz sign alone.
            pairs = paired_rows(rows, next(c for c in sidecar["contrasts"] if c["name"] == t["contrast"]))
            d = [a["quantities"][f"roi:{t['roi']}"] - b["quantities"][f"roi:{t['roi']}"] for a, b in pairs
                 if f"roi:{t['roi']}" in a["quantities"] and f"roi:{t['roi']}" in b["quantities"]
                 and a.get("parcels") is not None and b.get("parcels") is not None]
            t["status"] = status(d, t["p_holm"], t["reverse_p_holm"], quantity["noise"]["min_detectable_delta"], t["parcel_rank"], t["alternative"])
            t["weight_multiplier"] = multiplier(t["status"], quantity["flags"])
    checks = {}
    severity = {"reversed": 0, "uninformative": 1, "not_run": 2, "null": 3, "pass_broad": 4, "pass": 5}
    for test in tests:
        if test.get("primary") is False:
            continue
        roi = test["roi"]
        if roi not in checks or severity[test["status"]] < severity[checks[roi]["status"]]:
            checks[roi] = {**test, "family": sidecar["family"], "weight_basis": "literature_prior"}
    return {"tests": tests, "checks": checks,
            "positive_control_passed": bool(checks) and all(t["status"] in {"pass", "pass_broad"} for t in tests if t.get("primary", True))}


def average_ranks(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(order), float)
    for block in np.split(order, np.flatnonzero(np.diff(np.asarray(values)[order])) + 1):
        ranks[block] = np.mean(np.flatnonzero(np.isin(order, block)))
    return ranks


def correlation(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None


def stage0_report(rows, run, B=2000):
    retest, structure = {}, {}
    for row in rows:
        if row.get("role") == "retest":
            retest.setdefault(row["pair_id"], []).append(row)
        if row.get("role") == "structure":
            structure.setdefault(row["pair_id"], {})[row["condition"]] = row
    l2 = []
    for group in retest.values():
        first = np.load(run / group[0]["prediction_file"], allow_pickle=False).astype(float)
        for row in group[1:]:
            other = np.load(run / row["prediction_file"], allow_pickle=False).astype(float)
            l2.append(float(np.linalg.norm(other - first) / max(np.linalg.norm(first), 1e-12)))
    structure_deltas = {}
    for key in sorted({k for r in rows for k in r["quantities"]}):
        d = [g["nopunct"]["quantities"][key] - g["scrambled"]["quantities"][key] for g in structure.values()
             if {"nopunct", "scrambled"} <= g.keys() and key in g["nopunct"]["quantities"] and key in g["scrambled"]["quantities"]]
        if d:
            structure_deltas[key] = {"mean_delta": float(np.mean(d)), "ci95": stats.boot_ci(d)}
    keys = ("roi:vmpfc_frontopolar", "axis:self_value", "signal:personal_relevance")
    corpus = [r for r in rows if r.get("role") in {"paraphrase", "independent"} and all(k in r["quantities"] for k in keys)]
    correlations = {key: correlation([r["quantities"][keys[0]] for r in corpus], [r["quantities"][key] for r in corpus]) for key in keys[1:]} if corpus else None
    # WS07 owns the ROI-augmented production anchor. Never invent its future API.
    shadow = [r for r in rows if "evidence_score_with_roi" in r["quantities"]]
    shadow_report = {"status": "not_run", "reason": "WS07_augmented_anchor_not_available"}
    if shadow:
        old = [r["quantities"]["evidence_score"] for r in shadow]
        new = [r["quantities"]["evidence_score_with_roi"] for r in shadow]
        old_q = reference_quantity(shadow, "evidence_score", B)
        new_q = reference_quantity(shadow, "evidence_score_with_roi", B)
        families = {}
        for r in shadow:
            if r.get("role") == "paraphrase":
                families.setdefault(r["pair_id"], []).append(r)
        changes = [np.sign(a["quantities"]["evidence_score"] - b["quantities"]["evidence_score"]) !=
                   np.sign(a["quantities"]["evidence_score_with_roi"] - b["quantities"]["evidence_score_with_roi"])
                   for group in families.values() for i, a in enumerate(group) for b in group[i + 1:]]
        shadow_report = {"status": "measured", "spearman": correlation(average_ranks(old), average_ranks(new)),
                         "baseline_noise": old_q["noise"] if old_q else None, "roi_noise": new_q["noise"] if new_q else None,
                         "rank_change_rate": float(np.mean(changes)) if changes else None,
                         "recommend_half_weight": bool(old_q and new_q and old_q["noise"]["icc1"] - new_q["noise"]["icc1"] > .1)}
    neutral = [r for r in rows if r.get("role") == "neutral"]
    pitches = [r for r in rows if r.get("role") in {"paraphrase", "independent"}]
    neutral_deltas = {key: float(np.mean([r["quantities"][key] for r in pitches if key in r["quantities"]]) -
                               np.mean([r["quantities"][key] for r in neutral if key in r["quantities"]]))
                      for key in set.intersection(*(set(r["quantities"]) for r in rows))} if neutral and pitches else {}
    return {"retest_relative_l2": l2, "retest_sanity": all(v <= .01 for v in l2) if l2 else None,
            "structure": structure_deltas, "roi_proxy_correlations": correlations, "shadow_comparison": shadow_report,
            "neutral_rows": len(neutral), "pitch_minus_neutral": neutral_deltas}


def analyze(args):
    metrics = json.loads((args.run / "metrics.json").read_text(encoding="utf-8"))
    if not metrics["complete"]:
        raise ValueError("Cannot analyze an incomplete scoring run")
    rows = [json.loads(line) for line in (args.run / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if not r["excluded"]]
    prov = metrics["provenance"]
    assets = load_assets()
    for row in rows:
        item = (args.run / row["prediction_file"]).resolve()
        if item.parent != args.run.resolve() or sha256(item) != row["prediction_sha256"]:
            raise ValueError("Stored prediction hash/path mismatch")
        if assets is not None:
            pred = np.load(item, allow_pickle=False)
            check_prediction(row["text"], pred)
            from tribe_service import engine, roi_features
            row["roi_summary"] = roi_features.roi_summary(pred)
            # Offline replay has no loaded model; retain the recorded encoder for scale lookup.
            with patch.object(engine, "_text_feature_metadata", return_value={
                "text_feature_model": prov["encoder"],
                "expected_text_feature_model": prov.get("runtime", {}).get("expected_text_feature_model", engine.TRIBE_CHECKPOINT_TEXT_MODEL),
                "text_feature_compatible": prov["text_feature_compatible"],
            }):
                measured = measure(row["text"], pred, row.get("channel", "general"), assets,
                                   prov["text_feature_compatible"])
            row["quantities"].update({k: v for k, v in measured["quantities"].items()
                                      if k.startswith("roi:") or k == "evidence_score_with_roi"})
            row["parcels"] = measured["parcels"]
            row["secondary_window"] = secondary_window(row, pred, assets)
    family = metrics["family"]
    report = {"family": family, "encoder": prov["encoder"], "date": metrics["date"],
              "provenance": prov, "excluded_rows": metrics["excluded_rows"], "n_rows": len(rows),
              "atlas_sha256": assets[4] if assets else prov["atlas_sha256"],
              "analysis_harness_sha256": sha256(Path(__file__))}
    if family == "stage0":
        quantities = {k: reference_quantity(rows, k) for k in sorted({k for r in rows for k in r["quantities"]})}
        quantities = {k: v for k, v in quantities.items() if v is not None}
        report.update(stage0_report(rows, args.run))
        reference = {"schema_version": 1, "kind": "roi_reference", "encoder": prov["encoder"],
                     "event_version": prov["event_version"], "text_input_mode": prov["text_input_mode"],
                     "atlas_sha256": report["atlas_sha256"], "harness_git_sha": prov["git_sha"],
                     "tribev2_commit": prov["tribev2_commit"], "hf_snapshot": prov["hf_snapshot"],
                     "analysis_harness_sha256": report["analysis_harness_sha256"],
                     "label": LABEL, "quantities": quantities, "checks": {}, "policy": POLICY.copy()}
        if assets:
            from tribe_service import roi_features
            from tribe_service.persuasion_features import ROI_PRIOR
            calibration_rows = []
            for i, row in enumerate(rows):
                if row.get("role") not in {"paraphrase", "independent"}:
                    continue
                z = roi_features.reference_z(row["roi_summary"], reference, row["lexical"]["words"])
                if ROI_PRIOR.keys() <= z.keys():
                    calibration_rows.append({"case": i, "roi_summary": {"asset_sha256": assets[4],
                        "rois": {key: {"z": value} for key, value in z.items()}}})
            try:
                reference["unit_calibration"] = {**roi_features.unit_calibration(
                    calibration_rows, prov["encoder"], f"stage0:{metrics['date']}:{prov['git_sha']}"),
                    "basis": "length_model_z"}
            except ValueError:
                report["unit_calibration_reason"] = "insufficient_or_degenerate_reference"
            reference["checks"] = {k: {"family": None, "status": "not_run", "d_z": None, "p_holm": None,
                                      "parcel_rank": None, "date": None, "weight_basis": "literature_prior",
                                      "weight_multiplier": multiplier("not_run", quantities.get(f"roi:{k}", {}).get("flags", []))}
                                   for group in assets[2:4] for k in group}
        report["reference"] = reference
    else:
        reference = json.loads(args.reference.read_text(encoding="utf-8")) if args.reference else None
        if reference and any(reference.get(k) != report.get(k, prov.get(k)) for k in ("encoder", "event_version", "text_input_mode", "atlas_sha256")):
            raise ValueError("Reference scale does not match the run provenance")
        control = json.loads(args.positive_control.read_text(encoding="utf-8")) if args.positive_control else None
        if control and (control["family"] != "S2" or control["encoder"] != prov["encoder"]
                        or control["atlas_sha256"] != report["atlas_sha256"]
                        or any(control.get("provenance", {}).get(k) != prov.get(k) for k in ("event_version", "text_input_mode"))):
            raise ValueError("Positive control must be S2 with matching encoder and atlas")
        report.update(analyze_suite(rows, metrics["sidecar"], reference, control.get("positive_control_passed") if control else None))
        report["secondary_windows"] = [{"pair_id": r["pair_id"], "condition": r["condition"], **r["secondary_window"]}
                                       for r in rows if r.get("secondary_window")]
        for check in report["checks"].values():
            check["date"] = metrics["date"]
        if reference:
            reference["checks"].update(report["checks"])
            report["reference"] = reference
    write_json(args.output or args.run / "analysis.json", report)
    if args.reference_out:
        if "reference" not in report:
            raise ValueError("A matching Stage 0 reference is required to export scale/checks")
        write_json(args.reference_out, report["reference"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gpu = commands.add_parser("score")
    gpu.add_argument("--family", choices=["stage0", "S2", "S4", "S1", "S3"], required=True)
    gpu.add_argument("--git-sha", required=True)
    gpu.add_argument("--repo", type=Path, default=ROOT)
    gpu.add_argument("--out", type=Path, required=True)
    cpu = commands.add_parser("analyze")
    cpu.add_argument("--run", type=Path, required=True)
    cpu.add_argument("--reference", type=Path)
    cpu.add_argument("--positive-control", type=Path)
    cpu.add_argument("--reference-out", type=Path)
    cpu.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        (score if args.command == "score" else analyze)(args)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"WS08: {exc}\n")


if __name__ == "__main__":
    main()
