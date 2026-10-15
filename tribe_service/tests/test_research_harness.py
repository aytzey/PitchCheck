import json
from collections import Counter
from pathlib import Path
import random
import re
from types import SimpleNamespace, ModuleType
import sys

import numpy as np
import pytest

from scripts.research import run_contrasts as harness
from tribe_service.tests.fixtures.ws08_synthetic import contrasts, reference


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)]


def pronouns(text):
    return Counter(re.findall(r"\b(?:i|me|my|mine|myself|we|us|our|ours|ourselves|you|your|yours|"
                              r"yourself|yourselves|he|him|his|himself|she|her|hers|herself|"
                              r"it|its|itself|they|them|their|theirs|themselves)\b", text.casefold()))


@pytest.mark.parametrize("family,count", [("stage0", 405), ("S2", 200), ("S4", 100), ("S1", 80), ("S3", 80)])
def test_preregistrations_and_fixture_contracts(family, count):
    sidecar, rows, _ = harness.load_fixtures(family)
    assert len(rows) == count
    assert sidecar["run_status"] == "not_run"
    assert sidecar["manipulation_check"]["blind_classifier"] == "pending"
    assert sidecar["manipulation_check"]["owner_sample_fraction"] == 0
    assert all(v <= 1 for v in sidecar["weight_policy"].values())
    assert sidecar["generation"]["model"] and sidecar["generation"]["prompt"]
    assert sidecar["generation"]["author"] == "OpenAI Codex gpt-6.1-sol, hand-written per item"
    assert sidecar["generation"]["temperature"] is None


def test_stage0_families_keep_facts_pronouns_cta_and_token_multisets():
    from tribe_service import engine, llm_layer
    _, rows, _ = harness.load_fixtures("stage0")
    originals = {r["pair_id"]: r for r in rows if r["condition"] == "p0"}
    assert len(originals) == 30
    channels = {"email", "linkedin", "cold-call-script", "landing-page", "ad-copy"}
    assert Counter(r["channel"] for r in originals.values()) == Counter({c: 6 for c in channels})
    for r in rows:
        if r.get("role") == "paraphrase":
            original = originals[r["pair_id"]]["text"]
            assert llm_layer._refine_concrete_details(r["text"]) == llm_layer._refine_concrete_details(original)
            assert sentences(r["text"])[-1] == sentences(original)[-1]
            assert pronouns(r["text"]) == pronouns(original)
        if r["condition"] == "scrambled":
            mate = next(x for x in rows if x["pair_id"] == r["pair_id"] and x["condition"] == "nopunct")
            assert sorted(engine._WORD_RE.findall(r["text"])) == sorted(engine._WORD_RE.findall(mate["text"]))
            index = int(r["pair_id"].removeprefix("structure"))
            source = originals[f"para{index:02d}"]["text"]
            assert engine._WORD_RE.findall(mate["text"]) == engine._WORD_RE.findall(source)
            shuffled = engine._WORD_RE.findall(source)
            random.Random(index).shuffle(shuffled)
            assert engine._WORD_RE.findall(r["text"]) == shuffled
        if r.get("role") == "retest":
            index = int(r["pair_id"].removeprefix("retest"))
            assert r["text"] == originals[f"para{index:02d}"]["text"]
    assert Counter(r["role"] for r in rows) == {
        "paraphrase": 150, "independent": 150, "neutral": 30, "structure": 60, "retest": 15,
    }
    assert Counter(r["pair_id"] for r in rows if r.get("role") == "retest") == {
        f"retest{i}": 3 for i in range(5)
    }
    for channel in channels:
        counts = [len(engine._WORD_RE.findall(r["text"])) for r in originals.values() if r["channel"] == channel]
        assert Counter(0 if n <= 60 else 1 if n <= 150 else 2 for n in counts) == {0: 2, 1: 2, 2: 2}
        independent = [r for r in rows if r.get("role") == "independent" and r["channel"] == channel]
        assert len(independent) == 30
        assert {0 if len(engine._WORD_RE.findall(r["text"])) <= 60 else
                1 if len(engine._WORD_RE.findall(r["text"])) <= 150 else 2 for r in independent} == {0, 1, 2}


@pytest.mark.parametrize("family,conditions", [("S1", ["T", "U_you", "U"]),
                                               ("S2", ["Fact", "How", "Whygain", "Whyloss"])])
def test_manipulations_keep_shared_sentences_and_cta(family, conditions):
    from tribe_service import engine
    _, rows, _ = harness.load_fixtures(family)
    if family == "S2":
        assert all(45 <= len(engine._WORD_RE.findall(r["text"])) <= 60 for r in rows)
    for pair in {r["pair_id"] for r in rows}:
        group = {r["condition"]: r["text"] for r in rows if r["pair_id"] == pair}
        texts = [sentences(group[c]) for c in conditions]
        assert len({len(t) for t in texts}) == 1
        changed = [i for i in range(len(texts[0])) if len({t[i] for t in texts}) > 1]
        assert len(changed) == 1 and changed[0] != len(texts[0]) - 1
        if family == "S1":
            assert all(len(sentences(t)) == 5 for t in group.values())
            a, b = group["T"], group["U_you"]
        elif "Fact_you" in group:
            a, b = group["Fact_you"], group["Whygain"]
            assert sentences(group["Fact_you"])[-1] == texts[0][-1]
        else:
            assert not re.search(r"\b(?:skin|body|hands?|faces?|arms?|ears?|neck|nose|lips?|legs?|"
                                 r"feet|foot|eyes?|fingers?|palms?|shoulders?|cheeks?|forehead|"
                                 r"scalp|wrists?|ankles?|elbows?|knees?|thumbs?|backs?|chests?|"
                                 r"toes?|heels?|fingertips?)\b", group["How_nobody"], re.I)
            assert sentences(group["How_nobody"])[-1] == texts[0][-1]
            continue
        assert Counter(re.findall(r"\b(?:you|your|yours|yourself|yourselves)\b", a.casefold())) == Counter(
            re.findall(r"\b(?:you|your|yours|yourself|yourselves)\b", b.casefold()))


def test_hash_mismatch_and_overlong_text_rejected(tmp_path):
    sidecar, rows, _ = harness.load_fixtures("S1")
    path = tmp_path / "S1.jsonl"
    path.write_text(json.dumps(rows[0]) + "\n")
    (tmp_path / "S1.json").write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="hash"):
        harness.load_fixtures("S1", tmp_path)
    rows[0]["text"] = "word " * 241
    path.write_text(json.dumps(rows[0]) + "\n")
    sidecar["stimuli_sha256"] = harness.sha256(path)
    (tmp_path / "S1.json").write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="240"):
        harness.load_fixtures("S1", tmp_path)


@pytest.mark.parametrize("role", [None, "independent"])
@pytest.mark.parametrize("duplicate", ["sentence", "phrase"])
def test_fixture_preflight_rejects_cross_pair_templates(tmp_path, role, duplicate):
    # These are excerpts of the rejected WS08 templates, with distinct topic nouns.
    a = "Your team can use this workspace for invoice review while keeping the current process visible."
    b = "Your team can use this workspace for shift planning while keeping the current process visible."
    if duplicate == "sentence":
        a += " A list records the current owner and the next action for each item."
        b += " A list records the current owner and the next action for each item."
    else:
        a += " A list records the current owner and the next action for each item."
        b += " A list records the current owner and the next action beside each request."
    family = "stage0" if role else "S1"
    rows = [{"family": family, "pair_id": pair, "condition": "original", "text": text, "role": role}
            for pair, text in [("one", a), ("two", b)]]
    path = tmp_path / f"{family}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    harness.write_json(tmp_path / f"{family}.json", {
        "schema_version": 1, "family": family, "stimuli_file": path.name,
        "stimuli_sha256": harness.sha256(path), "contrasts": [],
    })
    with pytest.raises(ValueError, match="diversity.*(?:sentence|8-gram)"):
        harness.load_fixtures(family, tmp_path)


def test_fixture_diversity_allows_matching_inside_pair(tmp_path):
    text = "A rural repair cooperative offers blade sharpening before the spring planting season."
    rows = [{"family": "S1", "pair_id": "same", "condition": c, "text": text} for c in ("a", "b")]
    path = tmp_path / "S1.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    harness.write_json(tmp_path / "S1.json", {
        "schema_version": 1, "family": "S1", "stimuli_file": path.name,
        "stimuli_sha256": harness.sha256(path), "contrasts": [],
    })
    assert len(harness.load_fixtures("S1", tmp_path)[1]) == 2


@pytest.mark.parametrize("mock,cache,mode", [(True, 0, "direct"), (False, 1, "direct"), (False, 0, "audio")])
def test_gpu_preflight_rejects_unsafe_runtime(mock, cache, mode):
    with pytest.raises(ValueError):
        harness.check_runtime(SimpleNamespace(TRIBE_ALLOW_MOCK=mock, TRIBE_PREDICTION_CACHE_SIZE=cache, TRIBE_TEXT_INPUT_MODE=mode))


def test_prediction_contract():
    harness.check_prediction("one two three", np.ones((2, 20484)))
    for pred in (np.ones((2, 20)), np.ones((3, 20484)), np.full((2, 20484), np.nan)):
        with pytest.raises(ValueError):
            harness.check_prediction("one two three", pred)


def test_unregistered_encoder_rejected_before_weights_load(monkeypatch, tmp_path):
    from tribe_service import engine
    monkeypatch.setattr(engine, "TRIBE_ALLOW_MOCK", False)
    monkeypatch.setattr(engine, "TRIBE_PREDICTION_CACHE_SIZE", 0)
    monkeypatch.setattr(engine, "TRIBE_TEXT_MODEL", "unregistered/encoder")
    with pytest.raises(ValueError, match="not pre-registered"):
        harness.score(SimpleNamespace(family="stage0", repo=tmp_path, git_sha="abc", out=tmp_path))


def test_single_roi_loader_missing_and_synthetic_signed_readout(monkeypatch):
    real_import = harness.importlib.import_module
    def missing(name):
        if name == "tribe_service.roi_features":
            raise ModuleNotFoundError(name=name)
        return real_import(name)
    monkeypatch.setattr(harness.importlib, "import_module", missing)
    assert harness.load_assets() is None
    assert harness.roi_readout(np.ones((1, 2)), None) == ({}, None)
    monkeypatch.setattr(harness.importlib, "import_module", real_import)
    module = ModuleType("tribe_service.roi_features")
    labels = np.resize(np.arange(401), 20484)
    indices = np.flatnonzero(labels == 1)
    assets = (labels, np.array([str(i) for i in range(401)]), {"target": indices}, {"location": indices}, "synthetic")
    module.load_roi_assets = lambda: assets
    module.roi_summary = lambda p: {group: {key: {"minus_cortex": float(p[:, idx].mean() - p[:, labels > 0].mean())}
                                          for key, idx in mapping.items()} for group, mapping in zip(("rois", "locations"), assets[2:4])}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    import tribe_service
    monkeypatch.setattr(tribe_service, "roi_features", module, raising=False)
    assert harness.load_assets() is assets
    pred = np.zeros((3, 20484)); pred[:, indices] = -2
    q, parcels = harness.roi_readout(pred, assets)
    assert q["roi:target"] < 0 and q["roi:location"] < 0
    assert len(parcels) == 400 and parcels[0] < 0


def test_status_and_policy_never_increase_weights():
    assert harness.status([4]*20, .01, .01, 2, .95, "greater") == "pass"
    assert harness.status([4]*20, .01, .01, 2, .50, "greater") == "pass_broad"
    assert harness.status([-4]*20, 1, .01, 2, .95, "greater") == "reversed"
    assert harness.status([1]*20, .01, .01, 2, .95, "greater") == "null"
    assert harness.multiplier("pass", ["low_stability"]) == .5
    assert harness.multiplier("reversed", ["low_stability"]) == 0
    assert harness.multiplier("reversed", ["low_stability"], informative=False) == 1


@pytest.mark.parametrize("effect,expected", [(4, "pass"), (-4, "reversed"), (0, "null")])
def test_suite_paired_tests_and_positive_control_protection(effect, expected):
    rows, sidecar = contrasts(effect)
    result = harness.analyze_suite(rows, sidecar, reference(), True, B=999)
    assert result["checks"]["target"]["status"] == expected
    assert all(t["p_holm"] >= t["p"] for t in result["tests"])
    result = harness.analyze_suite(rows, sidecar, reference(), False, B=999)
    assert all(t["status"] == "uninformative" and t["weight_multiplier"] == 1 for t in result["tests"])
    result = harness.analyze_suite(rows, sidecar, None, True, B=999)
    assert all(t["status"] == "uninformative" for t in result["tests"])
    sidecar["manipulation_check"]["blind_classifier"] = "failed"
    result = harness.analyze_suite(rows, sidecar, reference(), True, B=999)
    assert all(t["status"] == "uninformative" and t["weight_multiplier"] == 1 for t in result["tests"])


def test_missing_roi_and_excluded_pairs_cannot_be_called_pass():
    rows, sidecar = contrasts()
    assert harness.analyze_suite(rows[:30], sidecar, reference(), True)["checks"]["target"]["status"] == "not_run"
    for r in rows:
        r["quantities"] = {}
    assert harness.analyze_suite(rows, sidecar, reference(), True)["checks"]["target"]["status"] == "not_run"


def test_compound_why_condition_is_paired_mean():
    rows = [{"pair_id": "one", "condition": c, "quantities": {"q": v}, "parcels": [v]*400,
             "lexical": {"motor_verbs": v}} for c, v in [("How", 4), ("Whygain", 1), ("Whyloss", 3)]]
    a, b = harness.paired_rows(rows, {"conditions": ["How", ["Whygain", "Whyloss"]]})[0]
    assert a["quantities"]["q"] - b["quantities"]["q"] == 2


def test_stage0_scale_uses_pitches_only_and_pooled_length_model():
    rng = np.random.default_rng(2)
    rows = []
    for i in range(30):
        effect = rng.normal(0, 3)
        for j in range(5):
            words = (40, 100, 200)[i % 3] + j
            rows.append({"pair_id": str(i), "condition": f"p{j}", "role": "paraphrase",
                         "quantities": {"q": 2 + .8*np.log(words) + effect + rng.normal(0, .2)},
                         "lexical": {"words": words, "sentences": 4, "mean_word_length": 5, "global_mean_abs": 1}})
    rows.append({"role": "neutral", "quantities": {"q": 1e6}})
    q = harness.reference_quantity(rows, "q", B=100)
    assert q["n_reference"] == 150 and q["n_paraphrase_families"] == 30
    assert q["noise"]["icc1"] > .9
    assert q["noise"]["min_detectable_delta"] == pytest.approx(1.96*np.sqrt(2)*q["noise"]["sigma_within"])
    assert set(q["resid_q"]) == {"p05", "p25", "p50", "p75", "p95"}
    assert "joint" in q["confound_r2"]


def test_preregistration_must_be_committed_before_scoring(monkeypatch, tmp_path):
    path = tmp_path / "S1.json"; path.write_text("{}")
    outputs = iter(["abc\n", "?? S1.json\n"])
    monkeypatch.setattr(harness.subprocess, "check_output", lambda *a, **kw: next(outputs))
    with pytest.raises(ValueError, match="clean checkout"):
        harness.committed_preregistration([path], tmp_path, "abc")


def test_gpu_score_mocked_io_excludes_fallback_and_retry(monkeypatch, tmp_path):
    from tribe_service import engine
    sidecar, _, _ = harness.load_fixtures("stage0")
    sidecar["manipulation_check"].update(blind_classifier="passed", owner_sample_fraction=.2,
                                       classifier_model="synthetic-test", owner_review="passed")
    rows = [{"pair_id": str(i), "condition": f"r{i}", "text": "one two three", "role": "retest"} for i in range(3)]
    monkeypatch.setattr(harness, "load_fixtures", lambda *a: (sidecar, rows, []))
    monkeypatch.setattr(harness, "committed_preregistration", lambda *a: "committed-before-run")
    monkeypatch.setattr(engine, "TRIBE_ALLOW_MOCK", False)
    monkeypatch.setattr(engine, "TRIBE_PREDICTION_CACHE_SIZE", 0)
    monkeypatch.setattr(engine, "load_runtime_models", lambda: None)
    monkeypatch.setattr(engine, "unload_model", lambda: None)
    monkeypatch.setattr(engine, "runtime_config", lambda: {"text_feature_compatible": True})
    monkeypatch.setattr(engine, "score_text", lambda text: np.ones((2, 20484)))
    runtimes = iter([{}, {"fallback_used": True}, {"retry_index": 1}])
    monkeypatch.setattr(engine, "last_score_metrics", lambda: next(runtimes))
    monkeypatch.setattr(harness, "load_assets", lambda: None)
    monkeypatch.setattr(harness, "provenance", lambda *a: {"encoder": engine.TRIBE_TEXT_MODEL})
    monkeypatch.setattr(harness, "measure", lambda *a: {"quantities": {"evidence_score": 50}})
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda seed: None))
    harness.score(SimpleNamespace(family="stage0", repo=tmp_path, git_sha="abc", out=tmp_path / "run"))
    metrics = json.loads((tmp_path / "run/metrics.json").read_text())
    result = [json.loads(line) for line in (tmp_path / "run/rows.jsonl").read_text().splitlines()]
    assert metrics["complete"] and metrics["excluded_rows"] == 2
    assert [r["excluded"] for r in result] == [False, True, True]
    assert all(np.load(tmp_path / "run" / r["prediction_file"]).dtype == np.float16 for r in result)


def test_cpu_analysis_exports_reference_consumable_by_product(monkeypatch, tmp_path):
    from tribe_service import engine
    rng = np.random.default_rng(2)
    rows = []
    run = tmp_path / "run"; run.mkdir()
    pred = run / "pred.npy"; np.save(pred, np.zeros((2, 20484), np.float16))
    for i in range(5):
        for j in range(5):
            rows.append({"pair_id": str(i), "condition": f"p{j}", "role": "paraphrase", "text": "one two three",
                         "quantities": {"evidence_score": 50 + i + rng.normal(0, .1)},
                         "lexical": {"words": 3, "sentences": 1, "mean_word_length": 3, "global_mean_abs": 1},
                         "excluded": False, "prediction_file": "pred.npy", "prediction_sha256": harness.sha256(pred)})
    prov = {"encoder": "meta-llama/Llama-3.2-3B", "event_version": "direct_v1", "text_input_mode": "direct",
            "atlas_sha256": None, "git_sha": "synthetic", "tribev2_commit": "synthetic", "hf_snapshot": "synthetic"}
    harness.write_json(run / "metrics.json", {"complete": True, "family": "stage0", "date": "2026-10-16",
                                              "provenance": prov, "excluded_rows": 0})
    (run / "rows.jsonl").write_text(''.join(json.dumps(row)+'\n' for row in rows))
    monkeypatch.setattr(harness, "load_assets", lambda: None)
    monkeypatch.setattr(engine, "__file__", str(tmp_path / "engine.py"))
    destination = tmp_path / "roi_assets/roi_reference_meta-llama__Llama-3.2-3B.json"
    harness.analyze(SimpleNamespace(run=run, output=None, reference_out=destination))
    report = json.loads((run / "analysis.json").read_text())
    engine.roi_reference.cache_clear()
    try:
        data = engine.roi_reference(prov["encoder"])
        assert data == report["reference"]
        assert data["quantities"]["evidence_score"]["noise"]["min_detectable_delta"] > 0
        assert report["shadow_comparison"]["status"] == "not_run"
    finally:
        engine.roi_reference.cache_clear()
    pred.write_bytes(b'changed')
    with pytest.raises(ValueError, match="hash"):
        harness.analyze(SimpleNamespace(run=run, output=None, reference_out=destination))


def test_lexical_condition_confound_is_not_reported_as_adjusted_signal():
    rows, sidecar = contrasts()
    for r in rows:
        r["lexical"]["motor_verbs"] = 3 if r["condition"] == "a" else 0
    result = harness.analyze_suite(rows, sidecar, reference(), True, B=99)
    adjustment = result["tests"][0]["lexical_residualized"]
    assert adjustment["identifiable"] is False and adjustment["adjusted_delta"] is None


def test_cost_sentence_position_and_secondary_window():
    from tribe_service import engine
    _, rows, _ = harness.load_fixtures("S3")
    for pair_id in {r["pair_id"] for r in rows}:
        group = {r["condition"]: r for r in rows if r["pair_id"] == pair_id}
        assert len({r["cost_word_index"] for r in group.values()}) == 1
        for r in group.values():
            matches = list(engine._WORD_RE.finditer(r["text"]))
            assert r["cost_word_index"] == len(engine._WORD_RE.findall(sentences(r["text"])[0]))
            assert r["text"][matches[r["cost_word_index"]].start():].startswith(sentences(r["text"])[1])
        for a, b in [("reasonable", "excessive"), ("number_low", "number_high")]:
            assert re.sub(r"\d+", "#", group[a]["text"]) == re.sub(r"\d+", "#", group[b]["text"])
        for cost, number in [("reasonable", "number_low"), ("excessive", "number_high")]:
            assert re.findall(r"\d+", group[cost]["text"]) == re.findall(r"\d+", group[number]["text"])
            for i, (a, b) in enumerate(zip(sentences(group[cost]["text"]), sentences(group[number]["text"]))):
                if i != 1:
                    assert a == b
    row = rows[0]
    n = len(engine._WORD_RE.findall(row["text"]))
    import math
    pred = np.ones((math.ceil(.4*(n-1)+.32), 20484))
    window = harness.secondary_window(row, pred, None)
    assert window["stop_segment"] - window["start_segment"] >= 4
    assert window["start_segment"] > 0 and window["stop_segment"] == len(pred)


def test_measurement_uses_production_anchor_without_atlas(monkeypatch):
    from tribe_service import app, engine
    text = "one two three"
    pred = np.random.default_rng(3).normal(size=(2, 20484)).astype(np.float32)
    measured = harness.measure(text, pred, "email", None)
    assert measured["quantities"]["evidence_score"] == pytest.approx(app._measure_prediction(text, pred, "", "email")["neural_score"], abs=.00051)
    assert measured["parcels"] is None
    assert not any(k.startswith("roi:") for k in measured["quantities"])


def test_unknown_co_primary_cannot_be_aggregated_to_pass():
    rows, sidecar = contrasts()
    sidecar["contrasts"].append({"name": "missing_co_primary", "conditions": ["a", "missing"],
                                 "rois": ["target"], "alternative": "greater"})
    result = harness.analyze_suite(rows, sidecar, reference(), True, B=99)
    assert result["checks"]["target"]["status"] == "not_run"
    assert result["positive_control_passed"] is False
