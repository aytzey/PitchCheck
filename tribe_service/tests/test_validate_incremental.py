"""WS10 synthetic checks run offline, without a GPU, credentials or API calls."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v = load("ws10_validate", ROOT / "scripts/validate_incremental.py")
s = load("ws10_scorer", ROOT / "scripts/score_validation_corpus.py")
d = load("ws10_prepare", ROOT / "scripts/prepare_validation_data.py")
fixture = load("ws10_synthetic", Path(__file__).parent / "fixtures/ws10_synthetic.py")


@pytest.mark.parametrize("kind", ["planted", "shuffled", "length", "correlated_null"])
def test_residual_permutation_synthetic(kind):
    base, candidate, y, w, g = fixture.make(effect=1.5 if kind in {"planted", "shuffled", "length"} else 0.)
    if kind == "shuffled":
        candidate = np.random.default_rng(10).permutation(candidate)
    elif kind == "length":
        candidate = base[:, :1].copy()
    folds = v.folds_for(g)
    result, _, _ = v.perm_test(base, candidate, y, w, g, folds, v.pairs(g, folds), B=199)
    if kind == "planted":
        assert result["delta"] > 0 and result["p"] < .05
    else:
        assert result["p"] > .05
        assert abs(result["delta"]) < .08


def test_folds_standardization_pairs_bootstrap_and_holm():
    base, candidate, y, w, g = fixture.make()
    folds = v.folds_for(g)
    assert all(len(set(folds[g == group])) == 1 for group in np.unique(g))
    P = v.pairs(g, folds)
    assert np.all(g[P[:, 0]] == g[P[:, 1]])
    assert np.all(folds[P[:, 0]] == folds[P[:, 1]])
    assert v.concordance(np.zeros(len(y)), y, P) == .5
    with pytest.raises(ValueError, match="No comparable"):
        v.concordance(y, np.ones(len(y)), P)
    assert v.holm([.01, .04, .03]) == [.03, .06, .06]
    result, old, new = v.perm_test(base, candidate, y, w, g, folds, P, B=1)
    assert result["ci"] == v.bootstrap_delta(old, new, y, g, P)
    assert np.allclose(v.demean(base, g, w)[g == 0].T @ w[g == 0], 0)


@pytest.mark.parametrize("groups,arms,sends", [(15, 4, 300), (8, 12, 300), (40, 4, 99)])
def test_minimum_n_is_conjunction(groups, arms, sends):
    rows = fixture.records(groups=groups, arms=arms)
    for r in rows:
        r["n_sends"] = sends
    assert v.decision(rows, .2, [.1, .3], .001, True, candidate="x=1", flag="flag") == "underpowered"


def test_decisions_never_change_product_or_tune_production_data():
    rows = fixture.records()
    assert v.decision(rows, .2, [.1, .3], .001, True, candidate="x=1") == "keep"
    for r in rows:
        r.update(synthetic=False, data_source="randomized_campaign")
    assert v.decision(rows, .2, [.1, .3], .001, True, candidate="x=1") == "tune:x=1"
    assert v.decision(rows, .2, [.1, .3], .001, False, candidate="x=1") == "keep"
    assert v.decision(rows, -.2, [-.3, -.1], flag="flag") == "rollback_alarm:flag"


@pytest.mark.parametrize("change", ["missing", "too_many", "nan", "unconsented", "bool", "negative", "score", "mock"])
def test_loader_rejects_invalid_data(change):
    row = fixture.records(groups=2)[0]
    if change == "missing":
        del row["group_id"]
    elif change == "too_many":
        row["positives"] = 301
    elif change == "nan":
        row["evidence_score"] = float("nan")
    elif change == "unconsented":
        row["text"] = "Private synthetic text"
    elif change == "bool":
        row["n_sends"] = True
    elif change == "negative":
        row["n_sends"] = -1
    elif change == "score":
        row["context_fit_score_blind"] = 101
    else:
        row["source"] = "refine_candidate"
    with pytest.raises(ValueError):
        v.validate_row(row)


def test_eligibility_strata_dedup_edits_and_refine_separation():
    rows = fixture.records(groups=4)
    rows[0]["mode"] = "mock"
    rows[1]["voxel_count"] = 12
    rows[2]["edited_after_score"] = True
    rows[3]["channel"] = "general"
    rows[4]["personal_invitation"] = True
    rows[5]["context_fit_missing"] = ["proof_credibility"]
    rows[6]["raw_llm_score_blind"] = None
    rows[7]["source"] = "refine_candidate"
    rows[8]["text_feature_model"] = "other-encoder"
    rows[9]["language"] = "tr"
    rows[10]["methodology_version"] = "other-method"
    rows[11]["llm_model"] = "other-LLM"
    extra = {**rows[12], "raw_llm_score_blind": 100., "scored_at": "2026-02-01T00:00:00+00:00"}
    strata, excluded = v.eligible_strata([*rows, extra])
    assert len(strata) == 5
    assert sum(excluded.values()) == 8
    kept = next(r for rs in strata.values() for r in rs if r["pitch_id"] == extra["pitch_id"])
    assert kept["raw_llm_score_blind"] == (rows[12]["raw_llm_score_blind"] + 100) / 2
    assert kept["scored_at"] == extra["scored_at"]


def test_join_aggregates_recipients_but_preserves_campaigns_and_rejects_ambiguous_persona():
    row = fixture.records(groups=2)[0]
    outcomes = [{k: row[k] for k in ("pitch_id", "group_id", "channel", "outcome_family", "outcome")} | {"n_sends": 1, "positives": k} for k in (0, 1)]
    joined = v.join_outcomes([row], outcomes)
    assert len(joined) == 1 and joined[0]["n_sends"] == 2 and joined[0]["positives"] == 1
    assert joined[0]["outcome"] == .5
    with pytest.raises(ValueError, match="Ambiguous"):
        v.join_outcomes([row, {**row, "persona_id": "other"}], outcomes)


def test_blind_prompt_preserves_doctrine_annex_and_exact_production_rubric():
    from tribe_service import llm_layer as llm
    lower = s.BLIND_SYSTEM_PROMPT.lower()
    assert not any(marker in lower for marker in ("tribe", "calibration", "neural prior", "self_value", "reward_affect", "temporal trace"))
    assert llm.PERSUASION_DOCTRINE in s.BLIND_SYSTEM_PROMPT
    assert llm.PERSUASION_RESEARCH_ANNEX in s.BLIND_SYSTEM_PROMPT
    assert s.BLIND_CONTEXT_SCHEMA in llm._build_user_prompt("Own synthetic pitch", "A careful reader", "email", {})
    prompt = s.blind_user_prompt("A private synthetic pitch", "A careful reader", "email")
    assert "Calibration Prior" not in prompt and "neural_signals" not in prompt


def test_blind_samples_temperature_missing_facets_and_scrambled_twin():
    from tribe_service import llm_layer as llm
    calls = []
    def call(prompt, **kwargs):
        calls.append(kwargs)
        return {"persuasion_score": 80, "context_fit": {key: {"score": 70} for key in llm.CONTEXT_FIT_KEYS}}
    result = s.blind_scores("Own synthetic pitch", "A careful reader", "email", call=call)
    assert len(calls) == 3 and all(c["temperature"] == .2 and c["system_prompt"] == s.BLIND_SYSTEM_PROMPT for c in calls)
    assert result["raw_llm_score_blind"] == 80 and result["context_fit_missing"] == []
    bad = s.blind_scores("Own synthetic pitch", "A careful reader", "email", call=lambda *a, **k: {"persuasion_score": 80, "context_fit": {}})
    assert set(bad["context_fit_missing"]) == set(llm.CONTEXT_FIT_KEYS)
    assert bad["context_fit_score_blind"] is None
    text = "one two three four five six seven"
    assert s.scrambled_text(text, "id") == s.scrambled_text(text, "id")
    assert sorted(s.scrambled_text(text, "id").split()) == sorted(text.split())


def test_compare_200_synthetic_rows_parity_and_refine_counterfactual():
    rows = fixture.comparison_records()
    rows.append({"source": "refine_candidate", "selected": True, "selected_id": "c1", "selected_id_semantic_only": "c2"})
    result = v.compare(rows)
    assert result["parity"] == "pass" and result["n"] == 200
    assert 0 < result["mean_absolute_delta"] <= 6 and result["refine_winner_change_rate"] == 1
    rows[0]["final_score"] += .1
    with pytest.raises(ValueError, match="parity"):
        v.compare(rows)


def test_prior_replay_budget_individual_flags_and_alarm():
    from tribe_service.llm_layer import _blend_final
    rows = fixture.records()
    for row in rows:
        e = row["blend_inputs"]["evidence"]
        row["prior_inputs"] = {"evidence_without_roi": 100 - e, "facet_without_boost": None,
                               "tailoring_points": 0., "specificity_points": 0.}
        row["blend_inputs"].update(raw_llm=None, facet=None)
        row["final_score"] = round(_blend_final(**row["blend_inputs"], priors=row["prior_inputs"])[0], 1)
        row.update(synthetic=False, data_source="randomized_campaign")
        # The active prior reverses an otherwise perfect within-group ordering.
        row["positives"] = int(round(300 * (100 - e) / 100))
    assert v.compare(rows)["mean_absolute_delta"] <= 6.0 + 1e-12
    assert v.replay(rows[0], disabled_flag="PITCHCHECK_ROI_PRIOR_WEIGHT")[0] == pytest.approx(100 - rows[0]["blend_inputs"]["evidence"])
    result = v.rollback_alarm(rows, "PITCHCHECK_ROI_PRIOR_WEIGHT", B=199)
    assert result["decision"] in {"keep", "rollback_alarm:PITCHCHECK_ROI_PRIOR_WEIGHT"}
    with pytest.raises(ValueError):
        v.replay({**rows[0], "prior_inputs": None}, disabled_flag="PITCHCHECK_ROI_PRIOR_WEIGHT")


def test_temporal_sweep_never_trains_on_latest_groups_or_tunes_synthetic():
    rows = fixture.records()
    train, test = v.temporal_split(rows)
    assert train.sum() == 120 and test.sum() == 40
    assert {r["group_id"] for r, m in zip(rows, train) if m}.isdisjoint({r["group_id"] for r, m in zip(rows, test) if m})
    result = v.sweep(rows, B=199)
    assert result["decision"] == "keep"
    assert any(c["constant"] == "CONTEXT_FIT_WEIGHTS" for c in result["candidates"])
    assert any(c["constant"] == "NEURO_AXIS_WEIGHTS" for c in result["candidates"])
    assert [c["value"] for c in result["candidates"] if c["constant"] == "SEMANTIC_BLEND_WEIGHT"] == [.35, .45, .55, .65, .75, .85]
    assert all("p_holm" in c for c in result["candidates"])


def test_upworthy_flags_image_control_merging_and_winner_sensitivity(monkeypatch):
    monkeypatch.setenv("PITCHCHECK_SCORE_LOG_KEY", "k" * 32)
    with (Path(__file__).parent / "fixtures/ws10_upworthy.csv").open() as f:
        source = list(csv.DictReader(f))
    rows, meta = d.prepare_upworthy(source)
    assert len(rows) == 2 and rows[0]["n_sends"] == 300 and rows[0]["positives"] == 30
    assert all(r["channel"] == "ad-copy" and r["calibration_basis"] == "proxy_upworthy" for r in rows)
    assert meta["exclusions"]["randomization_risk_tests"] == 2
    assert meta["exclusions"]["nonconstant_or_missing_eyecatcher_tests"] == 1
    sensitive, _ = d.prepare_upworthy(source, exclude_winners=True)
    assert sensitive == []  # Only one distinct non-winner title remains.
    with pytest.raises(ValueError, match="risk flag"):
        d.prepare_upworthy([{k: value for k, value in row.items() if k != "problem"} for row in source])


def test_campaign_assignment_is_stable_balanced_and_never_sends(monkeypatch):
    monkeypatch.setenv("PITCHCHECK_SCORE_LOG_KEY", "k" * 32)
    args = ("a" * 64, "group", "a", "b", {"a": "pitch-a", "b": "pitch-b"})
    assert d.assign_campaign(*args) == d.assign_campaign(*args)
    assignments = [d.assign_campaign(f"{i:064x}", *args[1:]) for i in range(1000)]
    assert .45 < np.mean([r["arm"] == "with_neural" for r in assignments]) < .55
    assert all("recipient_hash" not in row for row in assignments)


def test_panel_latin_square_scales_reliability_and_scope():
    stimuli = [{"group_id": f"g{g}", "pitch_id": f"p{g}-{i}"} for g in range(20) for i in range(4)]
    assignments = d.panel_assign(stimuli)
    counts = {}
    ratings = []
    for row in assignments:
        counts[row["pitch_id"]] = counts.get(row["pitch_id"], 0) + 1
        value = int(row["pitch_id"].split("-")[1]) + 1
        ratings.append({**row, "channel": "email", **dict.fromkeys(d.PANEL_SCALES, value)})
    assert set(counts.values()) == {15}
    rows, meta = d.panel_aggregate(ratings, panel_id="synthetic")
    assert len(rows) == 80 and all(r["n_raters"] == 15 for r in rows)
    assert meta["spearman_brown"] == pytest.approx(1.)
    assert meta["scope"] == v.PME_SCOPE
    with pytest.raises(ValueError, match="at most one"):
        d.panel_aggregate([*ratings, ratings[0]], panel_id="synthetic")


def test_report_wording_and_cli_checksums(tmp_path):
    assert "lever" not in v.REPORT_TEMPLATE.lower() and "drives" not in v.REPORT_TEMPLATE.lower()
    source = tmp_path / "synthetic.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in fixture.records()))
    output = tmp_path / "comparison"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/validate_incremental.py"), "compare", str(source), "--output", str(output), "--dataset-id", "synthetic-ci"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["dataset_id"] == "synthetic-ci" and report["strata"][0]["parity"] == "pass"
    for line in output.with_suffix(".sha256").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest


def test_randomized_policy_loss_emits_alarm_with_sufficient_n():
    rows = fixture.records(groups=40, arms=2)
    for i, row in enumerate(rows):
        row.update(data_source="randomized_campaign", synthetic=False,
                   arm="with_neural" if i % 2 == 0 else "semantic_only", positives=10 if i % 2 == 0 else 60)
    result = v.randomized_arm_alarm(rows, "PITCHCHECK_ROI_PRIOR_POINTS", B=199)
    assert result["ci"][1] < 0
    assert result["decision"] == "rollback_alarm:PITCHCHECK_ROI_PRIOR_POINTS"
    assert v.randomized_arm_alarm(rows[:60], "flag", B=199)["decision"] == "underpowered"


def test_selection_points_sweep_four_candidates_and_parity():
    rows = fixture.records()
    for row in rows:
        row.update(data_source="randomized_campaign", synthetic=True, roi_prior_points=.5,
                   selection_score_without_roi=50., roi_selection_unit=(row["evidence_score"] - 50) / 20)
        row["selection_score"] = 50 + row["roi_selection_unit"] * .5
    result = v.selection_sweep(rows, B=199)
    assert result["decision"] == "keep"
    assert [c["value"] for c in result["candidates"]] == [0, .25, .5, 1.]
    rows[0]["selection_score"] += 1
    with pytest.raises(ValueError, match="parity"):
        v.selection_sweep(rows, B=199)


def test_blind_join_keeps_product_formula_and_rejects_encoder_mismatch():
    rows = fixture.records(groups=2)
    blind = copy.deepcopy(rows)
    blind[0]["final_score"] = 0.
    joined = v.join_blind(rows, blind)
    assert joined[0]["final_score"] == rows[0]["final_score"]
    blind[0]["text_feature_model"] = "incompatible"
    with pytest.raises(ValueError, match="matching blind"):
        v.join_blind(rows, blind)


def test_increment_optional_roi_reverse_tests_pme_r2_and_sensitivities():
    rows = fixture.records(groups=4)
    for i, row in enumerate(rows):
        row.update(outcome_family="pme", outcome=1 + (i % 4), data_source="pme_panel", n_raters=15,
                   roi={"rois": {"mofc": {"z": i % 3}}}, raw_features={"global_mean_abs": i % 5})
    result = v.increment(rows, B=1)
    names = {r["name"] for r in result["tests"]}
    assert {"B3|B0+B1+B2", "B2|B0+B1+B3", "raw_features|B0+B1+B2", "B1|B0+B2"} <= names
    assert all("OOF_R2" in r and "F" in r["ols"] for r in result["tests"])
    assert result["decision"] == "underpowered"
    rows[0]["edited_after_score"] = True
    reports = v.sensitivity_reports(rows, B=1)
    assert any(r["sensitivity"] == "include_edited_arms" for r in reports)
    assert all(r["decision"] == "descriptive_sensitivity_only" for r in reports)


def test_optional_log_fields_never_copy_prose_or_unknown_roi():
    from tribe_service import score_log
    report = {"score_terms": {"roi_prior": {"value": 1., "weight": .15, "provenance": "literature_prior_unvalidated", "note": "private prose"}},
              "prior_inputs": {"evidence_without_roi": 60., "facet_without_boost": None, "tailoring_points": 1., "specificity_points": 0., "message": "private prose"},
              "active_flags": {"PITCHCHECK_ROI_PRIOR_WEIGHT": .15, "message": "private prose"}}
    fmri = {"roi_summary": {"rois": {"mofc": {"z": 1., "mean_signed": .2, "note": "private prose"}, "private_prose": {"z": 1.}}}}
    output = score_log.monitoring_fields(report, fmri, {"recipient_stance": "skeptical", "personalization_level": "private prose"})
    assert "private prose" not in json.dumps(output) and "private_prose" not in json.dumps(output)
    assert output["recipient_stance"] == "skeptical" and output["personalization_level"] is None
    assert output["prior_inputs"]["facet_without_boost"] is None


def test_corpus_checks_encoder_and_consent_before_blind_calls(monkeypatch):
    from tribe_service import engine
    monkeypatch.setenv("PITCHCHECK_SCORE_LOG_KEY", "k" * 32)
    monkeypatch.setattr(engine, "TRIBE_ALLOW_MOCK", False)
    calls = []
    monkeypatch.setattr(engine, "score_text", lambda text: calls.append("score") or np.ones((2, 2)))
    monkeypatch.setattr(engine, "analyze_predictions", lambda p: ({}, {"text_feature_compatible": False}, {}))
    monkeypatch.setattr(s.llm, "_call_openrouter_once", lambda *a, **k: pytest.fail("No paid call before encoder check"))
    corpus = {"text": "An owned synthetic pitch", "persona": "A careful reader", "channel": "email", "group_id": "g1", "language": "en", "consent_text": True}
    with pytest.raises(ValueError, match="compatibility"):
        s.score_corpus([corpus])
    assert calls == ["score"]
    calls.clear()
    with pytest.raises(ValueError, match="consent"):
        s.score_corpus([corpus, {**corpus, "consent_text": False}])
    assert calls == []


def test_no_outcome_variation_keeps_parity_but_reports_undefined_metric():
    rows = fixture.records(groups=2)
    for row in rows:
        row["positives"] = 0
    assert v.compare(rows)["active_C_w"] is None
    assert v.increment(rows, B=1)["decision"] == "underpowered"
    for row in rows:
        row["scored_at"] = "2026-01-01T00:00:00+00:00"
    assert v.sweep(rows, B=199)["decision"] == "underpowered"
