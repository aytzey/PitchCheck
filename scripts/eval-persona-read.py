#!/usr/bin/env python3
"""WS06 label stability monitoring; no persuasion evidence and no release gate.

Offline: python scripts/eval-persona-read.py --selftest
Offline report: python scripts/eval-persona-read.py --report /tmp/persona-read.json
Only after separate owner authorization (never in CI):
  python scripts/eval-persona-read.py --allow-paid-run --model PINNED_MODEL --output /tmp/persona-read.json
Requires OPENROUTER_API_KEY and the service Python environment; no GPU.
Twenty synthetic EN/TR personas, three evaluator reads each, six title-only pairs.
The production evaluator prompt and persona normalizer remain the single source of definitions.
Outputs contain IDs, normalized labels and fact counts, never persona/fact text or credentials.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).with_name("fixtures") / "persona_read_cases.json"
sys.path.insert(0, str(ROOT))
RUNS = 3
SYNTHETIC_SIGNALS = {"emotional_engagement": 72., "personal_relevance": 65., "social_proof_potential": 58.,
           "memorability": 60., "attention_capture": 70., "cognitive_friction": 30.}


def validate_fixtures(data):
    cases, pairs = data["cases"], data["title_pairs"]
    if len(cases) != 20 or Counter(row["language"] for row in cases) != {"en": 10, "tr": 10}:
        raise ValueError("Need twenty synthetic cases, ten per language")
    by_id = {row["id"]: row for row in cases}
    if len(by_id) != len(cases):
        raise ValueError("Duplicate fixture id")
    for case in cases:
        if any(not isinstance(case.get(key), str) or not case[key].strip()
               for key in ("id", "title", "facts", "message", "platform")):
            raise ValueError("Fixture fields must be nonempty strings")
        if case["platform"] not in {"email", "linkedin", "ad-copy", "landing-page", "general"}:
            raise ValueError("Invalid fixture platform")
    if len(pairs) != 6 or any(not isinstance(pair, list) or len(pair) != 2 for pair in pairs):
        raise ValueError("Need six title-only pairs")
    paired_ids = [case_id for pair in pairs for case_id in pair]
    if len(set(paired_ids)) != 12 or not set(paired_ids) <= by_id.keys():
        raise ValueError("Title pairs must use twelve distinct existing cases")
    for first, second in pairs:
        a, b = by_id[first], by_id[second]
        if a["title"] == b["title"] or any(a[key] != b[key] for key in ("facts", "message", "language", "platform")):
            raise ValueError("Title-pair inputs may differ only in title")
    if Counter(by_id[pair[0]]["language"] for pair in pairs) != {"en": 3, "tr": 3}:
        raise ValueError("Need three title pairs in each language")
    return data


def load_fixtures():
    return validate_fixtures(json.loads(FIXTURES.read_text(encoding="utf-8")))


def sample(llm, case, model):
    persona = case["title"] + ". " + case["facts"]
    prompt = llm._build_user_prompt(case["message"], persona, case["platform"], SYNTHETIC_SIGNALS)
    raw = llm._call_openrouter_once(prompt, model=model, temperature=.2)
    if not isinstance(raw, dict) or not isinstance(raw.get("context_fit"), dict):
        raise ValueError("Evaluator returned no context-fit block")
    fit = llm._normalise_context_fit(raw["context_fit"], persona=persona, message=case["message"])
    return {"recipient_stance": fit["recipient_stance"], "personalization_level": fit["personalization_level"],
            "unused_facts_count": len(fit["unused_persona_facts"])}


def summarize(rows, fixtures):
    validate_fixtures(fixtures)
    by_id = {case["id"]: case for case in fixtures["cases"]}
    grouped = {case_id: {} for case_id in by_id}
    for row in rows:
        if not isinstance(row, dict) or row.get("error") or row.get("id") not in grouped:
            raise ValueError("Failed or unknown sample; do not interpret incomplete data")
        case_id, run = row["id"], row.get("run")
        if type(run) is not int or not 0 <= run < RUNS or run in grouped[case_id]:
            raise ValueError("Invalid or duplicate case/run")
        if row.get("recipient_stance") not in {None, "receptive", "neutral", "skeptical"} or row.get(
                "personalization_level") not in {None, "generic", "category", "recipient_specific"}:
            raise ValueError("Report contains an unnormalized persona label")
        if type(row.get("unused_facts_count")) is not int or not 0 <= row["unused_facts_count"] <= 3:
            raise ValueError("Invalid unused-fact count")
        grouped[case_id][run] = row
    if any(len(group) != RUNS for group in grouped.values()):
        raise ValueError("Every fixture needs exactly three runs")

    def metrics(ids):
        selected = [grouped[case_id][run] for case_id in ids for run in range(RUNS)]
        agreement = sum(max(Counter(row["recipient_stance"] for row in grouped[case_id].values()
                                    if row["recipient_stance"] is not None).values(), default=0) for case_id in ids) / len(selected)
        pairs = [pair for pair in fixtures["title_pairs"] if pair[0] in ids]
        mismatches = [{"ids": pair, "run": run} for pair in pairs for run in range(RUNS)
                      if grouped[pair[0]][run]["recipient_stance"] != grouped[pair[1]][run]["recipient_stance"]]
        empty_rate = sum(row["unused_facts_count"] == 0 for row in selected) / len(selected)
        return {"cases": len(ids), "samples": len(selected), "stance_agreement_rate": agreement,
                "stance_agreement_definition": "Non-null modal votes per persona / all three runs; null never counts as agreement",
                "recipient_stance_null_rate": sum(row["recipient_stance"] is None for row in selected) / len(selected),
                "personalization_level_null_rate": sum(row["personalization_level"] is None for row in selected) / len(selected),
                "unused_facts_empty_rate": empty_rate, "title_pairs": len(pairs),
                "title_pair_consistency_rate": 1 - len(mismatches) / (len(pairs) * RUNS),
                "title_pair_mismatches": mismatches,
                "monitoring_thresholds": {"stance_agreement_at_least_80_percent": agreement >= .8,
                                          "unused_facts_empty_at_most_70_percent": empty_rate <= .7,
                                          "title_pairs_same_stance": not mismatches}}

    return {"purpose": "Model-label behavior only; unvalidated priors, not persuasion evidence or a release gate",
            "all": metrics(set(by_id)),
            "tr": metrics({case_id for case_id, case in by_id.items() if case["language"] == "tr"})}


def run(llm, fixtures, model):
    rows = []
    for run_index in range(RUNS):
        for case in fixtures["cases"]:
            row = {"id": case["id"], "run": run_index}
            try:
                row.update(sample(llm, case, model))
            except Exception as exc:
                row["error"] = type(exc).__name__  # Never serialize exception bodies, prompts or API responses.
            rows.append(row)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "requested_model": model,
              "fixture_sha256": hashlib.sha256(FIXTURES.read_bytes()).hexdigest(),
              "system_prompt_sha256": hashlib.sha256(llm.SYSTEM_PROMPT.encode()).hexdigest(),
              "prompt_doctrine_version": llm.PROMPT_DOCTRINE_VERSION,
              "context_fit_rubric_version": llm.CONTEXT_FIT_RUBRIC_VERSION,
              "persona_read_priors_enabled": llm.PERSONA_READ_PRIORS,
              "measurements": "Equal synthetic neural controls; no GPU or persuasion outcomes", "samples": rows}
    report["summary"] = None if any("error" in row for row in rows) else summarize(rows, fixtures)
    return report


def selftest():
    from contextlib import redirect_stderr
    from copy import deepcopy
    import io
    from tribe_service import llm_layer as llm

    fixtures = load_fixtures()
    calls = []
    def fake_evaluator(prompt, **kwargs):
        assert llm.STANCE_RULE in prompt and llm.PERSONALIZATION_RULE in prompt
        assert kwargs == {"model": "test/offline", "temperature": .2}
        calls.append(1)
        case = fixtures["cases"][(len(calls) - 1) % 20]
        return {"context_fit": {"recipient_stance": " Skeptical ", "personalization_level": " category ",
                                "unused_persona_facts": [case["facts"].split(". ")[-1], "invented detail is dropped"]}}

    with patch.object(llm.httpx, "post", side_effect=AssertionError("Network forbidden in selftest")), patch.object(
            llm, "_call_openrouter_once", fake_evaluator):
        report = run(llm, fixtures, "test/offline")
    assert len(calls) == 60 and report["summary"]["all"]["stance_agreement_rate"] == 1
    assert report["summary"]["tr"]["samples"] == 30
    assert report["summary"]["all"]["unused_facts_empty_rate"] == 0
    assert all(row["unused_facts_count"] == 1 for row in report["samples"])
    assert all(report["summary"]["all"]["monitoring_thresholds"].values())
    assert "facts" not in report["samples"][0] and "persona" not in report["samples"][0]

    rows = deepcopy(report["samples"])
    for row in rows:
        row["unused_facts_count"] = 0 if row["id"] in {case["id"] for case in fixtures["cases"][:14]} else 1
    assert summarize(rows, fixtures)["all"]["monitoring_thresholds"]["unused_facts_empty_at_most_70_percent"]
    rows[14]["unused_facts_count"] = 0
    assert not summarize(rows, fixtures)["all"]["monitoring_thresholds"]["unused_facts_empty_at_most_70_percent"]
    rows = deepcopy(report["samples"])
    for row in rows[:12]:
        row["recipient_stance"] = "neutral"
    assert summarize(rows, fixtures)["all"]["stance_agreement_rate"] == .8
    rows[12]["recipient_stance"] = "neutral"
    assert not summarize(rows, fixtures)["all"]["monitoring_thresholds"]["stance_agreement_at_least_80_percent"]
    mismatched = deepcopy(report["samples"])
    mismatched[0]["recipient_stance"] = "neutral"
    assert summarize(mismatched, fixtures)["all"]["title_pair_mismatches"] == [{"ids": fixtures["title_pairs"][0], "run": 0}]
    assert summarize(mismatched, fixtures)["tr"]["title_pair_consistency_rate"] == 1
    for row in rows:
        row["recipient_stance"] = None
    assert summarize(rows, fixtures)["all"]["stance_agreement_rate"] == 0
    assert summarize(rows, fixtures)["all"]["recipient_stance_null_rate"] == 1
    invalid_pair = deepcopy(fixtures)
    invalid_pair["cases"][1]["facts"] += " Extra fact."
    try:
        validate_fixtures(invalid_pair)
    except ValueError:
        pass
    else:
        raise AssertionError("Accepted a pair that differs beyond title")
    for invalid in (rows[:-1], rows + [rows[0]], [{**row, "run": True} for row in rows],
                    [{**row, "unused_facts_count": True} for row in rows],
                    [{**row, "recipient_stance": "unknown"} for row in rows],
                    [{**row, "error": "ValueError"} for row in rows]):
        try:
            summarize(invalid, fixtures)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted invalid or incomplete monitoring data")
    with patch.object(llm, "_call_openrouter_once", return_value=None):
        failed = run(llm, fixtures, "test/offline")
    assert failed["summary"] is None and len(failed["samples"]) == 60
    assert all(row["error"] == "ValueError" for row in failed["samples"])
    for flags, env in (([], {}), (["--allow-paid-run"], {"CI": "1"})):
        with patch.object(sys, "argv", ["eval-persona-read.py", "--model", "test/offline", "--output", "/unused", *flags]), patch.dict(
                os.environ, {"OPENROUTER_API_KEY": "offline-test", **env}), redirect_stderr(io.StringIO()):
            try:
                main()
            except SystemExit as exc:
                assert exc.code == 2
            else:
                raise AssertionError("Accepted a paid run without opt-in or inside CI")
    print("selftest: 20 EN/TR fixtures, 60 mocked reads, six title pairs, thresholds, nulls, trust boundary and invalid-input checks passed; 0 API calls")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-paid-run", action="store_true", help="Requires separate explicit owner authorization")
    parser.add_argument("--model", help="Explicit pinned model ID; no automatic model selection")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    fixtures = load_fixtures()
    if args.report:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        if report["fixture_sha256"] != hashlib.sha256(FIXTURES.read_bytes()).hexdigest():
            parser.error("Report fixture revision differs from the current frozen cases")
        print(json.dumps(summarize(report["samples"], fixtures), ensure_ascii=False, indent=2))
        return
    if not args.allow_paid_run or os.getenv("CI") or not args.model or not args.output or not os.getenv("OPENROUTER_API_KEY", "").strip():
        parser.error("Paid runs require separate owner authorization, --allow-paid-run, --model, --output and OPENROUTER_API_KEY; forbidden in CI")
    with args.output.open("x", encoding="utf-8") as handle:
        from tribe_service import llm_layer as llm
        report = run(llm, fixtures, args.model)
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {args.output}; {len(report['samples'])} samples; summary {'complete' if report['summary'] else 'unavailable (failed samples)'}")
    if report["summary"] is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
