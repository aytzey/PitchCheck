#!/usr/bin/env python3
"""Post-release model-behavior monitoring, not evidence of persuasion or a release gate.

Run later with OPENROUTER_API_KEY and the service's existing Python environment:
  python scripts/ab-prompt-doctrine.py --old-worktree /repo/before --new-worktree /repo/after --model MODEL --output /tmp/doctrine-ab.jsonl
The worktrees must already exist: old is pre-WS04 Faz 0 (f0da33d), new adds only WS04.
--limit 12 permits a small first run; default is all 27.
Workers run in each worktree with --arm old|old-repeat|new and --case CASE_ID.
  python scripts/ab-prompt-doctrine.py --report /tmp/doctrine-ab.jsonl
  python scripts/ab-prompt-doctrine.py --selftest

JSONL keeps generated text for manual review of benefit, next step and personal tone.
Fill the manual_review fields before reporting if desired. No keys or headers are saved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from unittest.mock import patch
import uuid

FIXTURES = Path(__file__).with_name("fixtures") / "ab_pairs.json"
ARMS = ("old", "old-repeat", "new")
# Same fixed inputs as test_llm_layer.py; no engine import, GPU or candidate measurements.
SIGNALS = {"emotional_engagement": 72.0, "personal_relevance": 65.0, "social_proof_potential": 58.0,
           "memorability": 60.0, "attention_capture": 70.0, "cognitive_friction": 30.0}
RAW_FEATURES = {"global_mean_abs": 0.25, "global_peak_abs": 0.6, "temporal_std": 0.08,
                "early_mean": 0.2, "late_mean": 0.3, "max_temporal_delta": 0.12,
                "spatial_spread": 0.07, "focus_ratio": 0.35, "sustain_ratio": 0.6, "arc_ratio": 1.2}
MINDSTATE = re.compile(r"risk-averse|conformist|your brain|reader['’]s brain|reward (?:fires|lights up|system)|"
                       r"trust signal|doubt (?:rises|spikes)|striatum|accumbens|amygdala|hippocamp\w*|"
                       r"\b(?:NAcc|caudate|putamen)\b|dlPFC load", re.I)
# ponytail: literal cortical names/acronyms; extend this list when the ROI vocabulary grows.
REGION = re.compile(r"\b(?:[dv]?[ml]?PFC|PCC|ACC|rIPL|pIFG|pSTS|r?TPJ|precuneus|insula|"
                    r"temporal[ -]pole|temporoparietal[ -]junction|angular(?: gyrus)?|"
                    r"(?:medial|ventromedial|dorsomedial|dorsolateral|ventrolateral)[ -]prefrontal(?: cortex)?|"
                    r"(?:anterior|posterior)[ -]cingulate(?: cortex)?|orbitofrontal(?: cortex)?|"
                    r"(?:inferior|superior)[ -](?:parietal|frontal|temporal)(?: cortex| gyrus| lobule)?|"
                    r"prefrontal korteks|singulat|singülat)\b", re.I)
QUOTED = re.compile(r'"([^"\n]+)"|“([^”\n]+)”|‘([^’\n]+)’')


def load_fixtures():
    data = json.loads(FIXTURES.read_text(encoding="utf-8"))
    pairs = data["pairs"]
    ids = [row["id"] for row in pairs]
    if not 25 <= len(pairs) <= 30 or len(ids) != len(set(ids)):
        raise ValueError("Fixtures must have 25–30 unique synthetic cases")
    if len(set(data["neutral_ids"])) != 8 or not set(data["neutral_ids"]) <= set(ids):
        raise ValueError("Declare exactly eight existing neutral cases")
    for row in pairs:
        if any(not isinstance(row.get(key), str) or not row[key].strip()
               for key in ("id", "message", "persona", "platform", "kind", "language")):
            raise ValueError("Fixture fields must be nonempty strings")
    personal = [row for row in pairs if row["kind"] in {"personal", "romantic"}]
    if len(personal) < 3 or not any(row["language"] == "tr" for row in personal):
        raise ValueError("Need three personal cases, including a Turkish case")
    if sum(row["language"] == "tr" and row["kind"] == "business" for row in pairs) < 2:
        raise ValueError("Need two Turkish business cases")
    return data


def strings(value, path=""):
    if isinstance(value, str):
        yield {"field": path, "text": value}
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from strings(item, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from strings(item, f"{path}[{index}]")


def generated_fields(parsed, phase):
    if phase == "score":
        selected = {key: parsed[key] for key in ("narrative", "top_moves") if key in parsed}
        rewrites = parsed.get("rewrite_suggestions")
        selected["rewrite_suggestions"] = [
            {key: row[key] for key in ("title", "after", "why") if key in row}
            for row in rewrites if isinstance(row, dict)] if isinstance(rewrites, list) else []
    elif phase == "refine":
        selected = {"candidates": parsed.get("candidates", [])}
    else:
        return []
    return list(strings(selected, phase))


def safety_hits(fields, listed_roi_quotes=()):
    hits = []
    for field in fields:
        text = field["text"]
        hits.extend({"field": field["field"], "kind": "mindstate_or_subcortical", "match": m.group()}
                    for m in MINDSTATE.finditer(text))
        # Only an entire, verbatim listed ROI statement inside quotation marks gets K14's exception.
        unquoted = QUOTED.sub(lambda m: "" if next(group for group in m.groups() if group is not None)
                             in listed_roi_quotes else m.group(), text)
        hits.extend({"field": field["field"], "kind": "unquoted_region", "match": m.group()}
                    for m in REGION.finditer(unquoted))
    return hits


def run_case(llm, case, model):
    calls, fields = [], []
    phase = "score"
    post = llm.httpx.post

    def recorded_post(*args, **kwargs):
        response = post(*args, **kwargs)
        try:
            body = response.json()
        except ValueError:
            body = {}
        body = body if isinstance(body, dict) else {}
        usage = body.get("usage")
        calls.append({"phase": phase, "status": response.status_code,
                      "requested_model": kwargs.get("json", {}).get("model"),
                      "served_model": body.get("model"),
                      "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage, dict) else None})
        choices = body.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        message = choice.get("message")
        content = message.get("content", "") if isinstance(message, dict) else ""
        parsed = llm._parse_json_content(content) if isinstance(content, str) else None
        if isinstance(parsed, dict):
            fields.extend(generated_fields(parsed, phase))
        return response

    with patch.object(llm.httpx, "post", recorded_post):
        report = llm.interpret_persuasion(case["message"], case["persona"], case["platform"],
                                        dict(SIGNALS), dict(RAW_FEATURES), None, openrouter_model=model)
        if report.get("robustness", {}).get("llm_model") != model:
            raise RuntimeError("Score fell back without a usable LLM response")
        phase = "refine"
        refine = llm.refine_pitch_message(case["message"], case["persona"], case["platform"], [],
                                         force_rewrite=True, openrouter_model=model)
        candidates = refine["candidates"]
        if len(candidates) != 3:
            raise RuntimeError("Refine did not produce three candidates")
        known = llm._refine_concrete_details(case["message"] + "\n" + case["persona"])
        minimum, maximum = llm._refine_length_bounds(case["message"])
        checks = [{"id": f"c{i}", "message": candidate, "words": len(candidate.split()),
                   "new_details": sorted(llm._refine_concrete_details(candidate) - known),
                   "length_rejected": not minimum <= len(candidate.split()) <= maximum}
                  for i, candidate in enumerate(candidates, 1)]
        phase = "critic"
        measurements = [{"id": "original", "message": case["message"]},
                        *({"id": row["id"], "message": row["message"]} for row in checks)]
        critic_prompt = llm._build_refine_critic_prompt(case["message"], case["persona"], case["platform"],
                                                       [], measurements)
        # No invented measurements: this run isolates prompts and asks the unchanged critic about text.
        critic_prompt = ("Offline prompt A/B: no candidate was measured with TRIBE. Assess factual support "
                         "and context from supplied text only; no candidate response differences are available.\n"
                         + critic_prompt.replace("Actual TRIBE measurements of the original and every candidate:",
                                                 "Original and candidates (no candidate measurements):"))
        critic = llm._parse_json_content(llm._post_refine_chat(
            llm.REFINE_CRITIC_SYSTEM_PROMPT, critic_prompt, model, temperature=llm._critic_temperature(model)))
        reviews = critic.get("evaluations", []) if isinstance(critic, dict) else []
        by_id = {row.get("id"): row for row in reviews if isinstance(row, dict)}
        if set(by_id) != {"original", "c1", "c2", "c3"} or len(reviews) != 4 or any(
                not isinstance(row.get("supported"), bool) for row in reviews):
            raise RuntimeError("Critic support judgments were incomplete")
    # Fixed SAMPLE inputs have no ROI items; future synthesis item formats need an explicit allowlist update.
    roi = report.get("robustness", {}).get("roi_readout", report.get("roi_readout", {}))
    listed = [row["text"] for row in strings(roi) if "predicted average-reader response" in row["text"]]
    return {"final_score": report["persuasion_score"], "context_fit": report.get("context_fit"),
            "api_calls": calls, "generated_fields": fields,
            "prompt_doctrine_version": report["robustness"].get("prompt_doctrine_version", "legacy_unversioned"),
            "candidate_checks": checks, "critic_evaluations": reviews,
            "safety_hits": safety_hits(fields, listed), "listed_roi_quotes": listed,
            "manual_review": {"persona_specific_benefit": None, "concrete_next_step": None,
                              "personal_tone": None, "notes": ""}}


def schedule(pairs):
    for index, case in enumerate(pairs):
        for arm in ARMS[index % 3:] + ARMS[:index % 3]:
            yield case, arm


def summarize(rows, neutral_ids):
    if not rows or any(row.get("error") for row in rows):
        raise ValueError("Run is empty or has failed cases; do not treat missing data as zero")
    if any(len({row[key] for row in rows}) != 1 for key in ("run_id", "day", "model", "fixture_sha256")):
        raise ValueError("Compare one run, UTC day, model and fixture revision at a time")
    served = {call["served_model"] for row in rows for call in row["api_calls"] if call["status"] == 200}
    if len(served) != 1 or None in served:
        raise ValueError("Successful calls must identify the same served model")
    for arms in (("old", "old-repeat"), ("new",)):
        if len({row["prompt_sha256"] for row in rows if row["arm"] in arms}) != 1:
            raise ValueError("Each arm must use a fixed prompt; old and old-repeat must match")
    grouped = {}
    for row in rows:
        key = row["case"]["id"]
        if row["arm"] in grouped.setdefault(key, {}):
            raise ValueError("Duplicate case/arm in this run")
        grouped[key][row["arm"]] = row
    if any(set(group) != set(ARMS) for group in grouped.values()):
        raise ValueError("Each case needs old, old-repeat and new")
    score_review = []
    for case_id, group in grouped.items():
        old, repeat, new = (group[arm]["final_score"] for arm in ARMS)
        delta = new - (old + repeat) / 2
        if abs(delta) > 8:
            score_review.append({"case_id": case_id, "old": old, "old_repeat": repeat, "new": new,
                                 "new_minus_aa_baseline": delta})
    rates = {}
    for arm in ARMS:
        selected = [group[arm] for group in grouped.values()]
        candidates = [item for row in selected for item in row["candidate_checks"]]
        reviews = [item for row in selected for item in row["critic_evaluations"] if item["id"] != "original"]
        rates[arm] = {"candidates": len(candidates),
                      "new_detail_rejection_rate": statistics.mean(bool(item["new_details"]) for item in candidates),
                      "length_rejection_rate": statistics.mean(item["length_rejected"] for item in candidates),
                      "critic_unsupported_rate": statistics.mean(item["supported"] is False for item in reviews),
                      "safety_hits": sum(len(row["safety_hits"]) for row in selected)}
    neutral = {}
    for case_id in neutral_ids:
        if case_id not in grouped:
            raise ValueError("Run must include the eight predeclared neutral cases")
        group = grouped[case_id]
        facets = [group[arm].get("context_fit") for arm in ARMS]
        if not all(isinstance(facet, dict) for facet in facets):
            raise ValueError("Neutral cases require context-fit facets")
        for key, item in facets[0].items():
            if not isinstance(item, dict) or "score" not in item:
                continue
            old, repeat, new = (facet[key]["score"] for facet in facets)
            neutral.setdefault(key, []).append({"case_id": case_id, "aa_delta": repeat - old,
                                               "ab_delta": new - (old + repeat) / 2})
    facet_summary = {}
    for key, values in neutral.items():
        spread = max(abs(value["aa_delta"]) for value in values)
        mean_delta = statistics.mean(value["ab_delta"] for value in values)
        max_delta = max(abs(value["ab_delta"]) for value in values)
        facet_summary[key] = {"aa_max_abs_delta": spread, "ab_mean_delta": mean_delta,
                              "ab_max_abs_delta": max_delta, "max_within_aa_spread": max_delta <= spread,
                              "mean_within_aa_spread": abs(mean_delta) <= spread, "cases": values}
    tokens = {}
    for phase in ("score", "refine", "critic"):
        deltas = []
        for group in grouped.values():
            by_arm = [[call["prompt_tokens"] for call in group[arm]["api_calls"]
                       if call["phase"] == phase and call["status"] == 200] for arm in ARMS]
            if any(not values or any(not isinstance(value, int) or value < 0 for value in values) for values in by_arm):
                raise ValueError("Successful calls need usage.prompt_tokens for every phase")
            old, repeat, new = (statistics.mean(values) for values in by_arm)
            deltas.append(new - (old + repeat) / 2)
        tokens[phase] = {"mean_new_minus_old": statistics.mean(deltas), "max_new_minus_old": max(deltas)}
    return {"purpose": "Model behavior monitoring only; not persuasion evidence or a release gate",
            "cases": len(grouped), "model": rows[0]["model"], "served_model": next(iter(served)),
            "rates": rates, "neutral_facets": facet_summary, "prompt_tokens": tokens,
            "score_prompt_growth_within_650": tokens["score"]["max_new_minus_old"] <= 650,
            "score_shift_over_8_manual_review": score_review,
            "manual_review": [{"case_id": row["case"]["id"], "arm": row["arm"],
                               "kind": row["case"]["kind"], **row["manual_review"]} for row in rows]}


def selftest():
    from types import SimpleNamespace

    fixtures = load_fixtures()
    pairs = fixtures["pairs"]
    order = list(schedule(pairs))
    assert len(order) == len(pairs) * 3
    assert [arm for _, arm in order[:9]] == ["old", "old-repeat", "new", "old-repeat", "new", "old", "new", "old", "old-repeat"]
    fields = generated_fields({"narrative": "Clear proposal", "top_moves": [{"do": "Name the benefit"}],
                               "rewrite_suggestions": [{"before": "your brain", "after": "Want a look?"}],
                               "robustness": {"research_synthesis": "your brain"}}, "score")
    assert not safety_hits(fields)
    assert not generated_fields({"rewrite_suggestions": None}, "score")
    assert not generated_fields({"rewrite_suggestions": 4}, "score")
    assert len(safety_hits([{"field": "score.narrative", "text": "Your brain trusts a risk-averse conformist"}])) == 3
    quote = "Medial prefrontal predicted average-reader response is higher."
    assert not safety_hits([{"field": "score.narrative", "text": f'Listed ROI: "{quote}"'}], [quote])
    assert safety_hits([{"field": "score.narrative", "text": quote}], [quote])
    assert len(safety_hits([{"field": "score.narrative", "text": "TPJ, pSTS and temporal-pole"}])) == 3
    assert safety_hits([{"field": "refine.candidates[0]", "text": '"your brain medial prefrontal"'}],
                       ["your brain medial prefrontal"])
    case = pairs[0]
    fake = SimpleNamespace()
    responses = {
        "score": {"narrative": "Your brain lights up", "rewrite_suggestions": [{"before": "risk-averse", "after": "Want a look?"}]},
        "refine": {"candidates": [case["message"], case["message"] + " 99 tickets", case["message"] * 5]},
        "critic": {"evaluations": [{"id": key, "supported": key != "c2"} for key in ("original", "c1", "c2", "c3")]},
    }

    def fake_post(*_args, **kwargs):
        payload = kwargs["json"]
        body = {"model": payload["model"], "usage": {"prompt_tokens": 123},
                "choices": [{"message": {"content": json.dumps(responses[payload["phase"]])}}]}
        return SimpleNamespace(status_code=200, json=lambda: body)

    def fake_interpret(*_args, **kwargs):
        fake.httpx.post(json={"phase": "score", "model": kwargs["openrouter_model"]})
        return {"persuasion_score": 50, "robustness": {"llm_model": "fake"},
                "context_fit": {"cta_ease": {"score": 50}}}

    def fake_refine(*_args, **kwargs):
        return json.loads(fake.httpx.post(json={"phase": "refine", "model": kwargs["openrouter_model"]})
                          .json()["choices"][0]["message"]["content"])

    def fake_critic(_system, prompt, model, **_kwargs):
        assert "no candidate was measured" in prompt and "Actual TRIBE measurements" not in prompt
        return fake.httpx.post(json={"phase": "critic", "model": model}).json()["choices"][0]["message"]["content"]

    fake.httpx = SimpleNamespace(post=fake_post)
    fake.interpret_persuasion, fake.refine_pitch_message = fake_interpret, fake_refine
    fake._post_refine_chat, fake._parse_json_content = fake_critic, json.loads
    fake._build_refine_critic_prompt = lambda *_args: "Actual TRIBE measurements of the original and every candidate:"
    fake._critic_temperature = lambda _model: 0.2
    fake._refine_concrete_details = lambda text: set(re.findall(r"\d+", text))
    fake._refine_length_bounds = lambda text: (max(1, len(text.split()) - 5), len(text.split()) + 5)
    fake.REFINE_CRITIC_SYSTEM_PROMPT = "existing critic"
    captured = run_case(fake, case, "fake")
    assert [call["phase"] for call in captured["api_calls"]] == ["score", "refine", "critic"]
    assert captured["candidate_checks"][1]["new_details"] == ["99"]
    assert captured["candidate_checks"][2]["length_rejected"]
    assert len(captured["safety_hits"]) == 1 and "before" not in str(captured["generated_fields"])
    assert fake.httpx.post is fake_post  # The wrapper must not escape its case.
    rows = []
    for case in pairs[:8]:
        for arm, score in zip(ARMS, (50, 54, 53)):
            rows.append({"run_id": "selftest", "day": "2026-01-01", "model": "fake", "fixture_sha256": "fake",
                         "case": case, "arm": arm, "prompt_sha256": "new" if arm == "new" else "old",
                         "context_fit": {"cta_ease": {"score": score}},
                         "final_score": score + (8 if arm == "new" else 0),
                         "api_calls": [{"phase": phase, "status": 200, "served_model": "fake",
                                        "prompt_tokens": 100 + (600 if arm == "new" and phase == "score" else 0)}
                                       for phase in ("score", "refine", "critic")],
                         "candidate_checks": [{"new_details": [], "length_rejected": False}],
                         "critic_evaluations": [{"id": "c1", "supported": arm != "new"}],
                         "safety_hits": [], "manual_review": {}})
    summary = summarize(rows, fixtures["neutral_ids"])
    assert summary["neutral_facets"]["cta_ease"]["aa_max_abs_delta"] == 4
    assert summary["neutral_facets"]["cta_ease"]["ab_mean_delta"] == 1
    assert summary["rates"]["new"]["critic_unsupported_rate"] == 1
    assert summary["score_prompt_growth_within_650"]
    assert len(summary["score_shift_over_8_manual_review"]) == 8
    for invalid in (rows[:-1], rows + [rows[0]],
                    [{**row, "day": str(index)} for index, row in enumerate(rows)],
                    [{**row, "prompt_sha256": "changed"} if row["arm"] == "old-repeat" else row for row in rows]):
        try:
            summarize(invalid, fixtures["neutral_ids"])
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted an incomplete, duplicate or mixed-day run")
    print(f"selftest: passed; {len(pairs)} synthetic fixtures, 8 neutral cases; mocked worker and offline report; no API calls")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--old-worktree", type=Path)
    parser.add_argument("--new-worktree", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=27)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--case")
    parser.add_argument("--run-id", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    fixtures = load_fixtures()
    if args.report:
        rows = [json.loads(line) for line in args.report.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(json.dumps(summarize(rows, fixtures["neutral_ids"]), indent=2, ensure_ascii=False))
        return
    if not args.output or not args.model or not os.getenv("OPENROUTER_API_KEY", "").strip():
        parser.error("Live runs require --output, --model and OPENROUTER_API_KEY")
    args.output = args.output.resolve()
    day = datetime.now(timezone.utc).date().isoformat()
    if args.arm:
        case = next((row for row in fixtures["pairs"] if row["id"] == args.case), None)
        if case is None:
            parser.error("Workers require --case with a fixture ID")
        sys.path.insert(0, str(Path.cwd()))
        from tribe_service import llm_layer as llm
        row = {"run_id": args.run_id or day, "day": day, "arm": args.arm, "model": args.model, "case": case,
               "fixture_sha256": hashlib.sha256(FIXTURES.read_bytes()).hexdigest(),
               "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "prompt_sha256": hashlib.sha256(llm.SYSTEM_PROMPT.encode()).hexdigest()}
        try:
            row.update(run_case(llm, case, args.model))
            if datetime.now(timezone.utc).date().isoformat() != day:
                raise RuntimeError("Run crossed UTC midnight")
        except Exception as exc:
            row["error"] = type(exc).__name__  # Do not serialize exception bodies, headers or credentials.
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if row.get("error"):
            raise SystemExit(f"{args.case}/{args.arm} failed ({row['error']}); inspect locally before retrying")
        return
    if not args.old_worktree or not args.new_worktree or not 8 <= args.limit <= len(fixtures["pairs"]):
        parser.error("Coordinator requires existing --old-worktree, --new-worktree and --limit between 8 and 27")
    worktrees = {"old": args.old_worktree.resolve(), "old-repeat": args.old_worktree.resolve(),
                 "new": args.new_worktree.resolve()}
    if worktrees["old"] == worktrees["new"] or any(
            not (root / "tribe_service" / "llm_layer.py").is_file() for root in worktrees.values()):
        parser.error("Use two distinct existing PitchCheck worktrees")
    with args.output.open("x", encoding="utf-8"):
        pass  # Never overwrite an existing run.
    run_id = str(uuid.uuid4())
    for case, arm in schedule(fixtures["pairs"][:args.limit]):
        if datetime.now(timezone.utc).date().isoformat() != day:
            raise SystemExit("Run crossed UTC midnight; start a new same-day run")
        print(f"{case['id']}: {arm}", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--arm", arm, "--case", case["id"],
                        "--model", args.model, "--output", str(args.output), "--run-id", run_id],
                       cwd=worktrees[arm], check=True)


if __name__ == "__main__":
    main()
