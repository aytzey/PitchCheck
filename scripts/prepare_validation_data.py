#!/usr/bin/env python3
"""Local WS10 collection helpers; no network, sending or product changes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tribe_service import score_log
from validate_incremental import PME_SCOPE, finite, read_jsonl, spearman

UPWORTHY_PERSONA = "An English-speaking reader considering whether to open this news story."
PME_ITEMS = ("pme_persuasive", "pme_convincing", "pme_effective")
PANEL_SCALES = {"persuasive": 4, "information": 4, "feelings": 4, "reply_likelihood": 7,
                "believable": 4, **dict.fromkeys(PME_ITEMS, 4)}


def flag(value):
    value = str(value).strip().lower()
    if value in {"", "0", "0.0", "false", "f", "no", "n"}:
        return False
    if value in {"1", "1.0", "true", "t", "yes", "y"}:
        return True
    raise ValueError("Unrecognized randomization/winner flag")


def integer(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}") from exc
    finite(number, name, 0)
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)


def prepare_upworthy(rows, *, exclude_winners=False):
    groups = defaultdict(list)
    exclusions = Counter()
    for row in rows:
        if not row.get("clickability_test_id"):
            raise ValueError("Missing clickability_test_id")
        groups[str(row["clickability_test_id"])].append(row)
    result = []
    for group, packages in groups.items():
        risk_keys = ("problem", "randomization_imbalace_risk", "randomization_imbalance_risk")
        if any(not any(k in row for k in risk_keys) for row in packages):
            raise ValueError("Randomization risk flag required for every package")
        risky = any(any(flag(row[k]) for k in risk_keys if k in row) for row in packages)
        dates = [str(row.get("created_at", row.get("test_date", "")))[:10] for row in packages]
        risky |= any("2013-06-01" <= value < "2014-02-01" for value in dates if value)
        if risky:
            exclusions["randomization_risk_tests"] += 1
            continue
        images = {row.get("eyecatcher_id") for row in packages}
        if len(images) != 1 or not next(iter(images)):
            exclusions["nonconstant_or_missing_eyecatcher_tests"] += 1
            continue
        titles = defaultdict(list)
        for row in packages:
            if exclude_winners and flag(row.get("winner", False)):
                exclusions["winner_packages"] += 1
                continue
            title = " ".join(str(row.get("headline", row.get("title", ""))).split())
            if not title:
                raise ValueError("Missing headline")
            identifier = score_log.pitch_id(title)
            if identifier is None:
                raise ValueError("Set the shared >=32-byte PITCHCHECK_SCORE_LOG_KEY")
            n = integer(row.get("impressions", row.get("n_sends")), "impressions")
            k = integer(row.get("clicks", row.get("positives")), "clicks")
            if n < 1 or k > n:
                raise ValueError("Invalid Upworthy binomial counts")
            titles[identifier].append({**row, "_title": title, "_n": n, "_k": k})
        if len(titles) < 2:
            exclusions["fewer_than_two_distinct_titles_tests"] += 1
            continue
        for identifier, duplicates in titles.items():
            n, k = sum(r["_n"] for r in duplicates), sum(r["_k"] for r in duplicates)
            title = duplicates[0]["_title"]
            result.append({"pitch_id": identifier, "group_id": group, "channel": "ad-copy", "platform": "ad-copy",
                           "language": "en", "text": title, "consent_text": True, "persona": UPWORTHY_PERSONA,
                           "data_source": "upworthy", "calibration_basis": "proxy_upworthy", "outcome_family": "click",
                           "n_sends": n, "positives": k, "outcome": k / n, "edited_after_score": False,
                           "winner": any(flag(r.get("winner", False)) for r in duplicates)})
    return result, {"exclusions": dict(exclusions), "scope": "Headline clicks are an ad-copy proxy, not sales replies.",
                    "calibration_basis": "proxy_upworthy", "license": "CC BY 4.0", "source_doi": "10.1038/s41597-021-00934-7"}


def assign_campaign(recipient_hash, group_id, selected_id, semantic_id, candidates):
    if not re.fullmatch(r"[0-9a-fA-F]{32,128}", recipient_hash):
        raise ValueError("Use a recipient hash, never a recipient address")
    if not group_id or not isinstance(group_id, str):
        raise ValueError("group_id required")
    if selected_id not in candidates or semantic_id not in candidates:
        raise ValueError("Both candidate pitch IDs are required")
    digest = score_log.pitch_id(json.dumps([group_id, recipient_hash.lower()], separators=(",", ":")))
    if digest is None:
        raise ValueError("Shared HMAC key required")
    arm = "semantic_only" if selected_id != semantic_id and int(digest[-1], 16) % 2 else "with_neural"
    identifier = candidates[semantic_id if arm == "semantic_only" else selected_id]
    return {"group_id": group_id, "arm": arm, "pitch_id": identifier, "score_log_id": identifier,
            "assignment_id": digest, "randomized": selected_id != semantic_id,
            "data_source": "randomized_campaign" if selected_id != semantic_id else "production_monitoring", "edited_after_score": False}


def panel_assign(stimuli, *, raters_per_message=15, seed=10):
    if type(raters_per_message) is not int or raters_per_message < 15:
        raise ValueError("At least 15 raters per message")
    groups = defaultdict(list)
    for row in stimuli:
        groups[row["group_id"]].append(row["pitch_id"])
    if not groups or any(len(ids) != 4 or len(set(ids)) != 4 for ids in groups.values()):
        raise ValueError("Each base-pitch family requires exactly four distinct variants")
    rng = random.Random(seed)
    families = sorted(groups)
    for ids in groups.values():
        rng.shuffle(ids)
    assignments = []
    for rater in range(raters_per_message * 4):
        order = list(enumerate(families))
        rng.shuffle(order)
        for family_index, group in order:
            assignments.append({"rater_id": f"r{rater + 1:04}", "group_id": group,
                                "pitch_id": groups[group][(rater + family_index) % 4]})
    return assignments


def panel_aggregate(ratings, *, panel_id, seed=10):
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+", panel_id):
        raise ValueError("panel_id must be a simple identifier")
    if len({r.get("channel") for r in ratings}) != 1:
        raise ValueError("Analyze panel channels separately")
    seen = set()
    messages = defaultdict(list)
    for row in ratings:
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("rater_id", "pitch_id", "group_id", "channel")):
            raise ValueError("Panel identifiers/channel are required")
        key = (row["rater_id"], row["group_id"])
        if key in seen:
            raise ValueError("A rater may see at most one variant per family")
        seen.add(key)
        for item, maximum in PANEL_SCALES.items():
            value = finite(row.get(item), item, 1, maximum)
            if int(value) != value:
                raise ValueError("Panel items require integer ratings")
        messages[(row["pitch_id"], row["group_id"], row["channel"])].append(row)
    rng = np.random.default_rng(seed)
    result, halves = [], []
    for (pitch, group, channel), rows in messages.items():
        if len(rows) < 15:
            raise ValueError("Each message requires at least 15 ratings")
        values = np.array([np.mean([r[k] for k in PME_ITEMS]) for r in rows])
        order = rng.permutation(len(values))
        a, b = np.array_split(values[order], 2)
        halves.append([float(a.mean()), float(b.mean())])
        result.append({"pitch_id": pitch, "group_id": group, "channel": channel, "outcome_family": "pme",
                       "outcome": float(values.mean()), "n_raters": len(rows), "data_source": "pme_panel",
                       "calibration_basis": f"pme_panel_{panel_id}", "edited_after_score": False,
                       "item_means": {k: float(np.mean([r[k] for r in rows])) for k in PANEL_SCALES}})
    r = spearman([h[0] for h in halves], [h[1] for h in halves])
    reliability = 2 * r / (1 + r) if r is not None and r > -1 else None
    return result, {"split_half_spearman": r, "spearman_brown": reliability,
                    "collect_more_ratings": reliability is None or reliability < .70,
                    "scope": PME_SCOPE, "calibration_basis": f"pme_panel_{panel_id}",
                    "fitted_scope": "calibrated to rated effectiveness, not replies"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-upworthy", "panel-assign", "panel-aggregate", "assign-campaign"))
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-winners", action="store_true")
    parser.add_argument("--panel-id", default="unregistered")
    parser.add_argument("--raters-per-message", type=int, default=15)
    args = parser.parse_args()
    meta = {}
    if args.command == "prepare-upworthy":
        with args.input.open(encoding="utf-8-sig", newline="") as handle:
            rows, meta = prepare_upworthy(list(csv.DictReader(handle)), exclude_winners=args.exclude_winners)
    elif args.command == "panel-assign":
        rows = panel_assign(read_jsonl(args.input), raters_per_message=args.raters_per_message)
    elif args.command == "panel-aggregate":
        rows, meta = panel_aggregate(read_jsonl(args.input), panel_id=args.panel_id)
    else:
        rows = [assign_campaign(**row) for row in read_jsonl(args.input)]
    with args.output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    with args.output.with_suffix(".metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
