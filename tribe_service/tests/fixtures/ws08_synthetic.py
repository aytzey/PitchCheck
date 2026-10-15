"""Synthetic WS08 scale/contrast data, independent of WS07 assets and TRIBE."""
import copy

import numpy as np

from scripts.research.run_contrasts import LABEL, POLICY


def reference(encoder="meta-llama/Llama-3.2-3B"):
    quantity = {"noise": {"sigma_within": .2, "icc1": .8, "icc1_ci95": [.7, .9],
                          "min_detectable_delta": 2.0},
                "length_model": {"a": 0.0, "b_log_words": 0.0}, "resid_sd": 1.0,
                "resid_q": {"p05": -1.5, "p25": -.7, "p50": 0, "p75": .7, "p95": 1.5},
                "flags": ["stable"]}
    return {"schema_version": 1, "kind": "roi_reference", "encoder": encoder,
            "event_version": "direct_v1", "text_input_mode": "direct", "atlas_sha256": "synthetic",
            "harness_git_sha": "synthetic", "tribev2_commit": "synthetic", "hf_snapshot": "synthetic",
            "label": LABEL, "quantities": {key: copy.deepcopy(quantity) for key in (
                "evidence_score", "roi:target", "roi:other")}, "checks": {}, "policy": POLICY.copy()}


def contrasts(effect=4.0):
    rng = np.random.default_rng(8)
    rows = []
    for i in range(20):
        difference = effect + rng.normal(0, .5)
        for condition, factor in (("a", 1), ("b", 0)):
            rows.append({"pair_id": str(i), "condition": condition,
                         "quantities": {"roi:target": factor * difference, "roi:other": factor * difference},
                         "parcels": rng.normal(0, 1, 400).tolist(),
                         "lexical": {"motor_verbs": 0, "body_words": 0}})
    sidecar = {"family": "S1", "minimum_pairs": 16,
               "manipulation_check": {"blind_classifier": "passed", "owner_review": "passed"}, "contrasts": [
        {"name": "a>b", "conditions": ["a", "b"], "rois": ["target", "other"], "alternative": "greater"}]}
    return rows, sidecar
