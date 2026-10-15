"""WS01 claims hygiene: registry provenance, no subcortical/load claims, no retired overclaims."""
from __future__ import annotations

import importlib
import importlib.util
import json
import re
from pathlib import Path
from unittest.mock import patch

from tribe_service.engine import PERSUASION_SIGNAL_LABELS
from tribe_service.llm_layer import (
    SYSTEM_PROMPT,
    _clean_llm_string,
    _format_refine_clarification_answers,
    _normalise_rewrites,
    interpret_persuasion,
)
from tribe_service.persuasion_features import (
    NEURO_AXIS_META,
    RESEARCH_SOURCES,
    SCIENTIFIC_CAVEATS,
    neuro_axes_from_analysis,
    neuro_axis_score_from_axes,
)
from tribe_service.research_synthesis import synthesize_research_findings
from tribe_service.tests.test_llm_layer import (
    SAMPLE_MESSAGE,
    SAMPLE_NEURAL_SIGNALS,
    SAMPLE_PERSONA,
    SAMPLE_PLATFORM,
    SAMPLE_RAW_FEATURES,
    VALID_LLM_RESPONSE,
    _mock_openrouter_response,
)

SUBCORTICAL = re.compile(r"\b(striatum|accumbens|NAcc|amygdala|hippocamp\w*|HPC|caudate|putamen)\b", re.I)
LOAD = re.compile(r"dlPFC load|control-load", re.I)
BANNED = ("better than self-report", "best neural predictor", "single best-evidenced", "strongest for message-consistent",
          "roughly twice", "region analogues", "stopped reading", "reliably increases", "03637751")
EVIDENCE_LEVELS = {"individual_differences_in_sample", "within_subject_condition_contrast", "within_subject_item_level",
                   "message_level_aggregate", "behavioral", "meta_analysis", "review", "model"}
ROOT = Path(__file__).resolve().parents[2]
TRACES = [[0.6, 0.5, 0.7, 0.5, 0.6], [0.30, 0.31, 0.30, 0.31, 0.30], [0.9, 0.6, 0.4, 0.3, 0.2], [0.2, 0.3, 0.35, 0.4, 0.9], [0.2, 0.3, 0.9, 0.5, 0.45, 0.5]]


def _syntheses():
    """Two axis grids (all gaps / all strengths, both routes) x the five archetype traces."""
    low = {"self_value": 30, "reward_affect": 70, "social_sharing": 40, "encoding_attention": 30, "processing_fluency": 30}
    high = {"self_value": 80, "reward_affect": 40, "social_sharing": 70, "encoding_attention": 80, "processing_fluency": 80}
    for grid in (low, high):
        axes = {key: {"score": value} for key, value in grid.items()}
        for trace in TRACES:
            yield synthesize_research_findings(axes, {"temporal_trace": trace}, {"sustain_ratio": 0.2})


def _synthesis_entries():
    for synthesis in _syntheses():
        yield from synthesis["items"]
        if synthesis["temporal_archetype"]:
            yield synthesis["temporal_archetype"]


def test_registry_entries_are_unique_https_and_typed():
    keys = [source["key"] for source in RESEARCH_SOURCES]
    assert len(keys) == len(set(keys)) and len(keys) <= 64
    for source in RESEARCH_SOURCES:
        assert source["url"].startswith("https://"), source["key"]
        if "evidence_level" in source:
            assert set(source["evidence_level"].split(";")) <= EVIDENCE_LEVELS, source["key"]


def test_every_source_key_resolves_in_registry():
    registry = {source["key"] for source in RESEARCH_SOURCES}
    used = {key for meta in NEURO_AXIS_META.values() for key in meta["source_keys"]}
    used |= {key for entry in _synthesis_entries() for key in entry["source_keys"]}
    for name in ("tribe_service.persuasion_features", "tribe_service.roi_features", "tribe_service.llm_layer"):
        if importlib.util.find_spec(name):  # WS07 ROI_PRIOR rows: (sign, weight, center, source_keys)
            used |= {key for row in getattr(importlib.import_module(name), "ROI_PRIOR", {}).values() for key in row[-1]}
    assert used - registry == set()


def test_axis_labels_caveats_prompt_and_synthesis_name_no_subcortical_or_load():
    texts = [str(meta[field]) for meta in NEURO_AXIS_META.values() for field in ("label", "analogue", "caveat")]
    texts += [*PERSUASION_SIGNAL_LABELS.values(), *SCIENTIFIC_CAVEATS, SYSTEM_PROMPT]
    texts += [str(entry.get(field, "")) for entry in _synthesis_entries()
              for field in ("label", "observation", "finding", "lever", "implication", "citation")]
    hits = [text for text in texts if SUBCORTICAL.search(text) or LOAD.search(text)]
    assert hits == []


def test_retired_overclaims_are_gone_from_service_and_desktop_sources():
    files = [*(ROOT / "tribe_service").glob("*.py"), *(ROOT / "components").glob("*.tsx")]
    hits = [(path.name, phrase) for path in files for phrase in BANNED if phrase in path.read_text(encoding="utf-8")]
    assert hits == []


def test_scientific_caveats_stay_short_and_keep_the_first_line():
    assert len(SCIENTIFIC_CAVEATS) <= 8
    assert all(len(caveat) <= 500 for caveat in SCIENTIFIC_CAVEATS)
    assert SCIENTIFIC_CAVEATS[0] == "TRIBE returns predicted neural-response analogues, not measured fMRI from this recipient."


def test_scrub_cleans_explanations_but_never_paste_ready_or_user_text():
    assert not SUBCORTICAL.search(_clean_llm_string("Here your ventral striatum lights up."))
    after = "Our reward system activates instantly and your brain will thank you."
    assert _normalise_rewrites([{"title": "t", "before": "your brain", "after": after, "why": "w"}], [])[0]["after"] == after
    answer = "Yes, our NAcc-themed brand is real."
    assert answer in _format_refine_clarification_answers([{"question": "Brand?", "answer": answer}])


def test_deterministic_score_pin_is_unchanged():
    axes = neuro_axes_from_analysis(SAMPLE_NEURAL_SIGNALS)
    assert round(neuro_axis_score_from_axes(axes), 2) == 66.30
    with patch("tribe_service.llm_layer.OPENROUTER_ENABLED", True), \
            patch("tribe_service.llm_layer.OPENROUTER_API_KEY", "sk-test-key"), \
            patch("tribe_service.llm_layer.httpx.post") as post:
        post.return_value = _mock_openrouter_response(json.dumps(VALID_LLM_RESPONSE))
        result = interpret_persuasion(SAMPLE_MESSAGE, SAMPLE_PERSONA, SAMPLE_PLATFORM, SAMPLE_NEURAL_SIGNALS, SAMPLE_RAW_FEATURES)
    assert result["persuasion_score"] == 73
