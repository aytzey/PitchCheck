"""CPU checks for the real-runtime calibration harness; no model is loaded."""
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from tribe_service.persuasion_features import ROI_PRIOR

benchmark = runpy.run_path(str(Path(__file__).parents[2] / "scripts/benchmark-runtime.py"))


def row(case, z, sha="atlas"):
    return {"case": case, "roi_summary": {"asset_sha256": sha, "rois": {k: {"z": z} for k in ROI_PRIOR}}}


def test_unit_calibration_deduplicates_warm_cases_and_centers_each_roi():
    rows = [row(0, -2), row(1, 1), row(2, 3)]
    result = benchmark["unit_calibration"]([*rows, rows[0]], "encoder", "run")
    assert result["n_cases"] == 3 and set(result["centers"].values()) == {1.0}
    coefficient = sum(s * w for s, w, _, _ in ROI_PRIOR.values()) / sum(w for _, w, _, _ in ROI_PRIOR.values())
    assert result["ROI_Z_SCALE"] == pytest.approx(np.std([-3, 0, 2]) * abs(coefficient))
    assert result["provenance"] == "unit_calibration:run" and result["validated"] is False


@pytest.mark.parametrize("rows", [[], [row(0, 1)], [row(0, 1), row(1, 1)],
                                 [row(0, 0), row(1, np.nan)], [row(0, 0), row(1, 1, "other")]])
def test_unit_calibration_rejects_degenerate_or_mixed_atlas_data(rows):
    with pytest.raises(ValueError, match="ROI unit calibration"):
        benchmark["unit_calibration"](rows, "encoder", "run")


@pytest.mark.parametrize("data", [{}, [], ["one"], ["one", "one"], ["one", " "], ["one", 2]])
def test_corpus_rejects_invalid_or_repeated_texts(tmp_path, data):
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Corpus must be"):
        benchmark["load_corpus"](corpus)


def test_fixed_calibration_corpus():
    corpus = Path(__file__).parents[2] / "scripts/fixtures/roi-unit-calibration.json"
    texts = benchmark["load_corpus"](corpus)
    assert len(texts) == 64
    assert texts[0].startswith("Jordan, Our deployment") and texts[2].count("Our deployment") == 9
