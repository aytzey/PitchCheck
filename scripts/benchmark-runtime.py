"""Run inside the GPU image: python scripts/benchmark-runtime.py --output /audit/after.

Use separate, empty TRIBE_CACHE_DIRs for before/after and the same GPU/thread limits.
The fixed synthetic pitches contain no customer data. --compare checks prediction parity.
"""
from __future__ import annotations

import argparse
import json
import resource
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.cwd()))
from tribe_service import engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    args.output.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import try_to_load_from_cache
    from neuralset.extractors.text import HuggingFaceText
    # Let the unpatched baseline use cached weights offline as well (same weights and math).
    assert isinstance(try_to_load_from_cache(engine.TRIBE_TEXT_MODEL, "config.json"), str)
    if engine.TRIBE_TEXT_MODEL not in HuggingFaceText._REPOS:
        HuggingFaceText._REPOS.append(engine.TRIBE_TEXT_MODEL)
    torch.cuda.set_per_process_memory_fraction(
        8 * 1024**3 / torch.cuda.get_device_properties(0).total_memory,
    )
    base = (
        "Our deployment workflow gives engineering managers a clear view of failed releases. "
        "Your team can compare the change history, find the owner and restore the last working "
        "version from one screen. Would a short demonstration next Tuesday help you evaluate it?"
    )
    messages = [f"Jordan, {base}", f"Morgan, {base} " * 3, f"Taylor, {base} " * 9]
    rows = []
    for index in [0, 1, 2, 1]:
        started = time.perf_counter()
        predictions = engine.score_text(messages[index])
        assert predictions.ndim == 2 and predictions.shape[1] == 20484
        assert np.isfinite(predictions).all()
        row = {"case": index, "wall_seconds": round(time.perf_counter() - started, 4),
               "metrics": engine.last_score_metrics()}
        row["feature_cache_entries"] = len(engine.get_model().data.text_feature.infra.cache_dict)
        if args.compare:
            assert row["feature_cache_entries"] == 0
            reference = np.load(args.compare / f"case-{index}.npy")
            assert reference.shape == predictions.shape
            row["max_abs_delta"] = float(np.abs(reference - predictions).max())
            row["prediction_parity"] = bool(np.allclose(reference, predictions, rtol=1e-4, atol=1e-5))
            assert row["prediction_parity"], row
        np.save(args.output / f"case-{index}.npy", predictions)
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "rows": rows}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    engine.unload_model()


if __name__ == "__main__":
    main()
