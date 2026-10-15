"""Run inside the GPU image: python scripts/benchmark-runtime.py --output /audit/after.

Use separate, empty TRIBE_CACHE_DIRs for before/after and the same GPU/thread limits.
The fixed synthetic pitches contain no customer data. --compare checks prediction parity.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import resource
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.cwd()))
from tribe_service import engine, roi_features
from tribe_service.persuasion_features import ROI_PRIOR, roi_prior_from_summary
from tribe_service.roi_features import unit_calibration


def load_corpus(path):
    messages = json.loads(path.read_text())
    if (not isinstance(messages, list) or len(messages) < 2
            or any(not isinstance(text, str) or not text.strip() for text in messages)
            or len(set(messages)) != len(messages)):
        raise ValueError("Corpus must be a JSON list of at least two distinct, non-empty texts")
    return messages



def default_floors(rows, encoder, run_id):
    """Provisional spatial/retest dead zone, never a paraphrase noise estimate."""
    unique = {row["case"]: row["roi_summary"] for row in rows}
    sd = float(np.median([summary["cortex_sd_signed"] for summary in unique.values()]))
    repeat = [row["roi_summary"] for row in rows if row["case"] == 1]
    keys = ("mpfc_falk2010", "ipl_ifg_how_gt_why_vezich2017_l")
    # ponytail: old benchmarks overwrote repeats; use spatial units until a retained-retest run replaces these floors.
    retest = {key: abs(repeat[-1]["locations"][key]["minus_cortex"] - repeat[0]["locations"][key]["minus_cortex"])
              for key in keys} if len(repeat) >= 2 else {}
    if len(unique) < 2 or not np.isfinite(sd) or sd <= 0:
        raise ValueError("Default floors need distinct, non-degenerate real predictions")
    return {"schema_version": 1, "atlas_sha256": next(iter(unique.values()))["asset_sha256"],
            "event_version": engine.DIRECT_TEXT_EVENT_VERSION, "text_input_mode": engine.TRIBE_TEXT_INPUT_MODE,
            "encoders": {encoder: {"floors": {key: max(0.25 * sd, 10 * retest.get(key, 0)) for key in keys},
                                   "floor_source": "default_prior", "median_cortex_sd_signed": sd,
                                   "test_retest_abs_delta": retest or None, "n_cases": len(unique),
                                   "provenance": f"benchmark_spatial_units:{run_id}" + (";retest_not_preserved" if not retest else "")}}}


def selftest():
    summary = lambda value: {"asset_sha256": "test", "rois": {key: {"z": value} for key in ROI_PRIOR}}
    rows = [{"case": i, "roi_summary": summary(v)} for i, v in enumerate((-1, 0, 2))]
    result = unit_calibration([*rows, rows[0]], "test-encoder", "selftest")
    assert set(result["centers"].values()) == {0.0} and result["n_cases"] == 3
    expected = np.std([-1, 0, 2]) * abs(sum(s * w for s, w, _, _ in ROI_PRIOR.values())) / sum(w for _, w, _, _ in ROI_PRIOR.values())
    assert np.isclose(result["ROI_Z_SCALE"], expected)
    assert result["provenance"] == "unit_calibration:selftest"
    summaries = [{"case": i, "roi_summary": {"asset_sha256": "test", "cortex_sd_signed": 0.4,
                 "locations": {key: {"minus_cortex": v} for key in ("mpfc_falk2010", "ipl_ifg_how_gt_why_vezich2017_l")}}}
                 for i, v in ((0, 0), (1, 0.2), (2, 0.3), (1, 0.22))]
    floors = default_floors(summaries, "test", "selftest")["encoders"]["test"]
    assert all(np.isclose(v, 0.2) for v in floors["floors"].values())
    assert "retest_not_preserved" not in floors["provenance"]
    assert default_floors(summaries[:3], "test", "selftest")["encoders"]["test"]["test_retest_abs_delta"] is None
    print("ROI unit calibration selftest: passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--corpus", type=Path, help="JSON list of distinct texts; default is the timing benchmark")
    parser.add_argument("--run-id", help="Provenance identifier for the encoder-specific unit artifact")
    parser.add_argument("--roi-floors-from", type=Path, help="CPU-only: reduce saved benchmark matrices; no models loaded")
    parser.add_argument("--encoder", help="Required with --roi-floors-from; must match the recorded benchmark runtime")
    parser.add_argument("--selftest", action="store_true", help="CPU-only unit-artifact check; loads no models")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if args.output is None:
        parser.error("--output is required unless --selftest is used")
    if args.roi_floors_from:
        metrics = json.loads((args.roi_floors_from / "metrics.json").read_text())
        rows, hashes = [], {}
        for path in sorted(args.roi_floors_from.glob("case-*.npy")):
            case = int(path.stem.split("-")[1])
            recorded = next(row["metrics"]["runtime"]["text_model"] for row in metrics["rows"] if row["case"] == case)
            if not args.encoder or recorded != args.encoder:
                parser.error("--encoder must match the saved benchmark runtime")
            rows.append({"case": case, "roi_summary": roi_features.roi_summary(np.load(path, allow_pickle=False))})
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        artifact = default_floors(rows, args.encoder, args.run_id or args.roi_floors_from.name)
        artifact["encoders"][args.encoder]["prediction_sha256"] = hashes
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "roi_default_floors.json").write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
        return
    messages = load_corpus(args.corpus) if args.corpus else None
    import torch
    assert not engine.TRIBE_ALLOW_MOCK, "Unit calibration requires real predictions"
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
    if messages is None:
        messages = [f"Jordan, {base}", f"Morgan, {base} " * 3, f"Taylor, {base} " * 9]
    # Timing constants in engine mirror the checkpoint config; fail loudly if upstream drifts.
    data_cfg = engine.get_model().data
    tribe_cfg = {"neuro_offset": data_cfg.neuro.offset, "tr": data_cfg.TR,
                 "text_hz": data_cfg.text_feature.frequency, "window_trs": data_cfg.duration_trs}
    assert tribe_cfg == {"neuro_offset": engine.TRIBE_HEMODYNAMIC_LAG_SECONDS, "tr": engine.TRIBE_TR_SECONDS,
                         "text_hz": engine.TRIBE_TEXT_FEATURE_HZ, "window_trs": engine.TRIBE_WINDOW_TRS}, tribe_cfg
    rows = []
    for index in (range(len(messages)) if args.corpus else [0, 1, 2, 1]):
        started = time.perf_counter()
        predictions = engine.score_text(messages[index])
        assert predictions.ndim == 2 and predictions.shape[1] == 20484
        assert np.isfinite(predictions).all()
        if engine.TRIBE_TEXT_INPUT_MODE == "direct":
            assert engine.direct_text_segment_words(
                messages[index], predictions.shape[0], engine.DIRECT_TEXT_TIMING_BASIS
            ) is not None, predictions.shape
        row = {"case": index, "words": len(engine._WORD_RE.findall(messages[index])),
               "wall_seconds": round(time.perf_counter() - started, 4),
               "metrics": engine.last_score_metrics()}
        row["feature_cache_entries"] = len(engine.get_model().data.text_feature.infra.cache_dict)
        row["roi_summary"] = roi_features.roi_summary(predictions)
        assert isinstance(row["roi_summary"], dict), "ROI asset unavailable or degenerate prediction"
        row["roi_prior"] = roi_prior_from_summary(row["roi_summary"],
                                                engine.TRIBE_TEXT_MODEL == engine.TRIBE_CHECKPOINT_TEXT_MODEL,
                                                encoder=engine.TRIBE_TEXT_MODEL, word_count=row["words"])
        if args.compare:
            assert row["feature_cache_entries"] == 0
            reference = np.load(args.compare / f"case-{index}.npy")
            assert reference.shape == predictions.shape
            row["max_abs_delta"] = float(np.abs(reference - predictions).max())
            row["prediction_parity"] = bool(np.allclose(reference, predictions, rtol=1e-4, atol=1e-5))
            assert row["prediction_parity"], row
        repeats = sum(previous["case"] == index for previous in rows)
        filename = f"case-{index}" + (f"-repeat-{repeats}" if repeats else "") + ".npy"
        np.save(args.output / filename, predictions)
        rows.append(row)
        print(json.dumps(row), flush=True)
    calibration = unit_calibration(rows, engine.TRIBE_TEXT_MODEL, args.run_id or args.output.name)
    calibration.update({"created_at_utc": datetime.now(timezone.utc).isoformat(), "gpu": torch.cuda.get_device_name(),
                        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes() if args.corpus else
                                                       json.dumps(messages).encode()).hexdigest()})
    slug = engine.TRIBE_TEXT_MODEL.replace("/", "__")
    (args.output / f"roi_unit_calibration_{slug}.json").write_text(json.dumps(calibration, indent=2) + "\n")
    (args.output / "roi_default_floors.json").write_text(json.dumps(
        default_floors(rows, engine.TRIBE_TEXT_MODEL, args.run_id or args.output.name), indent=2, allow_nan=False) + "\n")
    result = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "encoder": engine.TRIBE_TEXT_MODEL, "roi_unit_calibration": calibration,
              "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "tribe_config": tribe_cfg, "rows": rows}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    engine.unload_model()


if __name__ == "__main__":
    main()
