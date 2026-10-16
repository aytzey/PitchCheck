"""GPU-only WS11 runner. Never run on production; see ws11_runbook.md."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from unittest.mock import patch

import numpy as np

try:
    from .fidelity_lib import ROOT, PREREG, N_VERTICES, outside_repo, sha256, write_json
except ImportError:
    from fidelity_lib import ROOT, PREREG, N_VERTICES, outside_repo, sha256, write_json

WORD_RE = re.compile(r"\b[\w'-]+\b")
LOCK_FIELDS = ("id", "condition", "sha256", "n_words", "pair_id")


def committed_file(path):
    """Do not infer measured results against an uncommitted or changed registration."""
    relative = Path(path).resolve().relative_to(ROOT)
    saved = subprocess.check_output(["git", "show", f"HEAD:{relative.as_posix()}"], cwd=ROOT)
    if saved != Path(path).read_bytes():
        raise ValueError(f"Commit registration/ID hashes before inference: {relative}")


def load_manifest(path):
    manifest = json.loads(outside_repo(path).read_text())
    items = manifest["items"]
    if not isinstance(items, list) or not items:
        raise ValueError("Manifest needs items")
    seen = set()
    for row in items:
        item_id = row.get("id", "")
        if not isinstance(item_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", item_id) or item_id in seen:
            raise ValueError("Item IDs must be unique safe filenames")
        seen.add(item_id)
        text_path = outside_repo(row["text_file"])
        if sha256(text_path) != row["sha256"] or len(WORD_RE.findall(text_path.read_text())) != row["n_words"]:
            raise ValueError(f"Stimulus hash/word count mismatch: {item_id}")
        if row["condition"] not in {"D", "S", "long", "short"}:
            raise ValueError("Unknown condition")
    return manifest


def check_registration(manifest, lock_path):
    committed_file(PREREG)
    committed_file(lock_path)
    locked = json.loads(Path(lock_path).read_text())
    expected = {"stimuli_set": manifest["stimuli_set"],
                "items": [{k: row[k] for k in LOCK_FIELDS if k in row} for row in manifest["items"]]}
    if locked != expected:
        raise ValueError("Manifest differs from committed stimulus ID/hash lock")


def timed_events(events, word_times):
    events = events.copy()
    words = events["type"] == "Word"
    times = np.asarray(word_times, dtype=float)
    if (times.shape != (int(words.sum()), 2) or not np.isfinite(times).all()
            or (times[:, 0] < 0).any() or (times[:, 1] <= 0).any() or (np.diff(times[:, 0]) < 0).any()):
        raise ValueError("word_times must match tokenizer: ordered [start, duration] seconds")
    events.loc[words, "start"] = times[:, 0]
    events.loc[words, "duration"] = times[:, 1]
    events.loc[words, "stop"] = times.sum(1)
    end = float(times.sum(1).max())
    events.loc[events["type"] == "Text", ["start", "duration", "stop"]] = [0, end, end]
    return events


def predict(engine, text, word_times=None):
    events = engine.build_direct_text_events(text)
    if word_times is not None:
        events = timed_events(events, word_times)
    max_stop = float(events["stop"].max())
    with patch.object(engine, "build_direct_text_events", return_value=events):
        matrix, _ = engine._score_text_once(text, retry_index=0)
    if matrix.shape != (math.ceil(max_stop), N_VERTICES) or not np.isfinite(matrix).all():
        raise ValueError(f"Unexpected TRIBE prediction geometry: {matrix.shape}")
    if len(engine.get_model().data.text_feature.infra.cache_dict):
        raise ValueError("Feature cache was not cleared by the product inference path")
    return matrix


def hf_sha(repo):
    from huggingface_hub import try_to_load_from_cache
    for name in ("config.yaml", "config.json", "model.safetensors", "pytorch_model.bin"):
        cached = try_to_load_from_cache(repo, name)
        if isinstance(cached, str) and "snapshots" in Path(cached).parts:
            parts = Path(cached).parts
            return parts[parts.index("snapshots") + 1]
    raise ValueError(f"Cannot identify cached HF snapshot for {repo}")


def provenance(engine, image_digest):
    dist = importlib.metadata.distribution("tribev2")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    commit = direct.get("vcs_info", {}).get("commit_id")
    if not commit:
        raise ValueError("tribev2 direct_url.json must identify its VCS commit")
    return {"text_model": engine.TRIBE_TEXT_MODEL, "tribe_hf_sha": hf_sha(engine.TRIBE_MODEL_ID),
            "text_hf_sha": hf_sha(engine.TRIBE_TEXT_MODEL), "tribev2_commit": commit,
            "packages": {name: importlib.metadata.version(name) for name in ("neuralset", "torch", "transformers")},
            "image_digest": image_digest, "pitchcheck_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "direct_text_event_version": getattr(engine, "DIRECT_TEXT_EVENT_VERSION", None),
            "date": datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--mode", choices=("product", "real"), default="product")
    args = parser.parse_args()
    output = outside_repo(args.output)
    manifest = load_manifest(args.manifest)
    check_registration(manifest, args.lock)
    sys.path.insert(0, str(ROOT))
    from tribe_service import engine
    if engine.TRIBE_ALLOW_MOCK or engine.TRIBE_TEXT_INPUT_MODE != "direct" or engine.TRIBE_TEXT_MODEL != "meta-llama/Llama-3.2-3B":
        raise ValueError("Research requires real direct-text Llama-3.2-3B")
    import torch
    if not torch.cuda.is_available():
        raise ValueError("Run on a separate research GPU")
    output.mkdir(parents=True, exist_ok=True)
    try:
        engine.get_model()
        if engine._text_feature_metadata()["text_feature_compatible"] is not True:
            raise ValueError("Loaded text encoder differs from the checkpoint")
        meta = provenance(engine, args.image_digest)
        for index, row in enumerate(manifest["items"]):
            text = Path(row["text_file"]).read_text()
            times = row["word_times"] if args.mode == "real" else None
            matrix = predict(engine, text, times)
            if index == 0:
                second = predict(engine, text, times)
                delta = float(np.linalg.norm(matrix - second) / max(np.linalg.norm(matrix), 1e-12))
                if delta > .01:
                    raise ValueError(f"Repeat relative L2 exceeds .01: {delta}")
                meta["repeat_relative_l2"] = delta
            dest = output / f"{row['id']}.npy"
            np.save(dest, matrix.astype(np.float32))
            row.update(prediction_file=str(dest), prediction_sha256=sha256(dest))
        manifest.update(provenance={**meta, "timing": "pitchcheck_direct_0.40s" if args.mode == "product" else "real_word_times"},
                        prereg_sha256=sha256(PREREG), stimulus_lock_sha256=sha256(args.lock))
        write_json(output / "predictions.json", manifest)
    finally:
        engine.unload_model()


if __name__ == "__main__":
    main()
