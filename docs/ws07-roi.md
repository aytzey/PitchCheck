# WS07: signed cortical literature priors

Implemented plan steps 1–11, K2/K3/K5/K8 and C.WS07. Step 11 now includes real local RTX 3070 unit assets for both encoders and the run record below. WS09 09-A/B/C1, WS12 scoring/contrasts and WS08/WS11 research jobs are separate work. No Rust scoring changes or release packaging are included.

The single builder consumes `scripts/roi_peaks.csv`, transcribed from the authoritative plan's PDF-image curation and prototype. It fetches the four CBIG annotations and MIT license at commit `634f676630929a71297852d01dd92a287103e861`, uses nibabel 5.4.2 / nilearn 0.14.1, and records their hashes, versions, coordinate transforms and every peak decision. Meshes are not redistributed. Runtime uses only numpy and stdlib, with no downloads or provenance-JSON reads.

Create a disposable environment, then build twice (use paths in your research scratch directory):

```sh
python3 -m venv "$SCRATCH/venv-roi"
"$SCRATCH/venv-roi/bin/python" -m pip install nibabel==5.4.2 nilearn==0.14.1 numpy
"$SCRATCH/venv-roi/bin/python" scripts/build_roi_assets.py --cache-dir "$SCRATCH/roi-build-cache"
"$SCRATCH/venv-roi/bin/python" scripts/build_roi_assets.py --cache-dir "$SCRATCH/roi-build-cache" \
  --output "$SCRATCH/roi-build-repeat" --ui-output "$SCRATCH/roi-centroids-repeat.json"
```

The two executed builds produced identical NPZ, provenance, license and UI centroid bytes. Committed NPZ SHA-256: `c5ca6bb873d2e0d140a83dd94ae30c2c1dfca2959e6b50a7ae6ed019424bfe0d` (23,239 bytes). The runtime SHA field is its first 16 hex characters. The nine non-overlapping parcel unions match the prototype: frontopolar 77, medial orbitofrontal 142, dorsomedial prefrontal 191, insula 93, right inferior frontal/anterior insula 100, posterior cingulate/precuneus 169, right temporoparietal junction 39, lateral temporal 264, left inferior frontal 76 vertices. All 13 requested location keys are present; they intentionally overlap.

The authoritative build records these geometry caveats:

- Right inferior frontal/anterior insula assignments differ between affine and no-affine paths: `RH_DefaultB_PFCv_1` and `RH_SalVentAttnB_Ins_1`; both are retained.
- Posterior cingulate/precuneus assignments differ on both hemispheres; all four parcels are retained.
- Left posterior superior temporal sulcus has a network mismatch (`LH_SomMotB_Aud_2`); the affine path selects the nearest allowed `LH_TempPar_3` vertex at 6.57771 mm. The no-affine alternative is 8.57778 mm away and is dropped.
- Cooper's unpublished `(0,-42,21)` centroid maps to the medial wall in both paths and is excluded. Subcortical/excluded and null/benchmark records never contribute selected parcels.
- The strict location rule drops the right dorsomedial prefrontal Chua 2011 peak at **6.09073 mm**, leaving the accepted left 22 vertices. The plan's approximate 39-vertex preview is not hardcoded. Left inferior frontal Chua 2011 has 47 vertices and ventromedial prefrontal Chua 2009 has 54; the builder JSON is authoritative.

`load_roi_assets()` is the single loader. It validates index types, bounds, hemisphere layout, medial wall, parcel count and non-overlap; missing/corrupt assets log a warning and disable overlays. Signed time means are referenced to labelled cortex, excluding the medial wall. ROI z is invariant to positive scaling and constant offsets. The reducer feeds Python and native analysis paths identically.

The six signed priors use the fixed K4 denominator. Compatible Llama gets weight 0.15; incompatible encoders get 0.075 by default. Robustness publishes the term, provenance, anchor-point shadow, total-prior counterfactual and K5 budget diagnostics. Legacy refine automatically uses the adjusted anchor and publishes `roi_prior_score`. Jev's whitelist and selection policy remain unchanged; neither Jev nor the legacy critic receives raw ROI fields. Only the evaluator's explicitly allowed WS07 block and deterministic synthesis contain ROI values. Desktop refine advice carries the ROI lever/citation without numeric observations.

Desktop normalizers preserve signed `rois`/`locations` and optional WS08/WS11 badges. The ROI row appears when summary/prior/points fields exist. Brain points use the builder's actual parcel-union centroids projected onto the existing top-down schematic; with absent fields, neutral whole-cortex codes remain. Both robustness and variant ranking share “About these signals”. Mobile UI is unchanged.

## Unit calibration run: 2026-10-16

WS07 item 11 is complete for both encoders on the local **NVIDIA GeForce RTX 3070 (8 GiB)**. The date is Europe/Istanbul (UTC+03); artifact timestamps use UTC. Both runs used real TRIBE v2, the documented offline GPU environment and memory-clamping launcher, `PITCHCHECK_REPO` set to this worktree, direct text mode, batch size 1 and a 7 GiB allocator cap. Runs were sequential. All 128 saved matrices are finite float32 with 20,484 vertices; timing-config assertions and text/segment mapping checks passed. Every case used CUDA with the Accelerate text device, zero retries/fallbacks and zero retained feature-cache entries. Accelerate CPU offloading is part of that documented configuration.

| Encoder | Run ID (`unit_calibration:` prefix in artifact) | Unique texts | ROI_Z_SCALE | Inference seconds | Peak board MiB |
|---|---|---:|---:|---:|---:|
| `NousResearch/Hermes-3-Llama-3.2-3B` | `ws07-hermes-20261016-rtx3070` | 64 | 0.1701008503 | 416.19 | 7220 |
| `meta-llama/Llama-3.2-3B` | `ws07-llama-20261016-rtx3070` | 64 | 0.1691673434 | 424.48 | 7274 |

Base commit: `067929d98019dcc5947f56375ce093a5bf3eba8d`. Runtime: Python 3.12.3, Torch 2.7.1+cu128, TRIBE `af58661791a351a448a489042a28f6c37e1c14b7`, Transformers `9914a3641f7aaaabb0bcdfcd73a54a1cfa70c3e7`. Benchmark harness SHA-256: `0af9e802f936e06223c91955f4615c7ab184bd9c7a76a72376c3ccd9b10d107e`.

The fixed corpus is [`scripts/fixtures/roi-unit-calibration.json`](../scripts/fixtures/roi-unit-calibration.json), SHA-256 `c4f2e065dabe26cf9cc3d624883259a5e9759473ea4071ced6005e976d6593e0`. It was fixed before either run and contains no customer data:

- Cases 0–2: the original benchmark deployment pitches, unchanged (42, 126 and 378 words, including its repeated-paragraph stress case).
- Cases 3–7: five existing English fixture texts: `test_llm_layer.py` SAMPLE_MESSAGE, `test_app.py` deployment-time score fixture, TestScoreLog.MESSAGE, the ROI score-test deployment text, and `test_engine.py` cache-test pitch (11, 10, 16, 11 and 10 words).
- Cases 8–63: 56 newly authored, independent English pitches (50–57 words each): business services, retail, community invitations/fundraising, education and personal invitations, including vague and dogmatic copy. They are synthetic examples, not labelled outcomes or paraphrase families.

There are 3,592 words in total; the median pitch has 53 words. WS07 item 11 permits the benchmark route without specifying a fixed corpus. The separate WS08 Stage 0 reference corpus has not been built here; this run does not substitute for that length-adjusted reference or its validation suites.

Each center is the median signed cortical z for that ROI across the 64 unique cases. `ROI_Z_SCALE` is the population SD (`ddof=0`) of the K4 weighted, centered raw composite, with the fixed prior denominator and multipliers 1. Weights remain 0.075 for Hermes (K8 mismatch factor 0.5) and 0.15 for checkpoint-aligned Llama. These are numeric unit adjustments; both assets retain `validated: false` and the score term remains `literature_prior_unvalidated`.

| ROI center | Hermes | Llama |
|---|---:|---:|
| `vmpfc_frontopolar` | -0.18900 | -0.18595 |
| `dmpfc` | 0.55270 | 0.51775 |
| `r_ifg_ai` | -0.47110 | -0.53525 |
| `mofc` | -0.42430 | -0.44025 |
| `insula` | -0.81830 | -0.84400 |
| `pcc_precuneus` | 0.30425 | 0.23610 |

Both assets match atlas SHA `c5ca6bb873d2e0d1` (the loader uses the first 16 characters of the full NPZ SHA above) and include the corpus hash, encoder, date, GPU, sample count and `unit_calibration:<run_id>` provenance. The runtime loader required no change: matching encoder artifacts are selected, invalid/missing or unknown encoders retain provisional 0/1 defaults, and a matching WS08 reference takes precedence when available.

### Before/after score distributions

Both states use the same saved GPU matrices and default weights. **Before** means provisional centers 0 / scale 1; **after** means the encoder-specific asset. Every case had quality weight q=1. `robustness.roi_prior_points` is the one-decimal, quality-adjusted **anchor** shadow; final-score effects below are computed separately through the existing `_blend_final`, including the ±6-point K5 budget. Other literature-prior contributions are held at zero.

| Encoder / state | ROI score median [p05, p95] | ROI anchor points mean; median [p05, p95] | Final, neural-only: median Δ; p95 absolute Δ | Final, fixed LLM=50: median Δ; p95 absolute Δ |
|---|---|---|---|---|
| Hermes / before | 61.35 [35.27, 66.45] | +0.655; +0.85 [-1.08, +1.20] | +0.851; 1.251 | +0.383; 0.563 |
| Hermes / after | 55.75 [0.21, 82.20] | -0.030; +0.45 [-3.70, +2.38] | +0.431; 3.734 | +0.194; 1.680 |
| Llama / before | 61.10 [35.72, 66.22] | +1.314; +1.70 [-2.14, +2.40] | +1.665; 2.486 | +0.749; 1.119 |
| Llama / after | 52.70 [0.23, 80.58] | -0.191; +0.40 [-7.49, +4.55] | +0.405; 6.000 | +0.182; 3.368 |

The fixed-LLM=50 scenario is a deterministic counterfactual with no facets; no paid or local LLM was called. This scenario includes band clamping where it occurs. An additional counterfactual in the audit JSON sets the semantic score to each pitch’s ROI-off anchor to isolate the unclamped blend (0.45 × the anchor effect). Neural-only results are the actual no-LLM score path. Integer headline-score effects and full ranges make the rounding and tails explicit:

| Encoder / state | ROI score range | Anchor point range | Final neural-only Δ range | Final fixed-LLM=50 Δ range | Integer final Δ ranges: neural / LLM |
|---|---|---|---|---|---|
| Hermes / before | [32.1, 67.4] | [-1.3, +1.3] | [-1.343, +1.305] | [-0.604, +0.587] | [-2, +2] / [-1, +1] |
| Hermes / after | [0.1, 85.5] | [-3.7, +2.7] | [-3.743, +2.663] | [-1.684, +1.198] | [-4, +3] / [-2, +1] |
| Llama / before | [31.5, 66.8] | [-2.8, +2.5] | [-2.775, +2.520] | [-1.249, +1.134] | [-3, +3] / [-1, +1] |
| Llama / after | [0.1, 82.7] | [-7.5, +4.9] | [-6.000, +4.905] | [-4.439, +2.207] | [-6, +5] / [-4, +2] |

Fixed-LLM=50 band-clamp case counts (before → after): Hermes 0 → 0 of 64; Llama 0 → 3 of 64.

The CPU analysis also checked fixed semantic scores 0 and 100, exercising band clamping, and verified |final − counterfactual| ≤ 6 for every tested path. Calibration removes most of the corpus mean offset while increasing variation; per-ROI median centering does not force the composite median to exactly 50.

### Estimate sensitivity

A seeded 2,000-resample bootstrap (NumPy seed 7, IID texts) and leave-one-out calculation quantify sampling sensitivity; they are not outcome validation.

| Encoder | Scale | Bootstrap 95% interval | Leave-one-out range | Synthetic cases 8–63 only |
|---|---:|---|---|---:|
| Hermes | 0.170101 | [0.114856, 0.212954] | [0.156371, 0.171446] | 0.066866 |
| Llama | 0.169167 | [0.114712, 0.210384] | [0.154610, 0.170505] | 0.064393 |

The mixed corpus provides a non-degenerate scale with modest single-case influence, but its short-pitch emphasis, related benchmark stress cases and synthetic-only sensitivity limit generalization. The bootstrap intervals show remaining sampling uncertainty. WS08 can replace these assets with a broader length-adjusted reference; this run makes no claim about behavioral accuracy or measured-fMRI fidelity.

### Reproduction and audit files

Use the already prepared local runtime, sequentially, from this worktree. Sourcing either environment resets the repository path, so set both repository variables afterward:

```sh
GPU_RUNTIME=/tmp/claude-1000/-home-aytzey-Desktop-pitch-check-research-papers/18d7cf70-afea-4b81-a3ed-75ef728c2a3c/scratchpad/gpu
source "$GPU_RUNTIME/env_hermes.sh"
export PITCHCHECK_REPO="$PWD" PYTHONPATH="$PWD"
python "$GPU_RUNTIME/benchmark.py" --corpus scripts/fixtures/roi-unit-calibration.json \
  --output "$GPU_RUNTIME/ws07-unit-20261016/hermes" --run-id ws07-hermes-20261016-rtx3070
source "$GPU_RUNTIME/env_llama.sh"
export PITCHCHECK_REPO="$PWD" PYTHONPATH="$PWD"
python "$GPU_RUNTIME/benchmark.py" --corpus scripts/fixtures/roi-unit-calibration.json \
  --output "$GPU_RUNTIME/ws07-unit-20261016/llama" --run-id ws07-llama-20261016-rtx3070
```

The benchmark retains its original timing cases when `--corpus` is omitted. `--selftest` remains CPU-only. The new corpus input rejects empty, non-string and duplicate texts; calibration rejects degenerate or mixed-atlas summaries. Per-case signed ROI/location summaries, raw priors and runtime metrics are in each output `metrics.json`; saved matrices are `case-*.npy`. CPU analysis independently recomputed centers/scale from those saved summaries, reopened every matrix, checked its ROI summary and verified corpus/atlas hashes before copying the two numeric assets into `tribe_service/roi_assets/`.

Audit directory: `$GPU_RUNTIME/ws07-unit-20261016/`. Each encoder folder contains `run.log`, sampled `vram.csv`, `metrics.json`, the numeric asset, and `score-distributions.json` (all per-case counterfactuals and full descriptive statistics). `analyze.py` and `write_docs.py` are retained at the audit root. The analysis explicitly uses provisional 0/1 defaults for its before state even if assets are already installed; a future benchmark itself reports whichever unit asset it loads.

Verification: `$S/venv/bin/python -m pytest tribe_service/tests -q -rs` finished with **265 passed, 4 skipped, 0 failed**, plus one existing FastAPI/Starlette deprecation warning. The four skips are optional `exca.cachedict` and Torch checks in the CPU test venv; the separate GPU venv ran all 128 real predictions. `npm run lint` and `npx tsc --noEmit -p .` passed; `npx vitest run` finished with **52 passed, 0 failed across 9 files**; `scripts/benchmark-runtime.py --selftest` passed. Logs are retained at the audit root. The deterministic no-ROI pin remains 66.30 / 73. No Rust source was changed. Authenticated API smoke, baseline prediction `--compare`, WS08/WS11 datasets and human/outcome validation were not run; no production host or paid API was contacted. Task GPU processes and VRAM monitors exited after both runs.
