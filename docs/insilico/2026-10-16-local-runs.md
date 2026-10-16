# WS08 local GPU execution — 2026-10-16

Stage 0 is complete for both encoders. Core suites and WS11 follow in the specified order.

## Stage 0: measured results

Each encoder ran all 405 frozen rows: 150 paraphrases (30 families × five), 150 independent pitches, 30 neutral controls, 60 structure controls, and 15 fresh retests. There were zero exclusions, retries or fallback runs. All backend key-set checks passed. The reference corpus has 300 pitches.

| Encoder | Evidence ICC(1) | σ within | MDD (points) | Length R² | Coherent minus scrambled (CI95) |
|---|---:|---:|---:|---:|---|
| meta-llama/Llama-3.2-3B | 0.724047 | 3.430644 | 9.509259 | 0.404623 | 4.434933 (2.014466, 6.873562) |
| NousResearch/Hermes-3-Llama-3.2-3B | 0.779176 | 3.158212 | 8.754117 | 0.501914 | 4.509133 (1.588322, 7.408270) |

All ten pairwise retest relative L2 values per encoder are zero in the runner’s stored float16 matrices. Retests bypassed both built-in and offline reuse. This is a determinism sanity check; paraphrase ICC and σ provide the noise floor.

The dated per-encoder [Llama](2026-10-16-stage0-meta-llama__Llama-3.2-3B.md) and [Hermes](2026-10-16-stage0-NousResearch__Hermes-3-Llama-3.2-3B.md) reports give all 46 quantities, their ICC/CI95, σ, MDD, length R² and residual SD. Adjacent JSONs also retain coefficients, residual p05/p25/p50/p75/p95, all four confound R² values and joint R², neutral differences, and language ROI structure controls.

Hermes evidence score is length-confounded (R²=.501914). The measured files supply length-residual scales rather than discarding this result.

| Encoder | Shadow Spearman ρ | ICC baseline → ROI | Paraphrase rank changes | Half-weight warning |
|---|---:|---|---:|---|
| meta-llama/Llama-3.2-3B | 0.980617 | 0.724047 → 0.720339 | 5.667% | False |
| NousResearch/Hermes-3-Llama-3.2-3B | 0.994364 | 0.779176 → 0.771179 | 2.667% | False |

0e was recomputed from the saved matrices after installing the measured scales, without another GPU run. Neither ICC decrease exceeds .1. No global ROI-weight change is recommended by that rule.

## Product consumption

Both `roi_reference_<encoder-slug>.json` files pass `engine.roi_reference` and match the shared atlas `c5ca6bb873d2e0d1`, direct mode and event version `direct_v1`. The product tests check 300 reference pitches, 30 paraphrase families, MDD consistency, the real calibration provenance, the noise-floor preference and all check multipliers. The old 64-pitch unit artifacts and default floors remain available as fallbacks.

| Encoder | ROI_Z_SCALE | mPFC selection MDD | Why/how guard MDD |
|---|---:|---:|---:|
| meta-llama/Llama-3.2-3B | 0.572743738 | 0.018820053 | 0.027807662 |
| NousResearch/Hermes-3-Llama-3.2-3B | 0.589846337 | 0.021582221 | 0.027734041 |

Calibration provenance is `unit_calibration:stage0:<date>:<revision>`, with `basis=length_model_z` and `n_cases=300`. Selection floors use `paraphrase_noise_floor`, ahead of `default_prior`. Legacy neural ties now use the measured evidence MDD (Llama 9.509259 points; Hermes 8.754117), ahead of the provisional 1-point floor. Literature-prior tags and K8’s separate encoder factor remain in place. These scales describe synthetic text predictions and do not establish human persuasion effectiveness.

## Runtime and bookkeeping

All TRIBE inference used the local RTX 3070, one process at a time, the supplied venv-gpu and encoder env files, with PYTHONPATH pointing at this worktree. Llama is checkpoint-aligned primary; Hermes remains the production-default robustness encoder. No production host or paid API was contacted. The full GPU-process wall monitor enforces 14,400 seconds including load and CPU time while each GPU process exists; Stage 0 consumed 133.35 minutes.

Raw float32 NPZ caches and metadata are under the authorized scratchpad’s `gpu/runs/cache/<encoder-slug>/`; the committed runner’s float16 NPY audit matrices, row hashes, runtime fields and logs are in its per-run directories. Float16 reanalysis has storage quantization. Non-retest predictions are reusable; retests are fresh. The cache assumes the frozen HF snapshots and must be cleared if those revisions change.

The scratch-only adapter checks frozen input bytes against the supplied HEAD tree through filesystem reads, replacing the runner’s git-command/whole-tree-clean preflight. Generated results and test adjustments may be uncommitted while inference proceeds. It does not change stimuli, ROI definitions, statistics, thresholds or verdict rules. All frozen WS08 JSON/JSONL hashes still match the initial snapshot and committed HEAD. The run anchor is `05f42df417bf3e92dfc467d67c72cda19125153f`.

Adapter versions actually used are retained outside git: v1 SHA `7ef53a9073d67717277da9f5f29bd4e329674e4b98d1858e0b4d1dc09ca67776` (Llama Stage 0), v2 SHA `3c7bfb5716961483d7a71eb259508a216acab1c1dd352e2d48aff37e3b5707c0` (Hermes Stage 0 and subsequent suites). Both original runner/stat hashes are in the reports. This native venv run has no container image digest; its provenance says so. Exact installed/runtime versions remain in the supplied `gpu/runtime-versions.json` and package manifests.

## Checks at the Stage 0 boundary

Python: 568 passed, 4 skipped, 1 existing Starlette/httpx deprecation warning. Five initial failures were tests assuming measured references were absent; their provisional-fallback setup is now explicit, while the added tests exercise actual shipped references. No product source logic changed. Full cross-language verification will run again after suites and WS11. Existing workers need a restart to clear reference caches; no service was restarted or deployed.
