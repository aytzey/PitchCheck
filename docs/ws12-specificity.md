# WS12: category-twin specificity contrast

The runtime comparison is a signed, predicted average-reader response to the draft versus a category-level twin. It measures lexical specificity. TRIBE never sees the persona; deterministic traceability supplies `grounded`, and WS06 `personalization_level` supplies an additional soft score gate. Neither specificity nor a cortical location establishes a recipient's mental state, truth, relevance, or persuasion probability.

The implementation uses WS07's single `load_roi_assets()` loader and six `loc_*` sets. Only the dorsomedial prefrontal and precuneus Chua 2011 locations enter the specificity average. The contrast uses signed differences, relative to all cortical parcels; a positive result can represent less deactivation. It is an unvalidated literature prior on synthetic text timing, with no rest baseline. `top_parcels` also reports where the contrast actually concentrates.

The evaluator receives only categorical direction/grounding and span/category pairs. Numeric ROI values, numeric contrasts, parcel scores and the control text remain report-only.

The feature is default-on and independently reversible:

| Environment variable | Default | Effect |
| --- | --- | --- |
| `PITCHCHECK_SPECIFICITY_CONTRAST` | `1` | `0` restores the previous path with no twin LLM call or additional TRIBE run. |
| `PITCHCHECK_SPECIFICITY_SCORE_WEIGHT` | `1.5` | `0` retains measurements and the shadow score, with zero specificity score effect. |
| `PITCHCHECK_SPECIFICITY_Z_MIN` | `1.0` | Dead zone; the monitoring probe proposes paraphrase absolute-z p95. |
| `PITCHCHECK_SPECIFICITY_REFINE_PENALTY` | `2.0` | `0` disables the refine detail-retention term and its additional calls. |

The specificity score is grounded-only, at most ±3 points, scaled by K4 ROI multipliers and the K8 encoder factor, then routed through the existing K5 priors dictionary and total ±6-point budget. The API records `score_without_specificity_prior` and an available contrast or an explicit skip reason. A missing asset, unsupported language/length, failed twin, mock runtime, or busy score queue never creates a fake zero measurement. The twin only acquires an idle score slot without blocking queued primary scores.

Refine penalizes dropping verified details from an above-dead-zone positive contrast, by at most two selection points. It never rewards invented specifics. The category twin stays out of the measured candidate list, critics and Jev payloads. The desktop `SPC` / `exp` row shows the signed score contribution, the category control under “Compared with”, and the shared “About these signals” caveats. The literature-prior label makes no outcome-validation claim.

## Parallel monitoring probe (written, GPU/API run unexecuted)

`scripts/specificity-probe.py` runs outside the image and CI, with the same import arrangement as `benchmark-runtime.py`. It uses installed numpy and runtime helpers; there is no new runtime dependency. `--selftest` imports only numpy/stdlib and checks input balance, control validation, tied-rank Spearman, missing-data handling and threshold reduction. It loads no weights and makes no network requests:

```sh
python scripts/specificity-probe.py --selftest
```

The bundled `scripts/fixtures/specificity-pairs.jsonl` contains **20 synthetic pairs: 10 health, 10 sales**. Each row has `id`, `domain`, `persona`, `original`, `human_category`, `human_chua`, and exact replacement `spans` (`span`, `category`, `persona_fact`). These are monitoring examples, not study participants or published verbatim stimuli. The smoking example adapts the plan's Chua 2009 example with added logging context. `human_category` changes only the listed details and no hedges; `human_chua` adds one `may` hedge to that category wording, exposing the certainty confound. Chua 2011 Untailored sentences are not used as twins. A future paper-transcribed dataset can use the same schema and record its own stimulus provenance; do not relabel these synthetic examples as paper transcriptions.

For each pair the probe records:

1. Three independent LLM category twins at temperature 0.7, checked by the live deterministic twin validator, and their signed contrasts.
2. Three independent paraphrases at temperature 0.7, preserving the listed detail spans verbatim, concrete details, second-person count, hedge count and the live ±10% / two-word bound. Invalid paraphrases retain an explicit failure reason.
3. A hedge-only control with exactly one `perhaps` added and every other character preserved; a separate contrast for `human_chua` records the category-plus-hedge condition.
4. An equally specific, persona-mismatched control: all original numeric facts receive a different value (+7), with the same words, person, hedge count, length and number of concrete facts. It is compared with the **same category control**. The mismatch-minus-original category-contrast differences document sensitivity to lexical facts without implying recipient fit. This deterministic synthetic manipulation covers numeric mismatches; broader tool/situation mismatches require additional pre-specified fixtures.
5. LLM-versus-human-category sign agreement and tied-rank Spearman rho.
6. The evaluator model's raw `personalization_fit` judgment for the original and first available LLM twin, using the same stated-persona alignment criterion. This manipulation check receives only text/persona, no neural values, ROI names, or specificity scores. It is reported separately from neural criteria.

The pre-specified monitoring thresholds are in the script header: ≥15/20 nonzero sign agreements; Spearman rho > 0; detail-removal absolute z greater than hedge-only absolute z in ≥15/20 pairs; pooled within-pair twin SD below half the median absolute LLM-twin contrast. Larger balanced datasets retain the same 75% count thresholds. Pooled SD is the square root of the mean of per-pair sample variances; the signal median uses per-pair median LLM z. Proposed `Z_MIN` is the p95 of every accepted paraphrase absolute z. Missing controls remain missing, never z=0, and make a run incomplete.

Each `results.json` records `_text_feature_metadata()`, configured encoder, device, exact source git SHA, input/script SHA-256, per-pair results, completeness and thresholds. Hermes and Llama use separate output directories and are never pooled. Raw synthetic controls live only in the research artifact; stdout contains numeric progress only. Review the recorded SHA against the source used for the run; use a reviewed source snapshot.

## Authorized GPU run, in a separate bounded container

The following commands are **instructions only; they were not run for this implementation**. They require owner authorization for the paid OpenRouter requests. Do not run them against production, mount production auth/state, or reuse a serving container. Choose an existing GPU image, dedicated cached model directory and dedicated empty output root. The API env file should contain only the authorized OpenRouter settings, with owner-only permissions; credentials are never printed. Both encoders must already be cached; Hugging Face access stays offline.

Set `WS12_IMAGE`, `WS12_MODELS`, `WS12_OUTPUT`, and `WS12_API_ENV` to those research resources. From the repository root, this runs the two encoders **serially**, each with the existing four-CPU, 12-GiB RAM/no-additional-swap, 256-PID and 8-GiB allocator ceilings. It bind-mounts the research script rather than adding it to the image:

```sh
WS12_SOURCE="$PWD"
WS12_GIT_SHA=$(git rev-parse HEAD)
mkdir -p "$WS12_OUTPUT"
for WS12_KIND in hermes llama; do
  case "$WS12_KIND" in
    hermes) WS12_ENCODER=NousResearch/Hermes-3-Llama-3.2-3B ;;
    llama) WS12_ENCODER=meta-llama/Llama-3.2-3B ;;
  esac
  docker run --rm --init --name "pitchcheck-ws12-$WS12_KIND" \
    --gpus device=0 --cpus 4 --memory 12g --memory-swap 12g --pids-limit 256 \
    --cap-drop ALL --security-opt no-new-privileges:true \
    --mount "type=bind,src=$WS12_SOURCE,dst=/source,readonly" \
    --mount "type=bind,src=$WS12_MODELS,dst=/models" \
    --mount "type=bind,src=$WS12_OUTPUT,dst=/audit" \
    --workdir /source --env-file "$WS12_API_ENV" \
    -e TRIBE_ALLOW_MOCK=0 -e TRIBE_DEVICE=cuda -e TRIBE_TEXT_DEVICE=auto \
    -e "TRIBE_TEXT_MODEL=$WS12_ENCODER" -e TRIBE_TEXT_INPUT_MODE=direct \
    -e TRIBE_TEXT_BATCH_SIZE=4 -e TRIBE_CUDA_MEMORY_LIMIT_GB=8 \
    -e TRIBE_CACHE_DIR=/models -e HF_HOME=/models/huggingface \
    -e HUGGINGFACE_HUB_CACHE=/models/huggingface/hub -e HF_HUB_OFFLINE=1 \
    -e XDG_CACHE_HOME=/models/.cache -e TRIBE_PREDICTION_CACHE_SIZE=8 \
    -e TRIBE_PREDICTION_CACHE_MAX_BYTES=67108864 \
    -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
    -e TOKENIZERS_PARALLELISM=false -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$WS12_IMAGE" python scripts/specificity-probe.py \
    --input scripts/fixtures/specificity-pairs.jsonl \
    --output "/audit/$WS12_KIND" --git-sha "$WS12_GIT_SHA"
done
```

Output directories must not already exist. Each pair makes three twin requests (with at most one validator retry each), three paraphrase requests and two evaluator requests, and up to twelve serial TRIBE runs. Check the paid-call budget before authorizing a run. The script does not change env files, deploy services, tune code, or call PitchServer.

Run the probe within **two weeks after the first deployment**. Review each encoder's artifact independently. When all monitoring criteria are met, apply the proposed `PITCHCHECK_SPECIFICITY_Z_MIN` through env only. When criteria fail or measurements are incomplete, set `PITCHCHECK_SPECIFICITY_SCORE_WEIGHT=0` and `PITCHCHECK_SPECIFICITY_REFINE_PENALTY=0` to keep score measurements in shadow mode. These actions are monitoring responses, never a precondition for shipping.

Weekly, summarize final minus `score_without_specificity_prior`: mean delta and fraction with absolute delta >2. Track primary `/score` 429 rates and p95 latency against the pre-feature baseline. If 429s rise above baseline, or p95 exceeds half `TRIBE_SCORE_TIMEOUT`, disable `PITCHCHECK_SPECIFICITY_CONTRAST=0` as an owner operational decision. Retain only numeric monitoring fields, never control text, persona or credentials. No production monitoring data, GPU contrast, paid manipulation check or measured dead zone was collected here.
