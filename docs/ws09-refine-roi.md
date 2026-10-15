# WS09: signed refine ROI evidence

09-0 was already implemented. Steps 7–15 (09-A/B), 16 (C1), 18 and the ROI readout UI are implemented here. WS12 is separate. C2's WS11-produced `hv2025_dogmatic_minus_suggestive_tribe_fsaverage5.npy` and sidecar are absent: no template key, weight, download or substitute is added.

The WS07 loader and `roi_summary.locations` supply signed `minus_cortex` measurements for the Falk J Neurosci, Falk JoCN and Vezich locations. These remain separate from `response_features`. Deltas require the same TRIBE model, feature encoder, mode, temporal basis and vertex count. Missing/mock overlays report `roi_unavailable`; mismatches report `runtime_not_comparable`. Neither produces a readout or selection term.

Jev adds only `mpfc_falk2010`: zero through the floor, linear to saturation at twice the floor, ±0.5 points by default, scaled by minimum prediction quality, the shared K4 multiplier, and the K8 encoder factor. An opposite, above-floor Vezich delta disables it. Trace plus ROI adjustment remains within ±3. Eligibility, context band, semantic secondary ordering and the Jev exact-tie preference are unchanged. The `basis` remains `context_first_experimental_trace_tiebreak`. Policy records source/evidence/provenance, floor source, `roi_counterfactual_id` and `roi_changed_selection`; each row records `selection_score_without_roi`. Legacy scores and winners are unchanged; `roi_shadow_selected_id` measures a hypothetical additional term.

`PITCHCHECK_ROI_PRIOR_POINTS=0` disables the refine term (default 0.5; configurable ceiling 1). `PITCHCHECK_ROI_ENCODER_MISMATCH_FACTOR=0` disables it for mismatched encoders. Turkish, detected with the existing helper, uses `default_prior` floors and the same half-weight factor, applied once if encoder mismatch also holds; provenance includes `;non_english_default_floor`. Other languages retain the existing detector's limitation. Future language-specific paraphrase references require an explicit integration change.

## Provisional floors and their limit

`roi_default_floors.json` is encoder- and atlas-specific, tagged `default_prior`, in signed TRIBE units. It was reduced on CPU from the existing real, direct-text benchmark matrices in the research scratchpad (`gpu/benchmark-hermes` and `gpu/benchmark-llama`; three distinct cases each). Input SHA-256 hashes and median cortical SD are stored in the artifact. Floors are **0.024015** for Hermes and **0.024550** for Llama, for each of the two keys.

The old benchmark overwrote the first `case-1.npy` with its repeat and its metrics omitted ROI summaries. Thus the repeat delta cannot be reconstructed. The committed provisional floors use `0.25 × median cortex_sd_signed` only, with `test_retest_abs_delta: null` and explicit `;retest_not_preserved` provenance. They are not paraphrase noise estimates. This is the step-12 deviation; no repeat delta of zero or completed unit calibration is claimed.

Reproduce those provisional artifacts independently for each encoder:

```sh
python scripts/benchmark-runtime.py --roi-floors-from "$SCRATCH/gpu/benchmark-hermes" \
  --encoder NousResearch/Hermes-3-Llama-3.2-3B --output "$SCRATCH/ws09-floors-hermes"
python scripts/benchmark-runtime.py --roi-floors-from "$SCRATCH/gpu/benchmark-llama" \
  --encoder meta-llama/Llama-3.2-3B --output "$SCRATCH/ws09-floors-llama"
```

Merge their `encoders` objects into the bundled file without changing the shared atlas/runtime metadata. No model or network access occurs in this mode. The regular GPU benchmark now preserves repeated matrices, records all ROI summaries, and writes the complete `max(0.25 × median SD, 10 × repeat |delta|)` artifact. Run the existing [WS07 benchmark commands](ws07-roi.md) for each encoder and replace the provisional entries after reviewing the numeric artifacts. No GPU job was run for this change.

WS08's matching `roi_reference_<encoder.replace('/', '__')>.json` takes precedence automatically, using `quantities["roi:<location>"].noise.min_detectable_delta` and `floor_source: paraphrase_noise_floor`. Encoder, event version (`direct_v1`), input mode and atlas must match; invalid or absent data falls back safely. The existing shared loader also reads future WS08 references when its `engine.roi_reference` hook is not installed yet.

## C1 and the LLM boundary

The Falk JoCN hypothesis requires finite core/left/right location values with matching left/right signs. With no reference it uses `core.z <= -0.67`, tagged `default_prior_within_draft`. Matching WS08 quantities use signed p25: `length_buckets` keys `le60`, `61_150`, `151_240`, each containing `p25`. Channel-specific quantities under `channels[platform]` require `n >= 50`; otherwise pooled quantities apply. The WS08 length-model schema is also supported as `a + b_log_words × log(words) + resid_q.p25`. Missing reference statistics retain the provisional trigger. Hypotheses follow gaps, precede strengths, keep the five-item cap and existing synthesis `_v2` basis, and display `HYP`. `PITCHCHECK_ROI_HYPOTHESES=0` disables C1 independently of WS07's score term.

No raw ROI value, delta, z, floor, prior score or numeric ROI sort score is sent to the evaluator, either writer, legacy critic, Jev planner or Jev reviewer. The user-requested all-LLM leak boundary supersedes WS07's earlier numeric evaluator block: it now contains categorical guidance only; numeric observations are removed from WS07 synthesis and numeric sort scores are omitted from its prompt copy. The API keeps its numeric ROI evidence. C1's explicitly allowed categorical observation/lever reaches /score synthesis. The deterministic score pin remains axis 66.30 / final 73.

GPU/image smoke testing, live Jev requests, WS08/WS11 measurements and reply-outcome validation remain unrun; no production host or paid API was contacted. The one-time retained-retest benchmark is the remaining data action for the complete provisional-floor formula.

Local verification: Python **285 passed, 4 skipped, 0 failed**; Vitest **57 passed, 0 failed** across 9 files; ESLint and TypeScript passed; benchmark `--selftest` passed; `git diff --check` passed. The skipped Python cases are existing dependency-dependent checks. The 66.30 / 73 pin and LLM payload leak tests pass. Changes remain uncommitted.
