# WS11 offline runbook

The [2026-10-16 measured HV report](ws11_hv2025_2026-10-16.md) is complete:
primary Spearman r=-0.01700766, p=.54534547, category-faithful verdict
`uninformative`. All 10,000 label permutations and full LOO rebuilds ran. The
positive-r template condition failed, so no C2 template was installed.

WS07 is merged. The current run uses its shared 17-network atlas (primary mask
12,443 vertices), not the historical scratch 7-network fallback (11,503).
The [analysis registration](ws11_prereg.md) is unchanged. The
[reviewed ID/hash lock](ws11_stimuli_category_faithful.json) was committed in
`9b5a8f78afb40c812d0f210f2052c67bdc5a6b8e` before the first prediction.
Inputs were the existing public NeuroVault cache with reverified hashes;
managed-browser refresh downloads timed out. Predictions used local RTX 3070,
checkpoint-aligned Llama, native venv-gpu, and direct product timing. The local
adapter replaces git-command checks with exact committed-blob checks and records
its hash; no container image digest is invented. Raw inputs, NPZ caches,
predictions and result JSON remain in the authorized scratchpad's `gpu/runs/`.

[Lebel 11.2 is skipped](ws11_lebel_2026-10-16.md) under this job's approximately
one-hour usable-input condition. Public custom HDF5 derivatives exist, but no
ready run-wise MNI/fsaverage5 bundle was obtained. There is no measured
`roi_fidelity.json`; runtime factors remain unity and labels not_benchmarked.
This track monitors literature priors and never gates their release.

## CPU target preparation (no inference)

Use a venv **outside the repo** and install only
`scripts/research/requirements.txt`. Set `PITCHCHECK_RESEARCH_DATA` to an external
directory. The service/test requirements are unchanged. Download the four public
ASCII surfaces into that directory:

`https://neurovault.org/media/images/17520/{886850,886851}.{L,R}.func.gii`

Supply WS07's shared atlas if available, otherwise the two scratch
Schaefer-400 7N fsaverage5 `.annot` files:

```sh
python scripts/research/fidelity_lib.py --selftest
python scripts/research/fidelity_hv2025.py --selftest
python scripts/research/fidelity_hv2025.py \
  --surfaces "$PITCHCHECK_RESEARCH_DATA/886850.L.func.gii" "$PITCHCHECK_RESEARCH_DATA/886850.R.func.gii" \
  --inverse "$PITCHCHECK_RESEARCH_DATA/886851.L.func.gii" "$PITCHCHECK_RESEARCH_DATA/886851.R.func.gii" \
  --annotations "$PITCHCHECK_RESEARCH_DATA/lh.400_7n.annot" "$PITCHCHECK_RESEARCH_DATA/rh.400_7n.annot" \
  --output "$PITCHCHECK_RESEARCH_DATA/target-check"
```

This produces only target coverage/sign checks, hashes and `pending_gpu`.
No labels, template, fidelity asset or inferential verdict are generated.
Review observed versus expected counts (1,743 medial-wall, 17,934 valid).
An inverse-sign failure stops the analysis.

## Stimulus lock and separate research GPU

Recover Appendix 1 if available. Otherwise author the category-faithful set
described in the registration; independently review its category/outcome clauses.
Use WS08 S9's tone pairs as a separate secondary manifest when available.
WS08 is merged; `tribe_predict.py` supplies this track's product inference path. No production host or paid API is contacted.

An external manifest is JSON with `stimuli_set` and `items`. Each item has a
safe unique `id`, `condition` (`D`, `S`, `long`, `short`), `sha256`, `n_words`,
and absolute external `text_file`. `tone_pairs` adds `pair_id`. Real-word timing
adds `word_times: [[start_seconds, duration_seconds], ...]`, relative to stimulus
start (the +10-s fMRI offset belongs to later alignment). UTF-8 file bytes are
hashed exactly, including any trailing newline. English tokenization matches
`engine._WORD_RE`. No stimulus text enters git.

Commit the registration and the stripped ID/hash lock described in the protocol
**before** inference. Both GPU inference and measured analysis verify the exact
committed files. The 2026-10-16 run committed the reviewed lock before inference; later runs must do the same.

```sh
TRIBE_ALLOW_MOCK=0 TRIBE_TEXT_INPUT_MODE=direct \
TRIBE_TEXT_MODEL=meta-llama/Llama-3.2-3B \
python scripts/research/tribe_predict.py \
  --manifest "$PITCHCHECK_RESEARCH_DATA/stimuli.json" \
  --lock docs/validation/ws11_stimuli_category_faithful.json \
  --output "$PITCHCHECK_RESEARCH_DATA/product" \
  --image-digest 'sha256:ACTUAL_RESEARCH_IMAGE_DIGEST'
```

Requires checkpoint/encoder weights, compatible loaded metadata, VCS/package/HF
provenance and a separate GPU. `_score_text_once` retains the product's feature
cache cleanup; the bounded prediction cache is bypassed. The first item runs
twice and must have relative L2 <= .01. `--mode real` uses Word start/duration
replacement without changing lexical context. Output geometry is checked against
ceil(max stop), 20,484 vertices; all values must be finite and the feature cache
empty. Never pass real-word predictions to the HV product analysis.

Then repeat the target command with:

```sh
  --manifest "$PITCHCHECK_RESEARCH_DATA/product/predictions.json" \
  --lock docs/validation/ws11_stimuli_category_faithful.json
```

CLI null counts are fixed: 10,000 unpaired permutations or 32,768 paired flips.
LOO rebuilds each training-fold template and repeats that procedure under the
same null. This CPU analysis can take considerably longer than GPU inference.
Reports include three masks, both correlations, sign agreement, raw means,
length adjustment, long–short control, p values and LOO AUC/p. A rank-deficient
length design is explicitly `non_identifiable`. Template output requires r > 0;
a nonpositive rerun removes task-owned old template files in that output dir.

Copy only the reviewed Markdown report to `docs/validation/ws11_hv2025_DATE.md`.
Keep result JSONs, inputs, texts, predictions and per-parcel tables outside git.
A positive-r result may supply the float16 template and sidecar to
`tribe_service/roi_assets/`; a future asset PR reviews attribution and hashes.
The sidecar uses "Not a mental-state measure" to obey the asset's prohibited-word
test while retaining the plan's intended scope disclaimer.

## 11.2 and runtime

11.2 remains **pre-registered, not executed**. Its input-format and timing gaps
are detailed in the dated availability report. Do not substitute native masked
HDF5 aggregates or the training repeated story for the registered run-wise
MNI/surface validation data. The deferred analysis runner and measured asset
were not fabricated around incompatible inputs.

The integrated fidelity overlay's stdlib scaling functions live in
`tribe_service/roi_fidelity.py`, not a replacement `roi_features.py`.
`_augment_persuasion_evidence` calls `annotate_fidelity` before shared scoring
and refinement consumers; it binds WS07's existing same-module
`fidelity_factor` hook. `roi_features.roi_multipliers` then multiplies WS08 ×
WS11 once, including locations. K4's denominator and K8's separate encoder
factor remain WS07's responsibility. Integrated WS09/WS12 consumption tests and the reserved-hook contract check
cover this merged implementation.

Missing atlas/asset or rejected metadata => unity scaling, `not_benchmarked`,
unchanged scores. Missing rows also use unity; measurable rows use
clip((r/ceiling)/.3, .25, 1), unmeasurable => unity. `on` applies it;
`shadow`/`off` report the proposed `would_be` without applying it. Unknown modes
fail back to off. Assets are cached per process: deploy/restart after changing
one. Provenance must match WS07's atlas, checkpoint encoder and product timing.
Hermes still applies the Llama-measured factor and reports encoder_match false;
its K8 factor is additional. A first measured asset selects methodology v2.4;
without one the current methodology is unchanged.

`FmriOutput.roi_summary` is optional so additive row annotations survive Pydantic.
Robustness includes mode, asset presence/hash and encoder match; shadow values
are report-only. Prompt additions contain categorical fidelity and applied
weights, never raw ROI/r/ceiling values. The WS07 UI/normalizers are merged. This execution adds no Rust, TS or mobile
behavior; no measured Lebel labels are available for badges.

For a measured asset PR, run all Python tests (the optional shipped-asset schema
check becomes mandatory automatically), review on/shadow median and p95 score
deltas on the WS08 reference corpus and include the measured 11.2 report.
