# Data-gated Faz 2 / Faz 3 remainder

State at 2026-10-16. English Stage 0 scales, WS08 shrink multipliers, Turkish
WS09-B paraphrase floors and additional parcel spins have been measured locally.
These are synthetic response measurements and literature priors. There is no
WS11 fidelity asset, no C2 template and no methodology v2.4 claim.

All commands below run from the repo root in the matching research environment.
`DATA` is an existing private directory outside Git; `DATASET_ID` identifies one
frozen dataset version; `MODEL` is an explicitly approved pinned provider model.
`SOURCE_SHA` is the reviewed 40-character commit recorded for that run.
`DOCTRINE_OLD` is an already prepared pre-WS04 Faz 0 source checkout (f0da33d),
and `DOCTRINE_NEW` is an already prepared checkout adding only WS04. Comparing
today's integrated branch against Faz 0 would mix later work packages into that
A/B. Credentials remain preconfigured in the environment and are never written
into these commands or reports. Use fresh output paths for each run.

## Outcomes and release evidence

| Item | Exact trigger | Command / next executable step | Resulting change |
|---|---|---|---|
| First real WS10 comparison | At least 200 eligible real numeric score records, matching encoder/methodology/LLM provenance and full-precision blend inputs; opt-in collection and join keys already configured | `python scripts/validate_incremental.py compare "$DATA/scores.jsonl" --dataset-id "$DATASET_ID" --output "$DATA/compare-first200"` | Record parity (absolute error ≤.05), rank association, score shifts, clamp frequency and available refine counterfactuals. This descriptive report neither fits weights nor changes flags. Synthetic 200-row replay is already complete and does not meet this trigger. |
| WS10 tune N rule | In each eligible randomized/proxy stratum: ≥80 message arms **and** ≥10 independent groups **and**, for binomial outcomes, ≥100 sends in every arm. PME instead needs its registered ≥15 ratings/message and reliability ≥.70. No observational production or synthetic data authorizes fitting | `python scripts/validate_incremental.py tune "$DATA/joined-arms.jsonl" --blind-input "$DATA/blind.jsonl" --dataset-id "$DATASET_ID" --output "$DATA/tune-v1"` | Recommendation only. Accept a winner only with positive group-CV Δ CI, Holm p<.05 across all candidates, and positive untouched final-25%-time holdout. A dated reviewed change updates only the winning constant, increments methodology and tags `calibration_basis=fitted:<dataset_id>`. |
| WS10 selection term fit | The same N rule plus randomized candidate outcomes, exact `selection_score_without_roi`, `roi_selection_unit`, `roi_prior_points`, and score parity | Add `--refine-input "$DATA/candidate-outcomes.jsonl"` to the tune command above | Selection-point recommendation remains separate from score/B1 fitting. Never fit selection effects from the fact that the current policy selected a candidate. |
| Legacy ROI term decision | Numeric WS09 shadow records plus randomized policy outcomes sufficient for the registered N rule; `roi_shadow_selected_id` alone establishes possible winner changes, not effectiveness | `python scripts/validate_incremental.py rollback_alarm "$DATA/legacy-policy-arms.jsonl" --randomized-arms --flag PITCHCHECK_ROI_PRIOR_POINTS --dataset-id "$DATASET_ID" --output "$DATA/legacy-policy-alarm"` | Decide whether to implement a bounded legacy term through a separately reviewed change, retain shadow-only behavior, or reduce/disable the implicated policy flag if its CI is wholly negative. Current legacy winners stay unchanged. The policy-arm file must contain disjoint aggregated `with_neural`/`semantic_only` outcomes per group. |
| WS05 v1 deletion | Two **released versions with v1-vs-v2 monitoring**, evidenced by release IDs/dates and saved rubric reports. A local commit or version string is not a release | Run the rubric-pairs command below for each monitored release; then remove `_v2`'s temporary v1 alternatives and the v1 rollback branch in `tribe_service/llm_layer.py`; verify `python -m pytest tribe_service/tests/test_context_fit_rubric.py tribe_service/tests/test_llm_layer.py -q` | Retire `PITCHCHECK_CONTEXT_FIT_RUBRIC=v1`, adjust its fixtures/tests, keep the v2 rubric and provenance. No deletion until the two-release record exists. |
| Jev on desktop/web | Owner chooses to expose Jev after reviewing real usage, latency/cost and the numeric WS09 counterfactual report; backend Jev availability alone is not the decision | Implementation targets: request contract in `shared/types.ts`, `lib/tribe-client.ts`, `app/api/refine/route.ts`, `src-tauri/src/lib.rs`, and the existing `ScoreForm` flow. There is no enable-desktop-Jev script. After that scoped change run `npx vitest run`, `npx tsc --noEmit -p .`, and the shared-target Cargo test command in the verification ledger | Forward the chosen strategy through all desktop/web transports and test the resulting selection/provenance. No silent default strategy or new provider call is introduced by this job. |

The tune command uses joined arms; a separate outcomes file can instead be
supplied with `--outcomes "$DATA/outcomes.jsonl"` when its HMAC join and disjoint
count schema match [WS10 monitoring](ws10-monitoring.md). All strata and exclusions
must be reported. Upward weights and fitted constants require the plan's dated,
human-reviewed change; a recommendation does not apply itself.

## Probe-triggered tuning tables

| Item | Trigger and command | Change when the trigger fires |
|---|---|---|
| WS06 stance/tailoring | Run the paid persona-read command below: 20 EN/TR personas ×3 reads, six title-only pairs. Stance agreement <80%, any title-only stance mismatch, or empty unused-fact lists >70% is a monitoring failure | Tighten the existing evaluator prompt or reduce `PITCHCHECK_SKEPTIC_FACET_BOOST` (default .4) / `PITCHCHECK_TAILORING_PRIOR_POINTS` (default 3), with a dated report. `PITCHCHECK_PERSONA_READ_PRIORS=0` is the existing rollback. No arbitrary fitted replacement value follows from these thresholds. |
| WS09 Jev selection | After the first **50 actual Jev refinements**, count `selection_policy.roi_changed_selection`; >20% recommends `PITCHCHECK_ROI_PRIOR_POINTS=0.25`. Use the existing opt-in numeric refine logs/retained policy fields and the WS10 compare command above for available winner changes. No dedicated WS09 paid-probe CLI exists; A/B/rubric probes below invoke the existing refine paths but do not replace this 50-request sample | Apply only the named reduction after owner review. For the registered why/how control, >60% How losses attributable to ROI **or** abs Spearman ρ(ΔmPFC, Δword-count)≥.5 recommends `PITCHCHECK_ROI_PRIOR_POINTS=0`. These require measured mPFC differences, not the S2 pIFG/IPL primary verdicts. Existing S2 alone does not establish those checks. Add the explicit counterfactual/confound reducer to the WS08 harness before claiming them measured; there is no complete current CLI for that table. |
| WS12 specificity | Run the paid GPU specificity command below with ≥20 complete pairs, separately per encoder. Require ≥15/20 nonzero LLM/human twin sign agreements, positive Spearman ρ, detail-removal abs z > hedge-only abs z in ≥15/20, and pooled within-pair twin SD < half median abs LLM contrast. Use p95 of accepted paraphrase abs z as the proposed `Z_MIN` | Review `PITCHCHECK_SPECIFICITY_Z_MIN`; failed criteria recommend `PITCHCHECK_SPECIFICITY_SCORE_WEIGHT=0` and `PITCHCHECK_SPECIFICITY_REFINE_PENALTY=0` (shadow). If p95 /score latency exceeds half the configured request budget, use `PITCHCHECK_SPECIFICITY_CONTRAST=0`. Preserve raw shadows, K5 and K8; Turkish remains skipped. |
| WS05 behavior regression | Rubric report shows ad-copy/landing improved-rate down >20 percentage points in v2, or doctrine report has new reader/mind overclaims (target zero), prompt growth >650 tokens, or abs final shift >8 needing manual review | Review flagged texts; the plan's rubric rollback is `PITCHCHECK_CONTEXT_FIT_RUBRIC=v1`. These are behavior checks, not outcome validation. |

## Paid API scripts built but not run

These commands are instructions for a later **explicitly authorized paid run**.
None was executed in this job. Their `--selftest` modes are offline. The
specificity probe also needs a separate single-process GPU slot and the matching
cached model runtime. Reports containing synthetic/generated text stay outside Git.

```sh
python scripts/ab-prompt-doctrine.py \
  --old-worktree "$DOCTRINE_OLD" --new-worktree "$DOCTRINE_NEW" \
  --model "$MODEL" --limit 27 --output "$DATA/doctrine-ab.jsonl"
python scripts/ab-prompt-doctrine.py --report "$DATA/doctrine-ab.jsonl"

python scripts/eval-rubric-pairs.py --rubric both \
  --model "$MODEL" --output "$DATA/rubric-v1-v2.json"

python scripts/eval-persona-read.py --allow-paid-run \
  --model "$MODEL" --output "$DATA/persona-read.json"
python scripts/eval-persona-read.py --report "$DATA/persona-read.json"

TRIBE_TEXT_MODEL=NousResearch/Hermes-3-Llama-3.2-3B \
  python scripts/specificity-probe.py --git-sha "$SOURCE_SHA" \
  --input scripts/fixtures/specificity-pairs.jsonl \
  --output "$DATA/specificity-hermes"
TRIBE_TEXT_MODEL=meta-llama/Llama-3.2-3B \
  python scripts/specificity-probe.py --git-sha "$SOURCE_SHA" \
  --input scripts/fixtures/specificity-pairs.jsonl \
  --output "$DATA/specificity-llama"
```

The specificity script uses the existing service model selection and
`OPENROUTER_API_KEY`; it has no `--model` option. Run the two encoders serially.
Its isolated-GPU memory requirements are documented in
[the WS12 runbook](ws12-specificity.md); the current script's explicit 8-GiB
allocator request must be made compatible with the chosen device before a local
3070 run. This was already a reason to use the documented isolated research
container, and is not permission to run paid probes here.

The additional built-but-unrun blind corpus scorer also calls paid APIs:

```sh
TRIBE_TEXT_MODEL=meta-llama/Llama-3.2-3B \
  python scripts/score_validation_corpus.py "$DATA/consented-corpus.jsonl" \
  --output "$DATA/blind.jsonl"
```

## Public-data, method and language backlog

These items need additional input/method implementation; elapsed time by itself
does not supply the missing evidence. No runnable analysis is claimed where a
runner has not been built.

| Item | Exact reopening trigger | Script / concrete next step | What changes |
|---|---|---|---|
| Li 2026 | Study-1 Mandarin audiovisual stimuli, original per-person outcomes and author processing notebooks are available with checked usage/timing provenance, plus an allocated research GPU run | No Li runner exists. Implement an offline Study-1 adapter using `ExtractWordsFromAudio(language="chinese")`, `zh_core_web_lg`, `remove_empty_segments=False`, GSR and within-run z-score. Do not route this through the direct-text CLI. Freeze input IDs/hashes and the analysis before inference; run the new adapter only after it exists | Add OOD robustness numbers only; name Schaefer-200 7N #90 `LH_Default_PFC_8`, not a mental process. n=25/n=15 outcome correlations go to descriptive WS10 analysis. No product ROI or shipping gate follows. |
| WS08 S11 TTS | Separate Vast run, authorized TTS/audio/WhisperX downloads and aligned audio for the frozen direct-text stimuli, with enough memory/time | No S11 runner or `--family S11` exists. Extend `scripts/research/run_contrasts.py` with an explicitly registered direct-vs-TTS/WhisperX arm and event/timing audit; first run `python scripts/benchmark-runtime.py --selftest` | Record modality/timing sensitivity separately from the registered direct-text suites. Production stays direct; no retrospective primary verdict changes. |
| RF-ANTs registration | Verified CBIG MNI152→fsaverage RF-ANTs transform resources and toolchain, with documented coordinate conventions and reproducible peak mapping | Extend `scripts/build_roi_assets.py` to consume the verified transform first. Then run `python scripts/build_roi_assets.py --cache-dir "$DATA/rfants-cache" --output "$DATA/rfants-assets" --ui-output "$DATA/rfants-centroids.json"`. The current builder performs affine approximation and does **not** implement RF-ANTs | Review changed vertex/parcel assignments and dropout/unstable peaks, version the atlas/provenance, regenerate compatible reference/fidelity/spin assets and UI centroids, and rerun ROI tests. Do not silently substitute a transform beneath existing atlas hashes. |
| Language ID beyond Turkish | A labeled representative language/mixed-language sample with an explicit ambiguity policy and measured error rates, plus a separate paraphrase baseline per new language/encoder | No language-ID evaluation CLI exists. Extend the current `_looks_turkish` dispatch in `tribe_service/llm_layer.py` and the request language/skip paths in `tribe_service/app.py` with a reviewed detector and tests. Verify `python -m pytest tribe_service/tests/test_ws09.py tribe_service/tests/test_specificity.py tribe_service/tests/test_app.py -q` | Add supported language keys only with matching measured floors/provenance. Keep unidentified languages disclosed and WS12 skips explicit; a Turkish reference is not evidence for another language. |
| WS11 Lebel fidelity | Verified coordinate-matched per-run surfaces/volumes, repeat-run identities, matched transcripts and timing/preprocessing provenance listed in the availability report | Follow [the WS11 runbook](validation/ws11_runbook.md) and [the missing-input inventory](validation/ws11_lebel_2026-10-16.md). `fidelity_lebel.py` has not been built; implement the unchanged registered analysis after the bundle is usable. Run `python scripts/research/fidelity_lib.py --selftest` and `python -m pytest tribe_service/tests/test_roi_fidelity.py -q` as prerequisite local checks | Only a measured, compatible `roi_fidelity.json` supplies the [0.25,1] factors and causes the specified v2.4 promotion. HV2025's negative category-faithful r supplies neither fidelity factors nor C2 template. |

The named absent runners are implementation backlog, not commands to execute.
Creating placeholder scripts or tuning values would conceal the missing inputs.
All current feature defaults, rollback switches and shadow records stay effective
while these tracks collect evidence.
