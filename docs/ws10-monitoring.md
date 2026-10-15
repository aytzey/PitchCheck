# WS10 offline monitoring

Implementation of plan 10B steps 6–9 and 10C steps 10–13. Read
[the pre-registration](validation-protocol.md) before opening real outcomes.
This track never applies a recommendation or gates shipment. The registration
was written before real-data access and is committed with the fixtures on
`research/integrated`; real collection/analysis remains pending.

## Inputs and privacy

Use the existing opt-in 10A log and the same `score_log.pitch_id()` HMAC key for
every join. Keep credentials, log files, outcomes, consented corpora and individual
ratings outside the repository and deployment image; `validation_data/` is also
ignored. Never copy a key into a command or a report. All examples assume a
separate research environment with environment variables already configured and
`DATA` pointing outside Git. No new dependency is needed beyond the test/runtime
requirements (stdlib and NumPy; no sklearn/statsmodels).
`validation_data/` is excluded from Docker build contexts as well.

Production changes are a formula extraction with unchanged defaults, an optional
system-prompt keyword with the identical default payload, and numeric log fields.
The public score response, default flags and 66.30 / 73 pin are preserved.
`blend_inputs` stores unrounded inputs; response fields alone can lose enough
precision to fail the .05-point replay tolerance. Older logs are accepted only
if they actually pass parity. Use matching code, methodology and encoder versions.

A joined arm JSONL has required `pitch_id`, `group_id`, `channel`,
`outcome_family`, `outcome`; binomial families (`reply`, `meeting`, `acceptance`,
`click`, `binomial`) also require integer `n_sends`, `positives`. PME uses the
message mean (1–4). Include `language`, `data_source`, `scored_at` (timezone),
`edited_after_score`, encoder/methodology/LLM provenance and surface/score fields.
Use `data_source=randomized_campaign|upworthy|pme_panel` only for those designs;
production traffic stays descriptive. Synthetic records must say `synthetic=true`.
The loader rejects nonfinite/out-of-range scores and text lacking
`consent_text=true`. General-channel/personal replies, mock or non-20,484-vertex
output, missing provenance/facets and edited arms are counted as exclusions.
Different languages, channels, outcome families, encoders and versions stay in
separate reports. Refine rows never enter the score/B1 analysis.

For separate files, `--outcomes outcomes.jsonl` joins by HMAC; repeat score reads
keep latest deterministic features and average LLM scores/facets. Outcome rows
are **disjoint counts**, summed per pitch × campaign × arm; do not repeat an
already-aggregated count. Add `persona_id` to resolve a multi-persona join. A
standalone joined input contains one row per arm (duplicate reads are collapsed).
Use `--blind-input blind.jsonl` to join blind reads by pitch, persona, encoder,
methodology and LLM model without replacing the product score. Surface metadata
must use the engine word tokenizer, not `str.split`.

## CPU harness

```sh
python scripts/validate_incremental.py compare "$DATA/scores.jsonl" \
  --dataset-id campaign-v1 --output "$DATA/compare-v1"
python scripts/validate_incremental.py describe "$DATA/scores.jsonl" \
  --dataset-id campaign-v1 --output "$DATA/describe-v1"
python scripts/validate_incremental.py increment "$DATA/scores.jsonl" \
  --blind-input "$DATA/blind.jsonl" --outcomes "$DATA/outcomes.jsonl" \
  --dataset-id campaign-v1 --output "$DATA/increment-v1"
python scripts/validate_incremental.py sweep "$DATA/joined-arms.jsonl" \
  --dataset-id campaign-v1 --output "$DATA/sweep-v1"
python scripts/validate_incremental.py rollback_alarm "$DATA/joined-arms.jsonl" \
  --flag PITCHCHECK_ROI_PRIOR_WEIGHT --dataset-id campaign-v1 \
  --output "$DATA/alarm-v1"
```

`tune` is an alias for `sweep`, producing a recommendation only. Real runs use
1,000 permutations and 2,000 bootstrap draws; `--permutations 199` is for CI.
Reports include input/protocol SHA-256, strata/exclusions, seed, provenance and
JSON/Markdown checksums. Existing report files are never overwritten. The
committed synthetic comparison is a replay check, not outcome evidence.

The [200-row synthetic comparison](validation/ws10-synthetic-compare.md) passes
parity with artificial priors. Reproduce its input by JSONL-serializing
`tribe_service.tests.fixtures.ws10_synthetic.comparison_records()` using
`json.dumps(row, sort_keys=True)` and a newline per row, then run `compare` with
`--dataset-id ws10-synthetic-ci-v1` and a fresh output path. The JSON carries
`synthetic=true` and hashes of the input, protocol and implementation.

`increment` reports primary B2 | B0 + blind B1, reverse order, available axes/raw
features/B3, and B3/B2 beyond each other. Missing future ROI blocks are omitted,
not imputed. `--conditioned-b1` explicitly permits production B1 and prints
"TRIBE-conditioned B1: blend-weight test, not incremental validity". OLS nested
F/ΔR² are descriptive; CV uses ridge λ=1. Edited arms, exclusion of Upworthy
winners, and |within-group outcome z|>2.5 are separate descriptive sensitivities.
Scrambled B1 and B2 controls are analyzed when present.

`compare` fails if a replay differs from the logged active final by >.05; it
reports rank association, mean |Δ|, neural-band clamp frequency, refine winner
changes and within-group concordance when outcomes exist. `describe` adds
correlations/VIF, memorability-to-axis ablation and reward-affect redundancy.
A null VIF means constant/collinear, not zero redundancy.

Sweeps use .35–.85 semantic blends and coarse .10 simplex moves from the current
facet/axis weights, plus ROI weights at 0/half/prior/double when exact
`prior_inputs` exist. ROI inputs must include `evidence_without_roi`,
`facet_without_boost`, `tailoring_points`, `specificity_points`; log budget,
term weights/provenance and active flags alongside them. Terms overlap and are
never subtracted to approximate a counterfactual. K5 attribution/budget support
is inert unless another work package explicitly supplies prior inputs.

For selection points, add `--refine-input candidate-outcomes.jsonl` to sweep/tune.
It is a separate randomized-candidate ranking analysis. Each candidate requires
outcome counts, provenance, `selection_score_without_roi`, `roi_selection_unit`,
`roi_prior_points`, `selection_score`; the last equals baseline + unit × points.
Units already incorporate encoder/fidelity/in-silico scaling. Ranking parity is
checked. Missing WS09 fields are reported as unavailable until WS09 supplies
them; no feature waits for that report. For a policy-arm alarm, use
`rollback_alarm policy-arms.jsonl --randomized-arms --flag PITCHCHECK_ROI_PRIOR_POINTS`
with the usual dataset/output flags. Supply exactly one aggregated row for each
`with_neural`/`semantic_only` policy per group; these arms never enter blind B1.

Candidate formulas are frozen before evaluation, selected on group CV in the
first 75% of timestamps, and only the winner is evaluated on the final 25%.
Training groups crossing the time boundary are purged. Fixed formula comparisons
use within-group concordance and paired group sign permutations; they fit no
coefficients, so folds only restrict comparable pairs. Incremental blocks use
residual permutation and refit the complete ridge pipeline each iteration.
Holm covers all candidates/strata in a run, including optional selection input.
Use one complete candidate manifest/run per dataset version; do not cherry-pick
separate runs. Bootstrap resamples groups over fixed OOF pairs without refitting.

Below 80 arms, 10 groups, or 100 sends/arm, the decision is
`underpowered — monitoring only`. No synthetic or selected production data can
authorize fitting. Positive CV CI + Holm p<.05 + positive temporal holdout are
all required for `tune`. A wholly negative 95% CI can emit a flag-specific
`rollback_alarm`; neither action changes an environment variable. Each accepted
tune needs a human-reviewed dated PR, methodology version increase and
`calibration_basis=fitted:<dataset_id>`. Do not claim comparable scores across
methodology versions. ROI scalars remain predicted-response analogues.

## Independent collection helpers

Owner campaigns: `prepare_validation_data.py assign-campaign` consumes JSONL
`{recipient_hash, group_id, selected_id, semantic_id, candidates:{id:pitch_id}}`.
It emits a stable keyed 50/50 assignment and chosen `score_log_id` when winners
disagree. It sends nothing and changes no product selection. The sending tool
must save assignment, sent time, edits, sends and positives outside Git.
Connecting Taskfulight or another sending tool is owner work, outside this repo.

Upworthy: manually obtain the public archive named in the protocol, then:

```sh
python scripts/prepare_validation_data.py prepare-upworthy "$DATA/archive.csv" \
  --output "$DATA/upworthy.jsonl"
```

The helper requires risk flags, excludes the risk period/tests, fixes images,
merges duplicate titles, and labels `proxy_upworthy`. Run again with
`--exclude-winners` for sensitivity. Output texts are public licensed stimuli;
the fixed persona/channel and shared HMAC are used by the corpus scorer.

PME: prepare 20–25 private base families/channel, exactly four variants each;
`panel-assign stimuli.jsonl --output assignments.jsonl` produces a Latin square
for 60 raters (15/message). Assignments contain IDs, not text. Record all items
named in the protocol using keys `persuasive`, `information`, `feelings`,
`reply_likelihood`, `believable`, `pme_persuasive`, `pme_convincing`,
`pme_effective`, together with `rater_id`, `pitch_id`, `group_id`, `channel`.
`panel-aggregate ratings.jsonl --panel-id <id> --output outcomes.jsonl` rejects
repeat-family exposure or <15 ratings/message and reports split-half
Spearman–Brown reliability. Below .70 collect more ratings. Output carries the
verbatim PME scope rule and `pme_panel_<id>` provenance; it is rated
effectiveness, not replies. No panel recruitment or payment is performed.

## Deferred GPU / provider runs

In a separate research GPU image with canonical gated Llama access and the
owner-configured OpenRouter key:

```sh
TRIBE_TEXT_MODEL=meta-llama/Llama-3.2-3B \
  python scripts/score_validation_corpus.py "$DATA/consented-corpus.jsonl" \
  --output "$DATA/blind.jsonl"
```

Rows require consented `text`, `persona`, `channel`, `group_id`, `language` and
data-source provenance. The script explicitly sets the canonical encoder and
rejects mock/incompatible output before any blind API call. It makes three
blind 0.2-temperature samples, flags missing facets, reports B1 sample reliability
alongside TRIBE retest features/L2, and exports only numbers/IDs. Its final score
is neural-only; use matching production logs plus `--blind-input` to sweep the
actual production blend. Only after a candidate passes, repeat with `--scrambled`
for deterministic word-order twins. Do not run this on pitchserver.

No production logs/outcomes, public corpus download, paid request, GPU run,
server deployment, sending-tool integration or panel collection was performed
by this implementation. Operational acceptance on the first 200 real records,
manual text-leak review and any owner flag change remain future operator work.

## Integration with literature priors

The sole K5 owner is `_blend_final(evidence, raw_llm, facet, confidence,
quality_weight, *, blend=..., priors=None)`, returning final score, semantic score,
semantic weight, guardrails and the prior report. Runtime and corpus scoring use
the same function through `_calibrate_result`. `prior_inputs` records the
ROI-free evidence, unboosted facet, tailoring and specificity points at full
precision. Logs also retain the three numeric score terms, single-prior shadows,
specificity scalars and active flags; replaced spans and control text are excluded.
Replay carries an archived budget through `priors["budget"]` so changing the
current service budget does not change historical parity. Display scores remain
rounded; the comparison tolerance is below 0.05 points.

The blind system prompt preserves the doctrine and annex verbatim. Their generic
claim-hygiene references to TRIBE contain no pitch-specific neural data; all
model-output instructions and pitch-specific neural payloads are removed from
the blind read. Tests cover both encoder paths, LLM and fallback scoring, and
zero/default budgets. No outcome-validation claim follows from these checks.
