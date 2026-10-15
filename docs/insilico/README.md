# WS08 in-silico track

Prepared 2026-10-16. No GPU inference, blind classifier, owner review, or live
runtime check has been run for this implementation. This track measures sensitivity
of TRIBE's average-subject text prediction to authored linguistic forms. It never
gates shipping literature priors and does not establish persuasion effectiveness,
reader-specific states, or human replication.

| Family | Class | Encoder | Date | Status | Key numbers / outstanding measurements |
|---|---|---|---|---|---|
| Stage 0 | Scale / sanity | Llama-3.2-3B | 2026-10-16 | not_run | 405 stimuli; 300 reference pitches; ICC(1), sigma_within, mdd, length R², 0d structure and 0e shadow comparison pending |
| Stage 0 | Scale / sanity | Hermes-3-Llama-3.2-3B | 2026-10-16 | not_run | Same fixture; independent encoder-specific ICC(1), mdd, R², structure and shadow measurements pending |
| S2 | A; positive control | Llama / Hermes separately | 2026-10-16 | not_run | 20 sunscreen + 20 B2B frames; How>Why and How>Fact, 2 location ROIs; 4 co-primary hypotheses per cohort |
| S4 | A; related ROI definition | Llama | 2026-10-16 | not_run | 20 B2B frames shared with S2; Whygain>Fact and Whygain>Fact_you |
| S1 | A | Llama | 2026-10-16 | not_run | 20 topics; T>U_you in dmPFC and precuneus locations; vmPFC report only |
| S3 | A price-period prior / B valuation limit | Llama | 2026-10-16 | not_run | 20 offers, 20 number-only control pairs; vmPFC primary decrease, right insula secondary increase |

These rows are a dated work ledger, not measured null results. Actual runs add
`<date>-<family>-<encoder-slug>.json` reports here and update this table, including
nulls and exclusions. No synthetic test output is installed as a product reference.

The implementation leaves all changes **uncommitted**, as required by the job
brief. The owner/supervisor must commit the fixtures and pre-registrations before
the first run. §9.3's measured Stage 0 reference files and S2 results remain pending
GPU execution; neither acceptance item is claimed complete.

## Frozen inputs and review

`scripts/research/fixtures/{stage0,S2,S4,S1,S3}.json` are the dated pre-registration
sidecars. Each fixes the text SHA-256, contrasts, directions, ROI names, Holm
groups, minimum 16 / target 20 pairs, seed, thresholds and multiplier policy.
S4 hashes and selects the B2B cohort from **S2.jsonl**, avoiding duplicate stimuli.
The JSONL text is original English material, never copied from papers or customers.
Generation used fixed templates authored by Codex (GPT-6); the prompt is recorded
in each sidecar. The harness does not expose its sampling temperature; that field
is explicitly null, and no paid generation or classification API was called.

Before freezing, use a different model family to classify shuffled texts without
hypotheses or ROI names, and have the owner read at least 20%. Record the classifier
model, labels/errors and owner review in the review record; revise failed items,
update hashes, and commit the sidecars with `blind_classifier="passed"`, a nonempty
`classifier_model`, `owner_sample_fraction>=0.2`, and `owner_review="passed"`.
Pending review deliberately prevents an unreviewed research run. This restriction
applies only to the offline study; product priors remain available.

Stage 0 contains 30 families × (original + 4 fact-preserving paraphrases), with
six families per non-general channel and all three length buckets (≤60, 61–150,
151–240). It adds 150 independent pitches, 30 neutral descriptions, 30
punctuation-free/scrambled pairs and five texts repeated three times. Paraphrases
retain facts, pronouns and CTA; structure controls retain the token multiset.
S1 has five statements per condition and matches second-person pronoun counts
between T and U_you. S3 holds the cost sentence at the same word index and has
number-only controls with the same numbers but no cost meaning. All texts obey
the engine tokenizer's 240-word ceiling, pair length ±10%, and S3's ≤2-word limit.

## Running later

The module docstring in `scripts/research/run_contrasts.py` gives the container
recipe. Use a clean normal checkout with `.git` accessible inside the container,
the production image/revision and cached model mounts, `HF_HUB_OFFLINE=1`, mock
off, prediction cache zero and direct mode. Set `PITCHCHECK_IMAGE_ID` to the
image digest when invoking the container. Do not run beside the live service on a
15.47 GiB card: stop/restart it in a maintenance window, or use Vast ≥24 GB.
No deployment, maintenance action, or remote connection was performed here.

```bash
python scripts/research/run_contrasts.py score --family stage0 \
  --git-sha "$(git rev-parse HEAD)" --out /audit/insilico/llama-stage0
python scripts/research/run_contrasts.py analyze --run /audit/insilico/llama-stage0 \
  --output docs/insilico/2026-xx-xx-stage0-meta-llama__Llama-3.2-3B.json \
  --reference-out tribe_service/roi_assets/roi_reference_meta-llama__Llama-3.2-3B.json
```

GPU commands run in the configured image; CPU analysis runs in the test environment
without torch/scipy/model weights. Repeat Stage 0 and S2 in **separate processes**
with `TRIBE_TEXT_MODEL=NousResearch/Hermes-3-Llama-3.2-3B` and separate directories
and reference filenames. Do not transfer pass statuses between encoders.

After Stage 0, run S2 → S4 → S1 → S3. For S2 analysis supply `--reference <same
encoder reference>`; for S4/S1/S3 also supply `--positive-control <S2 analysis.json>`.
Use `--reference-out <reference>` to export updated checks, then inspect the report
and commit the dated artifacts. A missing/failed S2 marks downstream results
`uninformative` with multiplier 1, never a penalizing null or reversal.

The runner checks the fixture hash and clean committed pre-registration **before**
loading weights and records the pre-registration commit. It rejects mock/cache/
mode mismatches, >240-word texts, bad pairing, nonfinite predictions and any shape
other than `(ceil(0.4*(N-1)+0.32), 20484)`. Retry/fallback rows are saved for audit,
excluded from analysis and counted. An existing run directory is never overwritten;
an interrupted run remains explicitly incomplete. Matrices are float16 `.npy`
files with individual hashes, alongside `rows.jsonl` and `metrics.json` outside
git (`/audit/insilico/<run_id>` or ignored `audit/insilico/`). Retests always perform
fresh inference. Float16 reanalysis has storage quantization; retest L2 is a sanity
check, while the noise floor comes from paraphrases, not deterministic retests.

Provenance includes tribev2 commit and local WhisperX patch note, HF snapshot,
encoder compatibility, language/mode/event version, 0.32/0.40 timing, runtime
device/batch configuration, prediction dtype, atlas hash, fixture and harness hashes,
git/pre-registration revisions, image id and Rust/Python key parity. No secrets or
credentials belong in these files. Scripts must be mounted into the runtime image;
the normal server image only copies `tribe_service/`.

## Optional WS07 assets and product contract

The only atlas loader is the planned
`roi_features.load_roi_assets() -> (labels, parcel_names, rois, locations, sha) | None`.
Both parcel-union and literature-location summaries come from WS07's `roi_summary`;
Schaefer-400 parcel means subtract the signed cortex mean and exclude medial wall.
No replacement atlas or WS07 implementation is added here. Missing assets return
`None`/skip ROI calculations. Stage 0's score/signal/axis/window quantities still
work. After merging WS07, repeat **CPU analysis of the saved matrices** to populate
ROI quantities and parcel ranks, without another GPU run. The current atlas hash
is attached to the exported reference. Future references must match encoder,
event version, mode and atlas before ROI consumers use them.

Runtime files are `tribe_service/roi_assets/roi_reference_<encoder.replace('/', '__')>.json`.
`engine.roi_reference(encoder)` is cached and returns a dict for compatible scale
data, including null/reversed checks; it returns None for missing, malformed,
wrong-encoder, wrong-event-version or wrong-mode files. Restart workers after
installing a reference to clear the cache. WS07/WS09 additionally compare
`atlas_sha256` against their loaded atlas. WS07 owns length-residual z, centers,
scale and K4 numerator multipliers; WS09 owns the ROI selection deadband.

The loader's quantities include `noise.{sigma_within,icc1,icc1_ci95,min_detectable_delta}`,
`length_model.{a,b_log_words}`, `resid_sd`, five residual quantiles and confound R².
Length z is `clip((roi - (a + b*log(words))) / resid_sd, -3, 3)` when scale is usable.
`checks` carry family, status, date, d_z, Holm p, parcel rank, weight basis and
weight multiplier. Policy is pass 1, pass_broad .75, null .5, reversed 0,
not_run/uninformative 1, with low_stability ×.5. Multipliers never exceed 1 and
only shrink K4 numerators; increasing weights requires a dated human PR.

Legacy refine uses `PITCHCHECK_NEURAL_TIE_DELTA` (finite, nonnegative) first, then
the encoder reference's **evidence_score** mdd, then 1 point (`default_prior`). A
neural difference strictly below the floor contributes the baseline neural value
to ranking; stored measurements remain unchanged. `0` restores previous ranking.
`tribe_guidance` adds `neural_delta`, `neural_tie_delta`,
`neural_tie_delta_source` (`env|paraphrase_noise_floor|default_prior`), and
`neural_delta_detectable`. `improved` still means selected id differs from original.
Jev's ±3 band and .0005 response geometry constant are unchanged. No TS, Rust,
mobile, LLM ROI payload or UI work is part of WS08.

## Analysis rules and limits

Stage 0 uses one-way random-effects ICC(1) over complete paraphrase families,
whole-family bootstrap CIs, and mdd = 1.96√2 × within-family SD. The pooled 300-pitch
reference fits `a+b*log(words)` and stores residual quantiles/SD. Neutral, structure
and retest rows never enter that fit. Reports include individual/joint confound R²,
ROI/proxy correlations, punctuation-free minus scrambled differences, and
ROI-on/off Spearman, paraphrase ICC and pair-rank changes. After WS07 is merged,
its actual `roi_prior_from_summary` and evidence anchor supply the shadow scores;
until then the shadow comparison explicitly says `not_run`. A >.1 ICC loss only
recommends halving weight.

Core inference is a one-sided paired sign-flip test in the pre-registered direction,
Holm per declared family/cohort, paired bootstrap CI and standardized d_z. For
How>Why, Why is the within-frame mean of Whygain and Whyloss. Primary pass needs
expected direction, Holm p<.05, |median delta|>mdd and parcel rank≥.90; a broader
effect is `pass_broad`. Opposite direction with two-sided Holm p<.05 is `reversed`.
Other informative results are `null`. Co-primary checks use the most conservative
status for a given ROI. Secondary/report-only tests do not alter check multipliers.
S2 reports motor/body covariate adjustment alongside its raw pre-registered test;
perfectly confounded lexical counts are flagged unidentifiable rather than
reported as an adjusted effect. Raw deltas are never used to compare ROI/family
effect magnitudes. d_z is across authored stimuli, not human participants.

S3's secondary right-insula value is the right-hemisphere subset of the loaded
insula vertex set, derived offline, not a new product ROI. Its cost-sentence window
uses `engine.direct_text_segment_words`, expands a segment and retains at least
four segments. Ancillary locations absent from WS07 are skipped. P2/P3 S5–S10
and direct-vs-TTS studies remain backlog; no extra atlas or network download is
needed by this implementation.

| Source modality | Harness analogue | Limit |
|---|---|---|
| Vezich / Chua: text with simultaneous audio | English direct synthetic read-aloud schedule | No natural timing, actual audio, subject tailoring or human responses |
| Knutson / Plassmann: food/images, subjective WTP, task choices | Passive cost wording plus number-only controls | Null is plausible; no inference about reader value or purchasing |
| TRIBE training: average-subject predicted fMRI analogue | Signed whole-text cortex/ROI summaries | Bidirectional context; sentence windows are descriptive; no baseline mental-state inference |

Finding → caveat wording for later reports:

- Positive contrast → “The predicted average-reader response differs for these
  authored linguistic forms; this does not establish a mental process in a reader.”
- Null → “The prior does not track this pre-registered contrast in silico for this
  encoder and fixture; the null is recorded alongside the literature prior.”
- High ROI/proxy correlation (|r|>.8) → ROI adds little distinct information; |r|<.3
  supports keeping neutral proxy labels, not an anatomical process claim.
- Weak structure discrimination → disclose that the score may not distinguish
  coherent from scrambled wording; retain the product score and report the caveat.
- Low paraphrase ICC → report low stability and a shrink-only multiplier; retest
  determinism alone is not reliability.

## Local verification

`pytest tribe_service/tests -q` exercises the NumPy statistics (including 500 null
simulations), synthetic block spatial nulls, loader compatibility, refine selection,
fixture hashes/lengths, missing atlas, committed-pre-registration preflight,
mocked scoring/exclusion and CPU analysis. No GPU, paid API, real research data,
server, or deployment is used. `scripts/smoke-runtime.py` contains the four-field
refine assertion for the owner's later live check; it was not executed here.
