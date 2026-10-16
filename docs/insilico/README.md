# WS08 in-silico track

Measured locally on 2026-10-16 using the RTX 3070, one GPU process at a time. Llama is primary; Hermes has its own Stage 0 and S2 robustness runs. Frozen stimuli and sidecars are unchanged. These are in-silico measurements, not human persuasion validation. Literature priors remain shipping features.

| Family | Class | Encoder | Date | Status | Key numbers / report |
|---|---|---|---|---|---|
| Stage 0 | Scale / sanity | meta-llama/Llama-3.2-3B | 2026-10-16 | measured | evidence ICC 0.724047; σw 3.43064; MDD 9.50926; length R² 0.404623; structure Δ 4.43493; shadow ρ 0.791243; [report](2026-10-16-stage0-meta-llama__Llama-3.2-3B.md) |
| Stage 0 | Scale / sanity | NousResearch/Hermes-3-Llama-3.2-3B | 2026-10-16 | measured | evidence ICC 0.779176; σw 3.15821; MDD 8.75412; length R² 0.501914; structure Δ 4.50913; shadow ρ 0.906485; [report](2026-10-16-stage0-NousResearch__Hermes-3-Llama-3.2-3B.md) |
| S2 | A | meta-llama/Llama-3.2-3B | 2026-10-16 | null, pass, reversed | pifg_vezich2017 sunscreen:How>Why: null (dz -0.175333, pH 1); ripl_vezich2017 sunscreen:How>Why: pass (dz 1.89936, pH 0.00039996); pifg_vezich2017 sunscreen:How>Fact: reversed (dz -1.07049, pH 1); ripl_vezich2017 sunscreen:How>Fact: pass (dz 1.8472, pH 0.00039996); pifg_vezich2017 b2b:How>Why: null (dz 0.127869, pH 1); ripl_vezich2017 b2b:How>Why: null (dz -0.192435, pH 1); pifg_vezich2017 b2b:How>Fact: null (dz -0.555697, pH 1); ripl_vezich2017 b2b:How>Fact: reversed (dz -0.570942, pH 1); [report](2026-10-16-S2-meta-llama__Llama-3.2-3B.md) |
| S2 | A | NousResearch/Hermes-3-Llama-3.2-3B | 2026-10-16 | null, pass_broad, reversed | pifg_vezich2017 sunscreen:How>Why: null (dz -0.357277, pH 1); ripl_vezich2017 sunscreen:How>Why: pass_broad (dz 1.72405, pH 0.00039996); pifg_vezich2017 sunscreen:How>Fact: reversed (dz -0.921311, pH 1); ripl_vezich2017 sunscreen:How>Fact: null (dz 1.64326, pH 0.00039996); pifg_vezich2017 b2b:How>Why: null (dz -0.0961446, pH 1); ripl_vezich2017 b2b:How>Why: null (dz -0.0819954, pH 1); pifg_vezich2017 b2b:How>Fact: null (dz -0.617839, pH 1); ripl_vezich2017 b2b:How>Fact: null (dz -0.357337, pH 1); [report](2026-10-16-S2-NousResearch__Hermes-3-Llama-3.2-3B.md) |
| S4 | A | meta-llama/Llama-3.2-3B | 2026-10-16 | uninformative | mpfc_falk2010 Whygain>Fact: uninformative (dz -0.402551, pH 1); mpfc_falk2010 Whygain>Fact_you: uninformative (dz -0.298233, pH 1); [report](2026-10-16-S4-meta-llama__Llama-3.2-3B.md) |
| S1 | A | meta-llama/Llama-3.2-3B | 2026-10-16 | uninformative | dmpfc_chua2011 T>U_you: uninformative (dz 0.0879686, pH 0.342066); precuneus_chua2011 T>U_you: uninformative (dz 0.610037, pH 0.0131987); [report](2026-10-16-S1-meta-llama__Llama-3.2-3B.md) |
| S3 | A / B limit | meta-llama/Llama-3.2-3B | 2026-10-16 | uninformative | vmpfc_frontopolar excessive<reasonable: uninformative (dz 0.18022, pH 0.783622); [report](2026-10-16-S3-meta-llama__Llama-3.2-3B.md) |

The dated reports include every measured quantity, exclusions and nulls. Product references are `tribe_service/roi_assets/roi_reference_<encoder-slug>.json`; NPZ caches, audit matrices and raw logs stay outside git under the authorized scratchpad. See [local execution notes](2026-10-16-local-runs.md) for bookkeeping and runtime deviations.

## Frozen inputs and review

`scripts/research/fixtures/{stage0,S2,S4,S1,S3}.json` are the dated pre-registration
sidecars. Each fixes the text SHA-256, contrasts, directions, ROI names, Holm
groups, minimum 16 / target 20 pairs, seed, thresholds and multiplier policy.
S4 hashes and selects the B2B cohort from **S2.jsonl**, avoiding duplicate stimuli.
The JSONL text is original English material, never copied from papers or customers.
The rejected template corpus was replaced on 2026-10-16. Each original,
paraphrase, independent pitch, neutral text, and contrast condition was written
individually as full prose by OpenAI Codex gpt-6.1-sol; only structure controls
and retest copies are derived. The prompt is recorded in each sidecar. The
harness does not expose its sampling temperature; that field is explicitly null,
and no paid generation or classification API was called.

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

Fixture preflight rejects any normalized sentence shared by different pairs and
any cross-pair word-8-gram overlap ≥0.10 (intersection divided by the smaller
text's distinct 8-gram count). Every condition is checked, including all Stage 0
independent and neutral texts. Within-pair matching is intentional; structure
and retest copies are excluded from diversity comparisons. This catches the
rejected template pattern. Semantic independence is a separate question from
this mechanical check. The frozen blind check and delegated sample review are
recorded in [the manipulation-check artifact](manipulation_check/blind_labels_2026-10-16.json):
360/360 condition labels and 30/30 paraphrase-equivalence groups matched. These
records were merged from `research/parallel`; integration did not repeat that
classification or represent it as a new owner review. The independent reference set has 30
pitches per channel, with 10 in each length bucket.

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

## Integrated reference units

Stage-0 CPU analysis exports `unit_calibration` beside `quantities` and `checks`
when the signed ROI reference is nondegenerate. Its `basis=length_model_z` uses
`clip((minus_cortex - a - b*log(words))/resid_sd, -3, 3)`. The shared calibration
reducer takes each ROI's median and the fixed-denominator composite SD, tagged
`unit_calibration:stage0:<date>:<revision>`. Runtime scoring and synthesis pass the
engine token count and consume the same units. Missing length data or a missing
reference falls back to the committed WS07 encoder unit artifact; WS09 retains
its separate `default_prior` noise floors. Measured Stage-0 files are now supplied for both encoders; the product prefers
them over the provisional WS07 calibration and WS09 default floors. No synthetic
test reference is installed for runtime.
