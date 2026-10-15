# WS10 validation protocol — pre-registration v1

Written on 2026-10-16 before inspecting any outcome data. Only synthetic data may
be used to develop this harness. This file must be committed and its commit SHA
recorded before collecting or opening a real dataset. The implementation brief
forbids committing during this job; the supervisor must commit this registration.

This is a parallel monitoring and tuning protocol. It never gates a feature,
changes a product score, edits a production constant, or disables a flag.
Literature priors retain their flags, shadow values and unvalidated provenance.

## Frozen analysis

The unit is a message arm (text × campaign); recipient rows are aggregated to
that unit. Group is campaign/test ID, or base-pitch family for the panel. Each
dataset version has one primary test: **B2 | B0 + blind B1**, where B2 is the
single `evidence_score` column. Do not choose predictors after viewing outcomes.

- B0: channel indicators, log word count (engine tokenizer), segment bins
  ≤2 / 3–8 / 9–24 / 25–64 / ≥65, words per sentence, digit count.
- Blind B1: `context_fit_score_blind`, `raw_llm_score_blind`, averaged over three
  independent samples at production temperature 0.2. Use the production doctrine,
  annex, channel norms and facet rubric verbatim, without response geometry,
  axes or calibration instructions. Missing/invalid facets exclude the row.
- B2: the frozen product evidence score, one degree of freedom.
- Exploratory only: the five existing axes, ten `FEATURE_KEYS`, and B3, the
  existing WS07 ROI `z` scalars (rois and locations separately named). Test
  B3 beyond B2 and B2 beyond B3. No connectivity features. These exploratory
  results cannot alone justify a change. Report reverse B1/B2 entry order.

Use weighted within-group centering, training-fold standardization, closed-form
ridge with λ=1 and five group-disjoint folds. For binomial arms use
`y=log((k+.5)/(n-k+.5))` and `w=1/(1/(k+.5)+1/(n-k+.5))`; for PME use the
message's mean rating and unit weight. Held-out groups have no fitted group
intercept; evaluate only within-group ordering. Report pooled OOF concordance
`C_w`, using pairs from the same group and fold (prediction ties count .5;
outcome ties are excluded). PME also gets within-group OOF R². Report unpenalized
OLS ΔR² and nested F in both entry orders as in-sample descriptions, including
rank deficiency and boundary/separation diagnostics; do not use ridge for F.

Residual (Freedman–Lane/Kennedy) candidate permutation preserves the B0/B1 fit;
refit the full CV pipeline in each of 1,000 permutations (199 in CI). Permute
within group when median group size ≥3, otherwise across rows. Use a two-sided
test for formula changes/alarms and a one-sided test for incremental blocks.
Use a 2,000-draw group bootstrap of fixed OOF pairs for 95% intervals, without
refitting. Apply Holm across **all** candidate changes in a dataset version,
including unsuccessful candidates. Seed is 10. No primary outlier exclusion;
report |z|>2.5 removal only as a sensitivity analysis.

Exclude mock/non-20,484-vertex rows, missing blind facets, personal invitations
and general-channel reply data; separate edits after scoring as sensitivity.
Keep latest deterministic features per pitch and average repeated LLM reads;
retain distinct campaign arms. Never pool different encoder, methodology,
LLM model, language, channel or outcome-family strata. Production B1 is labelled
"TRIBE-conditioned B1: blend-weight test, not incremental validity". Production
traffic is descriptive monitoring only because sending is selected/confounded.

## Decisions and shadow formula tests

`compare` replays the shared pure blend and persuasion features; fail on active
score parity error >.05. Report Spearman ρ, mean |Δ|, band crossing and refine
winner changes. Never infer a formula by subtracting overlapping prior terms.
Sweep semantic blend .35:.10:.85; ROI weight and selection points at
0, half, prior and double; and coarse simplex context-fit/axis weights. Freeze
all candidates before scoring outcomes. Select on the first 75% by timestamp
using group CV; test against the current formula on the untouched latest 25%.
Keep families disjoint across that boundary; do not use holdout results to select.

Decisions: `keep`; `underpowered` ("underpowered — monitoring only");
`tune:<constant>=<value>` only with positive CV Δ interval, Holm p<.05, a
positive temporal holdout comparison and sufficient N;
`rollback_alarm:<flag>` only with sufficient N and an entirely negative 95%
CV Δ interval for the flagged formula (or randomized arm loss). Sufficient N
means ≥80 arms **and** ≥10 groups **and**, for binomial data, ≥100 sends in
every arm. No synthetic or observational production dataset may authorize tuning.
Tuning requires a dated, human-reviewed PR, a methodology version increase and
`calibration_basis="fitted:<dataset_id>"`. Reports are JSON and Markdown with
SHA-256 checksums and dataset/protocol provenance. Recommendations never apply
themselves. Report only predictive association; avoid causal language in reports.

## Independent data collection tracks

**Owner campaigns:** store `score_log_id`, group, sends, positives, outcome,
`sent_at`, `edited_after_score` and assignment. When refine winners disagree,
use a keyed recipient hash for stable 50/50 assignment to `with_neural` or
`semantic_only`; save the chosen pitch ID. Randomization is the only test of
the selection policy. No sending or product selection changes are part of WS10.

**Upworthy:** manually obtain the archive (Matias et al. 2021,
doi:10.1038/s41597-021-00934-7, CC BY 4.0). Exclude flagged randomization risks
(`problem` / `randomization_imbalace_risk`, including the correctly spelled alias)
and the June 2013–January 2014 risk period. Require a constant `eyecatcher_id`
within test, merge duplicate titles by summing clicks/impressions, use
`clickability_test_id` as group, fixed persona, `ad-copy` channel. Repeat without
winner packages. Label `calibration_basis="proxy_upworthy"`; headline clicks
are a proxy for ad-copy only, and are not sales replies.

**PME panel:** owner-authored private stimuli, 20–25 base pitches ×4 variants
per channel, including original and actual refine candidates, not exclusively
optimized drafts. Latin-square assignment, at most one variant of a family per
rater; ≥15 raters/message (target 15–30). Persona roles are self-reported.
Freeze items: "This message, as a whole, is PERSUASIVE", "based on information",
"based on feelings" (each 1–4); "How likely would you be to reply?" (1–7);
"The claims in this message are believable" (1–4); three PME items (persuasive,
convincing, effective, each 1–4). The primary outcome is their mean. These PME
item/scale choices are PitchCheck's own. Use base-family fixed effects and
GroupKFold; report split-half Spearman–Brown reliability; below .70 collect
more ratings. Compare blind B1 and B2 with message-specific scrambled twins,
generated only after a candidate passes, with deterministic pitch-ID seeds.

PME is a self-report outcome; it tests perceived effectiveness only. A TRIBE null on PME is not evidence against behavioural validity, and a TRIBE positive is not evidence for reply rates.

Panel fits carry `calibration_basis="pme_panel_<id>"` and "calibrated to rated
effectiveness, not replies". Anatomical/process interpretations remain
predicted-response analogues regardless of any predictive result.

Logs, outcomes, consented texts, recipient assignments and fetched corpora stay
outside Git and deployment images (`validation_data/` is ignored). Raw text
requires `consent_text: true`; log records contain no text/persona/facet notes.
Never store credentials or publish individual ratings. No real collection,
paid API call, GPU run or production access is authorized by this registration.
