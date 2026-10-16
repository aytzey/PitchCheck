# WS11 pre-registration — offline fMRI fidelity

Status: protocol only; no TRIBE predictions or measured fidelity results have been
inspected. Commit this file and the stimulus ID/hash lock before the first GPU
analysis. The commit is the registration; each result records this file's SHA-256.
No result gates a product feature. All thresholds below are **unvalidated priors**.

## 11.1: fixed analysis

Target: NeuroVault collection 17520, unthresholded group T image 886850 (D−S),
with 886851 (S−D) as an inverse-sign check. Public collections are CC0 (FAQ 13).
Use the first 10,242 vertices of each surface, left then right, and WS07's
`roi_features.load_roi_assets()` labels. Until WS07 merges, CBIG Schaefer-400
7-network fsaverage5 annotations may supply labels; retain their hashes and mark
the fallback. Never substitute a newly projected volume.

Stimulus hierarchy, fixed before inference:

1. `original_en`: the original 15 D and 15 S items, translated into English;
   `original_es` is a separately reported sensitivity analysis. Items are unpaired.
2. `category_faithful`: if Appendix 1 cannot be obtained, 15 unique D messages
   (authority, imperative, now) and 15 unique S messages (question, perhaps).
   Each has 10–30 words and a distinct outcome clause. Label this conceptual
   replication; a failure is uninformative about TRIBE fidelity.
3. `tone_pairs`: secondary WS08 S9, 15 pairs with matched word counts and
   differently worded action clauses. Never treat the original items as paired.
4. All analyses require 15 long (25–30 words) and 15 short (10–14 words)
   neutral health statements as the specificity control.

Texts remain outside git. Before inference, commit a JSON lock containing only
`stimuli_set` and `items: [{id, condition, sha256, n_words, pair_id?}]` to
`docs/validation/ws11_stimuli_<set>.json`. SHA-256 is over the exact UTF-8 text
file bytes. Conditions are `D`, `S`, `long`, `short`. No stimulus IDs or hashes
exist yet; do not invent them. An external manifest adds `text_file` and
`prediction_file` to those same records. No text is copied into reports.

Item maps are signed temporal means. Compute Welch T, raw D−S means, and the
D-coefficient T in `m ~ 1 + D + n_words`. No absolute-value transform. Masks:
`valid = labels > 0 & T != 0 & finite(T)`; primary = valid minus Vis/SomMot;
secondary = valid with |T| > 3.1. Reproduce 1,743 medial-wall and 17,934 valid
vertices; report discrepancies. Report Spearman (primary), Pearson, significant
vertex sign agreement, and corresponding long−short correlations.

Unpaired sets use 10,000 random balanced label permutations (seed 0), not all
C(30,15) assignments. Pairs use all 32,768 sign flips (within-pair label swaps),
recomputing the same Welch statistic.
Use one-sided permutation p for primary-mask Spearman and significant-vertex
sign agreement. Report both nulls separately when both sets are run.
`agree` requires r > 0, both p < .05, and r(D−S) > r(long−short).
Otherwise `no_agreement`, or `uninformative` for `category_faithful`.

Only if primary r > 0, export TRIBE's own D−S T values, restricted to parcels
whose measured/predicted means have the same sign and whose measured |mean T| > 1.
Export float16[20484] plus the specified provenance sidecar. Compute leave-one-item
out D/S projection AUC and its label-permutation p with each fold rebuilt without
the held-out item. It is monitoring only, with score/selection weight zero.
Never use the measured T map itself as the runtime template.

Provenance retains the source discrepancies: n=33 (NeuroVault), 37 (paper), 35
(OSF SPM script); paper display 6 s versus onsets 8 s; neutral trials are at
900.632–972.683 s and are not used. Language likely Spanish, unconfirmed.
The original item appendix is not present in the supplied manuscript; the local
OpenNeuro events TSV is empty. Neither is evidence of original item recovery.

## 11.2: fixed protocol, execution pending

Source: OpenNeuro ds003020 (Lebel 2023), English story listening, CC0. Seven
stimulus-held-out validation stories: `birthofanation`, `exorcism`, `food`,
`golfclubbing`, `jugglingandjesus`, `lifeanddeathontheoregontrail`,
`thepostmanalwayscalls`. Subjects were in training; the best checkpoint was
selected on validation, so estimates are optimistic. `wheretheressmoke` is
**training**, not held out: SplitEvents overwrites loader splits, deterministic
SHA-256 seed 42 assigns it train (0.800 < .9). Use it only for ceilings and the
seen-versus-unseen stimulus diagnostic, never as the fidelity source.

Verify per-subject S3 availability before download. Take only UTS01–03's seven
val stories and ten repeated-story runs, UTS04–08's exorcism and common repeat
runs 1–5, T1w and TextGrids. Missing runs must be disclosed; no silent subject
substitution. Use fMRIPrep 23.x `--fs-no-reconall
--output-spaces MNI152NLin6Asym:res-2`, no GSR. Project with TRIBE's own
`TribeSurfaceProjector(mesh="fsaverage5", radius=3, kind="ball",
center_depth=.5, extract_fsaverage_from_mni=True)`, then detrend and z-score each
run. Use signed WS07 ROI/location/parcel means; discard first/last 10 s.

Read TextGrid Word events using tribev2's `_create_audio_events`. Match transcript
tokens from `engine._WORD_RE` via SequenceMatcher; exclude stories below 95%
matching. Real onsets are +10 s relative to fMRI. Run text-only `real` (A) and
`product` synthetic .40-s (B); optional audio+text (C) is descriptive only.
Only B determines asset labels. Map product rows to real word time using
`synthetic_to_real`. BOLD t=k*2+1 s, checking slice-time reference in its sidecar;
sample prediction at q=t−10−5−lag_res, without extrapolation.

Choose **one global** lag from −4 to +6 s in .5-s steps by the positive-control
maximum on food, golfclubbing, birthofanation. Freeze it for the other four
stories and exorcism; report fitting and evaluation stories separately. Never
maximize per parcel. Expected lag is near zero (the model already includes 5 s).

Ceiling is sqrt(reliability) of a **single-presentation group mean**, estimated
as mean pairwise correlation between run-wise group means of the same subjects:
UTS01–03's ten repeats for seven-story analysis, UTS01–08's common five repeats
for exorcism sensitivity. Also report exorcism 4-vs-4 subject split-half with
Spearman–Brown. Bootstrap repeats for reliability CI (2,000 resamples, seed 0).
Reliability < .1 or a CI containing zero means `unmeasurable`, factor 1.

Per story: r, r/ceiling, 1,000 circular shifts with shifts at least 20 samples
from either end (seed 0). Combine story p by Fisher and BH-FDR independently
for nine WS07 ROIs, all K2(b) locations, and 400 descriptive parcels. Aggregate
r/ceiling by median and story-bootstrap CI (2,000, seed 0). Record tuning-story
inclusion sensitivity to expose lag-selection optimism.

Positive control: left temporal `*_LH_*TempPar*` and `*_LH_Default_Temp*`
parcels must reach median product r/ceiling >= .3. Otherwise mark
`pipeline_suspect` and do not write an asset (pipeline error detector only).
Report seen-versus-unseen repeated-story fidelity, real minus product, and
audio+text minus real if C exists. A missing C is explicitly `not_run`.

Labels: `ok` iff q < .05 and product r/ceiling >= .3; `unmeasurable` as above;
otherwise `weak`. f=clip((r/ceiling)/.3, .25, 1), unmeasurable/missing => 1.
These are unvalidated scaling priors, never process/recipient labels.

## Assets, runtime and release

Only 11.2 may produce `tribe_service/roi_assets/roi_fidelity.json`: schema 1,
validated false; thresholds .3/.25 tagged unvalidated_prior; atlas/prereg hashes;
Llama-3.2-3B/HF/VCS/package/image/timing provenance; source `lebel2023_val`,
subjects_in_training true, stimuli_in_training false; finite ROI/location rows;
network summaries; separately reported HV summary. No placeholder measured asset.
K2(a) and K2(b) keys come only from `load_roi_assets()`. Runtime modes on/shadow/off
default on; only on applies f. Shadow/off preserve `would_be`. Multiply only the
numerator in K4; never alter the denominator or override WS08's reversed => 0.
Keep the K8 encoder factor separate. No asset => byte-identical scores.

First measured-asset PR supplies methodology v2.4 and corpus shadow/on median
and p95 |score delta|, includes the measured report, and reviews provenance
against current timing/atlas. Registration must be committed before inference;
this implementation leaves changes uncommitted for supervisor review.

Backlog only: Li 2026 Study 1 needs Chinese ASR word timing (not direct-text
events), GSR/run z-score and Schaefer-200 names; it never supplies factors.
Huskey gPPI MFG–SPL (high-risk subgroup only) belongs to WS10, requiring data.
Ntoumanis is excluded; do not download ds005518 for WS11.
