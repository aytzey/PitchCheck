# Turkish Stage 0b — meta-llama/Llama-3.2-3B

150 individually authored Turkish paraphrases: 30 families ×5, ten families /
50 texts per ≤60 / 61–150 / 151–240-word bucket. Each of the five English
channels has two families per bucket. Business and personal proposals are
authored directly in Turkish; this is not a translated English comparison.
Facts, requests and addressee are preserved within each family; vocabulary
and sentence structure are rewritten. Shared diversity and ≤10% within-family
length guards pass. Synthetic text only; no customer data.

Scope: 0b paraphrase reliability/noise only. There is no Turkish 0a deterministic
retest, independent 0c corpus, 0d structure study or new 0e recommendation.
English 0b has identical counts; its independent references are not replicated
or pooled into the Turkish floor. The pooled Turkish length fit has n=150 and
is not installed as WS07 score units. No independent classifier or owner review
is claimed; documented author review preceded inference. This remains a limit.

Rows: 150; excluded: 0; checkpoint encoder compatible: True.
Input SHA-256: `63dbea48549e03d6d96d5cecbb50115c75863a5fed9c86a43b840c4f0f352929`.
Atlas: `c5ca6bb873d2e0d1`; run anchor commit (implementation initially uncommitted): `bd2e7f97bed16a286b19cbcb9339f32584552846`.

ICC(1) uses whole families; CI95 uses 2,000 family bootstrap draws, seed 0.
MDD=1.96√2 × sigma_within. Columns compare separately authored language sets,
not a paired translation experiment or a test of Turkish training coverage.

| Quantity | TR ICC(1) [CI95] | EN ICC(1) | TR sigma_w | EN sigma_w | TR MDD | EN MDD |
|---|---|---:|---:|---:|---:|---:|
| axis:encoding_attention | 0.889354 [0.824887, 0.922061] | 0.845028 | 3.48958928 | 3.65687025 | 9.67264799 | 10.13632718 |
| axis:processing_fluency | 0.426781 [0.246239, 0.564698] | 0.516281 | 4.65415585 | 4.33992703 | 12.90066183 | 12.02966400 |
| axis:reward_affect | 0.858092 [0.776463, 0.902337] | 0.831515 | 3.54957744 | 3.20501690 | 9.83892669 | 8.88385360 |
| axis:self_value | 0.448970 [0.291492, 0.575157] | 0.469937 | 4.84762485 | 4.26558320 | 13.43693054 | 11.82359339 |
| axis:social_sharing | 0.824489 [0.737991, 0.874794] | 0.753405 | 3.59052921 | 3.25907451 | 9.95243920 | 9.03369365 |
| evidence_score | 0.764568 [0.646573, 0.830831] | 0.724047 | 3.63861389 | 3.43064379 | 10.08572313 | 9.50925944 |
| evidence_score_with_roi | 0.755159 [0.643886, 0.829626] | 0.760173 | 3.89622267 | 3.70413487 | 10.79977823 | 10.26733802 |
| neural_score_from_signals | 0.787372 [0.676732, 0.847910] | 0.740273 | 3.56544912 | 3.38356385 | 9.88292075 | 9.37876049 |
| quality_weight | 0.000000 [0.000000, 0.000000] | 0.000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| response_shape:half:closing | 0.905436 [0.860193, 0.928067] | 0.813102 | 0.04215710 | 0.04281353 | 0.11685353 | 0.11867305 |
| response_shape:half:continuity | 0.866199 [0.784160, 0.908653] | 0.769795 | 0.07494045 | 0.07631109 | 0.20772433 | 0.21152355 |
| response_shape:half:max_relative_drop | 0.866199 [0.784160, 0.908653] | 0.769795 | 0.07494045 | 0.07631109 | 0.20772433 | 0.21152355 |
| response_shape:half:opening | 0.897809 [0.848710, 0.922050] | 0.812032 | 0.04313430 | 0.04207361 | 0.11956217 | 0.11662211 |
| response_shape:quarter:closing | 0.925588 [0.881078, 0.946220] | 0.897144 | 0.06492612 | 0.06432070 | 0.17996603 | 0.17828787 |
| response_shape:quarter:continuity | 0.844141 [0.754143, 0.894102] | 0.846728 | 0.07351017 | 0.06328606 | 0.20375979 | 0.17542002 |
| response_shape:quarter:max_relative_drop | 0.844141 [0.754143, 0.894102] | 0.846728 | 0.07351017 | 0.06328606 | 0.20375979 | 0.17542002 |
| response_shape:quarter:opening | 0.788535 [0.704722, 0.843818] | 0.761704 | 0.06293803 | 0.05497690 | 0.17445531 | 0.15238820 |
| roi:dmpfc | 0.829853 [0.760247, 0.866465] | 0.886308 | 0.01947800 | 0.01766837 | 0.05399025 | 0.04897423 |
| roi:dmpfc_chua2011 | 0.831294 [0.761537, 0.870887] | 0.900329 | 0.02440915 | 0.02089917 | 0.06765871 | 0.05792953 |
| roi:falk2010jocn_core | 0.712262 [0.583246, 0.787095] | 0.866116 | 0.02128664 | 0.01876097 | 0.05900355 | 0.05200276 |
| roi:ifg47_l_chua2011 | 0.753801 [0.593469, 0.839935] | 0.859788 | 0.01802657 | 0.01752258 | 0.04996709 | 0.04857012 |
| roi:insula | 0.761482 [0.650995, 0.824564] | 0.860020 | 0.01440107 | 0.01459265 | 0.03991772 | 0.04044876 |
| roi:insula_right | 0.710229 [0.580903, 0.779894] | 0.885443 | 0.01301349 | 0.01301102 | 0.03607155 | 0.03606471 |
| roi:ipl_ifg_how_gt_why_vezich2017_l | 0.603068 [0.369910, 0.740905] | 0.902683 | 0.00956881 | 0.01003214 | 0.02652339 | 0.02780766 |
| roi:l_angular_chua2011 | 0.847156 [0.769821, 0.887998] | 0.877036 | 0.02300607 | 0.02190493 | 0.06376956 | 0.06071736 |
| roi:l_ifg | 0.601480 [0.417140, 0.716064] | 0.848234 | 0.01541462 | 0.01770878 | 0.04272714 | 0.04908623 |
| roi:lateral_temporal | 0.856529 [0.754023, 0.906980] | 0.945145 | 0.02429849 | 0.02112970 | 0.06735197 | 0.05856855 |
| roi:mofc | 0.711695 [0.570725, 0.794716] | 0.809381 | 0.01184272 | 0.01137730 | 0.03282634 | 0.03153628 |
| roi:mpfc_falk2010 | 0.768976 [0.646419, 0.842270] | 0.937241 | 0.00927997 | 0.00678969 | 0.02572277 | 0.01882005 |
| roi:pcc_precuneus | 0.538681 [0.360111, 0.668670] | 0.911320 | 0.02552180 | 0.02188136 | 0.07074283 | 0.06065202 |
| roi:pifg_vezich2017 | 0.520921 [0.343872, 0.635698] | 0.900590 | 0.01379035 | 0.01677998 | 0.03822490 | 0.04651173 |
| roi:precuneus_chua2011 | 0.580450 [0.374390, 0.713027] | 0.935139 | 0.02420881 | 0.01948613 | 0.06710339 | 0.05401279 |
| roi:r_ifg_ai | 0.710183 [0.576816, 0.796490] | 0.924033 | 0.01377791 | 0.01008828 | 0.03819043 | 0.02796329 |
| roi:r_tpj | 0.814192 [0.722213, 0.860082] | 0.878910 | 0.02781526 | 0.02607707 | 0.07709997 | 0.07228194 |
| roi:ripl_vezich2017 | 0.740162 [0.551044, 0.849603] | 0.919447 | 0.01063842 | 0.00814169 | 0.02948818 | 0.02256762 |
| roi:rmpfc_chua2009 | 0.712348 [0.570134, 0.788948] | 0.905133 | 0.01245209 | 0.00917171 | 0.03451544 | 0.02542268 |
| roi:tp_l | 0.627632 [0.492269, 0.703465] | 0.813055 | 0.02991162 | 0.02314453 | 0.08291079 | 0.06415338 |
| roi:tp_r | 0.706856 [0.547020, 0.796236] | 0.901356 | 0.02502000 | 0.01998455 | 0.06935190 | 0.05539435 |
| roi:vmpfc_chua2009 | 0.450355 [0.257883, 0.579643] | 0.690727 | 0.01889870 | 0.01575515 | 0.05238452 | 0.04367106 |
| roi:vmpfc_frontopolar | 0.733846 [0.611309, 0.810945] | 0.906540 | 0.00777038 | 0.00620781 | 0.02153839 | 0.01720716 |
| signal:attention_capture | 0.851912 [0.785765, 0.888708] | 0.765253 | 4.78275882 | 5.06178549 | 13.25713107 | 14.03055355 |
| signal:cognitive_friction | 0.218675 [0.052584, 0.354161] | 0.422722 | 5.57539943 | 4.92555862 | 15.45421877 | 13.65295193 |
| signal:emotional_engagement | 0.879126 [0.798188, 0.923936] | 0.873653 | 4.37136216 | 3.49409742 | 12.11679773 | 9.68514392 |
| signal:memorability | 0.822225 [0.724132, 0.881769] | 0.796655 | 5.48617286 | 5.04787798 | 15.20689532 | 13.99200390 |
| signal:personal_relevance | 0.297150 [0.083247, 0.459967] | 0.365327 | 6.21549965 | 5.43553679 | 17.22848605 | 15.06653931 |
| signal:social_proof_potential | 0.824243 [0.751257, 0.867803] | 0.720604 | 4.11420343 | 3.67593955 | 11.40399007 | 10.18918458 |

## Product consumption

Runtime file: `tribe_service/roi_assets/roi_reference_meta-llama__Llama-3.2-3B_tr.json`.
The existing loader checks encoder, language, direct event version, input mode,
finite noise/length statistics and atlas. WS09-B takes only the two measured
location MDDs. `floor_source=paraphrase_noise_floor_tr`; applied provenance
ends in `;measured_turkish_paraphrase_floor`. The language reduction is removed;
K8 still applies independently to Hermes. `selection_score_without_roi`, K4,
the ±.5 point ceiling and eligibility rules are unchanged. WS12 skips Turkish.

`PITCHCHECK_ROI_INSILICO_TUNING=0` restores the old default floor/half language
weight. Missing or incompatible Turkish assets do the same; no English floor
is silently reused for Turkish. Existing points=0 rollback remains available.
WS07 English units and primary suite check multipliers are not overwritten.
Restart workers after installing references to clear the existing loader cache.

## Local execution

One local RTX 3070 GPU process at a time: Hermes then Llama, supplied env files
and venv-gpu, product `engine.score_text` and cached offline model snapshots.
The production direct-event language metadata remains `english`; actual stimulus
language is separately `tr`. No translation, audio/ASR or new timing is added.
Float32 NPZ caches are under `scratchpad/gpu/runs/turkish/cache/`; float16 audit
matrices are replayed as in the English analysis. CPU replay restores recorded
encoder identity. No fallback/retry, paid API or production host was used.

The scratch-only adapter checks a pre-inference SHA256 lock.
Unlike hypothesis suites, this descriptive Turkish scale was not committed
before inference; author review is recorded without claiming independent review.
Final input bytes still match that lock. The lock, adapter, manifests, NPZs
and complete logs stay outside Git. No inference was repeated for analysis.

Full quantities, provenance and unused descriptive fields: [2026-10-16-stage0-tr-meta-llama__Llama-3.2-3B.json](2026-10-16-stage0-tr-meta-llama__Llama-3.2-3B.json).
