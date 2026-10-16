# Turkish Stage 0b — NousResearch/Hermes-3-Llama-3.2-3B

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

Rows: 150; excluded: 0; checkpoint encoder compatible: False.
Input SHA-256: `63dbea48549e03d6d96d5cecbb50115c75863a5fed9c86a43b840c4f0f352929`.
Atlas: `c5ca6bb873d2e0d1`; run anchor commit (implementation initially uncommitted): `bd2e7f97bed16a286b19cbcb9339f32584552846`.

ICC(1) uses whole families; CI95 uses 2,000 family bootstrap draws, seed 0.
MDD=1.96√2 × sigma_within. Columns compare separately authored language sets,
not a paired translation experiment or a test of Turkish training coverage.

| Quantity | TR ICC(1) [CI95] | EN ICC(1) | TR sigma_w | EN sigma_w | TR MDD | EN MDD |
|---|---|---:|---:|---:|---:|---:|
| axis:encoding_attention | 0.898155 [0.837106, 0.925434] | 0.877712 | 3.61101094 | 3.43393360 | 10.00921166 | 9.51837833 |
| axis:processing_fluency | 0.418474 [0.226384, 0.558395] | 0.546763 | 4.59213458 | 4.04861705 | 12.72874765 | 11.22219391 |
| axis:reward_affect | 0.887963 [0.821316, 0.921253] | 0.861377 | 3.42970844 | 3.07546744 | 9.50666678 | 8.52476083 |
| axis:self_value | 0.486708 [0.298879, 0.628246] | 0.482815 | 4.63742026 | 3.93553893 | 12.85427315 | 10.90875736 |
| axis:social_sharing | 0.877235 [0.818022, 0.910928] | 0.801073 | 3.35243692 | 3.15634282 | 9.29248106 | 8.74893594 |
| evidence_score | 0.809923 [0.699704, 0.867562] | 0.779176 | 3.53384369 | 3.15821215 | 9.79531496 | 8.75411746 |
| evidence_score_with_roi | 0.809831 [0.704913, 0.864047] | 0.757038 | 3.65060524 | 3.19547175 | 10.11896145 | 8.85739579 |
| neural_score_from_signals | 0.831769 [0.735019, 0.880699] | 0.795136 | 3.45871964 | 3.12170996 | 9.58708171 | 8.65293853 |
| quality_weight | 0.000000 [0.000000, 0.000000] | 0.000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| response_shape:half:closing | 0.916138 [0.872839, 0.938382] | 0.820839 | 0.04111143 | 0.04378496 | 0.11395508 | 0.12136572 |
| response_shape:half:continuity | 0.881008 [0.811214, 0.921380] | 0.787179 | 0.07615387 | 0.07715619 | 0.21108776 | 0.21386604 |
| response_shape:half:max_relative_drop | 0.881008 [0.811214, 0.921380] | 0.787179 | 0.07615387 | 0.07715619 | 0.21108776 | 0.21386604 |
| response_shape:half:opening | 0.909053 [0.862984, 0.932129] | 0.818228 | 0.04201038 | 0.04323447 | 0.11644683 | 0.11983982 |
| response_shape:quarter:closing | 0.923220 [0.873913, 0.948189] | 0.888626 | 0.06648623 | 0.06893328 | 0.18429041 | 0.19107331 |
| response_shape:quarter:continuity | 0.837287 [0.750285, 0.879059] | 0.829901 | 0.07521931 | 0.07069663 | 0.20849728 | 0.19596106 |
| response_shape:quarter:max_relative_drop | 0.837287 [0.750285, 0.879059] | 0.829901 | 0.07521931 | 0.07069663 | 0.20849728 | 0.19596106 |
| response_shape:quarter:opening | 0.794976 [0.715351, 0.844578] | 0.774220 | 0.06181886 | 0.05690042 | 0.17135313 | 0.15771991 |
| roi:dmpfc | 0.812457 [0.729124, 0.859062] | 0.849586 | 0.01868718 | 0.02012919 | 0.05179822 | 0.05579526 |
| roi:dmpfc_chua2011 | 0.816251 [0.739269, 0.861676] | 0.869336 | 0.02381048 | 0.02353616 | 0.06599927 | 0.06523890 |
| roi:falk2010jocn_core | 0.678914 [0.545480, 0.755554] | 0.836785 | 0.02017849 | 0.02072849 | 0.05593191 | 0.05745645 |
| roi:ifg47_l_chua2011 | 0.715526 [0.545906, 0.815117] | 0.801680 | 0.01717626 | 0.01977798 | 0.04761018 | 0.05482176 |
| roi:insula | 0.762922 [0.639256, 0.832281] | 0.848673 | 0.01379990 | 0.01520992 | 0.03825137 | 0.04215973 |
| roi:insula_right | 0.705416 [0.548585, 0.788663] | 0.868408 | 0.01235992 | 0.01346716 | 0.03425996 | 0.03732906 |
| roi:ipl_ifg_how_gt_why_vezich2017_l | 0.680130 [0.497418, 0.788280] | 0.870025 | 0.00837322 | 0.01000558 | 0.02320937 | 0.02773404 |
| roi:l_angular_chua2011 | 0.822508 [0.727685, 0.875081] | 0.849274 | 0.02101570 | 0.02268788 | 0.05825256 | 0.06288758 |
| roi:l_ifg | 0.541274 [0.346249, 0.668474] | 0.791178 | 0.01485636 | 0.01948152 | 0.04117972 | 0.05400000 |
| roi:lateral_temporal | 0.864176 [0.769104, 0.910147] | 0.934374 | 0.02314098 | 0.02375795 | 0.06414353 | 0.06585368 |
| roi:mofc | 0.687655 [0.511718, 0.788255] | 0.800258 | 0.01052584 | 0.01122393 | 0.02917613 | 0.03111114 |
| roi:mpfc_falk2010 | 0.815374 [0.714532, 0.874969] | 0.924595 | 0.00837717 | 0.00778619 | 0.02322032 | 0.02158222 |
| roi:pcc_precuneus | 0.530886 [0.345440, 0.673881] | 0.889904 | 0.02298124 | 0.02246355 | 0.06370075 | 0.06226577 |
| roi:pifg_vezich2017 | 0.370918 [0.179719, 0.510452] | 0.865701 | 0.01300615 | 0.01764494 | 0.03605120 | 0.04890928 |
| roi:precuneus_chua2011 | 0.611615 [0.399909, 0.742425] | 0.920405 | 0.02112391 | 0.02011630 | 0.05855249 | 0.05575955 |
| roi:r_ifg_ai | 0.661645 [0.507494, 0.765901] | 0.911527 | 0.01279253 | 0.00994601 | 0.03545908 | 0.02756893 |
| roi:r_tpj | 0.755485 [0.631734, 0.819830] | 0.866024 | 0.02693322 | 0.02673888 | 0.07465507 | 0.07411638 |
| roi:ripl_vezich2017 | 0.770594 [0.617593, 0.856738] | 0.912493 | 0.00942797 | 0.00836848 | 0.02613301 | 0.02319624 |
| roi:rmpfc_chua2009 | 0.685708 [0.521227, 0.777497] | 0.880307 | 0.01133409 | 0.00971967 | 0.03141649 | 0.02694154 |
| roi:tp_l | 0.608493 [0.491325, 0.676210] | 0.795202 | 0.02720213 | 0.02380962 | 0.07540047 | 0.06599690 |
| roi:tp_r | 0.703549 [0.543810, 0.795097] | 0.885233 | 0.02348582 | 0.02210154 | 0.06509938 | 0.06126235 |
| roi:vmpfc_chua2009 | 0.434282 [0.221724, 0.580206] | 0.672642 | 0.01643494 | 0.01551639 | 0.04555532 | 0.04300924 |
| roi:vmpfc_frontopolar | 0.752714 [0.629631, 0.830333] | 0.902674 | 0.00684964 | 0.00631442 | 0.01898623 | 0.01750269 |
| signal:attention_capture | 0.863933 [0.804954, 0.901704] | 0.798015 | 4.91058476 | 4.96015608 | 13.61144650 | 13.74885121 |
| signal:cognitive_friction | 0.110706 [-0.042027, 0.250629] | 0.412318 | 5.45010141 | 4.74825172 | 15.10691036 | 13.16148229 |
| signal:emotional_engagement | 0.902253 [0.837765, 0.939579] | 0.882138 | 4.16701488 | 3.56690944 | 11.55037596 | 9.88696854 |
| signal:memorability | 0.835483 [0.733965, 0.892814] | 0.828302 | 5.47483258 | 4.82292644 | 15.17546166 | 13.36847006 |
| signal:personal_relevance | 0.152658 [-0.036905, 0.306189] | 0.261903 | 5.97296306 | 5.02227891 | 16.55620892 | 13.92104690 |
| signal:social_proof_potential | 0.882172 [0.825659, 0.919357] | 0.753468 | 3.78897611 | 3.80411813 | 10.50250594 | 10.54447750 |

## Product consumption

Runtime file: `tribe_service/roi_assets/roi_reference_NousResearch__Hermes-3-Llama-3.2-3B_tr.json`.
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

Full quantities, provenance and unused descriptive fields: [2026-10-16-stage0-tr-NousResearch__Hermes-3-Llama-3.2-3B.json](2026-10-16-stage0-tr-NousResearch__Hermes-3-Llama-3.2-3B.json).
