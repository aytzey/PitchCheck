# WS08 stage0 — NousResearch/Hermes-3-Llama-3.2-3B, 2026-10-16

Rows: 405; excluded: 0. Real TRIBE on local RTX 3070.
PitchCheck revision: `05f42df417bf3e92dfc467d67c72cda19125153f`; atlas: `c5ca6bb873d2e0d1`.

Full numbers and provenance: [2026-10-16-stage0-NousResearch__Hermes-3-Llama-3.2-3B.json](2026-10-16-stage0-NousResearch__Hermes-3-Llama-3.2-3B.json).

Effects describe variation across authored texts in an average-subject prediction; they are not human-subject effect sizes or recipient mental states.

0a: five texts × three fresh, uncached inferences.
Retest relative L2 maximum: 0; sanity: True.

0b/0c: 30 families × five forms; 300 pooled reference pitches; 30 neutral controls. MDD = 1.96√2 σw. Residual quantiles p05/p25/p50/p75/p95 and length-model coefficients are in the JSON.

| Quantity | ICC(1) | σ within | MDD | Length R² | Residual SD |
|---|---:|---:|---:|---:|---:|
| axis:encoding_attention | 0.877712 | 3.43393 | 9.51838 | 0.613838 | 5.67096 |
| axis:processing_fluency | 0.546763 | 4.04862 | 11.2222 | 0.286337 | 4.87999 |
| axis:reward_affect | 0.861377 | 3.07547 | 8.52476 | 0.598694 | 4.87699 |
| axis:self_value | 0.482815 | 3.93554 | 10.9088 | 0.190525 | 4.95344 |
| axis:social_sharing | 0.801073 | 3.15634 | 8.74894 | 0.493135 | 4.69028 |
| evidence_score | 0.779176 | 3.15821 | 8.75412 | 0.501914 | 4.45432 |
| evidence_score_with_roi | 0.771179 | 3.15906 | 8.75646 | 0.504909 | 4.3913 |
| neural_score_from_signals | 0.795136 | 3.12171 | 8.65294 | 0.511039 | 4.50941 |
| quality_weight | 0 | 0 | 0 | 0 | 1.66771e-16 |
| response_shape:half:closing | 0.820839 | 0.043785 | 0.121366 | 0.553973 | 0.0631147 |
| response_shape:half:continuity | 0.787179 | 0.0771562 | 0.213866 | 0.546905 | 0.101338 |
| response_shape:half:max_relative_drop | 0.787179 | 0.0771562 | 0.213866 | 0.546905 | 0.101338 |
| response_shape:half:opening | 0.818228 | 0.0432345 | 0.11984 | 0.543923 | 0.0627584 |
| response_shape:quarter:closing | 0.888626 | 0.0689333 | 0.191073 | 0.589495 | 0.118946 |
| response_shape:quarter:continuity | 0.829901 | 0.0706966 | 0.195961 | 0.525018 | 0.106657 |
| response_shape:quarter:max_relative_drop | 0.829901 | 0.0706966 | 0.195961 | 0.525018 | 0.106657 |
| response_shape:quarter:opening | 0.77422 | 0.0569004 | 0.15772 | 0.341172 | 0.088544 |
| roi:dmpfc | 0.849586 | 0.0201292 | 0.0557953 | 0.0979165 | 0.0466995 |
| roi:dmpfc_chua2011 | 0.869336 | 0.0235362 | 0.0652389 | 0.124658 | 0.0584887 |
| roi:falk2010jocn_core | 0.836785 | 0.0207285 | 0.0574565 | 0.0109997 | 0.0472855 |
| roi:ifg47_l_chua2011 | 0.80168 | 0.019778 | 0.0548218 | 0.000365053 | 0.0432619 |
| roi:insula | 0.848673 | 0.0152099 | 0.0421597 | 0.0148516 | 0.033907 |
| roi:insula_right | 0.868408 | 0.0134672 | 0.0373291 | 0.095214 | 0.0307229 |
| roi:ipl_ifg_how_gt_why_vezich2017_l | 0.870025 | 0.0100056 | 0.027734 | 0.0899663 | 0.023481 |
| roi:l_angular_chua2011 | 0.849274 | 0.0226879 | 0.0628876 | 0.260736 | 0.047758 |
| roi:l_ifg | 0.791178 | 0.0194815 | 0.054 | 0.277808 | 0.0347857 |
| roi:lateral_temporal | 0.934374 | 0.023758 | 0.0658537 | 0.652631 | 0.0528763 |
| roi:mofc | 0.800258 | 0.0112239 | 0.0311111 | 0.0781546 | 0.0245583 |
| roi:mpfc_falk2010 | 0.924595 | 0.00778619 | 0.0215822 | 0.367855 | 0.0223919 |
| roi:pcc_precuneus | 0.889904 | 0.0224635 | 0.0622658 | 0.00939714 | 0.060918 |
| roi:pifg_vezich2017 | 0.865701 | 0.0176449 | 0.0489093 | 0.480214 | 0.0323928 |
| roi:precuneus_chua2011 | 0.920405 | 0.0201163 | 0.0557595 | 1.24007e-05 | 0.0638392 |
| roi:r_ifg_ai | 0.911527 | 0.00994601 | 0.0275689 | 0.575885 | 0.021409 |
| roi:r_tpj | 0.866024 | 0.0267389 | 0.0741164 | 0.490855 | 0.0502024 |
| roi:ripl_vezich2017 | 0.912493 | 0.00836848 | 0.0231962 | 0.165853 | 0.0230119 |
| roi:rmpfc_chua2009 | 0.880307 | 0.00971967 | 0.0269415 | 0.260231 | 0.0232785 |
| roi:tp_l | 0.795202 | 0.0238096 | 0.0659969 | 0.00701368 | 0.0496347 |
| roi:tp_r | 0.885233 | 0.0221015 | 0.0612623 | 0.195572 | 0.0535988 |
| roi:vmpfc_chua2009 | 0.672642 | 0.0155164 | 0.0430092 | 0.00039826 | 0.0275697 |
| roi:vmpfc_frontopolar | 0.902674 | 0.00631442 | 0.0175027 | 0.221557 | 0.0177419 |
| signal:attention_capture | 0.798015 | 4.96016 | 13.7489 | 0.348791 | 8.2398 |
| signal:cognitive_friction | 0.412318 | 4.74825 | 13.1615 | 0.140627 | 5.59278 |
| signal:emotional_engagement | 0.882138 | 3.56691 | 9.88697 | 0.671125 | 5.63396 |
| signal:memorability | 0.828302 | 4.82293 | 13.3685 | 0.633553 | 6.78677 |
| signal:personal_relevance | 0.261903 | 5.02228 | 13.921 | 0.0360441 | 6.16337 |
| signal:social_proof_potential | 0.753468 | 3.80412 | 10.5445 | 0.431908 | 5.36479 |

0d: punctuation-free minus scrambled (30 matched pairs).

| Quantity | Mean Δ | Bootstrap CI95 |
|---|---:|---|
| evidence_score | 4.50913 | [1.588321666666667, 7.40827] |
| roi:lateral_temporal | 0.062978 | [0.044624683333333325, 0.08636771666666666] |
| roi:l_ifg | 0.0293417 | [0.015261875000000005, 0.04554294999999999] |
| roi:tp_l | 0.0958683 | [0.07953005833333333, 0.11507134166666665] |
| roi:tp_r | 0.0689693 | [0.05317158333333334, 0.08778840833333332] |

0e: current versus ROI-augmented evidence score, recomputed from saved float16 matrices after installing the measured reference (no GPU rerun).
Spearman ρ: 0.994364; ICC current: 0.779176; ICC with ROI: 0.771179; paraphrase rank-change rate: 0.0266667; half-weight recommendation: False.

vmPFC/proxy correlations: `{'axis:self_value': -0.1524774107725131, 'signal:personal_relevance': 0.2480866879961808}`.

Paraphrase stability flags shrink the literature-prior numerators; deterministic retests do not establish reliability. The product uses the encoder-specific length-residual scale and paraphrase noise floors.
