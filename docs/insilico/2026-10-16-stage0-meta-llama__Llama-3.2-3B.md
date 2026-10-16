# WS08 stage0 — meta-llama/Llama-3.2-3B, 2026-10-16

Rows: 405; excluded: 0. Real TRIBE on local RTX 3070.
PitchCheck revision: `05f42df417bf3e92dfc467d67c72cda19125153f`; atlas: `c5ca6bb873d2e0d1`.

Full numbers and provenance: [2026-10-16-stage0-meta-llama__Llama-3.2-3B.json](2026-10-16-stage0-meta-llama__Llama-3.2-3B.json).

Effects describe variation across authored texts in an average-subject prediction; they are not human-subject effect sizes or recipient mental states.

0a: five texts × three fresh, uncached inferences.
Retest relative L2 maximum: 0; sanity: True.

0b/0c: 30 families × five forms; 300 pooled reference pitches; 30 neutral controls. MDD = 1.96√2 σw. Residual quantiles p05/p25/p50/p75/p95 and length-model coefficients are in the JSON.

| Quantity | ICC(1) | σ within | MDD | Length R² | Residual SD |
|---|---:|---:|---:|---:|---:|
| axis:encoding_attention | 0.845028 | 3.65687 | 10.1363 | 0.581589 | 5.58408 |
| axis:processing_fluency | 0.516281 | 4.33993 | 12.0297 | 0.195609 | 5.27259 |
| axis:reward_affect | 0.831515 | 3.20502 | 8.88385 | 0.551457 | 4.86377 |
| axis:self_value | 0.469937 | 4.26558 | 11.8236 | 0.095905 | 5.38607 |
| axis:social_sharing | 0.753405 | 3.25907 | 9.03369 | 0.40806 | 4.68483 |
| evidence_score | 0.724047 | 3.43064 | 9.50926 | 0.404623 | 4.68614 |
| evidence_score_with_roi | 0.760173 | 3.70413 | 10.2673 | 0.24607 | 6.48252 |
| neural_score_from_signals | 0.740273 | 3.38356 | 9.37876 | 0.419141 | 4.69611 |
| quality_weight | 0 | 0 | 0 | 0 | 1.66771e-16 |
| response_shape:half:closing | 0.813102 | 0.0428135 | 0.118673 | 0.571346 | 0.0611744 |
| response_shape:half:continuity | 0.769795 | 0.0763111 | 0.211524 | 0.539461 | 0.0995274 |
| response_shape:half:max_relative_drop | 0.769795 | 0.0763111 | 0.211524 | 0.539461 | 0.0995274 |
| response_shape:half:opening | 0.812032 | 0.0420736 | 0.116622 | 0.562039 | 0.0608342 |
| response_shape:quarter:closing | 0.897144 | 0.0643207 | 0.178288 | 0.636683 | 0.111123 |
| response_shape:quarter:continuity | 0.846728 | 0.0632861 | 0.17542 | 0.559952 | 0.098244 |
| response_shape:quarter:max_relative_drop | 0.846728 | 0.0632861 | 0.17542 | 0.559952 | 0.098244 |
| response_shape:quarter:opening | 0.761704 | 0.0549769 | 0.152388 | 0.365512 | 0.0849664 |
| roi:dmpfc | 0.886308 | 0.0176684 | 0.0489742 | 0.123649 | 0.0473959 |
| roi:dmpfc_chua2011 | 0.900329 | 0.0208992 | 0.0579295 | 0.145101 | 0.0593506 |
| roi:falk2010jocn_core | 0.866116 | 0.018761 | 0.0520028 | 0.0031051 | 0.0482471 |
| roi:ifg47_l_chua2011 | 0.859788 | 0.0175226 | 0.0485701 | 0.00382554 | 0.0457282 |
| roi:insula | 0.86002 | 0.0145926 | 0.0404488 | 0.0319169 | 0.0339276 |
| roi:insula_right | 0.885443 | 0.013011 | 0.0360647 | 0.123964 | 0.0314487 |
| roi:ipl_ifg_how_gt_why_vezich2017_l | 0.902683 | 0.0100321 | 0.0278077 | 0.133968 | 0.0260275 |
| roi:l_angular_chua2011 | 0.877036 | 0.0219049 | 0.0607174 | 0.311178 | 0.0504836 |
| roi:l_ifg | 0.848234 | 0.0177088 | 0.0490862 | 0.280007 | 0.0365351 |
| roi:lateral_temporal | 0.945145 | 0.0211297 | 0.0585685 | 0.649596 | 0.0520466 |
| roi:mofc | 0.809381 | 0.0113773 | 0.0315363 | 0.0520679 | 0.0250428 |
| roi:mpfc_falk2010 | 0.937241 | 0.00678969 | 0.0188201 | 0.263599 | 0.0231334 |
| roi:pcc_precuneus | 0.91132 | 0.0218814 | 0.060652 | 0.0309927 | 0.0650581 |
| roi:pifg_vezich2017 | 0.90059 | 0.01678 | 0.0465117 | 0.483217 | 0.035136 |
| roi:precuneus_chua2011 | 0.935139 | 0.0194861 | 0.0540128 | 0.0100329 | 0.0678485 |
| roi:r_ifg_ai | 0.924033 | 0.0100883 | 0.0279633 | 0.575864 | 0.0230976 |
| roi:r_tpj | 0.87891 | 0.0260771 | 0.0722819 | 0.502129 | 0.0520797 |
| roi:ripl_vezich2017 | 0.919447 | 0.00814169 | 0.0225676 | 0.113136 | 0.0241291 |
| roi:rmpfc_chua2009 | 0.905133 | 0.00917171 | 0.0254227 | 0.319551 | 0.0239911 |
| roi:tp_l | 0.813055 | 0.0231445 | 0.0641534 | 0.0170776 | 0.0512331 |
| roi:tp_r | 0.901356 | 0.0199846 | 0.0553943 | 0.171069 | 0.0537315 |
| roi:vmpfc_chua2009 | 0.690727 | 0.0157552 | 0.0436711 | 5.99086e-05 | 0.0276966 |
| roi:vmpfc_frontopolar | 0.90654 | 0.00620781 | 0.0172072 | 0.131511 | 0.0185922 |
| signal:attention_capture | 0.765253 | 5.06179 | 14.0306 | 0.331309 | 7.97794 |
| signal:cognitive_friction | 0.422722 | 4.92556 | 13.653 | 0.093159 | 5.86711 |
| signal:emotional_engagement | 0.873653 | 3.4941 | 9.68514 | 0.654919 | 5.48134 |
| signal:memorability | 0.796655 | 5.04788 | 13.992 | 0.583301 | 6.89585 |
| signal:personal_relevance | 0.365327 | 5.43554 | 15.0665 | 0.0763686 | 6.65947 |
| signal:social_proof_potential | 0.720604 | 3.67594 | 10.1892 | 0.35531 | 5.14889 |

0d: punctuation-free minus scrambled (30 matched pairs).

| Quantity | Mean Δ | Bootstrap CI95 |
|---|---:|---|
| evidence_score | 4.43493 | [2.0144658333333343, 6.873562499999997] |
| roi:lateral_temporal | 0.0638247 | [0.04526600833333334, 0.08497881666666665] |
| roi:l_ifg | 0.023452 | [0.007455466666666667, 0.040824574999999974] |
| roi:tp_l | 0.100215 | [0.08368730833333335, 0.11870116666666668] |
| roi:tp_r | 0.0712247 | [0.05551838333333334, 0.08879704166666665] |

0e: current versus ROI-augmented evidence score, recomputed from saved float16 matrices after installing the measured reference (no GPU rerun).
Spearman ρ: 0.791243; ICC current: 0.724047; ICC with ROI: 0.760173; paraphrase rank-change rate: 0.116667; half-weight recommendation: False.

vmPFC/proxy correlations: `{'axis:self_value': 0.11881450431714717, 'signal:personal_relevance': 0.41372888824288645}`.

Paraphrase stability flags shrink the literature-prior numerators; deterministic retests do not establish reliability. The product uses the encoder-specific length-residual scale and paraphrase noise floors.
