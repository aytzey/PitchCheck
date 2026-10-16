# WS08 S1 — meta-llama/Llama-3.2-3B, 2026-10-16

Rows: 80; excluded: 0. Real TRIBE on local RTX 3070.
PitchCheck revision: `4a8fcd45ffcb2951fec78319fab988852b74eb7a`; atlas: `c5ca6bb873d2e0d1`.

Full numbers and provenance: [2026-10-16-S1-meta-llama__Llama-3.2-3B.json](2026-10-16-S1-meta-llama__Llama-3.2-3B.json).

Effects describe variation across authored texts in an average-subject prediction; they are not human-subject effect sizes or recipient mental states.

All co-primary tests pass/pass_broad: False.
Downstream contrasts retain uninformative status and multiplier 1 if the pre-registered S2 positive control failed. Raw effects and p values are still reported.

| Contrast | ROI | Primary | n | dz | Mean Δ | CI95 | p Holm | Reverse p Holm | Parcel rank | Verdict | Multiplier |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| T>U_you | dmpfc_chua2011 | True | 20 | 0.0879686 | 0.001537 | [-0.005649737499999993, 0.0090818125] | 0.342066 | 0.693831 | 0.605 | uninformative | 1 |
| T>U_you | precuneus_chua2011 | True | 20 | 0.610037 | 0.009411 | [0.00293085, 0.01582605] | 0.0131987 | 0.0271973 | 0.99 | uninformative | 1 |
| T>U | dmpfc_chua2011 | False | 20 | 0.641386 | 0.0113175 | [0.003877487499999999, 0.018638149999999992] | 0.00859914 | 0.0137986 | 0.7975 | uninformative | 1 |
| T>U | precuneus_chua2011 | False | 20 | 0.688941 | 0.015087 | [0.0046519625, 0.023624537499999997] | 0.00859914 | 0.0137986 | 0.82 | uninformative | 1 |
| IFG_report | r_ifg_ai | False | 20 | 0.312944 | 0.001928 | [-0.0006872624999999995, 0.004563099999999998] | 0.268173 | 0.547145 | 0.8175 | uninformative | 1 |
| IFG_report | l_ifg | False | 20 | -0.08593 | -0.0012595 | [-0.007252874999999999, 0.0051267875] | 1 | 0.69533 | 0.445 | uninformative | 1 |
| genre_sanity | dmpfc_chua2011 | False | 20 | 1.09347 | 0.0991725 | [0.060408187499999995, 0.13879667499999998] | 0.00039996 | 0.00039996 | 0.8575 | uninformative | 1 |
| genre_sanity | precuneus_chua2011 | False | 20 | 1.68789 | 0.194545 | [0.14405092500000002, 0.24234566250000003] | 0.00039996 | 0.00039996 | 0.9625 | uninformative | 1 |
| generic_genre_sanity | dmpfc_chua2011 | False | 20 | 1.03028 | 0.087855 | [0.0520613375, 0.1251156875] | 0.00039996 | 0.00039996 | 0.84 | uninformative | 1 |
| generic_genre_sanity | precuneus_chua2011 | False | 20 | 1.61796 | 0.179458 | [0.1320712125, 0.22648456249999988] | 0.00039996 | 0.00039996 | 0.9575 | uninformative | 1 |
| vmPFC_report | vmpfc_frontopolar | False | 20 | -0.229061 | -0.001385 | [-0.004014087499999999, 0.0011160874999999995] | 1 | 0.626337 | 0.3375 | uninformative | 1 |

Motor/body covariate adjustments and descriptive cost-sentence windows are retained in the JSON. Primary inference follows the frozen contrasts, Holm families, MDD and parcel-rank rules.

Policy: pass 1; pass_broad .75; null .5; reversed 0; not_run/uninformative 1; low_stability multiplies by .5. Only primary checks update product multipliers.
