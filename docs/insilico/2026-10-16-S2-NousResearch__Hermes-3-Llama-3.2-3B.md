# WS08 S2 — NousResearch/Hermes-3-Llama-3.2-3B, 2026-10-16

Rows: 200; excluded: 0. Real TRIBE on local RTX 3070.
PitchCheck revision: `4a8fcd45ffcb2951fec78319fab988852b74eb7a`; atlas: `c5ca6bb873d2e0d1`.

Full numbers and provenance: [2026-10-16-S2-NousResearch__Hermes-3-Llama-3.2-3B.json](2026-10-16-S2-NousResearch__Hermes-3-Llama-3.2-3B.json).

Effects describe variation across authored texts in an average-subject prediction; they are not human-subject effect sizes or recipient mental states.

All co-primary tests pass/pass_broad: False.
Downstream contrasts retain uninformative status and multiplier 1 if the pre-registered S2 positive control failed. Raw effects and p values are still reported.

| Contrast | ROI | Primary | n | dz | Mean Δ | CI95 | p Holm | Reverse p Holm | Parcel rank | Verdict | Multiplier |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| sunscreen:How>Why | pifg_vezich2017 | True | 20 | -0.357277 | -0.003807 | [-0.008528587499999999, 0.0006989875000000006] | 1 | 0.128587 | 0.4175 | null | 0.5 |
| sunscreen:How>Why | ripl_vezich2017 | True | 20 | 1.72405 | 0.0226945 | [0.016897412499999997, 0.0281835625] | 0.00039996 | 0.00039996 | 0.87 | pass_broad | 0.75 |
| sunscreen:How>Fact | pifg_vezich2017 | True | 20 | -0.921311 | -0.014673 | [-0.021509587499999993, -0.007622250000000006] | 1 | 0.00119988 | 0.1875 | reversed | 0 |
| sunscreen:How>Fact | ripl_vezich2017 | True | 20 | 1.64326 | 0.018597 | [0.013859099999999997, 0.023543025000000002] | 0.00039996 | 0.00039996 | 0.96 | null | 0.5 |
| b2b:How>Why | pifg_vezich2017 | True | 20 | -0.0961446 | -0.00205375 | [-0.011574087499999997, 0.00710821875] | 1 | 1 | 0.5075 | null | 0.5 |
| b2b:How>Why | ripl_vezich2017 | True | 20 | -0.0819954 | -0.00092325 | [-0.005966100000000001, 0.004119018749999997] | 1 | 1 | 0.5125 | null | 0.5 |
| b2b:How>Fact | pifg_vezich2017 | True | 20 | -0.617839 | -0.011984 | [-0.0200464125, -0.0033596375] | 1 | 0.0559944 | 0.185 | null | 0.5 |
| b2b:How>Fact | ripl_vezich2017 | True | 20 | -0.357337 | -0.00388 | [-0.008791962500000002, 0.0004981499999999997] | 1 | 0.369263 | 0.36 | null | 0.5 |
| How>How_nobody | pifg_vezich2017 | False | 20 | 0.115612 | 0.0009465 | [-0.0026819749999999966, 0.0044369499999999985] | 0.618338 | 1 | 0.6175 | null | 0.5 |
| How>How_nobody | ripl_vezich2017 | False | 20 | 0.105927 | 0.001241 | [-0.0036057750000000003, 0.005877862499999998] | 0.618338 | 1 | 0.6 | null | 0.5 |

Motor/body covariate adjustments and descriptive cost-sentence windows are retained in the JSON. Primary inference follows the frozen contrasts, Holm families, MDD and parcel-rank rules.

Policy: pass 1; pass_broad .75; null .5; reversed 0; not_run/uninformative 1; low_stability multiplies by .5. Only primary checks update product multipliers.
