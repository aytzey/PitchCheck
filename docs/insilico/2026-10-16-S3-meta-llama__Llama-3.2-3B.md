# WS08 S3 — meta-llama/Llama-3.2-3B, 2026-10-16

Rows: 80; excluded: 0. Real TRIBE on local RTX 3070.
PitchCheck revision: `4a8fcd45ffcb2951fec78319fab988852b74eb7a`; atlas: `c5ca6bb873d2e0d1`.

Full numbers and provenance: [2026-10-16-S3-meta-llama__Llama-3.2-3B.json](2026-10-16-S3-meta-llama__Llama-3.2-3B.json).

Effects describe variation across authored texts in an average-subject prediction; they are not human-subject effect sizes or recipient mental states.

All co-primary tests pass/pass_broad: False.
Downstream contrasts retain uninformative status and multiplier 1 if the pre-registered S2 positive control failed. Raw effects and p values are still reported.

| Contrast | ROI | Primary | n | dz | Mean Δ | CI95 | p Holm | Reverse p Holm | Parcel rank | Verdict | Multiplier |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| excessive<reasonable | vmpfc_frontopolar | True | 20 | 0.18022 | 0.0003785 | [-0.0004960125000000006, 0.0013970749999999996] | 0.783622 | 0.434557 | 0.435 | uninformative | 1 |
| insula_secondary | insula_right | False | 20 | 0.76091 | 0.00304812 | [0.0012671468615816734, 0.004686880345907793] | 0.00259974 | 0.00679932 | 0.8025 | uninformative | 1 |
| number_only_control | vmpfc_frontopolar | False | 20 | -0.0548013 | -4e-05 | [-0.0003566249999999993, 0.0002715624999999986] | 0.40246 | 0.80472 | 0.6075 | uninformative | 1 |

Motor/body covariate adjustments and descriptive cost-sentence windows are retained in the JSON. Primary inference follows the frozen contrasts, Holm families, MDD and parcel-rank rules.

Policy: pass 1; pass_broad .75; null .5; reversed 0; not_run/uninformative 1; low_stability multiplies by .5. Only primary checks update product multipliers.
