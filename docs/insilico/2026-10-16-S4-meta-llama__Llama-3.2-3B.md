# WS08 S4 — meta-llama/Llama-3.2-3B, 2026-10-16

Rows: 100; excluded: 0. Real TRIBE on local RTX 3070.
PitchCheck revision: `4a8fcd45ffcb2951fec78319fab988852b74eb7a`; atlas: `c5ca6bb873d2e0d1`.

Full numbers and provenance: [2026-10-16-S4-meta-llama__Llama-3.2-3B.json](2026-10-16-S4-meta-llama__Llama-3.2-3B.json).

Effects describe variation across authored texts in an average-subject prediction; they are not human-subject effect sizes or recipient mental states.

All co-primary tests pass/pass_broad: False.
Downstream contrasts retain uninformative status and multiplier 1 if the pre-registered S2 positive control failed. Raw effects and p values are still reported.

| Contrast | ROI | Primary | n | dz | Mean Δ | CI95 | p Holm | Reverse p Holm | Parcel rank | Verdict | Multiplier |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| Whygain>Fact | mpfc_falk2010 | True | 20 | -0.402551 | -0.003499 | [-0.007369024999999999, 0.00019588749999999535] | 1 | 0.173383 | 0.22 | uninformative | 1 |
| Whygain>Fact_you | mpfc_falk2010 | True | 20 | -0.298233 | -0.0024375 | [-0.0062248749999999995, 0.0007216749999999987] | 1 | 0.211879 | 0.2975 | uninformative | 1 |
| Whygain>Whyloss | mpfc_falk2010 | False | 20 | -0.64206 | -0.0051265 | [-0.008473137499999998, -0.0018984624999999994] | 0.9962 | 0.0119988 | 0.22 | uninformative | 1 |
| Why>How | mpfc_falk2010 | False | 20 | 0.609926 | 0.00441875 | [0.0015551812499999997, 0.007887275] | 0.00619938 | 0.0119988 | 0.7375 | uninformative | 1 |

Motor/body covariate adjustments and descriptive cost-sentence windows are retained in the JSON. Primary inference follows the frozen contrasts, Holm families, MDD and parcel-rank rules.

Policy: pass 1; pass_broad .75; null .5; reversed 0; not_run/uninformative 1; low_stability multiplies by .5. Only primary checks update product multipliers.
