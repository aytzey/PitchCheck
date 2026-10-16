# WS08 S2 — meta-llama/Llama-3.2-3B, 2026-10-16

Rows: 200; excluded: 0. Real TRIBE on local RTX 3070.
PitchCheck revision: `4a8fcd45ffcb2951fec78319fab988852b74eb7a`; atlas: `c5ca6bb873d2e0d1`.

Full numbers and provenance: [2026-10-16-S2-meta-llama__Llama-3.2-3B.json](2026-10-16-S2-meta-llama__Llama-3.2-3B.json).

Effects describe variation across authored texts in an average-subject prediction; they are not human-subject effect sizes or recipient mental states.

All co-primary tests pass/pass_broad: False.
Downstream contrasts retain uninformative status and multiplier 1 if the pre-registered S2 positive control failed. Raw effects and p values are still reported.

| Contrast | ROI | Primary | n | dz | Mean Δ | CI95 | p Holm | Reverse p Holm | Parcel rank | Spin p | Verdict | Multiplier |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |
| sunscreen:How>Why | pifg_vezich2017 | True | 20 | -0.175333 | -0.002848 | [-0.009793187500000003, 0.004234512499999995] | 1 | 0.440356 | 0.4525 | 0.47552448 | null | 0.5 |
| sunscreen:How>Why | ripl_vezich2017 | True | 20 | 1.89936 | 0.0224283 | [0.0171847375, 0.027378174999999998] | 0.00039996 | 0.00039996 | 0.92 | 0.06293706 | pass | 1 |
| sunscreen:How>Fact | pifg_vezich2017 | True | 20 | -1.07049 | -0.0233015 | [-0.0326936375, -0.014019687500000003] | 1 | 0.00039996 | 0.1875 | 0.78921079 | reversed | 0 |
| sunscreen:How>Fact | ripl_vezich2017 | True | 20 | 1.8472 | 0.0217605 | [0.016726537500000003, 0.026816537499999994] | 0.00039996 | 0.00039996 | 0.965 | 0.00999001 | pass | 1 |
| b2b:How>Why | pifg_vezich2017 | True | 20 | 0.127869 | 0.002778 | [-0.006758168750000004, 0.012287212499999997] | 1 | 0.80072 | 0.605 | 0.40959041 | null | 0.5 |
| b2b:How>Why | ripl_vezich2017 | True | 20 | -0.192435 | -0.002247 | [-0.007373868750000002, 0.0030030937499999897] | 1 | 0.80072 | 0.4575 | 0.60039960 | null | 0.5 |
| b2b:How>Fact | pifg_vezich2017 | True | 20 | -0.555697 | -0.0095585 | [-0.016933762499999998, -0.002418212500000008] | 1 | 0.0692931 | 0.315 | 0.67632368 | null | 0.5 |
| b2b:How>Fact | ripl_vezich2017 | True | 20 | -0.570942 | -0.004919 | [-0.009070250000000004, -0.0013493875] | 1 | 0.0407959 | 0.3075 | 0.70529471 | reversed | 0 |
| How>How_nobody | pifg_vezich2017 | False | 20 | -0.0406167 | -0.0003175 | [-0.003839575000000003, 0.003129824999999992] | 0.986901 | 1 | 0.45 | 0.56943057 | null | 0.5 |
| How>How_nobody | ripl_vezich2017 | False | 20 | 0.0064217 | 7.2e-05 | [-0.004676225000000001, 0.004684525] | 0.986901 | 1 | 0.54 | 0.48251748 | null | 0.5 |

Motor/body covariate adjustments and descriptive cost-sentence windows are retained in the JSON. Primary inference follows the frozen contrasts, Holm families, MDD and parcel-rank rules.

Policy: pass 1; pass_broad .75; null .5; reversed 0; not_run/uninformative 1; low_stability multiplies by .5. Only primary checks update product multipliers.

Additional robustness: 1,000 parcel spins, seed 0. Spin p uses the
vertex-weighted parcel approximation for partial literature locations; it does
not replace the original ROI statistic or change any verdict/multiplier.
Observed parcel d_z, sphere hashes and all methods are in the
[spin report](2026-10-16-spins.md).
