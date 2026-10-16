# WS11 11.1 — HV2025 report/verdict template

Status: **pending GPU predictions**. This is a template, not an agreement result.
Pre-registration commit: pending supervisor commit.
Pre-registration SHA-256: populate from the exact committed `ws11_prereg.md`.
Stimulus set and ID/hash-lock SHA-256: pending recovery/authoring and commit.

Record image 886850, inverse 886851, collection 17520, CC0 (FAQ 13), hashes,
left/right order, atlas identity, and observed/expected 1,743/17,934 counts.
Record n=33/37/35, 6-s/8-s discrepancy and neutral block 900.632–972.683 s.
Include checkpoint/encoder HF SHAs, tribev2 commit, package versions, research
image digest, PitchCheck SHA, timing/event version and repeat-relative-L2 check.

| Analysis | Mask | N | Spearman r | Pearson r | p |
|---|---|---:|---:|---:|---:|
| D−S | valid | pending | pending | pending | — |
| D−S (primary) | valid minus Vis/SomMot | pending | pending | pending | pending |
| D−S | abs(T)>3.1 | pending | pending | pending | — |
| Raw mean D−S | all three | pending | pending | pending | — |
| Length-adjusted D coefficient | all three | pending | pending | pending | — |
| Long−short control | all three | pending | pending | pending | — |

Sign agreement on abs(T)>3.1: pending; permutation p: pending.
Null: 10,000 random balanced labels, or 32,768 exact pair flips for secondary S9.
Record the separate primary/paired reports; do not pool stimulus sources.
LOO D/S projection AUC: pending; full-refit label-null p: pending (monitoring only).
Table 1/2 peak sign inspection: pending original coordinate/hemisphere checks.

Verdict: pending. Use `agree` only if primary r>0, r and sign p<.05, and
r(D−S)>r(long−short). Otherwise `no_agreement`; a category-faithful failure is
`uninformative`: conceptual replication; a fail is uninformative about TRIBE fidelity.
The verdict never gates a feature or creates a numeric fidelity factor.

Template: pending; only positive primary r permits the TRIBE T-map template,
restricted to sign-concordant parcels with measured abs(mean T)>1.
Record shape float16[20484], hashes, sidecar and retained parcels. Its weight is 0.
Runtime fidelity scaling still requires the separate Lebel 11.2 asset.

Scope: similarity to one measured group reading contrast (health messages,
language likely Spanish). Not sales-copy, self-paced-reading or recipient
validation. A cortical highlight is a predicted-response analogue, not a mental state.
