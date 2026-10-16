# WS11 fallback stimulus review — 2026-10-16

The original Appendix 1 was not recovered. The Granada repository links the
same anonymized manuscript without the appendix; the publisher page returned
a browser challenge, and the supplied OpenNeuro events file is empty.
This category-faithful set follows the committed fallback protocol and is a
conceptual replication, not the paper's original stimuli. A failure is
uninformative about TRIBE fidelity.

OpenAI Codex authored 15 D and 15 S messages plus 15 long and 15 short neutral
health descriptions outside git. No WS08 frozen input was edited. D messages
include authority, an imperative and “now”; S messages include a question and
“perhaps”. Outcome clauses are distinct within each condition. Word bounds are
10–30 for D/S, 25–30 for long and 10–14 for short; all 60 hashes are unique.
The D/S set is unpaired. S9 tone pairs were not supplied and are not invented.

Independent blind review used local NousResearch/Hermes-3-Llama-3.2-3B on CPU
(CUDA hidden). It saw wording and category definitions, without ROI names,
hypotheses, measured maps or TRIBE outputs. All 60/60 labels matched: 15 D,
15 S, 30 neutral. The author also reviewed the authority/question markers,
distinct outcome clauses and neutral controls. This is not a new owner review.

Only IDs, conditions, exact UTF-8 byte SHA-256, and tokenizer word counts enter
the committed JSON lock. Texts, authoring source, review prompts and outputs
remain under scratchpad/gpu/runs/ws11-stimuli-draft/. The lock must be committed
before inference. The analysis registration remains unchanged.

Authored source SHA-256: `1a50ea3d546cc3683ebb55f0d5cb4ba8377ca2dd35be7dbb2a8df344528efa28`.

Blind review SHA-256: `113aebc77f18248fd17a3f694d8fa090a4b0dccb9d9ded702ba0c015481ef420`.

Lock SHA-256: `09719190621ba7f50d2ba103103378168835c0477e39890b3b254646693a5d8e`.
