<div align="center">

<br>

# PitchCheck

**Know if your pitch will land before you send it.**

Score sales emails, cold outreach, LinkedIn DMs, and call scripts against a neuroscience model that predicts an average listener's cortical response analogue to the text; recipient attitudes are handled by the persona-aware LLM layer.

<br>

[![Screenshot](docs/assets/pitchcheck-screenshot.png)](https://github.com/aytzey/PitchCheck/releases)

<br>

[![CI](https://img.shields.io/github/actions/workflow/status/aytzey/PitchCheck/ci.yml?branch=main&label=CI&style=flat-square)](https://github.com/aytzey/PitchCheck/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/aytzey/PitchCheck?display_name=tag&sort=semver&style=flat-square)](https://github.com/aytzey/PitchCheck/releases)
[![License](https://img.shields.io/github/license/aytzey/PitchCheck?style=flat-square)](LICENSE)
![Tauri 2](https://img.shields.io/badge/Tauri-2.x-24c8db?style=flat-square)
![Next.js 16](https://img.shields.io/badge/Next.js-16-111827?style=flat-square)

[Download](https://github.com/aytzey/PitchCheck/releases) · [Quick Start](#quick-start) · [How the Scoring Works](#how-the-scoring-works) · [Architecture](#architecture)

</div>

---

## What This Does

You paste a message, set a recipient ("Jordan Park, Staff Engineer"), pick a channel (email, LinkedIn, cold call, landing page, ad copy), and hit score.

PitchCheck runs your text through Meta's [TRIBE v2](https://github.com/facebookresearch/tribev2) brain encoding model — the same model used in computational neuroscience research to predict fMRI responses to naturalistic stimuli. It maps your message onto ~20,000 cortical vertices and treats the output as **predicted neural-response analogues**, not measured fMRI from the recipient. It returns:

- **Persuasion score** (0–100) with calibration confidence
- **Six neural signals** — affective salience, early attention salience, self-value proxy, social cognition proxy, encoding potential, cognitive friction
- **Neuro-persuasive axes** — self/value, affective salience, social cognition, encoding/attention, processing fluency (whole-brain proxies)
- **Segment-by-segment predicted-response timeline** — where the predicted response peaks and drops
- **Verdict and narrative** — an LLM reads the neural output and explains what's working and what isn't, in plain language
- **Top moves** — the 1-3 highest-leverage changes, LLM-ranked by expected impact on whether the reader acts
- **Rewrite suggestions** — before/after alternatives with reasoning

The persuasion doctrine starts from the actual relationship, tailors with persona-stated facts, earns the ask relative to its value, uses domain-relevant proof, keeps requests voluntary, and makes the next step easy to picture: **cut words, not reasons**. Business proof rules do not turn personal invitations into sales copy. Python and the Rust desktop fallback share [persuasion_doctrine.txt](tribe_service/persuasion_doctrine.txt), embedded with Rust's `include_str!`.

The evaluator's evidence annex covers route matching, framing, same-setting norms, reactance, fluency, precise-number credibility, psychological targeting, and commitment gradients. Neuroscience entries (Falk 2010/2011, Cooper 2015, Chua 2009/2011, Vezich 2017, Huskey 2017, Klucharev 2008, Yomogida 2017, Cascio 2015, Hernández-Vergara 2025, Di Plinio 2025, Ntoumanis 2024, Knutson 2007) are tagged analogies from small human studies; TRIBE does not measure them. These are unvalidated literature priors for text, not evidence of persuasion for this reader. The annex stays with the evaluator; the non-Jev rewrite writer receives only the doctrine.

WS04 responses carry `prompt_doctrine_version="doctrine_v2"` in score `robustness` and refine `tribe_guidance`, including clarification responses. CI caps the annex at 4350 characters and the doctrine at 2450. WS04 has **no rollback env switch**: revert its implementation commit with `git revert <WS04-commit>`, rebuild the service using `tribe_service/Dockerfile.server` (which copies Python and the doctrine onto the existing GPU base image), and redeploy. Rebuild/release the desktop to roll back its embedded fallback doctrine too. The same rollback policy is recorded in `.env.example`.

WS05 shares one context-fit rubric across the evaluator, refine critic and Jev reviewer, with aligned writer instructions. Its default `v2` is an unvalidated literature prior applied by analogy: recipient-stated facts, relevant proof, value relative to costs, a concrete next step, voluntary requests and neutral peer information. A single standard imperative CTA is allowed only on ad-copy/landing pages as a product decision. Set `PITCHCHECK_CONTEXT_FIT_RUBRIC=v1` and restart the service to restore the old rubric and Python writer instructions. Every score (including neural-only fallback) records `robustness.context_fit_rubric_version`; every refine result (including clarification) records it in `tribe_guidance`. Facet weights, calibration and neural formulas are unchanged. The desktop fallback embeds the aligned writer wording at build time.

WS06 adds persona-read literature priors (ELM; Huskey 2017 by analogy; Chua 2009/2011): bounded, flag-controlled, shadow-logged, and unvalidated on outcomes. `context_fit` records stated recipient stance, tailoring level and up to three verified, unused verbatim work facts. Unknown labels stay `null`; job titles never determine stance, and merely using a competitor or disliking outreach is neutral. Private or sensitive details are excluded by the evaluator instructions. These are an LLM read of your persona, with no ROI link. Personalization Fit and Credibility retain WS02's existing facet anchors.

With `PITCHCHECK_PERSONA_READ_PRIORS=1` (default), skeptical reads multiply proof/objection facet weights by 1.4 and recipient-specific tailoring adds 3 semantic points before the existing neural band clamp. Generic and category-level messages receive the same prior. Every score includes `robustness.persona_read_prior` (`null` on neural-only fallback), with provenance, sources and `final_score_without_prior`; confidence is unchanged. K5 caps the final-score prior difference at ±6 points and records `score_terms` and `final_score_without_literature_priors`. The opt-in numbers-only log records labels, fact counts and shadow scores, never the facts themselves. The skeptic/weak-proof synthesis item is tagged `persona_llm_read`. Non-Jev refine applies the same priors and records each candidate's `semantic_score_without_prior`; Jev selection is unchanged. The desktop shows the persona read, unused details and material score adjustment, and includes two bounded lines in the Rust fallback brief only.

Rollback: set `PITCHCHECK_PERSONA_READ_PRIORS=0` and restart the service to restore pre-prior scoring/selection and remove the skeptic prompt/synthesis lever; structured labels remain available. Adjust `PITCHCHECK_SKEPTIC_FACET_BOOST` and `PITCHCHECK_TAILORING_PRIOR_POINTS` downward independently, or set both to `0` for score-only shadow monitoring. `PITCHCHECK_LITERATURE_PRIOR_BUDGET=0` disables final-score prior effects while retaining shadow metadata; it does not change refine ranking.

The label-stability monitor is written for 20 synthetic EN/TR personas × 3 runs, including six title-only pairs. Run `python scripts/eval-persona-read.py --selftest` without credentials; `--report <reads.json>` aggregates an existing run offline. It reports stance agreement (target ≥80%), null rate, fact-empty rate (≤70%), title-pair consistency and the Turkish subset. A separately authorized paid run requires `--allow-paid-run --model <model> --output <reads.json>` and `OPENROUTER_API_KEY`. **No paid label-stability run was performed for this implementation**; this monitoring does not gate release.

The whole loop takes a few seconds. Write, score, tweak, re-score.

## How the Scoring Works

The pipeline has three stages. All of this runs locally if you have a GPU; otherwise PitchCheck rents one on [Vast.ai](https://vast.ai) and tears it down when you disconnect.

### 1. Text → TRIBE Prediction

Your message gets tokenized into word-level events with synthetic timing: one word starts every 0.40 s and lasts 0.32 s (~150 wpm, no sentence pauses), so TRIBE models hearing the pitch read aloud, not self-paced reading. TRIBE's predictions are already shifted 5 s for hemodynamic lag (PitchCheck never shifts them again). Each 1-s segment presents ~2–3 words, but its predicted response blends several seconds of surrounding text. Numbers, URLs and e-mail addresses are split into several word events. TRIBE v2 takes those events and produces a prediction matrix of shape `(n_segments, 20484)` — each row is a time segment, each column is a cortical vertex on the fsaverage5 mesh.

No audio synthesis, no image generation — the `direct` text input mode skips gTTS/WhisperX entirely and works straight from the text.

### 2. Prediction → Neural Features → Persuasion Signals

Ten raw features get extracted from the prediction matrix:

| Feature | What it captures |
|---------|-----------------|
| `global_mean_abs` | Mean `\|prediction\|` over all vertices and segments |
| `global_peak_abs` | Maximum vertex `\|prediction\|` |
| `temporal_std` | How much predicted response varies across segments |
| `early_mean` | First-quartile predicted response (opener strength) |
| `late_mean` | Last-quartile predicted response (close strength) |
| `max_temporal_delta` | Largest consecutive segment shift |
| `spatial_spread` | Fraction of vertices above the global mean |
| `focus_ratio` | Top-10% vertex mean vs. global mean |
| `sustain_ratio` | Fraction of segments above mean |
| `arc_ratio` | (max − min) / mean of temporal trace |

These get ratio-normalized against `global_mean_abs` and mapped through hand-set band ranges (literature-motivated, unvalidated) into six predicted-response signals (0–100):

| Signal | Interpretive label (literature anchor; whole-brain proxy) | Key inputs |
|--------|----------------------|------------|
| Affective Salience (response intensity) | Affective/value salience findings; valence-agnostic | peak intensity, temporal variation, arc, delta |
| Early Attention Salience | Salience findings | early predicted response, peak, spatial spread |
| Self-Value Proxy | mPFC/vmPFC self/value findings | sustain, focus, late predicted response |
| Social Cognition Proxy | dmPFC/pSTS/temporal-pole/TPJ social-cognition findings | peaks, temporal dynamics, spread |
| Encoding Potential | Encoding/memorability findings; not a recall estimate | arc, sustain, peak, focus |
| Cognitive Friction | Processing-fluency findings (inverse) | inverse of sustain, focus, spread |

Scores are compressed to [8, 92] ([4, 84] for cognitive friction) to avoid false confidence at extremes. Cognitive friction is inverted — a high number means high friction, which is bad. A prediction-quality gate then shrinks the final score toward neutral when the TRIBE output is near-zero, flat, too short, or too low-resolution to support a strong claim.

The final score is calibrated from these signals through five TRIBE-derived neuro-persuasive axes:

| Axis | What it means |
|------|---------------|
| Self/value response proxy | Response-shape composite named after mPFC/vmPFC self/value findings; identical for every persona |
| Affective salience proxy | Intensity/variability composite; not liking, willingness to pay or purchase intent |
| Social-cognition proxy | Composite named after dmPFC/pSTS/temporal-pole/TPJ findings; not a social-proof detector |
| Encoding/attention proxy | Salience/encoding heuristic; memorability is not persuasion |
| Processing-fluency proxy | Inverse whole-brain friction heuristic; a comprehension hint |

### 3. Neural Output → LLM Interpretation

The raw signals and prediction summary go to an evaluator LLM (`google/gemini-3.8-flash` by default) via [OpenRouter](https://openrouter.ai), along with the recipient persona, channel norms for the selected platform, and a segment map that ties each temporal-trace segment to the approximate sentence of the pitch it covers. Rewrites are produced by a separate refiner model (`google/gemini-3.8-flash` by default; `z-ai/glm-5.3-flash` as the Jev writer); both are OpenRouter model ids and fully swappable, with first-class handling for reasoning models (`<think>` stripping, `reasoning.effort` hints, JSON-mode fallback).

Before the LLM sees anything, a deterministic **neural × research synthesis** layer does the heavy interpretive lifting from the TRIBE output so the LLM mostly verbalizes and rewrites rather than guesses:

- **Segment localization.** The per-segment temporal trace is mapped back onto the actual words of the pitch to pinpoint — in Python, with no LLM — the opener strength, the strongest moment, the single weakest span, the close/CTA strength, and the largest predicted-response drop. The LLM and the report both receive these approximate text regions as places to inspect instead of being asked to find them.
- **Research findings.** Axis geometry and raw TRIBE features (e.g. `sustain_ratio`) are linked to published findings they are motivated by, not validated against: weak self/value response → the Falk et al. self-relevance lever; low fluency → Alter & Oppenheimer; reward-vs-social route dominance → Cohen et al. 2024; "the predicted response stays at or above average for only N% of the pitch" → a memorability prior (Chan et al. 2024; Chua et al. 2011).
- **Temporal archetype.** The trace shape is classified (strong-open-fade, late peak, buried lede, flat, sustained) by a PitchCheck heuristic inspired by Chan et al. 2024 (video ads), each with a concrete lever.

All of it is citation-anchored and appears in the report's deep dive and the evaluator prompt (only the desktop Rust fallback refine reads it as a brief); the evaluator is told to localize every judgment to these spans and execute the strongest levers in its top moves. The LLM produces the verdict, narrative, strength/risk analysis, a structured context-fit read (persona pain alignment, objection coverage, proof credibility, CTA ease, channel fit), and rewrite suggestions.

PitchCheck no longer uses deterministic keyword/regex persuasion heuristics for scoring. The TRIBE-predicted neural prior anchors the final score, and the semantic side is derived from the rubric-scored context-fit facets (not from a single self-reported LLM number), clamped to the neural calibration band, and blended in. When TRIBE evidence is weak, the quality-shrunk neural prior carries less of the score and the semantic read carries more — so persona and channel fit genuinely move the score while prompt-injection text stays bounded by the band clamp.

Repeated TRIBE predictions are cached in-process, but OpenRouter prompt/output caps and fast-model routing are intentionally disabled; quality is preferred over latency/cost.

WS07 adds signed Schaefer-400 cortical ROI analogues and a default-on **unvalidated literature prior** (`neural_only_v2.3`). `PITCHCHECK_ROI_PRIOR_WEIGHT=0` restores scoring without the ROI term and disables its evaluator block and WS07 synthesis items; the optional `fmri_output.roi_summary` remains available. Compatible Llama features use weight 0.15; Hermes uses 0.075 with `;encoder_mismatch` provenance. Set `PITCHCHECK_ROI_ENCODER_MISMATCH_FACTOR=0` to disable that contribution, or `PITCHCHECK_ROI_INSILICO_TUNING=0` to ignore future WS08 multipliers. `PITCHCHECK_LITERATURE_PRIOR_BUDGET=6.0` bounds the final total prior contribution; 0 removes it. These settings are passed through Docker Compose; PitchServer accepts them in its service env file. Restart the service after changes.

The atlas and encoder-specific numeric unit assets are bundled for offline operation. Hermes and checkpoint-aligned Llama use per-ROI median centers and centered-raw SDs from a local 64-text RTX 3070 run, tagged `unit_calibration:<run_id>`. The ROI term remains an unvalidated literature prior; unknown encoders or missing/invalid unit assets retain the provisional 0/1 defaults. See [the WS07 build and calibration procedure](docs/ws07-roi.md) for reproducible assets, encoder-specific benchmark artifacts, and geometry caveats.

Refine reports measured candidates and signed location differences separately from LLM judgments. Jev uses one unvalidated Falk 2010 medial-prefrontal location prior, bounded to ±0.5 selection points by default (`PITCHCHECK_ROI_PRIOR_POINTS=0` disables it), with a noise dead zone, a why/how route guard and a full ROI-free counterfactual in every response. Legacy ranking stays unchanged and reports `roi_shadow_selected_id`. Encoder-keyed `default_prior` floors ship now; WS08 reference floors take precedence when available. Turkish uses the provisional floor with half weight, as do incompatible encoders (the factor is applied once when both apply). `PITCHCHECK_ROI_HYPOTHESES=0` disables the separate Falk JoCN hypothesis. No WS09 numeric ROI readout or delta enters an LLM payload; its /score synthesis is categorical. See [WS09 details and provisional-floor limits](docs/ws09-refine-roi.md).

### Specificity contrast (WS12, literature prior)

`PITCHCHECK_SPECIFICITY_CONTRAST=1` enables an internal category-worded control for English drafts of 12–250 engine tokens. The LLM proposes at most four verbatim spans; deterministic replacements preserve second-person count, certainty and length. A single additional TRIBE pass measures a signed **average-reader** contrast using WS07's Chua locations. The model does not see the persona and cannot establish whether details are true or relevant. The desktop shows an `SPC` / `exp` row and a folded “Compared with” control. This is an **unvalidated literature prior**, not recipient-level outcome evidence.

Only persona-traceable details can affect scoring: `PITCHCHECK_SPECIFICITY_SCORE_WEIGHT=1.5` and `PITCHCHECK_SPECIFICITY_Z_MIN=1.0` set the slope and dead zone, capped at ±3 points before K4/K8 scaling and the shared K5 ±6 total budget. The response includes `robustness.specificity_contrast` (available or an explicit skip reason), `prior_points`, `score_terms.specificity_prior`, and `score_without_specificity_prior`. WS06's `personalization_level`, when supplied, additionally requires `recipient_specific`. The evaluator sees only the explicitly allowed derived contrast summary and span/category pairs, never raw six-ROI values or the control text, and is told to leave its own scores unchanged by that block.

`PITCHCHECK_SPECIFICITY_REFINE_PENALTY=2.0` subtracts up to two selection points when a candidate drops grounded details carrying a positive contrast; `specificity_retention` records the dropped spans. The twin stays outside candidate measurements and critic/Jev payloads. Set the penalty to `0` to skip refine's extra calls, the score weight to `0` for score shadow mode, or the contrast flag to `0` to restore the previous path with zero extra calls. Mock mode also makes zero extra calls. Incompatible encoders use the shared default 0.5 factor; setting that factor to `0` skips both twin calls with `encoder_mismatch_disabled`. Busy GPU queues skip optional measurement with `skipped_busy` rather than delaying queued primary scores.

All four switches are in `.env.example` and the local Compose mapping; PitchServer reads overrides from its raw `service.env` (documented in the deployment Compose file, without masking those overrides). Restart after changing them. Methodology remains `neural_only_v2.3` for the shared Faz 1b train. The [WS12 probe and monitoring procedure](docs/ws12-specificity.md) includes 20 synthetic health/sales pairs, CPU self-checks and separate bounded GPU runs for Hermes/Llama. GPU/API probes and live smoke checks are written but unrun; run the probe within two weeks of deployment, then review the dead zone and weekly shadow-score differences, p95 latency and primary-score 429 rates. These monitoring results never gate shipping.

The neuroscience layer follows a conservative triangulation rule from consumer-neuroscience and neuroforecasting work: TRIBE output is useful evidence, but it is not treated as recipient-level measured fMRI. Region labels are interpretive anchors, reverse-inference risk is exposed through confidence reasons and scientific caveats, and weak prediction matrices lower both confidence and score extremity.

Research basis for the calibration layer includes Falk et al. on neural responses predicting persuasion-induced behavior change ([2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC3027351/)), neural focus groups predicting population-level media effects ([2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3725133/)), and MPFC/self-value signals predicting health behavior change and campaign response ([2011](https://doi.org/10.1037/a0022259), [2016](https://doi.org/10.1093/scan/nsv108)); Venkatraman et al. on fMRI predicting market-level advertising response ([2015](https://doi.org/10.1509/jmr.13.0593)); Baek, Scholz et al. on value/self/social signals and information sharing ([2017](https://doi.org/10.1177/0956797617695073)); Chan, Scholz et al. on cross-cultural neural prediction of information sharing ([2023](https://doi.org/10.1073/pnas.2313175120)); Chan et al. on temporal neural signals of ad liking ([2024](https://doi.org/10.1177/00222437231194319)); Cohen et al. on reward vs. mentalizing predictors of persuasiveness by narrative type ([2024](https://doi.org/10.1038/s41598-024-62341-3)); Scholz, Chan & Falk et al. on a 16-study message-effectiveness mega-analysis ([2025](https://doi.org/10.1093/pnasnexus/pgaf287)); Cao & Reimann on triangulation and reverse-inference limits ([2020](https://doi.org/10.3389/fpsyg.2020.550204)); and TRIBE v2 as an in-silico fMRI foundation model ([GitHub](https://github.com/facebookresearch/tribev2), [Hugging Face](https://huggingface.co/facebook/tribev2)). The full source registry, with modality and evidence level per study, is `RESEARCH_SOURCES` in `tribe_service/persuasion_features.py`.

This layer is optional — scoring works without an OpenRouter key, you just won't get the written interpretation.

## Quick Start

### Desktop App

Grab the installer from [Releases](https://github.com/aytzey/PitchCheck/releases) — Linux (.deb, .rpm), macOS (universal .dmg), Windows (.msi).

The app auto-detects your GPU situation:
- **NVIDIA GPU available** → runs the TRIBE service locally via Docker on `127.0.0.1:8090`
- **No GPU** → enter a Vast.ai API key, PitchCheck finds the cheapest matching instance, deploys, scores, and destroys on disconnect

API keys stay in memory. Nothing hits disk.

### From Source

```bash
git clone https://github.com/aytzey/PitchCheck.git
cd PitchCheck
npm install
npm run desktop:dev
```

On Ubuntu/Debian, you'll need Tauri's system deps first:

```bash
sudo apt-get install -y libgtk-3-dev libwebkit2gtk-4.1-dev \
  libayatana-appindicator3-dev librsvg2-dev
```

### Docker Compose (Web Only)

```bash
cp .env.example .env
docker compose up -d --build
```

App runs on `http://localhost:3000`.

### Mobile App (Expo, iOS-first, TestFlight-ready)

PitchCheck includes a production-oriented Expo mobile client in `mobile/` with the same dark design language as desktop and native runtime controls:

- **PitchServer** runtime
- **Vast AI** runtime
- Transport mode toggle (`auto`, `next-api`, `direct`)
- Runtime health probing before scoring
- Secure local credential storage (`expo-secure-store`)
- EAS build profiles for development/preview/production and iOS submit scripts

Run it:

```bash
npm run mobile:install
npm run mobile:start
```

For App Store release flow, see `mobile/README.md` (`build:ios`, `submit:ios`) and configure `mobile/.env` from `.env.example`.

### One-Line Install

```bash
npm run install:local
```

Handles everything — git pull, env setup, npm install, lint, test, build, TRIBE Docker image, desktop bundle, system package install.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Tauri 2 Shell                     │
│          (Rust · signed installers · auto-update)    │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │              Next.js 16 + React 19            │  │
│  │          (Tailwind v4 · Lucide icons)         │  │
│  └────────────────────┬──────────────────────────┘  │
│                       │                             │
│           ┌───────────┴───────────┐                 │
│           │   Rust Runtime Mgr    │                 │
│           │  (Docker / Vast.ai)   │                 │
│           └───────────┬───────────┘                 │
└───────────────────────┼─────────────────────────────┘
                        │
            ┌───────────┴───────────┐
            │  FastAPI TRIBE Svc    │
            │  facebook/tribev2     │
            │  ~20K cortical verts  │
            ├───────────────────────┤
            │  OpenRouter LLM       │
            │  (interpretation)     │
            └───────────────────────┘
                    runs on
           ┌────────┴────────┐
      Local NVIDIA      Vast.ai
        Docker GPU      Cloud GPU
```

The Rust layer in Tauri manages the entire GPU lifecycle — detecting `nvidia-smi`, pulling the Docker image, starting/stopping containers, or provisioning Vast.ai instances with the cheapest verified offer within configured price/VRAM limits.

## API

### `POST /api/score`

```json
{
  "message": "Hey Jordan — caught your QCon talk on observability...",
  "persona": "Staff Engineer, infrastructure background",
  "platform": "email"
}
```

Returns persuasion score, neural signals, predicted-response timeline, verdict, strengths, risks, and rewrite suggestions.

### `GET /api/health`

Service readiness check.

## Development

```bash
npm run dev              # Next.js dev server
npm run desktop:dev      # Tauri desktop dev
npm run lint             # ESLint
npm test                 # Vitest
npm run build            # Production web build
npm run desktop:build    # Native installers
```

TRIBE service standalone (mock mode for development):

```bash
cd tribe_service
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
TRIBE_ALLOW_MOCK=1 uvicorn tribe_service.app:app --host 0.0.0.0 --port 8090
```

Desktop Rust checks:

```bash
npm run build:desktop-web
npm run desktop:check
cargo test --manifest-path src-tauri/Cargo.toml
```

WS04 prompt monitoring uses 27 handwritten synthetic pitch/persona pairs, including
eight predeclared neutral cases and English/Turkish personal and business messages.
It compares `old`, `old-repeat` (A/A noise), and `new` in rotating order on the same
UTC day and model, with fixed neural inputs and no GPU. Prepare two existing
worktrees: the pre-WS04 baseline (`f0da33d` for this change) and the WS04 version.
In the service's Python environment, with `OPENROUTER_API_KEY` already set, run later:

```bash
python scripts/ab-prompt-doctrine.py --old-worktree /path/to/before --new-worktree /path/to/after --model MODEL_ID --output /tmp/doctrine-ab.jsonl
python scripts/ab-prompt-doctrine.py --report /tmp/doctrine-ab.jsonl
```

The script records actual prompt-token usage, detail/length rejection rates,
critic support judgments, user-text safety matches, and neutral facet shifts
against A/A spread. Generated text and manual-review fields in JSONL support
checking persona-specific benefits, next steps, and personal tone. This measures
model behavior, not persuasion, and never gates release or runs in CI.
`python scripts/ab-prompt-doctrine.py --selftest` checks the runner offline without API calls.

WS05's frozen English/Turkish minimal pairs are a rubric manipulation check, not evidence of persuasion; thresholds are design choices. The runner compares v1/v2 in the same process with three raw evaluator samples per text, synthetic equal neural measurements, critic checks, writer orientation probes and refine regression cases. It records requested/served models, raw-score medians and spread, ties, controls and rollback warnings. It never runs in CI or gates release. Run later, with an API key and a pinned model:

```bash
python scripts/eval-rubric-pairs.py --rubric both --model MODEL_ID --output /tmp/rubric-pairs.json
python scripts/eval-rubric-pairs.py --selftest
```

The first command uses a paid API (roughly 1100 calls); it has not been run for this implementation. `--selftest` is entirely offline. By default a completed monitoring run exits 0; `--strict` exits 1 for unmet design thresholds. A greater than 20 percentage-point drop in ad-copy/landing improved-rate warns to revert to v1. No WS06 persona priors are required; personalization-level labels are recorded only if present.

## Configuration

| Variable | Default | What it does |
|----------|---------|--------------|
| `OPENROUTER_API_KEY` | — | Turns on LLM verdicts and rewrites |
| `OPENROUTER_MODEL` | `google/gemini-3.8-flash` | High-quality model for interpreting neural output |
| `OPENROUTER_REFINER_MODEL` | `google/gemini-3.8-flash` | Which model writes rewrite drafts for requests without `jevStrategy` |
| `OPENROUTER_REASONING_EFFORT` | — | Optional reasoning-effort hint for reasoning-capable models (DeepSeek V4: `high`/`xhigh`); dropped automatically when a provider rejects it |
| `PITCHCHECK_SEMANTIC_BLEND_WEIGHT` | `0.55` | Base share of the final score carried by the band-clamped context-fit read; grows automatically as TRIBE prediction quality drops (0 = neural-only) |
| `PITCHCHECK_PERSONALIZATION_FACET_WEIGHT` | `1.0` | Share of the Personalization Fit breakdown item taken from the persona read (`persona_pain_alignment`); TRIBE never sees the persona. `0` restores the neural proxy |
| `PITCHCHECK_CREDIBILITY_FACET_WEIGHT` | `0.5` | Share of the Credibility breakdown item taken from `proof_credibility` (hand-set prior); `0` restores the neural mix |
| `PITCHCHECK_PERSONA_READ_PRIORS` | `1` | WS06 unvalidated literature priors for score, skeptic prompt/synthesis and non-Jev selection; `0` + service restart restores pre-prior behavior |
| `PITCHCHECK_SKEPTIC_FACET_BOOST` | `0.4` | Additional proof/objection facet weight for stated skeptical recipients; `0` disables this score/selection term |
| `PITCHCHECK_TAILORING_PRIOR_POINTS` | `3` | Semantic points for recipient-specific tailoring only; `0` disables this score/selection term |
| `PITCHCHECK_LITERATURE_PRIOR_BUDGET` | `6` | K5 maximum absolute final-score difference from the no-priors shadow; `0` disables final-score prior effects |
| `PITCHCHECK_CONTEXT_FIT_RUBRIC` | `v2` | WS05 unvalidated literature prior for shared rubric and Python writers; `v1` + service restart restores the old wording. Version recorded in score/refine responses |
| `PITCHCHECK_LOCALIZATION_CONTEXT_SEGMENTS` | `1` | Neighbouring 1-s segments shown around each located span (HRF-dispersion prior, 0–3) |
| `PITCHCHECK_NEURAL_TIE_DELTA` | reference / `1.0` | Legacy refine treats neural differences below the encoder-specific paraphrase noise floor as ties. `0` restores the previous ranking. See [WS08](docs/insilico/README.md) |
| `PITCHCHECK_SCORE_LOG_PATH` / `PITCHCHECK_SCORE_LOG_KEY` | unset | Opt-in numbers-only JSONL log of `/score` and `/refine` for later outcome joins. Needs both; the key (≥32 chars) HMACs message/persona ids. Text is never written |
| `OPENROUTER_TIMEOUT_SECONDS` | `60` | LLM request timeout |
| `TRIBE_DEVICE` | `cuda` | `cuda`, `cpu`, or `auto` |
| `TRIBE_TEXT_DEVICE` | `auto` | Device for the 3B text feature model |
| `TRIBE_ALLOW_MOCK` | `0` | Deterministic mock for tests |
| `TRIBE_PREDICTION_CACHE_SIZE` | `8` | In-memory LRU cache for repeated TRIBE text predictions |
| `TRIBE_SCORE_TIMEOUT_SECONDS` | `900` | Timeout (first run downloads ~8GB of weights) |

Requests with `jevStrategy: true` use one `z-ai/glm-5.3-flash` writer call after Jev's
decision, regardless of the client's `openRouterModel`. GLM requests prefer
Baseten FP8 (`baseten/fp8`), Fireworks (`fireworks`) and CoreWeave (`coreweave/nvfp4`),
in that order. When upstream capacity is unavailable, OpenRouter can use another
provider of the same model that supports the requested parameters; latency may
increase. The Jev writer sends one request without application retries or
regeneration. It uses low reasoning and excludes reasoning text. Both refine
writer paths scale their completion allowance with the source length for three
complete JSON drafts, from 1,536 tokens up to 65,536 tokens. Explicit
GLM analysis requests default to low reasoning and cap completion at 4,096 tokens.
Refine responses include the served model and provider in `writer_call` when the
provider returns them; `model` uses the served model when available.

Each rewrite preserves the original meaning, supporting details and paragraph
coverage while actively rewriting its phrasing and organization, targeting
80–120% of its word count with a five-word tolerance for
short inputs. This overrides channel brevity suggestions and also applies to
the plain-ask candidate. The three drafts must have different openings and
recognizable strategies; copied source text or punctuation-only variants are
not rewrite alternatives. The original and all three drafts still receive TRIBE
measurements; length-mismatched drafts cannot win even if the critic approves
them. If no eligible improvement remains, the original is retained. The existing
30,000-character input limit is unchanged; outputs are never padded or sliced to
meet the target. Business rewrites use supplied minimum requirements and product
coverage to explain a factual fit or gap, without naming or denigrating competitors,
inventing another option's limitations, or assuming unstated units.

<details>
<summary>Full variable list</summary>

| Variable | Default | Description |
|----------|---------|-------------|
| `TRIBE_MODEL_ID` | `facebook/tribev2` | Model identifier |
| `TRIBE_TEXT_BATCH_SIZE` | `auto` | Batch 1 on sub-10GB GPUs |
| `TRIBE_TEXT_INPUT_MODE` | `direct` | `direct` skips TTS/WhisperX |
| `TRIBE_MAX_SCORE_CONCURRENCY` | `1` | Keep at 1 on local GPUs |
| `TRIBE_OOM_FALLBACK_TEXT_DEVICE` | `accelerate,cpu` | Comma-separated retry devices after CUDA OOM |
| `TRIBE_ACCELERATE_MAX_GPU_MEMORY_GB` | `auto` | GPU memory cap for Accelerate offload |
| `TRIBE_ACCELERATE_MAX_CPU_MEMORY_GB` | `32` | CPU RAM cap for Accelerate offload |
| `TRIBE_ACCELERATE_OFFLOAD_FOLDER` | `/models/offload` | Disk folder for Accelerate offloaded weights |
| `TRIBE_SERVICE_URL` | `http://tribe-service:8090` | Backend URL for frontend |
| `PITCHCHECK_TRIBE_IMAGE` | `ghcr.io/aytzey/pitchcheck-tribe:latest` | Runtime image override |

</details>

## CI/CD

Three GitHub Actions workflows:

- **ci.yml** — lint, test, build (web + desktop Rust checks + Compose validation) on every push/PR
- **tribe-image.yml** — builds and pushes `ghcr.io/<owner>/pitchcheck-tribe:latest` when `tribe_service/` changes on main
- **desktop-installers.yml** — cross-platform release builds (Linux/macOS/Windows) on `v*` tags, with Tauri updater signing

## Tech Stack

| Layer | Tech |
|-------|------|
| Desktop shell | [Tauri 2](https://tauri.app) (Rust) |
| Frontend | [Next.js 16](https://nextjs.org), React 19, Tailwind CSS v4 |
| Scoring backend | [FastAPI](https://fastapi.tiangolo.com), Python 3.11 |
| Neural model | [TRIBE v2](https://github.com/facebookresearch/tribev2) (Meta FAIR) |
| Text features | `NousResearch/Hermes-3-Llama-3.2-3B` by default (TRIBE checkpoint trained on `meta-llama/Llama-3.2-3B`; reported in `text_feature_compatible`) |
| LLM interpretation | [OpenRouter](https://openrouter.ai) |
| Cloud GPU | [Vast.ai](https://vast.ai) |
| CI/CD | GitHub Actions |

## TRIBE v2 Model — License and Attribution

PitchCheck uses Meta's **TRIBE v2** (`facebook/tribev2`) for neural response prediction.

TRIBE v2 is a foundation model for in-silico neuroscience developed by Meta FAIR. It combines `meta-llama/Llama-3.2-3B` text features (PitchCheck defaults to Hermes-3-Llama-3.2-3B and reports compatibility in `text_feature_compatible`), V-JEPA2, and Wav2Vec-BERT to predict fMRI brain responses across ~20,000 cortical vertices on the fsaverage5 mesh. The model was trained on naturalistic fMRI datasets and is described in:

> Stéphane d'Ascoli, Jérémy Rapin, Yohann Benchetrit, Teon Brookes, Katelyn Begany, Joséphine Raugel, Hubert Banville, Jean-Rémi King. *"A foundation model of vision, audition, and language for in-silico neuroscience."* Meta FAIR, 2026.

**The TRIBE v2 model weights are licensed under [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/).** This means the model cannot be used for commercial purposes. PitchCheck's own source code is MIT-licensed, but any deployment that includes the TRIBE v2 weights is subject to the non-commercial restriction of CC-BY-NC-4.0.

```bibtex
@article{dAscoli2026TribeV2,
  title   = {A foundation model of vision, audition, and language
             for in-silico neuroscience},
  author  = {d'Ascoli, St{\'e}phane and Rapin, J{\'e}r{\'e}my and
             Benchetrit, Yohann and Brookes, Teon and Begany, Katelyn
             and Raugel, Jos{\'e}phine and Banville, Hubert and
             King, Jean-R{\'e}mi},
  year    = {2026}
}
```

## License

PitchCheck source code is [MIT](LICENSE).

TRIBE v2 model weights (`facebook/tribev2`) are [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) — non-commercial use only. See [TRIBE v2 Model — License and Attribution](#tribe-v2-model--license-and-attribution) above.

## WS10 monitoring

The offline [WS10 harness](docs/ws10-monitoring.md) compares shadow formulas,
tests incremental predictive association and reports tuning/rollback advice.
It never changes a product score or gates a feature. The
[validation protocol](docs/validation-protocol.md) must be committed before real
outcome collection or inspection. Outcome files and consented corpora stay
outside Git; `validation_data/` is ignored.
