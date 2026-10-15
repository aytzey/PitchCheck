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

Every judgment and rewrite is held to a built-in persuasion doctrine — open inside the reader's problem, specificity over adjectives, earn the ask, pre-empt the default objection, honest proof hierarchy, one message one idea, status-safe CTAs — backed by an evidence annex from published persuasion science: self-relevance and neural message effectiveness (Falk et al. 2010/2011/2016; Cooper et al. 2015), route matching (Petty & Cacioppo's Elaboration Likelihood Model), framing (Tversky & Kahneman 1981) and, separately, loss aversion (Kahneman & Tversky 1979), same-setting norms (Goldstein, Cialdini & Griskevicius 2008), reactance and the "but you are free" effect (Carpenter 2013 meta-analysis), processing fluency (Alter & Oppenheimer), precise-number credibility (Janiszewski & Uy 2008), psychological targeting (Matz et al. 2017), and the commitment gradient (Freedman & Fraser 1966). Each top move names the principle it rests on.

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
| `PITCHCHECK_LOCALIZATION_CONTEXT_SEGMENTS` | `1` | Neighbouring 1-s segments shown around each located span (HRF-dispersion prior, 0–3) |
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
