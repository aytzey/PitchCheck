# PitchScore — Neural Persuasion Intelligence

Score your sales pitches with TRIBE brain-response analysis + LLM-powered persuasion interpretation.

**Domain**: pitch.machinity.ai

## What It Does

1. You input a **sales message** (cold email, pitch, ad copy) + a **target persona** description
2. TRIBE neural model analyzes the text for brain engagement patterns
3. LLM interprets those neural signals in the context of your target persona
4. You get a **persuasion score (0-100)** with breakdown, strengths, risks, and rewrite suggestions

## Stack

- **Frontend**: Next.js 16, TypeScript, Tailwind 4
- **Backend**: Python FastAPI, facebook/tribev2 (TRIBE neural model)
- **LLM Layer**: OpenRouter (GPT-4.1-mini) for persona-aware interpretation
- **Infra**: Docker Compose, Traefik reverse proxy, NVIDIA GPU

## Quick Start

### Docker (recommended)

```bash
cp .env.example .env
# Edit .env with your OPENROUTER_API_KEY
docker compose up -d --build
# Visit http://localhost:3000
```

### Local Development

**Frontend:**
```bash
npm install
npm run dev
# http://localhost:3000
```

**TRIBE Service:**
```bash
cd tribe_service
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# For GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
TRIBE_ALLOW_MOCK=1 uvicorn tribe_service.app:app --host 0.0.0.0 --port 8090
```

Set `TRIBE_SERVICE_URL=http://127.0.0.1:8090` for the frontend.

## API

### `POST /api/score`

Score a pitch for persuasion effectiveness.

**Request:**
```json
{
  "message": "Our platform reduces deployment time by 80% for enterprise teams...",
  "persona": "CTO, 40 years old, startup background, technical but pragmatic",
  "platform": "email"
}
```

**Response:**
```json
{
  "report": {
    "persuasion_score": 75,
    "verdict": "Strong pitch for technical audience",
    "narrative": "The pitch effectively addresses deployment pain points...",
    "breakdown": [
      { "key": "emotional_resonance", "label": "Emotional Resonance", "score": 70, "explanation": "..." },
      { "key": "clarity", "label": "Clarity", "score": 85, "explanation": "..." },
      { "key": "urgency", "label": "Urgency", "score": 60, "explanation": "..." },
      { "key": "credibility", "label": "Credibility", "score": 80, "explanation": "..." },
      { "key": "personalization_fit", "label": "Personalization Fit", "score": 65, "explanation": "..." }
    ],
    "neural_signals": [
      { "key": "emotional_engagement", "label": "Emotional Engagement", "score": 72, "direction": "up" },
      { "key": "attention_capture", "label": "Attention Capture", "score": 78, "direction": "up" }
    ],
    "strengths": ["Clear value proposition", "Strong technical credibility"],
    "risks": ["Missing social proof", "Could improve urgency"],
    "rewrite_suggestions": [
      { "title": "Strengthen opener", "before": "Our platform...", "after": "[Name], your team...", "why": "Personal hook" }
    ],
    "persona_summary": "Technical CTO at early-stage startup",
    "platform": "email",
    "scored_at": "2026-04-04T12:00:00Z"
  }
}
```

### `GET /api/health`

Returns service status.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | For LLM | — | OpenRouter API key for persona-aware analysis |
| `OPENROUTER_MODEL` | No | `openai/gpt-4.1-mini` | LLM model to use |
| `TRIBE_MODEL_ID` | No | `facebook/tribev2` | TRIBE model identifier |
| `TRIBE_DEVICE` | No | `auto` | `cuda`, `cpu`, or `auto` |
| `TRIBE_ALLOW_MOCK` | No | `0` | Set to `1` for testing without GPU |
| `TRIBE_SERVICE_URL` | No | `http://tribe-service:8090` | Backend URL (for frontend) |

## Platform Options

| Platform | Description |
|----------|-------------|
| `email` | Cold email or follow-up |
| `linkedin` | LinkedIn message or InMail |
| `cold-call-script` | Phone call script |
| `landing-page` | Landing page copy |
| `ad-copy` | Advertisement copy |
| `general` | General-purpose pitch |

## Testing

```bash
# Python tests (39 tests)
PYTHONPATH=. TRIBE_ALLOW_MOCK=1 pytest tribe_service/tests/ -v

# TypeScript tests (22 tests)
npx vitest run

# All 61 tests
npm test && PYTHONPATH=. TRIBE_ALLOW_MOCK=1 pytest tribe_service/tests/
```

## Architecture

```
User → Next.js (:3000) → /api/score → tribe-service (:8090) → TRIBE model → LLM → Report
                                         ↓
                                    OpenRouter (GPT-4.1-mini)
```

Behind Traefik at `pitch.machinity.ai` with auto-SSL via Let's Encrypt.
