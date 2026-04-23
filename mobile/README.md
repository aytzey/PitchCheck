# PitchCheck Mobile (Expo)

iOS-first mobile client designed to be shippable through TestFlight and the App Store.

## What is production-ready here
- Runtime switching for **PitchServer** and **Vast AI**
- Secure local storage for runtime credentials via `expo-secure-store`
- Transport compatibility modes:
  - `auto` (`/api/score` then `/score`)
  - `next-api` (`/api/score` only)
  - `direct` (`/score` only)
- Runtime connectivity check (`/api/health` or `/health`)
- URL validation + request timeout/retry behavior for unstable mobile networks
- Score payload normalization + safe defaults for malformed backend responses
- Request ID + idempotency key headers for runtime observability/safety
- Optional strict HTTPS policy for non-local runtimes
- Production build profiles auto-enforce strict HTTPS policy
- In-app recent-score panel + pending draft queue persistence (SecureStore)
- Runtime telemetry export (JSON) for debugging/support
- EAS build profiles for development, preview, and production
- Dynamic app config (`app.config.ts`) for bundle IDs and build numbers via env vars

## Setup

```bash
cp .env.example .env
npm install
npm run start
```

## iOS release flow

```bash
npm run build:ios
npm run submit:ios
```

## Runtime contract
Both runtime modes expect a compatible scoring service:
- `POST {baseUrl}/score` **or** `POST {baseUrl}/api/score`
- Body: `{ message, persona, platform, openRouterModel? }`

Health checks:
- `GET {baseUrl}/health` **or** `GET {baseUrl}/api/health`

Vast mode can optionally attach `Authorization: Bearer <key>`.
