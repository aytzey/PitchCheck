# Contributing

PitchCheck is a desktop-first AI app with three moving parts: the Next.js UI,
the Tauri/Rust runtime manager, and the FastAPI TRIBE service. Keep changes
small and test the layer you touch.

## Local Setup

```bash
npm install
cp .env.example .env
```

Run the web app:

```bash
npm run dev
```

Run the desktop app:

```bash
npm run desktop:dev
```

Run the model service in mock mode:

```bash
TRIBE_ALLOW_MOCK=1 uvicorn tribe_service.app:app --host 0.0.0.0 --port 8090
```

## Checks

```bash
npm run lint
npm test
npm run build
npm run build:desktop-web
npm run desktop:check
cargo test --manifest-path src-tauri/Cargo.toml
```

For GPU runtime changes, also run:

```bash
docker compose config
```

Paid Vast.ai testing should use `scripts/vast-e2e.mjs`. It creates a temporary
instance, scores once, and destroys the instance unless `KEEP_VAST_INSTANCE=1`
is set.

## Pull Requests

- Include the runtime path you tested: web, local GPU, Vast.ai, or Docker.
- Do not commit API keys, updater private keys, `.env`, model caches, or build
  output.
- Prefer focused PRs over broad refactors.
- Update README or docs when the setup, release, or runtime behavior changes.
