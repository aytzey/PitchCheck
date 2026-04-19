# Security Policy

## Supported Versions

Security fixes are handled on the default branch and published through signed
GitHub Releases.

## Reporting a Vulnerability

Please open a private GitHub security advisory for vulnerabilities that could
expose API keys, execute untrusted code, leak user pitch content, or leave paid
Vast.ai instances running unexpectedly.

Avoid posting secrets, tokens, or live endpoint URLs in public issues.

## Secret Handling

- Vast.ai keys are entered in the desktop UI and passed to Rust in memory.
- The Tauri updater private key belongs in the `TAURI_SIGNING_PRIVATE_KEY`
  GitHub secret, never in git.
- `.env`, `.secrets/`, model caches, and generated installers are ignored.
- `scripts/vast-e2e.mjs` destroys rented instances by default after the score
  request completes.
