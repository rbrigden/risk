# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this repository, please report it privately using **GitHub Security Advisories** ("Report a vulnerability" in the repository's **Security** tab) rather than opening a public issue.

Please include:
- A clear description of the issue and potential impact.
- Reproduction steps or a minimal proof of concept.
- Any affected binary, feature flag, or HTTP endpoint.

## Security Design Principles

- **Local-only tools by default**: Local development servers (`riskbot-web`, `web/dashboard/serve`, `web/serve`) bind to loopback (`127.0.0.1`) by default and are intended for trusted local use—not direct exposure to the public internet.
- **No wildcard CORS**: Local HTTP APIs do not emit `Access-Control-Allow-Origin: *` headers, preventing untrusted websites from interacting with local servers.
- **Path validation**: Run-telemetry and checkpoint-loading endpoints reject path traversal sequences and restrict checkpoint loading to `.mpk` files.
- **No secrets or production credentials**: No API keys, OAuth credentials, database URLs, or private deployment assets are stored in this repository.
