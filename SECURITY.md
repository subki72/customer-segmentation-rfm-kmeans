# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in this repository:

1. **Do NOT open a public GitHub issue** for undisclosed vulnerabilities.
2. Submit a private report via GitHub Security Advisories (`Security -> Advisories -> Report a vulnerability`).
3. Include affected file paths, line numbers, reproduction steps, and estimated severity.
4. Initial acknowledgement will be provided within **48 hours**.

## Security Controls

- **Secret Hygiene**: `.env`, key files, and local data outputs are excluded via `.gitignore`; zero credentials are stored in source control.
- **API Input Validation & CORS**: All FastAPI payloads are validated strictly via Pydantic v2 schemas (`src/schemas.py`), CORS origins are constrained via `ALLOWED_ORIGINS`, and unhandled exceptions return generic 500 JSON responses without internal stack traces.
- **Automated Vulnerability Management**: Weekly Dependabot updates (`semver-major` ignored for stability), Semgrep Python SAST, and Trivy container vulnerability scanning in CI.
- **Container Hardening**: Multi-stage non-root Docker runtime running as `appuser` (UID `10001`).
