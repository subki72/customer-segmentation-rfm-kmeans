# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-24

### Added
- Validated runtime configuration (`src/config.py`) and `.env.example` template with structured leveled stdout logging.
- Strict Pydantic v2 API contracts (`src/schemas.py`) for transaction items, segmentation requests, cluster summaries, and health responses.
- FastAPI microservice (`src/api.py`) exposing `GET /health` and `POST /analyze` with CORS allowlisting and safe global exception handling.
- Synthetic retail transaction generator (`generate_sample_transactions`) and CLI modes (`--input`, `--clusters`, `--output-dir`, `--demo`, `--health`).
- Five offline Pytest suites (`tests/`) covering data preprocessing, RFM scoring, K-Means clustering, headless plotting, schemas/config, and FastAPI integration.
- Multi-stage non-root `Dockerfile`, `docker-compose.yml`, `Makefile`, `pyproject.toml`, `.flake8`, `.pre-commit-config.yaml`, `requirements-lock.txt`, and `uv.lock`.
- GitHub Actions CI pipeline (`.github/workflows/ci.yml`), Dependabot configuration, `SECURITY.md`, `CONTRIBUTING.md`, and `docs/RUNBOOK.md`.

### Changed
- Refactored `src/data_pipeline.py`, `src/rfm_engine.py`, `src/ml_clustering.py`, `src/visualizer.py`, and `main.py` with full type hints, vectorized RFM aggregation, small-cohort quintile guards, and headless `Agg` plotting.
- Preserved original exploratory notebook `User Segmentation.ipynb` untouched.
