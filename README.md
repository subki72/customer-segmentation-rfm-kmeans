# Customer Segmentation via RFM Analysis & K-Means Clustering

[![CI Pipeline](https://github.com/subki72/customer-segmentation-rfm-kmeans/actions/workflows/ci.yml/badge.svg)](https://github.com/subki72/customer-segmentation-rfm-kmeans/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Executive Summary

This project delivers a production-ready customer analytics pipeline and FastAPI microservice combining **Recency, Frequency, Monetary (RFM)** behavioral modeling with **K-Means clustering**. It enables retail and e-commerce teams to identify high-value customer cohorts, surface churn-risk accounts early, and serve real-time segmentation results over HTTP or headless batch jobs while preserving the original exploratory Jupyter notebook (`User Segmentation.ipynb`).

## Architecture Overview

```text
Raw Transactions (CSV / API JSON / Synthetic Demo)
  └── DataPreprocessor (Schema validation, cancellation & anomaly filtering)
        └── RFMEngine (Vectorized R/F/M metrics, quintile scoring, segment tiering)
              └── CustomerClustering (log1p + StandardScaler + K-Means + Silhouette)
                    ├── FastAPI Microservice (GET /health, POST /analyze)
                    └── Headless Batch CLI (JSON summary + Agg PNG visualizations)
```

### Key Capabilities

- **Dual-Mode Non-Breaking Design**: Preserves `User Segmentation.ipynb` intact while exposing modular `src/` Python packages, a parameterized batch CLI (`main.py`), and a FastAPI HTTP service (`src/api.py`).
- **Resilient RFM & Clustering Engine**: Vectorized pandas aggregation, safe quantile scoring even for small cohorts (`< 5` customers), deterministic `log1p` + `StandardScaler` normalization, and silhouette quality scoring.
- **Strict API Contracts & Guardrails**: Pydantic v2 schemas (`src/schemas.py`), explicit CORS allowlisting (`ALLOWED_ORIGINS`), structured leveled stdout logging, and safe `500` error boundaries.
- **Production Packaging & CI/CD**: Pinned dependencies (`requirements.txt`, `requirements-lock.txt`, `uv.lock`), multi-stage non-root `Dockerfile`, `docker-compose.yml`, `Makefile`, Ruff + Flake8 linting, Pytest coverage gates, Semgrep SAST, and Trivy scanning.

## Repository Structure

```text
├── .github/
│   ├── dependabot.yml           # Weekly pip & github-actions updates (semver-major ignored)
│   └── workflows/ci.yml         # Lint, Pytest coverage, Docker build, Semgrep & Trivy
├── docs/
│   └── RUNBOOK.md               # Operations, configuration & incident triage playbook
├── src/
│   ├── __init__.py
│   ├── api.py                   # FastAPI microservice (GET /health, POST /analyze)
│   ├── config.py                # Validated SegmentationConfig & structured logging
│   ├── data_pipeline.py         # DataPreprocessor & deterministic synthetic generator
│   ├── ml_clustering.py         # CustomerClustering (log1p, StandardScaler, K-Means)
│   ├── rfm_engine.py            # RFMEngine quintile scoring & segment mapping
│   ├── schemas.py               # Pydantic v2 request/response schemas
│   └── visualizer.py            # Headless-safe Agg SegmentVisualizer
├── tests/
│   ├── test_api_integration.py  # FastAPI TestClient & error/CORS boundary tests
│   ├── test_data_pipeline.py    # Ingestion, cleaning & synthetic generator tests
│   ├── test_ml_clustering.py    # Clustering, headless plots & CLI mode tests
│   ├── test_rfm_engine.py       # RFM calculation & small-cohort boundary tests
│   └── test_schemas_and_config.py
├── User Segmentation.ipynb      # Original exploratory analysis notebook (preserved)
├── main.py                      # Parameterized batch & demo CLI entrypoint
├── Dockerfile                   # Multi-stage non-root runtime (appuser UID 10001)
├── docker-compose.yml           # Local container orchestration with healthcheck
├── Makefile                     # Single-command install, lint, test, demo, run, build
├── pyproject.toml               # Project metadata, Ruff & Pytest configuration
├── requirements.txt             # Exact pinned runtime & test dependencies
├── requirements-lock.txt        # Compiled Linux/Python 3.11 dependency lockfile
├── uv.lock                      # Reproducible uv lockfile
├── .env.example                 # Environment variable template
├── SECURITY.md                  # Vulnerability reporting & security controls
├── CHANGELOG.md                 # Semantic version release notes
├── CONTRIBUTING.md              # Local development & PR guidelines
└── LICENSE                      # MIT License
```

## Quickstart

### 1. Local Setup

```bash
cp .env.example .env
make install
```

### 2. Run Automated Linting & Tests

```bash
make lint
make test
```

### 3. Execute Pipeline via CLI

```bash
# Run on built-in synthetic transactions (exports PNGs + JSON summary to outputs/)
python main.py --demo --clusters 4 --output-dir outputs

# Run on a custom CSV dataset
python main.py --input "Online Retail Data.csv" --clusters 4 --output-dir outputs

# Verify runtime configuration and health
python main.py --health
```

### 4. Run FastAPI Microservice

```bash
make run
# Or with Docker Compose:
docker compose up --build
```

Verify liveness and run real-time segmentation:

```bash
curl -s http://127.0.0.1:8000/health
```

```bash
curl -s -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "n_clusters": 2,
    "transactions": [
      {"order_id": "ORD-1", "customer_id": 101, "order_date": "2026-02-01", "quantity": 3, "price": 25.0},
      {"order_id": "ORD-2", "customer_id": 101, "order_date": "2026-02-15", "quantity": 2, "price": 30.0},
      {"order_id": "ORD-3", "customer_id": 102, "order_date": "2026-01-05", "quantity": 1, "price": 12.0}
    ]
  }'
```

## License

Released under the [MIT License](LICENSE).
