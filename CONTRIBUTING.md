# Contributing Guide

Thank you for your interest in contributing to **Customer Segmentation RFM & K-Means**.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install ruff flake8 pre-commit
cp .env.example .env
```

Install local git hooks:

```bash
pre-commit install
```

## Coding Standards

- Follow PEP 8 with a `120`-character maximum line length enforced by Ruff and Flake8 (`make lint`).
- Use explicit Python type annotations and Google-style docstrings on all public modules, classes, and functions.
- Replace `print()` calls with `logging.getLogger(__name__)` structured logging.
- Keep functions concise (`< 50` lines) with shallow nesting (`< 3` levels) and upfront guard clauses.
- Leave `User Segmentation.ipynb` untouched unless explicitly updating notebook analysis.

## Pull Request Checklist

1. Run `make lint` (`ruff check`, `ruff format --check`, and `flake8`) with zero errors.
2. Run `make test` (`python -m pytest tests/ -v --cov=src --cov-fail-under=70`) with 100% passing tests.
3. Verify headless execution via `python main.py --demo`.
4. Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `ci:`, `chore:`).
