# Stage 1: Dependency Builder
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Stage 2: Hardened Non-Root Runtime
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PATH="/opt/venv/bin:$PATH"

RUN groupadd --gid 10001 appgroup \
    && useradd --uid 10001 --gid appgroup --shell /sbin/nologin --create-home appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup main.py ./

RUN mkdir -p /app/outputs && chown -R appuser:appgroup /app/outputs

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
