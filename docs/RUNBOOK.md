# Operational Runbook

## Service Overview

- **Service**: `customer-segmentation-rfm-kmeans`
- **Primary Interfaces**:
  - **Batch CLI**: `python main.py` (supports `--input`, `--clusters`, `--output-dir`, `--demo`, `--health`)
  - **HTTP Microservice**: `uvicorn src.api:app --host 0.0.0.0 --port 8000`
- **Health Endpoint**: `GET /health` -> HTTP `200` with `{"status": "healthy", ...}`

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `Online Retail Data.csv` | Default input CSV path for batch CLI mode |
| `OUTPUT_DIR` | `outputs` | Directory for exported charts and JSON summary |
| `DEFAULT_CLUSTERS` | `4` | Default K-Means cluster count (`2..20`) |
| `RANDOM_SEED` | `42` | Reproducibility seed for K-Means and synthetic demo |
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://127.0.0.1:8000` | Comma-separated CORS allowlist |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |

## Common Verification Commands

```bash
# 1. Verify configuration and health via CLI
python main.py --health

# 2. Run synthetic end-to-end demo
python main.py --demo --clusters 4 --output-dir outputs

# 3. Verify API container locally
docker compose up --build -d
curl -f http://127.0.0.1:8000/health
```

## Incident Triage Playbook

### Symptom: `422 Unprocessable Entity` on `POST /analyze`
- **Cause**: Payload has empty `transactions`, non-positive `quantity`/`price`, malformed ISO `order_date`, or fewer valid non-cancelled unique `customer_id` records than `n_clusters`.
- **Action**: Inspect validation message in response `detail` field; ensure non-cancelled customer cardinality `>= n_clusters`.

### Symptom: Batch CLI exits with code `1` (`dataset could not be loaded`)
- **Cause**: `DATA_PATH` CSV is missing, empty, or lacks required columns (`customer_id`, `order_id`, `order_date`, `quantity`, `price`).
- **Action**: Confirm file permissions and CSV UTF-8 encoding, or run `python main.py --demo` to verify runtime integrity independently of external data files.

### Symptom: `500 Internal Server Error` from API
- **Cause**: Unexpected runtime failure inside preprocessing, RFM calculation, or clustering.
- **Action**: Inspect structured container logs (`docker compose logs -f api`) using `LOG_LEVEL=DEBUG` in non-production environments to capture full exception tracebacks (never exposed in HTTP responses).
