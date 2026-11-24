"""Integration tests for FastAPI customer segmentation microservice endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import create_app
from src.config import SegmentationConfig
from src.data_pipeline import generate_sample_transactions


def test_health_endpoint_returns_liveness_status() -> None:
    """Verify GET /health returns 200 and service metadata."""
    # Arrange
    app = create_app(SegmentationConfig())
    client = TestClient(app)

    # Act
    response = client.get("/health")

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == "customer-segmentation-rfm-kmeans"
    assert "timestamp" in body


def test_analyze_endpoint_segments_valid_transactions() -> None:
    """Verify POST /analyze returns customer segments and cluster summaries."""
    # Arrange
    app = create_app(SegmentationConfig(random_seed=42))
    client = TestClient(app)
    sample_df = generate_sample_transactions(n_customers=20, seed=42)
    payload = {
        "n_clusters": 4,
        "transactions": sample_df.to_dict("records"),
    }

    # Act
    response = client.post("/analyze", json=payload)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["total_customers"] == 20
    assert data["n_clusters"] == 4
    assert len(data["cluster_summary"]) == 4
    assert len(data["segments"]) == 20
    assert all("actionable_segment" in seg and "ml_cluster" in seg for seg in data["segments"])


def test_analyze_endpoint_rejects_empty_and_invalid_payloads() -> None:
    """Verify 422 responses for empty lists and invalid field constraints."""
    # Arrange
    client = TestClient(create_app(SegmentationConfig()))

    # Act
    empty_resp = client.post("/analyze", json={"transactions": [], "n_clusters": 4})
    negative_price_resp = client.post(
        "/analyze",
        json={
            "n_clusters": 2,
            "transactions": [
                {
                    "order_id": "O1",
                    "customer_id": 1,
                    "order_date": "2026-02-01",
                    "quantity": 1,
                    "price": -10.0,
                },
                {
                    "order_id": "O2",
                    "customer_id": 2,
                    "order_date": "2026-02-02",
                    "quantity": 1,
                    "price": 10.0,
                },
            ],
        },
    )

    # Assert
    assert empty_resp.status_code == 422
    assert negative_price_resp.status_code == 422


def test_global_exception_handler_masks_internal_errors(monkeypatch) -> None:
    """Verify unhandled runtime exceptions return safe 500 JSON without tracebacks."""
    # Arrange
    app = create_app(SegmentationConfig())
    client = TestClient(app, raise_server_exceptions=False)
    sample_df = generate_sample_transactions(n_customers=6, seed=9)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated internal failure token_xyz")

    monkeypatch.setattr("src.api.RFMEngine.generate_rfm", _boom)

    # Act
    response = client.post(
        "/analyze",
        json={"n_clusters": 2, "transactions": sample_df.to_dict("records")},
    )

    # Assert
    assert response.status_code == 500
    body = response.json()
    assert body == {"detail": "Internal server error processing segmentation request."}
    assert "token_xyz" not in response.text


def test_cors_headers_allow_configured_origin_and_block_untrusted() -> None:
    """Verify CORS allowlist permits configured origin and omits untrusted origins."""
    # Arrange
    config = SegmentationConfig(allowed_origins=("https://trusted.example.com",))
    client = TestClient(create_app(config))

    # Act
    allowed = client.get("/health", headers={"Origin": "https://trusted.example.com"})
    blocked = client.get("/health", headers={"Origin": "https://untrusted.example.org"})

    # Assert
    assert allowed.headers.get("access-control-allow-origin") == "https://trusted.example.com"
    assert "access-control-allow-origin" not in blocked.headers
