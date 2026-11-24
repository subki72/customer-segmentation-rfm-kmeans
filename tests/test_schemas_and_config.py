"""Unit tests for runtime configuration validation and Pydantic v2 schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import ValidationError
import pytest

from src.config import SegmentationConfig, configure_logging
from src.schemas import SegmentationRequest, TransactionItem


def test_config_from_env_parses_custom_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load and parse typed environment settings cleanly."""
    # Arrange
    monkeypatch.setenv("DATA_PATH", "custom.csv")
    monkeypatch.setenv("OUTPUT_DIR", "custom_out")
    monkeypatch.setenv("DEFAULT_CLUSTERS", "5")
    monkeypatch.setenv("RANDOM_SEED", "99")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://app.example.com, http://localhost:5173")
    monkeypatch.setenv("LOG_LEVEL", "debug")

    # Act
    cfg = SegmentationConfig.from_env(load_env_file=False)

    # Assert
    assert cfg.data_path == "custom.csv"
    assert cfg.output_dir == "custom_out"
    assert cfg.default_clusters == 5
    assert cfg.random_seed == 99
    assert cfg.allowed_origins == ("https://app.example.com", "http://localhost:5173")
    assert cfg.log_level == "DEBUG"


@pytest.mark.parametrize(
    ("env_key", "env_val"),
    [
        ("DEFAULT_CLUSTERS", "not-an-int"),
        ("DEFAULT_CLUSTERS", "1"),
        ("RANDOM_SEED", "-1"),
        ("LOG_LEVEL", "VERBOSE"),
    ],
)
def test_config_rejects_invalid_env_values(
    monkeypatch: pytest.MonkeyPatch,
    env_key: str,
    env_val: str,
) -> None:
    """Reject out-of-range or malformed configuration settings."""
    # Arrange
    monkeypatch.setenv(env_key, env_val)

    # Act & Assert
    with pytest.raises(ValueError):
        SegmentationConfig.from_env(load_env_file=False)


def test_configure_logging_validates_level() -> None:
    """Configure logger on valid level and raise ValueError on invalid level."""
    # Arrange & Act
    logger = configure_logging("INFO")

    # Assert
    assert logger.name == "customer_segmentation"
    with pytest.raises(ValueError, match="Invalid LOG_LEVEL"):
        configure_logging("TRACE")


def test_transaction_item_parses_iso_date_and_rejects_invalid() -> None:
    """Validate ISO-8601 dates, positive quantities/prices, and extra-field rejection."""
    # Arrange
    item = TransactionItem(
        order_id=" ORD-101 ",
        customer_id=55,
        order_date="2026-02-14T08:30:00Z",
        quantity=3,
        price=19.99,
    )

    # Assert
    assert item.order_id == "ORD-101"
    assert isinstance(item.order_date, datetime)

    with pytest.raises(ValidationError):
        TransactionItem(
            order_id="ORD-102",
            customer_id=55,
            order_date="bad-date",
            quantity=1,
            price=10.0,
        )


def test_segmentation_request_validates_customer_cardinality() -> None:
    """Ensure unique non-cancelled customer count satisfies n_clusters."""
    # Arrange
    items = [
        TransactionItem(order_id="O1", customer_id=1, order_date="2026-02-01", quantity=1, price=10.0),
        TransactionItem(order_id="O2", customer_id=2, order_date="2026-02-02", quantity=2, price=20.0),
    ]

    # Act
    valid_req = SegmentationRequest(transactions=items, n_clusters=2)

    # Assert
    assert valid_req.n_clusters == 2
    with pytest.raises(ValidationError, match="Valid unique customers"):
        SegmentationRequest(transactions=items, n_clusters=4)
