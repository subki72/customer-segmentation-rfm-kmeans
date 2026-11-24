"""Unit tests for data ingestion, validation, cleaning, and synthetic demo generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_pipeline import DataPreprocessor, generate_sample_transactions


def test_generate_sample_transactions_deterministic() -> None:
    """Verify deterministic sample dataset generation across identical seeds."""
    # Arrange
    n_customers = 12
    seed = 77

    # Act
    first_df = generate_sample_transactions(n_customers=n_customers, seed=seed)
    second_df = generate_sample_transactions(n_customers=n_customers, seed=seed)

    # Assert
    assert not first_df.empty
    assert first_df["customer_id"].nunique() == n_customers
    assert first_df.equals(second_df)


def test_generate_sample_transactions_invalid_count_raises() -> None:
    """Reject non-positive customer counts upfront."""
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="n_customers"):
        generate_sample_transactions(n_customers=0)


def test_data_preprocessor_requires_source() -> None:
    """Raise ValueError when neither file_path nor df is provided."""
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="Either file_path or df"):
        DataPreprocessor()


def test_data_preprocessor_cleans_anomalies_and_computes_revenue() -> None:
    """Filter cancellations, non-positive rows, nulls, and compute total_revenue."""
    # Arrange
    raw_df = pd.DataFrame(
        [
            {"order_id": "1001", "customer_id": 101, "order_date": "2026-01-10", "quantity": 2, "price": 15.0},
            {"order_id": "C1002", "customer_id": 102, "order_date": "2026-01-11", "quantity": 1, "price": 20.0},
            {"order_id": "1003", "customer_id": None, "order_date": "2026-01-12", "quantity": 3, "price": 10.0},
            {"order_id": "1004", "customer_id": 103, "order_date": "2026-01-13", "quantity": 0, "price": 10.0},
            {"order_id": "1005", "customer_id": 104, "order_date": "2026-01-14", "quantity": 3, "price": -5.0},
            {"order_id": "1006", "customer_id": 105, "order_date": "2026-01-15", "quantity": 4, "price": 12.5},
        ]
    )

    # Act
    cleaned = DataPreprocessor(df=raw_df).process_data()

    # Assert
    assert cleaned is not None
    assert len(cleaned) == 2
    assert list(cleaned["customer_id"]) == [101, 105]
    assert list(cleaned["total_revenue"]) == [30.0, 50.0]


def test_data_preprocessor_csv_and_missing_file(tmp_path: Path) -> None:
    """Load CSV files successfully and return None cleanly when file is missing."""
    # Arrange
    csv_path = tmp_path / "retail.csv"
    sample = generate_sample_transactions(n_customers=5, seed=42)
    sample.to_csv(csv_path, index=False)

    # Act
    loaded = DataPreprocessor(file_path=csv_path).process_data()
    missing = DataPreprocessor(file_path=tmp_path / "nonexistent.csv").process_data()

    # Assert
    assert loaded is not None
    assert len(loaded) == len(sample)
    assert missing is None


def test_data_preprocessor_missing_columns_raises() -> None:
    """Raise explicit ValueError when required columns are absent."""
    # Arrange
    bad_df = pd.DataFrame([{"order_id": "1", "customer_id": 10}])

    # Act & Assert
    with pytest.raises(ValueError, match="missing required columns"):
        DataPreprocessor(df=bad_df).process_data()
