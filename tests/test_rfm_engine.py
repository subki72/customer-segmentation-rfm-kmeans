"""Unit tests for RFM metric calculation, quintile scoring, and segment mapping."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_pipeline import DataPreprocessor, generate_sample_transactions
from src.rfm_engine import RFMEngine


def test_rfm_engine_computes_metrics_and_segments() -> None:
    """Verify recency, frequency, monetary, quintile scores, and segment labels."""
    # Arrange
    sample_raw = generate_sample_transactions(n_customers=20, seed=42)
    cleaned = DataPreprocessor(df=sample_raw).process_data()
    assert cleaned is not None

    # Act
    rfm_df = RFMEngine(cleaned).generate_rfm()

    # Assert
    assert len(rfm_df) == 20
    assert set(rfm_df.columns) >= {
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "r_score",
        "f_score",
        "m_score",
        "rf_score",
        "base_segment",
        "value_tier",
        "actionable_segment",
    }
    assert (rfm_df["recency"] >= 1).all()
    assert (rfm_df["frequency"] >= 1).all()
    assert (rfm_df["monetary"] > 0).all()
    assert rfm_df["r_score"].between(1, 5).all()
    assert rfm_df["f_score"].between(1, 5).all()
    assert rfm_df["m_score"].between(1, 5).all()


def test_rfm_engine_handles_small_customer_cohorts() -> None:
    """Handle fewer than five unique customers without qcut bin-edge failures."""
    # Arrange
    small_df = pd.DataFrame(
        [
            {"order_id": "O1", "customer_id": 1, "order_date": "2026-02-01", "total_revenue": 100.0},
            {"order_id": "O2", "customer_id": 2, "order_date": "2026-02-10", "total_revenue": 250.0},
            {"order_id": "O3", "customer_id": 3, "order_date": "2026-02-20", "total_revenue": 50.0},
        ]
    )

    # Act
    rfm_df = RFMEngine(small_df).generate_rfm()

    # Assert
    assert len(rfm_df) == 3
    assert rfm_df["r_score"].tolist() == [1, 3, 5]
    assert rfm_df["m_score"].tolist() == [3, 5, 1]
    assert rfm_df["actionable_segment"].notna().all()


def test_rfm_engine_single_customer_cohort() -> None:
    """Handle single-customer edge case deterministically."""
    # Arrange
    single_df = pd.DataFrame([{"order_id": "O1", "customer_id": 99, "order_date": "2026-02-15", "total_revenue": 75.0}])

    # Act
    rfm_df = RFMEngine(single_df).generate_rfm()

    # Assert
    assert len(rfm_df) == 1
    assert int(rfm_df.iloc[0]["recency"]) == 1
    assert int(rfm_df.iloc[0]["frequency"]) == 1


def test_rfm_engine_rejects_empty_or_invalid_schema() -> None:
    """Reject empty DataFrames and missing required RFM columns."""
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="must not be empty"):
        RFMEngine(pd.DataFrame())

    with pytest.raises(ValueError, match="missing required columns"):
        RFMEngine(pd.DataFrame([{"customer_id": 1, "order_date": "2026-02-01"}]))
