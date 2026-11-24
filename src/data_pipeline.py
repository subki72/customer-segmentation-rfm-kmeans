"""Data ingestion, validation, cleaning, and synthetic demo generation."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "customer_id",
    "order_id",
    "order_date",
    "quantity",
    "price",
)


def generate_sample_transactions(
    n_customers: int = 40,
    seed: int = 42,
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    """Generate a deterministic synthetic retail transaction dataset for demos and tests."""
    if n_customers < 1:
        raise ValueError(f"n_customers must be >= 1, got {n_customers}.")

    rng = np.random.default_rng(seed)
    anchor_date = reference_date or datetime(2026, 3, 1, 12, 0, 0)
    rows: list[dict[str, object]] = []
    order_seq = 10000

    for idx in range(n_customers):
        customer_id = 1001 + idx
        n_orders = int(rng.integers(1, 7))
        for _ in range(n_orders):
            order_seq += 1
            days_ago = int(rng.integers(1, 180))
            rows.append(
                {
                    "order_id": f"ORD-{order_seq}",
                    "customer_id": customer_id,
                    "order_date": (anchor_date - timedelta(days=days_ago)).isoformat(),
                    "quantity": int(rng.integers(1, 16)),
                    "price": round(float(rng.uniform(5.0, 150.0)), 2),
                }
            )

    return pd.DataFrame(rows)


class DataPreprocessor:
    """Pipeline component handling CSV/DataFrame ingestion, validation, and cleaning."""

    def __init__(self, file_path: str | Path | None = None, df: pd.DataFrame | None = None) -> None:
        """Initialize preprocessor from either a CSV path or an in-memory DataFrame."""
        if file_path is None and df is None:
            raise ValueError("Either file_path or df must be provided to DataPreprocessor.")
        self.file_path: Path | None = Path(file_path) if file_path is not None else None
        self._input_df: pd.DataFrame | None = df.copy() if df is not None else None
        self.df: pd.DataFrame | None = None

    def _load_source(self) -> pd.DataFrame | None:
        """Load raw transactions from memory or CSV file."""
        if self._input_df is not None:
            return self._input_df.copy()

        assert self.file_path is not None
        logger.info("Loading dataset from %s", self.file_path)
        try:
            frame = pd.read_csv(self.file_path, encoding="utf-8")
        except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            logger.error("Failed to load dataset from %s: %s", self.file_path, exc)
            return None

        logger.info("Loaded %d raw rows from %s", len(frame), self.file_path)
        return frame

    @staticmethod
    def _validate_schema(frame: pd.DataFrame) -> None:
        """Verify all mandatory columns exist before transformation."""
        missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"Input dataset is missing required columns: {missing}")

    def process_data(self) -> pd.DataFrame | None:
        """Execute end-to-end validation, anomaly removal, and revenue computation."""
        raw_df = self._load_source()
        if raw_df is None:
            return None

        self._validate_schema(raw_df)
        cleaned = raw_df.dropna(subset=list(REQUIRED_COLUMNS)).copy()
        cleaned["order_id"] = cleaned["order_id"].astype(str).str.strip()
        cleaned = cleaned[~cleaned["order_id"].str.upper().str.startswith("C")]

        cleaned["quantity"] = pd.to_numeric(cleaned["quantity"], errors="coerce")
        cleaned["price"] = pd.to_numeric(cleaned["price"], errors="coerce")
        cleaned = cleaned.dropna(subset=["quantity", "price"])
        cleaned = cleaned[(cleaned["quantity"] > 0) & (cleaned["price"] > 0)].copy()

        cleaned["total_revenue"] = (cleaned["quantity"] * cleaned["price"]).round(4)
        cleaned["order_date"] = pd.to_datetime(cleaned["order_date"], errors="coerce")
        cleaned = cleaned.dropna(subset=["order_date"]).copy()
        cleaned["customer_id"] = pd.to_numeric(cleaned["customer_id"], errors="coerce")
        cleaned = cleaned.dropna(subset=["customer_id"]).copy()
        cleaned["customer_id"] = cleaned["customer_id"].astype(int)

        self.df = cleaned.reset_index(drop=True)
        if self.df.empty:
            logger.warning("Data cleaning completed with zero valid transactions remaining.")
            return self.df

        logger.info(
            "Data cleaning completed: %d valid transactions across %d unique customers.",
            len(self.df),
            self.df["customer_id"].nunique(),
        )
        return self.df
