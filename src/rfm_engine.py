"""RFM metric computation, quintile scoring, and actionable segment mapping."""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_RFM_COLUMNS: tuple[str, ...] = (
    "customer_id",
    "order_date",
    "order_id",
    "total_revenue",
)

SEGMENT_MAP: dict[str, str] = {
    r"[1-2][1-2]": "Hibernating",
    r"[1-2][3-4]": "At Risk",
    r"[1-2]5": "Cannot Lose Them",
    r"3[1-2]": "About To Sleep",
    r"33": "Need Attention",
    r"[3-4][4-5]": "Loyal Customers",
    r"41": "Promising",
    r"51": "New Customers",
    r"[4-5][2-3]": "Potential Loyalists",
    r"5[4-5]": "Champions",
}


def _compute_quintile_scores(series: pd.Series, descending: bool = False) -> pd.Series:
    """Assign 1-5 scores safely even when fewer than five customers exist."""
    n_items = len(series)
    if n_items == 0:
        return pd.Series(dtype=int)
    if n_items == 1:
        default_score = 5 if descending else 3
        return pd.Series([default_score], index=series.index, dtype=int)

    n_bins = min(5, n_items)
    raw_labels = np.linspace(5, 1, n_bins) if descending else np.linspace(1, 5, n_bins)
    labels = [int(round(val)) for val in raw_labels]

    # Rank with method="first" ensures unique bin edges regardless of ties.
    ranked = series.rank(method="first")
    scored = pd.qcut(ranked, q=n_bins, labels=labels)
    return scored.astype(int)


class RFMEngine:
    """Compute Recency, Frequency, Monetary metrics and actionable business tiers."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Validate preprocessed transaction DataFrame for RFM aggregation."""
        if df is None or df.empty:
            raise ValueError("Transaction DataFrame provided to RFMEngine must not be empty.")
        missing = sorted(set(REQUIRED_RFM_COLUMNS) - set(df.columns))
        if missing:
            raise ValueError(f"RFMEngine missing required columns: {missing}")

        self.df = df.copy()
        self.df["order_date"] = pd.to_datetime(self.df["order_date"])
        self.rfm: pd.DataFrame | None = None

    def generate_rfm(self) -> pd.DataFrame:
        """Aggregate customer RFM metrics, quintile scores, and actionable segments."""
        logger.info("Computing RFM metrics for %d transactions...", len(self.df))
        snapshot_date = self.df["order_date"].max() + dt.timedelta(days=1)

        rfm = self.df.groupby("customer_id", as_index=False).agg(
            last_order_date=("order_date", "max"),
            frequency=("order_id", "nunique"),
            monetary=("total_revenue", "sum"),
        )
        rfm["recency"] = (snapshot_date - rfm["last_order_date"]).dt.days.astype(int)
        rfm = rfm.drop(columns=["last_order_date"])
        rfm["customer_id"] = rfm["customer_id"].astype(int)
        rfm["frequency"] = rfm["frequency"].astype(int)
        rfm["monetary"] = rfm["monetary"].round(2).astype(float)

        rfm["r_score"] = _compute_quintile_scores(rfm["recency"], descending=True)
        rfm["f_score"] = _compute_quintile_scores(rfm["frequency"], descending=False)
        rfm["m_score"] = _compute_quintile_scores(rfm["monetary"], descending=False)
        rfm["rf_score"] = rfm["r_score"].astype(str) + rfm["f_score"].astype(str)

        rfm["base_segment"] = rfm["rf_score"].replace(SEGMENT_MAP, regex=True)
        rfm["value_tier"] = np.where(rfm["m_score"] >= 4, "High Value", "Standard Value")
        rfm["actionable_segment"] = rfm["value_tier"] + " " + rfm["base_segment"]

        ordered_cols = [
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
        ]
        self.rfm = rfm[ordered_cols].sort_values("customer_id").reset_index(drop=True)
        logger.info("RFM segmentation completed for %d customers.", len(self.rfm))
        return self.rfm
