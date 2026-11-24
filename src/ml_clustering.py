"""Machine learning feature normalization and K-Means customer clustering."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

REQUIRED_FEATURES: tuple[str, ...] = ("recency", "frequency", "monetary")


class CustomerClustering:
    """Normalize RFM metrics via log1p + StandardScaler and fit K-Means clusters."""

    def __init__(self, rfm_df: pd.DataFrame, random_seed: int = 42) -> None:
        """Initialize clustering pipeline with validated RFM feature DataFrame."""
        if rfm_df is None or rfm_df.empty:
            raise ValueError("RFM DataFrame provided to CustomerClustering must not be empty.")
        missing = sorted(set(REQUIRED_FEATURES) - set(rfm_df.columns))
        if missing:
            raise ValueError(f"CustomerClustering missing required features: {missing}")

        self.rfm_df = rfm_df.copy()
        self.random_seed = random_seed
        self.scaled_data: np.ndarray | None = None
        self.scaler: StandardScaler | None = None
        self.model: KMeans | None = None
        self.silhouette_score_: float | None = None

    def preprocess_for_ml(self) -> np.ndarray:
        """Apply log1p transformation and z-score standardization to RFM features."""
        logger.info("Normalizing RFM features for %d customers...", len(self.rfm_df))
        features = self.rfm_df[list(REQUIRED_FEATURES)].astype(float).clip(lower=0.0)
        log_features = np.log1p(features)

        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(log_features)
        logger.info("RFM features log-transformed and standardized successfully.")
        return self.scaled_data

    def fit_kmeans(self, n_clusters: int = 4) -> pd.DataFrame:
        """Fit K-Means model and attach cluster labels and silhouette quality metric."""
        if n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {n_clusters}.")
        if n_clusters > len(self.rfm_df):
            raise ValueError(f"n_clusters ({n_clusters}) cannot exceed customer count ({len(self.rfm_df)}).")
        if self.scaled_data is None:
            self.preprocess_for_ml()

        assert self.scaled_data is not None
        logger.info("Fitting K-Means model with n_clusters=%d...", n_clusters)
        self.model = KMeans(n_clusters=n_clusters, random_state=self.random_seed, n_init=10)
        self.model.fit(self.scaled_data)

        labels = self.model.labels_.astype(int)
        self.rfm_df["ml_cluster"] = labels

        unique_labels = len(set(labels))
        if 2 <= unique_labels < len(self.rfm_df):
            score = silhouette_score(self.scaled_data, labels)
            self.silhouette_score_ = round(float(score), 4)
        else:
            self.silhouette_score_ = None

        logger.info(
            "K-Means clustering completed (clusters=%d, silhouette=%s).",
            n_clusters,
            self.silhouette_score_,
        )
        return self.rfm_df

    def summarize_clusters(self) -> list[dict[str, Any]]:
        """Compute per-cluster behavioral averages and dominant RFM segment."""
        if "ml_cluster" not in self.rfm_df.columns:
            raise ValueError("Cluster assignments unavailable; run fit_kmeans() first.")

        summaries: list[dict[str, Any]] = []
        grouped = self.rfm_df.groupby("ml_cluster", sort=True)
        for cluster_id, group in grouped:
            dominant = "N/A"
            if "actionable_segment" in group.columns and not group["actionable_segment"].empty:
                dominant = str(group["actionable_segment"].mode().iloc[0])

            summaries.append(
                {
                    "cluster_id": int(cluster_id),
                    "customer_count": int(len(group)),
                    "avg_recency": round(float(group["recency"].mean()), 2),
                    "avg_frequency": round(float(group["frequency"].mean()), 2),
                    "avg_monetary": round(float(group["monetary"].mean()), 2),
                    "dominant_segment": dominant,
                }
            )
        return summaries
