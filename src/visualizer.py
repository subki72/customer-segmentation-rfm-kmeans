"""Headless-safe visualization generator for RFM segments and ML clusters."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import matplotlib

# Force non-interactive Agg backend only before pyplot loads so headless CI/tests never hang.
if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class SegmentVisualizer:
    """Generate publication-ready distribution and cluster scatter charts."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize visualizer with segmented customer DataFrame."""
        if df is None or df.empty:
            raise ValueError("DataFrame provided to SegmentVisualizer must not be empty.")
        self.df = df.copy()
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({"figure.autolayout": True})

    @staticmethod
    def _finalize_figure(
        fig: plt.Figure,
        output_path: str | Path | None,
        show: bool,
    ) -> Path | None:
        """Save figure to disk if requested, optionally show, and always close figure."""
        saved_path: Path | None = None
        try:
            if output_path is not None:
                saved_path = Path(output_path)
                saved_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(saved_path, dpi=150, bbox_inches="tight")
                logger.info("Saved visualization to %s", saved_path)
            if show:
                plt.show()
        finally:
            plt.close(fig)
        return saved_path

    def plot_segment_distribution(
        self,
        output_path: str | Path | None = None,
        show: bool = False,
    ) -> Path | None:
        """Render horizontal bar chart of customer counts per actionable RFM segment."""
        if "actionable_segment" not in self.df.columns:
            raise ValueError("Column 'actionable_segment' is required for segment distribution plot.")

        logger.info("Generating Actionable Segment Distribution plot...")
        segment_counts = self.df["actionable_segment"].value_counts().reset_index()
        segment_counts.columns = ["Segment", "Count"]

        fig, ax = plt.subplots(figsize=(14, 7))
        sns.barplot(
            data=segment_counts,
            x="Count",
            y="Segment",
            palette="viridis",
            hue="Segment",
            legend=False,
            ax=ax,
        )
        ax.set_title("Customer Distribution by Actionable Segment", fontsize=16, fontweight="bold")
        ax.set_xlabel("Number of Customers", fontsize=12)
        ax.set_ylabel("Actionable Segment", fontsize=12)
        return self._finalize_figure(fig, output_path=output_path, show=show)

    def plot_ml_clusters_scatter(
        self,
        output_path: str | Path | None = None,
        show: bool = False,
    ) -> Path | None:
        """Render log-scale Recency vs Frequency scatter plot colored by K-Means cluster."""
        required = {"recency", "frequency", "monetary", "ml_cluster"}
        missing = sorted(required - set(self.df.columns))
        if missing:
            raise ValueError(f"Missing columns for cluster scatter plot: {missing}")

        logger.info("Generating Machine Learning Clusters scatter plot...")
        plot_df = self.df.copy()
        plot_df["recency"] = plot_df["recency"].clip(lower=1)
        plot_df["frequency"] = plot_df["frequency"].clip(lower=1)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(
            data=plot_df,
            x="recency",
            y="frequency",
            hue="ml_cluster",
            palette="Set1",
            size="monetary",
            sizes=(20, 400),
            alpha=0.7,
            ax=ax,
        )
        ax.set_title("K-Means Clusters: Recency vs Frequency", fontsize=16, fontweight="bold")
        ax.set_xlabel("Recency (Days)", fontsize=12)
        ax.set_ylabel("Frequency (Transactions)", fontsize=12)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(title="ML Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
        return self._finalize_figure(fig, output_path=output_path, show=show)
