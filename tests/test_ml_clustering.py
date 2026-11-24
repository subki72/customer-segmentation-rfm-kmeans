"""Unit tests for K-Means clustering, headless visualization, and CLI runner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from main import main
from src.data_pipeline import DataPreprocessor, generate_sample_transactions
from src.ml_clustering import CustomerClustering
from src.rfm_engine import RFMEngine
from src.visualizer import SegmentVisualizer


def test_customer_clustering_fit_and_summarize() -> None:
    """Verify log1p scaling, K-Means labeling, silhouette score, and summaries."""
    # Arrange
    raw_df = generate_sample_transactions(n_customers=30, seed=42)
    cleaned = DataPreprocessor(df=raw_df).process_data()
    assert cleaned is not None
    rfm_df = RFMEngine(cleaned).generate_rfm()
    clustering = CustomerClustering(rfm_df, random_seed=42)

    # Act
    scaled = clustering.preprocess_for_ml()
    clustered = clustering.fit_kmeans(n_clusters=4)
    summaries = clustering.summarize_clusters()

    # Assert
    assert scaled.shape == (30, 3)
    assert clustered["ml_cluster"].nunique() == 4
    assert clustering.silhouette_score_ is not None
    assert -1.0 <= clustering.silhouette_score_ <= 1.0
    assert len(summaries) == 4
    assert sum(item["customer_count"] for item in summaries) == 30


def test_customer_clustering_guards_invalid_inputs() -> None:
    """Reject invalid cluster bounds, missing features, and premature summaries."""
    # Arrange
    rfm_df = pd.DataFrame(
        [
            {"customer_id": 1, "recency": 5, "frequency": 2, "monetary": 100.0},
            {"customer_id": 2, "recency": 20, "frequency": 1, "monetary": 50.0},
        ]
    )
    clustering = CustomerClustering(rfm_df)

    # Act & Assert
    with pytest.raises(ValueError, match="run fit_kmeans"):
        clustering.summarize_clusters()

    with pytest.raises(ValueError, match="n_clusters must be >= 2"):
        clustering.fit_kmeans(n_clusters=1)

    with pytest.raises(ValueError, match="cannot exceed customer count"):
        clustering.fit_kmeans(n_clusters=5)


def test_segment_visualizer_renders_headless_pngs(tmp_path: Path) -> None:
    """Render both charts headlessly to disk without interactive display windows."""
    # Arrange
    raw_df = generate_sample_transactions(n_customers=15, seed=7)
    cleaned = DataPreprocessor(df=raw_df).process_data()
    assert cleaned is not None
    rfm_df = RFMEngine(cleaned).generate_rfm()
    clustered = CustomerClustering(rfm_df, random_seed=7).fit_kmeans(n_clusters=3)
    visualizer = SegmentVisualizer(clustered)

    dist_path = tmp_path / "charts" / "distribution.png"
    scatter_path = tmp_path / "charts" / "scatter.png"

    # Act
    saved_dist = visualizer.plot_segment_distribution(output_path=dist_path, show=False)
    saved_scatter = visualizer.plot_ml_clusters_scatter(output_path=scatter_path, show=False)

    # Assert
    assert saved_dist == dist_path and dist_path.exists() and dist_path.stat().st_size > 0
    assert saved_scatter == scatter_path and scatter_path.exists() and scatter_path.stat().st_size > 0


def test_cli_demo_and_health_modes(tmp_path: Path) -> None:
    """Verify CLI --health and --demo modes complete cleanly with expected artifacts."""
    # Arrange
    output_dir = tmp_path / "cli_out"

    # Act
    health_code = main(["--health"])
    demo_code = main(["--demo", "--clusters", "3", "--output-dir", str(output_dir)])

    # Assert
    assert health_code == 0
    assert demo_code == 0
    assert (output_dir / "segmentation_summary.json").exists()
    assert (output_dir / "segment_distribution.png").exists()
    assert (output_dir / "ml_clusters_scatter.png").exists()
