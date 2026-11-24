"""CLI entrypoint for running the RFM & K-Means customer segmentation pipeline."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from typing import Any

from src.config import SegmentationConfig, configure_logging
from src.data_pipeline import DataPreprocessor, generate_sample_transactions
from src.ml_clustering import CustomerClustering
from src.rfm_engine import RFMEngine
from src.visualizer import SegmentVisualizer

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser for batch and demo execution."""
    parser = argparse.ArgumentParser(description="Run end-to-end RFM & K-Means Customer Segmentation pipeline.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to raw transaction CSV file (defaults to DATA_PATH env var).",
    )
    parser.add_argument(
        "--clusters",
        "-k",
        type=int,
        default=None,
        help="Number of K-Means customer clusters (defaults to DEFAULT_CLUSTERS env var).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory for generated plots and JSON summary (defaults to OUTPUT_DIR).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run pipeline on built-in deterministic synthetic retail transactions.",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Validate configuration and report service health status.",
    )
    return parser


def run_pipeline(
    config: SegmentationConfig,
    input_path: str | None = None,
    clusters: int | None = None,
    output_dir: str | None = None,
    demo: bool = False,
) -> dict[str, Any]:
    """Execute ingestion, RFM scoring, clustering, and artifact generation."""
    n_clusters = clusters if clusters is not None else config.default_clusters
    target_dir = Path(output_dir or config.output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if demo:
        logger.info("Running pipeline in synthetic demo mode...")
        sample_df = generate_sample_transactions(seed=config.random_seed)
        preprocessor = DataPreprocessor(df=sample_df)
    else:
        dataset_path = input_path or config.data_path
        preprocessor = DataPreprocessor(file_path=dataset_path)

    cleaned_df = preprocessor.process_data()
    if cleaned_df is None or cleaned_df.empty:
        raise RuntimeError("Pipeline aborted: dataset could not be loaded or contained no valid rows.")

    rfm_df = RFMEngine(cleaned_df).generate_rfm()
    clustering = CustomerClustering(rfm_df, random_seed=config.random_seed)
    clustering.preprocess_for_ml()
    segmented_df = clustering.fit_kmeans(n_clusters=n_clusters)
    cluster_summary = clustering.summarize_clusters()

    visualizer = SegmentVisualizer(segmented_df)
    dist_plot = visualizer.plot_segment_distribution(output_path=target_dir / "segment_distribution.png")
    scatter_plot = visualizer.plot_ml_clusters_scatter(output_path=target_dir / "ml_clusters_scatter.png")

    summary_payload = {
        "total_transactions": int(len(cleaned_df)),
        "total_customers": int(segmented_df["customer_id"].nunique()),
        "n_clusters": int(n_clusters),
        "silhouette_score": clustering.silhouette_score_,
        "cluster_summary": cluster_summary,
        "artifacts": {
            "segment_distribution_plot": str(dist_plot) if dist_plot else None,
            "ml_clusters_scatter_plot": str(scatter_plot) if scatter_plot else None,
        },
    }
    summary_file = target_dir / "segmentation_summary.json"
    summary_file.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    logger.info("Pipeline completed successfully. Summary saved to %s", summary_file)
    return summary_payload


def main(argv: list[str] | None = None) -> int:
    """Parse CLI options, execute requested workflow, and return exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = SegmentationConfig.from_env()
        configure_logging(config.log_level)

        if args.health:
            health_payload = {
                "status": "healthy",
                "service": "customer-segmentation-rfm-kmeans",
                "timestamp": datetime.now(UTC).isoformat(),
            }
            logger.info("Health check passed: %s", json.dumps(health_payload))
            return 0

        run_pipeline(
            config=config,
            input_path=args.input,
            clusters=args.clusters,
            output_dir=args.output_dir,
            demo=args.demo,
        )
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        logger.error("Pipeline execution failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
