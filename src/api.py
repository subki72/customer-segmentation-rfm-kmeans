"""FastAPI microservice exposing RFM + K-Means segmentation and liveness probes."""

from __future__ import annotations

from datetime import UTC, datetime
import logging

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd

from src.config import SegmentationConfig, configure_logging
from src.data_pipeline import DataPreprocessor
from src.ml_clustering import CustomerClustering
from src.rfm_engine import RFMEngine
from src.schemas import (
    ClusterSummary,
    CustomerSegmentResult,
    HealthResponse,
    SegmentationRequest,
    SegmentationResponse,
)

logger = logging.getLogger(__name__)


def _execute_segmentation(payload: SegmentationRequest, random_seed: int) -> SegmentationResponse:
    """Run preprocessing, RFM scoring, and K-Means clustering for an API payload."""
    raw_df = pd.DataFrame([item.model_dump() for item in payload.transactions])
    cleaned_df = DataPreprocessor(df=raw_df).process_data()
    if cleaned_df is None or cleaned_df.empty:
        raise HTTPException(
            status_code=422,
            detail="No valid non-cancelled transactions available after preprocessing.",
        )

    rfm_df = RFMEngine(cleaned_df).generate_rfm()
    unique_customers = int(rfm_df["customer_id"].nunique())
    if unique_customers < payload.n_clusters:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unique valid customer count ({unique_customers}) is smaller than "
                f"requested n_clusters ({payload.n_clusters})."
            ),
        )

    clustering = CustomerClustering(rfm_df, random_seed=random_seed)
    clustering.preprocess_for_ml()
    segmented_df = clustering.fit_kmeans(n_clusters=payload.n_clusters)
    summaries = [ClusterSummary(**entry) for entry in clustering.summarize_clusters()]
    segments = [CustomerSegmentResult(**record) for record in segmented_df.to_dict("records")]

    return SegmentationResponse(
        total_transactions=len(cleaned_df),
        total_customers=unique_customers,
        n_clusters=payload.n_clusters,
        silhouette_score=clustering.silhouette_score_,
        cluster_summary=summaries,
        segments=segments,
    )


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log internal traceback safely and return generic 500 JSON to clients."""
    logger.exception("Unhandled error while processing %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error processing segmentation request."},
    )


def create_app(config: SegmentationConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI segmentation microservice."""
    settings = config or SegmentationConfig.from_env()
    configure_logging(settings.log_level)

    service = FastAPI(
        title="Customer Segmentation RFM & K-Means API",
        version="1.0.0",
        description="Production RFM scoring and K-Means customer clustering microservice.",
    )
    service.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )
    service.add_exception_handler(Exception, _unhandled_exception_handler)

    def get_settings() -> SegmentationConfig:
        """Provide validated runtime configuration via dependency injection."""
        return settings

    @service.get("/health", response_model=HealthResponse, tags=["Operations"])
    def health_check() -> HealthResponse:
        """Liveness probe endpoint confirming API availability."""
        return HealthResponse(
            status="healthy",
            service="customer-segmentation-rfm-kmeans",
            version="1.0.0",
            timestamp=datetime.now(UTC),
        )

    @service.post("/analyze", response_model=SegmentationResponse, tags=["Segmentation"])
    def analyze_customers(
        payload: SegmentationRequest,
        runtime_config: SegmentationConfig = Depends(get_settings),
    ) -> SegmentationResponse:
        """Segment retail customers using RFM metrics and K-Means clustering."""
        return _execute_segmentation(payload=payload, random_seed=runtime_config.random_seed)

    return service


app = create_app()
