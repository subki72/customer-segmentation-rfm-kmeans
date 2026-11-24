"""Pydantic v2 request and response schemas for customer segmentation API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TransactionItem(BaseModel):
    """Single retail line-item transaction record."""

    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(..., min_length=1, max_length=64, description="Unique invoice/order identifier.")
    customer_id: int = Field(..., gt=0, description="Positive integer customer identifier.")
    order_date: datetime | str = Field(..., description="ISO-8601 transaction timestamp or date string.")
    quantity: int = Field(..., gt=0, le=1_000_000, description="Purchased item quantity (> 0).")
    price: float = Field(..., gt=0.0, le=10_000_000.0, description="Unit item price (> 0).")

    @field_validator("order_id")
    @classmethod
    def _validate_order_id(cls, value: str) -> str:
        """Ensure order_id is non-blank after stripping whitespace."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("order_id must not be blank.")
        return cleaned

    @field_validator("order_date")
    @classmethod
    def _parse_order_date(cls, value: datetime | str) -> datetime:
        """Parse ISO-8601 date/datetime strings consistently."""
        if isinstance(value, datetime):
            return value
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("order_date must not be empty.")
        try:
            return datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid ISO-8601 order_date format: {value!r}") from exc


class SegmentationRequest(BaseModel):
    """API request payload for real-time RFM and K-Means segmentation."""

    model_config = ConfigDict(extra="forbid")

    transactions: list[TransactionItem] = Field(
        ...,
        min_length=1,
        max_length=50_000,
        description="List of retail transactions to segment.",
    )
    n_clusters: int = Field(
        default=4,
        ge=2,
        le=10,
        description="Number of K-Means customer clusters (2-10).",
    )

    @model_validator(mode="after")
    def _validate_customer_cardinality(self) -> SegmentationRequest:
        """Ensure non-cancelled customer count supports requested cluster count."""
        valid_customers = {item.customer_id for item in self.transactions if not item.order_id.upper().startswith("C")}
        if len(valid_customers) < self.n_clusters:
            raise ValueError(
                f"Valid unique customers ({len(valid_customers)}) must be >= n_clusters ({self.n_clusters})."
            )
        return self


class CustomerSegmentResult(BaseModel):
    """Per-customer RFM metrics, actionable tier, and assigned K-Means cluster."""

    customer_id: int
    recency: int
    frequency: int
    monetary: float
    r_score: int
    f_score: int
    m_score: int
    rf_score: str
    base_segment: str
    value_tier: str
    actionable_segment: str
    ml_cluster: int


class ClusterSummary(BaseModel):
    """Aggregated behavioral statistics for a single K-Means cluster."""

    cluster_id: int
    customer_count: int
    avg_recency: float
    avg_frequency: float
    avg_monetary: float
    dominant_segment: str


class SegmentationResponse(BaseModel):
    """Complete RFM + K-Means segmentation API response."""

    total_transactions: int
    total_customers: int
    n_clusters: int
    silhouette_score: float | None = None
    cluster_summary: list[ClusterSummary]
    segments: list[CustomerSegmentResult]


class HealthResponse(BaseModel):
    """Liveness health-check response payload."""

    status: str = "healthy"
    service: str = "customer-segmentation-rfm-kmeans"
    version: str = "1.0.0"
    timestamp: datetime
