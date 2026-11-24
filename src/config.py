"""Runtime configuration and structured logging setup."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import sys

from dotenv import load_dotenv

VALID_LOG_LEVELS: frozenset[str] = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
MIN_CLUSTERS = 2
MAX_CLUSTERS = 20
DEFAULT_ALLOWED_ORIGINS: tuple[str, ...] = (
    "http://localhost:3000",
    "http://127.0.0.1:8000",
)


def _parse_int_env(name: str, default: int) -> int:
    """Parse an integer environment variable with explicit error context."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}.") from exc


def _parse_origins(raw_origins: str | None) -> tuple[str, ...]:
    """Parse comma-separated CORS origins into a normalized tuple."""
    if raw_origins is None or not raw_origins.strip():
        return DEFAULT_ALLOWED_ORIGINS
    origins = tuple(item.strip() for item in raw_origins.split(",") if item.strip())
    return origins or DEFAULT_ALLOWED_ORIGINS


@dataclass(frozen=True)
class SegmentationConfig:
    """Validated runtime settings for the RFM pipeline and FastAPI service."""

    data_path: str = "Online Retail Data.csv"
    output_dir: str = "outputs"
    default_clusters: int = 4
    random_seed: int = 42
    allowed_origins: tuple[str, ...] = DEFAULT_ALLOWED_ORIGINS
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, load_env_file: bool = True) -> SegmentationConfig:
        """Load configuration from environment variables and validate bounds."""
        if load_env_file:
            load_dotenv()

        config = cls(
            data_path=os.getenv("DATA_PATH", "Online Retail Data.csv").strip(),
            output_dir=os.getenv("OUTPUT_DIR", "outputs").strip(),
            default_clusters=_parse_int_env("DEFAULT_CLUSTERS", 4),
            random_seed=_parse_int_env("RANDOM_SEED", 42),
            allowed_origins=_parse_origins(os.getenv("ALLOWED_ORIGINS")),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate all runtime bounds and invariants upfront."""
        if not self.data_path:
            raise ValueError("DATA_PATH must not be empty.")
        if not self.output_dir:
            raise ValueError("OUTPUT_DIR must not be empty.")
        if not (MIN_CLUSTERS <= self.default_clusters <= MAX_CLUSTERS):
            raise ValueError(
                f"DEFAULT_CLUSTERS must be between {MIN_CLUSTERS} and {MAX_CLUSTERS}, got {self.default_clusters}."
            )
        if self.random_seed < 0:
            raise ValueError(f"RANDOM_SEED must be non-negative, got {self.random_seed}.")
        if self.log_level.upper() not in VALID_LOG_LEVELS:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(VALID_LOG_LEVELS)}, got {self.log_level!r}.")
        if not self.allowed_origins:
            raise ValueError("ALLOWED_ORIGINS must contain at least one origin.")


Settings = SegmentationConfig


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure structured leveled stdout logging without breaking pytest caplog."""
    normalized = level.strip().upper()
    if normalized not in VALID_LOG_LEVELS:
        raise ValueError(f"Invalid LOG_LEVEL {level!r}; expected one of {sorted(VALID_LOG_LEVELS)}.")

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        root_logger.addHandler(handler)

    root_logger.setLevel(getattr(logging, normalized))
    return logging.getLogger("customer_segmentation")
