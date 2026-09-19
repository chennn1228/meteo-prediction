"""Canonical per-sample prediction contract (not a model feature interface)."""

from .schema import QUANTILES, QUANTILE_COLUMNS, REQUIRED_COLUMNS, validate_predictions

__all__ = ["QUANTILES", "QUANTILE_COLUMNS", "REQUIRED_COLUMNS", "validate_predictions"]
