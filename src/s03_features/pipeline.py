"""Single formal clean → returned-service physics → causal features chain."""
from __future__ import annotations

import pandas as pd

from s01_core.config_loader import ProtocolError
from .engineering import build_forecast_features, formal_feature_columns
from .physics import add_returned_service_physics


def prepare_formal_features(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if "fcst_issue_time_utc" in cleaned:
        raise ProtocolError("old issue-time alias is legacy-only; use forecast_issue_time_utc")
    physical = add_returned_service_physics(cleaned)
    featured = build_forecast_features(physical)
    columns = tuple(formal_feature_columns(featured))
    return featured, columns
