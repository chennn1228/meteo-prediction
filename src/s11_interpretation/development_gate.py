"""Prevent final-test labels from influencing feature selection or SHAP sampling."""
from __future__ import annotations

import pandas as pd

from s01_core.config_loader import ProtocolError, load_manifest


def require_development_rows(frame: pd.DataFrame) -> None:
    if "target_time_utc" not in frame:
        raise ProtocolError("feature analysis requires target_time_utc")
    target = pd.to_datetime(frame["target_time_utc"], utc=True)
    config = load_manifest()
    start = pd.Timestamp(config["development_period"]["start"])
    end = pd.Timestamp(config["development_period"]["end"])
    if frame.empty or target.isna().any() or (target < start).any() or (target > end).any():
        raise ProtocolError("feature mechanism analysis must use development period only")
