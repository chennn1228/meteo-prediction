"""Preprocessing objects fitted only on the current training fold."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from s01_core.config_loader import ProtocolError
from s01_core.schemas import assert_model_features


@dataclass
class FoldPreprocessor:
    columns: tuple[str, ...]
    medians: pd.Series | None = None
    means: pd.Series | None = None
    scales: pd.Series | None = None
    fit_end_utc: pd.Timestamp | None = None
    train_rows: int = 0

    def fit(self, training: pd.DataFrame, *, fit_end_utc: str | pd.Timestamp) -> "FoldPreprocessor":
        assert_model_features(self.columns)
        boundary = pd.Timestamp(fit_end_utc)
        boundary = boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")
        issue = pd.to_datetime(training["forecast_issue_time_utc"], utc=True)
        if training.empty or issue.isna().any() or (issue > boundary).any():
            raise ProtocolError("preprocessor fit contains later/invalid issue times")
        matrix = training.loc[:, list(self.columns)].apply(pd.to_numeric, errors="coerce")
        if matrix.notna().sum().eq(0).any():
            raise ProtocolError("training fold contains all-missing formal feature")
        self.medians = matrix.median()
        imputed = matrix.fillna(self.medians)
        self.means = imputed.mean()
        scale = imputed.std(ddof=0)
        self.scales = scale.mask(scale.eq(0), 1.0)
        self.fit_end_utc = boundary
        self.train_rows = len(training)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.medians is None or self.means is None or self.scales is None:
            raise ProtocolError("preprocessor must be fit on the training fold")
        matrix = frame.loc[:, list(self.columns)].apply(pd.to_numeric, errors="coerce")
        values = (matrix.fillna(self.medians) - self.means) / self.scales
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ProtocolError("non-finite value after frozen fold preprocessing")
        return values
