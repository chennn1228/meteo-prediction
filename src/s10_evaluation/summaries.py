"""Evaluation outputs keep probability ranking separate from point auxiliaries."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EvaluationTables:
    probability_primary: pd.DataFrame
    point_secondary: pd.DataFrame
    reliability: pd.DataFrame


@dataclass(frozen=True)
class EvaluationReport:
    overview: EvaluationTables
    grouped: dict[str, EvaluationTables]


def rank_probability_models(table: pd.DataFrame) -> pd.DataFrame:
    """Rank only seven-quantile models by the protocol's mean pinball score."""
    required = {"prediction_type", "mean_pinball", "model_id"}
    if not required.issubset(table.columns):
        raise ValueError(f"probability table missing {sorted(required - set(table.columns))}")
    if (table.prediction_type != "quantile").any():
        raise ValueError("point-only baselines cannot enter probability model ranking")
    return table.sort_values(["mean_pinball", "model_id"], kind="stable").reset_index(drop=True)
