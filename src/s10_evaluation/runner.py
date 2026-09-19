"""Read-only formal evaluation entry; it never trains or writes results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from s07_prediction.reader import read_predictions
from s07_prediction.schema import validate_predictions

from .grouped import evaluate_groups
from .summaries import EvaluationReport


def evaluate_predictions(predictions: pd.DataFrame | str | Path,
                         *, formal: bool = False,
                         groupings: tuple[str, ...] | None = None) -> EvaluationReport:
    """Evaluate a canonical prediction table without creating official output.

    ``formal=True`` is a gate, not a mode that upgrades provisional rows. It
    accepts only validated/official/official rows and still performs no IO.
    Raw GFS point forecasts appear only in point_secondary, never the
    probability ranking. Each grouped table is scored from its own rows.
    """
    clean = (read_predictions(predictions) if isinstance(predictions, (str, Path))
             else validate_predictions(predictions))
    if formal and not (
        (clean.implementation_level == "validated") &
        (clean.execution_level == "official") &
        (clean.result_status == "official")
    ).all():
        raise ValueError("formal evaluation rejects prototype, development or nonofficial rows")
    dimensions = groupings if groupings is not None else (
        "lead_time", "location_id", "outer_fold", "season"
    )
    overview = evaluate_groups(clean, require_references=formal)
    grouped = {
        name: evaluate_groups(clean, (name,), require_references=formal)
        for name in dimensions if name in clean.columns
    }
    return EvaluationReport(overview=overview, grouped=grouped)
