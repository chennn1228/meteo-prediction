"""Serializable split specifications and sample-sufficiency diagnostics."""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from nwp.core.config import to_plain

from .rolling import purge_days, times, utc, window


def split_specs(validation: Mapping[str, Any]) -> dict[str, Any]:
    return to_plain(
        {
            "gap_days": purge_days(validation),
            "inner_folds": validation["inner_folds"],
            "early_stop_days": validation["early_stop_days"],
            "scoring_days": validation["scoring_days"],
            "outer_folds": validation["outer_folds"],
        }
    )


def _count(frame: pd.DataFrame, sequence_hours: int) -> tuple[int, int, int]:
    if frame.empty:
        return 0, 0, 0
    if "solar_elevation" not in frame:
        raise KeyError("solar_elevation required for formal daytime diagnostic")
    raw = len(frame)
    day = int((frame["solar_elevation"] > 0).sum())
    ordered = frame.assign(_time=times(frame)).sort_values("_time")
    run_id = ordered["_time"].diff().ne(pd.Timedelta(hours=1)).cumsum()
    run_length = ordered.groupby(run_id).cumcount() + 1
    sequence_valid = int(
        ((run_length >= sequence_hours) & (ordered["solar_elevation"] > 0)).sum()
    )
    return raw, day, sequence_valid


def first_outer_sample_report(
    frame: pd.DataFrame,
    validation: Mapping[str, Any],
    development_period: Mapping[str, str],
    *,
    gap_days: int | None = None,
    service_column: str | None = None,
) -> pd.DataFrame:
    """Report fit/early/score sufficiency without relaxing the protocol."""
    service = service_column or (
        "location_id" if "location_id" in frame else "station_id"
    )
    if service not in frame or "lead_time" not in frame:
        raise KeyError(f"{service} and lead_time are required")
    gap = purge_days(validation) if gap_days is None else int(gap_days)
    if gap not in tuple(int(value) for value in validation["gap_sensitivity_days"]):
        raise ValueError("gap_days must be a registered sensitivity value")
    outer_spec = validation["outer_folds"][0]
    outer_start = utc(outer_spec["train_start"])
    outer_end = utc(outer_spec["validation_start"]) - pd.Timedelta(days=gap)
    outer = window(frame, outer_start, outer_end)
    score_days = int(validation["scoring_days"])
    early_days = int(validation["early_stop_days"])
    n = int(validation["inner_folds"])
    sequence_hours = int(validation["maximum_sequence_lookback_hours"])
    origin = utc(development_period["start"])
    rows = []
    keys = frame[[service, "lead_time"]].drop_duplicates().itertuples(
        index=False, name=None
    )
    for location, lead in keys:
        group = outer[
            (outer[service] == location) & (outer["lead_time"] == lead)
        ]
        for index in range(n):
            score_end = outer_end - pd.Timedelta(days=(n - index - 1) * score_days)
            score_start = score_end - pd.Timedelta(days=score_days)
            early_end = score_start - pd.Timedelta(days=gap)
            early_start = early_end - pd.Timedelta(days=early_days)
            fit_end = early_start - pd.Timedelta(days=gap)
            blocks = {
                "fit": window(group, origin, fit_end),
                "early_stop": window(group, early_start, early_end),
                "scoring": window(group, score_start, score_end),
            }
            counts = {
                label: _count(block, sequence_hours)
                for label, block in blocks.items()
            }
            rows.append(
                {
                    "outer_fold": outer_spec["id"],
                    "inner_fold": f"inner_{index + 1}",
                    service: location,
                    "lead_time": lead,
                    "raw_rows": sum(value[0] for value in counts.values()),
                    "daytime_rows": sum(value[1] for value in counts.values()),
                    "sequence_168_valid": sum(value[2] for value in counts.values()),
                    **{
                        f"{name}_rows": values[0]
                        for name, values in counts.items()
                    },
                    "fit_end": fit_end.isoformat(),
                    "protocol_feasible": fit_end > origin
                    and all(value[0] > 0 for value in counts.values()),
                }
            )
    return pd.DataFrame(rows)
