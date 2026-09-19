"""Evidence for outer × inner × service-point × lead sample sufficiency."""
from __future__ import annotations

import pandas as pd

from .rolling import OuterFold, times, window
from .split_registry import manifest, protocol, purge_days, utc


def _count(frame: pd.DataFrame, sequence_hours: int) -> tuple[int, int, int]:
    if frame.empty:
        return 0, 0, 0
    if "solar_elevation" not in frame:
        raise KeyError("solar_elevation required for formal daytime diagnostic")
    raw = len(frame)
    day = int((frame["solar_elevation"] > 0).sum())
    # A valid 168-hour window requires unbroken observed hourly records from
    # this station/lead. The target row must itself be daytime.
    ordered = frame.assign(_time=times(frame)).sort_values("_time")
    delta = ordered["_time"].diff()
    run_id = delta.ne(pd.Timedelta(hours=1)).cumsum()
    run_length = ordered.groupby(run_id).cumcount() + 1
    sequence_valid = int(((run_length >= sequence_hours) &
                          (ordered["solar_elevation"] > 0)).sum())
    return raw, day, sequence_valid


def first_outer_sample_report(frame: pd.DataFrame, *, gap_days: int | None = None,
                              service_column: str | None = None) -> pd.DataFrame:
    """Report requested six counts even if a configured inner fold is empty.

    This is a diagnostic only and does not authorize shortening the protocol.
    """
    service = service_column or ("location_id" if "location_id" in frame else "station_id")
    if service not in frame or "lead_time" not in frame:
        raise KeyError(f"{service} and lead_time are required")
    gap = purge_days() if gap_days is None else gap_days
    outer_spec = protocol()["outer_folds"][0]
    outer_start = utc(outer_spec["train_start"])
    outer_end = utc(outer_spec["validation_start"]) - pd.Timedelta(days=gap)
    outer = window(frame, outer_start, outer_end)
    score_days = int(protocol()["inner_scoring_days"])
    early_days = int(protocol()["inner_early_stop_days"])
    n = int(manifest()["tuning_budget"]["common_inner_folds"])
    sequence_hours = int(protocol()["maximum_sequence_lookback_hours"])
    rows = []
    keys = frame[[service, "lead_time"]].drop_duplicates().itertuples(index=False, name=None)
    for location, lead in keys:
        group = outer[(outer[service] == location) & (outer["lead_time"] == lead)]
        for j in range(n):
            scoring_end = outer_end - pd.Timedelta(days=(n - j - 1) * score_days)
            scoring_start = scoring_end - pd.Timedelta(days=score_days)
            early_end = scoring_start - pd.Timedelta(days=gap)
            early_start = early_end - pd.Timedelta(days=early_days)
            fit_end = early_start - pd.Timedelta(days=gap)
            blocks = {
                "fit": window(group, outer_start, fit_end),
                "early_stop": window(group, early_start, early_end),
                "scoring": window(group, scoring_start, scoring_end),
            }
            counts = {label: _count(block, sequence_hours) for label, block in blocks.items()}
            rows.append({"outer_fold": outer_spec["id"], "inner_fold": f"inner_{j + 1}",
                         service: location, "lead_time": lead,
                         "raw_rows": sum(v[0] for v in counts.values()),
                         "daytime_rows": sum(v[1] for v in counts.values()),
                         "sequence_168_valid": sum(v[2] for v in counts.values()),
                         **{f"{name}_rows": values[0] for name, values in counts.items()},
                         "fit_end": fit_end.isoformat(),
                         "protocol_feasible": fit_end > outer_start and all(v[0] > 0 for v in counts.values())})
    return pd.DataFrame(rows)
