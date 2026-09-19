"""Fail-closed contract for monthly Open-Meteo raw payloads and caches.

Raw API JSON is kept unchanged. A separate sidecar ties it to the current
protocol data version and complete request; an old 15-variable file without
this sidecar cannot silently satisfy the 18-variable forecast contract.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping


class DataContractError(ValueError):
    """A raw response or cleaned frame does not satisfy the protocol."""


def expected_hourly_fields(source: str, cfg: Mapping) -> tuple[str, ...]:
    if source == "previous_runs":
        variables = tuple(cfg["forecast_variables"])
        if len(variables) != 18 or len(set(variables)) != 18:
            raise DataContractError("Forecast contract requires 18 distinct variables")
        leads = tuple(cfg["leads"])
        if leads != (1, 2, 3):
            raise DataContractError("Forecast contract requires D+1/D+2/D+3")
        return tuple(f"{name}_previous_day{lead}" for name in variables for lead in leads)
    key = {"satellite": "satellite_variables", "era5": "era5_variables"}.get(source)
    if key is None:
        raise DataContractError(f"Unknown source: {source}")
    return tuple(cfg[key])


def expected_utc_hours(start: dt.date, end: dt.date) -> list[dt.datetime]:
    if start > end:
        raise DataContractError("start exceeds end")
    first = dt.datetime.combine(start, dt.time(), dt.timezone.utc)
    count = (end - start).days * 24 + 24
    return [first + dt.timedelta(hours=i) for i in range(count)]


def _parse_hour(value: str) -> dt.datetime:
    try:
        instant = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"Invalid hourly timestamp: {value!r}") from exc
    if instant.tzinfo is None:
        # Open-Meteo emits naive UTC strings when timezone=UTC.
        instant = instant.replace(tzinfo=dt.timezone.utc)
    return instant.astimezone(dt.timezone.utc)


def validate_raw_payload(payload: Mapping, *, source: str, cfg: Mapping,
                         start: dt.date, end: dt.date) -> None:
    if not isinstance(payload, Mapping) or payload.get("error"):
        raise DataContractError("API response is not a successful JSON object")
    for axis, bound in (("latitude", 90), ("longitude", 180)):
        value = payload.get(axis)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > bound:
            raise DataContractError(f"Missing or invalid returned service {axis}")
    hourly = payload.get("hourly")
    if not isinstance(hourly, Mapping):
        raise DataContractError("Missing hourly object")
    expected = expected_utc_hours(start, end)
    values = hourly.get("time")
    if not isinstance(values, list) or len(values) != len(expected):
        raise DataContractError(f"Hourly length differs from expected {len(expected)}")
    parsed = [_parse_hour(value) for value in values]
    if parsed != expected:
        raise DataContractError("Hourly timestamps are duplicated, missing, unordered, or out of range")
    for field in expected_hourly_fields(source, cfg):
        series = hourly.get(field)
        if not isinstance(series, list) or len(series) != len(expected):
            raise DataContractError(f"Missing or length-mismatched field: {field}")
        if all(value is None for value in series):
            raise DataContractError(f"All values are null despite field presence: {field}")


def cache_metadata(*, source: str, cfg: Mapping, start: dt.date, end: dt.date,
                   params: Mapping, data_version: str) -> dict:
    if not data_version:
        raise DataContractError("Missing data_version")
    return {
        "schema": "openmeteo-monthly-v2",
        "data_version": data_version,
        "source": source,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "required_fields": list(expected_hourly_fields(source, cfg)),
        "request": dict(sorted(params.items())),
    }
