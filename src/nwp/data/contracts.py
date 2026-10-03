"""Receipt-backed catalog and partition contracts for persisted research data."""
from __future__ import annotations

import json
import datetime as dt
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from nwp.core.paths import RunPaths
from nwp.core.provenance import read_receipt
from nwp.core.hashing import file_sha256
from nwp.core.schema import ContractError


CATALOG_SCHEMA_VERSION = 1
DATA_STAGES = frozenset({"raw", "clean", "features"})
DATASET_STATUSES = frozenset({"ready", "incomplete", "quarantined"})
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_HEX = re.compile(r"^[0-9a-f]{8,64}$")


def _source_variables(source: str, data_config: Mapping[str, Any]) -> tuple[str, ...]:
    if source == "previous_runs":
        forecast = data_config["forecast"]
        variables = tuple(forecast["variables"])
        leads = tuple(int(value) for value in forecast["leads"])
        if not variables or len(variables) != len(set(variables)):
            raise ContractError("forecast variables must be nonempty and unique")
        if not leads or any(value <= 0 or value % 24 for value in leads):
            raise ContractError("Previous Runs leads must be positive whole days expressed in hours")
        return tuple(
            f"{variable}_previous_day{lead // 24}"
            for variable in variables
            for lead in leads
        )
    if source == "satellite":
        return tuple(data_config["truth"]["primary"]["variables"])
    if source == "era5":
        return tuple(data_config["truth"]["supplementary"]["variables"])
    raise ContractError(f"unknown data source: {source}")


def expected_hourly_fields(source: str, data_config: Mapping[str, Any]) -> tuple[str, ...]:
    return _source_variables(source, data_config)


def expected_utc_hours(start: dt.date, end: dt.date) -> list[dt.datetime]:
    if start > end:
        raise ContractError("start exceeds end")
    first = dt.datetime.combine(start, dt.time(), dt.timezone.utc)
    count = (end - start).days * 24 + 24
    return [first + dt.timedelta(hours=index) for index in range(count)]


def _parse_hour(value: str) -> dt.datetime:
    try:
        instant = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ContractError(f"invalid hourly timestamp: {value!r}") from exc
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=dt.timezone.utc)
    return instant.astimezone(dt.timezone.utc)


def validate_raw_payload(
    payload: Mapping[str, Any],
    *,
    source: str,
    data_config: Mapping[str, Any],
    start: dt.date,
    end: dt.date,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("error"):
        raise ContractError("API response is not a successful JSON object")
    returned: dict[str, float | None] = {}
    for axis, bound in (("latitude", 90), ("longitude", 180)):
        value = payload.get(axis)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > bound:
            raise ContractError(f"missing or invalid returned service {axis}")
        returned[axis] = float(value)
    elevation = payload.get("elevation")
    returned["elevation"] = float(elevation) if isinstance(elevation, (int, float)) and math.isfinite(elevation) else None
    hourly = payload.get("hourly")
    if not isinstance(hourly, Mapping):
        raise ContractError("missing hourly object")
    expected = expected_utc_hours(start, end)
    values = hourly.get("time")
    if not isinstance(values, list) or len(values) != len(expected):
        raise ContractError(f"hourly length differs from expected {len(expected)}")
    if [_parse_hour(value) for value in values] != expected:
        raise ContractError("hourly timestamps are duplicated, missing, unordered, or out of range")
    missingness: dict[str, float] = {}
    for field in expected_hourly_fields(source, data_config):
        series = hourly.get(field)
        if not isinstance(series, list) or len(series) != len(expected):
            raise ContractError(f"missing or length-mismatched field: {field}")
        missing = sum(value is None for value in series)
        if missing == len(series):
            raise ContractError(f"all values are null despite field presence: {field}")
        missingness[field] = missing / len(series)
    return {
        "returned_coordinates": returned,
        "row_count": len(expected),
        "missingness": missingness,
    }


def expected_month_slices(start: dt.date, end: dt.date) -> Iterable[tuple[dt.date, dt.date]]:
    if start > end:
        raise ContractError("start exceeds end")
    current = start
    while current <= end:
        following = (current.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        stop = min(end, following - dt.timedelta(days=1))
        yield current, stop
        current = stop + dt.timedelta(days=1)


def validate_clean_frame(
    frame: Any,
    *,
    start: dt.date,
    end: dt.date,
    required_columns: Iterable[str],
    leads: Iterable[int],
) -> dict[str, Any]:
    import pandas as pd

    required = tuple(required_columns)
    missing_columns = set(required) - set(frame.columns)
    if missing_columns:
        raise ContractError(f"missing cleaned columns: {sorted(missing_columns)}")
    target = pd.to_datetime(frame["target_time_utc"], utc=True)
    issue = pd.to_datetime(frame["forecast_issue_time_utc"], utc=True)
    lead = pd.to_numeric(frame["lead_time"], errors="coerce")
    if target.isna().any() or issue.isna().any() or lead.isna().any():
        raise ContractError("null or unparsable target/issue/lead")
    registered_leads = {int(value) for value in leads}
    if set(int(value) for value in lead.unique()) != registered_leads:
        raise ContractError("cleaned data does not contain exactly the registered leads")
    if not ((target - issue) == pd.to_timedelta(lead, unit="h")).all():
        raise ContractError("forecast issue/target/lead inconsistency")
    if pd.DataFrame({"target": target, "lead": lead}).duplicated().any():
        raise ContractError("duplicate target time by lead records")
    expected = pd.date_range(start, end + dt.timedelta(days=1), freq="h", inclusive="left", tz="UTC")
    actual = pd.DatetimeIndex(target.drop_duplicates().sort_values())
    if not actual.equals(expected):
        raise ContractError("cleaned hourly timeline is incomplete or out of bounds")
    if len(frame) != len(expected) * len(registered_leads):
        raise ContractError("each target hour must contain every registered forecast lead")
    coordinate_columns = (
        "requested_latitude",
        "requested_longitude",
        "gfs_service_latitude",
        "gfs_service_longitude",
        "himawari_service_latitude",
        "himawari_service_longitude",
        "era5_service_latitude",
        "era5_service_longitude",
    )
    for column in coordinate_columns:
        if column not in frame or frame[column].isna().any():
            raise ContractError(f"missing coordinate: {column}")
    return {
        "expected_months": [f"{month:%Y-%m}" for month, _ in expected_month_slices(start, end)],
        "actual_months": sorted(set(actual.strftime("%Y-%m"))),
        "hourly_count": len(expected),
        "row_count": len(frame),
        "lead_hours": sorted(registered_leads),
        "missingness": {column: float(frame[column].isna().mean()) for column in required},
        "service_coordinates": {
            source: sorted(
                {
                    (float(lat), float(lon))
                    for lat, lon in zip(
                        frame[f"{source}_service_latitude"],
                        frame[f"{source}_service_longitude"],
                    )
                }
            )
            for source in ("gfs", "himawari", "era5")
        },
    }


def _relative_path(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"dataset {label} must be nonempty")
    pure = PurePosixPath(value.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise ContractError(f"dataset {label} must stay under data_root")
    return pure.as_posix()


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    stage: str
    config_hash: str
    sites: tuple[str, ...]
    time_range: str
    source_hashes: Mapping[str, str]
    status: str
    path: str
    receipt_path: str

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.dataset_id):
            raise ContractError(f"unsafe dataset_id: {self.dataset_id!r}")
        if self.stage not in DATA_STAGES:
            raise ContractError(f"unsupported data stage: {self.stage}")
        if not _HEX.fullmatch(self.config_hash):
            raise ContractError("dataset config_hash must be lowercase hex")
        if not self.sites or len(self.sites) != len(set(self.sites)) or any(not _SAFE_ID.fullmatch(site) for site in self.sites):
            raise ContractError("dataset sites must be nonempty, unique, safe IDs")
        if not isinstance(self.time_range, str) or not self.time_range:
            raise ContractError("dataset time_range must be nonempty")
        if self.status not in DATASET_STATUSES:
            raise ContractError(f"unsupported dataset status: {self.status}")
        hashes = dict(self.source_hashes)
        if any(not isinstance(key, str) or not key or not isinstance(value, str) or not _HEX.fullmatch(value) for key, value in hashes.items()):
            raise ContractError("dataset source_hashes must contain named lowercase hex hashes")
        object.__setattr__(self, "source_hashes", MappingProxyType(hashes))
        object.__setattr__(self, "path", _relative_path(self.path, "path"))
        object.__setattr__(self, "receipt_path", _relative_path(self.receipt_path, "receipt_path"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "stage": self.stage,
            "config_hash": self.config_hash,
            "sites": list(self.sites),
            "time_range": self.time_range,
            "source_hashes": dict(self.source_hashes),
            "status": self.status,
            "path": self.path,
            "receipt_path": self.receipt_path,
        }


class DataCatalog:
    """The only resolver for persisted data; callers never concatenate paths."""

    def __init__(self, paths: RunPaths):
        self.paths = paths
        self.data_root = paths.data_root
        self.path = paths.catalog_path

    def _absolute(self, relative: str) -> Path:
        candidate = (self.data_root / _relative_path(relative, "path")).resolve()
        try:
            candidate.relative_to(self.data_root)
        except ValueError as exc:
            raise ContractError("catalog path escapes data_root") from exc
        return candidate

    @staticmethod
    def _record(item: Mapping[str, Any]) -> DatasetRecord:
        try:
            return DatasetRecord(
                dataset_id=item["dataset_id"],
                stage=item["stage"],
                config_hash=item["config_hash"],
                sites=tuple(item["sites"]),
                time_range=item["time_range"],
                source_hashes=dict(item["source_hashes"]),
                status=item["status"],
                path=item["path"],
                receipt_path=item["receipt_path"],
            )
        except (KeyError, TypeError) as exc:
            raise ContractError("catalog dataset entry is malformed") from exc

    @classmethod
    def record_from_dict(cls, item: Mapping[str, Any]) -> DatasetRecord:
        """Rehydrate a manifest record through the canonical validation path."""
        return cls._record(item)

    def _read(self) -> list[DatasetRecord]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError("data/catalog.json is unreadable") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != CATALOG_SCHEMA_VERSION or not isinstance(payload.get("datasets"), list):
            raise ContractError("data/catalog.json has an invalid schema")
        records = [self._record(item) for item in payload["datasets"]]
        ids = [record.dataset_id for record in records]
        if len(ids) != len(set(ids)):
            raise ContractError("data/catalog.json contains duplicate dataset IDs")
        return records

    def _write(self, records: Iterable[DatasetRecord]) -> None:
        ordered = sorted(records, key=lambda record: record.dataset_id)
        self.data_root.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        if temporary.exists():
            raise ContractError(f"stale catalog temporary file exists: {temporary}")
        try:
            temporary.write_text(
                json.dumps(
                    {"schema_version": CATALOG_SCHEMA_VERSION, "datasets": [record.as_dict() for record in ordered]},
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def dataset_path(self, record: DatasetRecord) -> Path:
        return self._absolute(record.path)

    def records(self) -> tuple[DatasetRecord, ...]:
        return tuple(self._read())

    def receipt_path(self, record: DatasetRecord) -> Path:
        return self._absolute(record.receipt_path)

    def _validate_ready(self, record: DatasetRecord) -> None:
        data_path = self.dataset_path(record)
        receipt_path = self.receipt_path(record)
        if not data_path.is_file() or not receipt_path.is_file():
            raise ContractError("ready datasets require an existing data file and receipt")
        expected_receipt = self.paths.data_receipt(data_path)
        if receipt_path != expected_receipt:
            raise ContractError("dataset receipt_path must be the canonical sidecar path")
        receipt = read_receipt(receipt_path)
        if receipt["stage"] != record.stage or receipt["config_hash"] != record.config_hash or receipt["status"] != "ready":
            raise ContractError("dataset receipt does not match its catalog record")
        if file_sha256(data_path) not in receipt["output_hashes"].values():
            raise ContractError("dataset content hash is absent from its receipt")
        if record.stage != "raw" and dict(record.source_hashes) != receipt["input_hashes"]:
            raise ContractError("processed dataset source hashes do not match its receipt inputs")

    def register(self, record: DatasetRecord) -> None:
        self.register_many((record,))

    def register_many(self, additions: Iterable[DatasetRecord]) -> None:
        """Validate and atomically add a batch without quadratic rewrites."""
        incoming = tuple(additions)
        incoming_ids = [record.dataset_id for record in incoming]
        if len(incoming_ids) != len(set(incoming_ids)):
            raise ContractError("catalog registration batch contains duplicate dataset IDs")
        records = self._read()
        by_id = {record.dataset_id: record for record in records}
        changed = False
        for record in incoming:
            existing = by_id.get(record.dataset_id)
            if existing is not None:
                if existing != record:
                    raise ContractError(
                        f"catalog dataset_id already identifies different data: {record.dataset_id}")
                continue
            if record.status == "ready":
                self._validate_ready(record)
            records.append(record)
            by_id[record.dataset_id] = record
            changed = True
        if changed:
            self._write(records)

    def resolve(
        self,
        *,
        stage: str,
        config_hash: str,
        sites: Iterable[str] | None = None,
        time_range: str | None = None,
        allow_site_superset: bool = False,
    ) -> DatasetRecord | None:
        required_sites = set(sites or ())
        matches = []
        for item in self._read():
            site_match = required_sites <= set(item.sites) if allow_site_superset else required_sites == set(item.sites)
            if (
                item.stage == stage
                and item.config_hash == config_hash
                and item.status == "ready"
                and site_match
                and (time_range is None or item.time_range == time_range)
            ):
                matches.append(item)
        if len(matches) > 1:
            raise ContractError(f"ambiguous catalog resolution for {stage}/{config_hash}")
        if not matches:
            return None
        self._validate_ready(matches[0])
        return matches[0]

    def raw_partition(self, source: str, site_id: str, month: str, *, suffix: str = ".parquet") -> Path:
        return self.paths.raw_partition(source, site_id, month, suffix=suffix)

    def processed_partition(self, stage: str, config_hash: str, site_id: str, month: str) -> Path:
        return self.paths.processed_partition(stage, config_hash, site_id, month)
