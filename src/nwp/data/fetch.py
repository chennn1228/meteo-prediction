"""Configuration-driven monthly acquisition planning and receipt registration."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from nwp.core.hashing import content_hash
from nwp.core.paths import RunPaths
from nwp.core.provenance import make_receipt
from nwp.core.schema import ContractError

from .client import FetchOutcome, fetch_json
from .contracts import DataCatalog, DatasetRecord, expected_hourly_fields, expected_month_slices, validate_raw_payload


SOURCE_ENDPOINTS = {
    "previous_runs": "https://previous-runs-api.open-meteo.com/v1/forecast",
    "satellite": "https://satellite-api.open-meteo.com/v1/archive",
}


@dataclass(frozen=True)
class FetchRequest:
    source: str
    site_id: str
    requested_latitude: float
    requested_longitude: float
    start: dt.date
    end: dt.date
    variables: tuple[str, ...]
    leads: tuple[int, ...]
    model: str | None

    def __post_init__(self) -> None:
        if self.source not in SOURCE_ENDPOINTS:
            raise ContractError(f"unknown fetch source: {self.source}")
        if not self.site_id or not self.variables:
            raise ContractError("fetch request requires a site and variables")
        if self.start > self.end or self.start.strftime("%Y-%m") != self.end.strftime("%Y-%m"):
            raise ContractError("one fetch request must cover one nonempty calendar-month slice")
        if self.source == "previous_runs" and not self.leads:
            raise ContractError("Previous Runs fetch requires registered leads")
        if self.source != "previous_runs" and self.leads:
            raise ContractError("truth-source fetch requests must not declare forecast leads")

    @property
    def month(self) -> str:
        return self.start.strftime("%Y-%m")


def raw_dependency_hash(request: FetchRequest, data_config: Mapping[str, Any]) -> str:
    return content_hash(
        {
            "data_version": data_config["version"],
            "source": request.source,
            "site_id": request.site_id,
            "requested_coordinates": [request.requested_latitude, request.requested_longitude],
            "start": request.start,
            "end": request.end,
            "variables": request.variables,
            "leads": request.leads,
            "model": request.model,
            "timezone": data_config["timezone"],
            "tilt": data_config["tilt"],
            "azimuth": data_config["azimuth"],
            "cell_selection": data_config["forecast"]["cell_selection"],
        }
    )


def build_requests(
    data_config: Mapping[str, Any],
    site_registry: Mapping[str, Mapping[str, Any]],
    *,
    site_ids: Iterable[str],
    start: dt.date,
    end: dt.date,
    sources: Iterable[str] = ("previous_runs", "satellite"),
) -> tuple[FetchRequest, ...]:
    if start > end:
        raise ContractError("fetch start exceeds end")
    requested_sites = tuple(site_ids)
    if not requested_sites or len(requested_sites) != len(set(requested_sites)):
        raise ContractError("fetch site IDs must be nonempty and unique")
    unknown = set(requested_sites) - set(site_registry)
    if unknown:
        raise ContractError(f"fetch site IDs are unregistered: {sorted(unknown)}")
    requested_sources = tuple(sources)
    if not requested_sources or len(requested_sources) != len(set(requested_sources)) or set(requested_sources) - set(SOURCE_ENDPOINTS):
        raise ContractError("fetch sources must be a nonempty unique registered selection")
    variables = {
        "previous_runs": tuple(data_config["forecast"]["variables"]),
        "satellite": tuple(data_config["truth"]["primary"]["variables"]),
    }
    models = {
        "previous_runs": data_config["forecast"]["model"],
        "satellite": data_config["truth"]["primary"]["model"],
    }
    output = []
    for source in requested_sources:
        for site_id in requested_sites:
            site = site_registry[site_id]
            for first, last in expected_month_slices(start, end):
                output.append(
                    FetchRequest(
                        source=source,
                        site_id=site_id,
                        requested_latitude=float(site["lat"]),
                        requested_longitude=float(site["lon"]),
                        start=first,
                        end=last,
                        variables=variables[source],
                        leads=tuple(int(value) for value in data_config["forecast"]["leads"]) if source == "previous_runs" else (),
                        model=models[source],
                    )
                )
    return tuple(output)


def request_parameters(request: FetchRequest, data_config: Mapping[str, Any]) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "latitude": request.requested_latitude,
        "longitude": request.requested_longitude,
        "start_date": request.start.isoformat(),
        "end_date": request.end.isoformat(),
        "timezone": data_config["timezone"],
        "cell_selection": data_config["forecast"]["cell_selection"],
        "hourly": ",".join(expected_hourly_fields(request.source, data_config)),
        "tilt": data_config["tilt"],
        "azimuth": data_config["azimuth"],
    }
    if request.model:
        parameters["models"] = request.model
    return parameters


def execute_request(
    request: FetchRequest,
    *,
    data_config: Mapping[str, Any],
    paths: RunPaths,
    catalog: DataCatalog,
    execution_level: str = "development",
    **client_options: Any,
) -> FetchOutcome:
    dependency_hash = raw_dependency_hash(request, data_config)
    parameters = request_parameters(request, data_config)
    data_path = paths.raw_partition(request.source, request.site_id, request.month, suffix=".json")
    receipt_path = paths.data_receipt(data_path)

    def validator(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return validate_raw_payload(
            payload,
            source=request.source,
            data_config=data_config,
            start=request.start,
            end=request.end,
        )

    def receipt_builder(
        payload: Mapping[str, Any],
        audit: Mapping[str, Any],
        digest: str,
    ) -> Mapping[str, Any]:
        returned = audit["returned_coordinates"]
        provider = (data_config["forecast"]["provider"] if request.source == "previous_runs"
                    else data_config["truth"]["primary"]["provider"])
        return make_receipt(
            root=paths.root,
            stage="raw",
            config_hash=dependency_hash,
            execution_level=execution_level,
            status="ready",
            output_hashes={"data": digest},
            source=request.source,
            provider=provider,
            model=request.model,
            requested_latitude=request.requested_latitude,
            requested_longitude=request.requested_longitude,
            returned_service_latitude=returned["latitude"],
            returned_service_longitude=returned["longitude"],
            returned_service_elevation=returned.get("elevation"),
            variables=list(request.variables),
            leads=list(request.leads),
            request_parameters=parameters,
            time_start=request.start.isoformat(), time_end=request.end.isoformat(),
            row_count=audit["row_count"],
            missingness=audit["missingness"],
            data_sha256=digest,
            data_version=data_config["version"],
        )

    outcome = fetch_json(
        SOURCE_ENDPOINTS[request.source],
        parameters,
        data_path=data_path,
        receipt_path=receipt_path,
        validator=validator,
        receipt_builder=receipt_builder,
        **client_options,
    )
    catalog.register(
        DatasetRecord(
            dataset_id=f"raw-{request.source}-{request.site_id}-{request.month}-{dependency_hash}",
            stage="raw",
            config_hash=dependency_hash,
            sites=(request.site_id,),
            time_range=f"{request.start.isoformat()}/{request.end.isoformat()}",
            source_hashes={},
            status="ready",
            path=data_path.relative_to(paths.data_root).as_posix(),
            receipt_path=receipt_path.relative_to(paths.data_root).as_posix(),
        )
    )
    return outcome
