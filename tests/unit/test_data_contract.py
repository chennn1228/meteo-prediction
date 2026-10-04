from __future__ import annotations

import datetime as dt
import json
import inspect

import pytest

from nwp.core.config import load_bundle, project_root, to_plain
from nwp.core.hashing import file_sha256
from nwp.core.paths import RunPaths
from nwp.core.provenance import make_receipt, write_receipt
from nwp.core.schema import ContractError
from nwp.data.audit import inventory_data
from nwp.data.contracts import DataCatalog, DatasetRecord, expected_hourly_fields
from nwp.data.fetch import FetchRequest, build_requests, execute_request


def test_catalog_resolves_by_stage_hash_and_sites(tmp_path):
    paths = RunPaths.create(
        project_root(),
        execution="development",
        run_id="catalog-test",
        data_root=tmp_path / "data",
        outputs_root=tmp_path / "outputs",
    )
    catalog = DataCatalog(paths)
    data_path = paths.raw_partition("gfs", "nanjing_1", "2024-02")
    data_path.parent.mkdir(parents=True)
    data_path.write_text("raw\n", encoding="utf-8")
    receipt_path = paths.data_receipt(data_path)
    receipt = make_receipt(
        root=project_root(),
        stage="raw",
        config_hash="abcdef12",
        execution_level="development",
        status="ready",
        output_hashes={"data": file_sha256(data_path)},
    )
    write_receipt(receipt_path, receipt)
    record = DatasetRecord(
        "raw_a",
        "raw",
        "abcdef12",
        ("nanjing_1",),
        "2024-02",
        {},
        "ready",
        data_path.relative_to(paths.data_root).as_posix(),
        receipt_path.relative_to(paths.data_root).as_posix(),
    )
    catalog.register(record)
    catalog.register(record)  # exact idempotent registration is allowed
    assert catalog.resolve(stage="raw", config_hash="abcdef12", sites=("nanjing_1",)) == record
    assert catalog.resolve(stage="raw", config_hash="abcdef12", sites=("wuxi_1",)) is None
    with pytest.raises(ContractError, match="under data_root"):
        DatasetRecord("bad", "raw", "abcdef12", ("nanjing_1",), "2024-02", {}, "ready", "../escape", "raw/receipt.json")


def test_catalog_bulk_registration_is_atomic_and_idempotent(tmp_path):
    paths = RunPaths.create(
        project_root(), execution="development", run_id="catalog-bulk-test",
        data_root=tmp_path / "data", outputs_root=tmp_path / "outputs")
    catalog = DataCatalog(paths)
    records = []
    for month in ("2024-02", "2024-03"):
        data_path = paths.raw_partition("gfs", "nanjing_1", month)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(month, encoding="utf-8")
        receipt_path = paths.data_receipt(data_path)
        write_receipt(receipt_path, make_receipt(
            root=project_root(), stage="raw", config_hash="abcdef12",
            execution_level="development", status="ready",
            output_hashes={"data": file_sha256(data_path)}))
        records.append(DatasetRecord(
            f"raw-{month}", "raw", "abcdef12", ("nanjing_1",), month,
            {}, "ready", data_path.relative_to(paths.data_root).as_posix(),
            receipt_path.relative_to(paths.data_root).as_posix()))
    catalog.register_many(records)
    catalog.register_many(records)
    assert catalog.records() == tuple(records)
    with pytest.raises(ContractError, match="duplicate dataset IDs"):
        catalog.register_many((records[0], records[0]))


def test_inventory_hashes_and_classifies_without_deletion(tmp_path):
    root = tmp_path / "data"
    raw = root / "raw" / "gfs" / "nanjing_1" / "2024-02.csv"
    raw.parent.mkdir(parents=True)
    raw.write_text("x\n1\n", encoding="utf-8")
    counts = inventory_data(root)
    assert counts["KEEP"] == 1
    assert raw.exists()
    assert (root / "data_inventory.csv").exists()
    assert "quarantine_unknown" not in inspect.signature(inventory_data).parameters


def test_inventory_marks_numbered_layout_for_move_and_registry_for_keep(tmp_path):
    root = tmp_path / "data"
    old = root / "01_raw" / "gfs" / "nanjing_1" / "2024-02.csv"
    registry = root / "registry" / "sites.json"
    old.parent.mkdir(parents=True)
    registry.parent.mkdir(parents=True)
    old.write_text("x\n", encoding="utf-8")
    registry.write_text("{}\n", encoding="utf-8")
    counts = inventory_data(root)
    assert counts == {"MOVE": 1, "KEEP": 1}


def test_fetch_planning_is_site_month_partitioned_and_config_driven():
    bundle = to_plain(load_bundle())
    requests = build_requests(
        bundle["data"],
        bundle["sites"]["registry"],
        site_ids=("nanjing_1",),
        start=dt.date(2024, 2, 1),
        end=dt.date(2024, 3, 31),
    )
    assert len(requests) == 6
    assert {request.source for request in requests} == {"previous_runs", "satellite", "era5"}
    assert {request.month for request in requests} == {"2024-02", "2024-03"}
    forecast = next(request for request in requests if request.source == "previous_runs")
    assert forecast.variables == tuple(bundle["data"]["forecast"]["variables"])
    assert forecast.leads == tuple(bundle["data"]["forecast"]["leads"])


def test_fetch_writes_and_reuses_immutable_raw_receipt_pair(tmp_path):
    bundle = to_plain(load_bundle())
    data_config = bundle["data"]
    site = bundle["sites"]["registry"]["nanjing_1"]
    request = FetchRequest(
        source="previous_runs",
        site_id="nanjing_1",
        requested_latitude=site["lat"],
        requested_longitude=site["lon"],
        start=dt.date(2024, 2, 1),
        end=dt.date(2024, 2, 1),
        variables=tuple(data_config["forecast"]["variables"]),
        leads=tuple(data_config["forecast"]["leads"]),
        model=data_config["forecast"]["model"],
    )
    times = [f"2024-02-01T{hour:02d}:00" for hour in range(24)]
    hourly = {"time": times}
    for field in expected_hourly_fields("previous_runs", data_config):
        hourly[field] = [1.0] * 24
    payload = {
        "latitude": site["lat"] + 0.01,
        "longitude": site["lon"] + 0.01,
        "elevation": 12.0,
        "hourly": hourly,
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    calls = []

    def opener(*args, **kwargs):
        calls.append((args, kwargs))
        return Response()

    paths = RunPaths.create(
        project_root(),
        execution="development",
        run_id="fetch-test",
        data_root=tmp_path / "data",
        outputs_root=tmp_path / "outputs",
    )
    catalog = DataCatalog(paths)
    first = execute_request(
        request,
        data_config=data_config,
        paths=paths,
        catalog=catalog,
        opener=opener,
        sleeper=lambda _: None,
    )
    assert not first.reused
    assert len(calls) == 1
    receipt = json.loads(first.receipt_path.read_text(encoding="utf-8"))
    assert receipt["requested_latitude"] == site["lat"]
    assert receipt["requested_longitude"] == site["lon"]
    assert receipt["returned_service_latitude"] == payload["latitude"]
    assert receipt["returned_service_longitude"] == payload["longitude"]
    assert receipt["returned_service_latitude"] == payload["latitude"]
    assert receipt["row_count"] == 24
    assert receipt["data_sha256"] == first.sha256

    second = execute_request(
        request,
        data_config=data_config,
        paths=paths,
        catalog=catalog,
        opener=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("cache reuse made a network call")),
    )
    assert second.reused
