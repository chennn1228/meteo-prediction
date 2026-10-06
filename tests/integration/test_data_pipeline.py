from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from nwp.core.config import load_bundle, project_root, to_plain
from nwp.core.paths import RunPaths
from nwp.core.provenance import read_receipt
from nwp.data.clean import build_clean_month
from nwp.data.contracts import DataCatalog, expected_hourly_fields
from nwp.data.fetch import build_requests, execute_request
from nwp.features.build import build_feature_month


def test_three_source_month_builds_receipted_clean_partition(tmp_path):
    bundle = to_plain(load_bundle())
    data_config = bundle["data"]
    sites = bundle["sites"]["registry"]
    paths = RunPaths.create(
        project_root(),
        execution="development",
        run_id="data_integration",
        data_root=tmp_path / "data",
        outputs_root=tmp_path / "outputs",
    )
    catalog = DataCatalog(paths)
    first = dt.date(2024, 2, 1)
    last = dt.date(2024, 2, 29)
    requests = build_requests(
        data_config,
        sites,
        site_ids=("nanjing_1",),
        start=first,
        end=last,
    )
    hours = pd.date_range(first, last + dt.timedelta(days=1), freq="h", inclusive="left")
    time_values = [value.strftime("%Y-%m-%dT%H:%M") for value in hours]

    for request in requests:
        hourly = {"time": time_values}
        for field in expected_hourly_fields(request.source, data_config):
            if "cloud_cover" in field:
                hourly[field] = [120.0] * len(hours)
            elif "radiation" in field or "irradiance" in field:
                hourly[field] = [-1.0] * len(hours)
            else:
                hourly[field] = [1.0] * len(hours)
        payload = {
            "latitude": sites["nanjing_1"]["lat"] + 0.01,
            "longitude": sites["nanjing_1"]["lon"] + 0.01,
            "elevation": 10.0,
            "hourly": hourly,
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        execute_request(
            request,
            data_config=data_config,
            paths=paths,
            catalog=catalog,
            opener=lambda *_args, **_kwargs: Response(),
            sleeper=lambda _: None,
        )

    record = build_clean_month(
        site_id="nanjing_1",
        month="2024-02",
        site_registry=sites,
        data_config=data_config,
        paths=paths,
        catalog=catalog,
    )
    frame = pd.read_parquet(catalog.dataset_path(record))
    assert len(frame) == len(hours) * len(data_config["forecast"]["leads"])
    assert frame["cloud_cover_fcst"].max() == 100.0
    assert frame["ghi_fcst"].min() == 0.0
    assert set(frame["lead_time"]) == set(data_config["forecast"]["leads"])
    receipt = read_receipt(catalog.receipt_path(record))
    assert receipt["stage"] == "clean"
    assert set(receipt["input_hashes"]) == {"previous_runs", "satellite", "era5"}
    assert receipt["row_count"] == len(frame)
    assert build_clean_month(
        site_id="nanjing_1",
        month="2024-02",
        site_registry=sites,
        data_config=data_config,
        paths=paths,
        catalog=catalog,
    ) == record

    feature_record = build_feature_month(
        site_id="nanjing_1",
        month="2024-02",
        clean_config_hash=record.config_hash,
        data_config=data_config,
        feature_config=bundle["features"]["build"],
        paths=paths,
        catalog=catalog,
    )
    featured = pd.read_parquet(catalog.dataset_path(feature_record))
    assert len(featured) == len(frame)
    assert featured["location_id"].str.startswith("om-gfs-land-").all()
    assert {"kt_raw", "kt_model", "ghi_fcst_lag1", "cloud_cover_change"} <= set(featured)
    feature_receipt = read_receipt(catalog.receipt_path(feature_record))
    assert feature_receipt["stage"] == "features"
    assert feature_receipt["input_hashes"] == {"clean": receipt["output_hashes"]["data"]}
    assert feature_receipt["physical_coordinate_basis"] == "gfs_service_coordinates"
    assert build_feature_month(
        site_id="nanjing_1",
        month="2024-02",
        clean_config_hash=record.config_hash,
        data_config=data_config,
        feature_config=bundle["features"]["build"],
        paths=paths,
        catalog=catalog,
    ) == feature_record
