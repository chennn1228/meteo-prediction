"""A provider quota error must stop new requests without losing verified cache."""

import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from shapely.geometry import box

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from nwp.data.audit import (  # noqa: E402
    ServiceRateLimitError, fetch_service_probe_batch, probe_service_round,
)


class ServiceProbeRateLimitTests(unittest.TestCase):
    def test_http_429_is_not_retried(self):
        error = urllib.error.HTTPError("https://api.open-meteo.com", 429,
                                       "Too Many Requests", {}, None)
        with patch("urllib.request.urlopen", side_effect=error) as opener:
            with self.assertRaises(ServiceRateLimitError):
                fetch_service_probe_batch([(119.0, 32.0)], model="gfs_seamless")
        opener.assert_called_once()

    def test_partial_round_preserves_cache_and_stops_new_requests(self):
        point = {"requested_longitude": 119.0, "requested_latitude": 32.0,
                 "service_longitude": 119.0, "service_latitude": 32.0,
                 "service_elevation_m": 10.0, "service_id": "point-1"}
        with tempfile.TemporaryDirectory() as directory:
            with patch("nwp.data.audit.service_boundary",
                       return_value=box(118, 31, 121, 34)), patch(
                "nwp.data.audit.service_request_lattice",
                return_value=[(119.0, 32.0), (119.1, 32.0), (119.2, 32.0)]
            ), patch("nwp.data.audit.fetch_service_probe_batch",
                     side_effect=[[point], ServiceRateLimitError("HTTP 429")]) as fetch:
                result = probe_service_round(
                    .025, batch_size=1, max_new_batches=3,
                    data_root=Path(directory),
                    boundary_path=Path(directory) / "unused.geojson",
                    model="gfs_seamless")
            self.assertEqual(fetch.call_count, 2)
            self.assertEqual(result["status"], "partial_not_frozen")
            self.assertEqual(result["stopped_reason"], "provider_http_429")
            self.assertEqual(result["completed_batches"], 1)
            self.assertFalse(result["frozen"])
            cached = Path(directory) / "04_service_probes" / "step_0.025" / "batch_00000.json"
            self.assertEqual(json.loads(cached.read_text(encoding="utf-8"))["returns"], [point])
            self.assertFalse(cached.with_name("batch_00001.json").exists())


if __name__ == "__main__":
    unittest.main()
