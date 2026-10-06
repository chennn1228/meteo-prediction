"""Base-only executable smoke: real pipeline reaches real split construction."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import tempfile

import pandas as pd

from nwp.core.config import project_root, resolve_config
from nwp.core.context import RunContext
from nwp.core.fingerprints import sha256_file, stable_object_hash
from nwp.core.schema import StageResult
from nwp.workflow import pipeline
import nwp.workflow.stages.prepare as prepare


def _bounded_handler(stage: str):
    def handler(context: RunContext) -> StageResult:
        path = context.paths.stage_dir(stage) / f"{stage}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"bounded": True}) + "\n", encoding="utf-8")
        dependency = stable_object_hash({
            "stage": stage, "config": context.config.config_hash})
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        return StageResult(
            status="success", inputs={}, artifact_outputs={stage: str(path)},
            metadata_outputs={}, dependency_fingerprint=dependency,
            implementation_fingerprint="b" * 64,
            input_hashes={stage: dependency},
            output_hashes={stage: sha256_file(path)},
            started_at=now, finished_at=now)
    return handler


def main() -> None:
    with tempfile.TemporaryDirectory(
            prefix="nwp-base-splits-", dir=os.environ.get("TEMP")) as temporary:
        root = Path(temporary)
        context = RunContext.create(
            project_root(),
            resolve_config("nanjing_cpu_diagnostic", models="raw_gfs"),
            data_root=root / "data", outputs_root=root / "outputs",
            run_id="base_bounded_splits")
        frame = pd.DataFrame({
            "target_time_utc": pd.date_range(
                "2024-02-01", "2025-09-01", freq="6h",
                inclusive="left", tz="UTC")})
        prepare._load_feature_frame = lambda _context: frame
        pipeline.HANDLERS["data"] = _bounded_handler("data")
        pipeline.HANDLERS["features"] = _bounded_handler("features")
        statuses = pipeline.run_pipeline(context, to_stage="splits")
        actual = {key: value["status"] for key, value in statuses.items()}
        expected = {stage: "success" for stage in
                    ("validate", "selection", "data", "features", "splits")}
        if actual != expected:
            raise AssertionError(actual)
        print("BASE_BOUNDED_TO_SPLITS=PASS", actual)


if __name__ == "__main__":
    main()
