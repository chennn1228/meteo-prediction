"""Thin explicit stage dispatcher; defaults to safe structural validation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from s14_pipeline.orchestrator import STAGES, run_stage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="validate", choices=[stage.name for stage in STAGES])
    parser.add_argument("--implementation-level", default="prototype",
                        choices=("prototype", "validated"))
    parser.add_argument("--execution-level", default="smoke",
                        choices=("smoke", "development", "official"))
    parser.add_argument("--validation-mode", default="structural",
                        choices=("structural", "data_ready", "cpu_ready", "deep_ready",
                                 "official_full"))
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--probe-step", type=float)
    parser.add_argument("--max-new-batches", type=int, default=0)
    args = parser.parse_args()
    report = run_stage(args.stage, implementation_level=args.implementation_level,
                       execution_level=args.execution_level,
                       validation_mode=args.validation_mode, input_path=args.input_path,
                       output_path=args.output_path, probe_step=args.probe_step,
                       max_new_batches=args.max_new_batches)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("status") in {"pass", "complete_empirical_round",
                                         "partial_not_frozen", "development_feature_build_not_official_result"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
