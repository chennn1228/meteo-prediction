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
    args = parser.parse_args()
    report = run_stage(args.stage, implementation_level=args.implementation_level,
                       execution_level=args.execution_level)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("status") == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
