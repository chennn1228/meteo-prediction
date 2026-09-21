"""Thin CLI for layered evidence-based readiness validation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from s15_validation.validate_project import result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("structural", "data_ready", "cpu_ready", "spatial_ready",
                                           "deep_ready", "official_full", "official"),
                        default="structural")
    args = parser.parse_args()
    report = result(args.mode)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
