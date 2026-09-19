"""Thin, budgeted service-point discovery CLI; never freezes partial rounds."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from s12_spatial.probe import probe_round  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=float, required=True)
    parser.add_argument("--max-new-batches", type=int, default=0,
                        help="0 means dry run; 40 locations per new batch")
    args = parser.parse_args()
    print(json.dumps(probe_round(args.step, max_new_batches=args.max_new_batches),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
