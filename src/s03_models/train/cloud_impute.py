"""Compatibility import; formal cloud imputation lives in s03_features."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from s03_features.preprocessing import FoldCloudImputer  # noqa: E402,F401


def impute_cloud_forecast(*args, **kwargs):
    raise RuntimeError("whole-dataset cloud imputation is retired; fit FoldCloudImputer on each fit fold")
