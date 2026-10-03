"""Feature interpretation must not sample the final-test year."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").is_file())
sys.path.insert(0, str(ROOT / "src"))

from nwp.core.config import load_bundle, to_plain  # noqa: E402
from nwp.core.schema import ContractError  # noqa: E402
from nwp.evaluation.interpretation import require_development_rows  # noqa: E402

PROTOCOL = to_plain(load_bundle())["protocol"]


def test_development_interpretation_accepts_outer_score_but_rejects_test():
    require_development_rows(pd.DataFrame({
        "target_time_utc": ["2024-06-11T00:00:00Z", "2025-07-31T23:00:00Z"]}),
        PROTOCOL)
    with pytest.raises(ContractError, match="development period only"):
        require_development_rows(pd.DataFrame({
            "target_time_utc": ["2025-07-31T23:00:00Z", "2025-09-01T00:00:00Z"]}),
            PROTOCOL)
