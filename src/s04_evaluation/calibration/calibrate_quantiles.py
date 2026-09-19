"""Legacy import compatibility only; the old calibration CLI is disabled.

The old server scripts used this file to recalibrate during the test period.
That workflow is scientifically invalid under the current time-ordered
protocol. Formal runs must use ``s08_calibration.AdditiveQuantileCalibrator``
with explicit fit/early-stop/calibration/test boundaries.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s08_calibration.quantile import additive_residual_offsets, apply_additive_offsets
from s09_metrics.probabilistic import probability_metrics

TAUS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
OLD_COLUMNS = [f"q{tau:g}" for tau in TAUS]


def calibrate(val: pd.DataFrame, test: pd.DataFrame,
              clip: tuple[float, float] | None = None):
    """Diagnostic-only legacy array adapter; no file output or official status.

    This has no temporal split inputs and must never be used as a formal run.
    Existing unit tests can verify the sign and rearrangement through it.
    """
    missing = [column for column in ["y", *OLD_COLUMNS] if column not in val]
    missing += [column for column in OLD_COLUMNS if column not in test]
    if missing:
        raise ValueError(f"missing legacy quantile columns: {sorted(set(missing))}")
    offsets = additive_residual_offsets(val.y.to_numpy(), val[OLD_COLUMNS].to_numpy(), TAUS)
    output = test.copy()
    output.loc[:, OLD_COLUMNS] = apply_additive_offsets(
        test[OLD_COLUMNS].to_numpy(), offsets,
        lower_bound=None if clip is None else clip[0],
        upper_bound=None if clip is None else clip[1],
    )
    return output, dict(zip(OLD_COLUMNS, offsets.tolist()))


def metrics(y, qmat):
    """Compatibility alias to the single formal seven-quantile implementation."""
    return probability_metrics(y, qmat)


def main() -> None:
    raise SystemExit(
        "BLOCKED: legacy calibration CLI can reuse test-period observations. "
        "Use s08_calibration.AdditiveQuantileCalibrator with explicit time boundaries."
    )


if __name__ == "__main__":
    main()
