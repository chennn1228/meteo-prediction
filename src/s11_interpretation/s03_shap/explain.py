"""Keep SHAP values, sampled rows and metadata on one shared index."""
from __future__ import annotations

import numpy as np
import pandas as pd

from s01_core.config_loader import ProtocolError
from s01_core.schemas import assert_model_features
from s11_interpretation.development_gate import require_development_rows


def sample_aligned(features: pd.DataFrame, metadata: pd.DataFrame, *, n: int,
                   seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample one positional index once; never re-sample region labels later."""
    if len(features) != len(metadata) or n <= 0:
        raise ProtocolError("SHAP features/metadata length mismatch or invalid n")
    require_development_rows(metadata)
    assert_model_features(tuple(features.columns))
    positions = np.sort(np.random.default_rng(seed).choice(len(features),
                      size=min(n, len(features)), replace=False))
    return features.iloc[positions].copy(), metadata.iloc[positions].copy()


def native_shap_plot(explanation, kind: str, *, max_display: int = 15):
    """Use upstream SHAP plots rather than a handwritten lookalike.

    This function is never called during import or formal preflight. The caller
    must provide an already fitted, frozen explanation on development rows.
    """
    import shap

    if kind == "beeswarm":
        return shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    if kind == "bar":
        return shap.plots.bar(explanation, max_display=max_display, show=False)
    if kind == "waterfall":
        return shap.plots.waterfall(explanation, max_display=max_display, show=False)
    if kind == "heatmap":
        return shap.plots.heatmap(explanation, max_display=max_display, show=False)
    raise ProtocolError(f"unsupported native SHAP plot: {kind}")
