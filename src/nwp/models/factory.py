"""The only model constructor, driven by models.yaml registry entries."""
from __future__ import annotations

from typing import Any, Mapping

from nwp.features.build import model_feature_config

from .base import BaseModel, ModelError
from .baselines import BASELINE_IMPLEMENTATIONS, UnavailableBaseline
from .deep import DeepModel
from .statistical import StatisticalModel
from .trees import TreeModel


def quantile_model_factory(model_id: str, model_config: Mapping[str, Any],
                           feature_config: Mapping[str, Any],
                           protocol_config: Mapping[str, Any], *,
                           parameters: Mapping[str, Any], rounds: Any = None,
                           seed: int = 0) -> BaseModel:
    """Construct a registered quantile adapter without stage-local estimators."""
    from .statistical import RidgeQuantileModel
    from .trees import TreeQuantileModel
    entry = model_config["registry"][model_id]
    feature_config = model_feature_config(feature_config, model_id)
    levels = tuple(float(value) for value in protocol_config["probability"]["quantiles"])
    adapter = entry.get("quantile_adapter")
    if adapter == "ridge":
        return RidgeQuantileModel(
            model_id, feature_config, float(parameters["alpha"]), levels, seed=seed)
    if adapter == "tree":
        if rounds is None:
            raise ModelError("tree quantile model requires selected rounds")
        return TreeQuantileModel(
            model_id, str(entry["implementation"]), dict(parameters), feature_config,
            levels, rounds, seed=seed)
    raise ModelError(f"no registered quantile adapter for {model_id}")


def model_factory(
    model_id: str,
    model_config: Mapping[str, Any],
    feature_config: Mapping[str, Any],
    protocol_config: Mapping[str, Any],
    *,
    params: Mapping[str, Any] | None = None,
) -> BaseModel:
    registry = model_config["registry"]
    if model_id not in registry:
        raise ModelError(f"unregistered model: {model_id}")
    entry, implementation = registry[model_id], registry[model_id]["implementation"]
    feature_config = model_feature_config(feature_config, model_id)
    if implementation in BASELINE_IMPLEMENTATIONS:
        return BASELINE_IMPLEMENTATIONS[implementation](
            model_id, model_config, feature_config)
    if entry["family"] == "statistical":
        return StatisticalModel(model_id, implementation, feature_config, params=params)
    if entry["family"] == "tree_ml":
        return TreeModel(model_id, implementation, feature_config, params=params)
    if entry["family"] == "deep":
        return DeepModel(
            model_id, implementation, model_config, protocol_config, params=params)
    if entry["family"] == "experimental_constrained":
        return DeepModel(
            model_id, implementation, model_config, protocol_config, params=params)
    return UnavailableBaseline(model_id)
