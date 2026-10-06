"""Generic factory-based model fitting."""
from __future__ import annotations

from typing import Any
from types import SimpleNamespace

from nwp.core.config import to_plain
from nwp.core.context import RunContext
from nwp.features.build import model_feature_config
from nwp.models.factory import model_factory, quantile_model_factory
from nwp.models.trees import final_tree_rounds, selected_tree_rounds


def fit_model(context: RunContext, model_id: str, features: Any, target: Any | None = None, *, params: dict[str, Any] | None = None):
    model = model_factory(
        model_id,
        to_plain(context.config.models),
        to_plain(context.config.features),
        to_plain(context.config.protocol),
        params=params,
    )
    return model.fit(features, target)


def fit_outer_quantile_model(
        context: RunContext, model_id: str, outer_fit: Any, inner_folds: Any,
        parameters: dict[str, Any], *, tree_receipts: list[dict[str, Any]] | None = None):
    levels = tuple(float(value) for value in
                   context.config.protocol["probability"]["quantiles"])
    seed = int(context.config.protocol["seed_policy"]["tuning_seed"])
    entry = context.config.models["registry"][model_id]
    feature_config = model_feature_config(
        to_plain(context.config.features), model_id)
    adapter = entry.get("quantile_adapter")
    if adapter == "ridge":
        return quantile_model_factory(
            model_id, to_plain(context.config.models),
            feature_config, to_plain(context.config.protocol),
            parameters=parameters, seed=seed).fit(
                {"outer_fit": outer_fit, "inner_folds": inner_folds})
    if adapter == "tree":
        if tree_receipts is None:
            raise ValueError("tree outer fitting requires selected inner round receipts")
        rounds = selected_tree_rounds(
            str(entry["implementation"]), parameters, seed, levels, tree_receipts,
            expected_inner_folds=int(
                context.config.protocol["validation"]["inner_folds"]))
        return quantile_model_factory(
            model_id, to_plain(context.config.models),
            feature_config, to_plain(context.config.protocol),
            parameters=parameters, rounds=rounds, seed=seed).fit(outer_fit)
    raise ValueError(f"no outer quantile fitter for {model_id}")


def fit_final_quantile_model(
        context: RunContext, model_id: str, fit: Any, early_stop: Any,
        parameters: dict[str, Any]):
    """Fit one final model using only fit and the declared early-stop block."""
    levels = tuple(float(value) for value in
                   context.config.protocol["probability"]["quantiles"])
    seed = int(context.config.protocol["seed_policy"]["tuning_seed"])
    entry = context.config.models["registry"][model_id]
    feature_config = model_feature_config(
        to_plain(context.config.features), model_id)
    adapter = entry.get("quantile_adapter")
    if adapter == "ridge":
        fold = SimpleNamespace(fit=fit, early_stop=early_stop)
        return quantile_model_factory(
            model_id, to_plain(context.config.models),
            feature_config, to_plain(context.config.protocol),
            parameters=parameters, seed=seed).fit(
                {"outer_fit": fit, "inner_folds": [fold]})
    if adapter == "tree":
        rounds = final_tree_rounds(
            str(entry["implementation"]), parameters, fit, early_stop, seed, levels,
            feature_config=feature_config)
        return quantile_model_factory(
            model_id, to_plain(context.config.models),
            feature_config, to_plain(context.config.protocol),
            parameters=parameters, rounds=rounds, seed=seed).fit(fit)
    raise ValueError(f"no final quantile fitter for {model_id}")
