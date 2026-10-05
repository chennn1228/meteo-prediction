from __future__ import annotations

from dataclasses import replace

from nwp.core.config import resolve_config, to_plain
from nwp.core.dependencies import (
    data_dependency_fingerprint, feature_dependency_fingerprint,
    model_dependency_fingerprint, split_dependency_fingerprint)
from nwp.core.fingerprints import stable_object_hash


MODELS = ("ridge_mos", "lgbm", "xgboost")


def _snapshot(config, *, figure_style="default"):
    data = data_dependency_fingerprint(
        config.data, [{"catalog-record": "a" * 64}])
    features = feature_dependency_fingerprint(
        data, config.features, {"feature-partition": "b" * 64})
    splits = split_dependency_fingerprint(
        features, config.selected_sites, config.protocol["validation"])
    result = {"Data": data, "Features": features, "Splits": splits}
    model_outputs = {}
    for model_id in MODELS:
        tuning = model_dependency_fingerprint(
            "tuning", model_id,
            upstream={"features": features, "splits": splits,
                      "fold": stable_object_hash("outer_1")},
            model_record=config.models["registry"][model_id],
            search_space=config.selected_search_spaces[model_id],
            stage_contract={
                "gap_days": config.protocol["validation"]["gap_days"],
                "metric": config.protocol["probability"]["selection_metric"]})
        fitting = model_dependency_fingerprint(
            "fitting", model_id,
            upstream={"features": features, "splits": splits,
                      "tuning": tuning, "fold": stable_object_hash("outer_1")},
            model_record=config.models["registry"][model_id],
            stage_contract={"selected_from": tuning})
        prediction = model_dependency_fingerprint(
            "prediction", model_id,
            upstream={"model_dependency": fitting, "model": "c" * 64},
            model_record=config.models["registry"][model_id],
            stage_contract={"scope": "outer_validation"})
        metrics = model_dependency_fingerprint(
            "metrics", model_id, upstream={"predictions": prediction},
            model_record=config.models["registry"][model_id],
            stage_contract=config.protocol["evaluation"])
        result[{"ridge_mos": "Ridge", "lgbm": "LGBM",
                "xgboost": "XGB"}[model_id]] = fitting
        model_outputs[model_id] = metrics
    result["analysis"] = stable_object_hash({
        "model_metrics": model_outputs, "analysis": config.analysis})
    result["figures"] = stable_object_hash({
        "analysis": result["analysis"], "style": figure_style})
    result["report"] = stable_object_hash({"figures": result["figures"]})
    return result


def _changed(base, *, xgb=False, gap=False, features=False, analysis=False):
    models = to_plain(base.models)
    protocol = to_plain(base.protocol)
    feature_config = to_plain(base.features)
    analysis_config = to_plain(base.analysis)
    selected_spaces = to_plain(base.selected_search_spaces)
    if xgb:
        selected_spaces["xgboost"][0]["max_depth"] += 1
    if gap:
        protocol["validation"]["gap_days"] += 1
    if features:
        feature_config["derived_features"] = tuple(
            feature_config["derived_features"]) + ("bounded_test",)
    if analysis:
        analysis_config["evidence"]["permutation_repeats"] += 1
    return replace(
        base, models=models, protocol=protocol, features=feature_config,
        analysis=analysis_config, selected_search_spaces=selected_spaces,
        config_hash=stable_object_hash({
            "models": models, "protocol": protocol, "features": feature_config,
            "analysis": analysis_config, "spaces": selected_spaces}))


def test_production_dependency_truth_table():
    base = resolve_config("nanjing_cpu_diagnostic", models=MODELS)
    original = _snapshot(base)
    variants = {
        "XGB search": _snapshot(_changed(base, xgb=True)),
        "Gap": _snapshot(_changed(base, gap=True)),
        "Feature build": _snapshot(_changed(base, features=True)),
        "Analysis-only": _snapshot(_changed(base, analysis=True)),
        "Figure style": _snapshot(base, figure_style="alternate"),
    }
    expected_changed = {
        "XGB search": {"XGB", "analysis", "figures", "report"},
        "Gap": {"Splits", "Ridge", "LGBM", "XGB", "analysis", "figures", "report"},
        "Feature build": {"Features", "Splits", "Ridge", "LGBM", "XGB",
                          "analysis", "figures", "report"},
        "Analysis-only": {"analysis", "figures", "report"},
        "Figure style": {"figures", "report"},
    }
    for change, snapshot in variants.items():
        changed = {key for key in original if original[key] != snapshot[key]}
        assert changed == expected_changed[change]
