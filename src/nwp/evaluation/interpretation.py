"""Downstream-only analysis tables and aligned interpretation sampling."""
from __future__ import annotations

from collections.abc import Callable, Mapping
import json

import numpy as np
import pandas as pd

from nwp.core.schema import ContractError, assert_model_features
from nwp.evaluation.metrics import probability_metrics
from nwp.features.preprocessing import fit_fold_preprocessing
from nwp.splits.rolling import inner_folds, outer_folds


def assert_interpretation_only(stage: str) -> None:
    if stage in {"tuning", "fitting"}:
        raise ValueError("interpretation must not affect model selection or fitting")


def require_development_rows(
        frame: pd.DataFrame, protocol_config: Mapping[str, object]) -> None:
    """Reject final-test or malformed rows from feature-mechanism analysis."""
    if "target_time_utc" not in frame:
        raise ContractError("feature analysis requires target_time_utc")
    target = pd.to_datetime(frame["target_time_utc"], utc=True, errors="coerce")
    period = protocol_config["development_period"]
    start, end = pd.Timestamp(period["start"]), pd.Timestamp(period["end"])
    if (frame.empty or target.isna().any() or (target < start).any()
            or (target > end).any()):
        raise ContractError(
            "feature mechanism analysis must use development period only")


def sample_aligned(
        features: pd.DataFrame, metadata: pd.DataFrame, *, n: int,
        protocol_config: Mapping[str, object],
        feature_policy: Mapping[str, object], seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample one positional index once for features and their metadata."""
    if len(features) != len(metadata) or n <= 0:
        raise ContractError("SHAP features/metadata length mismatch or invalid n")
    require_development_rows(metadata, protocol_config)
    assert_model_features(tuple(features.columns), feature_policy)
    positions = np.sort(np.random.default_rng(seed).choice(
        len(features), size=min(n, len(features)), replace=False))
    return features.iloc[positions].copy(), metadata.iloc[positions].copy()


def native_shap_plot(explanation: object, kind: str, *, max_display: int = 15):
    """Render an upstream SHAP plot for an already frozen development model."""
    import shap

    functions = {
        "beeswarm": shap.plots.beeswarm,
        "bar": shap.plots.bar,
        "waterfall": shap.plots.waterfall,
        "heatmap": shap.plots.heatmap,
    }
    if kind not in functions:
        raise ContractError(f"unsupported native SHAP plot: {kind}")
    return functions[kind](explanation, max_display=max_display, show=False)


def model_summary(probability: pd.DataFrame,
                  point: pd.DataFrame) -> pd.DataFrame:
    """Create a compact source table without changing any model ranking rule."""
    assert_interpretation_only("analysis")
    point_columns = [
        name for name in (
            "model_id", "prediction_type", "n", "mae", "rmse", "bias",
            "r2", "rmse_skill", "rmse_skill_reference", "rmse_skill_status")
        if name in point.columns]
    output = point.loc[:, point_columns].copy()
    if output.empty:
        raise ValueError("point metric table is empty")
    output["point_rmse_rank"] = output["rmse"].rank(
        method="min", ascending=True).astype(int)
    if not probability.empty:
        probabilistic = probability.loc[:, [
            name for name in (
                "model_id", "mean_pinball", "crps_q7_trunc",
                "coverage_90", "interval_width_90", "crossing_rate")
            if name in probability.columns]].copy()
        probabilistic["probability_rank"] = probabilistic["mean_pinball"].rank(
            method="min", ascending=True).astype(int)
        output = output.merge(probabilistic, on="model_id", how="left",
                              validate="one_to_one")
    return output.sort_values(
        ["point_rmse_rank", "model_id"], kind="stable").reset_index(drop=True)


def _fit_quantile_estimators(
        implementation: str, parameters: Mapping[str, object], x_fit: pd.DataFrame,
        x_early: pd.DataFrame, y_fit: np.ndarray, y_early: np.ndarray,
        levels: tuple[float, ...], seed: int) -> list[object]:
    estimators = []
    for level in levels:
        if implementation == "lightgbm":
            import lightgbm as lgb
            estimator = lgb.LGBMRegressor(
                objective="quantile", alpha=level, metric="quantile",
                n_estimators=3000, random_state=seed, verbose=-1,
                **dict(parameters))
            estimator.fit(
                x_fit, y_fit, eval_set=[(x_early, y_early)],
                callbacks=[lgb.early_stopping(80, verbose=False)])
        elif implementation == "xgboost":
            import xgboost as xgb
            estimator = xgb.XGBRegressor(
                objective="reg:quantileerror", quantile_alpha=level,
                n_estimators=3000, tree_method="hist",
                early_stopping_rounds=80, random_state=seed,
                **dict(parameters))
            estimator.fit(
                x_fit, y_fit, eval_set=[(x_early, y_early)], verbose=False)
        else:
            raise ValueError(
                "group mechanism evidence requires a registered tree implementation")
        estimators.append(estimator)
    return estimators


def _tree_predictions(estimators: list[object], matrix: pd.DataFrame) -> np.ndarray:
    values = []
    for estimator in estimators:
        if estimator.__class__.__module__.startswith("lightgbm"):
            values.append(estimator.predict(
                matrix, num_iteration=estimator.best_iteration_))
        else:
            values.append(estimator.predict(
                matrix, iteration_range=(0, estimator.best_iteration + 1)))
    return np.column_stack(values)


def grouped_permutation_predictions(
        estimators: list[object], x_score: pd.DataFrame,
        score_metadata: pd.DataFrame, group_columns: list[str], *,
        lead_values: tuple[int, ...], seed: int) -> np.ndarray:
    """Permute one feature group within lead, preserving all row alignment."""
    if any(column not in x_score for column in group_columns):
        raise ValueError("permutation group contains an absent feature")
    random = np.random.default_rng(seed)
    permuted = x_score.copy()
    positions = [permuted.columns.get_loc(column) for column in group_columns]
    for lead in lead_values:
        rows = np.flatnonzero(score_metadata.lead_time.to_numpy() == lead)
        source = random.permutation(rows)
        permuted.iloc[rows, positions] = x_score.iloc[
            source, positions].to_numpy()
    return _tree_predictions(estimators, permuted)


def group_mechanism_evidence(
        frame: pd.DataFrame, selection: Mapping[str, object], *,
        protocol_config: Mapping[str, object],
        feature_config: Mapping[str, object],
        analysis_config: Mapping[str, object],
        model_config: Mapping[str, object],
        eligible: Callable[[pd.DataFrame], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run downstream-only group ablation and grouped permutation evidence."""
    assert_interpretation_only("analysis")
    settings = analysis_config["evidence"]
    model_id = str(settings["reference_model"])
    repeats = int(settings["permutation_repeats"])
    if repeats < 1:
        raise ValueError("permutation repeats must be positive")
    entry = model_config["registry"].get(model_id)
    if not isinstance(entry, Mapping) or entry.get("quantile_adapter") != "tree":
        raise ValueError("reference_model must be a supported tree model")
    implementation = str(entry["implementation"])
    levels = tuple(float(value) for value in
                   protocol_config["probability"]["quantiles"])
    leads = tuple(sorted(int(value) for value in frame.lead_time.unique()))
    validation = protocol_config["validation"]
    development = protocol_config["development_period"]
    gap = int(validation["gap_days"])
    groups = {str(group): list(columns) for group, columns in
              feature_config["feature_groups"].items()}
    seed = int(protocol_config["seed_policy"]["tuning_seed"])
    ablation_rows, permutation_rows = [], []
    outer_selection = selection["outer"]
    for outer_index, outer in enumerate(
            outer_folds(frame, validation, development, gap_days=gap), start=1):
        choice = outer_selection[outer.fold_id][model_id]
        parameters = choice["parameters"]
        trial_by_inner = {
            row["inner"]: row for row in choice["trials"]
            if int(row["candidate"]) == int(choice["selected_candidate"])}
        for inner_index, fold in enumerate(
                inner_folds(outer, validation, development, gap_days=gap), start=1):
            fit, early, score = (
                eligible(block) for block in
                (fold.fit, fold.early_stop, fold.score))
            if any(block.cloud_cover_fcst.isna().any()
                   for block in (fit, early, score)):
                raise ValueError(
                    "group ablation requires a group-specific imputer when cloud is missing")
            (x_fit, x_early, x_score), _ = fit_fold_preprocessing(
                fit, early, score, feature_config=feature_config, seed=seed)
            y_fit, y_early, y_score = (
                block.y.to_numpy(dtype=float) for block in (fit, early, score))
            estimators = _fit_quantile_estimators(
                implementation, parameters, x_fit, x_early, y_fit, y_early,
                levels, seed)
            baseline_q = _tree_predictions(estimators, x_score)
            baseline = float(probability_metrics(
                y_score, baseline_q, levels=levels)["mean_pinball"])
            original = trial_by_inner[fold.fold_id]
            if not np.isclose(
                    baseline, original["mean_pinball"], rtol=1e-6, atol=1e-6):
                raise ValueError(
                    f"{outer.fold_id}/{fold.fold_id}: tuning baseline did not reproduce")
            identity = {"outer_fold": outer.fold_id,
                        "inner_fold": fold.fold_id, "model_id": model_id,
                        "baseline_mean_pinball": baseline}
            for group_index, (group, columns) in enumerate(groups.items()):
                if any(column not in x_fit for column in columns):
                    raise ValueError(
                        f"{group}: declared feature absent from frozen matrix")
                keep = [column for column in x_fit if column not in columns]
                ablated = _fit_quantile_estimators(
                    implementation, parameters, x_fit[keep], x_early[keep],
                    y_fit, y_early, levels, seed)
                loss = float(probability_metrics(
                    y_score, _tree_predictions(ablated, x_score[keep]),
                    levels=levels)["mean_pinball"])
                ablation_rows.append({
                    **identity, "group": group,
                    "dropped_features": json.dumps(columns),
                    "mean_pinball": loss, "delta_vs_full": loss - baseline})
                for repeat in range(repeats):
                    values = grouped_permutation_predictions(
                        estimators, x_score, score, columns, lead_values=leads,
                        seed=(10000 * outer_index + 1000 * inner_index
                              + 10 * group_index + repeat))
                    loss = float(probability_metrics(
                        y_score, values, levels=levels)["mean_pinball"])
                    permutation_rows.append({
                        **identity, "group": group, "repeat": repeat,
                        "mean_pinball": loss,
                        "delta_vs_full": loss - baseline})
    return pd.DataFrame(ablation_rows), pd.DataFrame(permutation_rows)
