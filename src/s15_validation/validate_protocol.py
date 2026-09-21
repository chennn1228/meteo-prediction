"""Cross-module protocol consistency checks without launching experiments."""
from __future__ import annotations

from datetime import date, timedelta

from s01_core.config_loader import ProtocolError


def check_contract_alignment(manifest: dict, variables: dict) -> dict[str, bool]:
    """Compare live scientific interfaces with the only registered protocol."""
    from s07_prediction.schema import QUANTILES, QUANTILE_COLUMNS, REQUIRED_COLUMNS
    from s05_tuning.search_space import candidates

    registered_quantiles = tuple(float(value) for value in manifest["quantiles"])
    registered_columns = tuple(manifest["prediction_contract"]["quantile_columns"])
    forecast = variables["forecast_variables"]
    searchable = ("ridge_mos", "lgbm", "xgboost", "mlp", "autoformer", "timesnet")
    return {
        "forecast_variables_and_leads": len(forecast) == 15 and len(set(forecast)) == 15
        and variables["leads"] == [1, 2, 3]
        and manifest["data_sources"]["forecast"]["leads_hours"] == [24, 48, 72],
        "prediction_quantiles": QUANTILES == registered_quantiles
        and QUANTILE_COLUMNS == registered_columns
        and set(registered_columns).issubset(REQUIRED_COLUMNS),
        "six_distinct_search_candidates": all(
            len(options := candidates(model_id)) == 6
            and len({repr(sorted(option.items())) for option in options}) == 6
            for model_id in searchable
        ),
        "density_nested_quotas": manifest["spatial_design"]["training_site_density"] == [5, 10, 15, 20]
        and {int(size): int(quota) for size, quota in
             manifest["spatial_design"]["density_region_quotas"].items()}
        == {5: 1, 10: 2, 15: 3, 20: 4},
    }


def chronological_boundaries(manifest: dict) -> dict[str, bool | int]:
    """Check declared order and expose first-outer infeasibility numerically."""
    validation = manifest["validation_protocol"]
    outer = validation["outer_folds"]
    final = validation["final_fit"]
    as_date = date.fromisoformat
    dev_start = as_date(manifest["development_period"]["start"][:10])
    dev_end = as_date(manifest["development_period"]["end"][:10])
    test_start = as_date(manifest["test_period"]["start"][:10])
    gap = timedelta(hours=int(validation["purge_hours"]))
    ordered_outer = all(
        as_date(spec["train_start"]) == dev_start
        and as_date(spec["train_end"]) < as_date(spec["validation_start"])
        and as_date(spec["validation_end"]) <= dev_end
        and as_date(spec["validation_start"]) - as_date(spec["train_end"]) > gap
        for spec in outer
    ) and all(as_date(left["validation_end"]) < as_date(right["validation_start"])
              for left, right in zip(outer, outer[1:]))
    ordered_final = (
        as_date(final["fit_end"]) < as_date(final["early_stop_start"])
        <= as_date(final["early_stop_end"]) < as_date(final["calibration_start"])
        <= as_date(final["calibration_end"]) < test_start
    )
    # Consecutive scoring blocks need this many days before any nonempty fit.
    minimum_before_fit = (int(validation["inner_folds_per_outer"])
                          * int(validation["inner_scoring_days"])
                          + int(validation["inner_early_stop_days"])
                          + 2 * int(validation["purge_hours"]) // 24)
    first_prefix_days = (as_date(outer[0]["train_end"]) - dev_start).days + 1
    return {
        "outer_order": ordered_outer,
        "final_fit_early_calibration_test_order": ordered_final,
        "inner_fold_count_consistent": validation["inner_folds_per_outer"]
        == manifest["tuning_budget"]["common_inner_folds"],
        "first_outer_has_nonempty_fit": first_prefix_days > minimum_before_fit,
        "first_outer_prefix_days": first_prefix_days,
        "minimum_days_before_fit": minimum_before_fit,
    }


def require_official_chronology(manifest: dict) -> None:
    boundaries = chronological_boundaries(manifest)
    if not all(boundaries[key] for key in (
        "outer_order", "final_fit_early_calibration_test_order",
        "inner_fold_count_consistent", "first_outer_has_nonempty_fit",
    )):
        raise ProtocolError(f"official chronology is not feasible: {boundaries}")
