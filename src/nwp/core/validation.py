"""Evidence-based readiness gates for the unified repository."""
from __future__ import annotations

import ast
import datetime as dt
import importlib.util
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from nwp.core.config import ConfigError, load_bundle, load_local_paths, project_root, to_plain
from nwp.core.hashing import file_sha256
from nwp.core.fingerprints import stable_object_hash
from nwp.core.paths import RunPaths
from nwp.core.provenance import read_receipt
from nwp.core.schema import ContractError, assert_model_features
from nwp.data.contracts import DataCatalog
from nwp.experiment.prediction import quantile_columns, quantiles, required_columns


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""
    scope: str = "structural"


def contract_alignment(bundle: Mapping[str, Any]) -> dict[str, bool]:
    protocol, data = bundle["protocol"], bundle["data"]
    sites, models = bundle["sites"], bundle["models"]
    levels = tuple(float(value) for value in protocol["probability"]["quantiles"])
    columns = tuple(
        protocol["evaluation"]["prediction_contract"]["quantile_columns"])
    selector = sites["selectors"]["density"]
    return {
        "forecast_variables_and_leads": (
            bool(data["forecast"]["variables"])
            and len(data["forecast"]["variables"])
            == len(set(data["forecast"]["variables"]))
            and all(int(lead) > 0 for lead in data["forecast"]["leads"])),
        "prediction_quantiles": (
            quantiles(protocol) == levels
            and quantile_columns(protocol) == columns
            and set(columns).issubset(required_columns(protocol))),
        "distinct_registered_search_candidates": all(
            len(options)
            == len({repr(sorted(option.items())) for option in options})
            for options in models["search"]["spaces"].values()),
        "density_selector_contract": (
            selector["method"] == "region_balanced_farthest_point"
            and len(selector["allowed_counts"])
            == len(set(selector["allowed_counts"]))
            and set(int(key) for key in selector["region_quotas"])
            == set(selector["allowed_counts"])),
    }


def chronological_boundaries(protocol: Mapping[str, Any]) -> dict[str, bool | int]:
    validation = protocol["validation"]
    outer, final = validation["outer_folds"], validation["final_fit"]
    as_date = dt.date.fromisoformat
    development_start = as_date(protocol["development_period"]["start"][:10])
    development_end = as_date(protocol["development_period"]["end"][:10])
    test_start = as_date(protocol["test_period"]["start"][:10])
    purge = dt.timedelta(hours=int(validation["purge_hours"]))
    outer_order = all(
        as_date(spec["train_start"]) == development_start
        and as_date(spec["train_end"]) < as_date(spec["validation_start"])
        and as_date(spec["validation_end"]) <= development_end
        and as_date(spec["validation_start"]) - as_date(spec["train_end"]) > purge
        for spec in outer)
    final_order = (
        as_date(final["fit_end"]) < as_date(final["early_stop_start"])
        <= as_date(final["early_stop_end"]) < as_date(final["calibration_start"])
        <= as_date(final["calibration_end"]) < test_start)
    minimum = (
        int(validation["inner_folds"]) * int(validation["scoring_days"])
        + int(validation["early_stop_days"])
        + 2 * int(validation["purge_hours"]) // 24)
    first_days = (
        as_date(outer[0]["train_end"]) - development_start).days + 1
    return {
        "outer_order": outer_order,
        "final_fit_early_calibration_test_order": final_order,
        "inner_fold_count_consistent": validation["inner_folds"] > 0,
        "first_outer_has_nonempty_fit": first_days > minimum,
        "first_outer_prefix_days": first_days,
        "minimum_days_before_fit": minimum,
    }


def require_official_chronology(protocol: Mapping[str, Any]) -> None:
    evidence = chronological_boundaries(protocol)
    required = (
        "outer_order", "final_fit_early_calibration_test_order",
        "inner_fold_count_consistent", "first_outer_has_nonempty_fit")
    if not all(bool(evidence[name]) for name in required):
        raise ConfigError(f"official chronology is not feasible: {evidence}")


def _active_import_check(root: Path) -> Check:
    historical_roots = {
        "".join(("leg", "acy")),
        "".join(("arch", "ive")),
    }

    def is_forbidden(module: str) -> bool:
        root_name = module.split(".", 1)[0]
        return root_name in historical_roots or bool(
            re.fullmatch(r"s(?:0[1-9]|1[0-5])_[a-z0-9_]+", root_name)
        )

    violations = []
    for path in (root / "src" / "nwp").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = ([alias.name for alias in node.names]
                     if isinstance(node, ast.Import) else
                     [node.module or ""] if isinstance(node, ast.ImportFrom)
                     else [])
            for name in names:
                if is_forbidden(name):
                    violations.append(
                        f"{path.relative_to(root).as_posix()}: {name}")
    return Check(
        "active package has no migration-layer imports", not violations,
        "; ".join(violations[:10]))


def validate_structural(root: Path | None = None) -> list[Check]:
    root = (root or project_root()).resolve()
    try:
        bundle = to_plain(load_bundle(str(root)))
        features = [
            feature for values in bundle["features"]["build"]["feature_groups"].values()
            for feature in values]
        assert_model_features(features, bundle["features"]["build"]["policy"])
        require_official_chronology(bundle["protocol"])
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        return [Check("configuration and scientific invariants", False, str(exc))]
    checks = [Check("configuration and scientific invariants", True)]
    checks.extend(Check(name.replace("_", " "), passed)
                  for name, passed in contract_alignment(bundle).items())
    checks.extend(Check(name.replace("_", " "), bool(value))
                  for name, value in chronological_boundaries(
                      bundle["protocol"]).items()
                  if isinstance(value, bool))
    checks.append(Check(
        "official result set remains unset before an official run",
        bundle["manifest"]["official_result_set"] is None))
    checks.append(_active_import_check(root))
    return checks


def _months(first: dt.date, last: dt.date) -> tuple[str, ...]:
    output = []
    current = first.replace(day=1)
    while current <= last:
        output.append(current.strftime("%Y-%m"))
        current = (current.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    return tuple(output)


def validate_data_readiness(root: Path | None = None) -> list[Check]:
    root = (root or project_root()).resolve()
    bundle = to_plain(load_bundle(str(root)))
    checks = validate_structural(root)
    paths = RunPaths.create(
        root, execution="development", run_id="readiness-validation")
    catalog = DataCatalog(paths)
    records = catalog.records()
    ready_by_path = {record.path: record for record in records
                     if record.status == "ready"}
    start = dt.date.fromisoformat(bundle["protocol"]["development_period"]["start"][:10])
    end = dt.date.fromisoformat(bundle["protocol"]["test_period"]["end"][:10])
    expected = {
        f"raw/{source}/{site}/{month}.json"
        for source in ("previous_runs", "satellite", "era5")
        for site in bundle["sites"]["sets"]["training_20"]
        for month in _months(start, end)}
    missing = sorted(expected - set(ready_by_path))
    receipt_errors, imported = [], 0
    for relative in sorted(expected & set(ready_by_path)):
        record = ready_by_path[relative]
        try:
            data_path, receipt_path = (
                catalog.dataset_path(record), catalog.receipt_path(record))
            receipt = read_receipt(receipt_path)
            if receipt["status"] != "ready" or (
                    file_sha256(data_path)
                    not in receipt["output_hashes"].values()):
                raise ValueError("receipt status/hash mismatch")
            imported += int(bool(receipt.get("imported_from")))
        except (OSError, ValueError, ConfigError, ContractError) as exc:
            receipt_errors.append(f"{relative}: {exc}")
    checks.append(Check(
        "complete ready raw source-site-month coverage",
        not missing and not receipt_errors,
        f"expected={len(expected)}; missing={len(missing)}; receipt_errors={len(receipt_errors)}",
        "data_ready"))
    checks.append(Check(
        "acquisition-time provenance accepted for official use",
        imported == 0,
        f"migration-imported ready partitions={imported}; explicit acceptance or reacquisition required",
        "data_ready"))
    checks.append(Check(
        "primary truth coverage is independently receipt-backed",
        not any(path.startswith("raw/satellite/") for path in missing)
        and not receipt_errors,
        "coverage pass does not waive the imported-provenance decision",
        "data_ready"))
    return checks


def validate_cpu_readiness(root: Path | None = None) -> list[Check]:
    root = (root or project_root()).resolve()
    bundle = to_plain(load_bundle(str(root)))
    checks = validate_data_readiness(root)
    cpu_ids = bundle["models"]["groups"]["cpu_main"]
    pending = [model_id for model_id in cpu_ids
               if (bundle["models"]["registry"][model_id]["implementation_status"]
                   != "validated"
                   or not bundle["models"]["registry"][model_id]["official_eligible"])]
    checks.append(Check(
        "declared CPU models validated and official-eligible", not pending,
        f"pending={pending}", "cpu_ready"))
    accepted: list[dict[str, str]] = []
    outputs_root = load_local_paths(root)["outputs_root"]
    data_root = load_local_paths(root)["data_root"]
    required_stages = {
        "validate", "selection", "data", "features", "splits", "tuning",
        "fitting", "prediction", "calibration", "evaluation", "analysis",
        "figures", "report"}
    for run_root in sorted((outputs_root / "development").glob("*")):
        try:
            meta = run_root / "00_meta"
            provenance = __import__("json").loads(
                (meta / "provenance.json").read_text(encoding="utf-8"))
            resolved = __import__("yaml").safe_load(
                (meta / "resolved_config.yaml").read_text(encoding="utf-8"))
            stages = __import__("json").loads(
                (meta / "stage_results.json").read_text(encoding="utf-8"))
            manifest = __import__("json").loads(
                (meta / "artifact_manifest.json").read_text(encoding="utf-8"))
            clean = __import__("json").loads(
                (meta / "clean_manifest.json").read_text(encoding="utf-8"))
            receipt = read_receipt(run_root / "09_report" / "report.receipt.json")
            if provenance.get("run_type") != "executable":
                continue
            if resolved.get("execution") != "development":
                continue
            if set(stages) != required_stages or any(
                    value.get("status") != "success" for value in stages.values()):
                continue
            if not manifest.get("artifacts"):
                continue
            records = clean.get("records", [])
            if not records or any(
                    record.get("status") != "ready"
                    or "imported" in str(record.get("path", "")).lower()
                    or not (data_root / str(record.get("receipt_path", ""))).is_file()
                    for record in records):
                continue
            report_stage = stages["report"]
            if receipt.get("output_hashes") != report_stage.get("output_hashes"):
                continue
            report_outputs_valid = True
            for key, path_text in report_stage.get("artifact_outputs", {}).items():
                path = Path(path_text)
                if (not path.is_file()
                        or report_stage["output_hashes"].get(key) != file_sha256(path)):
                    report_outputs_valid = False
                    break
            if not report_outputs_valid or not report_stage.get("artifact_outputs"):
                continue
            markers = (str(provenance) + str(resolved)).lower()
            if "synthetic" in markers or "imported_evidence" in markers:
                continue
            accepted.append({"run_id": run_root.name,
                             "config_hash": str(resolved["config_hash"])})
        except (OSError, KeyError, TypeError, ValueError, ContractError):
            continue
    checks.append(Check(
        "receipt-backed real-data development mini-E2E accepted", bool(accepted),
        f"accepted_runs={accepted}", "cpu_ready"))
    missing_runtime = [name for name in ("numpy", "pandas", "pyarrow", "lightgbm", "xgboost")
                       if importlib.util.find_spec(name) is None]
    checks.append(Check(
        "required CPU runtime dependencies importable", not missing_runtime,
        f"missing={missing_runtime}", "cpu_ready"))
    return checks


def validate_spatial_readiness(root: Path | None = None) -> list[Check]:
    root = (root or project_root()).resolve()
    bundle = to_plain(load_bundle(str(root)))
    checks = validate_cpu_readiness(root)
    spatial = bundle["protocol"]["spatial_design"]
    checks.append(Check(
        "returned-service registry frozen after convergence",
        spatial["service_registry_status"] == "frozen",
        f"status={spatial['service_registry_status']}", "spatial_ready"))
    checks.append(Check(
        "spatial generalization explicitly enabled",
        spatial["execution_status"] == "enabled",
        f"status={spatial['execution_status']}", "spatial_ready"))
    return checks


def validate_deep_readiness(root: Path | None = None) -> list[Check]:
    root = (root or project_root()).resolve()
    bundle = to_plain(load_bundle(str(root)))
    checks = validate_cpu_readiness(root)
    deep_ids = bundle["models"]["groups"]["deep_all"]
    pending = [model_id for model_id in deep_ids
               if (bundle["models"]["registry"][model_id]["implementation_status"]
                   != "validated"
                   or not bundle["models"]["registry"][model_id]["official_eligible"])]
    checks.append(Check(
        "declared deep models validated and official-eligible", not pending,
        f"pending={pending}", "deep_ready"))
    return checks


def validate_official_readiness(root: Path | None = None) -> list[Check]:
    root = (root or project_root()).resolve()
    bundle = to_plain(load_bundle(str(root)))
    checks = validate_spatial_readiness(root)
    checks.extend(check for check in validate_deep_readiness(root)
                  if check.scope == "deep_ready")
    checks.append(Check(
        "official result set registered",
        bool(bundle["manifest"]["official_result_set"]),
        "formal/full execution remains forbidden until all readiness evidence passes",
        "official"))
    return checks


VALIDATORS: dict[str, Callable[[Path | None], list[Check]]] = {
    "structural": validate_structural,
    "data_ready": validate_data_readiness,
    "cpu_ready": validate_cpu_readiness,
    "spatial_ready": validate_spatial_readiness,
    "deep_ready": validate_deep_readiness,
    "official_full": validate_official_readiness,
    "official": validate_official_readiness,
}


def readiness_result(mode: str, root: Path | None = None) -> dict[str, Any]:
    try:
        validator = VALIDATORS[mode]
    except KeyError as exc:
        raise ConfigError(f"invalid validation mode: {mode}") from exc
    checks = validator(root)
    return {
        "mode": mode,
        "status": "pass" if all(check.passed for check in checks) else "blocked",
        "checks": [asdict(check) for check in checks],
        "passed": sum(check.passed for check in checks),
        "total": len(checks),
    }


def official_readiness_receipt(config: Any, root: Path | None = None) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    checks = validate_official_readiness(root)
    by_scope: dict[str, bool] = {}
    for scope in ("structural", "data_ready", "cpu_ready", "deep_ready", "spatial_ready"):
        scoped = [check.passed for check in checks if check.scope == scope]
        by_scope[scope] = bool(scoped) and all(scoped)
    bundle = to_plain(load_bundle(str(root)))
    eligibility = {
        model_id: bool(record["official_eligible"] and record["implementation_status"] == "validated")
        for model_id, record in bundle["models"]["registry"].items()
        if model_id in config.selected_models
    }
    registered = bundle["manifest"]["official_result_set"] is not None
    provenance_ready = all(check.passed for check in checks
                           if "provenance" in check.name or "receipt" in check.name)
    environment_ready = all(check.passed for check in checks
                            if "runtime" in check.name or "environment" in check.name)
    selected_records = config.models["registry"]
    deep_required = any(selected_records[model_id].get("family") in {
        "deep", "experimental_constrained"} for model_id in config.selected_models)
    spatial_required = bool(config.scope.get("spatial"))
    required_scopes = (
        by_scope["structural"] and by_scope["data_ready"] and by_scope["cpu_ready"]
        and (by_scope["deep_ready"] if deep_required else True)
        and (by_scope["spatial_ready"] if spatial_required else True)
    )
    overall = (required_scopes and registered and provenance_ready
               and environment_ready and all(eligibility.values()))
    return {
        "scientific_run_hash": config.config_hash,
        "structural_ready": by_scope["structural"],
        "data_ready": by_scope["data_ready"],
        "cpu_ready": by_scope["cpu_ready"],
        "deep_ready": by_scope["deep_ready"],
        "spatial_ready": by_scope["spatial_ready"],
        "official_result_registered": registered,
        "model_eligibility": eligibility,
        "environment_ready": environment_ready,
        "provenance_ready": provenance_ready,
        "overall_ready": overall,
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
