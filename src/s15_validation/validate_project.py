"""Evidence-based readiness gates; a structural pass never authorizes training."""
from __future__ import annotations

import ast
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
import datetime as dt
import json

from s01_core.config_loader import ProtocolError, load_data_config, load_manifest, project_root
from s01_core.schemas import assert_model_features
from s15_validation.validate_protocol import check_contract_alignment, chronological_boundaries
from s02_data.clean_contract import validate_cached_month


ACTIVE_PACKAGES = (
    "s01_core", "s02_data", "s03_features", "s04_splits", "s05_tuning",
    "s06_models", "s07_prediction", "s08_calibration", "s09_metrics",
    "s10_evaluation", "s11_interpretation", "s12_spatial",
    "s13_visualization", "s14_pipeline", "s15_validation",
)
FORBIDDEN_IMPORT_PARTS = ("legacy", "cv_month_balanced_quantile")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mini_e2e_receipt_valid(data_root: Path, manifest: dict) -> bool:
    base = data_root / "05_cpu_mini_e2e"
    receipt = base / "receipt.json"
    if not receipt.is_file():
        return False
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        expected = {
            "status": "pass", "result_status": "diagnostic", "execution_level": "smoke",
            "official_eligible": False, "site": "nanjing_1", "outer_fold": "outer_1",
            "inner_fold": "inner_1", "quantile_count": 7,
            "data_version": manifest["data_version"],
            "feature_version": manifest["feature_version"],
            "protocol_version": manifest["protocol_version"],
            "daylight_definition": manifest["daylight_definition"]["formal"],
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            return False
        if set(payload.get("fixed_models_tested", [])) != {
            "climatology", "persistence", "smart_persistence", "optimal_convex",
            "raw_gfs", "bias_correction", "linear_mos",
        } or payload.get("quantile_models_tested") != ["ridge_mos", "lgbm", "xgboost"]:
            return False
        if min(payload.get("fit_rows", 0), payload.get("early_stop_rows", 0),
               payload.get("scoring_rows", 0)) <= 0:
            return False
        paths = {
            "source_feature_sha256": data_root / "03_featured" / "nanjing_1_featured_2024-02_2026-08.parquet",
            "source_clean_sha256": data_root / "02_clean" / "nanjing_1_clean_2024-02_2026-08.parquet",
            "prediction_sha256": base / "predictions.parquet",
        }
        return all(path.is_file() and payload.get(key) == _sha256(path)
                   for key, path in paths.items())
    except (OSError, ValueError, TypeError, KeyError):
        return False


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""
    scope: str = "structural"


def _check_imports(root: Path) -> list[Check]:
    violations = []
    for package in ACTIVE_PACKAGES:
        for path in (root / "src" / package).rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:
                violations.append(f"{path.relative_to(root)}: syntax {exc.lineno}")
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    if any(part in name.split(".") for part in FORBIDDEN_IMPORT_PARTS):
                        violations.append(f"{path.relative_to(root)}: {name}")
    return [Check("active formal modules never import legacy/month-balanced", not violations,
                  "; ".join(violations[:10]))]


def validate_structural(root: Path | None = None) -> list[Check]:
    root = root or project_root()
    checks: list[Check] = []
    variables: dict = {}
    try:
        manifest = load_manifest(root / "project_manifest.yaml")
        checks.append(Check("unique valid manifest and core invariants", True))
    except (ProtocolError, KeyError, ValueError) as exc:
        return [Check("unique valid manifest and core invariants", False, str(exc))]
    try:
        variables = load_data_config("02_variables.yaml", root)
        forecast = variables["forecast_variables"]
        expected = {"cloud_cover"}
        excluded = {"cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"}
        checks.append(Check("archive-aligned 15-variable forecast contract", len(forecast) == 15
                            and len(set(forecast)) == 15 and expected <= set(forecast)
                            and not excluded.intersection(forecast)
                            and manifest["cpu_experiment"]["forecast_variable_count"] == 15,
                            f"count={len(forecast)}; layered_cloud_excluded={not excluded.intersection(forecast)}"))
    except (ProtocolError, KeyError, ValueError) as exc:
        checks.append(Check("archive-aligned 15-variable forecast contract", False, str(exc)))
    checks.append(Check("seven-quantile probabilistic primary objective",
                        len(manifest["quantiles"]) == 7
                        and manifest["selection_metric"] == "mean_pinball"))
    checks.append(Check("distinct six-candidate tuning budget",
                        manifest["tuning_budget"]["trials_per_model"] == 6
                        and manifest["tuning_budget"]["trial_definition"]
                        == "distinct_preregistered_hyperparameter_candidate"))
    checks.append(Check("10-day primary purge and 7/10/14 sensitivity",
                        manifest["validation_protocol"]["purge_hours"] == 240
                        and manifest["validation_protocol"]["gap_sensitivity_days"] == [7, 10, 14]))
    checks.append(Check("separate inner fit/early-stop/score ordering",
                        manifest["validation_protocol"]["inner_order"]
                        == ["fit", "purge", "early_stop", "purge", "score"]))
    checks.append(Check("one formal daylight definition",
                        manifest["daylight_definition"]["formal"] == "solar_elevation_gt_0"))
    checks.append(Check("24 semantic registered models",
                        len(manifest["models"]) == manifest["model_count"] == 24))
    checks.append(Check("GFS returned service points are spatial study object",
                        manifest["spatial_design"]["study_object"]
                        == "open_meteo_returned_service_points"
                        and manifest["spatial_design"]["requested_coordinates_role"]
                        == "provenance_only"))
    feature_names = [feature for group, names in manifest["feature_groups"].items()
                     if group not in {"selection_evidence", "shap_role"} for feature in names]
    try:
        assert_model_features(feature_names)
        checks.append(Check("identity/truth excluded from formal model feature registry", True))
    except ProtocolError as exc:
        checks.append(Check("identity/truth excluded from formal model feature registry", False, str(exc)))
    checks.append(Check("official_result_set remains unset before rerun",
                        manifest["official_result_set"] is None))
    try:
        for name, passed in check_contract_alignment(manifest, variables).items():
            checks.append(Check(name.replace("_", " "), passed))
        chronology = chronological_boundaries(manifest)
        for name in ("outer_order", "final_fit_early_calibration_test_order",
                     "inner_fold_count_consistent"):
            checks.append(Check(name.replace("_", " "), bool(chronology[name])))
    except (KeyError, ValueError, ImportError, TypeError) as exc:
        checks.append(Check("cross-module protocol alignment", False, str(exc)))
    checks.append(Check("frozen website isolated from research execution",
                        (root / "NWP_website_handoff").is_dir()
                        and not (root / "website").exists()))
    checks.extend(_check_imports(root))
    return checks


def validate_data_readiness(root: Path | None = None) -> list[Check]:
    root = root or project_root()
    manifest = load_manifest(root / "project_manifest.yaml")
    checks = validate_structural(root)
    data_root = root / manifest["data_layout"]["root"]
    sites = load_data_config("01_sites.yaml", root)["sites"]
    first = dt.date.fromisoformat(manifest["development_period"]["start"][:10])
    last = dt.date.fromisoformat(manifest["test_period"]["end"][:10])
    months = []
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    sources = (("previous_runs", "01_gfs"), ("era5", "02_era5"),
               ("satellite", "03_satellite"))
    missing = []
    for site in sites:
        for month_key in months:
            for _, folder in sources:
                path = data_root / "01_raw" / folder / site["id"] / f"{site['id']}_{month_key}.json"
                if not path.is_file() or not path.with_name(path.name + ".meta.json").is_file():
                    missing.append(str(path.relative_to(root)))
    checks.append(Check("complete archive-aligned monthly raw grid with sidecars", not missing,
                        f"expected={len(sites)*len(months)*len(sources)} files; missing={len(missing)}; "
                        f"first={missing[:2]}", "data_ready"))
    pilot = data_root / "01_raw" / "01_gfs" / "nanjing_1" / "nanjing_1_2024-02.json"
    if pilot.is_file():
        try:
            site = next(site for site in sites if site["id"] == "nanjing_1")
            validate_cached_month(pilot, source="previous_runs",
                                  cfg=load_data_config("02_variables.yaml", root),
                                  start=dt.date(2024, 2, 1), end=dt.date(2024, 2, 29),
                                  data_version=manifest["data_version"],
                                  requested_lat=site["lat"], requested_lon=site["lon"])
            checks.append(Check("real pilot GFS satisfies archive-aligned 15-variable contract", True,
                                scope="data_ready"))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            checks.append(Check("real pilot GFS satisfies archive-aligned 15-variable contract", False,
                                str(exc), "data_ready"))
    else:
        checks.append(Check("real pilot GFS satisfies archive-aligned 15-variable contract", False,
                            "no current-protocol pilot file", "data_ready"))
    checks.append(Check("returned-service coordinate provenance complete", not missing,
                        "requires every current-protocol GFS sidecar and raw payload", "data_ready"))
    checks.append(Check("independently audited hourly Himawari truth for fixed 20-site CPU cohort", False,
                        "20-site current-protocol truth receipt absent; province-wide spatial gate is deferred",
                        "data_ready"))
    return checks


def validate_cpu_readiness(root: Path | None = None) -> list[Check]:
    root = root or project_root()
    manifest = load_manifest(root / "project_manifest.yaml")
    checks = validate_data_readiness(root)
    chronology = chronological_boundaries(manifest)
    checks.append(Check("first outer prefix fits all three purged inner folds",
                        bool(chronology["first_outer_has_nonempty_fit"]),
                        f"available={chronology['first_outer_prefix_days']}d; "
                        f"minimum before nonempty fit={chronology['minimum_days_before_fit']}d",
                        "cpu_ready"))
    sites = load_data_config("01_sites.yaml", root)["sites"]
    cohort_ok = (manifest["cpu_experiment"]["cohort"]
                 == "configured_20_sites_with_returned_service_coordinates"
                 and manifest["cpu_experiment"]["spatial_generalization"] == "deferred"
                 and len(sites) == 20 and len({site["id"] for site in sites}) == 20)
    checks.append(Check("fixed 20-site CPU cohort registered without spatial claim",
                        cohort_ok, "forecast and truth must still carry verified returned coordinates",
                        "cpu_ready"))
    cpu_ids = manifest["cpu_experiment"]["model_ids"]
    models = {item["id"]: item for item in manifest["models"]}
    invalid = [model_id for model_id in cpu_ids if model_id not in models or
               models[model_id]["execution_device"] != "cpu" or
               models[model_id]["implementation_status"] != "validated" or
               not models[model_id]["official_eligible"]]
    checks.append(Check("declared CPU models validated and eligible", not invalid,
                        f"pending={invalid}; deep-model status intentionally ignored", "cpu_ready"))
    tuned = [model_id for model_id in cpu_ids if models[model_id]["tuning_required"]]
    candidate_ok = all(len(manifest["tuning_search_spaces"].get(model_id, [])) == 6
                       for model_id in tuned)
    fixed_ok = all(not models[model_id]["tuning_required"]
                   for model_id in cpu_ids if model_id not in tuned)
    checks.append(Check("six candidates only for tuned CPU models", candidate_ok and fixed_ok,
                        f"tuned={tuned}", "cpu_ready"))
    from s03_features.engineering import formal_feature_columns
    from s01_core.schemas import assert_model_features
    try:
        # The registry is the sole selection source; it also rejects IDs/truth.
        columns = [column for group, values in manifest["feature_groups"].items()
                   if group not in {"selection_evidence", "shap_role"} for column in values]
        assert_model_features(columns)
        feature_ok = callable(formal_feature_columns)
    except (ValueError, KeyError, TypeError):
        feature_ok = False
    checks.append(Check("one formal feature registry and identity exclusion", feature_ok,
                        scope="cpu_ready"))
    from s03_features.preprocessing import fit_fold_preprocessing
    checks.append(Check("fold-local preprocessing entry point available",
                        callable(fit_fold_preprocessing), scope="cpu_ready"))
    from s07_prediction.schema import validate_predictions
    from s09_metrics.deterministic import point_metrics
    from s10_evaluation.runner import evaluate_predictions
    checks.append(Check("prediction, probability and point metric interfaces available",
                        all(callable(fn) for fn in
                            (validate_predictions, point_metrics, evaluate_predictions)),
                        "interface presence does not substitute for mini-E2E", "cpu_ready"))
    receipt_ok = _mini_e2e_receipt_valid(root / manifest["data_layout"]["root"], manifest)
    checks.append(Check("real-data CPU mini-E2E receipt", receipt_ok,
                        "one-site diagnostic interface test only, with matching artifact hashes; not an official score",
                        "cpu_ready"))
    from s01_core.provenance import runtime_provenance
    try:
        snapshot = runtime_provenance(root)
        missing_packages = [name for name, version in snapshot["packages"].items()
                            if version is None]
        lock = (root / "requirements-lock.txt").read_text(encoding="utf-8")
        pinned = all(f"{name}=={version}" in lock for name, version in
                     snapshot["packages"].items() if version is not None)
        provenance_ok = not missing_packages and pinned
        detail = f"missing={missing_packages}; pinned_installed_versions={pinned}"
    except (ValueError, OSError, KeyError) as exc:
        provenance_ok, detail = False, str(exc)
    checks.append(Check("Git/environment provenance capture and package pins",
                        provenance_ok, detail, "cpu_ready"))
    return checks


def validate_spatial_readiness(root: Path | None = None) -> list[Check]:
    root = root or project_root()
    manifest = load_manifest(root / "project_manifest.yaml")
    checks = validate_cpu_readiness(root)
    registry = root / manifest["data_layout"]["root"] / "04_service_probes" / "frozen_registry.json"
    frozen = manifest["spatial_design"]["service_registry_status"] == "frozen" and registry.is_file()
    checks.append(Check("Jiangsu returned-service registry frozen after convergence", frozen,
                        "spatial generalization deferred; province-wide probe remains partial",
                        "spatial_ready"))
    checks.append(Check("spatial generalization explicitly enabled",
                        manifest["spatial_design"].get("execution_status") == "enabled",
                        "deferred by user; no spatial-generalization results may be claimed",
                        "spatial_ready"))
    return checks


def validate_deep_readiness(root: Path | None = None) -> list[Check]:
    checks = validate_cpu_readiness(root)
    manifest = load_manifest((root or project_root()) / "project_manifest.yaml")
    deep = [item for item in manifest["models"] if item["execution_device"] == "gpu"]
    checks.append(Check("14 GPU model implementations validated", len(deep) == 14 and all(
        item["implementation_status"] == "validated" and item["official_eligible"]
        for item in deep), "pinned experimental model is not silently promoted", "deep_ready"))
    return checks


def validate_official_readiness(root: Path | None = None) -> list[Check]:
    checks = validate_spatial_readiness(root) + [
        check for check in validate_deep_readiness(root)
        if check.scope == "deep_ready"
    ]
    checks.append(Check("official experiment result set intentionally not yet created", False,
                        "full benchmark forbidden in this preparation round", "official_full"))
    return checks


def result(mode: str = "structural", root: Path | None = None) -> dict:
    modes = {"structural": validate_structural, "data_ready": validate_data_readiness,
             "cpu_ready": validate_cpu_readiness, "spatial_ready": validate_spatial_readiness,
             "deep_ready": validate_deep_readiness,
             "official_full": validate_official_readiness, "official": validate_official_readiness}
    if mode not in modes:
        raise ProtocolError(f"invalid validation mode: {mode}")
    checks = modes[mode](root)
    return {
        "mode": mode,
        "status": "pass" if all(check.passed for check in checks) else "blocked",
        "checks": [asdict(check) for check in checks],
        "passed": sum(check.passed for check in checks),
        "total": len(checks),
    }
