"""Strong structural checks and an explicit, conservative official gate."""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path

from s01_core.config_loader import ProtocolError, load_data_config, load_manifest, project_root
from s01_core.schemas import assert_model_features
from s15_validation.validate_protocol import check_contract_alignment, chronological_boundaries


ACTIVE_PACKAGES = (
    "s01_core", "s02_data", "s03_features", "s04_splits", "s05_tuning",
    "s06_models", "s07_prediction", "s08_calibration", "s09_metrics",
    "s10_evaluation", "s11_interpretation", "s12_spatial",
    "s13_visualization", "s14_pipeline", "s15_validation",
)
FORBIDDEN_IMPORT_PARTS = ("legacy", "cv_month_balanced_quantile")


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
        expected = {"cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"}
        checks.append(Check("18-variable forecast contract", len(forecast) == 18
                            and expected <= set(forecast), f"count={len(forecast)}"))
    except (ProtocolError, KeyError, ValueError) as exc:
        checks.append(Check("18-variable forecast contract", False, str(exc)))
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


def validate_official_readiness(root: Path | None = None) -> list[Check]:
    root = root or project_root()
    manifest = load_manifest(root / "project_manifest.yaml")
    checks = validate_structural(root)
    chronology = chronological_boundaries(manifest)
    checks.extend([
        Check("first outer prefix supports three purged inner folds",
              bool(chronology["first_outer_has_nonempty_fit"]),
              f"available={chronology['first_outer_prefix_days']}d; minimum before nonempty fit="
              f"{chronology['minimum_days_before_fit']}d; protocol change requires explicit decision",
              "official"),
        Check("Jiangsu returned-service registry convergence frozen",
              manifest["spatial_design"]["service_registry_status"] == "frozen",
              "0.05-degree probe: 707 distinct returned coordinates from in-province requests; "
              "654 returned coordinates inside the Jiangsu boundary; no convergence audit", "official"),
        Check("province hourly Himawari truth gate passed", False,
              "no complete independently gated hourly truth is registered", "official"),
        Check("v2 raw/clean/featured data regenerated and audited", False,
              "current local raw cache predates strict 18-variable version sidecars", "official"),
        Check("all compared models audited as validated implementations", False,
              "per-model architecture and six-candidate tuner integration remains pending", "official"),
    ])
    return checks


def result(mode: str = "structural", root: Path | None = None) -> dict:
    if mode not in {"structural", "official"}:
        raise ProtocolError(f"invalid validation mode: {mode}")
    checks = validate_structural(root) if mode == "structural" else validate_official_readiness(root)
    return {
        "mode": mode,
        "status": "pass" if all(check.passed for check in checks) else "blocked",
        "checks": [asdict(check) for check in checks],
        "passed": sum(check.passed for check in checks),
        "total": len(checks),
    }
