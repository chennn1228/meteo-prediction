"""The sole YAML entry point and deterministic run configuration resolver."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import yaml

from .hashing import content_hash
from .schema import ContractError


class ConfigError(ContractError):
    """A configuration violates the one-source scientific contract."""


CONFIG_NAMES = ("protocol", "data", "features", "sites", "models", "experiments")
CONFIG_PATHS = {name: f"config/{name}.yaml" for name in CONFIG_NAMES}
CONFIG_ROOT_KEYS = {
    "protocol": {
        "target", "development_period", "test_period", "validation",
        "probability", "evaluation", "daylight_definition", "execution_gate",
        "seed_policy", "spatial_design",
    },
    "data": {
        "version", "forecast", "truth", "radiation_semantics",
        "requested_coordinates_role", "timezone", "tilt", "azimuth", "storage",
        "receipt_required", "clean_contract",
    },
    "features": {"build", "analysis"},
    "sites": {"registry", "sets", "selectors"},
    "models": {"registry", "groups", "search"},
    "experiments": {"profiles"},
}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _tuning_enabled(record: Mapping[str, Any]) -> bool:
    tuning = record.get("tuning")
    return bool(tuning.get("enabled")) if isinstance(tuning, Mapping) else False


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ConfigError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def project_root(start: Path | None = None) -> Path:
    probe = (start or Path(__file__)).resolve()
    candidates = (probe, *probe.parents) if probe.is_dir() else probe.parents
    for parent in candidates:
        if (parent / "project_manifest.yaml").is_file():
            return parent
    raise ConfigError("project_manifest.yaml not found")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"required configuration is missing: {path}")
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    if not isinstance(value, dict):
        raise ConfigError(f"configuration must be a YAML mapping: {path}")
    return value


def read_resolved_run_config(path: Path) -> dict[str, Any]:
    """Read a persisted resolved snapshot through the sole YAML parser."""
    return _read_yaml(Path(path))


def write_resolved_run_config(path: Path, config: "RunConfig") -> None:
    """Create, never overwrite, one resolved run snapshot."""
    path = Path(path)
    if path.exists():
        raise ConfigError(f"resolved configuration already exists: {path}")
    path.write_text(
        yaml.safe_dump(config.as_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def load_local_paths(root: Path | None = None) -> Mapping[str, Path]:
    """Resolve optional machine-only roots through the sole YAML reader."""
    root = (root or project_root()).resolve()
    local_path = root / "config" / "local.yaml"
    values: Mapping[str, Any] = {}
    if local_path.exists():
        local = _read_yaml(local_path)
        if (set(local) != {"paths"} or not isinstance(local.get("paths"), Mapping)
                or set(local["paths"]) - {"data_root", "outputs_root"}):
            raise ConfigError("config/local.yaml may contain only paths.data_root and paths.outputs_root")
        values = local["paths"]
    resolved: dict[str, Path] = {}
    for name, default in (("data_root", "data"), ("outputs_root", "outputs")):
        raw_value = values.get(name, default)
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ConfigError(f"local path {name} must be a nonempty string")
        candidate = Path(raw_value)
        resolved[name] = (candidate if candidate.is_absolute() else root / candidate).resolve()
        if resolved[name] == root:
            raise ConfigError(f"local path {name} must not be the project root")
    if resolved["data_root"] == resolved["outputs_root"]:
        raise ConfigError("data_root and outputs_root must be distinct")
    return MappingProxyType(resolved)


@lru_cache(maxsize=4)
def load_bundle(root_text: str | None = None) -> Mapping[str, Any]:
    root = Path(root_text) if root_text else project_root()
    manifest = _read_yaml(root / "project_manifest.yaml")
    if set(manifest) != {"project_id", "protocol_version", "official_result_set", "configs"}:
        raise ConfigError("project_manifest.yaml is an entry point only; move facts to one child config")
    config_paths = manifest["configs"]
    if set(config_paths) != set(CONFIG_NAMES):
        raise ConfigError(f"manifest must name exactly {CONFIG_NAMES}")
    if config_paths != CONFIG_PATHS:
        raise ConfigError(f"manifest configuration paths must be exactly {CONFIG_PATHS}")
    bundle: dict[str, Any] = {"manifest": manifest}
    for name in CONFIG_NAMES:
        relative = Path(config_paths[name])
        if relative.is_absolute() or ".." in relative.parts:
            raise ConfigError(f"manifest configuration path must stay under project root: {relative}")
        bundle[name] = _read_yaml(root / relative)
    _validate_bundle(bundle)
    return _freeze(bundle)


def _validate_bundle(bundle: Mapping[str, Any]) -> None:
    manifest = bundle["manifest"]
    for name, expected in CONFIG_ROOT_KEYS.items():
        if not isinstance(bundle[name], Mapping):
            raise ConfigError(f"{name}.yaml must contain a mapping")
        actual = set(bundle[name])
        if actual != expected:
            raise ConfigError(
                f"{name}.yaml top-level keys must be exactly {sorted(expected)}; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
    protocol, data = bundle["protocol"], bundle["data"]
    features = bundle["features"]["build"]
    def exact_keys(value: Any, allowed: set[str], label: str) -> None:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{label} must be a mapping")
        extra, missing = set(value) - allowed, allowed - set(value)
        if extra or missing:
            raise ConfigError(
                f"{label} keys must be exactly {sorted(allowed)}; "
                f"missing={sorted(missing)}, extra={sorted(extra)}")

    exact_keys(protocol["validation"], {
        "name", "maximum_sequence_lookback_hours", "maximum_lead_hours",
        "purge_hours", "purge_derivation", "gap_sensitivity_days",
        "inner_folds", "early_stop_days", "scoring_days", "inner_order",
        "first_outer_insufficiency_policy", "outer_folds", "final_fit"},
        "protocol.validation")
    exact_keys(protocol["probability"], {
        "quantiles", "selection_metric", "construction", "calibration"},
        "protocol.probability")
    exact_keys(protocol["evaluation"], {
        "deterministic_metrics", "probabilistic_metrics",
        "point_metric_references", "prediction_contract"},
        "protocol.evaluation")
    exact_keys(protocol["execution_gate"], {
        "implementation_levels", "execution_levels", "official",
        "readiness_scope", "model_execution_requires_explicit_flag"},
        "protocol.execution_gate")
    exact_keys(protocol["spatial_design"], {
        "execution_status", "province_model_forbids_station_id", "levels",
        "truth_gate", "study_object", "service_registry_status",
        "empirical_probe_step_degrees", "service_probe_registry",
        "convergence_required_before_official"}, "protocol.spatial_design")
    exact_keys(data["truth"], {"primary", "supplementary"}, "data.truth")
    exact_keys(data["truth"]["primary"], {
        "provider", "model", "semantics", "variables"}, "data.truth.primary")
    exact_keys(data["truth"]["supplementary"], {
        "provider", "role", "variables"}, "data.truth.supplementary")
    exact_keys(data["storage"], {
        "raw_partitioning", "clean_partitioning", "feature_partitioning",
        "raw_overwrite_forbidden"}, "data.storage")
    exact_keys(data["clean_contract"], {
        "radiation_negative_policy", "cloud_range", "missing_policy", "row_unit"},
        "data.clean_contract")
    exact_keys(features, {
        "version", "status", "clear_sky", "feature_groups",
        "derived_features", "policy"}, "features.build")
    exact_keys(features["policy"], {
        "issue_time_availability_required", "identity_fields_forbidden",
        "truth_fields_forbidden", "location_for_solar_geometry",
        "solar_geometry_time_basis", "clear_sky_time_basis",
        "circular_encodings", "clear_sky_denominator", "diffuse_denominator",
        "diagnostic_ratio_columns", "diagnostic_ratio_clipping",
        "model_ratio_columns", "model_ratio_transform", "preprocessing",
        "lag_source", "cloud_imputation_predictors"}, "features.build.policy")
    exact_keys(bundle["features"]["analysis"], {
        "evidence", "diagnostics"}, "features.analysis")
    for selector_id, selector in bundle["sites"]["selectors"].items():
        exact_keys(selector, {"method", "status", "official_eligible", "source_set",
                              "allowed_counts", "region_quotas"},
                   f"sites.selectors.{selector_id}")
    model_allowed = {"family", "role", "internal_version", "implementation",
                     "quantile_adapter", "algorithm", "device", "tuning",
                     "implementation_status", "official_eligible",
                     "workflow_status", "constraints"}
    for model_id, model in bundle["models"]["registry"].items():
        if set(model) - model_allowed:
            raise ConfigError(f"models.registry.{model_id} contains unknown keys: "
                              f"{sorted(set(model) - model_allowed)}")
        exact_keys(model["tuning"], {"enabled"},
                   f"models.registry.{model_id}.tuning")
    exact_keys(bundle["models"]["search"], {"budget", "spaces"}, "models.search")
    exact_keys(bundle["models"]["search"]["budget"], {
        "trials_per_model", "equal_budget", "report_compute_budget",
        "trial_definition", "outer_validation_forbidden_during_selection",
        "final_test_forbidden_during_selection"}, "models.search.budget")
    for profile_id, profile in bundle["experiments"]["profiles"].items():
        allowed = {"execution", "sites", "models", "spatial", "locked_config_hash"}
        if set(profile) - allowed:
            raise ConfigError(f"experiments.profiles.{profile_id} contains unknown keys: "
                              f"{sorted(set(profile) - allowed)}")
    for value, label in ((manifest.get("project_id"), "project ID"),):
        if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
            raise ConfigError(f"unsafe {label}: {value!r}")
    for section, label in ((bundle["sites"].get("registry"), "site"),
                           (bundle["sites"].get("sets"), "site set"),
                           (bundle["sites"].get("selectors"), "site selector"),
                           (bundle["models"].get("registry"), "model"),
                           (bundle["models"].get("groups"), "model group"),
                           (bundle["experiments"].get("profiles"), "profile")):
        if not isinstance(section, Mapping):
            raise ConfigError(f"{label} registry must be a mapping")
        unsafe = [key for key in section if not isinstance(key, str) or not _SAFE_ID.fullmatch(key)]
        if unsafe:
            raise ConfigError(f"unsafe {label} ID: {unsafe[0]!r}")
    if not isinstance(protocol.get("target"), str) or not protocol["target"]:
        raise ConfigError("protocol target must be declared")
    if not isinstance(data.get("version"), str) or not data["version"]:
        raise ConfigError("data version must be declared in data.yaml")
    if not isinstance(features.get("version"), str) or not features["version"]:
        raise ConfigError("feature version must be declared in features.yaml")
    validation = protocol.get("validation", {})
    if validation.get("purge_hours") != validation.get("maximum_sequence_lookback_hours", -1) + validation.get("maximum_lead_hours", -1):
        raise ConfigError("purge_hours must cover maximum sequence lookback plus lead")
    if validation.get("purge_derivation") != "maximum_sequence_lookback_hours + maximum_lead_hours":
        raise ConfigError("purge derivation must name the registered lookback-plus-lead rule")
    if validation.get("inner_order") != ["fit", "purge", "early_stop", "purge", "score"]:
        raise ConfigError("inner validation must separate fit, early-stop, and score blocks with purges")
    outer_folds = validation.get("outer_folds", [])
    if not outer_folds or len({fold.get("id") for fold in outer_folds}) != len(outer_folds):
        raise ConfigError("outer folds must be a nonempty uniquely identified list")
    for fold in outer_folds:
        boundaries = [fold.get(key) for key in ("train_start", "train_end", "purge_start", "purge_end", "validation_start", "validation_end")]
        if any(not isinstance(value, str) for value in boundaries) or boundaries != sorted(boundaries):
            raise ConfigError(f"outer fold chronology is invalid: {fold.get('id')}")
    final_fit = validation.get("final_fit", {})
    final_boundaries = [final_fit.get(key) for key in ("fit_end", "early_stop_start", "early_stop_end", "calibration_start", "calibration_end")]
    if any(not isinstance(value, str) for value in final_boundaries) or final_boundaries != sorted(final_boundaries):
        raise ConfigError("final fit, early-stop, and calibration chronology is invalid")
    if final_fit.get("calibration_end", "") >= protocol.get("test_period", {}).get("start", "")[:10]:
        raise ConfigError("calibration must end before the isolated final test period")
    quantiles = protocol.get("probability", {}).get("quantiles", [])
    if not quantiles or len(set(quantiles)) != len(quantiles) or list(quantiles) != sorted(quantiles) or any(not 0 < float(item) < 1 for item in quantiles):
        raise ConfigError("probability quantiles must be strictly ordered values in (0, 1)")
    if protocol.get("probability", {}).get("selection_metric") != "mean_pinball":
        raise ConfigError("formal model selection requires mean_pinball")
    if protocol.get("daylight_definition", {}).get("formal") != "solar_elevation_gt_0":
        raise ConfigError("formal daylight definition must be solar_elevation_gt_0")
    forecast = data.get("forecast", {})
    if not isinstance(forecast, Mapping):
        raise ConfigError("forecast must be a mapping")
    expected_forecast = {"provider", "model", "product_class", "cell_selection", "source_grid_id_exposed", "resolution", "leads", "variables"}
    if set(forecast) != expected_forecast:
        raise ConfigError("forecast contains unknown or missing keys")
    if forecast.get("cell_selection") not in {"land"} or forecast.get("resolution") not in {"hourly"}:
        raise ConfigError("invalid forecast enum")
    if not forecast.get("variables") or len(forecast["variables"]) != len(set(forecast["variables"])):
        raise ConfigError("forecast variables must be declared in data.yaml")
    leads = forecast.get("leads", [])
    if not leads or len(leads) != len(set(leads)) or any(int(lead) <= 0 for lead in leads):
        raise ConfigError("forecast leads must be unique positive hours")
    if max(leads) != validation["maximum_lead_hours"]:
        raise ConfigError("maximum forecast lead must match validation.maximum_lead_hours")
    if "root" in data.get("storage", {}):
        raise ConfigError("machine-specific data roots belong only in optional local configuration")
    prediction_columns = protocol.get("evaluation", {}).get("prediction_contract", {}).get("quantile_columns", [])
    expected_columns = [f"q{float(item):0.2f}" for item in quantiles]
    if prediction_columns != expected_columns:
        raise ConfigError("prediction quantile columns must correspond exactly to probability quantiles")
    if not features.get("policy", {}).get("identity_fields_forbidden") or not features["policy"].get("truth_fields_forbidden"):
        raise ConfigError("feature policy must forbid identity and truth fields")
    feature_names = [name for group in features.get("feature_groups", {}).values() for name in group]
    if len(feature_names) != len(set(feature_names)):
        raise ConfigError("formal feature groups must not duplicate columns")
    forbidden_features = set(feature_names) & (
        set(features["policy"]["identity_fields_forbidden"])
        | set(features["policy"]["truth_fields_forbidden"])
    )
    if forbidden_features:
        raise ConfigError(f"formal feature groups contain forbidden fields: {sorted(forbidden_features)}")
    registry = bundle["models"].get("registry", {})
    if not registry or len(registry) != len(set(registry)):
        raise ConfigError("model registry IDs must be unique")
    for group, ids in bundle["models"].get("groups", {}).items():
        unknown = set(ids) - set(registry)
        if unknown:
            raise ConfigError(f"model group {group} contains unregistered IDs: {sorted(unknown)}")
        if not ids or len(ids) != len(set(ids)):
            raise ConfigError(f"model group {group} must be nonempty and unique")
    search = bundle["models"].get("search", {})
    if set(search.get("budget", {})) & {"seed", "objective"}:
        raise ConfigError("tuning seed and objective have unique owners in protocol.yaml")
    trials = search.get("budget", {}).get("trials_per_model")
    if not isinstance(trials, int) or trials < 1:
        raise ConfigError("search budget must define a positive trials_per_model")
    spaces = search.get("spaces", {})
    for model_id, spec in registry.items():
        if not _tuning_enabled(spec):
            continue
        space_id = model_id if model_id in spaces else "deep_shared_prototype" if spec.get("family") == "deep" else None
        if space_id is None or len(spaces[space_id]) != trials:
            raise ConfigError(f"tuned model {model_id} must resolve to exactly {trials} registered candidates")
    sites = bundle["sites"]
    if not sites.get("registry") or not sites.get("sets"):
        raise ConfigError("site registry and named site sets must be declared")
    for set_id, members in sites["sets"].items():
        if not members or len(members) != len(set(members)) or set(members) - set(sites["registry"]):
            raise ConfigError(f"site set {set_id} must contain unique registered sites")
    for selector_id, selector in sites.get("selectors", {}).items():
        if selector.get("source_set") not in sites["sets"]:
            raise ConfigError(f"site selector {selector_id} has an unknown source set")
        regions = {str(sites["registry"][site_id]["region"]) for site_id in sites["sets"][selector["source_set"]]}
        for count in selector.get("allowed_counts", []):
            quota = selector.get("region_quotas", {}).get(count)
            if quota is None:
                raise ConfigError(f"site selector {selector_id} lacks a quota for {count}")
            quotas = {region: int(quota) for region in regions} if isinstance(quota, int) else {str(region): int(value) for region, value in quota.items()}
            if set(quotas) != regions or sum(quotas.values()) != int(count):
                raise ConfigError(f"site selector {selector_id} quotas do not sum to {count}")
    for profile_id, profile in bundle["experiments"].get("profiles", {}).items():
        if not _SAFE_ID.fullmatch(profile_id):
            raise ConfigError(f"unsafe experiment profile ID: {profile_id!r}")
        if profile.get("execution") not in protocol["execution_gate"]["execution_levels"]:
            raise ConfigError(f"profile {profile_id} has an invalid execution level")
        site_spec = profile.get("sites", {})
        if site_spec.get("value") is not None and site_spec["value"] not in sites["sets"]:
            raise ConfigError(f"profile {profile_id} has an unknown site set")
        if site_spec.get("selector") is not None and site_spec["selector"] not in sites.get("selectors", {}):
            raise ConfigError(f"profile {profile_id} has an unknown site selector")
        model_group = profile.get("models", {}).get("group")
        if model_group not in bundle["models"].get("groups", {}):
            raise ConfigError(f"profile {profile_id} has an unknown model group")
        if profile["execution"] == "official" and "locked_config_hash" not in profile:
            raise ConfigError(f"official profile {profile_id} must declare locked_config_hash")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def to_plain(value: Any) -> Any:
    """Return an ordinary dict/list tree suitable for YAML, JSON, and hashing."""
    return _plain(value)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _farthest_sites(registry: Mapping[str, Mapping[str, Any]], candidates: Sequence[str], count: int, region_quotas: Mapping[int, Any]) -> list[str]:
    """Deterministic region-balanced farthest-point selection; never list slicing."""
    if count < 1 or count > len(candidates):
        raise ConfigError(f"n_sites must be between 1 and {len(candidates)}")
    regions: dict[str, list[str]] = {}
    for site_id in candidates:
        regions.setdefault(str(registry[site_id]["region"]), []).append(site_id)
    registered_quota = region_quotas.get(count)
    if registered_quota is None:
        raise ConfigError(f"density selector does not define a regional quota for {count} sites")
    quotas = ({region: int(registered_quota) for region in regions} if isinstance(registered_quota, int) else {str(region): int(value) for region, value in registered_quota.items()})
    if set(quotas) != set(regions) or sum(quotas.values()) != count:
        raise ConfigError("density region quotas must cover every region and sum to n_sites")
    selected: list[str] = []
    for region, members in sorted(regions.items()):
        members = sorted(members)
        def dist2(a: str, b: str) -> float:
            return ((float(registry[a]["lat"]) - float(registry[b]["lat"])) ** 2 + (float(registry[a]["lon"]) - float(registry[b]["lon"])) ** 2)
        centroid = (sum(float(registry[item]["lat"]) for item in members) / len(members), sum(float(registry[item]["lon"]) for item in members) / len(members))
        first = max(members, key=lambda item: ((float(registry[item]["lat"]) - centroid[0]) ** 2 + (float(registry[item]["lon"]) - centroid[1]) ** 2, item))
        chosen = [first]
        while len(chosen) < quotas[region]:
            remaining = [item for item in members if item not in chosen]
            chosen.append(max(remaining, key=lambda item: (min(dist2(item, old) for old in chosen), item)))
        selected.extend(chosen)
    return selected


@dataclass(frozen=True)
class RunConfig:
    project_id: str
    protocol_version: str
    profile: str
    execution: str
    protocol: Mapping[str, Any]
    data: Mapping[str, Any]
    features: Mapping[str, Any]
    analysis: Mapping[str, Any]
    sites: Mapping[str, Any]
    models: Mapping[str, Any]
    selected_sites: tuple[str, ...]
    selected_models: tuple[str, ...]
    selected_site_records: Mapping[str, Any]
    selected_model_records: Mapping[str, Any]
    selected_search_spaces: Mapping[str, Any]
    scope: Mapping[str, Any]
    site_registry_hash: str
    model_registry_hash: str
    overrides: Mapping[str, Any]
    official_result_set: str | None
    locked_config_hash: str | None
    config_hash: str

    def as_dict(self) -> dict[str, Any]:
        return _plain({
            "project_id": self.project_id, "protocol_version": self.protocol_version,
            "profile": self.profile, "execution": self.execution,
            "protocol": self.protocol, "data": self.data, "features": self.features,
            "analysis": self.analysis,
            "sites": self.sites, "models": self.models, "selected_sites": self.selected_sites,
            "selected_models": self.selected_models,
            "selected_site_records": self.selected_site_records,
            "selected_model_records": self.selected_model_records,
            "selected_search_spaces": self.selected_search_spaces,
            "scope": self.scope,
            "overrides": self.overrides,
            "config_hash": self.config_hash,
        })


def resolve_config(profile: str | None, *, root: Path | None = None, models: str | Sequence[str] | None = None, sites: str | Sequence[str] | None = None, site_set: str | None = None, n_sites: int | None = None, gap_days: int | None = None, seed: int | None = None) -> RunConfig:
    """Resolve base configs, a profile, then development-only CLI overrides once."""
    root = root or project_root()
    bundle = load_bundle(str(root))
    profiles = bundle["experiments"]["profiles"]
    if profile not in profiles:
        raise ConfigError(f"unknown experiment profile: {profile}")
    chosen = profiles[profile]
    execution = chosen["execution"]
    raw_overrides = {key: value for key, value in {"models": models, "sites": sites, "site_set": site_set, "n_sites": n_sites, "gap_days": gap_days, "seed": seed}.items() if value is not None}
    if execution == "official" and raw_overrides:
        raise ConfigError("official runs prohibit CLI scientific overrides")
    if sum(value is not None for value in (sites, site_set, n_sites)) > 1:
        raise ConfigError("--sites, --site-set, and --n-sites cannot be combined")
    if n_sites is not None and int(n_sites) < 1:
        raise ConfigError("n_sites must be a positive integer")
    if seed is not None and int(seed) < 0:
        raise ConfigError("seed must be a nonnegative integer")
    site_cfg, registry = bundle["sites"], bundle["sites"]["registry"]
    if sites is not None:
        chosen_sites = tuple(part.strip() for part in (sites.split(",") if isinstance(sites, str) else sites) if part.strip())
    else:
        site_spec = dict(chosen["sites"])
        source_set = site_set or site_spec.get("value") or site_cfg["selectors"][site_spec["selector"]]["source_set"]
        if source_set not in site_cfg["sets"]:
            raise ConfigError(f"unknown site set: {source_set}")
        candidates = site_cfg["sets"][source_set]
        requested_count = n_sites if n_sites is not None else site_spec.get("n_sites")
        if requested_count is None:
            chosen_sites = tuple(candidates)
        else:
            selector = site_spec.get("selector")
            if selector is None:
                matches = [
                    selector_id for selector_id, selector_value in site_cfg["selectors"].items()
                    if selector_value.get("source_set") == source_set
                ]
                if len(matches) != 1:
                    raise ConfigError(f"site set {source_set} does not resolve to exactly one selector")
                selector = matches[0]
            selector_spec = site_cfg["selectors"].get(selector)
            if not selector_spec or requested_count not in selector_spec["allowed_counts"]:
                raise ConfigError(f"{selector} does not allow {requested_count} sites")
            chosen_sites = tuple(_farthest_sites(registry, candidates, int(requested_count), selector_spec["region_quotas"]))
    if not chosen_sites or len(chosen_sites) != len(set(chosen_sites)) or set(chosen_sites) - set(registry):
        raise ConfigError("selected sites must be a nonempty unique registered set")
    if models is not None:
        chosen_models = tuple(part.strip() for part in (models.split(",") if isinstance(models, str) else models) if part.strip())
    else:
        group = chosen["models"].get("group")
        chosen_models = tuple(bundle["models"]["groups"].get(group, ()))
    model_registry = bundle["models"]["registry"]
    if not chosen_models or len(chosen_models) != len(set(chosen_models)) or set(chosen_models) - set(model_registry):
        raise ConfigError("selected models must be a nonempty unique registered set")
    protocol = _plain(bundle["protocol"])
    if gap_days is not None:
        if int(gap_days) not in protocol["validation"]["gap_sensitivity_days"]:
            raise ConfigError("gap_days is not a registered sensitivity value")
        protocol["validation"]["gap_days"] = int(gap_days)
    else:
        protocol["validation"]["gap_days"] = protocol["validation"]["purge_hours"] // 24
    if seed is not None:
        protocol["seed_policy"]["run_seed"] = int(seed)
    selected_site_records = {item: _plain(registry[item]) for item in chosen_sites}
    selected_model_records = {item: _plain(model_registry[item]) for item in chosen_models}
    all_spaces = bundle["models"]["search"]["spaces"]
    selected_search_spaces = {
        item: _plain(all_spaces[item] if item in all_spaces else
                     all_spaces["deep_shared_prototype"])
        for item in chosen_models
        if _tuning_enabled(model_registry[item])
    }
    selected_models_config = {
        "registry": selected_model_records,
        "groups": {"selected": list(chosen_models)},
        "search": {"budget": _plain(bundle["models"]["search"]["budget"]),
                   "spaces": selected_search_spaces},
    }
    selector_id = (dict(chosen["sites"]).get("selector")
                   if n_sites is not None or dict(chosen["sites"]).get("n_sites") is not None
                   else None)
    resolved_site_set = (site_set or dict(chosen["sites"]).get("value")
                         if sites is None else None)
    selector_parameters: dict[str, Any] = {}
    if selector_id is not None:
        selector_record = _plain(site_cfg["selectors"][selector_id])
        selector_parameters = {
            "selector_id": selector_id,
            "method": selector_record["method"],
            "source_set": source_set,
            "allowed_counts": selector_record["allowed_counts"],
            "region_quotas": selector_record["region_quotas"],
            "requested_n_sites": len(chosen_sites),
            "status": selector_record["status"],
            "official_eligible": selector_record["official_eligible"],
        }
    payload = {
        "project_id": bundle["manifest"]["project_id"],
        "protocol_version": bundle["manifest"]["protocol_version"],
        "profile": profile,
        "execution": execution,
        "protocol": protocol,
        "data": _plain(bundle["data"]),
        "features": _plain(bundle["features"]["build"]),
        "analysis": _plain(bundle["features"]["analysis"]),
        "sites": {"registry": selected_site_records},
        "models": selected_models_config,
        "selected_sites": chosen_sites,
        "selected_models": chosen_models,
        "selected_site_records": selected_site_records,
        "selected_model_records": selected_model_records,
        "selected_search_spaces": selected_search_spaces,
        "scope": {
            "site_set": resolved_site_set,
            "selector": selector_id,
            "selector_parameters": selector_parameters,
            "model_group": dict(chosen["models"]).get("group"),
            "spatial": bool(chosen.get("spatial", False)),
        },
        "overrides": _plain(raw_overrides),
    }
    registry_hashes = {
        "site_registry_hash": content_hash(_plain(bundle["sites"]["registry"]), length=64),
        "model_registry_hash": content_hash(_plain(bundle["models"]["registry"]), length=64),
    }
    return RunConfig(
        config_hash=content_hash(payload),
        official_result_set=bundle["manifest"]["official_result_set"],
        locked_config_hash=chosen.get("locked_config_hash"),
        **_freeze(payload), **registry_hashes,
    )


def assert_official_ready(config: RunConfig, *, readiness_passed: bool = False) -> None:
    if config.execution != "official":
        return
    gate = config.protocol["execution_gate"]["official"]
    if gate["required_execution_level"] != config.execution:
        raise ConfigError("official profile does not satisfy the required execution level")
    if config.overrides:
        raise ConfigError("official runs must not contain CLI scientific overrides")
    if not config.locked_config_hash or config.locked_config_hash != config.config_hash:
        raise ConfigError("official run requires a profile with its exact locked_config_hash")
    required_status = gate["required_implementation_level"]
    invalid_models = [
        model_id for model_id in config.selected_models
        if config.models["registry"][model_id]["implementation_status"] != required_status
        or not config.models["registry"][model_id]["official_eligible"]
    ]
    if invalid_models:
        raise ConfigError(f"official run contains models without validated official eligibility: {invalid_models}")
    if not readiness_passed:
        raise ConfigError("official run requires an explicit readiness pass from the validation stage")
    if not config.official_result_set:
        raise ConfigError("official run is blocked until the official result set and readiness evidence are registered")
