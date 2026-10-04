from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from nwp.core.artifacts import ArtifactRecord, ArtifactResolver
from nwp.core.config import ConfigError, assert_official_ready, load_bundle, project_root, resolve_config, to_plain, _validate_bundle
from nwp.core.dependencies import aggregate_dependency_fingerprint, model_dependency_fingerprint
from nwp.core.lifecycle import require_writable, transition, write_initial_state
from nwp.core.fingerprints import sha256_file, stable_object_hash
from nwp.core.paths import RunPaths
from nwp.core.context import RunContext
from nwp.core.schema import ContractError, assert_model_features


def test_unknown_root_key_and_unsafe_ids_fail_closed():
    bundle = to_plain(load_bundle())
    bundle["data"]["unknown"] = True
    with pytest.raises(ConfigError, match="top-level keys"):
        _validate_bundle(bundle)
    bundle = to_plain(load_bundle())
    bundle["sites"]["registry"]["../escape"] = bundle["sites"]["registry"]["nanjing_1"]
    with pytest.raises(ConfigError, match="unsafe site ID"):
        _validate_bundle(bundle)


def test_unselected_records_do_not_change_model_local_dependency():
    config = resolve_config("nanjing_cpu_diagnostic", models="ridge_mos")
    before = model_dependency_fingerprint(
        "tuning", "ridge_mos", upstream={"features": "a" * 64},
        model_record=config.selected_model_records["ridge_mos"],
        search_space=config.selected_search_spaces["ridge_mos"])
    unrelated = {"xgboost": {"new": "candidate"}}
    after = model_dependency_fingerprint(
        "tuning", "ridge_mos", upstream={"features": "a" * 64},
        model_record=config.selected_model_records["ridge_mos"],
        search_space=config.selected_search_spaces["ridge_mos"])
    assert before == after
    assert before != model_dependency_fingerprint(
        "tuning", "ridge_mos", upstream={"features": "a" * 64},
        model_record={**to_plain(config.selected_model_records["ridge_mos"]), **unrelated},
        search_space=config.selected_search_spaces["ridge_mos"])


def test_selected_xgboost_change_is_local_but_aggregate_changes():
    base = {"ridge": "a" * 64, "lgbm": "b" * 64, "xgboost": "c" * 64}
    changed = {**base, "xgboost": "d" * 64}
    assert base["ridge"] == changed["ridge"] and base["lgbm"] == changed["lgbm"]
    assert aggregate_dependency_fingerprint("comparison", base, contract={}) != aggregate_dependency_fingerprint("comparison", changed, contract={})


@pytest.mark.parametrize("state", ["BLOCKED", "FAILED", "COMPLETE", "FROZEN", "IMPORTED"])
def test_terminal_lifecycle_states_are_read_only(tmp_path: Path, state: str):
    path = tmp_path / "status.json"
    write_initial_state(path, state)
    with pytest.raises(ContractError, match="read-only"):
        require_writable(path)
    with pytest.raises(ContractError, match="illegal"):
        transition(path, "RUNNING")


def test_running_is_writable_and_success_receipt_is_write_once(tmp_path: Path):
    path = tmp_path / "status.json"
    write_initial_state(path)
    transition(path, "RUNNING")
    assert require_writable(path) == "RUNNING"


def test_final_test_artifact_cannot_enter_selection():
    values = dict(
        artifact_id="a", artifact_type="model", stage="tuning", model_id="ridge",
        fold_id="outer_1", scope={}, data_scope={}, time_scope="final_test",
        split_scope={}, dependency_fingerprint="a" * 64,
        implementation_fingerprint="b" * 64, environment_fingerprint="c" * 64,
        sha256="d" * 64, path="x", portable=True, execution_level="development",
        result_status="diagnostic", official_reuse_eligible=False,
        created_at="2026-10-04T00:00:00+00:00")
    with pytest.raises(ContractError, match="final-test"):
        ArtifactRecord(**values)


def test_identity_fields_are_never_model_features():
    policy = {"identity_fields_forbidden": ["location_id"], "truth_fields_forbidden": ["y"]}
    for field in ("site_id", "station_id", "target", "truth", "row_id",
                  "requested_latitude", "location_id", "y"):
        with pytest.raises(ContractError):
            assert_model_features([field], policy)


def test_official_null_result_and_unvalidated_models_remain_blocked():
    config = resolve_config("official_cpu_20site")
    with pytest.raises(ConfigError):
        assert_official_ready(config, readiness_passed=True)
    development = resolve_config("nanjing_cpu_diagnostic", models="raw_gfs")
    eligible = replace(development, execution="official", overrides=MappingProxyType({}),
                       locked_config_hash=development.config_hash,
                       official_result_set="synthetic")
    assert_official_ready(eligible, readiness_passed=True)


def _artifact_record(path: Path, *, digest: str, implementation: str = "b" * 64,
                     environment: str = "c" * 64, execution: str = "development",
                     official_eligible: bool = False, scope=None) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id="ridge-outer-1", artifact_type="model", stage="fitting",
        model_id="ridge_mos", fold_id="outer_1", scope=scope or {},
        data_scope={"sites": ["nanjing_1"]}, time_scope="outer_validation",
        split_scope={"fold": "outer_1"}, dependency_fingerprint="a" * 64,
        implementation_fingerprint=implementation,
        environment_fingerprint=environment, sha256=digest,
        path=path.as_posix(), portable=True, execution_level=execution,
        result_status="diagnostic", official_reuse_eligible=official_eligible,
        created_at="2026-10-04T00:00:00+00:00")


def test_artifact_resolver_rejects_byte_implementation_environment_and_official_mismatch(tmp_path: Path):
    config = resolve_config("nanjing_cpu_diagnostic", models="ridge_mos")
    source = RunContext.create(project_root(), config, data_root=tmp_path / "data",
                               outputs_root=tmp_path / "outputs", run_id="source")
    artifact = source.paths.model_fold("ridge_mos", "outer_1") / "model.bin"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"model-v1")
    relative = artifact.relative_to(source.paths.run_root)
    record = _artifact_record(relative, digest=sha256_file(artifact))
    source.artifact_resolver.append_record(record)
    resolver = ArtifactResolver(source.paths)
    query = dict(artifact_type="model", dependency_fingerprint="a" * 64,
                 implementation_fingerprint="b" * 64,
                 data_scope={"sites": ["nanjing_1"]}, time_scope="outer_validation",
                 split_scope={"fold": "outer_1"}, execution_level="development",
                 environment_fingerprint="c" * 64)
    assert resolver.find_compatible_artifact(**query)[1] == record
    assert resolver.find_compatible_artifact(**{**query, "implementation_fingerprint": "d" * 64}) is None
    assert resolver.find_compatible_artifact(**{**query, "environment_fingerprint": "e" * 64}) is None
    assert resolver.find_compatible_artifact(**{**query, "execution_level": "official"}) is None
    artifact.write_bytes(b"model-v2")
    assert resolver.find_compatible_artifact(**query) is None


def test_bounded_three_model_child_run_dependency_reuse_matrix():
    models = {
        "ridge_mos": ({"alpha": 1}, [{"alpha": 1}]),
        "lgbm": ({"leaves": 15}, [{"leaves": 15}]),
        "xgboost": ({"depth": 3}, [{"depth": 3}]),
    }
    upstream = {"features": "f" * 64}
    run_a = {name: model_dependency_fingerprint(
        "tuning", name, upstream=upstream, model_record=record, search_space=space)
        for name, (record, space) in models.items()}
    changed = {**models, "xgboost": (models["xgboost"][0], [{"depth": 6}])}
    run_b = {name: model_dependency_fingerprint(
        "tuning", name, upstream=upstream, model_record=record, search_space=space)
        for name, (record, space) in changed.items()}
    assert run_a["ridge_mos"] == run_b["ridge_mos"]
    assert run_a["lgbm"] == run_b["lgbm"]
    assert run_a["xgboost"] != run_b["xgboost"]
    assert aggregate_dependency_fingerprint("comparison", run_a, contract={}) != aggregate_dependency_fingerprint("comparison", run_b, contract={})


def test_feature_and_visualization_fingerprint_invalidation_boundaries():
    base = {"data": "a" * 64, "features": "b" * 64, "model": "c" * 64,
            "visualization": "d" * 64}
    feature_changed = {**base, "features": "e" * 64}
    visual_changed = {**base, "visualization": "f" * 64}
    feature_downstream = lambda item: stable_object_hash({
        "features": item["features"], "model": item["model"]})
    figure = lambda item: stable_object_hash({
        "training": feature_downstream(item), "visualization": item["visualization"]})
    assert feature_downstream(base) != feature_downstream(feature_changed)
    assert feature_downstream(base) == feature_downstream(visual_changed)
    assert figure(base) != figure(visual_changed)


def test_child_run_provenance_records_parent_and_changed_dependencies(tmp_path: Path):
    config = resolve_config("nanjing_cpu_diagnostic", models="raw_gfs")
    child = RunContext.create(
        project_root(), config, data_root=tmp_path / "data",
        outputs_root=tmp_path / "outputs", run_id="child",
        parent_run_id="parent", change_reason="bounded synthetic dependency test",
        changed_dependencies=["xgboost.search_space"])
    assert child.provenance["parent_run_id"] == "parent"
    assert child.provenance["changed_dependencies"] == ["xgboost.search_space"]
