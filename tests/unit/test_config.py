from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest

from nwp.core.config import ConfigError, assert_official_ready, load_bundle, load_local_paths, resolve_config
from nwp.cli import main


def test_manifest_is_an_entry_point_to_exactly_six_configs():
    bundle = load_bundle()
    assert set(bundle["manifest"]["configs"]) == {"protocol", "data", "features", "sites", "models", "experiments"}
    assert bundle["data"]["forecast"]["variables"]
    assert list(bundle["protocol"]["probability"]["quantiles"]) == sorted(bundle["protocol"]["probability"]["quantiles"])
    assert bundle["manifest"]["configs"] == {
        "protocol": "config/protocol.yaml",
        "data": "config/data.yaml",
        "features": "config/features.yaml",
        "sites": "config/sites.yaml",
        "models": "config/models.yaml",
        "experiments": "config/experiments.yaml",
    }


def test_loaded_bundle_and_resolved_config_are_deeply_immutable():
    bundle = load_bundle()
    with pytest.raises(TypeError):
        bundle["protocol"]["target"] = "changed"
    config = resolve_config("nanjing_cpu_diagnostic")
    with pytest.raises(TypeError):
        config.protocol["validation"]["purge_hours"] = 0
    assert isinstance(config.selected_sites, tuple)
    assert isinstance(config.selected_models, tuple)


def test_profile_then_cli_overrides_are_resolved_once():
    config = resolve_config("cpu_20site", models="raw_gfs,ridge_mos", n_sites=10, gap_days=7, seed=11)
    assert config.selected_models == ("raw_gfs", "ridge_mos")
    assert len(config.selected_sites) == 10
    assert config.protocol["validation"]["gap_days"] == 7
    assert config.protocol["seed_policy"]["run_seed"] == 11
    assert config.overrides["n_sites"] == 10
    assert config.config_hash == resolve_config("cpu_20site", models="raw_gfs,ridge_mos", n_sites=10, gap_days=7, seed=11).config_hash
    assert config.config_hash != resolve_config("cpu_20site", models="raw_gfs,ridge_mos", n_sites=10, gap_days=14, seed=11).config_hash


def test_site_override_forms_cannot_conflict():
    with pytest.raises(ConfigError, match="cannot be combined"):
        resolve_config("cpu_20site", sites="nanjing_1", n_sites=1)
    with pytest.raises(ConfigError, match="cannot be combined"):
        resolve_config("cpu_20site", sites="nanjing_1", site_set="training_20")
    selected = resolve_config("nanjing_cpu_diagnostic", site_set="training_20", n_sites=5)
    assert len(selected.selected_sites) == 5


def test_density_selector_is_deterministic_and_not_list_slicing():
    first = resolve_config("spatial_density")
    second = resolve_config("spatial_density")
    configured_order = tuple(load_bundle()["sites"]["sets"]["training_20"][:10])
    assert first.selected_sites == second.selected_sites
    assert first.selected_sites != configured_order


def test_official_cli_scientific_override_and_unlocked_profile_are_rejected():
    with pytest.raises(ConfigError, match="prohibit CLI"):
        resolve_config("official_cpu_20site", models="raw_gfs")
    with pytest.raises(ConfigError, match="locked_config_hash"):
        assert_official_ready(resolve_config("official_cpu_20site"))
    with pytest.raises(SystemExit):
        main(["run", "--profile", "official_cpu_20site", "--to-stage", "selection"])


def test_official_gate_requires_validated_models_readiness_and_result_registration():
    official = resolve_config("official_cpu_20site")
    locked = replace(official, locked_config_hash=official.config_hash, official_result_set="candidate")
    with pytest.raises(ConfigError, match="models without validated"):
        assert_official_ready(locked, readiness_passed=True)

    raw_only = resolve_config("nanjing_cpu_diagnostic", models="raw_gfs")
    gated = replace(
        raw_only,
        execution="official",
        overrides=MappingProxyType({}),
        locked_config_hash=raw_only.config_hash,
        official_result_set="candidate",
    )
    with pytest.raises(ConfigError, match="readiness pass"):
        assert_official_ready(gated)
    assert_official_ready(gated, readiness_passed=True)


def test_duplicate_manifest_keys_and_noncanonical_config_paths_fail_closed(tmp_path):
    (tmp_path / "project_manifest.yaml").write_text(
        "project_id: one\nproject_id: two\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="duplicate YAML key"):
        load_bundle(str(tmp_path))

    other = tmp_path / "other"
    other.mkdir()
    (other / "project_manifest.yaml").write_text(
        """project_id: p
protocol_version: v
official_result_set: null
configs:
  protocol: protocol.yaml
  data: config/data.yaml
  features: config/features.yaml
  sites: config/sites.yaml
  models: config/models.yaml
  experiments: config/experiments.yaml
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="configuration paths must be exactly"):
        load_bundle(str(other))


def test_local_config_controls_only_machine_paths(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "local.yaml").write_text(
        "paths:\n  data_root: local-data\n  outputs_root: local-outputs\n",
        encoding="utf-8",
    )
    paths = load_local_paths(tmp_path)
    assert paths["data_root"] == (tmp_path / "local-data").resolve()
    assert paths["outputs_root"] == (tmp_path / "local-outputs").resolve()
    (config_dir / "local.yaml").write_text(
        "paths:\n  data_root: .\n  outputs_root: local-outputs\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="must not be the project root"):
        load_local_paths(tmp_path)
