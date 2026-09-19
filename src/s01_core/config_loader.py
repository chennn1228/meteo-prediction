"""Load the single active scientific protocol; reject duplicate YAML keys."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


class ProtocolError(ValueError):
    """A manifest violates an active scientific protocol invariant."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ProtocolError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def project_root() -> Path:
    return next(parent for parent in Path(__file__).resolve().parents
                if (parent / "project_manifest.yaml").is_file())


@lru_cache(maxsize=4)
def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Read and check protocol-wide invariants without running research work.

    Callers must not duplicate protocol values in their own constants. Tests may
    pass a temporary path to probe malformed manifests.
    """
    manifest_path = Path(path) if path is not None else project_root() / "project_manifest.yaml"
    config = yaml.load(manifest_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    if not isinstance(config, dict):
        raise ProtocolError("project manifest must be a mapping")
    if config.get("primary_target") != "ghi":
        raise ProtocolError("GHI is the only formal primary target")
    quantiles = config.get("quantiles")
    if quantiles != [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
        raise ProtocolError("seven quantiles must match the registered contract")
    validation = config.get("validation_protocol", {})
    if validation.get("purge_hours") != (
        validation.get("maximum_sequence_lookback_hours", -1)
        + validation.get("maximum_lead_hours", -1)
    ):
        raise ProtocolError("purge must cover sequence lookback and maximum lead")
    if config.get("selection_metric") != "mean_pinball":
        raise ProtocolError("formal model selection must use mean_pinball")
    if config.get("tuning_budget", {}).get("trials_per_model") != 6:
        raise ProtocolError("six candidates per model are required")
    if config.get("daylight_definition", {}).get("formal") != "solar_elevation_gt_0":
        raise ProtocolError("formal daylight definition must be solar_elevation_gt_0")
    model_ids = [model["id"] for model in config.get("models", [])]
    if len(model_ids) != len(set(model_ids)):
        raise ProtocolError("model IDs must be unique")
    if config.get("official_result_set") is not None and config.get("status") != "official":
        raise ProtocolError("an official result set cannot have provisional status")
    return config


def load_data_config(name: str, root: Path | None = None) -> dict[str, Any]:
    """Load data-only site or variable configuration, not protocol settings."""
    if name not in {"01_sites.yaml", "02_variables.yaml"}:
        raise ProtocolError(f"not a data-only configuration: {name}")
    path = (root or project_root()) / "config" / name
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    if not isinstance(data, dict):
        raise ProtocolError(f"invalid data configuration: {name}")
    return data
