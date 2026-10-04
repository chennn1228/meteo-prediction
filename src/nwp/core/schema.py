"""Immutable records and scientific boundary checks shared by all stages."""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence


class ContractError(ValueError):
    """A configuration or artifact violates a project contract."""


def assert_model_features(columns: Sequence[str], feature_policy: Mapping[str, Any]) -> None:
    if not columns or len(columns) != len(set(columns)):
        raise ContractError("model inputs must be a nonempty unique sequence")
    forbidden = set(columns) & set(feature_policy["identity_fields_forbidden"])
    forbidden |= set(columns) & set(feature_policy["truth_fields_forbidden"])
    forbidden |= {
        name
        for name in columns
        if name.startswith("station_")
        or name.endswith("_target_encoded")
        or name.endswith("_obs")
    }
    forbidden |= set(columns) & {
        "site_id", "target", "truth", "row_id", "requested_latitude",
        "requested_longitude", "requested_coordinates",
    }
    forbidden |= {name for name in columns if name.startswith("requested_")}
    if forbidden:
        raise ContractError(f"identity or truth fields cannot enter model inputs: {sorted(forbidden)}")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _instant(value: str, label: str) -> dt.datetime:
    try:
        instant = dt.datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} must be ISO-8601") from exc
    if instant.tzinfo is None:
        raise ContractError(f"{label} must include a timezone")
    return instant


@dataclass(frozen=True)
class StageResult:
    status: str
    artifact_outputs: Mapping[str, str]
    metadata_outputs: Mapping[str, Any]
    inputs: Mapping[str, Any]
    dependency_fingerprint: str
    implementation_fingerprint: str
    input_hashes: Mapping[str, str]
    output_hashes: Mapping[str, str]
    started_at: str
    finished_at: str
    message: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"success", "blocked", "failed", "reused", "skipped"}:
            raise ContractError(f"invalid stage status: {self.status}")
        for label, digest in (("dependency_fingerprint", self.dependency_fingerprint),
                              ("implementation_fingerprint", self.implementation_fingerprint)):
            if not re.fullmatch(r"[0-9a-f]{8,64}", digest):
                raise ContractError(f"stage {label} must be lowercase hex")
        if any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or not re.fullmatch(r"[0-9a-f]{8,64}", value)
            for key, value in self.input_hashes.items()
        ):
            raise ContractError("stage input_hashes must contain named lowercase hex hashes")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            or not re.fullmatch(r"[0-9a-f]{8,64}", value)
            for key, value in self.output_hashes.items()
        ):
            raise ContractError("stage output_hashes must contain actual lowercase hex hashes")
        if any(not isinstance(value, str) for value in self.artifact_outputs.values()):
            raise ContractError("artifact_outputs may contain only file or directory paths")
        started = _instant(self.started_at, "started_at")
        finished = _instant(self.finished_at, "finished_at")
        if finished < started:
            raise ContractError("stage finished_at precedes started_at")
        object.__setattr__(self, "inputs", _freeze(self.inputs))
        object.__setattr__(self, "artifact_outputs", _freeze(self.artifact_outputs))
        object.__setattr__(self, "metadata_outputs", _freeze(self.metadata_outputs))
        object.__setattr__(self, "input_hashes", _freeze(self.input_hashes))
        object.__setattr__(self, "output_hashes", _freeze(self.output_hashes))

    @property
    def outputs(self) -> Mapping[str, Any]:
        return _freeze({**_plain(self.artifact_outputs), **_plain(self.metadata_outputs)})

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "artifact_outputs": _plain(self.artifact_outputs),
            "metadata_outputs": _plain(self.metadata_outputs),
            "inputs": _plain(self.inputs),
            "dependency_fingerprint": self.dependency_fingerprint,
            "implementation_fingerprint": self.implementation_fingerprint,
            "input_hashes": _plain(self.input_hashes),
            "output_hashes": _plain(self.output_hashes),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "message": self.message,
        }
