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
    inputs: Mapping[str, Any]
    outputs: Mapping[str, Any]
    config_hash: str
    input_hashes: Mapping[str, str]
    started_at: str
    finished_at: str
    message: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"success", "blocked", "failed", "reused", "skipped"}:
            raise ContractError(f"invalid stage status: {self.status}")
        if not re.fullmatch(r"[0-9a-f]{8,64}", self.config_hash):
            raise ContractError("stage config_hash must be lowercase hex")
        if any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or not re.fullmatch(r"[0-9a-f]{8,64}", value)
            for key, value in self.input_hashes.items()
        ):
            raise ContractError("stage input_hashes must contain named lowercase hex hashes")
        started = _instant(self.started_at, "started_at")
        finished = _instant(self.finished_at, "finished_at")
        if finished < started:
            raise ContractError("stage finished_at precedes started_at")
        object.__setattr__(self, "inputs", _freeze(self.inputs))
        object.__setattr__(self, "outputs", _freeze(self.outputs))
        object.__setattr__(self, "input_hashes", _freeze(self.input_hashes))

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "inputs": _plain(self.inputs),
            "outputs": _plain(self.outputs),
            "config_hash": self.config_hash,
            "input_hashes": _plain(self.input_hashes),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "message": self.message,
        }
