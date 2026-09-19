"""Execution and result-state rules shared by all formal stages."""
from __future__ import annotations

from .config_loader import ProtocolError, load_manifest


def validate_run_levels(implementation_level: str, execution_level: str) -> None:
    manifest = load_manifest()
    if implementation_level not in manifest["implementation_levels"]:
        raise ProtocolError(f"unknown implementation level: {implementation_level}")
    if execution_level not in manifest["execution_levels"]:
        raise ProtocolError(f"unknown execution level: {execution_level}")
    if execution_level == "official" and implementation_level != "validated":
        raise ProtocolError("official execution requires a validated implementation")


def validate_result_status(
    implementation_level: str,
    execution_level: str,
    result_status: str,
) -> None:
    validate_run_levels(implementation_level, execution_level)
    if result_status not in {"smoke", "development", "provisional", "official", "legacy"}:
        raise ProtocolError(f"unknown result status: {result_status}")
    if result_status == "official" and (implementation_level, execution_level) != (
        "validated", "official"
    ):
        raise ProtocolError("official results require validated + official")
