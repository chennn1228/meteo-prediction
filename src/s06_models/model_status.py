"""Manifest-backed implementation gates; do not couple CPU to deep audits."""
from __future__ import annotations

from s01_core.config_loader import load_manifest
from s01_core.registry import assert_formal_model, model


def require_deep_execution(model_id: str, implementation_level: str,
                           execution_level: str) -> None:
    manifest = load_manifest()
    if implementation_level not in manifest["implementation_levels"]:
        raise ValueError("implementation_level must be prototype or validated")
    if execution_level not in manifest["execution_levels"]:
        raise ValueError("execution_level must be smoke, development or official")
    item = model(model_id)
    if item["family"] == "experimental_constrained" and execution_level == "official":
        raise PermissionError("PINN physical-unit constraint and ablations remain unresolved")
    if implementation_level == "validated" and item["implementation_status"] != "validated":
        raise PermissionError(f"{model_id} has no completed validated architecture audit")
    if execution_level == "official" and implementation_level != "validated":
        raise PermissionError("official execution requires a validated implementation")
    if execution_level == "official":
        try:
            assert_formal_model(model_id)
        except ValueError as exc:
            raise PermissionError(str(exc)) from exc
