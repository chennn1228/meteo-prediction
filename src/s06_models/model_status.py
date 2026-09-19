"""Prevent prototype or unaudited deep implementations from official runs."""
from __future__ import annotations

# Architecture smoke tests do not establish standard-method equivalence.
# Models enter this registry only after mechanism, shape, loss, scale, early
# stopping and reference-behaviour audits are recorded in the migration audit.
VALIDATED_DEEP_MODELS: frozenset[str] = frozenset()
EXPERIMENTAL_MODELS: frozenset[str] = frozenset({"pinn"})


def require_deep_execution(model_id: str, implementation_level: str,
                           execution_level: str) -> None:
    if implementation_level not in {"prototype", "validated"}:
        raise ValueError("implementation_level must be prototype or validated")
    if execution_level not in {"smoke", "development", "official"}:
        raise ValueError("execution_level must be smoke, development or official")
    if model_id in EXPERIMENTAL_MODELS and execution_level == "official":
        raise PermissionError("PINN physical-unit constraint and ablations remain unresolved")
    if implementation_level == "validated" and model_id not in VALIDATED_DEEP_MODELS:
        raise PermissionError(f"{model_id} has no completed validated architecture audit")
    if execution_level == "official" and implementation_level != "validated":
        raise PermissionError("official execution requires a validated implementation")
