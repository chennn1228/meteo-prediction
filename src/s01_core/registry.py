"""Semantic model registry; historical vN labels are provenance only."""
from __future__ import annotations

from .config_loader import ProtocolError, load_manifest


def model(model_id: str) -> dict:
    matches = [item for item in load_manifest()["models"] if item["id"] == model_id]
    if len(matches) != 1:
        raise ProtocolError(f"model is not registered exactly once: {model_id}")
    return matches[0]


def assert_formal_model(model_id: str) -> None:
    item = model(model_id)
    if item.get("family") == "experimental_constrained" or item.get("formal_eligible") is False:
        raise ProtocolError(f"model is not eligible for official comparison: {model_id}")
    if item.get("implementation_level") != "validated":
        raise ProtocolError(f"validated implementation not established: {model_id}")
