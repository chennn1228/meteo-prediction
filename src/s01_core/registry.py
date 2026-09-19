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
    if not item["official_eligible"]:
        raise ProtocolError(f"model is not eligible for official comparison: {model_id}")
    if item["implementation_status"] != "validated":
        raise ProtocolError(f"validated implementation not established: {model_id}")


def assert_declared_experiment(model_ids: list[str] | tuple[str, ...], *, device: str) -> None:
    """Check only models selected for this experiment; CPU ignores DL status."""
    if not model_ids or len(model_ids) != len(set(model_ids)):
        raise ProtocolError("experiment needs a nonempty unique model set")
    for model_id in model_ids:
        item = model(model_id)
        if item["execution_device"] != device:
            raise ProtocolError(f"{model_id} is not a {device} model")
        assert_formal_model(model_id)
        if item["tuning_required"] and model_id not in load_manifest()["tuning_search_spaces"]:
            raise ProtocolError(f"tuned model lacks registered candidates: {model_id}")
