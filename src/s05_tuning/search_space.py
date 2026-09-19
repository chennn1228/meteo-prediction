"""Read six pre-registered parameter candidates from the protocol manifest."""
from __future__ import annotations

from s04_splits.split_registry import manifest


def candidates(model_id: str) -> tuple[dict, ...]:
    config = manifest()
    registry = {entry["id"]: entry for entry in config["models"]}
    if model_id == "pinn":
        raise ValueError("PINN remains experimental; no official tuning space")
    if model_id not in registry:
        raise KeyError(f"Unregistered model: {model_id}")
    key = "deep_shared_prototype" if registry[model_id]["family"] == "deep" else model_id
    space = config["tuning_search_spaces"]
    if key not in space:
        raise KeyError(f"No pre-registered six-candidate search for {model_id}")
    result = tuple(dict(item) for item in space[key])
    expected = int(config["tuning_budget"]["trials_per_model"])
    if len(result) != expected or len({repr(sorted(item.items())) for item in result}) != expected:
        raise ValueError(f"{model_id}: candidate search must have {expected} distinct settings")
    return result
