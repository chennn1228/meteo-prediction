"""Deep architecture factory, execution gates, and uniform model adapter."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from nwp.models.base import BaseModel, ModelError


def require_deep_execution(model_id: str, implementation_level: str,
                           execution_level: str,
                           model_config: Mapping[str, Any],
                           protocol_config: Mapping[str, Any]) -> None:
    levels = protocol_config["execution_gate"]["execution_levels"]
    if implementation_level not in ("prototype", "validated"):
        raise ValueError("implementation_level must be prototype or validated")
    if execution_level not in levels:
        raise ValueError(
            "execution_level must be one of the registered execution levels")
    try:
        entry = model_config["registry"][model_id]
    except KeyError as exc:
        raise ModelError(f"unregistered deep model: {model_id}") from exc
    if (entry["family"] == "experimental_constrained"
            and execution_level == "official"):
        raise PermissionError(
            "PINN physical-unit constraint and ablations remain unresolved")
    if (implementation_level == "validated"
            and entry["implementation_status"] != "validated"):
        raise PermissionError(
            f"{model_id} has no completed validated architecture audit")
    if execution_level == "official" and implementation_level != "validated":
        raise PermissionError(
            "official execution requires a validated implementation")
    if execution_level == "official" and not entry["official_eligible"]:
        raise PermissionError(f"{model_id} is not eligible for official execution")


class DeepModel(BaseModel):
    """Sequence model requiring an explicit, disjoint early-stop block."""

    def __init__(self, model_id: str, implementation: str,
                 model_config: Mapping[str, Any],
                 protocol_config: Mapping[str, Any], *,
                 params: Mapping[str, Any] | None = None) -> None:
        self.model_id = model_id
        self.implementation = implementation
        self.model_config = model_config
        self.protocol_config = protocol_config
        self.params = dict(params or {})
        self.network = None
        self.receipt: dict[str, Any] | None = None

    def fit(self, features: Any, target: Any | None = None) -> "DeepModel":
        if not isinstance(features, Mapping) or not isinstance(target, Mapping):
            raise ModelError(
                "deep fit requires {'fit', 'early_stop'} feature and target mappings")
        fit_x = np.asarray(features["fit"], dtype=np.float32)
        early_x = np.asarray(features["early_stop"], dtype=np.float32)
        fit_y = np.asarray(target["fit"], dtype=np.float32)
        early_y = np.asarray(target["early_stop"], dtype=np.float32)
        if (fit_x.ndim != 3 or early_x.ndim != 3
                or fit_x.shape[1:] != early_x.shape[1:]):
            raise ModelError("deep input must share [sequence, feature] dimensions")
        from .architectures import build_deep_model
        from .training import fit_quantile_model

        quantiles = tuple(float(value) for value in
                          self.protocol_config["probability"]["quantiles"])
        self.network = build_deep_model(
            self.implementation, fit_x.shape[2], seq_len=fit_x.shape[1],
            out_dim=len(quantiles))
        self.network, self.receipt = fit_quantile_model(
            self.network, fit_x, fit_y, early_x, early_y,
            quantiles=quantiles,
            learning_rate=float(self.params.get("learning_rate", 1e-3)),
            max_epochs=int(self.params.get("max_epochs", 15)),
            batch_size=int(self.params.get("batch_size", 1024)),
            patience=int(self.params.get("patience", 4)),
            seed=int(self.protocol_config["seed_policy"]["tuning_seed"]),
            device=str(self.params.get("device", "cpu")))
        return self

    def predict(self, features: Any) -> np.ndarray:
        prediction = self.predict_quantiles(
            features,
            tuple(float(value) for value in
                  self.protocol_config["probability"]["quantiles"]))
        return prediction[:, prediction.shape[1] // 2]

    def predict_quantiles(self, features: Any,
                          quantiles: tuple[float, ...]) -> np.ndarray:
        registered = tuple(float(value) for value in
                           self.protocol_config["probability"]["quantiles"])
        if quantiles != registered:
            raise ModelError("deep model emits only the registered quantile grid")
        if self.network is None:
            raise ModelError("model must be fit before predict")
        from .training import predict_quantiles
        return predict_quantiles(
            self.network, np.asarray(features, dtype=np.float32),
            device=str(self.params.get("device", "cpu")))


__all__ = ["DeepModel", "require_deep_execution"]
