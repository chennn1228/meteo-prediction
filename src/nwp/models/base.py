"""Mandatory model interface for raw, statistical, tree, and deep models."""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class ModelError(RuntimeError):
    pass


class BaseModel(ABC):
    model_id: str

    @abstractmethod
    def fit(self, features: Any, target: Any | None = None) -> "BaseModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, features: Any) -> np.ndarray:
        raise NotImplementedError

    def predict_quantiles(self, features: Any, quantiles: tuple[float, ...]) -> np.ndarray:
        point = self.predict(features)
        return np.column_stack([point for _ in quantiles])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(self, stream)

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        with path.open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, BaseModel):
            raise ModelError("serialized object does not implement BaseModel")
        return model
