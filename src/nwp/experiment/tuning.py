"""Generic, auditable, equal-budget nested-fold candidate selection."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from nwp.evaluation.metrics import mean_pinball
from nwp.splits.rolling import InnerFold, assert_gap, purge_days


@dataclass(frozen=True)
class Trial:
    model: str
    candidate: int
    parameters: dict[str, Any]
    outer: str
    inner: str
    seed: int
    fit_rows: int
    early_stop_rows: int
    scoring_rows: int
    mean_pinball: float | None
    wall_seconds: float
    device: str
    peak_memory_bytes: int | None
    epoch: int | None
    status: str
    error_reason: str | None


class IncompleteTrialsError(RuntimeError):
    def __init__(self, message: str, ledger: list[Trial]):
        super().__init__(message)
        self.ledger = ledger


def as_frame(trials: Sequence[Trial]) -> pd.DataFrame:
    records = []
    for trial in trials:
        record = asdict(trial)
        record["parameters"] = json.dumps(record["parameters"], sort_keys=True)
        records.append(record)
    return pd.DataFrame.from_records(records)


def write_ledger(trials: Sequence[Trial], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    as_frame(trials).to_csv(destination, index=False)


def registered_candidates(model_id: str,
                          model_config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    registry = model_config["registry"]
    if model_id not in registry:
        raise KeyError(f"unregistered model: {model_id}")
    if not registry[model_id]["tuning"]["enabled"]:
        raise KeyError(f"model does not use candidate tuning: {model_id}")
    key = ("deep_shared_prototype"
           if registry[model_id]["family"] == "deep" else model_id)
    spaces = model_config["search"]["spaces"]
    if key not in spaces:
        raise KeyError(f"no pre-registered search for {model_id}")
    result = tuple(dict(item) for item in spaces[key])
    expected = int(model_config["search"]["budget"]["trials_per_model"])
    identities = {repr(sorted(item.items())) for item in result}
    if len(result) != expected or len(identities) != expected:
        raise ValueError(
            f"{model_id}: candidate search must have {expected} distinct settings")
    return result


# The concise name is retained for callers, but configuration remains explicit.
candidates = registered_candidates


def choose_candidate(candidates_: Sequence[Mapping[str, Any]],
                     scorer: Callable[[Mapping[str, Any]], float]
                     ) -> tuple[dict[str, Any], float]:
    scored = [(dict(candidate), float(scorer(candidate)))
              for candidate in candidates_]
    if not scored:
        raise ValueError("cannot select from an empty candidate set")
    return min(scored, key=lambda item: item[1])


def run_trials(
        model_id: str, outer_id: str, folds: Sequence[InnerFold],
        fit_predict: Callable, *, model_config: Mapping[str, Any],
        protocol_config: Mapping[str, Any], device: str = "cpu",
        smoke: bool = False, gap_days: int | None = None
        ) -> tuple[int, list[Trial]]:
    """Evaluate every registered candidate on the same disjoint inner folds."""
    validation = protocol_config["validation"]
    expected = int(validation["inner_folds"])
    gap = purge_days(validation) if gap_days is None else int(gap_days)
    if gap not in tuple(int(value)
                        for value in validation["gap_sensitivity_days"]):
        raise ValueError("gap_days must be a preregistered sensitivity value")
    if not smoke and len(folds) != expected:
        raise ValueError(
            f"expected {expected} common inner folds, received {len(folds)}")
    seed = int(protocol_config["seed_policy"]["tuning_seed"])
    quantiles = tuple(float(value)
                      for value in protocol_config["probability"]["quantiles"])
    all_candidates = registered_candidates(model_id, model_config)
    selected_candidates = all_candidates[:1] if smoke else all_candidates
    selected_folds = folds[:1] if smoke else folds
    ledger: list[Trial] = []
    for candidate_id, parameters in enumerate(selected_candidates):
        for fold in selected_folds:
            assert_gap(fold.fit, fold.early_stop, gap)
            assert_gap(fold.early_stop, fold.score, gap)
            start = time.perf_counter()
            error = None
            loss = None
            epoch = None
            memory = None
            status = "ok"
            try:
                predictions, epoch, memory = fit_predict(
                    dict(parameters), fold.fit.copy(), fold.early_stop.copy(),
                    fold.score.copy(), seed, quantiles)
                predictions = np.asarray(predictions, dtype=float)
                if predictions.shape != (len(fold.score), len(quantiles)):
                    raise ValueError(
                        "estimator did not return one seven-quantile row per "
                        "scoring sample")
                if not np.isfinite(predictions).all():
                    raise ValueError("non-finite scoring prediction")
                if "y" not in fold.score:
                    raise KeyError("scoring fold requires y")
                loss = mean_pinball(
                    fold.score["y"].to_numpy(dtype=float), predictions, quantiles)
            except Exception as exc:
                status, error = "error", f"{type(exc).__name__}: {exc}"
            ledger.append(Trial(
                model=model_id, candidate=candidate_id,
                parameters=dict(parameters), outer=outer_id,
                inner=fold.fold_id, seed=seed, fit_rows=len(fold.fit),
                early_stop_rows=len(fold.early_stop),
                scoring_rows=len(fold.score), mean_pinball=loss,
                wall_seconds=time.perf_counter() - start, device=device,
                peak_memory_bytes=memory, epoch=epoch, status=status,
                error_reason=error))
    if smoke:
        return -1, ledger
    scores = {}
    for candidate_id in range(len(all_candidates)):
        records = [row for row in ledger if row.candidate == candidate_id]
        if len(records) == expected and all(row.status == "ok" for row in records):
            scores[candidate_id] = float(np.mean(
                [row.mean_pinball for row in records]))
    if len(scores) != len(all_candidates):
        raise IncompleteTrialsError(
            f"{len(scores)}/{len(all_candidates)} candidates completed every "
            "inner fold; fair six-candidate selection is blocked", ledger)
    return min(scores, key=lambda key: (scores[key], key)), ledger
