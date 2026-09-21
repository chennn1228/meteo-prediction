"""Common nested-fold selector for seven-quantile models.

The caller supplies an estimator factory; this layer owns the immutable fold
contract, score, candidate equality, and trial ledger. It never reads the
outer-validation or final-test rows while selecting a candidate.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
import time

import numpy as np

from s04_splits.rolling import InnerFold, assert_gap
from s04_splits.split_registry import manifest, protocol, purge_days, quantiles
from s09_metrics.probabilistic import probability_metrics

from .search_space import candidates
from .trial_ledger import Trial


class IncompleteTrialsError(RuntimeError):
    def __init__(self, message: str, ledger: list[Trial]):
        super().__init__(message)
        self.ledger = ledger


def run_trials(model_id: str, outer_id: str, folds: Sequence[InnerFold],
               fit_predict: Callable, *, device: str = "cpu",
               smoke: bool = False, gap_days: int | None = None) -> tuple[int, list[Trial]]:
    """Score six parameter candidates on the same three disjoint inner folds.

    ``fit_predict(parameters, fit, early_stop, score, seed, quantiles)`` must
    fit any imputer/scaler/model on ``fit`` only, use ``early_stop`` solely to
    select training duration, and return ``(score_predictions, epoch, bytes)``.
    The callback must not receive outer-validation or final-test data.
    Smoke execution is diagnostic and may use one candidate/fold, never yields
    an official selection.
    """
    expected = int(manifest()["tuning_budget"]["common_inner_folds"])
    gap = purge_days() if gap_days is None else int(gap_days)
    if gap not in tuple(int(value) for value in protocol()["gap_sensitivity_days"]):
        raise ValueError("gap_days must be a preregistered sensitivity value")
    if not smoke and len(folds) != expected:
        raise ValueError(f"expected {expected} common inner folds, received {len(folds)}")
    seed = int(manifest()["tuning_budget"]["tuning_seed"])
    taus = quantiles()
    all_candidates = candidates(model_id)
    selected_candidates = all_candidates[:1] if smoke else all_candidates
    selected_folds = folds[:1] if smoke else folds
    ledger = []
    for candidate_id, parameters in enumerate(selected_candidates):
        for fold in selected_folds:
            assert_gap(fold.fit, fold.early_stop, gap)
            assert_gap(fold.early_stop, fold.score, gap)
            start = time.perf_counter()
            error, loss, epoch, memory, status = None, None, None, None, "ok"
            try:
                predictions, epoch, memory = fit_predict(
                    dict(parameters), fold.fit.copy(), fold.early_stop.copy(),
                    fold.score.copy(), seed, taus)
                predictions = np.asarray(predictions, dtype=float)
                if predictions.shape != (len(fold.score), len(taus)):
                    raise ValueError("estimator did not return one seven-quantile row per scoring sample")
                if not np.isfinite(predictions).all():
                    raise ValueError("non-finite scoring prediction")
                if "y" not in fold.score:
                    raise KeyError("scoring fold requires y")
                loss = float(probability_metrics(
                    fold.score["y"].to_numpy(), predictions)["mean_pinball"])
            except Exception as exc:
                status, error = "error", f"{type(exc).__name__}: {exc}"
            ledger.append(Trial(
                model=model_id, candidate=candidate_id, parameters=dict(parameters),
                outer=outer_id, inner=fold.fold_id, seed=seed,
                fit_rows=len(fold.fit), early_stop_rows=len(fold.early_stop),
                scoring_rows=len(fold.score), mean_pinball=loss,
                wall_seconds=time.perf_counter() - start, device=device,
                peak_memory_bytes=memory, epoch=epoch,
                status=status, error_reason=error))
    if smoke:
        return -1, ledger  # no selection from an incomplete candidate comparison
    scores = {}
    for candidate_id in range(len(all_candidates)):
        records = [r for r in ledger if r.candidate == candidate_id]
        if len(records) == expected and all(r.status == "ok" for r in records):
            scores[candidate_id] = float(np.mean([r.mean_pinball for r in records]))
    if len(scores) != len(all_candidates):
        raise IncompleteTrialsError(
            f"{len(scores)}/{len(all_candidates)} candidates completed every inner fold; "
            "fair six-candidate selection is blocked", ledger)
    return min(scores, key=lambda candidate_id: (scores[candidate_id], candidate_id)), ledger
