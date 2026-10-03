# Phase F models and experiment result

Recorded: 2026-10-03 (Asia/Shanghai)

## Models

- Replaced the placeholder model factory with one registry-driven constructor
  covering all 24 registered semantic model IDs and every registered family.
- Migrated the seven fixed CPU value-ladder models, including strictly causal
  persistence, clear-sky smart persistence, the closed-form convex blend,
  lead-specific bias correction, and fold-preprocessed linear MOS.
- Statistical and tree models now consume only the formal feature registry and
  fit preprocessing on the current fit block.  Ridge seven-quantile residual
  construction, LightGBM/XGBoost quantile early stopping, per-quantile inner
  round receipts, and outer refitting without outer-score early stopping were
  retained.
- Migrated all 14 deep/experimental architectures into
  `src/nwp/models/deep/`, including independent AutoCorrelation top-k lags.
  Added a reusable seven-quantile trainer with disjoint fit and early-stop
  blocks, crossing penalty, deterministic seed, best-epoch receipt, and the
  existing PINN official-execution block.  No deep training was launched.

## Experiment layer

- Consolidated the distinct six-candidate search space, 6 x 3 common-fold
  fairness rule, mean-pinball selection, failure blocking, smoke no-selection,
  and complete trial ledger in `nwp.experiment.tuning`.
- Fitting is routed through the unique model factory and explicit RunConfig.
- Replaced the prediction placeholder with the complete row schema,
  coordinate/time/type/official gates, generation, CSV/Parquet read/write, and
  canonical column ordering.  Configuration is passed explicitly.
- Migrated additive residual calibration, quantile rearrangement, static and
  forecast-issue-causal calibration, and actual UTC-calendar-day block
  bootstrap into `nwp.experiment.calibration`.

## Removal evidence

After every active Python caller and test was moved, repository searches found
no active imports of `s05_tuning`, `s06_models`, or `s08_calibration`, nor of
the retired `s03_models` registry/train modules.  Only then were the 16 tracked
Phase F source files listed in `COMPLETED_PHASE_F_SOURCES` removed.  They remain
recoverable from tag `pre-nwp-refactor-20261002` and are marked
`COMPLETE_SOURCE_MIGRATED_REMOVED` in the migration map.

`s07_prediction` is deliberately retained for now because the old evaluation
layer still imports it.  Its behavior is already present in
`nwp.experiment.prediction`; callers and source removal belong to Phase H, not
an early Phase F deletion.  The blocked legacy calibration CLI is likewise
retained until the Phase K residual-source cleanup.

Focused checks completed during the phase:

```text
fixed baselines and model factory:                    6 passed
statistical/tree/tuning protocol:                    19 passed
deep architecture, registry, and factory coverage:  19 passed
prediction/calibration/bootstrap/E2E:                12 passed
```

Final old-plus-new suite was run with the Anaconda interpreter for `pyarrow`
and the system Python package path for `pvlib`/`shapely`:

```text
........................................................................ [ 67%]
..................................                                       [100%]
106 passed in 17.87s
```

No formal/full training, network request, data move, quarantine action,
figure deletion, report deletion, result promotion, or Git push was performed
in Phase F.
