# Phase G workflow result

Recorded: 2026-10-03 (Asia/Shanghai)

## Executable DAG

- `nwp.workflow.pipeline` now contains the real ordered stages from validation
  through report generation; the stage registry points to implementations, not
  placeholder callbacks.
- Every stage returns the common `StageResult` contract with inputs, outputs,
  dependency hash, timestamps, and status.  Receipts use the single provenance
  schema.
- `RunContext.resume` verifies the saved resolved configuration, provenance,
  stage status, receipts, and output hashes before reuse.  An inconsistent
  recorded stage is rejected rather than silently overwritten.
- The CLI exposes the sole `python -m nwp` entry point.  `run` supports stage
  bounds and run resumption; `evaluate` and `figures` operate on an existing
  run ID instead of reconstructing scientific settings from new arguments.

## Scientific chronology

- Outer tuning uses only registered inner folds and common completed
  candidates.  Final candidate selection aggregates the registered outer-inner
  evidence without reading final-test outcomes.
- Outer and final models fit only on the declared fit block and choose stopping
  rounds only on the declared early-stop block.
- Prediction produces separate outer-score, final-calibration, and final-test
  artifacts.  Calibration is causal by forecast issue time.  Evaluation keeps
  outer, final uncalibrated, and final calibrated evidence separate.
- Formal calibrated evaluation carries unchanged point baselines alongside
  calibrated quantile rows so preregistered RMSE-skill references remain
  available; it never substitutes uncalibrated quantile rows.

## Removal evidence

After callers and tests were migrated, active Python searches found no
references to `s14_pipeline`, `scripts/01_validate`, `scripts/05_cpu`, or
`scripts/run_stage.py`.  The 11 tracked Phase G sources were then removed and
marked `COMPLETE_SOURCE_MIGRATED_REMOVED` in `migration/migration_map.csv`.  They remain
recoverable from tag `pre-nwp-refactor-20261002`.

Focused workflow/config/evaluation checks:

```text
30 passed in 6.80s
```

Final old-plus-new suite after Phase G/H closure:

```text
........................................................................ [ 67%]
...................................                                      [100%]
107 passed in 20.27s
```

No formal/full training, network request, data move, quarantine action,
historical result promotion, or Git push was performed in Phase G.
