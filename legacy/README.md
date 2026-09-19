# Legacy execution boundary

This directory holds retired, provisional or historical executable code. It is
not part of the active scientific pipeline and must never produce official
results. The old server queue and PowerShell deep-training launcher remain
hard-blocked. Reproduction of historical results requires a separate,
explicitly marked legacy environment and must not update active reports.

Current formal modules under `src/s01_core` through `src/s15_validation` must
not import from this directory.

## Experiment and model entries migrated in this round

| Previous path | Current path | Classification | Reason |
|---|---|---|---|
| `src/s02_experiment/cv_month_balanced_quantile.py` | `legacy/s02_experiment/cv_month_balanced_quantile.py` | legacy | Month-balanced, non-causal model selection with old 3-quantile scoring and station-encoding ablations. Requires explicit `--allow-legacy`. |
| `src/s03_models/train/train_quantile_v1.py` | `legacy/s03_models/train/train_quantile_v1.py` | legacy | Historical v1 trainer with pre-nested selection and test-period outputs. Requires explicit `--allow-legacy-reproduction`; not a formal path. Its old cloud branch is intentionally blocked by the retired full-data imputer. |
| `src/s03_models/predict/predict_daily.py` | `archive/code/predict_daily.py` | archive | Unimplemented live inference sketch (`NotImplementedError`), not a current prediction interface. |
| `src/s02_experiment/split_protocol.py` | unchanged | compatibility / delete candidate | Existing historical tests and prototype code still import it; current selection instead imports `src/s04_splits`. Remove only after callers and tests are migrated. |
| `src/s03_models/train/train_deep.py` | unchanged | current prototype | Smoke/development only; validated registry is empty and official execution is blocked. Final test predictions are not produced in this entry. |
| `src/s03_models/train/train_v1.py` | unchanged | transitional shared code | Current fold-local adapters use its physical feature list and model pipes; station encodings were removed. Replace with the dedicated formal feature registry after its issue-time availability audit. |

No current `s04_splits`, `s05_tuning`, or `s06_models` module imports any file
under `legacy/` or `archive/`. Historical `vN` identifiers are provenance only.
