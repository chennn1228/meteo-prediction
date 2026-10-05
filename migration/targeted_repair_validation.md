# Targeted repair validation

Date: 2026-10-06 (Asia/Shanghai)

Baseline: `71729cfb67e9d0248e7243a4707f28bfaa474ad1`

| requirement | result | evidence |
|---|---|---|
| Gap invalidates splits and all model/downstream stages, but not data/features | PASS | `tests/unit/test_dependency_truth_table.py` |
| XGBoost search change preserves Ridge/LGBM and recomputes XGBoost | PASS | real tuning/fitting child workflow in `tests/integration/test_child_run_reuse_workflow.py` |
| Critical dependencies are production-calculated | PASS | production data/feature/split/model dependency functions and real tuning handler are exercised |
| Figure reuse checks before render | PASS | `tests/integration/test_figure_reuse.py`; child render count remains zero |
| Implementation fingerprint follows registry implementation | PASS | baseline/ridge/tree directional invalidation tests |
| Tree quantile routing follows implementation | PASS | LightGBM alias test and implementation-keyed receipts |
| Deep capability is not overstated | PASS | all deep/PINN records are architecture-only, untuned and non-official |
| Deep environment contains Torch | PASS | targeted environment-package test |
| Base-only import boundary and bounded splits | PASS | `migration/base_environment_validation.txt` |
| Full environment | PASS | 153 tests in `migration/full_environment_validation.txt` |
| Frozen science | PASS | `tests/unit/test_targeted_science_regression.py` compares the baseline commit |
| Pseudo per-file audit removed | PASS | no unsupported per-file responsibility audit or reference remains |
| Historical unique recovery evidence untouched | PASS | 118 files remain in `migration/recovered_artifacts/` |

No formal training, GPU benchmark, multi-seed experiment, official run, bulk
remote acquisition, or official result registration was performed.
