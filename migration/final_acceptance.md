# Targeted repair acceptance

STATUS: MERGE_READY

Date: 2026-10-06 (Asia/Shanghai)

Branch: `codex/nwp-unified-refactor`

Baseline: `82fa80549c5ca34d81fd7c9936b70b67c80306b1`

## Acceptance gates

| gate | status | evidence |
|---|---|---|
| gap dependency propagation | PASS | production dependency truth table |
| XGBoost-only invalidation | PASS | real child tuning/fitting workflow; Ridge/LGBM calls are zero |
| figure check-before-render | PASS | compatible child figure render count is zero |
| implementation fingerprints | PASS | baseline/ridge/tree directional invalidation tests |
| tree implementation routing | PASS | arbitrary LightGBM model alias test |
| deep capability status | PASS | architecture-only, tuning disabled, non-official |
| deep environment fingerprint | PASS | Torch included for every deep implementation |
| base environment | PASS | import, validate, status, inventory and bounded split workflow |
| full environment | PASS | 160 tests passed in 54.20 seconds |
| active documentation | PASS | acceptance, status and repository guide agree |
| pseudo audit removal | PASS | no unsupported per-file responsibility audit or reference remains |
| recovery evidence preservation | PASS | all 118 unique files retained |
| scientific regression | PASS | baseline time/split/purge/quantile/data/site/spatial/model/profile assertions |
| analysis runtime | PASS | independent analysis configuration and bounded real-LightGBM group evidence execution |
| gap sensitivity | PASS | 7/10/14-day rolling folds differ; frozen final chronology remains canonical and reachable |
| final-test isolation | PASS | development prediction/evaluation is closed; frozen official readiness gate controls hold-out access |
| interrupted fitting resume | PASS | committed current-run model artifact is resumed in place without cross-run materialization |
| active machine-ID naming regression | PASS | canonical snake_case IDs reject artificial numeric-version suffixes |

Detailed evidence is recorded in `migration/targeted_repair_validation.md`.

## Scope boundary

No scientific protocol, time boundary, validation fold, purge, probability
contract, 15-variable data contract, site cohort, spatial evaluation semantics,
DataCatalog design, raw immutability rule, lifecycle, directory structure, or
historical evidence was redesigned.

No formal training, GPU benchmark, multi-seed experiment, official run, bulk
remote acquisition, or official result registration was performed.
