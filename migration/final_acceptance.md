# Targeted repair acceptance

STATUS: MERGE_READY

Date: 2026-10-06 (Asia/Shanghai)

Branch: `codex/nwp-unified-refactor`

Baseline: `71729cfb67e9d0248e7243a4707f28bfaa474ad1`

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
| full environment | PASS | 153 tests passed in 45.54 seconds |
| active documentation | PASS | acceptance, status and repository guide agree |
| pseudo audit removal | PASS | no unsupported per-file responsibility audit or reference remains |
| recovery evidence preservation | PASS | all 118 unique files retained |
| scientific regression | PASS | baseline time/split/purge/quantile/data/site/spatial/model/profile assertions |

Detailed evidence is recorded in `migration/targeted_repair_validation.md`.

## Scope boundary

No scientific protocol, time boundary, validation fold, purge, probability
contract, 15-variable data contract, site cohort, spatial evaluation semantics,
DataCatalog design, raw immutability rule, lifecycle, directory structure, or
historical evidence was redesigned.

No formal training, GPU benchmark, multi-seed experiment, official run, bulk
remote acquisition, or official result registration was performed.
