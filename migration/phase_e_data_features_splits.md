# Phase E data, features, and splits result

Recorded: 2026-10-03 (Asia/Shanghai)

## Data

- Added immutable, receipt-backed monthly API fetches with exact request,
  returned-coordinate, variable, row-count, missingness, and content-hash
  validation.
- Added catalog-only raw resolution and monthly clean construction for Previous
  Runs, Himawari, and ERA5.  Radiation/cloud physical policies, issue/target/lead
  chronology, all registered leads, hourly coverage, and requested/returned
  coordinates are validated before registration.
- Data inventory remains non-destructive.  Automatic quarantine is disabled;
  only an explicit reviewed path list can be moved in Phase I.
- Migrated the data tests, audit scripts, and validation entry points to the
  canonical configuration and data contracts.

## Features

- Migrated returned-GFS-service-point solar geometry and preceding-hour
  clear-sky semantics.
- Migrated issue-time validation, unclipped diagnostic/model ratios, circular
  encodings, forecast lags, cloud changes/rolling means, formal feature
  selection, fold-only cloud imputation, and fold-only numeric preprocessing.
- Added monthly feature partitions keyed by the clean content hash plus the
  complete feature configuration.  Every partition has a canonical receipt and
  DataCatalog record and is reused only after physical hash validation.
- The integration test executes synthetic `raw -> clean -> features` and checks
  lineage, coordinates, schema, row count, and idempotent reuse.

## Splits

- Consolidated the two previous split implementations into one explicit-config
  implementation.
- Preserved development/test boundaries, all five registered outer folds,
  registered 7/10/14-day sensitivity gaps, the 240-hour purge derivation, and
  three disjoint `fit -> purge -> early-stop -> purge -> score` inner folds.
- Preserved the final causal order
  `fit -> purge -> early-stop -> purge -> calibration -> final test` and added
  a consistency check between registered `fit_end` and the derived purge.
- Migrated sample-sufficiency diagnostics and the tuning adapters without
  allowing a compatibility implementation to choose different windows.

## Source removal evidence

After callers and tests were migrated, searches found no active imports of
`s01_data`, `s02_data`, `s03_features`, `s04_splits`, or `split_protocol`.
Only then were the 19 tracked Phase E source files removed.  Each remains
recoverable from tag `pre-nwp-refactor-20261002`; their migration-map state is
`COMPLETE_SOURCE_MIGRATED_REMOVED`, not `RECOVERY_REQUIRED`.

Focused checks completed during the phase:

```text
data reconciliation:                         15 passed
feature unit/integration/E2E checks:          17 passed
split consolidation and tuning checks:       15 passed
```

Final old-plus-new suite after all Phase E source removals:

```text
........................................................................ [ 69%]
................................                                         [100%]
104 passed in 18.91s
```

No formal training, network data fetch, data move, quarantine action, figure
deletion, report deletion, or result promotion was performed in Phase E.  The
117 files already placed under `data/quarantine` by the earlier erroneous pass
were not moved again and remain for the explicit Phase I audit.

The repository-root `.pytest_cache` from the earlier pass remains inaccessible
to the current Windows process because its ACL denies even ownership takeover.
Pytest's cache provider is disabled for all verification runs, and every
accessible per-run temporary directory created in this phase was removed.
