# Phase D core result

Recorded: 2026-10-03 (Asia/Shanghai)

## Completed core contracts

- `core/config.py`: canonical six-file loading, immutable `RunConfig`, optional
  machine-only local path loading, deterministic profile/CLI resolution, and
  fail-closed official gates.
- `core/hashing.py`: structural canonical JSON; deterministic mapping, sequence,
  set, date, dataclass, and path handling; rejection of non-finite or unsupported
  values; full file SHA-256 support.
- `core/paths.py`: the only constructor for data roots, run roots, fixed stage
  directories, model/fold paths, prediction paths, partition paths, and receipt
  sidecars. Unsafe path segments and data-root escapes are rejected.
- `core/provenance.py`: one validated receipt schema, Git/environment capture,
  no-overwrite provenance writes, timezone-aware timestamps, and atomic writes.
- `core/schema.py`: immutable validated `StageResult` and central model-feature
  leakage checks.
- `core/context.py`: fixed run initialization, resolved config, runtime
  provenance, initial stage status, central selected sites/models, catalog, and
  atomic stage-result recording.
- `data/contracts.py`: schema-versioned, atomic, receipt-backed `DataCatalog`
  with unique IDs, safe relative paths, exact/superset site resolution controls,
  physical content-hash verification, and processed-data lineage checks.

The old provenance module now delegates environment capture and writes to the
canonical mechanism; its old-shaped field projection remains a temporary
migration adapter until Phase K.

## Run layout verification

Context initialization creates exactly the fixed directories `00_meta` through
`09_report`.  Tests use explicit temporary data/output roots, so validation no
longer creates additional test runs under the repository's real `outputs/`.

## Tests

Focused Phase D tests:

```text
8 passed in 1.17s
```

Full old-plus-new suite after final Phase D changes:

```text
........................................................................ [ 71%]
.............................                                            [100%]
101 passed in 16.87s
```

No formal model training was started. No old source, data, figure, report, or
migration layer was deleted in Phase D.
