# Phase I data and inherited-artifact result

Recorded: 2026-10-03 (Asia/Shanghai)

## Data inventory and correction of the earlier quarantine error

- Rebuilt `data/data_inventory.csv` from the live filesystem with byte size,
  SHA-256, extension, estimated stage, site, date range, Parquet schema,
  receipt presence, quarantine state, and action.
- The 117 files moved prematurely into `data/quarantine` were reviewed before
  any further action.  They were not unknown data: 2 geographic inputs, 21
  returned-service-point probe batches, and 94 inherited audit/CPU diagnostic
  artifacts.  All 117 were first restored to their exact original relative
  paths, correcting the earlier move.
- Geographic inputs, fetch logs, historical site selections, and probe caches
  were then hash-verified into `data/registry/`.  The 94 inherited diagnostic
  results and 24 inherited data-audit records were moved out of the active data
  tree to `migration/recovered_artifacts/`; they are retained for traceability
  and are not represented as new-pipeline results.
- Quarantine is now absent.  The final data inventory contains 11,439 files,
  all classified `KEEP`; there are no `UNKNOWN`, `MOVE`, or
  `DUPLICATE_DELETE` rows.

## Raw API source data

- Confirmed that the inherited `data/01_raw` JSON files are Open-Meteo API
  source responses, not disposable generated results.
- Validated every inherited primary raw response against the current 15-variable
  contract and migrated all 5,521 byte-identical payloads to
  `data/raw/<source>/<site>/<month>.json` with one receipt per payload.
- Catalog state is fail-closed: 4,281 payloads pass the current monthly
  contract and are `ready`; 1,240 are preserved as `incomplete`.  The latter
  are predominantly pre-availability Previous Runs months with all-null
  forecast fields, plus partial 2026-09 responses.  No incomplete payload can
  be resolved by the active pipeline.
- Imported 98 raw files from `protocol15temp` and `protocol2` as immutable,
  content-hash-named variants.  Ninety-five are scientific-payload duplicates
  of canonical responses (different transport metadata); three are distinct
  rejected 18-variable GFS variants.  All 98 are catalogued `incomplete`, and
  their old metadata is embedded and hashed in their receipts.
- Only after source/target/receipt SHA-256 verification reported zero errors
  were the old `01_raw`, `protocol15temp/01_raw`, and `protocol2/01_raw`
  directory trees removed.

The per-file evidence is retained in:

- `migration/data_raw_import_plan.csv` (5,521 mappings)
- `migration/data_raw_variant_plan.csv` (98 mappings)
- `migration/data_nonpartition_disposition.csv` (147 reviewed moves)

## Inherited clean/features data

- Preserved 86 inherited Parquet files byte-for-byte under content-addressed
  `data/clean/<hash>/...` and `data/features/<hash>/...` paths: 43 clean and 43
  feature artifacts.
- Because the inherited files do not have complete trustworthy lineage, each
  has a blocked receipt and an `incomplete` catalog record.  Existing legacy
  receipt payloads (three files) are embedded with their hashes.  The active
  catalog cannot reuse these files as ready data.
- Source/target/receipt hashes were checked before the numbered and manually
  versioned processed-data trees were removed.  Exact mappings are in
  `migration/data_processed_import_plan.csv`.

Final catalog totals:

```text
records:     5,705
raw:         5,619
clean:          43
features:       43
ready:       4,281
incomplete:  1,424
```

## Development-run residue

Thirty small refactor-generated development run skeletons were inspected.
They contained only validate/selection/data/features/splits metadata and no
tuning, models, predictions, calibration, metrics, analysis, figures, or
reports.  Their 233 files were inventoried and the 30 scaffolds were removed.
Evidence is in `migration/development_run_inventory.csv` and
`migration/development_run_file_inventory.csv`.

## Figures and reports

`migration/figure_inventory.csv` preserves the complete pre-cleanup inventory of 428
inherited figure/report artifacts:

```text
LEGACY_PROTOCOL:  396
WRONG_PROCESSING:  13
SUPERSEDED:         10
DUPLICATE:           9
CURRENT_VALID:       0
```

The 146 files currently present are tracked and recoverable from tag
`pre-nwp-refactor-20261002`; they remain in place until the Phase K deletion.
The earlier erroneous pass deleted 282 untracked items (10 figures and 272
reports).  They are explicitly marked `RECOVERY_REQUIRED` with their former
paths, sizes, and SHA-256 values.  They are not in the freeze commit/tag and
cannot truthfully be claimed as recovered.  No inherited artifact is promoted
into a new run or represented as a current result.

No network request, formal/full training, model fitting, result regeneration,
or scientific-protocol change was performed in Phase I.
