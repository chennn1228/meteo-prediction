# Data contract

## Authoritative definitions

`config/data.yaml` is the sole authority for provider, product/model, forecast
variables, leads, truth roles, radiation semantics, timezone, tilt, azimuth,
and storage policy. `config/sites.yaml` is the sole authority for requested
site coordinates. Code must not carry a second variable or site registry.

Previous Runs supplies the location-specific GFS forecast fields at the
configured lead hours. A lead is the elapsed time from
`forecast_issue_time_utc` to `target_time_utc`; those three values must be
arithmetically consistent. Himawari is primary truth. ERA5 is supplementary and
exploratory and may not silently replace missing GFS or Himawari values. GHI
and the registered radiation variables use the configured preceding-hour
semantics.

## Coordinates and fields

Every raw receipt distinguishes the requested site coordinates from the
coordinates returned by each service. Requested coordinates are provenance;
returned coordinates drive service-point physics and spatial analysis. Raw
validation checks registered fields, ordered hourly timestamps, calendar-month
coverage, row counts, missingness, and returned coordinates. An all-null
required field or incomplete month is not `ready`.

## Storage stages

```text
data/raw/<source>/<site>/<month>.*
data/clean/<clean-hash>/<site>/<month>.parquet
data/features/<feature-hash>/<site>/<month>.parquet
data/registry/
data/catalog.json
```

`raw` is immutable acquired material. `clean` normalizes chronology, fields,
coordinates, and physical contracts. `features` is issue-time-safe model
material. Active code resolves data through `DataCatalog`; it does not infer a
dataset from a filename or concatenate a historical directory name.

Each materialized active partition has a canonical sidecar receipt recording
its stage, configuration/dependency hashes, source, request facts, requested
and returned coordinates, variables, time range, row count, missingness,
content hash, execution level, Git commit, and creation time. Ready catalog
records are accepted only when the file, receipt, stage, configuration hash,
status, and content hash agree. Inherited artifacts without sufficient lineage
are retained as `incomplete` and cannot satisfy resolution.

`data/data_inventory.csv` is the non-destructive filesystem audit. Its actions
are restricted to `KEEP`, `MOVE`, `MERGE`, `DUPLICATE_DELETE`, and `UNKNOWN`;
automatic quarantine is forbidden.
