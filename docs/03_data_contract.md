# Data contract

## Authoritative definitions

`config/data.yaml` is the sole authority for provider, product/model, forecast
variables, leads, truth roles, radiation semantics, timezone, tilt, azimuth,
and storage policy. `config/sites.yaml` is the sole authority for requested
site coordinates. Code must not carry a second variable or site registry.

Previous Runs supplies location-specific fields from the recorded
`gfs_seamless` product at 24, 48, and 72 h offsets. The derived
`forecast_issue_time_utc = target_time_utc - lead_time` boundary is used for
causal availability; it is not claimed to be a verified native GFS
initialization timestamp. Himawari `shortwave_radiation` is the sole active
satellite-derived GHI reference. ERA5 is not part of active acquisition,
cleaning, features, audit, or evaluation.

The raw acquisition contract and model-specific feature contracts are
separate. Linear and Ridge use the declared compact statistical groups and may
apply fold-local dimensionality reduction. LightGBM and XGBoost use the richer
tabular groups. Deep models may use the richer raw-derived table, temporal
sequences, and later spatial neighbourhood inputs. The project does not create
an artificial all-model-identical feature layer. Variables available from an
API but absent from the preserved Previous Runs bytes, including low/mid/high
cloud and pressure-level fields, are not active contract variables.

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

`data/data_inventory.csv` is the complete local filesystem audit. Its actions
are restricted to `KEEP`, `MOVE`, `MERGE`, `DUPLICATE_DELETE`, and `UNKNOWN`;
automatic quarantine is forbidden.

## Local ERA5 archive

ERA5 is outside the active repository and catalog. Its preserved local archive
is `../meteo prediction.local-archive/era5_20261006/data/raw/era5` relative to
the repository. It contains 3,744 files and 78,002,911 bytes. The archive
inventory is `era5_inventory_sha256.csv`; its SHA-256 is
`38784f71fc060ea4aebe58c35ade0c52526fc7ac0426bad86be368668edaca9e`.
The archive bytes and per-file inventory are local-only and are not uploaded to
GitHub.
