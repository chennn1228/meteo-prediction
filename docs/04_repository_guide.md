# Repository guide

## Where changes belong

- `project_manifest.yaml` points to the six authoritative configurations.
- `config/protocol.yaml` holds scientific rules.
- `config/data.yaml` holds source and raw-data facts.
- `config/features.yaml` holds feature groups and policies.
- `config/sites.yaml` holds sites, sets, and selectors.
- `config/models.yaml` holds model registrations and search spaces.
- `config/experiments.yaml` composes those facts into profiles.
- `src/nwp/` contains implementation, never a second experiment definition.
- `data/` contains catalogued research material; `outputs/` contains isolated
  runs.

Only `nwp.core.config` reads YAML. A run resolves configuration once into an
immutable `RunConfig`; later components receive it through `RunContext`.
`RunPaths` constructs project paths, `DataCatalog` resolves persisted data, and
the shared provenance functions own receipts.

## Normal changes

Add or replace a site in `sites.registry`, then reference it from a set/profile
or use the development CLI overrides. Change site count through a configured
selector; never take the first N entries. Add an existing model type by
registering it. For a genuinely new algorithm, implement the common model
interface under `nwp.models` and register it. Do not create model-, city-,
fold-, gap-, or site-count-specific execution scripts.

## Runs and traceability

Use only `python -m nwp`. Every run owns `00_meta` through `09_report` beneath
`outputs/<execution>/<run-id>/`. Resume by run ID. Reuse validates dependency,
implementation, environment, and persisted bytes before materialization.

Metrics and analysis tables belong in `06_metrics` and `07_analysis`.
`08_figures` contains only SVG, PNG, and `figure_index.json`. Trace a figure
through its index to a source table, predictions, resolved configuration, and
data receipts. Never infer provenance from a human version suffix.

## History and local-only material

Prior source is retrieved through the Git tags listed in `docs/07_history.md`.
Do not create historical-code directories, numbered source packages, copied
documentation trees, or a new migration tree. Local literature is intentionally
ignored by Git. Historical tracked material is recovered from commits and tags.
