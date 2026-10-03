# Phase C configuration-layer result

Recorded: 2026-10-03 (Asia/Shanghai)

## Canonical ownership

`project_manifest.yaml` contains only project identity, protocol version,
official-result registration, and the exact six canonical config paths.  The
six files are:

- `config/protocol.yaml`
- `config/data.yaml`
- `config/features.yaml`
- `config/sites.yaml`
- `config/models.yaml`
- `config/experiments.yaml`

The loader rejects missing/extra top-level ownership keys, noncanonical config
paths, duplicate YAML keys, duplicated tuning seed/objective ownership, and
invalid cross-file references.

## Frozen-protocol parity

`tests/unit/test_config_parity.py` compares the frozen overloaded manifest and
the restored old site/variable files against their new canonical owners.  The
comparison covers versions, target/task/status, periods, all outer/final
boundaries, purge derivation, quantiles, metrics, prediction contract, feature
groups/policy, spatial rules, all 20 sites, all 15 forecast variables, all 24
models, all 14 GPU/internal-version models, tuning budgets/search spaces, PINN
constraints, and evidence status.  No formal scientific value was intentionally
changed.

Fields missed by the first refactor attempt were restored before the parity test
was allowed to pass, including model roles/internal versions, `tsmixer`, PINN
constraints, version identifiers, request-coordinate semantics, diagnostic
rules, trial definition, and provisional result-set metadata.

## Resolution and gates

The canonical loader now produces a deeply immutable `RunConfig`.  Resolution
order is base scientific config, experiment profile, development-only CLI
override, then resolved config/hash.  Ambiguous site override combinations are
rejected.  Density selection is deterministic and never takes the first N
configured sites.  Official profiles reject CLI scientific overrides and fail
closed unless the config hash is locked, every selected model is validated and
officially eligible, readiness is explicitly passed, and an official result set
is registered.

## Single read entry

The only source file that parses YAML configuration is
`src/nwp/core/config.py`.  The old `s01_core.config_loader` is now a temporary
projection adapter over that canonical loader; it no longer reads old YAML.
Other old modules that previously read config files directly were switched to
the adapter.  The adapter and old model registry remain only to keep the
unmigrated modules testable and are scheduled for Phase K removal.

## Tests

Configuration-only run:

```text
9 passed in 0.31s
```

Full old-plus-new suite after restoring complete test discovery:

```text
........................................................................ [ 75%]
........................                                                 [100%]
96 passed in 13.97s
```

No old config, source package, script, figure, report, or data file was deleted
in Phase C.
