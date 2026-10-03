# Final residual searches

Search date: 2026-10-04. Scope for active logic: `src/nwp`, `config`, `tests`,
`README.md`, and `docs`, excluding `docs/07_history.md` where explicitly noted.
Generated caches were excluded.

## 1. Path residues

The case-insensitive search covered every numbered source-package name from
`s01_core` through `s15_validation`, plus `scripts/05_cpu` and
`nanjing_15var_diagnostic`.

Result: **zero matches in active code, configuration, tests, README, or active
documentation**.

The structural CLI independently parses every import in `src/nwp` and reports
`active package has no migration-layer imports: pass`.

## 2. Historical-word residues

The case-insensitive search covered `legacy`, `deprecated`, `retired`,
`superseded`, `do_not_cite`, `protocol2`, `18 variable` (including separator
variants), and `month_balanced`.

Result: **zero matches outside `docs/07_history.md` and `migration/` evidence**.
The history index intentionally contains the former result-set identifier and
its exclusion reason. Migration inventories retain original paths and labels as
audit evidence; they are not runtime logic.

## 3. Hard-coded scientific parameters

The case-insensitive search covered `nanjing_1`, integer `20`, `9564`,
`2024-02`, `2025-09`, `q0.05`, `ridge_mos`, and `xgboost`.

Result in `src/nwp/workflow` and `src/nwp/cli.py`: **zero matches**. Site IDs,
site counts, dates, quantiles, and registered model IDs are therefore not
hard-coded in orchestration.

The remaining runtime matches were reviewed individually:

- `src/nwp/data/audit.py` contains a generic `20YY-MM` filename-recognition
  regular expression, not a configured date or site;
- `src/nwp/core/provenance.py` and `src/nwp/core/validation.py` name the
  installed `xgboost` Python distribution for environment provenance/readiness;
- `src/nwp/models/trees.py` contains LightGBM/XGBoost implementation branches,
  which are the genuinely distinct algorithms registered by configuration;
- `src/nwp/evaluation/interpretation.py` contains the corresponding
  algorithm-specific estimator construction for interpretation evidence.

These are implementation facts under the instruction's explicit exception for
genuinely new/different algorithms. Model selection remains driven by
`config/models.yaml`; workflow dispatch uses `quantile_adapter` and `algorithm`
from the registry rather than model IDs. The searched site, count, row-count,
date, and quantile literals have no runtime matches.

Test fixtures may name registered models and representative values in order to
assert scientific contracts. They do not influence production configuration or
workflow selection.
