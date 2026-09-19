# Active project structure and data flow

Version 2.0.0-provisional | 2026-09-15

## Canonical flow

```text
config + project_manifest
→ raw data
→ clean data with explicit requested/returned coordinates
→ feature groups
→ nested rolling model selection
→ final fit / early stop / calibration / test
→ reports and figures
```

## Active paths

```text
project_manifest.yaml
config/
  01_sites.yaml
  02_variables.yaml
docs/
  01_research.md
  02_project_status.md
  03_model_config.md
  04_project_structure.md
  05_literature.md
  issues_open.md
  archive/
src/
  s01_data/{fetch,clean,features}/
  s02_experiment/
  s03_models/
  s04_evaluation/{analysis,calibration,verification}/
scripts/
  06_validate/validate_project.py
reports/
figs/
NWP_website_handoff/       # sealed, standalone external delivery; not a research dependency
```

Only the six named files directly under `docs/` are active project context. `docs/archive/` is historical and must not be used to infer current protocol or status.

## Naming and identity

- `location_id`: stable site or grid identifier;
- `source_grid_id`: local semantic ID for an API-returned location, not an upstream GRIB ID;
- `model_id`: stable semantic model ID; internal `vN_*` is provenance only;
- `lead`: hours (`24`, `48`, `72`);
- `target_time_utc`: valid time;
- `fcst_issue_time_utc`: valid time minus lead;
- `status=provisional` until the manifest names an official result set.

## Result-state policy

| State | Use |
|---|---|
| official | frozen v2 result set only |
| provisional | real but not validated under complete v2 protocol |
| exploratory | hypothesis generation / supplementary |
| legacy | reproducibility of retired protocol only |
| smoke/debug | pipeline checks only; never scientific display |

## External website delivery

`NWP_website_handoff/` is a frozen, independently sendable snapshot. All website
source, data contracts and selected research references live inside that one
directory. Active research scripts neither read from nor write to it.

## Sensitive and large paths

The website delivery contains no `secrets/`, `.venv/`, `.cache/`, `.trash/`, `deploy/`, raw data, checkpoints, logs, `__pycache__`, `node_modules`, `dist`, smoke/debug outputs or historical backups.
