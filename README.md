# Jiangsu GFS–GHI probabilistic post-processing

Protocol 2.0.0-provisional

The project asks one ordered scientific question: does D+1/D+2/D+3 GFS GHI
provide value beyond non-NWP baselines, and how much incremental value is added
by statistical correction, tree ML, deep learning and probability calibration?

> Current model numbers and website data are **provisional**. The former
> month-balanced protocol is retired, and `official_result_set` remains `null`
> until the new nested purged rolling-origin workflow is rerun.

## Read first

1. `project_manifest.yaml` — machine-readable Single Source of Truth;
2. `docs/01_research.md` — RQs, hypotheses, evidence ladder and boundaries;
3. `docs/02_project_status.md` — completed work, blockers and run order;
4. `docs/03_model_config.md` — registry, tuning budget and seeds;
5. `docs/04_project_structure.md` — paths, naming and data flow;
6. `docs/issues_open.md` — only unresolved P0/P1 work.

Files under `docs/archive/` are historical and do not define the current
protocol, status, model numbering or scientific claims.

## Formal protocol

- Primary target: GHI; cloud is input, error mechanism and supplementary experiment.
- Development: 2024-02-01 through 2025-08-31.
- Final test: 2025-09-01 through 2026-08-31.
- Validation: nested purged rolling-origin with 10-day primary purge (168 h lookback + 72 h lead).
- Calibration order: fit → early stop → later calibration → final test.
- Spatial evaluation: stratified 20-site holdout → province-wide unseen locations → optional regional stress test.
- Website delivery is frozen and isolated in `NWP_website_handoff/`; research runs do not read from or write to it.

## Current data object

The forecast source is the Open-Meteo Previous Runs API with `gfs_seamless` and
land-cell selection. It is a location-specific product, not direct raw-GFS
GRIB. D+1/D+2/D+3 are fixed 24/48/72-hour offsets. GFS surface output is hourly
through 120 hours; solar-radiation values are preceding-hour means.

## Verify the refactor

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests\01_unit -p "test_*.py" -v
.venv\Scripts\python.exe src\s04_evaluation\analysis\audit_gfs_semantics.py
.venv\Scripts\python.exe src\s04_evaluation\analysis\audit_source_grid.py
.venv\Scripts\python.exe scripts\06_validate\validate_project.py
```

A dated copy of the former active documentation is preserved in
`docs/archive/legacy_v11_20260915/`. Local data, reports, secrets and the
standalone website delivery are excluded from the GitHub upload by `.gitignore`.
