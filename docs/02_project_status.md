# Project status

Version 2.0.0-provisional | 2026-09-15

## Completed in the v2 refactor

- Created `project_manifest.yaml` as the machine-readable Single Source of Truth（单一事实源）.
- Replaced the formal split with nested purged rolling-origin validation（嵌套式带间隔滚动起点验证）.
- Bounded the final test to 2025-09-01 through 2026-08-31; 2026-09 is no longer silently included.
- Derived the primary purge as 168 h lookback + 72 h lead = 240 h; registered 7/10/14-day sensitivity.
- Separated fit, early stopping, calibration and final test in time.
- Added additive quantile-calibration sign tests; current code uses `y - q_hat` and `q + delta`.
- Retired month-balanced tuning from formal use; legacy execution now requires `--allow-legacy`.
- Added low/mid/high GFS cloud fields to the v2 data contract (18 forecast variables total).
- Separated requested site coordinates from returned service coordinates in the clean-data contract.
- Audited the 20 stations: 20 distinct returned locations; maximum request offset about 7.79 km.
- Sampled 3997 in-province request locations at 0.05° and enumerated 707 distinct
  Open-Meteo land-selected service cells; this is not an authoritative raw-GFS grid count.
- Applied the hourly-truth gate: exactly 0 province service cells are currently
  eligible for formal evaluation because full-period hourly Himawari has not been fetched.
- Disabled the dimensionally invalid/unsupported PINN clear-sky hard ceiling.
- Added 8 dependency-free unit tests; all pass.
- Added reproducible website exporter, schemas and a working zero-dependency website demo.

## Current artifacts

| Artifact | Status | Meaning |
|---|---|---|
| `project_manifest.yaml` | active | protocol/data/model/feature SSOT |
| `docs/01_research.md` | active | scientific argument and protocol |
| `docs/03_model_config.md` | active | model registry and equal budget |
| `reports/01_data_audit/gfs_semantics/` | active audit | local + official temporal semantics |
| `reports/01_data_audit/source_grid/` | active audit | 20-site mapping + sampled province service-cell inventory |
| `NWP_website_handoff/` | sealed delivery | standalone website package; isolated from active research |
| existing `reports/03_modeling` | legacy/provisional | not v2 official evidence |
| existing `figs/` | provisional/reference | visual reference, not v2 official result set |

`official_result_set` is deliberately `null` until the new protocol is executed.

## Blocked or incomplete evidence

| Priority | Item | Blocking condition | Required output |
|---|---|---|---|
| P0 | v2 data regeneration | archived raw lacks low/mid/high GFS cloud | new raw/clean/featured version |
| P0 | formal model comparison | v2 features and GPU equal-budget run not completed | outer/inner OOF + final frozen results |
| P0 | province-wide test | full-period hourly Himawari truth not gated | eligible grid list + missingness report |
| P0 | recalibration | predictions under new folds do not exist | time-ordered calibrated predictions |
| P1 | service-grid convergence | current 0.05° lattice is an empirical inventory, not an exact raw grid | stable counts at finer meshes |
| P1 | raw-GFS grid claim | no direct GRIB/archive pipeline | raw grid audit or explicit exclusion |
| P1 | PINN redesign | site-adapted clear sky/tolerance untested | constraint-ablation report |

These are scientific/compute dependencies, not documentation excuses. No placeholder result is promoted to official.

## Next executable sequence

1. Re-fetch Previous Runs fields for 2024-02 through 2026-08 using the 18-variable contract.
2. Regenerate clean/featured data and run temporal/semantic/missingness gates.
3. Run `cv_nested_rolling_quantile.py` and the equal-budget DL runner on the same inner folds.
4. Freeze features, model settings, calibration strategy and spatial design in the manifest.
5. Retrain on final fit/early-stop slices; fit calibration on 2025-08-11–08-31.
6. Evaluate the untouched 2025-09 through 2026-08 test once.
7. Fetch/gate hourly Himawari truth for province candidates; then run Level 2 and density experiments.
8. Mark and freeze the new scientific result set; website maintenance is outside the research workflow.

## Reproduction commands

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests\01_unit -p "test_*.py" -v
.venv\Scripts\python.exe src\s04_evaluation\analysis\audit_gfs_semantics.py
.venv\Scripts\python.exe src\s04_evaluation\analysis\audit_source_grid.py
.venv\Scripts\python.exe scripts\06_validate\validate_project.py
```

The old full server script is not a v2 reproduction path until its month-balanced calls are replaced.
