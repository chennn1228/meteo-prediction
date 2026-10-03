# Final acceptance

Acceptance date: 2026-10-04 (Asia/Shanghai).

## Scope and scientific safety

- No formal/full training was started.
- No network data acquisition was performed.
- No inherited result was regenerated or presented as current evidence.
- The frozen scientific protocol was preserved while configuration ownership,
  code ownership, data layout, and run artifact ownership were unified.

## Required deliverables

1. Unified repository: `config/`, `src/nwp/`, `data/`, and `outputs/` implement
   the four ownership layers; supporting docs/tests/migration evidence are
   separated.
2. Directory tree: `migration/final_directory_tree.md`.
3. Old-to-new map: `migration/migration_map.csv`, 7,127 rows, legal action on
   every row.
4. Canonical configuration: the six requested YAML files plus
   `local.example.yaml`; `project_manifest.yaml` is only the entry point.
5. Sole CLI: `python -m nwp` with `validate`, `data`, `run`, `evaluate`,
   `figures`, and `status`.
6. Development workflow: real dependency-aware stages from validation through
   report, with explicit opt-in for model execution.
7. Data evidence: `data/data_inventory.csv` has 11,439 `KEEP` rows;
   `data/catalog.json` has 5,705 entries (5,619 raw, 43 clean, 43 features),
   comprising 4,281 `ready` and 1,424 `incomplete` entries.
8. Figure/result evidence: `migration/figure_inventory.csv` has 428 rows. The
   146 present tracked files were verified against the freeze tag before
   removal. The 282 earlier-lost untracked files remain `RECOVERY_REQUIRED`.
9. Documentation: the exact seven-file active documentation set is present.
10. Tests: the post-cleanup suite completed with `105 passed in 39.33s` using
    `D:\anaconda3\python.exe -m pytest -q -p no:cacheprovider` and a disposable
    base temporary directory.
11. Residual searches: `migration/final_residual_search.md`; path and
    historical-word searches are zero in active logic, and workflow hard-coded
    parameter search is zero.
12. Official blockers: recorded below and in `docs/06_issues.md`.

## CLI verification

- `python -m nwp validate --profile nanjing_cpu_diagnostic --mode config`:
  passed and resolved a development configuration.
- `python -m nwp validate --mode structural`: 11/11 checks passed.
- `python -m nwp data inventory`: `KEEP=11439`.
- `python -m nwp status`: returned the four registered profiles and the single
  outputs root.
- `python -m nwp validate --mode data_ready`: correctly failed closed at 13/14
  checks. All 1,860 required current-period partitions passed coverage and
  receipt integrity; the sole failure requires explicit acceptance of
  migration-imported acquisition provenance or reacquisition.

## Deletion and recovery evidence

The removed historical trees, numbered packages, scripts, tests, configuration,
figures, and reports remain recoverable from
`pre-nwp-refactor-20261002` where they were tracked. For the 146 tracked
figure/report artifacts, 54 files matched the tag byte-for-byte and 92 text
files matched exactly after accounting for Git's LF-to-CRLF checkout
conversion. No other difference existed.

The 282 already-missing items were never tracked. Their paths, sizes, prior
SHA-256 values, classifications, and recovery status are preserved in the
figure inventory; Git cannot recover them.

## Remaining official-research blockers

- accept the reconstructed imported receipts or reacquire with immutable
  acquisition-time receipts;
- validate all intended comparison models on real data and accept a bounded
  receipt-backed development mini-E2E;
- reproduce the complete pinned training environment;
- complete the returned-service-point and spatial truth gates;
- lock an official profile/config hash and deliberately register an official
  result set;
- recover the 282 local-only artifacts from a matching external backup, if one
  exists.
