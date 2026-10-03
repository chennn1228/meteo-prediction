# Phase K — migration-layer cleanup

Status: implementation complete; final verification evidence is recorded in
`migration/final_acceptance.md`.

Removed from the active repository after ownership and recovery checks:

- both historical code/document trees, after confirming zero active imports and
  exact presence in `pre-nwp-refactor-20261002`;
- all numbered source packages and compatibility adapters;
- all standalone execution scripts;
- numbered migration-period test directories;
- the two replaced configuration files;
- 146 tracked inherited figure/report artifacts, after path, tracking, current
  SHA-256, and freeze-tag content verification;
- migration-only inventory/import/generator wrappers.

The 428-row figure inventory is retained at
`migration/figure_inventory.csv`. It records 146 verified removals and 282
already-missing local-only artifacts as `RECOVERY_REQUIRED`. The latter were
not restored and are not represented as recovered.

The 7,127-row old-to-new disposition table is retained at
`migration/migration_map.csv`. Evidence documents, data import plans, and the
isolated recovered-artifact evidence remain under `migration/`; executable
migration wrappers do not.
