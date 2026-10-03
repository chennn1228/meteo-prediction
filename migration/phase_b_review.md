# Phase B migration-map review

Recorded: 2026-10-03 (Asia/Shanghai)

## Coverage

`migration/migration_map.csv` contains one row for every row in
`migration/repository_inventory.csv`:

- inventory rows: 7,073
- migration-map rows: 7,073
- unique migration-map paths: 7,073
- inventory paths without a migration row: 0
- migration paths without an inventory row: 0
- rows with an invalid action or a blank required decision field: 0

The required decision fields are `current_path`, `role`, `target_path`,
`action`, `dependencies`, `tests`, and `rationale`.  An empty target is allowed
only for a planned deletion that produces no replacement file.

## Planned actions

| Action | Rows | Meaning at this phase |
|---|---:|---|
| KEEP | 122 | Retain at the existing path. |
| MOVE | 5,627 | Move only in the assigned later phase after dependency checks. Most are data files awaiting receipt-backed repartitioning. |
| MERGE | 201 | Extract and verify behavior/content into the listed canonical target. |
| DELETE | 395 | Planned deletion only; no deletion is authorized until the assigned phase and tests are complete. |
| REVIEW | 728 | Retain without mutation until provenance or ownership is resolved. |

Of the `REVIEW` rows, 282 are the already-missing untracked legacy
figure/report artifacts recorded before the erroneous deletion.  Their target
is `external_recovery_required`; they cannot be restored from the Git tag.  The
446 present `REVIEW` rows are data or deployment artifacts and remain in place.

## Deletion gate

Every present file marked `DELETE` is still physically present at the end of
Phase B.  `DELETE` is a future disposition, not evidence that deletion has
already happened.  In particular:

- `archive/`, `legacy/`, old numbered source packages, old scripts, and old
  numbered tests cannot be removed until their mapped behavior and scientific
  assertions have been absorbed and verified;
- tracked `figs/` and `reports/` cannot be removed until Phase I classification
  and the Phase K residual checks;
- partial runs under `outputs/` cannot be removed until Phase I confirms they
  contain no unique scientific result;
- files with unknown provenance remain `REVIEW`, never `DELETE`.

## Phase B result

Coverage and schema checks pass.  This permits work to begin on Phase C only;
it does not authorize any Phase E-K move or deletion.
