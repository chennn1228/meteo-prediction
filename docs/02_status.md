# Status

## Post-acceptance repair in progress

- Unified configuration and run-scoped workflow structure exist. Artifact-level
  cross-run reuse remains under repair and is not yet accepted as complete.
- Inherited API responses are stored by source/site/month with receipts and a
  catalog. Valid data are `ready`; invalid, partial, variant, and unverified
  processed imports are retained but fail-closed as `incomplete`.
- Data and inherited figure/report inventories are complete. No formal/full
  training, network acquisition, or result regeneration occurred.
- Historical raw-audit evidence was imported from the freeze tag with exact
  hashes: 42 current audit files are materialized, of which 34 came from Git
  and 8 were exact local files. Sixty Nanjing diagnostic files were restored
  from Git and materialized read-only. Imported evidence is non-official.
- Of the original 282 recovery-required paths, 121 have exact local recovery
  and 161 remain `RECOVERY_REQUIRED`. The canonical current snapshot is
  `migration/recovery_required_register.csv`.
- The earlier 124-test acceptance is superseded by the post-acceptance audit.
  Current tests are useful regression evidence but are not final acceptance.

## Blocked

- Imported raw receipts reconstruct some request facts from configuration; the
  project must decide whether that evidence is sufficient or reacquire data
  with acquisition-time receipts before an official run.
- Models not marked eligible in `config/models.yaml` still need real-data
  validation, including one receipt-backed development mini-E2E.
- Spatial evaluation remains deferred pending its configured returned-point
  convergence and truth gates.
- `project_manifest.yaml::official_result_set` is intentionally unset.
- The 161 unresolved recovery paths have no exact local or Git-tag copy found;
  they remain recorded by path and SHA-256 and must not be represented as
  restored.

## Safe now

Configuration validation, data inventory, status inspection, and a development
run through `splits` are safe. They do not start model training. Model stages
remain opt-in through `--execute-model-stages`; official execution remains
blocked by the gate.

## Next

Validate the chosen development model set on receipt-backed data. Current
readiness values are snapshots; canonical scientific state remains in
`config/`. Only after
closing the open evidence items in `docs/06_issues.md` should an official
profile/hash and result set be registered.
