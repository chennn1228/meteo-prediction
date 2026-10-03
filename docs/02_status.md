# Status

## Completed

- Unified configuration, core contracts, data/features/splits, model and
  experiment interfaces, executable workflow, evaluation, visualization, and
  run-scoped outputs are implemented.
- Inherited API responses are stored by source/site/month with receipts and a
  catalog. Valid data are `ready`; invalid, partial, variant, and unverified
  processed imports are retained but fail-closed as `incomplete`.
- Data and inherited figure/report inventories are complete. No formal/full
  training, network acquisition, or result regeneration occurred.
- The post-cleanup full suite passes; the exact command and result are recorded
  in `migration/final_acceptance.md`.

## Blocked

- Imported raw receipts reconstruct some request facts from configuration; the
  project must decide whether that evidence is sufficient or reacquire data
  with acquisition-time receipts before an official run.
- Models not marked eligible in `config/models.yaml` still need real-data
  validation, including one receipt-backed development mini-E2E.
- The reproducible training environment is incomplete on the current machine;
  official readiness must verify all pinned runtime dependencies.
- Spatial evaluation remains deferred pending its configured returned-point
  convergence and truth gates.
- `project_manifest.yaml::official_result_set` is intentionally unset.
- The earlier erroneous pass removed 282 untracked inherited figure/report
  artifacts. Their hashes and paths remain recorded as `RECOVERY_REQUIRED` in
  `migration/figure_inventory.csv`, but they are not recoverable from Git.

## Safe now

Configuration validation, data inventory, status inspection, and a development
run through `splits` are safe. They do not start model training. Model stages
remain opt-in through `--execute-model-stages`; official execution remains
blocked by the gate.

## Next

Validate the chosen development model set on receipt-backed data. Only after
closing the open evidence items in `docs/06_issues.md` should an official
profile/hash and result set be registered.
