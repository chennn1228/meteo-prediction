# Status

## Current repository and local data state

- `main` is the active branch; refactor history is recoverable from Git tags,
  including `nwp-refactor-closeout-20261006`.
- Active local data cover the configured 20 sites using immutable Previous
  Runs `gfs_seamless` bytes and Himawari shortwave-radiation bytes.
- ERA5 has left the active project and is retained only in a checksummed local
  archive outside the repository.
- Clean and feature partitions are regenerated from reusable raw bytes through
  the current receipt-backed pipeline; no network acquisition was performed.
- Raw acquisition fields and model-specific feature contracts are separate.
- CI runs the complete test suite on every push and pull request.

## Blocked

- Models not marked eligible in `config/models.yaml` still need real-data
  validation, including one receipt-backed development mini-E2E.
- Spatial evaluation remains deferred pending its configured returned-point
  convergence and truth gates.
- `project_manifest.yaml::official_result_set` is intentionally unset.

## Safe now

Configuration validation, local data inventory, and development execution are
safe. Model stages remain opt-in through `--execute-model-stages`; official
execution remains blocked by the gate.

## Next

Validate the chosen development model set on receipt-backed data. Canonical
scientific state remains in `config/`. Province-wide GFS025 acquisition is out
of scope and must not begin without an explicit user instruction.
