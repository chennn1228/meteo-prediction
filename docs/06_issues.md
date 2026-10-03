# Open issues

| ID | priority | problem | impact | required evidence | status |
|---|---|---|---|---|---|
| P0-DATA-01 | P0 | Imported raw receipts reconstruct some request facts rather than preserving acquisition-time receipts. | May be insufficient for an official data provenance claim. | Explicit acceptance audit or reacquisition with immutable acquisition receipts. | open |
| P0-MODEL-01 | P0 | Declared comparison models remain ineligible except where the registry says otherwise. | Prevents an official model comparison. | Model-specific contract checks and real-data validation receipts. | open |
| P0-E2E-01 | P0 | No current-pipeline real-data development mini-E2E has been accepted. | End-to-end scientific integration remains unverified. | One bounded, receipt-backed development run with artifact hashes; no final-test selection. | open |
| P0-ENV-01 | P0 | The current runtime does not satisfy the complete pinned training environment. | A formal run would not be reproducible. | Successful clean-environment dependency and runtime provenance audit. | open |
| P0-OFFICIAL-01 | P0 | No locked official profile/hash or official result set exists. | Official execution must remain blocked. | All readiness checks pass and an official result set is deliberately registered. | open |
| P1-SPATIAL-01 | P1 | Returned-service-point convergence and the configured spatial truth gate are incomplete. | Province-wide spatial claims remain deferred. | Converged returned-point registry and truth-coverage audit. | open |
| P1-RECOVERY-01 | P1 | 282 inherited untracked figure/report files were deleted in an earlier erroneous pass. | Some historical local evidence cannot be inspected or migrated. | Restore from an external backup matching the recorded SHA-256 values, if one exists. | open |
