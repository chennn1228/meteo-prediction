# Final repair acceptance

STATUS: MERGE_READY

SUPERSEDES THE PREMATURE 2026-10-04 ACCEPTANCE

Date: 2026-10-05

Branch: `codex/nwp-unified-refactor`

Audit base: `21ca995a46c75ec594c55eeee026f6e2aaf12421`

Historical recovery tag: `pre-nwp-refactor-20261002`

## Final gates

| gate | status | evidence |
|---|---|---|
| 0–219 closure ledger | PASS | `migration/final_closure_register.csv`; 220 numbered rows with evidence |
| 635-item repair ledger | PASS | `migration/final_repair_register.csv`; no pending/not-verified PASS and no blank PASS evidence |
| scientific protocol unchanged | PASS | protocol/data/features/model exports and regression checks; science-contract tests |
| real workflow integration | PASS | physical prepare/modeling/analysis modules; `implementation.py` absent |
| artifact reuse and manifests | PASS | real child-run integration uses `RunContext`, `run_pipeline`, Resolver, manifests and byte materialization |
| selective invalidation | PASS | XGBoost-only change recomputes XGBoost; gap/permutation/style-only children reuse model artifacts |
| resume integrity | PASS | receipts validate dependency, implementation, environment, inputs, outputs and actual artifact SHA |
| artifact safety | PASS | same-run tampering and incompatible cross-run bytes are rejected; final-test and diagnostic promotion gates pass |
| scientific hash | PASS | resolved config, readiness receipt and report use the same `config.config_hash` |
| configuration boundaries | PASS | unrelated registries excluded from scientific snapshot; nested unknown keys fail closed |
| CLI and local roots | PASS | parent lifecycle/diff and configured data-root inventory/probe/boundary tests |
| historical figures/reports | PASS | 94 original tag paths reproduce exact manifest SHA-256; staging is redundant and absent |
| local data reconciliation | PASS | `local_inventory_diff.csv`; no unexplained row and no source-observation byte loss |
| Git deletion reconciliation | PASS | `final_git_diff.csv`; all 322 deletions have disposition and recoverability |
| migration closure | PASS | recovery staging and all eleven phase files absent; unique unrecoverable artifacts retained |
| base-only environment | PASS | import/config/status/inventory; execution extras and pytest absent |
| independent full environment | PASS | 147 tests passed in 66.10 s |
| working environment | PASS | 147 tests passed in 69.65 s |

Detailed range-to-evidence mapping is in
`migration/post_acceptance_verification.md`.

## Scope boundary

This acceptance covers repository, code, configuration, documentation and
artifact-management repair only. It does not authorize or claim completion of
formal 20-site training, a GPU benchmark, formal multi-seed experiments, an
official full run, official result registration, or remote bulk acquisition.
