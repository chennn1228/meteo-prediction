# Post-acceptance verification ledger

Date: 2026-10-05

This ledger supersedes the premature acceptance dated 2026-10-04.  A row is
eligible for `PASS` only when its named evidence exists and the final clean
test run succeeds.

## Instruction evidence map

| instruction range | verification evidence |
|---|---|
| 0–6 | `migration/final_acceptance.md`; `tests/unit/test_repair_register.py` |
| 7–22 | `tests/unit/test_config.py`; `tests/unit/test_core.py`; `tests/unit/test_workflow_contract.py` |
| 23–42 | `tests/integration/test_workflow.py`; `tests/unit/test_core.py`; workflow receipts validated by the full suite |
| 43–84 | `tests/integration/test_child_run_reuse_workflow.py`; `tests/unit/test_final_repair_contracts.py`; non-empty manifests and byte-backed materialization asserted |
| 85–91 | `tests/unit/test_fixed_models.py::test_raw_alias`; implementation-based dispatch |
| 92–100 | `tests/unit/test_repository_layout.py`; `implementation.py` absent; physical stage modules imported by the pipeline |
| 101–112 | `tests/e2e/test_cli.py`; parent lifecycle, recursive config diff, configured local data root and boundary assertions |
| 113–122 | `tests/unit/test_config.py`; `tests/unit/test_core.py`; nested unknown-key and source-fingerprint assertions |
| 123–145 | `tests/integration/test_child_run_reuse_workflow.py`; real `RunContext`, `run_pipeline`, Resolver, manifest, and materialization calls |
| 146–153 | `tests/unit/test_config.py`; `tests/integration/test_workflow.py`; resolved/readiness/report scientific hash equality |
| 154–160 | `tests/unit/test_repository_layout.py`; staging and eleven phase files absent; cleanup register retains only long-term records |
| 161–178 | `README.md`; `migration/base_environment_validation.txt`; `migration/full_environment_validation.txt`; CLI official-readiness tests |
| 179–188 | `migration/import_manifests/*.json`; `tests/unit/test_manifest.py`; 94 original tag paths reproduced byte-for-byte |
| 189–194 | `migration/protocol_regression_check.json`; `migration/features_regression_check.json`; final protocol/data/features/model exports; full science-contract tests |
| 195–200 | `migration/local_inventory_after.csv`; `migration/local_inventory_diff.csv`; exact move of `jiangsu_main.geojson` to `jiangsu.geojson` has identical SHA-256; zero changed bytes and zero unexplained removals |
| 201–214 | config/core/workflow/reuse/tamper/final-test/official-gate tests in the final full suite |
| 215–219 | repository-layout test, zero-unverified-register test, clean-environment logs, final Git reconciliation and this ledger |

## Local-data reconciliation

The baseline has 11,559 rows and the final inventory has 11,596 rows. There
are 11,557 byte-identical paths, 38 additions, one regenerated derived
inventory file, and one path move. `data/data_inventory.csv` is the regenerated
local inventory audit, not source observations. `data/registry/geography/jiangsu_main.geojson` moved to
`data/registry/geography/jiangsu.geojson`; both hashes are
`983d15e1964dbd349ec3aac41a30c7b252fbf2b87ff3bca19af2efcd6f16e1b9`.
No source-observation bytes were removed or changed, and every non-identical
row has an explicit disposition in `migration/local_inventory_diff.csv`.

## Historical imported evidence

The raw-audit and Nanjing diagnostic manifests retain `source_tag`, original
`figs/` or `reports/` source path, and SHA-256.  The manifest test executes
`git show <source_tag>:<source_path>` for all 94 historically tracked files
and compares bytes.  Recovery staging is therefore redundant and was removed;
the source tag remains the canonical recoverable copy.

## Prohibited execution

No formal 20-site training, GPU benchmark, formal multi-seed experiment,
official full run, official result registration, or remote bulk acquisition
was performed.
