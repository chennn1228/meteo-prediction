# Final repair acceptance

STATUS: MERGE_READY

Date: 2026-10-04

Branch: `codex/nwp-unified-refactor`

Starting safety tag: `pre-final-repair-20261004`

Historical recovery tag: `pre-nwp-refactor-20261002`

No formal 20-site training, GPU benchmark, formal multi-seed run, official full
run, network acquisition, or official result registration was performed.

| gate | status | evidence |
|---|---|---|
| evidence recovery | PASS | 121/282 exact recoveries; 161 explicitly `RECOVERY_REQUIRED`; two import manifests validate 42 raw-audit and 60 Nanjing diagnostic files with exact SHA-256, read-only materialization, and non-official status |
| protocol invariance | PASS | `protocol_regression_check.json` |
| data-contract invariance | PASS | `final_data_contract_export.json` and baseline comparison |
| model-registry invariance | PASS | `final_model_registry_export.json`; IDs/groups unchanged, no eligibility upgrade, six trials |
| config ownership | PASS | manifest points to exactly six scientific YAML files |
| strict parsing | PASS | duplicate-key, unknown-key, enum, type, and unsafe-ID tests |
| RunConfig | PASS | immutable resolved selection-only records and hashes |
| RunContext | PASS | fixed run tree, resolver, catalog, lifecycle, and immutable metadata |
| site selection | PASS | mutually exclusive CLI forms and deterministic registered density selection |
| output tree | PASS | `00_meta` through `09_report` |
| hash isolation | PASS | unselected model/site isolation and three-model child dependency tests |
| implementation fingerprint | PASS | feature/model/analysis/visualization source fingerprint tests |
| artifact resolver | PASS | exact scope, SHA, implementation, environment, and official eligibility checks |
| time-scope safety | PASS | final-test artifacts rejected from selection |
| official reuse safety | PASS | diagnostic artifacts do not auto-upgrade to official |
| run lifecycle | PASS | terminal states read-only; successful receipts write-once |
| child runs | PASS | parent, change reason, and changed dependencies recorded |
| cross-run reuse | PASS | copy materialization plus exact SHA verification exercised in bounded tests |
| environment compatibility | PASS | incompatible environment fingerprint rejects reuse |
| model-level reuse | PASS | Ridge/LightGBM stable and XGBoost/aggregate invalidated in bounded synthetic child test |
| data catalog | PASS | missing, ambiguous, incomplete, mismatched, and idempotent paths fail closed |
| raw immutability | PASS | same bytes reuse; conflicting bytes reject overwrite |
| data partitions | PASS | source/site/month raw and dependency/site/month processed layout |
| receipts | PASS | canonical sidecars and real content hashes |
| service coordinate | PASS | requested coordinates provenance-only; returned coordinates drive physics/spatial identity |
| spatial protocol | PASS | three levels retained; 707/654/53 preserved as historical probe evidence, never a hard-coded target |
| official readiness | PASS | individual blockers produce `BLOCKED`; synthetic all-ready gate test does not train |
| report | PASS | run-scoped summary and report manifest |
| dependencies | PASS | clean install from `pyproject.toml`; resolved lock regenerated; no active Paramiko |
| gitignore | PASS | root figures/reports/scripts visible; data/outputs ignored; literature index tracked |
| literature | PASS | tracked index and valid documentation link |
| figures | PASS | run-scoped SVG/PNG/index contract; source tables remain in metrics/analysis |
| docs | PASS | exactly seven active documents; no second scientific authority |
| local diff | PASS | 11,558 unchanged, one SHA-identical move, 150 additions, zero removed, zero changed |
| Git deletion diff | PASS | 340 deletions individually mapped and recoverable from immutable tags |
| tests | PASS | ordered groups passed; final complete suite 124 passed |
| clean environment | PASS | Python 3.13.2, no system site packages, import/structural/full pytest all pass |
| remaining blockers | PASS (documented) | official scientific execution remains blocked by unresolved data/model/spatial/result-registration evidence in `docs/06_issues.md` |

Merge readiness applies to this repository refactor only. It does not promote
diagnostic or imported evidence into an official scientific result.
