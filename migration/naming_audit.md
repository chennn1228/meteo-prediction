# Naming audit

Date: 2026-10-06 (Asia/Shanghai)

Scope: `project_manifest.yaml`, `config/`, `src/`, `tests/`, `docs/`,
`README.md`, and `migration/`.

“Scientific hash” below means the dependency fingerprints that decide whether
data, features, tuning, models, or predictions can be reused. The run config
hash changes when its canonical metadata changes, but no scientific parameter,
algorithm, data byte, or immutable historical artifact changed. Data and
feature display-name fields are deliberately excluded from their scientific
dependency fingerprints and are covered by a regression test.

| old name | canonical name | reason | affected files | scientific hash impact | historical adapter |
|---|---|---|---|---|---|
| `jiangsu-gfs-ghi-postprocessing` | `jiangsu_gfs_ghi_postprocessing` | machine IDs use `snake_case` | `project_manifest.yaml` | No scientific-semantic change; run config identity changes | No |
| `2.1.0-provisional-15var-archive-aligned` | `protocol_version: 2.1.0`; `protocol_status: provisional` | keep the managed semantic version separate from status and scientific facts | `project_manifest.yaml`; `src/nwp/core/config.py`; tests | No scientific-semantic change; run config identity changes | No |
| `openmeteo-point-15var-archive-aligned-2024-02_2026-08-v1` | `openmeteo_point_15var_archive_aligned_2024_02_2026_08` | remove artificial revision suffix and normalize the active data ID | `config/data.yaml`; dependency regression test | No; the `version` label is excluded from the data dependency fingerprint | No |
| `feature-mechanism-raw-ratios` | `feature_mechanism_raw_ratios` | normalize the active feature-build ID | `config/features.yaml`; dependency regression test | No; the `version` label is excluded from the feature dependency fingerprint | No |
| model `internal_version: v2` through `v15` | stable registry model IDs (`mlp`, `cnn`, `tcn`, `lstm`, `transformer`, `autoformer`, `informer`, `fedformer`, `itransformer`, `patchtst`, `dlinear`, `timesnet`, `tsmixer`, `pinn`) | implementation order is not model identity; Git and implementation fingerprints carry implementation provenance | `config/models.yaml`; `src/nwp/core/config.py`; `tests/unit/test_manifest.py` | No scientific algorithm change | No |
| `deep_shared_prototype` | `deep_shared` | workflow status must not be encoded in the search-space ID | `config/models.yaml`; `src/nwp/core/config.py`; `src/nwp/experiment/tuning.py` | No; search-space values are byte-for-byte unchanged | No |
| `fold-cloud-lgbm-v2` | `fold_local_cloud_lightgbm` | name the fit scope and implementation instead of an artificial revision | `src/nwp/features/preprocessing.py`; tests exercise emitted metadata | No; implementation and parameters are unchanged | No |
| cloud metadata key `model_version` | `imputer_id` | identify the cloud imputer directly rather than treating it as a prediction-model version | `src/nwp/features/preprocessing.py` | No | No |
| `validated_hourly_himawari_v1` | `validated_hourly_himawari` | active evidence schema has one semantic canonical ID | `src/nwp/evaluation/spatial.py`; spatial contract tests | No | No; immutable historical evidence remains untouched |
| prediction field `protocol_revision` | `protocol_version` | one canonical field for the same protocol identity across config and prediction provenance | `config/protocol.yaml`; prediction, calibration, evaluation, workflow, and tests | No scientific-semantic change; the active machine contract changes coherently | No |
| `--max-new-batches` / `max_new_batches` | `--max-batches` / `max_batches` | the probe scope already means newly fetched batches; avoid state words in a machine ID | CLI, service probe, and unit test | No | No |
| test-only `data-v1`, `features-v1`, `p-v2`, `exp-1`, `exp-2` | semantic synthetic fixture IDs | tests must not normalize active contracts around artificial revision examples | evaluation/calibration/prediction unit tests | No | No |
| hyphenated synthetic run/result/service IDs | `snake_case` fixture IDs | apply the same machine-ID convention to executable tests and validation probes | workflow, core, data, spatial, and reuse tests; readiness validation | No | No |

## Remaining search matches

Every remaining match from the requested `v1`/`v2`/`vN`, `new`/`old`/`latest`,
and `temp`/`tmp` scan belongs to one of these explicit classes:

| remaining class | locations | reason retained |
|---|---|---|
| formal version | `project_manifest.yaml` | `2.1.0` is the managed protocol version; status is separate |
| third-party API version | `src/nwp/data/fetch.py`; `src/nwp/data/audit.py` | `/v1/` is part of the official Open-Meteo endpoint |
| atomic-write marker | `.tmp` paths and `temporary` local variables in `src/nwp/core/`, `src/nwp/data/`, `src/nwp/features/`, and `src/nwp/workflow/stages/common.py` | transient write safety mechanism, not a persistent object ID |
| operating-system/library temporary facility | `tests/base_bounded_splits_smoke.py`; `tests/unit/test_visualization.py` | Windows `TEMP` and Python `TemporaryDirectory` are external runtime interfaces |
| historical compatibility test | `tests/unit/test_data_contract.py`; `tests/unit/test_spatial_contracts.py` | deliberately constructs an old layout or stale file to verify rejection/migration behavior |
| naming regression guard | `tests/unit/test_manifest.py` | mentions the removed `internal_version` key only to assert that it cannot return |
| ordinary prose or local comparison variables | `docs/04_repository_guide.md`; `src/nwp/models/baselines.py`; `src/nwp/features/engineering.py`; `src/nwp/core/config.py`; focused tests | natural language and short-lived `old`/`new` comparison variables are not machine object names |
| immutable provenance and historical paths | existing `migration/*.csv`, `migration/*.json`, `migration/*.txt`, `migration/recovered_artifacts/`, and `migration/final_repair_baseline/` | original names, paths, hashes, Git/tag evidence, and recovered bytes must remain exact |

No active model ID contains an artificial version. No active object name encodes
`prototype`, `provisional`, `architecture_only`, `validated`, `deprecated`,
`imported`, `incomplete`, or `official_eligible`; those terms remain only in
explicit status fields or status values. No compatibility alias was added.
