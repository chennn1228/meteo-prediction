# 模型注册与调参政策

Version 2.0.0-provisional | Generated logically from `project_manifest.yaml`

Do not edit model counts or versions here first. Change the manifest and run the manifest tests.

## GFS Value Ladder registry

| Family | Stable semantic IDs | Count |
|---|---|---:|
| Non-NWP baseline | `climatology`, `persistence`, `smart_persistence`, `optimal_convex` | 4 |
| Raw NWP | `raw_gfs` | 1 |
| Statistical | `bias_correction`, `linear_mos`, `ridge_mos` | 3 |
| Tree ML | `lgbm`, `xgboost` | 2 |
| Deep | `mlp`, `cnn`, `tcn`, `lstm`, `transformer`, `autoformer`, `informer`, `fedformer`, `itransformer`, `patchtst`, `dlinear`, `timesnet`, `tsmixer` | 13 |
| Experimental constrained | `pinn` | 1 |
| **Total** |  | **24** |

The deep architecture benchmark contains **14** models including PINN. Internal versions v2–v15 are provenance only; website and reports use stable semantic IDs.

## Internal deep mapping

| Internal | Semantic ID | Status |
|---|---|---|
| v2 | mlp | benchmark |
| v3 | cnn | benchmark |
| v4 | tcn | benchmark |
| v5 | lstm | benchmark |
| v6 | transformer | benchmark |
| v7 | autoformer | benchmark |
| v8 | informer | benchmark |
| v9 | fedformer | benchmark |
| v10 | itransformer | benchmark |
| v11 | patchtst | benchmark |
| v12 | dlinear | benchmark |
| v13 | timesnet | benchmark |
| v14 | tsmixer | benchmark |
| v15 | pinn | experimental constrained model |

## Equal-budget policy

| Rule | Value |
|---|---|
| Candidate trials | 6 per model |
| Tuning data | current outer-training prefix only |
| Inner validation | 3 purged rolling folds |
| Selection metric | mean pinball over registered quantiles |
| Tuning seed | 0 |
| Final repeated seeds | 0, 1, 2, 3, 4 for top models |
| Early stopping | identical causal slice policy; patience reported |
| Compute report | wall time, device, peak memory, epochs, sample count |

DL search uses two learning rates, two model scales and at most two dropout values while respecting the six-trial cap. Tree ML has six preregistered configurations. Ridge has six alpha values. No model receives outer-validation feedback during tuning.

## Feature modes

- 所有正式与诊断模型均禁止把 `station_id`、`location_id`、`source_grid_id` 及其编码送入特征矩阵；这些字段只作分组与溯源。
- 使用 API 实际返回的 GFS 服务点经纬度与海拔计算物理特征；请求坐标只作溯源，不参与太阳几何。
- Feature selection uses group evidence in inner folds. SHAP is interpretation-only.

## 当前 CPU 准入状态

固定定义、无需伪造六候选：`climatology`、`persistence`、`smart_persistence`、`optimal_convex`、`raw_gfs`、`bias_correction`、`linear_mos`。需要六个预注册候选：`ridge_mos`、`lgbm`、`xgboost`。除 `raw_gfs` 外，目前均为 `pending_validation`，不得进入正式比较；GPU 模型状态不影响 CPU 专属门禁。概率分位数是主评价，MAE/RMSE/Bias/R² 是点预测辅助指标，RMSE skill 必须标明参照模型。

## Seed policy

- seed 0 selects configurations;
- no seed is selected using final test performance;
- seeds 0–4 quantify final stochastic variability for top methods;
- report every attempted trial, including failed and early-terminated trials.

## Current selection status

No v2 selection file is official. Existing `reports/02_experiment/cv/month_balanced/*/selection.json` belongs to the retired protocol and is retained only in legacy artifacts.
