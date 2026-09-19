# 项目组织、文件与可视化规范

版本 v3.3 ｜ 2026-09-13 ｜ 适用：全仓库目录、命名、结果与图件、脚本管线。
本文件是项目内唯一的“存放规则与图件索引”文档；任何子目录都不再放 README 或散落说明，
规则、图意、脚本清单一律登记在本文件。

## 0 编号与分级管理规则

### 0.1 编号语义

目录编号与 `docs/01_research.md` 的流程编号一一对应：

| 编号 | 含义 | 对应流程 |
|---|---|---|
| 01 | 数据获取与审计 | 抓取、清洗、特征、数据可用性 |
| 02 | 实验设计 | 切分协议、月分层五折超参选择 |
| 03 | 建模 | v1_ml 与 v2–v15 训练、预测、结果 |
| 04 | 误差分析 | 时效/站点/季节/天气分组误差 |
| 05 | 稳健性 | LOSO、多种子、DM 检验 |
| 06 | 概率验证 | 可靠性、覆盖率、区间宽度、mean_pinball/crps_q7_trunc |
| 00 | 汇总与开发中间产物 | 跨版本对比、日志、smoke |

旧协议、旧布局、旧编码与冻结脚本不设归档目录：产物直接删除，需要复现时从数据与当前代码重跑。

### 0.2 src 的 `s` 前缀

`src/` 下子包统一写成 `s01_data`、`s02_experiment` … ：
Python 包名不能以数字开头，加 `s` 前缀后既能保留编号语义，又能被正常 import；
其余目录（`data/`、`reports/`、`figs/`、`scripts/`）用纯数字编号。

### 0.3 命名规则

- 模型版本（**现行新编号**，2026-09-12 按族顺序重排）：`v1_ml`、`v2_mlp`、`v3_cnn`、`v4_tcn`、
  `v5_lstm`、`v6_transformer`、`v7_autoformer`、`v8_informer`、`v9_fedformer`、`v10_itransformer`、
  `v11_patchtst`、`v12_dlinear`、`v13_timesnet`、`v14_tsmixer`、`v15_pinn`；
- **✅ 编号现状（2026-09-13 M1 已完成）**：文档与磁盘**统一用新编号**——`reports/03_modeling/`
  目录、`train_deep.py` 的 `MODEL_VNUM` 及下游脚本读盘列表均已重排（旧 `v3_lstm→v5_lstm`、
  `v4_cnn→v3_cnn`、`v5_tcn→v4_tcn`，其余不变）。历史留痕：`fixedkt_top/` 与 `v5_lstm_fixedkt/`
  早前已用新编号。服务器端目录在下次会话用新编号重跑时自然对齐。
- 预算：`lite`（本地快速复现）与 `full`（正式结论唯一来源），差异见
  `docs/01_research.md` §4.16；
- 损失：现行只有 `quantile`（七个分位数）；点损失 `mae`/`mse` 为退役协议；
- 目标：`ghi`（Himawari 真值，白天）与 `cloud`（ERA5 总云量，全时次）；
- 图件文件名：`fig_<主题>.{svg,png}`，同名文件不得表示不同含义。

### 0.4 例外目录

| 目录 | 说明 |
|---|---|
| `.cache/matplotlib` | Matplotlib 字体缓存，可随时删除重建 |
| `secrets/` | AutoDL 凭据（`autodl_password.txt`），不参与训练、不得上传 |
| `tests/01_unit`、`tests/02_integration`、`tests/03_e2e` | 测试占位目录 |
| `deploy/01_payload`、`deploy/02_results`、`deploy/99_temp` | 服务器传输载荷与回收目录 |

### 0.5 禁止事项

1. 任何子目录不得新建 README/说明文件，规则只写本文件；
2. 非全年跑、非冻结协议、lite 混入 full 的结果不得写入正式结果目录；
3. 旧协议数字不得出现在现行结论、图件或状态文档中；
4. 未标准化 PCA、裸整数类别编码（树模型）禁止出图与建模；
5. 图内不放长标题，编号与结论写在图注与本文件索引中。

### 0.6 路径解析规则

脚本不写死项目根目录，统一用向上查找标记文件的方式定位：

```python
CODE_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / "config" / "01_sites.yaml").exists())
```

所有子目录常量（`data/`、`reports/`、`figs/`）都由 `CODE_ROOT` 拼接；
移动脚本文件位置后无需再改根路径，只有“目录改名”才需要同步常量。
约束：脚本必须放在项目树内运行（`deploy/`、`scripts/` 均在树内）；
如果把脚本复制到项目之外，向上查不到 `config/01_sites.yaml` 会立即报错，属预期行为。

## 1 现行目录树

```
config/                     01_sites.yaml, 02_variables.yaml
data/
  00_geo/jiangsu.geojson    江苏省+13市边界（DataV GeoJSON，供空间图轮廓/掩膜）
  00_geo/grid_annual_ghi_2025.csv  真实 0.1° Himawari 年辐射栅格缓存(fetch_grid_ghi 生成)
  01_raw/01_gfs/            20 站分月 JSON（原 previous_runs）
  01_raw/02_era5/           20 站分月 JSON
  01_raw/03_satellite/      20 站分月 JSON（Himawari）
  02_clean/                 {site}_clean_{tag}.parquet + quality_report_*.md
  03_featured/              {site}_featured_{tag}.parquet
src/
  s01_data/fetch|clean|features/
  s02_experiment/           split_protocol.py, cv_month_balanced_quantile.py
  s03_models/train|predict/
  s04_evaluation/analysis|calibration|verification/
scripts/
  01_fetch/ 02_clean/ 03_train/ 04_evaluate/ 05_server/
reports/
  01_data_audit/{diagnostics,features,sites}/
  02_experiment/cv/month_balanced/{ghi,cloud}/
  03_modeling/00_logs|00_smoke|00_summary|vX_{model}/{lite|full}/quantile/{ghi,cloud}/
  04_error_analysis/ 05_robustness/ 06_probability/
              old_experiment_evidence,analysis_legacy}/
figs/
  01_data_audit/ 02_site_design/ 03_modeling/ 04_error_analysis/
  05_robustness/ 06_probability/
docs/                       01_research.md … 05_review.md, 本文件
literature/                 00_index.md + 01–10 论文 PDF
secrets/                 凭据
tests/                     01_unit, 02_integration, 03_e2e
deploy/                    01_payload, 02_results, 99_temp
```

## 2 数据目录规范

### 2.1 raw 数据源目录

| CLI 源键 | 目录 | 内容 |
|---|---|---|
| `previous_runs` | `data/01_raw/01_gfs/` | GFS previous runs 预报（15 组变量） |
| `era5` | `data/01_raw/02_era5/` | ERA5 辐射 + 总/低/中/高云量 |
| `satellite` | `data/01_raw/03_satellite/` | Himawari 卫星反演辐射 |

源键与目录的映射只允许在 `src/s01_data/fetch/fetch_data.py`、`verify_data.py`、
`clean/clean_data.py` 的 `SOURCE_DIRS` 常量中定义，修改时三处同步。

### 2.2 清洗与特征表 tag

| tag | 状态 |
|---|---|
| `2024-02_2026-09` | 正式建模窗口（GFS 有效预报起点 2024-02） |
| `2019-02_2026-01` | 数据可用性审计证据，禁止建模 |

旧 tag（`2024-02_2026-01`、`2024-02_2026-08`）已删除。

### 2.3 质量报告

`data/02_clean/quality_report_<site>.md` 由清洗脚本覆盖式生成，与 clean parquet 同层。

## 3 结果目录规范

### 3.1 顶层职责

| 目录 | 内容 |
|---|---|
| `reports/01_data_audit/` | 数据可用性、特征共线/PCA、站点体系、天气廓线 |
| `reports/02_experiment/` | 月分层五折 CV 的 `results.csv`、`summary.md`、`selection.json` |
| `reports/03_modeling/` | 各版本分位数结果、日志、跨模型汇总 |
| `reports/04_error_analysis/` | 分组误差分析结果 |
| `reports/05_robustness/` | 稳健性检验结果 |
| `reports/06_probability/` | 概率验证结果 |

旧协议结果不再保留归档目录：过时结果直接删除，避免被误引用。

### 3.2 建模结果层级

```
reports/03_modeling/vX_{model}/full/quantile/ghi/
    results.csv             分组指标（all/lead/station/season）
    summary.md              摘要
    test_predictions.csv    整年测试集七个分位数预测
    val_predictions.csv     月分层验证折预测（校准输入）
    calibrated/             conformal 校准后的结果
```

日志统一放 `reports/03_modeling/00_logs/{deep,quantile,smoke,server}/`；
跨模型汇总表放 `00_summary/`；smoke 证据放 `00_smoke/vX_{model}/`。

### 3.3 过时结果处理

1. 协议或编码变更后，旧结果目录整体删除，不保留、不改名、不归档；
2. 服务器队列 `run_server_full.sh` 在正式开跑前删除克隆实例上的旧 `v*_*` 结果；
3. 需要追溯历史数字时，从数据与当前冻结协议重跑，而不是引用旧表。
   协议每次变更的内容、原因与旧口径影响登记在 `docs/06_protocol_changelog.md`。

## 4 图件目录规范

### 4.1 路径规则

`figs/<步骤编号>_<主题>/[<模型版本或 00 汇总>/][<预算>/][<目标>/]fig_*.{svg,png}`。
每张图只允许由一个脚本写入，脚本docstring 必须标注结论、输入与输出路径。

### 4.2 现行图件索引

| 图 | 脚本 | 内容与结论 |
|---|---|---|
| `01_data_audit/fig_gfs_availability` | `figures_data_audit.py` | GFS 双窗口可用性热图；2019–2023 基本为空，有效预报自 2024-02 起 |
| `01_data_audit/fig_collinearity_valid` | `collinearity_ablation.py` | a=标准化 PCA 累计方差；b/c=33 特征 vs PCA95（13 个主成分，95.3% 方差）的验证 MAE/RMSE |
| `01_data_audit/fig_quantile_proto` | `figures_data_audit.py` | 原型：a=可靠性、b=中心区间覆盖率、c=单日分位区间示例 |
| `02_site_design/fig_jiangsu_20sites` | `plot_site_figs.py` | 江苏 20 站分层布点 |
| `02_site_design/fig_site_coverage_map_*` | `plot_site_figs.py` | 站点覆盖与分层代表性 |
| `02_site_design/fig_jiangsu_20sites_geo` | `plot_spatial_jiangsu.py` | 20 站叠加江苏省轮廓+市界，**Albers 等积投影**，按层代号(SI/MI/MC/NI/NC)着色 |
| `02_site_design/fig_jiangsu_annual_ghi` | `plot_spatial_jiangsu.py` | 2025 年辐射量：`fetch_grid_ghi.py` 抓取的**真实 0.1° Himawari 栅格**（非插值），Albers 投影，含省界/市界/站点 |
| `03_modeling/00_comparison/fig_weather_profiles` | `plot_weather_profiles.py` | 天气类型 × D+1/2/3 的 raw 与 q50 订正误差廓线 |
| `03_modeling/v1_ml/full/fig_v1_ml_{ghi,cloud}_agreement` | `plot_agreement_full.py` | 现行 full v1(lgbm) pred/obs 散点+残差+边缘分布（旧 lite 旧切分版已废弃至 .trash） |
| `03_modeling/00_comparison/fig_model_comparison_full` | `batch1_eval.py` | 真实 full 数据 Fig.5：pinball/RMSE/coverage/skill 按模型（绿=v1、蓝=深度） |
| `06_probability/00_comparison/fig_calibration_effect_full` | `batch1_eval.py` | 真实 full 数据 Fig.9：校准前后 coverage/width/pinball |
| `04_error_analysis/fig_exploratory_error` | `plot_batch_figures.py` | raw GFS 误差探索：bias 热图/RMSE 分天气/cloud-GHI 耦合/日变化/分 lead/站点地图 |
| `01_data_audit/fig_data_overview` | `plot_batch_figures.py` | 数据概览：单站月序列(GHI+云)+季节箱线 |
| `01_data_audit/fig_feature_correlation` | `plot_batch_figures.py` | 33 特征相关热图（层次聚类序） |
| `01_data_audit/fig_cloud_identifiability` | `plot_batch_figures.py` | GFS-vs-ERA5 云量可辨识性（r/bias/RMSE + 分水平 bias） |
| `00_preview/fig_prev_*` | `plot_pipeline_preview.py` | 全流程预览图（lite/smoke 数据，带 PREVIEW 水印，非结论） |
| `06_probability/00_comparison/fig_reliability_full` | `downstream_full.py` | Fig.6：逐样本可靠性 + PIT + 尖锐度（v1+头部深度） |
| `04_error_analysis/fig_grouped_full` | `downstream_full.py` | Fig.7：分组 MAE（季节/天气）+ 条件覆盖率 |
| `03_modeling/00_comparison/fig_case_studies_full` | `downstream_full.py` | Fig.10：clear/partly/overcast 案例分位带 |
| `05_robustness/fig_dm_conformal_full` | `downstream2_local.py` | DM 检验（BH 显著性）+ block-bootstrap conformal 覆盖 |
| `03_modeling/00_comparison/fig_model_comparison_dual` | `downstream2_local.py` | 双版 Fig.5：主 broken-kt 柱 + fixed-kt 菱形稳健性叠加 |
| `04_error_analysis/fig_shap_quantile` | `shap_analysis.py` | 分位级 SHAP 重要性热图 + τ 间 Spearman |
| `04_error_analysis/fig_shap_lead_region` | `shap_analysis.py` | 跨时效 / 跨区域 SHAP 重要性对比 |
| `04_error_analysis/fig_shap_interaction` | `shap_analysis.py` | SHAP 交互值热图 + top 交互对 |
| `04_error_analysis/fig_ablation_feature_groups` | `diagnostics_local.py` | 特征组消融 Δpinball/ΔRMSE |
| `06_probability/fig_pit_grouped` | `diagnostics_local.py` | 分组 PIT 尾质量（按 lead / 天气） |
| `06_probability/v1_ml/{ghi,cloud}/fig_probability_*` | `plot_quantile_results.py` | v1 可靠性、pinball 分解、分组覆盖率与区间宽度 |
| `06_probability/00_comparison/{ghi,cloud}/fig_deep_probability` | `plot_deep_quantile.py` | v2–v15 的 mean pinball、覆盖率、区间宽度横向对比 |

### 4.3 图意详解

**fig_quantile_proto** 是流程原型：对每个名义分位 τ 统计真值落在预测分位以下的实际比例
（可靠性，目标为对角线）；中心区间覆盖率统计 q25–q75、q10–q90、q5–q95 实际包住真值的
比例并标注平均宽度；单日示例画出 q5–q95 / q25–q75 区间与 q50 中位线。
原型偏保守/欠覆盖的问题由正式分位数族 + conformal 校准解决。

**fig_weather_profiles** 中的 “q50 订正” 指分位数模型 τ=0.5 的输出作为条件中位数点预测，
与 raw GFS 按天气类型和北京时逐时对比 MAE，用于定位最难订正的天气与时段。

### 4.4 过时图件处理

旧布局、旧协议、旧编码的图件一律删除，不保留归档目录；正文与图注只允许引用
由当前冻结协议、当前脚本、当前数据重跑出的图件。

## 5 统一绘图风格

### 5.1 唯一入口

所有出图脚本必须 import `src/s04_evaluation/analysis/plot_common.py`：

- `apply_pub_style()`：字体、坐标轴、图例等全局 rcParams；
- `PALETTE`：唯一色卡；
- `save_pub(fig, out_dir, name)`：SVG（主）+ PNG（300 dpi）双格式；
- `panel_label(ax, "a")`：统一面板编号。

### 5.2 色卡

| 用途 | hex |
|---|---|
| 主蓝（主线） | `#0F4D92` |
| 次蓝（次要方法） | `#3775BA` |
| 绿（对照/基线） | `#8BCF8B` |
| 红（信号/增益/警示） | `#B64342` |
| 青（补充面板） | `#42949E` |
| 紫（第四方法族） | `#9A4D8E` |
| 中性灰（误差/参考线） | `#767676` |

柱形图、折线图、散点图一律从 `PALETTE` 取色，禁止新造色值；
地理图允许使用专用分层色（苏南内陆 `#6ba3c7`、苏中内陆 `#77b38a`、
苏中沿海 `#d0a54f`、苏北内陆 `#b58fc2`、苏北沿海 `#d0786f`）。

### 5.3 字体与导出

- 字体 Arial / DejaVu Sans / Liberation Sans；
- SVG `fonttype='none'`（文字可编辑），PDF `fonttype=42`；
- 期刊图字号 7–9，面板标签 11 bold；
- 主格式 `.svg`，预览 `.png`（≥300 dpi）；
- 图内不放长标题，编号（Fig.1a 等）与结论写入图注/本文件。

### 5.4 投稿前 QA

1. 核对样本量 n、误差棒定义、验证窗口、数据版本；
2. 检查字体嵌入、裁切、图例遮挡、面板标签；
3. 确认配色与同义同色规则；
4. 图与 `reports/` 中对应数据同步更新。

### 5.5 出图硬标准（所有出图脚本强制，2026-09-12 起）

1. **图例不遮数据**：一律置于轴外/轴上方（`loc="lower center", bbox_to_anchor=(0.5,1.08)`），多面板统一；
2. **双轴标签**：每个 axes 必须同时有 xlabel 与 ylabel；
3. **分类刻度**：旋转 45°、`ha="right"`、`rotation_mode="anchor"`，禁止 90° 悬空错位；
4. **实心填充**：柱/直方 `edgecolor="none"` 实心；禁止 `histtype="step"` 空心描边；box/violin facecolor alpha≥0.7；
5. **PREVIEW 纪律**：lite/smoke 数据出的图必须加 PREVIEW 水印且不得作正式结论。

## 6 脚本管线

### 6.1 层级

```
src/s04_evaluation/analysis/plot_common.py     统一样式（唯一依赖）
src/s04_evaluation/analysis/*.py               现行分析与出图
src/s04_evaluation/calibration/*.py            分位数校准
src/s04_evaluation/verification/*.py           稳健性检验
scripts/{01_fetch,02_clean,03_train,04_evaluate,05_server}/
```

### 6.2 现行脚本清单

| 脚本 | 角色 | 输入 → 输出 |
|---|---|---|
| `s01_data/fetch/fetch_data.py` | 抓取 | Open-Meteo API → `data/01_raw/` |
| `s01_data/fetch/verify_data.py` | 原始数据体检 | raw JSON → 控制台报告 |
| `s01_data/clean/clean_data.py` | 清洗 | raw → `data/02_clean/` |
| `s01_data/features/features.py` | 特征工程 | clean → `data/03_featured/` |
| `s02_experiment/split_protocol.py` | 冻结切分 | 任意长表 → train/val/test 掩码 |
| `s02_experiment/cv_month_balanced_quantile.py` | 五折调参 | 特征表 → `reports/02_experiment/cv/month_balanced/` |
| `s03_models/train/train_quantile_v1.py` | v1 分位数 | 特征表 → `reports/03_modeling/v1_ml/full/quantile/` |
| `s03_models/train/train_deep.py` | v2–v15 分位数 | 特征表 → `reports/03_modeling/vX_*/` |
| `s03_models/train/cloud_impute.py` | 云量输入插补 | 特征表 → 插补后特征 |
| `s03_models/train/formal_architectures.py` | 正式结构实现 | Autoformer/Informer/FEDformer/PatchTST/TimesNet/PINN |
| `s04_evaluation/calibration/calibrate_quantiles.py` | conformal 校准 | val/test 预测 → `calibrated/` |
| `s04_evaluation/analysis/data_availability.py` | 数据门禁 | 特征表 → `reports/01_data_audit/diagnostics/` |
| `s04_evaluation/analysis/collinearity_ablation.py` | 共线消融 | 特征表 → `fig_collinearity_valid` |
| `s04_evaluation/analysis/figures_data_audit.py` | 审计图 | → `fig_gfs_availability`、`fig_quantile_proto` |
| `s04_evaluation/analysis/plot_site_figs.py` | 站点图 | → `figs/02_site_design/` |
| `s04_evaluation/analysis/plot_quantile_results.py` | v1 概率图 | v1 预测 → `figs/06_probability/v1_ml/` |
| `s04_evaluation/analysis/plot_deep_quantile.py` | 深度对比图 | v2–v15 预测 → `figs/06_probability/00_comparison/` |
| `s04_evaluation/analysis/plot_agreement.py` | 一致性散点（**已废弃**，旧 lite 旧切分） | 由 `plot_agreement_full.py` 取代 |
| `s04_evaluation/analysis/plot_weather_profiles.py` | 天气廓线 | 原型预测 → `fig_weather_profiles` |
| `s04_evaluation/analysis/error_analysis.py` | 误差分析 | 特征表 → `reports/04_error_analysis/` |
| `s04_evaluation/verification/verification_robustness.py` | 稳健性 | LOSO/多种子/DM（不含双真值）；DM 读取 `reports/03_modeling/v1_ml/full/quantile/*/per_lead_predictions.csv` 的 q0.5 |
| `s04_evaluation/verification/kt_grouping_stability.py` | 分组阈值稳定性 | → `reports/05_robustness/verification/` |
| `s04_evaluation/analysis/audit_data_quality.py` | 数据质量审计 | 特征表 → 控制台（除法爆炸/离群/越界扫描） |
| `s04_evaluation/analysis/plot_batch_figures.py` | 数据概览/误差探索/特征/云量图 | 特征表 → `figs/01_data_audit`,`figs/04_error_analysis` |
| `s04_evaluation/analysis/plot_pipeline_preview.py` | 全流程预览图 | lite/smoke 结果 → `figs/00_preview` |
| `s04_evaluation/analysis/plot_agreement_full.py` | v1 一致性图(full) | v1 预测 → `figs/03_modeling/v1_ml/full/` |
| `s04_evaluation/analysis/batch1_eval.py` | Batch-1 聚合评估 | 聚合 results + 特征表 → `reports/batch1_skill_ghi.csv`、`site_data`、Fig5/Fig9 full |
| `s04_evaluation/analysis/downstream_full.py` | 逐样本诊断+Fig6/7/10 | 逐样本预测 → `reports/06_probability/per_sample_diagnostics.csv`、site_data、图 |
| `s04_evaluation/analysis/downstream2_local.py` | DM+block-bootstrap conformal+双版Fig5 | 逐样本 → `reports/05_robustness/dm_test_ghi.csv`、`conformal_blockbootstrap.csv`、图 |
| `s04_evaluation/analysis/shap_analysis.py` | 分位/时效/站点/交互 SHAP | 特征表 → `reports/04_error_analysis/shap_*.csv`、`figs/04_error_analysis/fig_shap_*` |
| `s04_evaluation/analysis/diagnostics_local.py` | 特征组消融+残差诊断+分组PIT | 特征表/逐样本 → `ablation_feature_groups.csv`、`residual_diagnostics.csv`、`pit_grouped.csv`、图 |
| `s04_evaluation/analysis/geo_jiangsu.py` | 江苏边界/轮廓/掩膜/层代号 helper | 读 `data/00_geo/jiangsu.geojson`(DataV)+`config/01_sites.yaml` |
| `s04_evaluation/analysis/fetch_grid_ghi.py` | 抓取真实 0.1° Himawari 年辐射栅格 | Open-Meteo satellite daily → `data/00_geo/grid_annual_ghi_2025.csv` |
| `s04_evaluation/analysis/plot_spatial_jiangsu.py` | 站点图+真实栅格年辐射面(Albers投影) | 网格CSV+边界+featured → `figs/02_site_design/fig_jiangsu_{20sites_geo,annual_ghi}` |
| `scripts/05_server/run_sens_top_deep.sh` | 服务器头部深度 fixed-kt 重跑队列 | 修复版 featured → `reports/03_modeling/vX_/full`（旧编号目录） |
| `scripts/05_server/remote.py` | 服务器助手 | 上传/下载/远程执行 |
| `scripts/05_server/run_server_full.sh` | 服务器全流程 | CV → v1 → v2–v15 → 校准 → 出图 |

### 6.3 退役脚本

旧协议脚本（点损失训练器、旧 CV、旧误差/稳健性/选点绘图）已于 2026-09-11 全部删除；
`src/` 只保留当前冻结协议需要的代码。需要历史复现时按当前协议重写，不恢复旧脚本。

## 7 数据与科学红线

1. PCA 一律先标准化；未标准化 PCA 的结论与图件禁止使用；
2. 树模型类别特征一律 one-hot（`station` 20 维 + `season` 4 维），禁止裸整数编码；
3. 测试年（2025-09 起）不参与任何开发、调参、出图；
4. 正式结论只引用 `full` 预算结果，lite 仅作本地验证；
5. 每次改数据、协议或代码后，同步更新本文件、`01_research.md`、`02_project_status.md`；
6. 任何“旧协议/旧编码”数字与脚本直接删除，不留在仓库内。

## 8 版本与更新记录

- v3.3（2026-09-13）：**M1 物理重命名完成**——`reports/03_modeling/` 目录、`train_deep.py` `MODEL_VNUM`、
  下游脚本读盘列表改新编号；§0.3 与 README/08 的"磁盘仍旧编号"警告**移除**，文档与磁盘统一；
- v3.2（2026-09-13）：§0.3 模型编号改**新编号**（当时磁盘仍旧编号，加过过渡警告，已由 v3.3 消除）；
  §5.5 出图硬标准；图件索引补 12 张新图（正式/审计/预览/SHAP/诊断/DM/双版）；
  脚本清单补 8 个下游/审计/服务器脚本；plot_agreement.py 标废弃；目录示例改中性 vX；
- v3.1（2026-09-11）：标准化与早停改为协议口径（拟合子集统计、训练子集内部早停、
  校准折独立）；根目录改为向上自动查找；稳健性去掉双真值；删除全部旧遗物
  （`reports/99_archive`、`figs/99_archive`、`data/99_archive`、`src/s05_legacy`、
  `scripts/99_legacy`、旧服务器载荷），仓库不再保留归档目录；
- v3.0（2026-09-11）：全仓编号重构——`data/01_raw` 源目录编号（01_gfs/02_era5/03_satellite）、
  CV 结果归位 `02_experiment`、建模结果统一 `vX_{model}/{lite|full}/quantile/{ghi,cloud}`、
  日志/smoke 归入 `00_*`、图件按步骤与版本分级、`src` 重组为 `s01–s04` 并新设
  calibration/verification、修复 `CODE_ROOT` 深度错误与 `tree_matrix` 索引错位；
- v2.0（2026-09-10）：合并 `visualization.md`，确立色卡与绘图规范；
- v1.x：审计图更名、33 特征并轨、fig_gfs_availability 格内百分比等修订。
