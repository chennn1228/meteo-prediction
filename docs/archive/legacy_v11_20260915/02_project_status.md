# 项目状态

更新 2026-09-13 ｜ 协议见 `docs/01_research.md`，目录与图件规则见 `docs/04_project_organization.md`。

## 1 正式协议

- 测试年：2025-09-01 至 2026-08-31；
- 训练池：2024-02-01 至 2025-08-31；
- 验证：训练池内按月等分五折连续时段，每折约 20%，覆盖 12 个日历月；
- 最终训练：四折训练，一折校准，早停集取自训练集内部；
- 输出：七个分位数（0.05/0.10/0.25/0.50/0.75/0.90/0.95），损失为 pinball；
- 评估：mean_pinball、crps_q7_trunc、覆盖率、区间宽度、分位交叉率、可靠性、PIT、
  q50 的 MAE/RMSE。

## 2 数据状态

- 20 站 GFS 有效预报自 2024-02 起；正式特征表 tag=`2024-02_2026-09`，每站 68,544 行；
- raw 目录按源编号：`01_gfs`、`02_era5`、`03_satellite`，源键映射见
  `docs/04_project_organization.md` §2.1；
- `2019-02_2026-01` 特征表保留为数据可用性审计证据，禁止建模；
  旧 tag（`2024-02_2026-01`、`2024-02_2026-08`）已删除；
- 云量输入缺失采用三级插补（同时刻其他时效中位数 → LightGBM → 站点/时效/月中位数），
  实现于 `src/s03_models/train/cloud_impute.py`；
- 辐射负值置零，云量裁剪至 0–100。

## 3 目录与代码重构（2026-09-11 完成）

- 全仓按流程编号重组：`data/01_raw` 源目录编号、
  CV 归位 `reports/02_experiment/cv/month_balanced`、建模结果统一为
  `reports/03_modeling/vX_{model}/{lite|full}/quantile/{ghi,cloud}`，
  日志/smoke 归入 `00_logs|00_smoke`；
- `src` 重组为 `s01_data`、`s02_experiment`、`s03_models`、`s04_evaluation`（analysis/
  calibration/verification）；`split_protocol.py` 移入 `s02_experiment`，
  `calibrate_quantiles.py` 移入 `s04_evaluation/calibration`；
- 修复 36 个脚本的 `CODE_ROOT` 深度错误（`parents[N]` 少一层，导致路径指向 `src/data/`）；
- 根目录改为向上查找 `config/01_sites.yaml` 自动定位，移动脚本文件后不会再断路径；
- 修复 `tree_matrix` 的索引错位：`pd.get_dummies(Categorical)` 丢失原索引，
  非连续索引下会与标签错位；`train_v1` 已修正并统一 one-hot（20 站 + 4 季），
  未使用的点损失云量训练器删除；
- 标准化与早停改为协议口径：数值特征/目标的均值方差只用拟合子集计算；
  早停集改为训练子集内部按日期排序的尾部 15%，校准折 0 只用于 conformal 校准；
- 删除与 `fig_collinearity_valid` 重复的 `fig_feature_diagnostics` 及其出图函数；
- 概率指标修正：`crps` 误名改为 `mean_pinball`（未加权平均 pinball）+
  `crps_q7_trunc`（截尾分位网格积分）；结果表新增 `crossing_rate`（排序前分位交叉率）；
- v1 线性族修正：`linear`=OLS、`ridge`=Ridge（α 并入月分层五折 CV 网格），
  原实现中二者实为 HuberRegressor；
- v1 端到端 smoke 通过（`--smoke`：3% 抽样 + 200 棵树，ghi/cloud 两目标），
  新指标列与 `crossing_rate` 正常产出；smoke 中 lgbm 排序前交叉率约 33%
  （D+1），说明"独立分位模型不自洽"真实存在，正式结果必须报告该列；
- 开机前排查修复（2026-09-11 晚）：
  ① `run_server_full.sh` 原先漏掉深度模型的 conformal 校准步骤，已补；
  ② 校准结果原先没有任何评估/出图读取（白做），现输出
     `calibrated/results_calibrated.csv`，`plot_deep_quantile.py` 优先读取校准结果；
  ③ CV 的 `--smoke` 会污染正式 `selection.json`，现 smoke 写入 `cv/00_smoke/`，
     且训练器只接受 `n_folds=5 且 smoke=false` 的选择文件；
  ④ 五折 CV 脚本缺 smoke 通道且存在索引 bug（smoke 下 `losses[1]` 越界），已修并跑通；
  ⑤ `requirements.txt` 缺 `torch`、`scipy`、`paramiko`，已补；
  ⑥ full 预算统一 168 小时输入（全部深度模型），早停与拟合子集之间加 15 天间隔，
     深度模型 pinball 损失加入分位单调惩罚；
- 服务器队列新增"预检"步骤：6 个正式结构（Autoformer/Informer/FEDformer/PatchTST/
  TimesNet/PINN）先在 GPU 上各跑 1 个 smoke，任一失败立即停止并写
  `00_logs/server/preflight/fail.log`；PatchTST/TimesNet 的批大小降为 512（内存安全），
  预检产物在正式队列前清除；
- 更新服务器脚本、PowerShell 队列与 SSH 助手的路径；
- 删除全部旧遗物（约 6.5 GB）：`reports/99_archive`、`figs/99_archive`、
  `data/99_archive`、`src/s05_legacy`、`scripts/99_legacy`、今早的非协议深度结果、
  旧服务器载荷 tar 包；仓库不再保留归档目录。

## 4 当前结果状态（2026-09-12 服务器跑完并回收后）

| 产物 | 状态 |
|---|---|
| `reports/02_experiment/cv/month_balanced/{ghi,cloud}/selection.json` | **已回收**（服务器五折 CV 产出） |
| `reports/03_modeling/v2_mlp…v15_pinn/full/quantile/{ghi,cloud}/` | **已回收**：results.csv + calibrated/results_calibrated.csv（14 深度×2 目标） |
| `reports/03_modeling/v1_ml/full/quantile/{ghi,cloud}/` | **本地 fixed-kt 重跑**，逐样本含 station/time/lead/season（新 schema）；旧 broken-kt 备份 `full_brokenkt_20260912/`、无 id 版 `full_noid_20260913/` |
| 逐样本 `test_predictions.csv` | **可用**：v1 全子模型 + 头部深度 fixed-kt（tcn/transformer/informer/autoformer，ghi，`05_robustness/kt_sensitivity/fixedkt_top/`）→ 已支撑分段/条件覆盖/PIT/案例；其余 10 个深度模型逐样本仍缺（需重训或旧实例） |
| `reports/batch1_skill_ghi.csv` | **新增**：baseline + skill score（白天基线，optimal-convex 最佳≈0.28） |
| `figs/03_modeling/00_comparison/fig_model_comparison_full` | **新增**（真实 full 数据 Fig.5） |
| `figs/06_probability/00_comparison/fig_calibration_effect_full` | **新增**（真实 full 数据 Fig.9） |
| `figs/03_modeling/v1_ml/full/fig_v1_ml_{ghi,cloud}_agreement` | **新增**（现行 full v1 一致性图）；旧 lite 版（旧切分）已废弃至 .trash |
| `figs/00_preview/`、`figs/01_data_audit/`、`figs/04_error_analysis/` | 已按 QA 硬标准重出（图例轴外/双轴标签/45°锚定/实心） |
| `data/03_featured/` | **已用修复版 kt 重生成**（门槛50+clip，max=1.5） |
| `reports/06_probability/per_sample_diagnostics.csv` | **新增**：白天/分段/条件覆盖/PIT-KS（v1+头部深度） |
| `reports/05_robustness/dm_test_ghi.csv` | **新增**：DM(HAC+Harvey+BH)，16/21 显著，tree-vs-deep 对不显著 |
| `reports/06_probability/conformal_blockbootstrap.csv` | **新增**：block-bootstrap conformal 覆盖（lgbm 0.79→0.89） |
| `figs/06_probability/00_comparison/fig_reliability_full`、`figs/04_error_analysis/fig_grouped_full`、`figs/03_modeling/00_comparison/fig_case_studies_full` | **新增**：Fig6/7/10（逐样本真实数据） |
| `figs/05_robustness/fig_dm_conformal_full`、`figs/03_modeling/00_comparison/fig_model_comparison_dual` | **新增**：DM+conformal、双版 Fig5 |
| `site_data/*.json` | 导出：model_comparison / reliability（供网站） |
| `reports/04_error_analysis/shap_*.csv` + `figs/04_error_analysis/fig_shap_*` | **新增**：分位/时效/站点/交互 SHAP（LightGBM TreeExplainer） |
| `reports/04_error_analysis/{ablation_feature_groups,residual_diagnostics}.csv`、`reports/06_probability/pit_grouped.csv` + `fig_ablation_feature_groups`、`fig_pit_grouped` | **新增**：特征组消融 + 残差诊断 + 分组 PIT |
| `figs/02_site_design/fig_jiangsu_{20sites_geo,annual_ghi}` | **新增**：站点图 + 年辐射量，均用**真实 0.1° Himawari 栅格**（`fetch_grid_ghi.py` 抓取，非插值）+ **Albers 等积投影**（修正此前"插值面 + lon/lat 压扁"两个错误）。`fig_exploratory_error` panel f 已加轮廓。订正增量全省面因需模型上网格，暂缓 |

## 5 服务器执行顺序

**执行历史（简述）**：主 full 训练在旧实例 `westc:37257`（RTX 4080 SUPER）跑完并回收
（CV→v1→v2–v15→校准→出图，ALL DONE）；随后新实例 `westb:21410`（RTX 4090）用于
kt 敏感性与头部模型 fixed-kt 重跑。**两实例现均已关机**（换平台/再训练需控制台开机）。
下文 1–5 为通用服务器执行流程（保留备查）。

1. 克隆实例就绪后自检：`bash scripts/05_server/check_server.sh`
   （打印环境、代码版本、20 个 featured 文件的 md5 与缺失清单，本地清单见
   `deploy/01_payload/featured_manifest_md5.txt`）；
2. 同步代码：上传 `deploy/01_payload/meteo_code_v3_2026-09-11.tar.gz`（约 85 KB）并解压覆盖，
   解压后删除服务器上残留的旧目录（`src/models`、`src/analysis`、`src/clean` 等）；
3. 数据：训练只读 `data/03_featured/*_featured_2024-02_2026-09.parquet`（20 个文件，102 MB）。
   自检显示缺失或 md5 不一致时，上传 `deploy/01_payload/meteo_data_featured_v3_2026-09-11.tar.gz`
   （120 MB，含 featured 与 clean 的当前 tag）；raw JSON 不需要上传，清洗在本地完成；
4. 执行 `bash scripts/05_server/run_server_full.sh`：脚本先清除克隆实例上的旧 `v*_*` 结果 → 五折 CV →
   v1_ml → v2–v15（长序列模型用正式结构 + 168 小时输入）→ 校准 → 出图；
5. 回传 `reports/02_experiment`、`reports/03_modeling`、`figs`（不含 `.cache`）。

### 5.1 当前执行状态与服务器待办（2026-09-13）

- **已完成（本地）**：kt 修复 + featured 重生成；v1 本地 fixed-kt 重跑（含身份列）；
  头部深度 fixed-kt 重跑（westb）；Batch-1 基线+skill；逐样本诊断（分段/条件覆盖/PIT-KS）；
  DM(HAC+Harvey+BH)；block-bootstrap conformal；双版 Fig5；SHAP（分位/时效/站点/交互）；
  特征组消融 + 残差诊断 + 分组 PIT；**M1 物理重命名（目录+train_deep+下游脚本，文档磁盘统一）**。
  图/表/文档均已同步。
- **仅剩服务器待办**：
  1. **LOSO / 多种子稳健性**（C5）与其余 10 个深度模型的逐样本预测（全模型条件覆盖/案例）；
  2. **R15 PINN 物理约束验证**（越界/约束-精度权衡/λ 扫描）；
  3. **深度模型 SHAP**（DeepExplainer，需服务器权重）。
- **kt 敏感性结论**：fixed-kt 与 broken-kt 差异 ±1–2% 噪声内（树 ±0.1）→
  主结果沿用 broken-kt 全套，fixed-kt 头部重跑作稳健性章节；"树≈深度"边界对 kt bug 不敏感。

## 6 已知风险与待确认

1. 五折 CV 已在服务器执行、`selection.json`（ghi/cloud）已回收；v1 已按选中超参训练（不再是默认超参临时结果）。
2. 测试年 2025-09~2026-01 在早期开发中曾暴露，论文需披露；
   完全干净的验证窗口从 2026-09 起冻结。
3. GFS 有效预报仅 2024-02 起，无法执行多年交叉验证。
4. DM 检验已按规范完成（Newey-West HAC + Harvey 小样本修正 + BH-FDR，见 07 R12）：
   raw vs 各订正模型显著，**tree-vs-deep 对不显著→坐实"树≈深度"边界**；
   完整 LOSO（20 站）与多种子（5 个种子）仍待服务器执行（C5）。
5. 评审遗留项已按用户批准全部执行（早停间隔、输入长度统一、编码消融、分位单调惩罚、
   v1 种子策略），处置表见 `docs/06_protocol_changelog.md` §3；GFS 版本漂移按项目
   决定不再追踪。
6. 深度模型不做事先超参搜索（结构与学习率固定，五折 CV 仅覆盖 v1 树/线性族）；
   深度多种子由稳健性章节对排名靠前模型补充，正式主表用 seed 0。
7. AutoDL 实例已连通并使用过（主训练 westc:37257、敏感性与头部重跑 westb:21410），
   现均已关机；再训练需控制台开机。`scripts/05_server/remote.py` 端点已更新为
   `connect.westb.seetacloud.com:21410`；调用前须 `export MSYS_NO_PATHCONV=1`。

## 7 文档

| 文件 | 内容 |
|---|---|
| `docs/01_research.md` | 研究问题、数据、实验设计、模型族、指标、流程、文件结构、限制 |
| `docs/02_project_status.md` | 本文件：进度、产物、服务器执行、风险 |
| `docs/03_hyperparameters.md` | 调参协议、搜索空间、校准参数 |
| `docs/04_project_organization.md` | 目录规范、命名规则、图件索引、绘图风格、脚本清单 |
| `docs/05_review.md` | 文献综述与引用清单 |
| `docs/06_protocol_changelog.md` | 协议变更日志、AutoPV 缺陷清单、外部评审 25 条处置、文献待复核清单 |
| `docs/07_issue_register.md` | 全项目问题台账（A–E 评审 + 数据审计新发现），含优先级/修复阶段/是否重训/批次规划 |
