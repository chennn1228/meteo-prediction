# Issue Register 问题台账

版本 v1.1 ｜ 2026-09-13 ｜ 定位：全项目问题/缺陷的统一追踪表。
来源：① 首轮技术评审（A 数据/B 特征/C 实验/D 模型/E 评估 五类）；② 数据审计脚本
`src/s04_evaluation/analysis/audit_data_quality.py`（2026-09-12）新发现。
处置原则：**服务器本轮跑完前不动训练代码**；能在校准/评估/报告阶段修的不重训；
需重训的集中在跑完后一次性处理。状态字段：`待办 / 进行中 / 已修 / 已豁免 / 待决策`。

---

## 0 阅读指南

- **Severity**：P0 = 投稿前必须修；P1 = 顶刊评审必挑；P2 = 锦上添花。
- **Fix Stage**：`eval` = 评估端改（不重训）；`calib` = 校准步骤改（不重训）；
  `feat` = 改特征/数据（需重生成 featured + 重训受影响模型）；`doc` = 写作补一段；
  `code` = 改训练/评估代码。
- **Retrain?**：`否` = 当前服务器结果可直接用；`深度` = 仅 v2–v15 需重训；`全` = 全部重训。
- 编号沿用首轮评审的 A/B/C/D/E 前缀；审计新发现用 `N` 前缀。

---

## 1 数据与真值（A 类 + 审计新发现）

| ID | Sev | 问题 | 证据 | Fix Stage | Retrain | 状态 |
|---|---|---|---|---|---|---|
| A1 | P1 | Himawari 真值误差未量化，缺 irreducible floor | 文献 rMBE −7.4%~4.9% | doc | 否 | 待办 |
| A2 | P0 | ERA5 与 GFS 云量同源风险只声明未量化 | 审计：r=0.605, bias=+4.7%, RMSE=39.9%（已出图 fig_cloud_identifiability） | eval | 否 | **部分已修**（图已出，正文待引） |
| A3 | P1 | 缺卫星云量第三真值交叉验证 | config 未抓 Himawari cloud | doc/feat | 否 | 待办(future work) |
| A4 | P1 | 20 站有效样本量被高估（GFS 网格自相关） | 0.25°≈25km，独立格点 4–8 | eval | 否 | 待办 |
| A5 | P0 | 夜间样本处理未定义（GHI 产品） | is_day_fcst 存在但未用于过滤 | eval | 否 | 待办 |
| A6 | P1 | 1h 为 3h 插值产物，序列模型学伪信号 | config resolution:hourly | doc | 否 | 待办 |
| N1 | P0 | **kt/kni/diffuse_fraction 除法爆炸（极端层）**：分母 ghi_clear_sky∈(0,1] | 审计：kt_fcst max=2.55e6（9.1% 坏）；kni max=1.9e5；门槛≥10 可消除极端层 | feat | 深度 | **特征层已修(09-12)；敏感性已验证(09-13)：无实质影响** |
| N2 | P0 | **kt 残留层（门槛修不掉）**：分母中位 46 W/m² 仍 kt∈(1.5,13] | 审计：门槛>100 仍 3.37% 坏、max=3 → 必须 clip kt∈[0,1.5] | feat | 深度 | **特征层已修(09-12)；敏感性已验证(09-13)** |
| N3 | P1 | **Ineichen 晴空模型在江苏系统性偏低**：10–15% 白天 GHI>1.25×clearsky | 审计：obs 15.2% / fcst 13.3% / era5 10.3% 超 1.25× | feat(大改) | 全 | 待决策（site-adapted clear-sky 留 future work） |
| N4 | P2 | GHI_obs_sat>1300 离群（仅 9 行，ratio 1.49×） | 审计 §3 | eval | 否 | 待办（评估端报 with/without） |
| N5 | P2 | sky_type 与 is_day_fcst 不一致（差 3,915 行，obs 缺失回退 night） | 审计 §5 | eval | 否 | 待办 |

## 2 特征工程（B 类）

| ID | Sev | 问题 | Fix Stage | Retrain | 状态 |
|---|---|---|---|---|---|
| B1 | ~~P0~~ 豁免 | hour_local 时区：实为**北京时**（tz_convert Asia/Shanghai），此前"UTC/相位差120°"为误判 | 无需修改 | 否 | **已豁免(09-12 更正)** |
| B2 | P1 | ghi_fcst_lag1/2 语义不明（索引结构未声明） | doc | 否 | 待办 |
| B3 | P1 | kt_weather_bins 注释[0.3,0.5,0.7]≠实际[0.44,0.73,0.97] | doc/eval | 否 | 待决策 |
| B4 | P2 | cloud_cover_roll 窗口对齐未声明 | doc | 否 | 待办 |
| B5 | P2 | PCA95 fit 时机疑似预处理泄漏 | code/doc | 否 | 待办（审计代码） |

## 3 实验设计（C 类）

| ID | Sev | 问题 | Fix Stage | Retrain | 状态 |
|---|---|---|---|---|---|
| C1 | P0 | Conformal 违背可交换性（折0块内自相关） | calib | 否 | **已做(09-13)**：滚动式 block-bootstrap conformal 已实现并复评（split≈blockbb 点覆盖，价值在降方差）；阴天仍欠覆盖→引出 Mondrian |
| C2 | P0 | 测试年历史暴露处理不严谨 | eval | 否 | **已做(09-13)**：暴露段/干净段分段报告；raw→xgb 相对增益两段相等→暴露未虚增 |
| C3 | P0 | 缺 persistence/smart-persistence/climatology baseline | code/eval | 否 | **已做(09-13)**：4 基线 + optimal-convex skill 已算（白天基线） |
| C4 | P1 | DM 检验缺小样本修正+多重比较校正 | eval | 否 | **已做(09-13)**：DM+NeweyWest(HAC)+Harvey+BH-FDR；tree-vs-deep 对不显著→坐实边界 |
| C5 | P1 | LOSO 空间自相关未处理 | eval | 否 | 待办（Regional LOSO） |
| C6 | P1 | 模型选择 winner's curse 未量化 | eval | 否 | 待办（CV-test gap） |
| C7 | P1 | 深度模型不调参的 framing 需声明 | doc | 否 | 待办 |
| C8 | P1 | 分位交叉 33% 说明独立分位范式有问题 | code | 深度 | 待办（NCQRNN/QRF） |

## 4 模型与损失（D 类）

| ID | Sev | 问题 | Fix Stage | Retrain | 状态 |
|---|---|---|---|---|---|
| D1 | P1 | PINN 物理约束太弱（缺几何/时间连续性） | code | 深度 | 待决策 |
| D2 | P1 | 无 ensemble/stacking | code/eval | 否 | 待办 |
| D3 | P2 | crps_q7_trunc 截尾对极端事件失真 | doc | 否 | 待办 |
| D4 | P2 | 无三层不确定性分解 | doc | 否 | 待办(future work) |

## 5 评估与报告（E 类）

| ID | Sev | 问题 | Fix Stage | Retrain | 状态 |
|---|---|---|---|---|---|
| E1 | P0 | 缺 RMSE skill score（Yang 2020 基准） | eval | 否 | 待办（依赖 C3） |
| E2 | P1 | 条件覆盖率未报告 | eval | 否 | 待办 |
| E3 | P1 | PIT 缺 KS/AD 统计检验 | eval | 否 | 待办 |
| E4 | P2 | 无经济价值评估 | doc | 否 | 待办(future work) |
| E5 | P2 | 可复现性细节缺失（lock/deterministic/容器） | doc/code | 否 | 待办 |

## 6 文档与编号（本轮新增）

| ID | Sev | 问题 | 处置 | 状态 |
|---|---|---|---|---|
| M1 | P2 | 模型 v 编号与族顺序不一致 | 已改文档编号（cnn4→3, tcn5→4, lstm3→5）；物理重命名暂缓 | **文档已修**，目录待跑完后重命名 |
| M2 | P2 | 01_research 缺 Introduction/RQ/Hypotheses | 已重构 v11.0 | **已修** |

---

## 7 处置批次规划（服务器跑完后）

| 批次 | 内容 | 是否重训 | 预估工作量 |
|---|---|---|---|
| Batch-1（评估端，不重训） | A2引图, A4, A5, C1, C2, C3, C4, C5, C6, D2, E1, E2, E3, N4, N5 | 否 | ~2 天 |
| Batch-2（特征修复+敏感性） | N1, N2 → 重生成 featured + 头部深度 fixed-kt 重跑 + v1 本地重跑 | 部分 | **已完成(09-13)** |
| Batch-3（视 Batch-2 结果） | 结论：差异 ±1–2% 噪声内 → **豁免深度全量重训** | 豁免 | **已完成(09-13)** |
| Batch-4（写作） | A1, A3, A6, B2, B3, B4, B5, C7, C8, D1, D3, D4, E4, E5 | 否 | ~2 天 |
| 豁免/待决策 | N3（site-adapted clear-sky）, B1, D1, D4, E4 → future work/豁免 | — | — |

## 8 结果回收后更新日志（2026-09-12~13）

| 项 | 内容 | 状态 |
|---|---|---|
| R1 | 服务器 full 跑完并回传：14 深度模型×{ghi,cloud} full+calibrated + 两目标 selection.json + figs | **完成** |
| R2 | tar **不含逐样本 test_predictions.csv**（仅聚合）→ 夜间过滤/分段测试/条件覆盖/PIT/案例图被阻塞 | **待 A**（重连后重打包补下载） |
| R3 | Batch-1 聚合级完成：baseline(persist/smart/clim/optimal-convex) + skill score；发现模型预测为**白天-only**，skill 须用白天基线；climatology(白天≈194)强、smart(≈247)弱 → 报 skill 用 optimal-convex（最佳≈0.28） | **完成**（reports/batch1_skill_ghi.csv） |
| R4 | kt 修复落地：features.py 门槛50+clip，重生成 featured，max=1.5 越界0；train_deep 用 nan_to_num 故 NaN 安全 | **完成**（待 B 敏感性验证） |
| R5 | 图件 QA 重出：图例轴外、双轴标签、45°锚定、实心填充；正式 Fig5/Fig9 + 审计4图 + 预览5图 + v1 一致性图 | **完成** |
| R6 | 旧协议 lite 一致性图（figs/03_modeling/v1_ml/lite，旧切分）废弃 → .trash；用现行 full v1 重出至 v1_ml/full | **完成** |
| M1 | **物理重命名（旧→新编号）**：reports/figs 目录、`train_deep.py` `MODEL_VNUM`、下游脚本读盘列表 | **已完成(09-13 本地)**：目录与代码已改新编号、文档磁盘统一；服务器端下次会话自然对齐 |
| M3 | B 敏感性仅 lstm（服务器旧编号 v3_lstm）：修复版 featured 上传 + 单模型重训 + 与 broken-kt(25.565)对比 | **已完成(09-13)**：fixed-kt lstm pinball=26.92（+5%，单种子噪声内，无提升）→ **豁免深度全量重训**；存档 `reports/05_robustness/kt_sensitivity/v5_lstm_fixedkt/` |
| R7 | kt 敏感性含义：修复 kt 未提升 lstm → 当前 broken-kt 深度结果可作主结果；并证明"树≈深度"边界结论对 kt bug **不敏感**，可据此回应审稿人 | **完成** |
| R8 | 头部模型 fixed-kt 重跑（新实例 09-13）：ghi pinball fixed/broken = tcn 25.87/25.88、transformer 25.94/26.30、informer 25.93/26.25、autoformer 26.36/26.36；cloud 普遍略优；差异 ±1–2% 噪声内 → **维持豁免全量重训**。存档 `reports/05_robustness/kt_sensitivity/fixedkt_top/`（含 ghi 逐样本）。建议主结果沿用 broken-kt 全套保持一致性，fixed-kt 头部重跑作**稳健性章节** | **完成** |
| R9 | v1_ml 本地 fixed-kt 重跑：树对 kt 修复几乎无感（lgbm 139.0/138.9、xgb 137.8/137.9）；旧 broken-kt v1 备份于 `v1_ml/full_brokenkt_20260912/` | **完成** |
| R10 | `train_quantile_v1.py` 补 station_id/target_time/lead_time/season 到逐样本预测 → v1 可分组/分段/案例（此前缺身份列，非下载问题）；旧无 id 版备份 `v1_ml/full_noid_20260913/` | **完成** |
| R11 | 下游逐样本评估完成（`downstream_full.py`）：Fig6 可靠性/PIT/尖锐度、Fig7 分组MAE+条件覆盖、Fig10 案例、`reports/06_probability/per_sample_diagnostics.csv`、site_data。**关键发现**：① 暴露段 vs 干净段 pinball 差异（~21 vs ~28.5）是**季节难度**所致——raw→xgb 相对增益两段几乎相等（−27% vs −28%）→ **历史暴露未虚增结果**（可回应审稿人）；② 阴天 cov90 普遍欠覆盖（lgbm 0.773/xgb 0.801）→ 支持 H3；③ PIT KS p=0（校准前欠拟合）→ 佐证需 conformal | **完成** |
| R12 | `downstream2_local.py`：DM 检验（16/21 BH 显著；**tree-vs-deep 对不显著→坐实"树≈深度"边界**）、block-bootstrap conformal（lgbm cov 0.79→0.89，split≈blockbb，阴天仍欠覆盖→Mondrian 依据）、双版 Fig.5（主 broken-kt + fixed-kt 菱形叠加）。产物 `reports/05_robustness/dm_test_ghi.csv`、`reports/06_probability/conformal_blockbootstrap.csv`、`figs/05_robustness/fig_dm_conformal_full`、`figs/03_modeling/00_comparison/fig_model_comparison_dual` | **完成** |
| R13 | SHAP 分析（`shap_analysis.py`，LightGBM TreeExplainer，GHI）：① **分位级**——ghi_fcst 主导中分位、terrestrial_fcst 主导高分位尾部；τ 间重要性 Spearman 相邻≈0.98、尾-心仅 0.72–0.77 → 驱动随分位分化，**支持分位专属 head/分支**；② **跨时效/站点**——D+1/2/3 与 5 区域重要性排序见 `fig_shap_lead_region`；③ **交互值**——max 主效应 198 vs max 交互 11.3（交互≈主效应 6%），且 top 交互几乎全是 ghi_fcst×辐射族（冗余量组合），**非** cloud×elevation 强耦合 → **主效应主导，深度交叉层增益有限**（呼应树≈深度）。产物 `reports/04_error_analysis/shap_*.csv`、`figs/04_error_analysis/fig_shap_{quantile,lead_region,interaction}` | **完成** |
| R14 | `diagnostics_local.py`：① **特征组消融**——met 组贡献最大(Δpinball +3.5%)；radiation 与 kt_index 高度冗余(去掉≈0)；**time_enc 去掉反而 −2.1%（净噪声，候选删除）**；cloud 仅 +1.1%（→ GFS 辐射已编码大部分云信息，"两步无增益"的深证）；② **残差诊断**——偏度~0.5、**超额峰度~3.7（厚尾）**→ 极端事件未捕获，佐证分布感知头；|res|–|pred| 相关~0.18（异方差）；**阴天 mean_res +107、晴天 −42**（系统高估阴天/低估晴天）；云量剧变分位残差最大→突变建模不足；③ **分组 PIT**——按 lead 均衡（非 U 形，全局 conformal 够用），但**按天气强烈分化**（晴 PIT>0.9 质量 0.24、阴 PIT<0.1 质量 0.23）→ **需天气条件化(Mondrian) conformal**。产物 `reports/04_error_analysis/ablation_feature_groups.csv`、`residual_diagnostics.csv`、`reports/06_probability/pit_grouped.csv`、`figs/04_error_analysis/fig_ablation_feature_groups`、`figs/06_probability/fig_pit_grouped` | **完成** |
| R15 | PINN 物理约束验证（越界检查/约束-精度权衡/λ=0.1 最优性）：需 v15_pinn 逐样本预测（未回传）+ 多 λ 重训（GPU）→ **服务器待办**，本地不做 | **待服务器** |

## 9 已豁免项（用户决定，不再追）

| ID | 内容 | 理由 |
|---|---|---|
| X1 | GFS 版本漂移不追踪 | 2024-02 前数据为空，用户决定忽略（06_changelog） |
| X2 | 稳健性删除双真值项 | 项目决定（06_changelog） |
