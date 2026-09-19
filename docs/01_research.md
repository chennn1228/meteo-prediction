# Research framework: GFS GHI probabilistic post-processing

Version 2.0.0-provisional | 2026-09-15

> `project_manifest.yaml` is the Single Source of Truth（单一事实源）. This
> document explains the scientific logic. Existing numerical results are
> provisional until regenerated under the v2 protocol.

## 1 One-sentence argument

**In a Jiangsu regional setting, we test whether the Open-Meteo
location-specific GFS product contains independent GHI forecast value and
quantify the incremental benefit of causal statistical, machine-learning,
deep-learning and calibrated probabilistic post-processing across lead times
and unseen locations, supported by nested rolling and spatial evaluation and
bounded by the availability and uncertainty of Himawari truth.**

中文核对：本研究不再预设复杂模型有效，而是先检验 GFS 自身价值，再依次量化统计订正、
树模型、深度模型和概率校准的增量；结论边界限于 Open-Meteo 地点化 GFS 产品、江苏区域和
通过质量门禁的 Himawari 真值。

## 2 Terminology Ledger（术语账本）

| Canonical term | 首次定义 | 废弃或受限变体 |
|---|---|---|
| GFS GHI probabilistic post-processing | GFS 辐照度概率后处理 | “GHI 与 cloud 双主产品” |
| rolling-origin validation（滚动起点验证） | 训练只使用验证时刻以前的数据 | month-balanced 5-fold 仅可复现旧结果 |
| incremental value（增量价值） | 相邻复杂度层级相对于前一层的收益 | 不用“模型提升”笼统代替 |
| spatial generalization（空间泛化） | 对未参与训练地点的泛化 | LOSO 不等同全省未见格点评价 |
| Open-Meteo location-specific GFS product（地点化 GFS 产品） | 当前 point API + `cell_selection=land` 返回产品 | 不称 raw GFS grid |
| raw GFS grid（原始 GFS 格点） | NOAA GFS GRIB 原生格点 | 当前仓库尚未直接获取 |
| mean pinball | 7 个分位 pinball 的未加权均值 | 不写成 CRPS |
| `crps_q7_trunc` | τ=0.05–0.95 分位网格的截尾积分近似 | 不写成完整 CRPS |
| development period（开发期） | 2024-02-01 至 2025-08-31 | 不用 train/test 污染叙事 |
| final test（最终测试） | 2025-09-01 至 2026-08-31 | 新协议冻结后只评一次 |
| experimental constrained model（实验性约束模型） | 当前 PINN 定位 | 不称可靠物理模型 |

## 3 Scientific questions and hypotheses

### RQ1 — GFS independent value

**GFS 在江苏 D+1/D+2/D+3 的原始 GHI 预报是否优于不使用 NWP 的基线？**

- H1a：raw GFS 的 RMSE 低于 climatology、persistence 和 smart persistence。
- H1b：GFS 的优势随 lead 增加而减弱，但不预先规定单调性。
- H1c：所有比较同时报告 MAE、RMSE、Bias 和相对于预先指定基线的 RMSE skill score。

### RQ2 — Extractable GFS information

**原始 GFS 未充分表达的信息能否被统计后处理稳定提取？**

- H2a：bias correction 主要修正系统偏差，不能替代条件化映射。
- H2b：linear/ridge MOS 相对 raw GFS 产生稳定增量。
- H2c：LightGBM/XGBoost 相对 MOS 的增量必须跨 outer folds、lead 和条件组稳定才成立。

### RQ3 — Complexity ladder

**从 tree ML 到 DL 的额外复杂度产生多少可重复的增量价值？**

- H3a：相同 inner folds、selection metric 和 6-trial 预算下比较 ML/DL。
- H3b：top models 用 seeds 0–4 报均值与不确定性。
- H3c：任何“tree≈deep”结论仅限区域、有限数据的 NWP 后处理场景。

### RQ4 — Feature mechanisms

**Cloud、clear-sky/kt、辐射、气象、几何、时序和空间特征的贡献是否稳定？**

- H4a：Cloud 只在 group ablation、grouped permutation、rolling stability 和区间估计共同支持时称为有效。
- H4b：kt/clear-sky 与 radiation 的冗余需显式量化。
- H4c：SHAP 只解释模型行为，不单独决定特征保留。

### RQ5 — Probability and calibration

**时间有序校准能否改善边际与条件覆盖而不过度增宽区间？**

- H5a：比较未校准、time-ordered split conformal 和 block-bootstrap uncertainty。
- H5b：仅在分组证据支持时采用 weather-conditioned Mondrian conformal（天气条件保形校准）。

### RQ6 — Spatial generalization and site sufficiency

**20 个代表站训练能否泛化到江苏省域未见地点，5/10/15/20 站何时趋于饱和？**

- H6a：省域模型不依赖固定 `station_id` one-hot。
- H6b：误差和 skill 必须按 distance to nearest training site 报告，不预设距离效应方向。
- H6c：全省正式评价只纳入通过完整时段小时级 Himawari truth 门禁的地点。

## 4 Scientific narrative and paper architecture

主证据链固定为：

1. **GFS 原始预报价值**：climatology / persistence / smart persistence / optimal convex 与 raw GFS；
2. **系统误差特征**：lead、季节、天气、太阳高度与空间条件；
3. **简单统计订正**：bias correction、linear MOS、ridge MOS；
4. **非线性机器学习增量**：LightGBM、XGBoost；
5. **深度学习增量**：14 个架构在等预算下的区域性 benchmark；
6. **概率预报**：七分位、mean pinball、coverage、width；
7. **概率校准**：时间有序 split、block uncertainty、候选 Mondrian；
8. **条件失效与空间泛化**：分组诊断、三层空间评价、训练站密度。

建议论文/技术文档目录：

1. Introduction（引言）
2. Problem Statement（问题定义）
3. Data and Temporal Semantics（数据与时间语义）
4. Methodology（方法）
   1. GFS Value Ladder
   2. Feature-mechanism protocol
   3. Statistical, tree and deep post-processing
   4. Probabilistic calibration
5. Evaluation Design（评价设计）
   1. Nested purged rolling-origin validation
   2. Three-level spatial evaluation
   3. Training-site density experiment
6. Results and Analysis（结果与分析）
7. Limitations and Threats to Validity（局限性与效度威胁）
8. Conclusions（结论）

每个 Results 段只承担一个任务：claim → evidence → comparator → boundary。

## 5 Data and temporal semantics

### 5.1 Current forecast object

当前代码请求 Open-Meteo Previous Runs API 的 `gfs_seamless`，并使用
`cell_selection=land`。因此研究对象是 **Open-Meteo location-specific GFS
product**，不是直接下载的 raw GFS GRIB grid。API 返回的经纬度和高程保存在新清洗契约的
`source_grid_latitude`、`source_grid_longitude`、`source_grid_elevation`；
请求坐标保存在 `lat`、`lon`。

如需声称 raw GFS grid 结论，必须新增 GRIB/archive 数据链并独立审计。

### 5.2 Lead and time alignment

- `_previous_day1/2/3` 分别是 valid time 前 24/48/72 小时做出的预报；
- `fcst_issue_time_utc = target_time_utc - lead_time`；
- GFS surface product 在 120 小时内为 hourly；D+1–D+3 不涉及 3-hourly→hourly 插值；
- GHI/DHI/DNI/GTI 与 Himawari `shortwave_radiation` 均采用 preceding-hour mean；
- 云量、温湿风等 instant/interval 语义按 API 字段定义，不混写成辐射平均量。

本地样本 2024-02 含 696 个连续小时，步长严格为 1 h。完整证据见
`reports/01_data_audit/gfs_semantics/audit.md`。

### 5.3 Forecast variables

旧 raw 有 15 个预报变量。v2 配置增加 low/mid/high cloud，共 18 个；现有 raw/featured
缺少新增 9 个 lead-specific 字段，正式重训前必须重新抓取、清洗和生成特征。

### 5.4 Truth

- 主真值：Himawari 小时 GHI；
- ERA5 辐射：交叉检查，不替代主真值；
- ERA5 cloud：补充性参考，存在与 GFS 的同源/依赖风险；
- 正式全省 test 必须先检查完整 development/test 的小时覆盖、missingness、空间匹配和成本。

## 6 Cloud repositioning

Cloud 不再是与 GHI 平级的主产品。

1. **物理输入**：GFS total/low/mid/high cloud 是候选特征；
2. **误差机制**：解释 clear / partly cloudy / overcast、cloud change 和 GHI error；
3. **辅助实验**：已有 ERA5-cloud correction 标为 supplementary exploratory experiment。

只有 corrected cloud 对最终 GHI 在因果验证中产生稳定、带区间的增量，或出现独立业务需求，
才恢复为主要任务。可检验 raw vs corrected cloud、GHI-only vs cloud auxiliary、single-task
vs multi-task 和 corrected-cloud cascade。

## 7 Feature-mechanism protocol

Feature groups 及字段由 `project_manifest.yaml` 定义。选择步骤只能在当前 outer-training
prefix 内的 inner folds 发生：

1. 预注册 group ablation；
2. grouped permutation importance；
3. rolling-fold rank/effect stability；
4. block bootstrap 或跨折置信区间；
5. 再用 SHAP 解释入选模型，不反向选择特征。

预先要求的对照：Base+cloud、Base+kt、Full−cloud、Full−kt、radiation/clear-sky/kt
冗余、time encoding 是否为噪声。当前旧协议消融提示 meteorology 最大、cloud 约 1%、
radiation 与 kt 高冗余、time encoding 可能有害；这些只是 provisional hypothesis-generating
evidence，不是 v2 入选结论。

## 8 GFS Value Ladder

| Tier | Models | Primary comparison |
|---|---|---|
| 0 Non-NWP | climatology, persistence, smart persistence, optimal convex | 原始 GFS 是否值得使用 |
| 1 Raw NWP | raw GFS | 独立 NWP value |
| 2 Statistical | bias correction, linear MOS, ridge MOS | 简单订正增量 |
| 3 Tree ML | LightGBM, XGBoost | 非线性增量 |
| 4 Deep | 14 deep architectures | 等预算下的架构增量 |
| 5 Calibrated probability | split / block / conditional conformal | 覆盖—宽度权衡 |

每一层按 D+1/D+2/D+3 报 MAE、RMSE、Bias 和 RMSE skill；概率模型另报 mean pinball、
coverage 与 width。不得只展示冠军模型。

## 9 Temporal validation

### 9.1 Outer folds

| Fold | Training prefix | Purge | Validation |
|---|---|---|---|
| outer_1 | 2024-02-01–2024-05-31 | 2024-06-01–06-10 | 2024-06-11–07-31 |
| outer_2 | 2024-02-01–2024-08-31 | 2024-09-01–09-10 | 2024-09-11–10-31 |
| outer_3 | 2024-02-01–2024-11-30 | 2024-12-01–12-10 | 2024-12-11–2025-01-31 |
| outer_4 | 2024-02-01–2025-02-28 | 2025-03-01–03-10 | 2025-03-11–04-30 |
| outer_5 | 2024-02-01–2025-05-31 | 2025-06-01–06-10 | 2025-06-11–07-31 |

Purge = 168 h maximum sequence lookback + 72 h maximum lead = 240 h = 10 days。
另做 7/10/14 天 gap sensitivity；如样本重叠或 ACF 证据要求更长，以证据更新 manifest。

### 9.2 Inner folds and final fit

每个 outer-training prefix 内建立 3 个 expanding inner folds，用于全部调参、feature-group
selection、early-stopping policy 和 calibration strategy。最终顺序：

`fit ≤ 2025-06-20 → purge → early stop 2025-07 → purge → calibration 2025-08-11–08-31 → test 2025-09–2026-08`

最终 test 不参与 architecture、feature、station count、calibration fitting 或阈值选择。

## 10 Equal-budget ML/DL tuning

| Item | Formal rule |
|---|---|
| Candidate budget | 每个模型 6 trials |
| Inner folds | 同一 3 个 rolling folds |
| Selection metric | mean pinball |
| Tuning seed | seed 0 |
| Learning rate | 0.0005 / 0.001（DL；tree 有 6 个预注册配置） |
| Scale | small / base |
| Dropout | 0.1 / 0.2，受总 trial 限制 |
| Repeats | top models seeds 0–4 |
| Reporting | trial 数、wall time、device、peak memory、停止 epoch |

## 11 Calibration protocol and formula audit

对分位 τ，唯一加性公式为：

`residual_i,τ = y_i − q_hat_i,τ`

`delta_τ = empirical_quantile(residual_τ, τ)`

`q_cal,τ = q_hat_τ + delta_τ`

因此若模型整体低估，residual 为正，校准后分位应上移。当前实现与该符号一致；相反的
`q_hat − y` + 加法公式错误。`tests/01_unit/test_calibration.py` 用最小例验证方向并验证
校准后单调性。

校准顺序必须是 fit → early stop → later calibration → validation/test。现有 calibrated
结果源自退役 split，全部需重算。天气分组 PIT 提示 conditional miscalibration，但只有在
outer-fold 证据支持时才采用 Mondrian。

## 12 Spatial design

### Level 1 — Stratified Spatial Holdout（分层空间留出）

20 站 development data 内按五区域留出，任何训练折保留每区代表；用于选择可迁移空间特征。

### Level 2 — Province-wide Unseen-grid Evaluation（全省未见格点评价）

冻结方案后以 20 站训练，评价其余通过 Himawari truth gate 的地点，按 lead、五区域、季节、
天气和 distance to nearest training site 报 raw/corrected error、skill、improvement、coverage。

### Level 3 — Regional Extrapolation Stress Test（区域外推压力测试）

train 4 regions / test fifth region，仅作 robustness，不代表主部署场景。

### Training-site Density Experiment（训练站密度实验）

比较 5/10/15/20 站，保持五区域尽量均衡并使用预注册 max–min 选择；站数只能在 development
spatial validation 内确定。当前 20 个请求站映射为 20 个不同的 API 返回位置，最大请求偏移
约 7.79 km；“仅 4–8 个独立格点”的旧说法无证据且已撤销。

当前 `grid_annual_ghi_2025.csv` 有 1000 个 0.1° Himawari 展示点（354 日有效），它不是 raw
GFS grid，也没有完成全时段小时级 truth gate，因此不能充当正式 Level-2 结果。

## 13 PINN scientific audit

代码审计发现：旧 PINN 将目标标准化空间的 prediction 与特征标准化空间的
`ghi_clear_sky` 直接比较，量纲不一致；同时 Ineichen 在江苏并非可信硬上限，观测存在
cloud enhancement。v2 已禁用 clear-sky hard ceiling，只保留 nonnegative soft penalty。

文档曾声称 kt consistency，但实际 loss 只有 nonnegative + clear-sky ceiling；因此
`kt_consistency_implemented=false`。在 site-adapted clear sky、soft tolerance、enhancement 和
uncertainty 评估完成前，PINN 只称 experimental constrained model。

## 14 Existing evidence: provisional only

旧协议结果可用于提出假设和网站 Demo，不进入 v2 主表：

- 总体白天 RMSE：climatology 193.94、raw GFS 163.07、linear MOS 137.54、XGBoost 137.77、
  TCN 138.05、LSTM 138.26 W m⁻²；
- mean pinball：raw GFS 35.35、XGBoost 25.62、LightGBM 25.77、LSTM 25.56；
- 旧分组诊断显示天气条件覆盖差异和厚尾残差；
- fixed-kt 子集结果只覆盖部分模型，不能替代完整重训。

这些数字只能写为“existing provisional analysis indicated”，不能写为 v2 已证明的结果。

## 15 Retrain/recompute decisions

### Must retrain or recompute

1. 重新抓取 total/low/mid/high cloud 后生成 v2 featured；
2. nested outer/inner 的 baseline、MOS、tree 和 14 DL equal-budget benchmark；
3. 全部 OOF predictions、5-seed top models 与不确定性；
4. time-ordered calibration 和候选 Mondrian；
5. Level-1 spatial holdout、Level-2 truth-gated province evaluation、5/10/15/20 density；
6. PINN（仅在新约束完成后）；
7. 所有旧 calibrated 与 official 表/图。

### Reusable as diagnostics or sensitivity

- raw data and Himawari/ERA5 alignment records, after semantic audit；
- 20-site design and Jiangsu boundary；
- baseline implementation、metric functions、plotting conventions；
- old feature ablation、SHAP、residual/PIT、DM and fixed-kt results as provisional diagnostics；
- existing SVG figures as information-design references。

### Forbidden from official results

- month-balanced selection；
- broken-kt main tables as new-protocol evidence；
- noncausal calibrated results；
- smoke/lite/debug outputs；
- cloud as co-equal main RQ；
- raw-grid claims inferred from nominal resolution；
- old “test exposure/clean segment” narrative。

## 16 Limitations and threats to validity

- 当前产品是 Open-Meteo 地点化产品，原始 GFS 格点结论尚未建立；
- 新增 cloud-level fields 尚未回填；
- 全省小时 Himawari truth 的完整性、成本和空间匹配尚未通过门禁；
- Himawari retrieval uncertainty 构成不可约误差边界；
- ERA5 cloud 不是独立观测；
- clear-sky reference 在 cloud enhancement 条件下不能当硬上限；
- 14 个 DL 是受限预算架构 benchmark，不支持普遍性 ML-vs-DL 排名；
- 当前仓库不是 Git 工作树，变更追踪依赖归档与 consistency report，建议尽快纳入版本控制。

## 17 Claim–evidence map

| Claim | Evidence | Status |
|---|---|---|
| D+1–D+3 GFS surface data are hourly | official API documentation + local 696-step sample | supported |
| 20 sites share only 4–8 cells | contradicted by 20 distinct returned coordinates | rejected |
| raw GFS has value over no-NWP baselines | provisional final-year comparison | needs v2 reporting by lead |
| tree and deep performance are similar | old protocol aggregate/DM analysis | provisional; needs equal-budget v2 |
| cloud has a stable feature contribution | old ablation only | needs evidence |
| weather-conditional calibration is needed | old grouped PIT/coverage | inferred; needs outer-fold validation |
| 20 sites generalize province-wide | no truth-gated full-grid evaluation | needs evidence |

