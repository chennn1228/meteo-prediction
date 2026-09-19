# Research Protocol 研究协议

版本 11.1 ｜ 2026-09-13 ｜ 定位：项目技术文档（方法与流程），非论文正文。
旧版归档于 `docs/archive/01_research_v10.0_2026-09-11.md`。

---

## 1 Introduction 引言

### 1.1 Motivation 研究动机

- 光伏/电力市场对 D+1/D+2/D+3 辐照度与云量**概率预报**有刚需（备用容量、偏差考核、现货报价）；
- GFS 在江苏区域存在系统性偏差（文献：云量低估 → 辐照高估，Hațegan 2023；Mabasa 2025）；
- 江苏区域缺乏公开的"NWP 辐照度 + 云场联合概率订正数据集"（Zhang 2020 覆盖全国 17 站但无江苏专项；张敏 2024 仅做 CMA-WSP2.0 确定性检验）。

### 1.2 Related Work 文献定位

| 谱系 | 代表文献 | 本研究差异 |
|---|---|---|
| 线性 MOS / CSI 中间量 | Verzijlbergh 2015; Pierro 2015 | 本研究扩展到非线性 + 概率输出 |
| 概率订正（QR/QRF/GBDT） | Bakker 2019; Verbois 2018 | 本研究增加 conformal 校准 + 15 种深度模型对照 |
| 深度学习订正 | Savchenko 2025 (SolarM2P); Bire 2026 (BRTCN); Phan 2025 (TSMixer) | 本研究为多站合并 + 站点 one-hot，非 map-to-point |
| 链条位置质疑 | Mayer & Yang 2024/2025 | 本研究交付"气象端订正数据集"，不构建功率链 |
| 系统级概率预报 | Terrén-Serrano 2026 (Nat. Commun.) | 本研究为站点级分位数，非节点联合分布 |
| 情境依赖误差归因 | Lipponen 2026 (EGUsphere) | 本研究含 SHAP/分组误差分析，但未做 XGBoost 误差预测 |

### 1.3 Gap 空白点

1. 江苏 20 站分层、D+1/D+2/D+3、GHI + 总云量**双产品独立**概率订正：无先例；
2. 小样本（训练池 19 个月）下 15 种深度模型 vs 树模型的系统对照：无先例；
3. 时序 conformal 校准在 NWP 辐照度订正中的应用：文献中仅 NCQRNN（Xia 2024）涉及，未用 block bootstrap；
4. 云量产品以 ERA5 为参考真值的可辨识性量化：文献中 Deo 2023 用 KRR 订正 GFS 云量但未讨论同源风险。

### 1.4 Contributions 预期贡献

| # | 贡献 | 对应 RQ |
|---|---|---|
| C1 | 江苏 20 站 GHI + 云量条件分位数订正数据集（7 分位 × 3 时效 × 2 产品） | RQ1 |
| C2 | 21 种模型（6 传统 + 15 深度）在小样本 NWP 订正场景的系统对照 | RQ2 |
| C3 | 时序 conformal 校准 + block bootstrap 在辐照度概率订正中的适配方案 | RQ3 |
| C4 | 云量订正产品的可辨识性评估（GFS-vs-ERA5 同源风险量化） | RQ4 |

---

## 2 Problem Statement 问题定义

### 2.1 Task Definition 任务定义

- 输入：GFS Previous Runs 预报（15 变量 × 3 时效）+ 站点静态特征 + 时间特征；
- 输出：条件分位数 q̂_τ(s, T, l)，τ ∈ T = {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95}；
- 时效：l ∈ {24, 48, 72} h（D+1, D+2, D+3）；
- 两个产品独立建模，互不输入。

| 产品 | 目标变量 | 真值来源 | 真值属性 | 样本约束 |
|---|---|---|---|---|
| A 辐照度 | GHI (W m⁻²) | Himawari 卫星反演 | 观测（含反演误差 ~20–40 W/m²） | 仅白天（solar_elevation > 0） |
| B 云量 | 总云量 (%) | ERA5 再分析 | 非独立观测（与 NWP 同源风险） | 全时次 |

### 2.2 Research Questions 研究问题

| RQ | 问题 | 验证指标 |
|---|---|---|
| RQ1 | GFS 辐照度/云量预报的系统性偏差能否通过条件分位数订正显著降低？ | RMSE skill score > 0（相对 optimal baseline） |
| RQ2 | 不同模型族（线性/树/深度）的相对优势如何随时效（D+1→D+3）变化？ | 分时效 mean_pinball 排名 |
| RQ3 | Conformal 校准能否使边际覆盖率接近名义水平？条件覆盖率是否均匀？ | 覆盖率偏差 < 3pp；分组覆盖率极差 < 10pp |
| RQ4 | 以 ERA5 为参考真值的云量订正是否具有可辨识性（GFS 与 ERA5 偏差是否足够大）？ | GFS-vs-ERA5 云量 r < 0.9 且 |bias| > 5% |

### 2.3 Hypotheses 假设

| H | 假设 | 对应 RQ | 推翻条件 |
|---|---|---|---|
| H1 | 树模型（LightGBM/XGBoost）在 D+1 优于所有深度模型 | RQ2 | 任一深度模型 D+1 pinball 显著低于 LGBM（DM p<0.05） |
| H2 | 深度序列模型在 D+3 相对优势增大（长上下文收益） | RQ2 | D+3 排名与 D+1 无显著变化 |
| H3 | Conformal 校准后边际覆盖率接近名义水平，但阴天条件覆盖率欠覆盖 | RQ3 | 阴天覆盖率 ≥ 名义 − 3pp |
| H4 | GFS 云量与 ERA5 云量存在显著系统偏差（r < 0.9），云量订正有空间 | RQ4 | r > 0.95 且 |bias| < 3% |

### 2.4 Deliverables 交付物

- 分位数预测 CSV（7τ × 3 时效 × 20 站 × 测试年）；
- Conformal 校准参数（δ_τ per model per lead）；
- 评价指标表（点指标 + 概率指标 + 分组指标）；
- 模型脚本与复现图件；
- 可复用的订正气象数据集（Parquet）。

---

## 3 Data 数据

### 3.1 Study Area & Site Design 研究区域与站点设计

- 区域：江苏省（30.8°–35.1°N, 116.3°–121.9°E）；
- 站点：20 个，分 5 层（苏南内陆、苏中内陆、苏中沿海、苏北内陆、苏北沿海）；
- 布点策略：层内最大最小间距；
- 配置：`config/01_sites.yaml`；
- 图件：Fig. 1(a) `figs/02_site_design/fig_jiangsu_20sites`。

| 分层 | 站点 | 数量 |
|---|---|---|
| 苏南内陆 | nanjing_1/2, suzhou_1, wuxi_1, changzhou_1 | 5 |
| 苏中内陆 | taizhou_1/2, nantong_1/2 | 4 |
| 苏中沿海 | yancheng_1/2/3 | 3 |
| 苏北内陆 | xuzhou_1/2, huai_an_1/2/3, suqian_1 | 6 |
| 苏北沿海 | lianyungang_1/2 | 2 |

### 3.2 Data Sources 数据源

| 来源 | API / 产品 | 变量数 | 时间分辨率 | 有效窗口 | 角色 |
|---|---|---|---|---|---|
| GFS Previous Runs | Open-Meteo `gfs_seamless` | 16 | 1h（3h 原生插值） | 2024-02 起 | 预报输入 |
| Himawari (JMA/JAXA) | Open-Meteo `jma_jaxa_himawari` | 5 | 1h | 2024-02 起 | GHI 真值（产品 A） |
| ERA5 | Open-Meteo `era5` | 9 | 1h | 2024-02 起 | 云量参考真值（产品 B）+ 辐射交叉验证 |

> **注意**：GFS 1h 输出为 Open-Meteo 对 3h 原生数据的时间插值产物，相邻 1h 值非独立观测。
> 对 168h 输入的序列模型，实际独立信息点约 56 个。此限制在 §8.3 讨论。

### 3.3 Descriptive Statistics 描述性统计

- 每站特征表行数：68,544（938 天 × 24h × 3 时效 + 边界）；
- 时间跨度：2024-02-01 ~ 2026-09-09；
- 白天样本占比：51.8%（`is_day_fcst == 1`）；
- 夜间样本：GHI ≡ 0，产品 A 不训练/不评估；
- 天气类型分布（白天）：clear 20,787 / partly 7,203 / overcast 7,287；
- 图件：Fig. 1(c)(d)（待出）。

| 变量 | 均值 | 标准差 | 最小 | 最大 | 缺失率 |
|---|---|---|---|---|---|
| ghi_fcst (W/m²) | 待补 | 待补 | 0 | 1036 | 0% |
| ghi_obs_sat (W/m²) | 待补 | 待补 | 0 | 1084 | 0.4% |
| cloud_cover_fcst (%) | 待补 | 待补 | 0 | 100 | 0.04% |
| cloud_cover_obs (%) | 待补 | 待补 | 0 | 100 | 0% |
| ghi_err_fcst (W/m², 白天) | −30.6 | 154.8 | −940 | 855 | 0.4% |

> "待补"项需排除 kt 异常值后按 20 站汇总重算（服务器结果落盘后补全）。

### 3.4 Quality Control 质量控制

| 步骤 | 操作 | 实现 |
|---|---|---|
| 真值缺失 | 剔除（不插补） | `clean_data.py` |
| 辐射负值 | 置零 | `clean_data.py` |
| 云量越界 | 裁剪至 [0, 100] | `clean_data.py` |
| 质量报告 | 每站生成 `quality_report_<site>.md` | `clean_data.py` |

### 3.5 Cloud Input Imputation 云量输入缺失插补

缺失时按优先级执行：

| 优先级 | 方法 | 说明 |
|---|---|---|
| 1 | 同时刻其他时效中位数 | median{c(s,T,l') : l' ≠ l} |
| 2 | LightGBM 回归插补 | 输入 = 其余数值特征；训练集 = 非缺失样本 |
| 3 | 站点×时效×月份中位数 | 兜底 |

- 插补值裁剪至 [0, 100]；输出 `cloud_imputed` 标志列；
- 实现：`src/s03_models/train/cloud_impute.py`。

### 3.6 Feature Engineering 特征工程

#### 3.6.1 特征清单（33 数值 + 2 类别）

| 类别 | 特征 | 数量 | 说明 |
|---|---|---|---|
| 辐射预报 | ghi_fcst, dhi_fcst, dni_fcst, gti_fcst, terrestrial_fcst | 5 | GFS 原始 |
| 晴空与指数 | ghi_clear_sky, dni_clear_sky, kt_fcst, kni, ghi_fcst_minus_clear, diffuse_fraction | 6 | pvlib Ineichen 晴空模型 |
| 云场 | cloud_cover_fcst, cloud_cover_change, cloud_cover_roll1, cloud_cover_roll3 | 4 | roll 为右对齐滚动均值 |
| 气象 | temp_fcst, rh_fcst, dewpoint_fcst, wind_speed_fcst, wind_dir_fcst, pressure_fcst, precip_fcst, sunshine_fcst | 8 | GFS 原始 |
| 几何与时间 | solar_elevation, solar_azimuth, hour_local_sin, hour_local_cos, doy_sin, doy_cos, month, lead_time | 8 | hour_local = 北京时（tz_convert Asia/Shanghai）；非真太阳时但省内差≤16min 可忽略 |
| 滞后 | ghi_fcst_lag1, ghi_fcst_lag2 | 2 | 前一/前二 target_time 的 GFS 预报（同 lead） |
| 类别 | station_id (one-hot 20d), season (one-hot 4d) | 2 | 树模型与深度模型均 one-hot |

#### 3.6.2 类别编码

- 树模型：one-hot（20 站 + 4 季）拼接 33 数值特征 → LightGBM / XGBoost；
- 深度模型：one-hot 作为常数通道拼接在每个时间步；
- 连续数值特征标准化（z-score，仅用拟合子集统计量）；one-hot 保持 0/1 不标准化；
- 编码消融（one-hot / native / target）写入五折 CV，结果存 `encoding_ablation.csv`。

#### 3.6.3 已知数据质量问题（⚠️ 待修复）

| 问题 | 影响 | 修复方案 | 状态 |
|---|---|---|---|
| `kt_fcst` 日出日落边缘爆炸（`ghi_clear_sky > 0` 阈值太低） | 白天 8.6% 样本 kt > 1.5（物理不可能），max = 190,877 | 阈值改 `ghi_clear_sky > 50 W/m²` + clip kt ∈ [0, 1.5] | **特征层已修(09-12)；敏感性已验证(09-13)：无实质影响** |
| `kni`、`diffuse_fraction` 同理爆炸 | kni max = 98,652；diffuse_fraction max = 17.2 | 同上（kni clip [0,1.5]、diffuse clip [0,1]） | **特征层已修(09-12)；敏感性已验证(09-13)** |
| `hour_local_sin/cos` 时区 | 实为北京时（非 UTC，此前误判）；与真太阳时差≤16min | 无需修改 | **已豁免(09-12 更正)** |
| `sky_type` 分组阈值与 config 注释不一致 | 注释写 [0.3, 0.5, 0.7]，实际 bins = [0.44, 0.73, 0.97] | 统一注释或统一 bins | **未修复** |

---

## 4 Methodology 方法

### 4.1 Pipeline Overview 流程概览

```
Raw Data (GFS / Himawari / ERA5)
  → Clean (QC + 缺失剔除)
    → Featured (33+2 特征)
      → Split (训练池 / 测试年冻结)
        → 5-Fold CV (超参选择, v1 only)
          → Final Train (4折拟合 + 内部早停)
            → Conformal Calibration (折0)
              → Test Evaluation (一次性)
                → Figures & Tables
```

- 图件：Fig. 2 `figs/00_overview/fig_pipeline`（待绘制）。

### 4.2 Experimental Design 实验设计

#### 4.2.1 Data Splitting 数据切分

| 集合 | 时间窗口 | 用途 | 约束 |
|---|---|---|---|
| 训练池 | 2024-02-01 ~ 2025-08-31（19 个月） | CV + 最终训练 | 不参与测试评估 |
| 测试年 | 2025-09-01 ~ 2026-08-31（12 个月） | 一次性最终评估 | 不参与训练/调参/校准 |
| 干净补充段 | 2026-02-01 ~ 2026-08-31（7 个月） | 测试年子集，未暴露 | 单独报告（§8.1） |
| 暴露段 | 2025-09-01 ~ 2026-01-31（5 个月） | 测试年子集，早期开发曾暴露 | 单独报告 + 披露 |

#### 4.2.2 Month-Stratified 5-Fold CV 月分层五折验证

- 对每个日历月 m，将日期等分为 5 个连续时段：D_m = ∪_{k=1}^5 V_{m,k}；
- 第 k 折验证集：V_k = ∪_m V_{m,k}（覆盖 12 个月，约 20%）；
- 选择准则：τ = 0.10/0.50/0.90 三分位平均 pinball；
- 输出：`selection.json`（最终训练自动读取）；
- 适用范围：仅 v1_ml（树/线性族）；深度模型固定超参（§4.3.7）。

#### 4.2.3 Final Training / Calibration / Early-Stop 最终训练、校准与早停

| 子集 | 来源 | 比例 | 用途 |
|---|---|---|---|
| 训练子集 | 训练池 − 折 0 | ~80% | 最终模型训练 |
| 拟合子集 | 训练子集前 85%（按日期排序） | ~68% of pool | 模型参数拟合 + 标准化统计量 |
| 间隔 | 拟合子集与早停集之间 | 15 天 | 消除时序自相关泄漏 |
| 早停集 | 训练子集尾部 15% | ~12% of pool | 早停（patience=4） |
| 校准折 | 折 0（每月前 ~20% 连续天数） | ~20% of pool | Conformal 校准（不参与训练/早停） |

- 标准化：z-score，μ/σ 仅用拟合子集计算；
- 目标缩放：ỹ = (y − μ_fit) / σ_fit，预测反变换 q̂ = q̃ · σ_fit + μ_fit；
- 物理单位：GHI = W m⁻²，云量 = %。

### 4.3 Model Families 模型族

#### 4.3.1 Baseline Models 基线模型

| 模型 | 方法 | 分位数来源 | 说明 |
|---|---|---|---|
| raw | GFS 原始预报 | 无（点预测） | 下界参照 |
| bias | 月×时效均值偏差订正 | 校准折残差经验分位 | 最简订正 |
| linear (OLS) | 普通最小二乘 | 校准折残差经验分位 | 线性 MOS 基线 |
| ridge | Ridge 回归（α 由 CV 选择） | 校准折残差经验分位 | 正则化线性 |
| persistence | GHI_obs(t − l) | — | **待新增**（skill score 基准） |
| smart persistence | kt_obs(t−l) · GHI_clear(t) | — | **待新增** |
| climatology | 月气候态 GHI | — | **待新增** |

> ⚠️ persistence / smart persistence / climatology 为 skill score 必需，当前未实现，
> 计划服务器 full 结果落盘后补充（~4h）。

#### 4.3.2 Tree-based Models 树模型

| 模型 | 分位数方法 | 变体 | 超参搜索 |
|---|---|---|---|
| LightGBM | 原生 quantile objective（7τ 各训练一棵） | per_lead / lead_feature | 5 组候选 |
| XGBoost | 原生 quantile objective | per_lead / lead_feature | 5 组候选 |

- 变体定义：
  - `per_lead`：D+1/D+2/D+3 各训练一套（3 个独立模型）；
  - `lead_feature`：全时效合并，lead_time 作为特征输入（1 个模型）。

#### 4.3.3 Shallow Deep Models 浅层深度模型

| 版本 | 模型 | 结构 | 关键参数 |
|---|---|---|---|
| v2 | MLP | 2 层全连接 + ReLU | hidden 256/128, dropout 0.1 |
| v3 | CNN | 2 层 1D 卷积 + GAP | channels 64/128, kernel 3 |
| v4 | TCN | 3 层膨胀因果卷积 | kernel 2, dilation 1/2/4, dropout 0.1 |

#### 4.3.4 Sequence & Attention Models 序列与注意力模型

| 版本 | 模型 | 核心机制 | 输入长度 | 原始文献 |
|---|---|---|---|---|
| v5 | LSTM | 2 层, hidden 64 | 168h | Hochreiter & Schmidhuber 1997 |
| v6 | Transformer | 2 层 encoder, 4 head, d=64 | 168h | Vaswani et al. 2017 |
| v7 | Autoformer | 序列分解 + Auto-Correlation | 168h | Wu et al. 2021 |
| v8 | Informer | ProbSparse attention + distilling | 168h | Zhou et al. 2021 |
| v9 | FEDformer | 频率增强 + 序列分解 | 168h | Zhou et al. 2022 |
| v10 | iTransformer | 特征维 token + 2 层 encoder | 168h | Liu et al. 2024 |
| v11 | PatchTST | patch=16, stride=8, 3 层, 8 head, d=128 | 168h | Nie et al. 2023 |
| v12 | DLinear | 趋势/季节分解 + 线性 | 168h | Zeng et al. 2023 |
| v13 | TimesNet | FFT 周期发现 + 2D Inception | 168h | Wu et al. 2023 |
| v14 | TSMixer | 时间混合 + 特征混合 MLP | 168h | Chen et al. 2023 |

> **编号映射（旧 → 新）**：仅 3 个模型改号，其余不变。
> `v3_lstm→v5_lstm`、`v4_cnn→v3_cnn`、`v5_tcn→v4_tcn`。
> 调整目的：使 v 编号与本文档族顺序（浅层 → 序列 → 物理）单调一致。
> ✅ **M1 物理重命名已完成(09-13)**：`reports/03_modeling/` 目录、`train_deep.py` 的 `MODEL_VNUM`
> 及下游脚本读盘列表均已改为新编号，磁盘与文档一致；上表仅作历史留痕。
> （服务器端目录在下次会话用新编号重跑时自然对齐。）

#### 4.3.5 Physics-Informed Models 物理约束模型

| 版本 | 模型 | 约束项 | 损失 |
|---|---|---|---|
| v15 | PINN (Quantile MLP + physics) | ① 非负 q̂ ≥ 0；② 晴空上限 q̂ ≤ GHI_clear；③ kt 一致性 | L = L_pinball + λ·L_phy |

- 物理损失：L_phy = ReLU(−q̂) + ReLU(q̂ − GHI_clear)；
- 待扩展（future work）：GHI = DHI + DNI·cos(θz) 几何一致性、|dGHI/dt| 时间连续性。

#### 4.3.6 Model Comparison Table 模型对比总表

| 族 | 模型数 | 参数量级 | 输入 | 超参搜索 | 种子 | 分位数方法 |
|---|---|---|---|---|---|---|
| Baseline | 4 (+3 待补) | 0 | 点 | 无 | — | 校准折残差 |
| Tree | 2 × 2 变体 | ~10⁵ (trees) | 表格 33+24 | 5 组 CV | 0 / 0–4 | 原生 quantile |
| Shallow Deep | 3 | 10⁴–10⁵ | 168h × 33ch | 固定 | 0–4 | 7τ 输出头 |
| Sequence/Attention | 10 | 10⁵–10⁶ | 168h × 33ch | 固定 | 0–4 | 7τ 输出头 |
| Physics-Informed | 1 | ~10⁴ | 168h × 33ch | 固定 | 0–4 | 7τ + 物理惩罚 |
| **Ensemble** | **待补** | — | — | CV 加权 / Stacking | — | 分位平均 |

#### 4.3.7 Deep Model Policy 深度模型超参策略

- **不做事先网格搜索**（结构/学习率/批大小/轮数固定）；
- 理由：训练池仅 19 个月（拟合子集 ~393 天），深度模型超参搜索的 CV 不稳定；
- Framing：深度模型作为**结构对照**，回答"小样本下深度模型是否优于树模型"（H1/H2）；
- 多种子证据：full 预算 seed 0–4，稳健性章节报告 mean ± SD；
- 若需调参：优先扫描学习率（1e-4 / 1e-3 / 1e-2）与 d_model（32 / 64 / 128）。

### 4.4 Loss Function 损失函数

#### Pinball Loss 分位数损失

ρ_τ(y, q) = max{τ(y − q), (τ − 1)(y − q)}

#### 总体损失

L = (1/|T|) Σ_τ (1/n) Σ_i ρ_τ(y_i, q̂_{τ,i})

#### 深度模型附加项

| 附加项 | 公式 | 权重 | 适用 |
|---|---|---|---|
| 分位单调惩罚 | L_mono = Σ_j ReLU(q̂_{τ_j} − q̂_{τ_{j+1}}) | 0.1 | v2–v15 |
| PINN 物理惩罚 | L_phy = ReLU(−q̂) + ReLU(q̂ − GHI_clear) | λ | v15 only |

### 4.5 Conformal Quantile Calibration 分位数校准

#### 4.5.1 当前方法（Split Conformal）

- 校准折（折 0）计算 conformity score：s_i^(τ) = y_i − q̂_τ(x_i)；
- 经验分位修正量：δ_τ = Q_τ{s_i^(τ)}；
- 测试预测修正：q̂'_τ(x) = q̂_τ(x) + δ_τ；
- 校准后重新排序（isotonic）并裁剪至物理范围。

#### 4.5.2 已知局限与改进方案

| 局限 | 原因 | 改进方案 | 优先级 |
|---|---|---|---|
| 可交换性假设违背 | 折 0 为每月前 20% 连续天数，块内自相关强 | Moving-block bootstrap（块长 7 天）——**已实现(09-13)**：滚动式复评，split≈blockbb（价值在降方差）；阴天仍欠覆盖→Mondrian 留后续 | P0（已做） |
| 边际覆盖 ≠ 条件覆盖 | Conformal 只保证整体覆盖率 | 分组报告 + Mondrian conformal（按 sky_type） | P1 |
| 分布漂移 | 训练池与测试年跨 2 个完整年 | ACI (Gibbs & Candès 2021) | P2 (future work) |

### 4.6 Quantile Crossing Handling 分位交叉处理

| 模型族 | 训练时约束 | 后处理 | 报告指标 |
|---|---|---|---|
| 树模型 | 无（独立 7τ 各训练一棵） | 排序（isotonic rearrangement） | crossing_rate（排序前） |
| 深度模型 | 单调惩罚 L_mono（权重 0.1） | 排序 | crossing_rate（排序前） |
| 待引入 | NCQRNN (Xia 2024) / QRF (Meinshausen 2006) | — | — |

---

## 5 Evaluation Metrics 评价指标

### 5.1 Point Forecast Metrics 点预测指标

| 指标 | 公式 | 用途 | 层次 |
|---|---|---|---|
| MAE | (1/n)Σ|y_i − q̂_{0.5,i}| | 中心预测绝对误差 | Primary |
| RMSE | √[(1/n)Σ(y_i − q̂_{0.5,i})²] | 对大误差敏感 | Primary |
| Bias (MBE) | (1/n)Σ(q̂_{0.5,i} − y_i) | 系统偏差方向 | Primary |
| **RMSE Skill Score** | 1 − RMSE_model / RMSE_baseline | 相对 optimal baseline 改善 | **Primary（待新增）** |

> Skill score baseline = Smart Persistence 与 Climatology 的最优凸组合（Yang et al. 2020）。

### 5.2 Probabilistic Metrics 概率预测指标

| 指标 | 说明 | 用途 | 层次 |
|---|---|---|---|
| mean_pinball | 未加权 7τ 平均 pinball（**不是 CRPS**） | 分位数综合质量 | Primary |
| crps_q7_trunc | 截尾 [0.05, 0.95] 分位网格积分近似 | 可比相对指标 | Secondary |
| Coverage Ĉ_α | (1/n)Σ 1[q_lo ≤ y ≤ q_hi] | 校准质量 | Primary |
| Interval Width Ŵ_α | (1/n)Σ(q_hi − q_lo) | 尖锐度 | Primary |
| Crossing Rate | 排序前相邻分位逆序的样本比例 | 分位自洽性 | Diagnostic |
| **Conditional Coverage** | 按 kt / sky_type / solar_elevation 分组 | 条件校准均匀性 | **Diagnostic（待新增）** |

### 5.3 Diagnostic Tools 诊断工具

| 工具 | 说明 | 统计检验 |
|---|---|---|
| Reliability Diagram | 名义分位 vs 经验分位（目标 = 对角线） | — |
| PIT Histogram | 概率积分变换（目标 = Uniform[0,1]） | KS + Anderson-Darling |
| Sharpness Plot | 区间宽度分布（小提琴图） | — |
| Coverage Calibration Curve | 名义 vs 实际覆盖率（按分组） | Binomial CI |

### 5.4 Statistical Tests 统计检验

| 检验 | 用途 | 修正 |
|---|---|---|
| Diebold-Mariano (DM) | 模型对预测精度差异显著性 | Harvey-Leybourne-Newbold (1997) 小样本修正 |
| 多重比较校正 | 21 模型 C(21,2)=210 对 | Benjamini-Hochberg FDR (q<0.05) |
| Bootstrap CI | 指标置信区间 | Moving-block bootstrap（块长 7 天） |
| Effective Sample Size | 空间/时间自相关修正 | GFS 网格聚类 + ACF 积分 |

### 5.5 Grouped Evaluation 分组评估

| 分组维度 | 水平 | 来源 |
|---|---|---|
| 时效 | D+1, D+2, D+3 | lead_time |
| 站点 | 20 站 | station_id |
| 区域 | 5 层 | config/01_sites.yaml |
| 月份 | 1–12 | month |
| 季节 | 春/夏/秋/冬 | season |
| 天气类型 | clear / partly / overcast | sky_type（⚠️ 阈值待统一） |
| 太阳高度角 | 低(<15°) / 中(15–45°) / 高(>45°) | solar_elevation |

---

## 6 Implementation 实现

### 6.1 Experimental Workflow 实验流程

| 步骤 | 操作 | 脚本 | 输出 |
|---|---|---|---|
| 1 | 数据抓取 | `s01_data/fetch/fetch_data.py` | `data/01_raw/` |
| 2 | 清洗 + QC | `s01_data/clean/clean_data.py` | `data/02_clean/` |
| 3 | 特征工程 | `s01_data/features/features.py` | `data/03_featured/` |
| 4 | 数据门禁 + 审计图 | `s04_evaluation/analysis/data_availability.py` | `reports/01_data_audit/` |
| 5 | 月分层五折 CV | `s02_experiment/cv_month_balanced_quantile.py` | `reports/02_experiment/cv/` |
| 6 | v1_ml 训练 | `s03_models/train/train_quantile_v1.py` | `reports/03_modeling/v1_ml/` |
| 7 | v2–v15 训练 | `s03_models/train/train_deep.py` | `reports/03_modeling/vX_*/` |
| 8 | Conformal 校准 | `s04_evaluation/calibration/calibrate_quantiles.py` | `*/calibrated/` |
| 9 | 测试集评估 + 出图 | `s04_evaluation/analysis/plot_*.py` | `figs/` + `reports/` |
| 10 | 稳健性检验 | `s04_evaluation/verification/` | `reports/05_robustness/` |

### 6.2 Hyperparameter Selection 超参选择

- 详见 `docs/03_hyperparameters.md`；
- 搜索范围：仅 v1_ml（LightGBM 5 组 × XGBoost 5 组 × Ridge 6α）；
- 选择准则：三分位（τ=0.1/0.5/0.9）平均 pinball（树）/ 验证 MAE（ridge）；
- 输出：`selection.json`，最终训练自动读取；缺失时回退默认 + 日志警告。

### 6.3 Software & Hardware 软硬件环境

| 项目 | 配置 |
|---|---|
| Python | 3.12+ |
| 关键库 | numpy, pandas, lightgbm, xgboost, scikit-learn, torch, pvlib, scipy |
| 本地 | CPU only（lite 预算） |
| 服务器 | AutoDL RTX 4090 24GB, torch 2.1.2+cu121, Python 3.10.8（`connect.westb.seetacloud.com:21410`；早期 4080 SUPER 实例已弃用） |
| 版本锁定 | `requirements.txt`（⚠️ 待补 pip freeze lock） |

### 6.4 Reproducibility 可复现性

| 项目 | 当前状态 | 待补 |
|---|---|---|
| 随机种子 | full: 0–4; lite: 0 | — |
| cudnn deterministic | 未设置 | `torch.backends.cudnn.deterministic = True` |
| PYTHONHASHSEED | 未设置 | `PYTHONHASHSEED=0` |
| 容器化 | 无 | Dockerfile（future work） |
| 代码可用性 | 本地仓库 | 投稿时附 GitHub/Zenodo DOI |

---

## 7 Expected Outcomes & Success Criteria 预期成果与成功标准

### 7.1 Primary Outcomes 主要预期

| 指标 | 目标 | 依据 |
|---|---|---|
| RMSE skill score (GHI, D+1) | > 0.15 | Bakker 2019: QR 相对 raw 改善 ~20% |
| RMSE skill score (GHI, D+3) | > 0.08 | 时效越长改善越小（Pierro 2015） |
| 90% 覆盖率偏差 | < 3pp | Conformal 理论保证 |
| Crossing rate（校准后） | < 5% | 单调惩罚 + 排序 |

### 7.2 Secondary Outcomes 次要预期

- 树模型在 D+1 排名第一（H1）；
- 深度模型在 D+3 相对改善（H2）；
- 阴天条件覆盖率低于晴天（H3）；
- GFS-vs-ERA5 云量 r < 0.9（H4，云量订正有空间）。

### 7.3 Falsification 推翻条件

| 条件 | 后果 |
|---|---|
| 所有模型 skill score ≤ 0 | 订正无效，检查数据质量/特征设计 |
| 深度模型全面优于树模型 | 推翻 H1，讨论小样本正则化效果 |
| GFS-vs-ERA5 云量 r > 0.95 | 产品 B 科学价值受限，重新定位交付物 |

---

## 8 Limitations & Threats to Validity 限制与效度威胁

### 8.1 Test Period Historical Exposure 测试期历史暴露

| 项目 | 说明 |
|---|---|
| 暴露窗口 | 2025-09 ~ 2026-01（早期开发中用于模型比较与误差分析） |
| 干净窗口 | 2026-02 ~ 2026-08（7 个月，未暴露） |
| 处理方式 | ① 测试年拆两段分别报告；② 正文对比两段差异；③ 论文显式披露 |
| 长期方案 | 2026-09 起冻结干净验证窗口，12 个月后替代 |
| 为什么不用 7 个月作主测试 | 缺秋冬，无法评价季节泛化 |

### 8.2 Data Length & Sample Size 数据长度与样本量

- GFS 有效预报仅 2024-02 起（previous runs 2019–2023 为空）；
- 训练池 19 个月、拟合子集 ~393 天；
- 168h 输入的有效独立窗口 ~2000（远小于 Autoformer 原论文 34k+）；
- 无法执行多年交叉验证；
- 结论适用边界：限定为"小样本 NWP 订正场景"。

### 8.3 Input Interpolation Artifacts 输入插值伪相关

- GFS Previous Runs 原生 3h，Open-Meteo 输出 1h 为时间插值；
- 168h 窗口实际独立信息点 ~56 个；
- 序列模型可能学到插值核平滑效应而非大气信号；
- 缓解：full 预算统一 168h（公平对照）；讨论中声明。

### 8.4 Truth Uncertainty 真值不确定性

- Himawari GHI 反演 RMSE ~20–40 W/m²（Azam 2026; Xu & Mao 2024）；
- 模型 MAE 降至 ~30 W/m² 时无法区分模型改善与真值噪声；
- ERA5 云量为再分析，非独立观测；
- 缓解：报告 irreducible floor；future work 引入地面辐射站校验。

### 8.5 Spatial Autocorrelation 空间自相关

- GFS 0.25°（~25 km），江苏 20 站可能仅 4–8 个独立格点；
- 站点间误差相关 r 可能 > 0.9；
- 后果：DM p 值被压低；LOSO 低估泛化误差；
- 缓解：报告 effective sample size；LOSO 改为 Regional LOSO（5 层各留一）。

### 8.6 Model Selection Bias 模型选择偏差

- 五折 CV 选超参存在 winner's curse；
- 缓解：报告 CV-test gap；多种子 + DM 确认排名稳定性。

### 8.7 Feature Timezone 特征时区（已更正/豁免）

- `hour_local_sin/cos` 由 `tz_convert("Asia/Shanghai")` 得到，为**北京时**（此前误判为 UTC，09-12 更正）；
- 与真太阳时差异 ≤16 min（江苏经度跨度），对逐时订正可忽略；
- 结论：**豁免**，不修改；如需更物理可后续加真太阳时特征（future work）。

### 8.8 kt Feature Explosion kt 特征爆炸（特征层已修 + 敏感性已验证）

- 原缺陷：`kt = ghi / ghi_clear_sky` 阈值 `>0`，日出日落边缘分母→0⁺ 致 kt→∞（max 190,877，白天 8.6% >1.5）；
- 09-12 修复：`features.py` 分母门槛改 50 W/m² + clip（kt/kni∈[0,1.5]、diffuse_fraction∈[0,1]）；重生成后 max=1.5、越界 0；
- 深度模型 NaN 安全：`train_deep.py` 用 `nan_to_num(nan=0)`，门槛提高带来的额外 NaN 不致训练失败；
- **敏感性已完成(09-13)**：头部深度模型（tcn/transformer/informer/autoformer）fixed-kt 重跑，差异 ±1–2% 噪声内；树 ±0.1 → kt bug 不改变结论；主结果沿用 broken-kt 全套保持一致性，fixed-kt 作稳健性章节。

### 8.9 GFS Version Drift GFS 版本漂移

- 2024-02 之前数据为空（v16→v17 切换期）；
- 按项目决定不追踪，误差分析不按版本分段；
- 记录于 `docs/06_protocol_changelog.md`。

---

## 9 Reproducibility Entry Points 复现入口

| 模块 | 文件 |
|---|---|
| 数据切分 | `src/s02_experiment/split_protocol.py` |
| 云量插补 | `src/s03_models/train/cloud_impute.py` |
| v1_ml 训练 | `src/s03_models/train/train_quantile_v1.py` |
| 深度模型训练 | `src/s03_models/train/train_deep.py` |
| 正式结构实现 | `src/s03_models/train/formal_architectures.py` |
| 超参选择 | `src/s02_experiment/cv_month_balanced_quantile.py` |
| Conformal 校准 | `src/s04_evaluation/calibration/calibrate_quantiles.py` |
| 评估出图 | `src/s04_evaluation/analysis/plot_quantile_results.py` |
| 深度对比出图 | `src/s04_evaluation/analysis/plot_deep_quantile.py` |
| 稳健性检验 | `src/s04_evaluation/verification/verification_robustness.py` |
| kt 分组稳定性 | `src/s04_evaluation/verification/kt_grouping_stability.py` |
| 服务器全流程 | `scripts/05_server/run_server_full.sh` |

---

## 10 File Structure Mapping 文件结构映射

| 文档章节 | 代码 | 结果 | 图件 |
|---|---|---|---|
| §3 Data | `src/s01_data/{fetch,clean,features}` | `data/{01_raw,02_clean,03_featured}` | `figs/01_data_audit`, `figs/02_site_design` |
| §4.2 Experimental Design | `src/s02_experiment/` | `reports/02_experiment/cv/` | `figs/01_data_audit` |
| §4.3 Model Families | `src/s03_models/train/` | `reports/03_modeling/vX_*/` | `figs/03_modeling/` |
| §5 Evaluation | `src/s04_evaluation/{analysis,calibration,verification}` | `reports/{04,05,06}_*` | `figs/{04,05,06}_*` |

完整目录树与命名规则见 `docs/04_project_organization.md`。

---

## References 参考文献（正文引用）

> 完整文献库见 `docs/05_review.md` 与 `literature/00_index.md`。以下仅列本文档直接引用条目。

| 键 | 引用 |
|---|---|
| Yang2020 | Yang, D., et al. (2020). Verification of deterministic solar forecasts. Solar Energy. |
| Bakker2019 | Bakker, K., et al. (2019). Comparison of statistical post-processing methods for probabilistic NWP forecasts of solar radiation. Solar Energy. |
| Mayer2024 | Mayer, M.J., Yang, D. (2024). Optimal place to apply post-processing in the deterministic PV power forecasting workflow. Applied Energy, 371, 123681. |
| Bire2026 | Bire, C., et al. (2026). Bounded residual TCN for multi-site day-ahead NWP GHI correction. Energy and AI. |
| Xia2024 | Xia, X., et al. (2024). Non-crossing quantile regression neural network for ensemble weather forecasts. Adv. Atmos. Sci. |
| TerrénSerrano2026 | Terrén-Serrano, G., et al. (2026). Probabilistic day-ahead forecasting of system-level renewable energy. Nature Communications, 17, 3307. |
| Gibbs2021 | Gibbs, I., Candès, E. (2021). Adaptive conformal inference under distribution shift. NeurIPS. |
| Harvey1997 | Harvey, D., Leybourne, S., Newbold, P. (1997). Testing the equality of prediction mean squared errors. Int. J. Forecasting, 13(2), 281–291. |
| Azam2026 | Azam, F., et al. (2026). CAMS radiation service v4.6: Evaluation of Himawari based surface solar irradiance products. Remote Sensing of Environment. |
| Deo2023 | Deo, R.C., et al. (2023). Cloud cover bias correction in numerical weather models using kernel ridge regression. Renewable Energy, 203, 113–130. |
| Lipponen2026 | Lipponen, A., et al. (2026). ML-based approach for solar radiation model uncertainty identification, attribution and bias correction. EGUsphere. |
| Meinshausen2006 | Meinshausen, N. (2006). Quantile regression forests. JMLR, 7, 983–999. |
| Chernozhukov2010 | Chernozhukov, V., Fernández-Val, I., Galichon, A. (2010). Quantile and probability curves without crossing. Econometrica, 78(3), 1093–1125. |
| Savchenko2025 | Savchenko, O., et al. (2025). SolarM2P: map-to-point deep neural network for post-processing of NWP-based solar irradiance forecasts. Przegląd Elektrotechniczny. |
| Mabasa2025 | Mabasa, B., et al. (2025). Evaluating the GHI projected by the GFS model in diverse climatic zones in South Africa. Weather and Forecasting, 40(7), 1047–1064. |
| Hațegan2023 | Hațegan, S.-M., et al. (2023). Calibration of GFS solar irradiation forecasts: A case study in Romania. Energies, 16(6), 2919. |
| Zhang2020 | Zhang, Y., et al. (2020). Validation of GFS day-ahead solar irradiance forecasts in China. arXiv:2007.01639. |
| Verzijlbergh2015 | Verzijlbergh, R.A., et al. (2015). Improved model output statistics of NWP based irradiance forecasts. Solar Energy, 118, 634–645. |
| Pierro2015 | Pierro, M., et al. (2015). Model output statistics cascade to improve day ahead solar irradiance forecast. Solar Energy, 117, 99–113. |
