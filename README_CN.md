# 江苏省短期 GHI 概率预测与 GFS 后处理框架

本项目面向江苏省短期太阳总辐照度（Global Horizontal Irradiance, GHI）预测，构建一套统一、可复现的数值天气预报（NWP）后处理与概率预测框架。

核心任务是利用 GFS 历史预报及相关气象信息，对 24、48 和 72 h 预测时效下的 GHI 进行统计与机器学习订正，并进一步研究不同模型、特征体系和空间采样策略下的预测性能与省域泛化能力。

项目采用配置驱动的统一工作流，将数据获取、预处理、特征构建、滚动验证、模型训练、概率校准、评价和结果管理组织在同一套可追溯流程中。

---

## 1. 研究对象与数据体系

### 1.1 预测数据

预测端基于 Open-Meteo 提供的 GFS 历史预报数据。

当前已落地数据按注册的 20 个站点组织：

```text
20 站江苏注册表
    ↓
site × target time × lead time
    ↓
GFS meteorological predictors
```

研究采用三个固定预测时效：

```text
24 h
48 h
72 h
```

Previous Runs 数据表示目标时刻之前相应提前量所保存的历史预报。预测时效用于定义信息可获得性和因果关系，不将其未经验证地解释为原始 GFS 模型初始化时刻。

### 1.2 辐照度参考数据

当前主要空间参考为 JMA/JAXA Himawari 卫星反演 GHI：

```text
Himawari shortwave radiation
→ satellite-derived GHI reference
```

卫星反演数据不是 NWP，也不是再分析资料。当前本地数据中，它与 Previous Runs 字段按相同的 20 个请求位置匹配。

高质量地面辐射观测可作为进一步的点尺度独立验证来源。

### 1.3 数据分层

本地数据遵循统一的三级结构：

```text
data/
├── raw/
├── clean/<clean_hash>/
├── features/<feature_hash>/
├── registry/
├── catalog.json
└── data_inventory.csv
```

其中：

- `raw`：不可覆盖的原始获取数据；
- `clean`：完成时间、空间、字段和物理约束统一后的结构化数据；
- `features`：面向具体模型构建的可训练输入；
- `registry`：空间网格、站点及其他稳定索引；
- `catalog.json`：数据集注册与解析入口；
- `data_inventory.csv`：本地数据资产清单。

每个正式数据产物均通过 receipt、内容哈希和依赖指纹记录来源及生成过程。

---

## 2. 实验协议

研究数据划分为：

```text
Development
2024-02-01 — 2025-08-31

Final test
2025-09-01 — 2026-08-31
```

模型开发采用 nested purged rolling-origin validation（嵌套带隔离间隔的滚动时间验证）。

核心设置：

| 项目 | 设置 |
|---|---:|
| 外层滚动折 | 5 |
| 每个外层折的内层折 | 3 |
| 最大历史回看 | 168 h |
| 最大预测时效 | 72 h |
| 标准隔离间隔 | 240 h |
| 隔离间隔敏感性 | 7 / 10 / 14 d |

滚动验证直接参与模型选择，而不是在模型确定后附加执行。

最终测试期独立于开发过程，在模型、配置和实验协议冻结前不得参与模型选择或开发反馈。

---

## 3. 特征体系

项目将**原始数据获取合同**与**模型特征合同**分离。

原始数据层负责保存具有长期研究价值的气象信息；不同模型根据自身数据结构和学习机制使用不同的 feature contract（特征合同）。

典型输入包括：

- GFS 辐射变量；
- 云量及变化信息；
- 温度、湿度、露点；
- 风速、风向；
- 气压和降水；
- 晴空辐照度与晴空指数；
- 太阳高度角、方位角；
- 时间周期特征；
- 历史可获得预报滞后项；
- GFS 网格空间坐标。

不同模型不强制使用完全相同的特征数量。

例如：

```text
Linear / Ridge
→ 紧凑的表格特征与适当正则化/降维

LightGBM / XGBoost
→ 较丰富的工程化表格特征

Deep Learning
→ 原始气象变量 + 时间序列 + 空间邻域结构
```

所有特征必须满足 forecast issue time（预测发布时刻）的因果可获得性。

站点 ID、网格 ID 等身份变量不得作为模型捷径输入；目标值及未来观测不得进入预测特征。

数据预处理仅在对应训练折内拟合，防止时间或空间信息泄漏。

---

## 4. 模型体系

项目包含从原始预报到统计、机器学习和深度学习的多层模型体系。

### 基准与统计模型

```text
climatology
persistence
smart_persistence
optimal_convex
raw_gfs
bias_correction
linear_mos
ridge_mos
```

### 树模型

```text
lgbm
xgboost
```

### 深度学习架构

```text
mlp
cnn
tcn
lstm
transformer
autoformer
informer
fedformer
itransformer
patchtst
dlinear
timesnet
tsmixer
pinn
```

不同模型可使用不同的特征合同，但所有模型共享相同的时间划分、数据因果规则和评价协议。

---

## 5. 概率预测与评价

概率预测采用固定分位数：

```text
0.05
0.10
0.25
0.50
0.75
0.90
0.95
```

主要模型选择指标为：

```text
mean pinball loss
```

确定性评价包括：

```text
MAE
RMSE
Bias
R²
RMSE skill
```

概率评价包括：

```text
mean pinball loss
coverage
interval width
```

概率预测可进一步通过独立的后置校准时段进行 conformal calibration（保形校准）。

模型解释包括特征组消融、分组置换重要性和 SHAP。解释分析不参与模型选择。

---

## 6. 空间数据与泛化

当前数据为有 receipt 和哈希证据的 20 站本地数据。

完整空间数据库的建立与实验中的训练/测试空间划分相互独立：

```text
完整 20 站数据
        ↓
实验设计
        ├─ 训练位置
        ├─ 空间留出位置
        └─ 区域外推位置
```

省域 GFS025 获取不属于当前仓库状态，未经用户明确指令不得启动。

空间评价包括：

1. stratified spatial holdout（分层空间留出）；
2. province-wide unseen grid（省域未见网格泛化）；
3. regional extrapolation stress test（区域外推压力测试）。

---

## 7. 软件工作流

完整流程为：

```text
configuration
    ↓
raw data
    ↓
clean data
    ↓
model-specific features
    ↓
rolling splits
    ↓
hyperparameter tuning
    ↓
model fitting
    ↓
prediction
    ↓
calibration
    ↓
evaluation
    ↓
analysis / figures
```

科学协议由配置文件统一定义：

```text
config/protocol.yaml
config/data.yaml
config/features.yaml
config/sites.yaml
config/models.yaml
config/experiments.yaml
```

代码不得维护第二套独立实验定义。

正式工作流统一通过：

```text
python -m nwp
```

进入。

---

## 8. 仓库结构

```text
config/          科学协议与实验配置
src/nwp/         Python实现
tests/           单元、集成与端到端测试
docs/            研究协议与技术说明

data/            本地科研数据
outputs/         独立实验运行与结果
```

`data/` 和 `outputs/` 与源代码分离管理。

---

## 9. 安装

基础环境：

```bash
python -m pip install -e .
```

CPU 模型：

```bash
python -m pip install -e ".[cpu]"
```

深度学习开发：

```bash
python -m pip install -e ".[cpu,deep]"
```

完整开发与测试：

```bash
python -m pip install -e ".[cpu,deep,dev]"
```

---

## 10. 使用

验证配置：

```bash
python -m nwp validate --profile nanjing_cpu_diagnostic
```

盘点本地数据：

```bash
python -m nwp data inventory
```

执行至数据切分阶段：

```bash
python -m nwp run \
  --profile nanjing_cpu_diagnostic \
  --to-stage splits
```

显式执行模型阶段：

```bash
python -m nwp run \
  --profile nanjing_cpu_diagnostic \
  --execute-model-stages
```

评价已有运行：

```bash
python -m nwp evaluate --run-id <run-id>
```

生成图表：

```bash
python -m nwp figures --run-id <run-id>
```

查看配置和运行入口：

```bash
python -m nwp status
```

每次实验独立写入：

```text
outputs/<execution>/<run_id>/
```

并保存解析后的配置、provenance、stage receipts、artifact manifest、模型、预测、评价及图表等可复现信息。
