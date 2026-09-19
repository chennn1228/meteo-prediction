# 辐照度与云量概率订正研究

版本 10.0 ｜ 2026-09-11 ｜ 方法与技术路径

## 1 研究问题

### 1.1 任务定义

本研究包含两个独立交付产品。

产品 A 为辐照度订正：输入 GFS 预报与站点特征，输出 D+1、D+2、D+3 的 GHI
条件分位数，主真值为 Himawari 反演 GHI。

产品 B 为云量订正：输入 GFS 预报与站点特征，输出 D+1、D+2、D+3 的总云量
条件分位数，参考真值为 ERA5 总云量。ERA5 属再分析产品，论文须声明其非独立观测属性。

两个产品彼此独立：云量不进入 GHI 模型的输入，GHI 也不进入云量模型；
交付定位是"可复用的订正气象数据集"，云量产品面向云敏感的下游应用
（光伏功率预测的云修正、负荷与辐照度联合分析、卫星/再分析交叉校验），
不承担 GHI 产品的中间量角色，因此不存在异源"云量→辐照度"传递误差。

给定站点 $s$、目标时刻 $T$ 与时效 $l\in\{24,48,72\}$，两个产品均输出
$\mathbf{x}_{s,T,l}$ 的条件分位数：

$$
\hat q_{\tau}(s,T,l),\quad \tau\in\{0.05,0.10,0.25,0.50,0.75,0.90,0.95\}
$$

### 1.2 产品与真值

辐照度产品真值为 Himawari 反演 GHI（卫星观测）；云量产品参考真值为 ERA5 总云量
（再分析产品，非独立观测，存在与 NWP 同源的风险）。

### 1.3 交付物

分位数预测、校准参数、评价指标、模型脚本、复现图件与结果数据。

## 2 数据

### 2.1 数据源

| 来源 | 变量 | 时间 | 用途 |
|---|---|---|---|
| GFS Previous Runs | 15 个气象与辐射变量，3 时效 | 2024-02 起有效 | 预报输入 |
| Himawari | GHI、DHI、DNI、GTI | 2024-02 起匹配 | 辐照度真值 |
| ERA5 | 总云量与分层云量 | 2024-02 起匹配 | 云量真值 |

### 2.2 站点

江苏分层 20 站：苏南内陆、苏中内陆、苏中沿海、苏北内陆、苏北沿海。
层内最大最小布点，配置见 `config/01_sites.yaml`。

### 2.3 质量控制

真值缺失剔除，辐射负值置零，云量裁剪至 $[0,100]$。

### 2.4 云量输入缺失插补

令 $c_{s,T,l}$ 为 GFS 云量预报。缺失时按顺序执行：

1. 同时刻其他时效中位数
$$
\tilde c_{s,T,l}=\operatorname{median}\{c_{s,T,l'}:l'\neq l,\ c_{s,T,l'}\ \text{可用}\}
$$
2. LightGBM 回归插补，输入为其余数值特征，训练集仅使用非缺失样本；
3. 站点、时效、月份中位数兜底。

插补值裁剪至 $[0,100]$，输出插补标志。实现见 `src/s03_models/train/cloud_impute.py`。

### 2.5 特征

| 类别 | 特征 |
|---|---|
| 辐射预报 | ghi_fcst、dhi_fcst、dni_fcst、gti_fcst、terrestrial_fcst |
| 晴空与指数 | ghi_clear_sky、dni_clear_sky、kt_fcst、kni、ghi_fcst_minus_clear、diffuse_fraction |
| 云场 | cloud_cover_fcst、cloud_cover_change、cloud_cover_roll1、cloud_cover_roll3 |
| 气象 | temp_fcst、rh_fcst、dewpoint_fcst、wind_speed_fcst、wind_dir_fcst、pressure_fcst、precip_fcst、sunshine_fcst |
| 几何与时间 | solar_elevation、solar_azimuth、hour_local_sin、hour_local_cos、doy_sin、doy_cos、month、lead_time |
| 滞后 | ghi_fcst_lag1、ghi_fcst_lag2 |
| 类别 | station_id、season |

共 33 个数值特征与 2 个类别特征。

### 2.6 类别特征编码

树模型与深度模型均使用 one-hot 编码，不使用裸整数编码：

- station 编码为 20 维 one-hot；
- season 编码为 4 维 one-hot；

树模型将 one-hot 与 33 个数值特征拼接后输入 LightGBM 与 XGBoost。
深度模型将 one-hot 作为常数通道拼接在每一个时间步上，与数值特征共同进入序列模型。
连续数值特征标准化，one-hot 保持 0/1，不参与标准化。

## 3 实验设计

### 3.1 数据切分

时间序列存在自相关，全部按连续时段切分。

#### 3.1.1 训练池与测试集

训练池为 2024-02-01 至 2025-08-31，测试集为 2025-09-01 至 2026-08-31。
测试集不参与训练、调参、校准。

#### 3.1.2 五折月分层验证

对每个日历月 $m$，将日期等分为五个连续时段：

$$
\mathcal D_m=\bigcup_{k=1}^{5}\mathcal V_{m,k}
$$

第 $k$ 折验证集为 $\mathcal V_k=\bigcup_m \mathcal V_{m,k}$，训练集为其余时段。
每折验证比例约 20%，覆盖 12 个月。

由于每个月的日期都被等分五段，折 0 取的是**每个月的前约 20% 连续天数**
（不是某一个季节），5 折轮转后任一折都覆盖 12 个月，因此折 0 具有全年代表性，
被固定用作最终模型的校准折只是约定，不引入季节性偏置。

#### 3.1.3 最终训练、校准与早停

- 训练子集：训练池中排除固定校准折（折 0），约 80%；
- 拟合子集：训练子集按日期排序后去掉尾部 15%，其余 85% 用于拟合模型参数；
- 早停集：训练子集内部按日期排序的尾部 15%；
- 校准折：五折中的固定一折，仅用于 conformal 分位校准；
- 标准化统计量与目标缩放参数只用拟合子集计算；目标缩放采用 z-score
  $\tilde y=(y-\mu_{\text{fit}})/\sigma_{\text{fit}}$，模型输出分位数再按
  $\hat q=\tilde q\,\sigma_{\text{fit}}+\mu_{\text{fit}}$ 反变换回物理单位
  （GHI 为 W m$^{-2}$，云量为 %）；
- 校准折与测试集不参与训练、早停与标准化。

### 3.2 概率预测

#### 3.2.1 分位数定义

$$
q_\tau(x)=\inf\{y:F(y\mid x)\ge\tau\}
$$

#### 3.2.2 损失函数

Pinball 损失：

$$
\rho_\tau(y,q)=\max\{\tau(y-q),(\tau-1)(y-q)\}
$$

总体损失为七个分位的平均：

$$
\mathcal L=\frac{1}{|\mathcal T|}\sum_{\tau\in\mathcal T}\rho_\tau(y,q_\tau)
$$

### 3.3 超参数选择

采用第 3.1.2 节五折验证，准则为三分位平均 pinball：

$$
\mathcal L_{CV}=\frac{1}{5}\sum_{k=1}^{5}
\frac{1}{3}\sum_{\tau\in\{0.1,0.5,0.9\}}\rho_\tau^{(k)}
$$

选择结果写入 `selection.json`，最终训练自动读取。

### 3.4 分位数校准

在校准折计算 conformity score：

$$
s_i^{(\tau)}=y_i-\hat q_\tau(x_i)
$$

取经验分位作为修正量：

$$
\delta_\tau=Q_\tau\{s_i^{(\tau)}\}
$$

测试预测修正为：

$$
\hat q_\tau'(x)=\hat q_\tau(x)+\delta_\tau
$$

校准后重新排序并裁剪至物理范围。

## 4 模型族

### 4.1 v1_ml

模型：raw、bias、linear（OLS）、ridge（Ridge，α 由月分层五折 CV 选择）、
LightGBM、XGBoost。变体含义：

- `per_lead`：D+1、D+2、D+3 各训练一套模型（3 个独立模型）；
- `lead_feature`：全时效合并为单模型，时效作为特征输入。

LightGBM 与 XGBoost 使用原生分位数目标；其余模型使用校准折残差经验分位。

### 4.2 v2_mlp

$$
h_1=\mathrm{ReLU}(W_1 \mathrm{vec}(X)+b_1),\quad
h_2=\mathrm{ReLU}(W_2 h_1+b_2),\quad
\hat q=W_3 h_2+b_3
$$

参数：隐藏维 256、128，Dropout 0.1。

### 4.3 v3_lstm

两层 LSTM，隐藏维 64，Dropout 0.1，末时刻映射至七个分位。

### 4.4 v4_cnn

两层一维卷积，通道 64、128，核长 3，全局平均池化。

### 4.5 v5_tcn

三层膨胀卷积，核长 2，膨胀率 1、2、4，Dropout 0.1。

### 4.6 v6_transformer

投影 64 维，位置编码，两层编码器，四注意力头，前馈 128 维。

### 4.7 v7_autoformer

序列分解与 Auto-Correlation 注意力，输入长度 168 小时。

分解：

$$
X_{trend}=\mathrm{AvgPool}(X),\quad X_{season}=X-X_{trend}
$$

### 4.8 v8_informer

ProbSparse 注意力与 distilling 编码器，输入长度 168 小时。

### 4.9 v9_fedformer

频率增强块与分解块，输入长度 168 小时。

### 4.10 v10_itransformer

特征维作为 token，序列维投影 64 维，两层编码器。

### 4.11 v11_patchtst

patch 长度 16，步长 8，三层编码器，八注意力头，隐藏维 128，输入 168 小时。

### 4.12 v12_dlinear

趋势与季节分解后线性建模：

$$
\hat y=W_{trend}X_{trend}+W_{season}X_{season}
$$

### 4.13 v13_timesnet

FFT 周期发现、二维卷积 inception 块、多周期融合，输入 168 小时。

### 4.14 v14_tsmixer

时间混合与特征混合 MLP。

### 4.15 v15_pinn

分位 MLP 加物理约束：

$$
\mathcal L_{phy}=\mathrm{ReLU}(-\hat q)+\mathrm{ReLU}(\hat q-GHI_{clear})
$$

### 4.16 lite 与 full

lite 与 full 的差异固定如下，正式结论只引用 full。
表中"结构相同"的模型，lite 与 full 的网络结构一致，差异只在输入长度、训练轮数、
抽样与种子；标记"近似/普通编码器"的模型给出的是 lite 实际使用的简化结构，
full 由 `src/s03_models/train/formal_architectures.py` 实现。

| 项目 | lite | full |
|---|---|---|
| 输入长度 | 24 小时（仅快速验证） | 全部模型 168 小时 |
| 训练轮数 | 2–4 | 50 |
| 早停耐心 | 2 | 4 |
| 训练抽样 | 1/8 | 无 |
| 验证抽样 | 1/8 | 无 |
| 批大小 | 1024 | 1024 |
| 随机种子 | 0 | 0–4 |
| v2 MLP | 隐藏维 256/128，1 组 | 隐藏维 256/128，5 种子 |
| v3 LSTM | 2 层，隐藏维 64 | 2 层，隐藏维 64，全量数据 |
| v4 CNN | 2 层，通道 64/128 | 2 层，通道 64/128，全量数据 |
| v5 TCN | 3 层，膨胀率 1/2/4 | 3 层，膨胀率 1/2/4，全量数据 |
| v6 Transformer | 2 层，4 头，64 维 | 2 层，4 头，64 维，全量数据 |
| v7 Autoformer | 滑动平均去趋势 + 普通 Transformer 编码器 | 序列分解块 + Auto-Correlation 注意力 |
| v8 Informer | 普通 Transformer 编码器 | ProbSparse 注意力 + distilling 编码器 |
| v9 FEDformer | 普通编码器 + FFT 乘标量 sigmoid 门 | 频率增强块 + 序列分解块 |
| v10 iTransformer | 2 层，64 维 | 2 层，64 维，全量数据 |
| v11 PatchTST | patch=12，1 层，32 维 | patch=16，stride=8，3 层，8 头，128 维 |
| v12 DLinear | 趋势与季节线性 | 同结构，全量数据 |
| v13 TimesNet | 特征维自适应池化到 8 维 + 两层 Conv2d | FFT 周期发现 + Inception 二维卷积 + 多周期融合 |
| v14 TSMixer | 时间与特征混合 MLP | 同结构，全量数据 |
| v15 PINN | MLP + 晴空惩罚项 | 非负约束、晴空上限、晴空指数一致性 |
| 输出路径 | `lite/quantile/` | `full/quantile/` |

## 5 评价指标

### 5.1 点预测

$$
\mathrm{MAE}=\frac{1}{n}\sum|y_i-\hat q_{0.5,i}|
$$

$$
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum(y_i-\hat q_{0.5,i})^2}
$$

$$
\mathrm{Bias}=\frac{1}{n}\sum(\hat q_{0.5,i}-y_i)
$$

### 5.2 概率预测

平均 pinball 损失（未加权，七个分位数的算术平均，**不是 CRPS**）：

$$
\mathrm{mean\_pinball}=\frac{1}{|\mathcal T|}\sum_{\tau\in\mathcal T}\rho_\tau,
\qquad \rho_\tau=\frac{1}{n}\sum_i \max\big(\tau(y_i-\hat q_{\tau,i}),\ (\tau-1)(y_i-\hat q_{\tau,i})\big)
$$

截尾 CRPS 近似（只在分位网格 $[\tau_{\min},\tau_{\max}]=[0.05,0.95]$ 上积分，
尾部未建模，因此是可比的相对指标而非完整 CRPS）：

$$
\mathrm{crps\_q7\_trunc}=2\sum_{j} w_j\,\rho_{\tau_j},\qquad
w_j=\tfrac{\tau_{j+1}-\tau_{j-1}}{2}\ (1\le j\le 6),\quad
w_0=\tfrac{\tau_1-\tau_0}{2},\quad w_6=\tfrac{\tau_6-\tau_5}{2}
$$

完整 CRPS 需要全分位函数或集合预报，本项目不报告"CRPS"单值。

分位交叉率（排序前统计相邻分位逆序的样本比例）：

$$
\mathrm{crossing}=\frac{1}{n}\sum_i \mathbf 1\big[\exists j:\ \hat q_{\tau_j,i}>\hat q_{\tau_{j+1},i}\big]
$$

预测输出前对分位数排序，但交叉率本身作为质量指标写入结果表：
交叉率高说明独立训练的分位模型不自洽，论文需报告。

覆盖率：

$$
\hat C_\alpha=\frac{1}{n}\sum \mathbf 1[q_{lo,i}\le y_i\le q_{hi,i}]
$$

区间宽度：

$$
\hat W_\alpha=\frac{1}{n}\sum(q_{hi,i}-q_{lo,i})
$$

### 5.3 分组

时效、站点、区域、月份、季节、天气类型。

## 6 实验流程

1. 数据清洗与特征工程；
2. 云量输入缺失插补；
3. 五折月分层验证选择超参数；
4. 按第 3.1.3 节切分最终训练、校准与早停；
5. 训练 v1_ml 与 v2–v15；
6. 校准折计算分位修正；
7. 测试集一次性评估；
8. 保存结果与图件。

## 7 文件结构

目录按本文档步骤编号组织（`src` 子包加 `s` 前缀以保证可 import）：

| 文档步骤 | 代码 | 结果 | 图件 |
|---|---|---|---|
| 2 数据 | `src/s01_data/{fetch,clean,features}` | `data/01_raw/{01_gfs,02_era5,03_satellite}`、`data/02_clean`、`data/03_featured` | `figs/02_site_design` |
| 3 实验设计 | `src/s02_experiment/{split_protocol,cv_month_balanced_quantile}.py` | `reports/02_experiment/cv/month_balanced/{ghi,cloud}` | `figs/01_data_audit` |
| 4 模型族 | `src/s03_models/train/{train_quantile_v1,train_deep}.py` | `reports/03_modeling/vX_{model}/{lite,full}/quantile/{ghi,cloud}` | `figs/03_modeling/vX_{model}` |
| 5 评价 | `src/s04_evaluation/{analysis,calibration,verification}` | `reports/04_error_analysis`、`05_robustness`、`06_probability` | `figs/{04,05,06}_*` |
完整目录树、命名规则、图件索引与脚本清单见 `docs/04_project_organization.md`。

## 8 限制

### 8.1 测试时段的历史暴露

测试集为 2025-09-01 至 2026-08-31。其中 2025-09 至 2026-01 在早期开发中曾用于
模型比较与误差分析，因此该时段并非完全未见样本。

处理方式：

- 所有正式超参数由五折月分层验证选择，不使用测试指标；
- 最终模型只训练四折，另一折用于校准；
- 论文中明确披露测试时段的历史暴露；
- 未来完全干净的验证集从 2026-09 起独立冻结，积累满 12 个月后替代当前测试年。

为什么不用 2026-02~2026-08 作干净测试段：该窗口只有 7 个月且不含秋冬，
无法覆盖完整年周期，会把"季节泛化"从评价里删掉；因此选择"整年测试 + 披露
2025-09~2026-01 的历史暴露"。该取舍记录于 `docs/06_protocol_changelog.md`。

### 8.2 数据长度

有效 GFS 预报自 2024-02 起，无法执行多年交叉验证。训练池 19 个月、测试年 12 个月，
训练/测试比例由"GFS 起点 + 整年测试"两个约束共同决定，不是常规的 8:1:1 划分；
对 15 个深度模型（尤其 Autoformer/TimesNet 类）而言训练样本偏少，属于已知限制，
论文需说明并在结论中限定适用边界。

关于 GFS 模式版本（v16/v17）漂移：2024-02 之前的数据在抓取阶段即为空值，
不属于有效训练数据；按项目决定不再追踪版本切换点，误差分析不按版本分段。
该决定记录于 `docs/06_protocol_changelog.md`。

### 8.3 输入长度

full 预算下全部深度模型统一使用 168 小时输入，保证"结构差异"不被"可见历史长度差异"
混淆；lite 预算仍用 24 小时输入，仅用于本地快速验证，不进入正式结论。

## 9 复现入口

| 模块 | 文件 |
|---|---|
| 数据切分 | `src/s02_experiment/split_protocol.py` |
| 云量插补 | `src/s03_models/train/cloud_impute.py` |
| v1_ml | `src/s03_models/train/train_quantile_v1.py` |
| 深度模型 | `src/s03_models/train/train_deep.py` |
| 超参选择 | `src/s02_experiment/cv_month_balanced_quantile.py` |
| 校准 | `src/s04_evaluation/calibration/calibrate_quantiles.py` |
| 评估 | `src/s04_evaluation/analysis/plot_quantile_results.py`、`src/s04_evaluation/analysis/plot_deep_quantile.py` |
| 稳健性 | `src/s04_evaluation/verification/verification_robustness.py`、`kt_grouping_stability.py` |
