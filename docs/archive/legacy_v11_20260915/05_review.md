# 文献综述

> 版本：v1.3 ｜ 日期：2026-09-11 ｜ 用途：为本项目（NWP 辐照度/云场订正）建立研究背景与方法谱系
> 说明：本综述正文（第 1–7 节）为文献证据，与项目协议无关；
> 第 8 节中标注「历史编号」的条目来自重构前的旧模型编号（旧 v1–v8 体系），
> 现项目为 v1_ml…v15_pinn 双产品独立建模，协议与变更记录见 `docs/06_protocol_changelog.md`。
> 检索说明：本会话学术检索 MCP 工具不可用，按技能降级链采用联网检索 + 原文/摘要页面核验完成。标注 ✓ 表示已核验原文或官方摘要页；标注 ~ 表示依据检索摘要整理，引用前建议复核全文。
> 2026-08-21 核验与补充：按 nature-academic-search 流程补充检索深度学习订正、云量—辐照度误差耦合与区域基准方向；原 34/46 条完成多源核验并补全题录，36 条补充著录（卷期待核）；新增条目见第 3.2/4/5/6/7 节与参考文献 I 组，v4 设计结论见第 8 节第 11–14 条。
> 2026-09-02 融合更新：补充 Nature Communications 系统级概率日前预报、EGUsphere 情境依赖误差归因与订正、SHADECast 扩散生成、skill score meta 分析等 5 条核验文献，结论已融入第 1/2/4/5/6/7/8 节对应小节（不单列模块）。

---

## 1. 研究链条全景

```
真值构建（观测站/卫星反演/再分析）
   ↓
NWP 预报（GFS / ECMWF / WRF-Solar / 区域模式）
   ↓
误差诊断（按天气、季节、时段、辐照度水平分层）
   ↓
订正 / 后处理（Bias → MOS → 概率 → 树模型 → 深度学习）
   ↓
（可选）辐照度→功率链路 + 功率级后处理
   ↓
评估（MAE/RMSE/Bias/R²/nRMSE/CRPS/skill score）
```

| 链条环节 | 要回答的问题 | 代表文献 |
|---|---|---|
| 真值构建 | "真实辐照度"用什么代表？卫星/观测/再分析？ | Vernay et al. 2021；Mayer & Yang 2024；Solargis |
| 误差诊断 | NWP 预报错在哪、什么条件下错？ | Dou et al. 2024；Hațegan et al. 2023 |
| 线性 MOS | 预报+物理特征→真值的线性映射？ | Verzijlbergh et al. 2015；Pierro et al. 2015 |
| 概率订正 | 如何给出带不确定性的区间？ | Bakker et al. 2019；Verbois et al. 2018 |
| 树模型 | 非线性表格特征谁最强？ | Hațegan et al. 2023；Verbois et al. 2018 |
| 深度学习 | CNN/Transformer/扩散能否进一步压低误差？ | Savchenko et al. 2025；TSMixer 2025；扩散模型 2025/2026 |
| 链条位置 | 到底该订正辐照度还是订正功率？ | Mayer & Yang 2024, 2025 |
| 评估基准 | 用什么指标和参照物才算公平？ | Yang et al. 2020；GEFCom2014 |
| 情境依赖误差归因 | 误差在哪些输入条件下最大、由哪个物理量驱动？ | Lipponen et al. 2026；Jiang & Peng 2026 |

---

## 2. 预测对象与目标变量（他们都预测什么）

订正/预测研究的目标变量分四层，从气象量到应用量：

| 目标变量 | 含义 | 代表文献 | 说明 |
|---|---|---|---|
| GHI | 水平面总辐照度 | Pierro 2015；Verzijlbergh 2015；Bakker 2019；Pereira 2019；Hațegan 2023；Savchenko 2025 | 绝对主流，光伏预测的直接气象输入 |
| DNI | 法向直接辐照度 | Han et al. 2026（CNN+晴空模型，CSP 场景） | 聚光太阳能（CSP）关注 |
| GTI | 倾斜面辐照度 | Energies 2026（线性回归+支持向量分位数回归） | 组件倾角/建筑 BIPV 场景 |
| CSI / Kt | 晴空指数 | Lorenz 2009；Verzijlbergh 2015；Bakker 2019；天空成像 SVM-CNN（2025） | 关键中间变量：先订正 Kt 再还原 GHI |
| 云量 | 总云量 | 干旱气象 2021；WRF 云同化（NCAR） | 少数直接订正云量，多数把云量作为特征 |
| 光伏功率 | 电站出力 | Alessandrini 2015；GEFCom2014；Mayer & Yang 2024/2025 | 最终应用目标 |
| 系统级净负荷（需求−风−光联合） | Terrén-Serrano et al. 2026（Nature Communications） | 节点/资源间联合概率分布，服务储备与调度 |

**对本课题的启示**：我们的任务书预测对象是 GHI（辅以云量），属于链条最主流的位置；同时学界普遍建议以 CSI/Kt 为中间量建模（见第 5.1 节），与本课题的晴空指数特征设计完全一致。

---

## 3. 他们用什么数据

### 3.1 真值（被当作"真实"的数据）

| 真值来源 | 代表文献 | 要点 |
|---|---|---|
| 地面辐射站观测 | Bakker 2019（荷兰 KNMI 30 站 CM11 日射计）；Pereira 2019（葡萄牙）；Mayer & Yang 2024（匈牙利 3 电站） | 最权威但站点稀疏，中国辐射观测更少 |
| 卫星反演辐射 | Mayer & Yang 2024（卫星 GHI 与地面 GHI 双真值对比）；Savchenko 2025 | 空间连续，适合区域研究；本身有反演误差 |
| 再分析（ERA5 类） | 大量验证研究默认采用 | 与 NWP 同源风险（订正 ECMWF 时需谨慎） |
| 站点适应（site adaptation） | Vernay et al.（核密度映射校准卫星库）；Himawari-8 站点适应（Renewable Energy 2022）；Solargis 工业实践 | 用地面观测校正卫星/模型辐射产品，思想可直接用于"卫星真值本地化" |
| 风云-4A 卫星 + 深度学习（RadNet） | Lu et al. 2026（npj Clean Energy） | 中国 4 km 分辨率 GHI 估计，时空泛化良好；再分析精度低于遥感产品 |

### 3.2 被订正的预报（NWP 来源）

| NWP | 代表文献 | 说明 |
|---|---|---|
| WRF / WRF-Solar | Pierro 2015；宁夏 EOF-MOS 2013；TSMixer 2025；Jimenez et al. 2016（WRF-Solar 官方介绍） | 区域模式，全球被广泛使用 |
| GFS | Verzijlbergh 2015；Hațegan 2023（罗马尼亚）；Savchenko 2025（乌克兰） | 与本课题 Previous Runs 的 GFS 档案同源 |
| GFS / CMA-WSP2.0（中国与江苏） | Zhang et al. 2020（全国 17 一级辐射站 + BSRN）；张敏等 2024（江苏南京/淮安/吕泗，CMA-WSP2.0） | 中国基准：GFS 整体弱于持久性、Gradient Boosting 混合最优；江苏三个标准辐射站存在官方检验先例 |
| ECMWF IFS / ENS | Pereira 2019（ANN 订正）；Massidda & Marrocu 2018；Mayer & Yang 2025（集合） | 高分辨率确定性 + 集合 |
| HARMONIE-AROME + CAMS | Bakker 2019 | 高分辨率区域模式 + 气溶胶成分模式组合 |
| 中国本地化模式 | 宁夏 WRF；FY-4B 云订正（2026 报告） | 中国本地实践路径 |

**对本课题的启示**：真值首选卫星 + 观测双源交叉（Mayer & Yang 2024 的做法）；预报用 GFS 档案起步与多数文献一致；若引入站点适应思想，卫星真值质量可进一步提升。

---

## 4. 误差诊断研究（先搞清楚"错在哪"）

这部分与课题任务书第 3 条高度重合，已有明确的文献证据支持"先诊断、按条件分别订正"：

| 文献 | 数据/场景 | 主要发现 |
|---|---|---|
| Dou et al. 2024（J. Renew. Sustain. Energy, ✓） | 公开数据 + 实际电站，日前 GHI | 阴天误差最大、多云次之、晴天最小；阴天/多云系统性**高估**；冬季整体误差最小；误差随时段与预报量级变化 → 建议按天气与季节分别设计订正 |
| Bai et al. 2024（Energy 297:131187, ✓） | 欧洲/北美多站点 NWP 辐照度 | 误差存在系统/随机结构；低云量误差与辐照度误差负相关、气温误差正相关 → 云量与温度为关键协变量 |
| Mabasa et al. 2025（Wea. Forecasting 40(7):1047–1064, ✓） | GFS GHI，南非 6 气候区 15 站 | 阴天系统性高估、晴/多云低估；误差随云量分数增大；漏报阴天日比例随时效显著上升；夏季最大 → 与本课题"多云下 bias 反而更大"的实测一致 |
| Mendes et al. 2024（JAMC 63(2):227–244, ✓） | 高分辨率模式地表短波辐射，南非 | 云量预报误差与辐射误差存在"不平衡"——云量误差被辐射通量放大，云量订正收益直接传导至辐射 |
| 乌拉圭 Bonete（Atmosphere 16(1):35–58, 2025, ~作者） | 模式辐照度 + 云量，亚热带湿润气候 | 云量低估导致 GHI 高估 10% 以上；晴空指数预报偏差存在日变化与季节变化 |
| Hațegan et al. 2023（Energies 16:2919, ✓） | 罗马尼亚 GFS 逐时辐照度 | 最大误差来自**云量低估**；校准后 nRMSE 降低约 16% |
| 云同化研究（NCAR/OSTI 2012, ~） | WRF 云同化 | NWP 普遍高估辐照 = 系统性低估云量；多云地区尤为突出 |
| 干旱气象 2021（中国, ~） | 基于云量的短时辐射预报订正 | 误差主要来自相位偏差与系统偏差；考虑云量的订正效果显著 |
| Lipponen et al. 2026（EGUsphere 预印本, ✓） | CAMS 辐射产品 + 地面观测 | XGBoost 情境依赖误差模型 + SHAP 归因：云光学厚度为第一误差驱动因子；阴天个例可归因至云/气溶胶/地表过程；订正后 GHI 中位偏差 5.0→−0.6 W/m² |

**对本课题的启示**：任务书要求的"按季节、时段、晴/多云/阴分组分析 + 云量误差与辐照度误差相关性"正是这一研究流的标配，且"云量低估→辐照高估"是跨文献一致的强规律，可作为我们误差分析的先验假设。

---

## 5. 订正方法谱系（核心）

### 5.1 简单偏差订正与线性 MOS

| 文献 | 方法 | 数据 | 关键结果 |
|---|---|---|---|
| Verzijlbergh et al. 2015（Solar Energy 118:634-645, ✓） | 逐步线性回归 MOS；先订正 CSI 再还原辐照度；基准方法=太阳天顶角+预报 CSI | GFS，荷兰 | 确立"以 CSI 为中间量"的 MOS 范式 |
| Pierro et al. 2015（Solar Energy 117:99-113, ✓） | 两个 MOS 级联（MOSRH→MOSNN，MOS cascade） | WRF/GFS，意大利 | 级联订正优于单级；测试 D+1/D+2 |
| 宁夏 EOF-MOS（2013, ~） | EOF 分解 + MOS 订正 WRF 辐射 | 本地 WRF + 电站功率 | 辐照度 MAPE 约 24%→15%，功率 MAPE 稳定约 22% |
| 华中师范大学 WRF 短波辐射统计订正（~） | MOS，辐射+温度+水汽组合 | WRF | MAPE 38.3%→22.0% |

### 5.2 概率订正（分位数回归 / 模拟集合 / 集合校准）

| 文献 | 方法 | 数据 | 关键结果 |
|---|---|---|---|
| Bakker et al. 2019（Solar Energy，arXiv:1904.07192, ✓） | 7 种方法对比：gamma/截断正态、QR、QRF、GBDT、广义随机森林、QR 神经网络 | HARMONIE-AROME + CAMS，荷兰 30 站 | QR 与广义随机森林总体最优；近晴空条件下非参数方法更优 |
| Verbois et al. 2018（Solar Energy, ✓） | 分位数梯度提升（QGB），基准= Lasso MOS、AnEn | ECMWF 类 NWP，新加坡全年 | QGB 确定性+概率双优 |
| Verbois et al. 2018（Solar Energy, ~） | 4 个 NWP + PCA + 逐步变量选择的多元后处理 | 新加坡 | 多元统计学习显著优于原始预报 |
| Alessandrini et al. 2015（Applied Energy 157, ✓） | 模拟集合 AnEn | 欧洲光伏电站 | AnEn 概率功率预测的经典 |
| NCQRNN（Adv. Atmos. Sci. 2024, ✓） | 非交叉分位数回归神经网络 | 集合 NWP | 用于集合辐照度校准，锐度-可靠性均衡 |
| AAS 2025（Adv. Atmos. Sci. 42:297-312, ✓） | 模型链概率后处理（统计+ML） | GHI 与功率集合 | 强调"模型链"整体后处理 |
| Terrén-Serrano et al. 2026（Nature Communications, ✓） | HRRR + 稀疏特征选择（Lasso/OMP/EN/GL）+ 多任务 GPR（SLGPR/NLGPR） | CAISO 三节点日前 | 多元 proper score（ES/VS0.5/IS）选模；较基准最高 skill +25%；模型链生成时间一致场景 |

### 5.3 树模型（GBDT / LightGBM / XGBoost）

| 文献 | 方法 | 场景 | 关键结果 |
|---|---|---|---|
| Hațegan et al. 2023 | RF/SVR 校准 GFS | 罗马尼亚 | 云量低估是主误差源；校准显著降 nRMSE |
| Zhang et al. 2020（arXiv:2007.01639, ✓） | Gradient Boosting 混合订正 GFS | 中国 17 一级辐射站 + 1 BSRN | GFS 日前辐照度整体弱于持久性；GB 混合最优 |
| Verbois et al. 2018（QGB） | 分位数 GBDT | 新加坡 | 树模型在表格气象特征上竞争力强 |
| IEEE 2026（XGBoost/LightGBM, ~） | 时间特征工程 + XGB/LGBM | 历史辐照度序列 | 两模型在短期辐照度预测均有效 |
| SHAP 可解释性研究（2026, ~） | Optuna-LightGBM + SHAP/PFI/PDP/ICE | 智能电网场景 | 强调树模型+季节可解释性分析 |
| GEFCom2014 树集成方案（IJF 2016, ~） | 广义可加树集成 | 竞赛太阳能轨道 | 获得第二名，验证树模型在概率太阳预测中的竞争力 |

### 5.4 深度学习（CNN / TSMixer / Transformer / 扩散模型）

| 文献 | 方法 | 数据 | 关键结果 |
|---|---|---|---|
| Pereira et al. 2019（Solar Energy 185:387-405, ✓） | ANN 订正算法 | ECMWF GHI，葡萄牙 | 早期 ANN 订正代表作 |
| Lauret et al. 2014（Energy Procedia 57:1044-1052, ✓） | ANN 订正 WRF 日前 GHI，输入含太阳天顶角与晴空指数 | 留尼汪，1 h 分辨率 | DL 订正最早基准之一，验证太阳几何 + 晴空归一化的标准输入 |
| Savchenko et al. 2025（Przegląd Elektrotechniczny, DOI:10.15199/48.2025.04.09, ✓） | SolarM2P：map-to-point 深度 CNN | GFS 2015–2023，乌克兰东北 | rRMSE 40.4%→33.8%，还能提升时间分辨率 |
| Phan, Wu & Phan 2025（IEEE/IAS 61st I&CPS, DOI:10.1109/ICPS64254.2025.11030368, ✓） | TSMixer 时序模型订正 WRF-Solar | WRF-Solar + 日前功率 | 首次将 TSMixer 用于辐照度偏差订正，并与 Transformer/Informer 对照；轻量序列模型即可胜任 |
| Bire, Lu & Ramezani 2026（Energy and AI, ✓） | BRTCN：有界残差 TCN，跨站点单模型 | 多站点日前 NWP GHI | regime 分组仅用于诊断、单一映射建模；有界残差提升物理合理性 → 直接支持 v1（统一）优于 v2（分层） |
| Dou, Wang & Shan 2025（Applied Energy 397:126295, ✓） | CoST 季节-趋势解耦 + MoE 稀疏激活 + encoder-decoder | 日前 NWP GHI | 建模前先统计 GHI 误差特征；MoE 按季节自适应激活特征，优于固定分组 → 软门控替代硬切桶 |
| Jiang & Peng 2026（Sci. Rep. 16, DOI:10.1038/s41598-026-46558-y, ✓） | 物理约束多模态视觉 Transformer | 卫星多通道 + 辐照度 | 辐射传输方程作为损失软约束；RMSE 较最优基线降 18.7%，系统性偏差 12.7→1.2 W/m²；阴天误差最小（67.4 W/m²）、破碎云最大（103.8 W/m²） |
| Li, Wu, Chen & Zhang 2026（Renewable Energy 256:124152, ✓） | RTI-Net：Beer–Lambert 辐射传输指数（RTI）中间量 | 天空图像 + 功率 | 物理中间量由网络端到端学习，而非两阶段硬传递 |
| Chen et al. 2026（Advances in Applied Energy, ~作者） | 2D-TFT：TFT 融合 3 km–1 h NWP + 卫星 | 日内辐照度 | NWP 辐照度特征重要性最高（达 22.07%） |
| Zidoum & Ibrahim 2026（Energy Informatics, DOI:10.1186/s42162-026-00672-3, ✓） | TFT vs LSTM 跨地点长周期 | 干旱环境多站 | LSTM 单站迁移严重退化；TFT 多站合并训练在特定站点 RMSE 最多降 57% |
| 全卷积网络重建（Solar Energy, 2025, ~作者） | FCN 插值 NWP 辐照度场 | 集合 NWP 逐时场 | 场级后处理可同时提升时空一致性 |
| Carpentieri et al. 2025（SHADECast, Applied Energy 377:124186, ✓） | 扩散生成模型（概率时空辐照度 nowcast） | 区域级卫星/模式场 | 深生成模型延长日内概率预报有效时效 |
| Rastgoo et al. 2025（IEEE Access, DOI:10.1109/access.2025.3600713, ✓） | 扩散概率模型 + 天空图像序列 | 超短期功率 | 云况快速变化场景下的概率预测 |
| XGBoost-LSTM 混合（Energy Reports, 2026, ~作者） | 物理特征（晴空指数、归一化 GHI 等）+ XGBoost–LSTM + 集成不确定性 | 光伏功率 | 物理特征与混合架构的近期实例 |
| Sensors 2026（Attention-CNN-LSTM + 空间降尺度, ~） | 空间降尺度 + 注意力 CNN-LSTM | 区域 NWP + 电站 | 空间细化的 NWP 输入改善日前预测，GHI/散射误差降约 40–55% |
| 可再生能源 2025（多源数据 + Transformer, ~） | LSTM + 跨模态注意力融合 | 多源气象 + 天空图像 | 超短期辐照度预测，跨模态融合 |
| Zhang & Xue 2026（Deep Fusion of Clear-Sky Physics and a Multimodal Transformer, IEEE, ~卷期待核） | 晴空物理模型 + 云图 + 气象要素多模态 Transformer | 辐照度时序 | 晴空物理约束深度嵌入网络，预测不超晴空上限 |
| 中国计量大学学报 2025（生成式扩散模型, ✓） | 扩散概率模型 + 改进 Transformer | 历史辐照度序列 | 短期多步预测，扩散路线进入国内期刊 |
| PV-MM-diffusion（Applied Energy 2026, ~） | 多模态扩散 U-Net（天空图像+功率联合预测） | 天空相机 + 功率 | 端到端超短期概率预测 |
| GAT-Adapt-Hybrid DDPM（IEEE 2026, ~） | 图注意力 + 扩散概率模型 | 太阳能/风能集合 | 概率预测新方向 |

### 5.5 空间订正与站点适应

- **map-to-point**：SolarM2P 用网格预报场直接映射到站点（Savchenko 2025）。
- **空间网格误差订正**：深度多变量空间注意力 CNN 订正 NWP 网格误差再用于日前功率（IEEE, ~）。
- **空间降尺度**：CNN-LSTM + 空间降尺度把粗网格 NWP 细化（Sensors 2026）。
- **站点适应**：核密度映射（Vernay et al., ✓）、Himawari-8 Heliosat+REST2 优化（Renewable Energy 2022, ~）、Solargis 工业流程——用高质量地面观测把卫星/模型辐射"锚定"到本地。

### 5.6 云量订正与云资料同化（与课题"云场"直接相关）

- 云量低估→辐照高估：跨文献一致（Hațegan 2023；NCAR 云同化 2012）。
- 基于云量的辐射订正：中国《干旱气象》2021 年工作表明云量订正可显著改善辐射预报。
- 云量直接订正：Deo et al. 2023（Renewable Energy 203:113-130, ✓）用核岭回归（KRR）订正 GFS 总云量（TCDC），2–8 天提前期效果显著，云量订正可独立成环。
- 云同化进 WRF：WRF-Solar + MADCast 卫星初始化（NCAR, ~）；FY-4B AGRI 云敏感通道亮温作为云订正因子进入自适应变分偏差订正（2026 会议报告, ~）。
- 中国场景：Himawari-8 AOD 同化进 WRF-Chem-Solar 改善晴空功率预测（Wang et al., ~）。

---

## 6. 链条位置研究：到底该订正哪一步（最重要的反面证据）

> 适用范围说明：本节讨论的是"辐照度→功率"文献链条。当前项目交付的是
> **两个独立气象产品**（GHI 条件分位数、总云量条件分位数），不构建功率链条；
> 本节结论只用于说明交付物定位与下游适用性边界。

| 文献 | 发现 | 对本课题的意义 |
|---|---|---|
| Mayer & Yang 2024（Applied Energy 371:123681, ✓） | 在匈牙利 3 个电站、4 年数据上对比 4 条工作流（MOS/KCDE 在不同阶段后处理）：**只要最终功率做了后处理，GHI 订正的额外收益甚微** | 直接挑战"先订正辐照→再转功率"的默认链条。我们作为气象环节子模块，必须明确交付物定位是"可复用的订正气象数据集"，而非仅服务于单一电站功率指标 |
| Mayer & Yang 2025（~） | 集合 NWP 较确定性 NWP 误差降低约 5%；整个流程中**唯一不可省的后处理是对最终功率的偏差订正** | 提示：订正收益要在"气象端"和"功率端"之间清晰归因 |
| Nguyen & Müsgens 2022（Applied Energy 323:119603, ✓） | 统计 180 篇光伏预测文献、1136 条误差观测：**引入 NWP 变量、数据归一化、重采样**是提升精度的最有效数据处理手段 | 支持本课题把"特征工程+NWP 输入"作为核心杠杆 |
| Danner & de Meer 2026（arXiv:2603.04132, ✓） | 两阶段分解：天气预报与电站特性分开建模、误差按来源分解 | 支持"气象订正作为独立可复用模块"的交付定位（本课题产出订正气象特征数据集） |
| Horat et al. 2025（同 24, AAS 42:297-312, ✓） | 辐照度级后处理对最终功率影响甚微，功率级后处理收益最大；直接 NN 与最优后处理相当 | 气象端收益需单独报告；本课题交付订正气象数据集应以气象端指标为主 |
| Terrén-Serrano et al. 2026（Nature Communications, ✓） | 系统级联合概率预报可直接量化运营储备需求 | 概率化场景是"气象→市场"链条的自然延伸 |

---

## 7. 评估与基准（怎么才算"订正有效"）

| 文献 | 贡献 |
|---|---|
| Yang et al. 2020（"Verification of deterministic solar forecasts", ✓） | 提出确定性太阳预报验证框架；建议**普遍报告 RMSE skill score**，基准取气候态与持久性的最优凸组合 |
| GEFCom2014（Hong et al. 2016, IJF, ✓） | 概率能量预测竞赛（负荷/电价/风电/太阳能四轨道），pinball loss 计分；是"业界对标"的现成基准平台 |
| Bakker et al. 2019 | CRPS skill score、Brier skill score、可靠性图等概率验证指标 |
| Dou et al. 2024 | 按天气/季节/时段分层评估的示范 |
| Mayer & Yang 2024 | 强调遵循 best-practice 验证流程，避免指标滥用 |
| Mayer et al. 2026（Solar Energy, ✓） | 7 种概率后处理方法系统对照：分位回归神经网络最优，相对原始预报 CRPS 改善 11.1%–14.7% |
| IEA PVPS T16 概率基准（Renewable Energy 2024, ~） | 国际概率辐照度预报基准：多源点预报最优混合 + 统计后处理生成分位数 |
| 误差分解形式化（IET RPG 2026, DOI:10.1049/rpg2.70217, ✓） | ε_total = ε_sys + ε_ran；按晴/非晴条件估计协方差 → 支撑系统/随机偏差分层与区间输出 |
| Nguyen & Müsgens 2026（JRSE 18(2):026102, DOI:10.1063/5.0300682, ✓） | 基于 skill score 的太阳预报 meta 分析，统一评价口径 |

常用指标速查：MAE、RMSE、MBE/Bias、R²、nRMSE/nMAE、MAPE、CRPS、pinball loss、skill score（相对基准的改善率）。

---

## 8. 对本项目的直接启示（浓缩）

1. **先诊断后建模**：Dou et al. 2024 明确建议按天气、季节分别设计订正方案——任务书的分析顺序正确且有据可依。
2. **以 CSI/Kt 为中间量**：Verzijlbergh 2015、Lorenz 2009、Bakker 2019 一致采用"订正晴空指数再还原辐照度"，支持我们的 Kt 特征设计。
3. **云量误差是主误差源**：跨文献一致（云量低估→辐照高估），我们的"云量误差-辐照度误差"相关分析有明确的先验支持。
4. **模型阶梯合理**：Bias → Linear MOS → Ridge → LightGBM/XGBoost 与文献主流的"简单基线→线性→树模型"完全对应；树模型在表格气象特征上通常是最强基线（QGB、RF、LightGBM）。
5. **真值用卫星+地面双源**：Mayer & Yang 2024 提供了"地面与卫星双真值"的成熟做法；可再叠加站点适应思想。
6. **分时效评估是标配**：Pierro 2015 即按 D+1/D+2 评估；任务书 D+1/D+2/D+3 分开报告符合惯例。
7. **评估要加 skill score**：除 MAE/RMSE/Bias/R² 外，建议按 Yang 2020 补 RMSE skill score（基准=气候态/持久性），这是"业界对标"的公平口径。
8. **警惕链条位置质疑**：Mayer & Yang 2024/2025 提示辐照订正收益可能在功率端被稀释；本项目交付"订正气象特征数据集"时应在报告中明确气象端收益（如 MAE/RMSE 改善率）与下游适用性边界。
9. **深度学习方法可选作上限对照**：TSMixer、map-to-point CNN、Transformer、扩散模型代表当前前沿，可作为 LightGBM 之外的进阶实验（若数据量允许）。
10. **中国场景的空白点**：公开的"中国区域 NWP 辐照度订正"文献多为宁夏/华中 WRF 实践，江苏地区未见系统性的 D+1/D+2/D+3 辐照-云场联合订正数据集研究，创新空间明确。
11. **深度学习不等于更强**：同数据对照下 XGBoost 仍是强基线；DL 的可验证增益集中在序列上下文、跨站点共享表示与概率输出（Bire 2026；Zidoum 2026；Mayer 2026），深度模型的预期收益应定位在这三点而非单纯增大容量。
12. **分层诊断、统一建模**：BRTCN 以 regime 分组做诊断、单一映射建模，MoE 以软门控替代硬切桶（Dou 2025）。
    （历史编号：旧体系 v1 统一 > v2 分层；当前体系不再有"统一 vs 分层"对照，现协议为多站合并 + 站点 one-hot 条件特征。）
13. **物理约束与 kt 的正确用法**：输出做残差/有界化（0 ≤ 订正 GHI ≤ 晴空上限），物理约束以损失软项进入网络（Jiang 2026；BRTCN）；kt 作为端到端中间监督而非两阶段硬传递。
    （历史编号：旧体系 v3 阶段一精度不足的教训；当前 v15_pinn 采用非负/晴空上限/晴空指数一致性约束。）
14. **多站点合并训练 + 站点条件特征**优于分站建模（Zidoum 2026），站点不做硬分层以保证泛化能力。
15. **阴天以系统性偏差为主、逐时形状不可约**：Mabasa 2025 实测 GFS 阴天高估且漏报阴天日随时效上升；Jiang & Peng 2026 表明阴天在“输入含云结构”时反而是误差最小场景——D+2/D+3 缺的正是云结构信息。对策是 kt 中间量 + 物理有界输出 + 概率区间，而非加大容量拟合逐时形状。
16. **情境依赖误差建模与归因**（Lipponen 2026）：以预报输入直接预测误差量级（XGBoost）+ SHAP 归因，确认云光学厚度为第一驱动因子；阴天偏差可结构化为云况、时效、季节的函数并直接扣除。
17. **概率/场景化交付**：Terrén-Serrano 2026 与 SHADECast 表明联合分布与时间一致场景可直接服务储备/调度决策；对应本项目分位数族 + conformal 校准 + mean_pinball/覆盖率/宽度验证路线。
18. **空间上下文特征（轻量版）**：Zhang et al. 2026（IEEE TSG）用全球 ERA5 场证明遥相关存在；本项目可先用周边 3×3/5×5 格点 GFS 云量与辐射做低成本空间上下文，暂不需要全局场。
19. **多站合并 + 站点静态特征**：Zidoum 2026 证实合并多站训练优于单站，站点条件编码方式（静态协变量）决定泛化。
20. **评估口径**：阴天绝对误差天然偏小，需以 rMAE/rRMSE、kt 尺度、skill score（Yang 2020；meta 分析 2026）与 CRPS 分况报告。

### 8.1 AutoPV（Applied Energy 2026）精读记录——流程/评估范式借鉴

AutoPV（Chen et al., Applied Energy 415:127851）本体是 NAS 自动设计光伏功率预测模型，但其
**实验流程与本项目自查清单直接对应**（缺陷 1–7 的当前可追溯版本见
`docs/06_protocol_changelog.md` §2）：
- 数据：单站 1 分钟原始记录，先按日执行“缺失/异常占比过高整日删除 + 夜间置零 + 邻域插补”，
  再降采样逐时，最终报告清洗后样本量（369 天 / 8856 样本）——对应缺陷 1“数据完整性门槛”；
- 切分：纯时序 6:2:2（train/val/test 不交叉、测试只用于最终报告）——对应缺陷 3/7；
- 重复：每个模型×任务重复 5 次并报 mean±SD——对应缺陷 2；
- 评估：MAE/RMSE/R² 三指标并列，并解释 L1 搜索目标与 RMSE/R² 排名差异——对应缺陷 6；
- 图件：Fig.4 预测可视化；Fig.5 三档天气辐照度叠合与平均廓线；Fig.6 分天气总 MAE；
  Fig.7 分天气×预报时步 MAE 廓线（Fig.6/7 为本项目 fig_weather_profiles 的直接模板）；
- 未来天气输入按“随时效不确定性增加”建模（加入相应噪声层/场景），而非当确定值——
  对应缺陷 5，是本项目概率版升级的参考点。

全文 DOI：10.1016/j.apenergy.2026.127851（本地 literature/01_1-s2.0-S0306261926005039-main.pdf）。

---

## 9. 参考文献

> 引用标注：`✓` 已核验原文或官方摘要页；`~` 依据检索摘要整理，投稿前须逐条复核
> （共 29 条，清单与状态见 `docs/06_protocol_changelog.md` §4）。

### A. 综述与全景

1. Sobri, S., Koohi-Kamali, S., Rahim, N.A. (2018). Solar photovoltaic generation forecasting methods: A review. Energy Conversion and Management, 156, 459–497. ✓
2. Antonanzas, J., et al. (2016). Review of photovoltaic power forecasting. Solar Energy, 136, 78–111. ✓
3. Inman, R.H., Pedro, H.T.C., Coimbra, C.F.M. (2013). Solar forecasting methods for renewable energy integration. Progress in Energy and Combustion Science, 39(6), 535–576. ✓
4. Yang, D., Kleissl, J., Gueymard, C.A., Pedro, H.T.C., Coimbra, C.F.M. (2018). History and trends in solar irradiance and PV power forecasting: A preliminary assessment and review using text mining. Solar Energy, 168, 60–101. ✓
5. Voyant, C., Notton, G., Kalogirou, S., Nivet, M.-L., Paoli, C., et al. (2017). Machine learning methods for solar radiation forecasting: A review. Renewable Energy, 105, 569–582. ✓
6. Ghodusinejad, M.H., Rashvand, N., Salmanpour, F., Danehkar, S., Yousefi, H. (2026). A systematic review of solar irradiance forecasting across time horizons using physical, satellite, and AI-based methods. Solar Compass, 17, 100154. ✓
7. Nguyen, T.N., Müsgens, F. (2022). What drives the accuracy of PV output forecasts? Applied Energy, 323, 119603. ✓（arXiv:2111.02092）
8. 光伏发电功率预测方法综述. 水电与抽水蓄能, 2024(2). ~
9. 基于深度学习的光伏功率预测方法研究综述. （国内核心期刊，被引 264 次）. ~

### B. 误差诊断

10. Dou, W., Wang, K., Shan, S., Li, C., Wen, J., Zhang, K., Wei, H., Sreeram, V. (2024). Evaluation of performance for day-ahead solar irradiance forecast using numerical weather prediction. Journal of Renewable and Sustainable Energy, 16(4), 043703. ✓
11. Hațegan, S.-M., Stefu, N., Paulescu, M. (2023). Calibration of GFS solar irradiation forecasts: A case study in Romania. Energies, 16(6), 2919. ✓
12. Improved solar power forecasting using cloud assimilation into WRF. (2012). NCAR/OSTI. ~
13. 基于云量的短时太阳辐射预报订正技术. 干旱气象, 2021, 39(6), 1006–1016. ~

### C. MOS 与线性订正

14. Verzijlbergh, R.A., Heijnen, P.W., de Roode, S.R., Los, A., Jonker, H.J.J. (2015). Improved model output statistics of numerical weather prediction based irradiance forecasts for solar power applications. Solar Energy, 118, 634–645. ✓
15. Pierro, M., Bucci, F., Cornaro, C., Maggioni, E., Perotto, A., Pravettoni, M., Spada, F. (2015). Model output statistics cascade to improve day ahead solar irradiance forecast. Solar Energy, 117, 99–113. ✓
16. Lorenz, E., et al. (2009). CSI 误差建模（误差标准差为预报 CSI 与天顶角的函数）. ~（经 Bakker 2019 引述）
17. 宁夏本地化 WRF 辐射预报订正及光伏发电功率预测方法初探. （2013, 会议论文）. ~
18. 关于 WRF 模式模拟到达地表短波辐射的统计订正. 华中师范大学学报（自然科学版）. ~

### D. 概率订正与集合

19. Bakker, K., Whan, K., Knap, W., Schmeits, M. (2019). Comparison of statistical post-processing methods for probabilistic NWP forecasts of solar radiation. Solar Energy.（arXiv:1904.07192）✓
20. Verbois, H., Rusydi, A., Thiery, A. (2018). Probabilistic forecasting of day-ahead solar irradiance using quantile gradient boosting. Solar Energy.（DOI:10.1016/j.solener.2018.07.071）✓
21. Verbois, H., Huva, R., Rusydi, A., Walsh, W. (2018). Solar irradiance forecasting in the tropics using numerical weather prediction and statistical learning. Solar Energy. ~
22. Alessandrini, S., Delle Monache, L., Sperati, S., Cervone, G. (2015). An analog ensemble for short-term probabilistic solar power forecast. Applied Energy, 157.（DOI:10.1016/j.apenergy.2015.08.011）✓
23. Xia, X., et al. (2024). Non-crossing quantile regression neural network as a calibration tool for ensemble weather forecasts. Advances in Atmospheric Sciences.（DOI:10.1007/s00376-023-3184-5）✓
24. Improving model chain approaches for probabilistic solar energy forecasting through post-processing and machine learning. Advances in Atmospheric Sciences, 2025, 42(2), 297–312.（DOI:10.1007/s00376-024-4219-2）✓
25. Combining quantiles of calibrated solar forecasts from ensemble numerical weather prediction. Renewable Energy, 2023. ~
26. Zamo, M., et al. (2014). Quantile regression 与 QRF 对比（Météo-France PEARP）. ~（经 Bakker 2019 引述）

### E. 树模型

27. Hațegan et al. 2023（同 11：RF/SVR 校准 GFS）. ✓
28. Verbois et al. 2018 QGB（同 20：分位数梯度提升）. ✓
29. Solar irradiance forecasting and its impact on active distribution network operation（XGBoost/LightGBM）. IEEE, 2026. ~
30. Seasonal explainability in machine learning-based solar forecasting（Optuna-LightGBM + SHAP/PFI/PDP/ICE）. 2026. ~
31. GEFCom2014: Probabilistic solar and wind power forecasting using a generalized additive tree ensemble approach. International Journal of Forecasting, 2016. ~

### F. 深度学习

32. Pereira, S., Canhoto, P., Salgado, R., Costa, M.J. (2019). Development of an ANN based corrective algorithm of the operational ECMWF global horizontal irradiation forecasts. Solar Energy, 185, 387–405. ✓
33. Savchenko, O., et al. (2025). SolarM2P: map-to-point deep neural network for post-processing of numerical weather prediction-based solar irradiance forecasts. Przegląd Elektrotechniczny.（DOI:10.15199/48.2025.04.09）✓
34. Phan, Q.-T., Wu, Y.-K., Phan, Q.-D. (2025). TSMixer: An innovative model for advanced bias correction of NWP solar irradiance and one-day-ahead power forecasting. 2025 IEEE/IAS 61st I&CPS, 1–6.（DOI:10.1109/ICPS64254.2025.11030368）✓
35. Attention-enhanced CNN-LSTM with spatial downscaling for day-ahead photovoltaic power forecasting. Sensors, 2026, 26(2), 593. ~
36. Zhang & Xue（2026）. Deep fusion of clear-sky physics and a multimodal Transformer for solar irradiance forecasting. IEEE.（作者与卷期待核）~
37. VMD-LSTM-Transformer: A hybrid approach for robust solar irradiance forecasting. 2026. ~
38. 基于多源数据和 Transformer 框架的超短期太阳辐照度预测模型. 可再生能源, 2025(12). ~
39. 基于生成式扩散模型的短期多步太阳辐照度预测. 中国计量大学学报, 2025(3). ✓
40. PV-MM-diffusion: An end-to-end multi-modal diffusion model for ultra-short-term probabilistic photovoltaic forecasting. Applied Energy, 2026. ~
41. Efficient probabilistic forecasting of solar and wind generation using a GAT-Adapt-hybrid diffusion model. IEEE, 2026. ~

### G. 卫星、站点适应与云

42. Vernay, C., et al. Kernel density mapping 站点适应（西欧 GHI）. ~（HAL:hal-03337523）
43. An optimized approach for mapping solar irradiance ... site-adaptation using Himawari-8 satellite imageries. Renewable Energy, 2022. ~
44. Solargis. Solar irradiance site adaptation（工业实践文档）. ✓
45. Jimenez, P.A., et al. (2016). WRF-Solar: Description and clear-sky assessment of an augmented NWP model for solar power prediction. Bulletin of the American Meteorological Society, 97(7). ✓
46. Shen, L., Yao, Y., Cui, Y., et al. (2026). A weather typing adaptive bias correction method for FY-4B satellite-driven solar radiation nowcasting. Solar Energy, 312, 114600.（DOI:10.1016/j.solener.2026.114600）✓
47. Wang, S., Dai, T., Li, C., Cheng, Y., Huang, G., Shi, G. Improving clear-sky solar power prediction over China by assimilating Himawari-8 aerosol optical depth with WRF-Chem-Solar. ~

### H. 链条位置与评估

48. Mayer, M.J., Yang, D. (2024). Optimal place to apply post-processing in the deterministic photovoltaic power forecasting workflow. Applied Energy, 371, 123681. ✓
49. Mayer, M.J., Yang, D. (2025). The complexity and dimensionality of making deterministic photovoltaic power forecasts from ensemble numerical weather prediction. ~（REAL:224374）
50. Yang, D., et al. (2020). Verification of deterministic solar forecasts. Solar Energy. ✓
51. Hong, T., et al. (2016). Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond. International Journal of Forecasting, 32(3), 896–913. ✓

### I. 2026-08-21 补充（多源核验 ✓）

52. Bai, M., Yao, P., Dong, H., Fang, Z., Jin, W., Yang, X., Liu, J., Yu, D. (2024). Spatial-temporal characteristics analysis of solar irradiance forecast errors in Europe and North America. Energy, 297, 131187. ✓
53. Mabasa, B., Langerman, K., Winkler, H. (2025). Evaluating the Global Horizontal Irradiance projected by the Global Forecast System (GFS) model in diverse climatic zones in South Africa. Weather and Forecasting, 40(7), 1047–1064.（DOI:10.1175/WAF-D-24-0074.1）✓
54. Mendes, J., Zwane, N., Mabasa, B., et al. (2024). An analysis of the effects of clouds in high-resolution forecasting of surface shortwave radiation in South Africa. Journal of Applied Meteorology and Climatology, 63(2), 227–244. ✓
55. Preliminary evaluation of a numerical system of prediction for surface solar irradiance and cloudiness in a site with a subtropical humid climate. Atmosphere, 2025, 16(1), 35–58.（作者待核）~
56. Zhang, Y., et al. (2020). Validation of GFS day-ahead solar irradiance forecasts in China. arXiv:2007.01639. ✓
57. 张敏, 袁心仪, 张顾, 王博妮, 孙明, 黄亮, 陈正洪, 葛行成, 周雪城 (2024). CMA-WSP2.0 在江苏地表太阳辐射预报中的检验评估. 气象研究与应用, 2024(1), 17–22. ✓
58. Deo, R.C., Ahmed, A.A.M., Casillas-Pérez, D., et al. (2023). Cloud cover bias correction in numerical weather models for solar energy monitoring and forecasting systems with kernel ridge regression. Renewable Energy, 203, 113–130. ✓
59. Lauret, P., Diagne, M., David, M. (2014). A neural network post-processing approach to improving NWP solar radiation forecasts. Energy Procedia, 57, 1044–1052.（DOI:10.1016/j.egypro.2014.10.089）✓
60. Mayer, M.J., Baran, Á., Lerch, S., Horat, N., Yang, D., Baran, S. (2026). Post-processing of ensemble photovoltaic power forecasts with distributional and quantile regression methods. Solar Energy.（CRPS 改善 11.1%–14.7%）✓
61. Bire, C., Lu, H.Y., Ramezani, F. (2026). Bounded residual Temporal Convolutional Network for multi-site day-ahead numerical weather prediction global horizontal irradiance correction. Energy and AI. ✓
62. Dou, W., Wang, K., Shan, S. (2025). A hybrid correction framework using disentangled seasonal-trend representations and MoE for NWP solar irradiance forecast. Applied Energy, 397, 126295.（DOI:10.1016/j.apenergy.2025.126295）✓
63. Jiang, Z., Peng, W. (2026). Physics-constrained multimodal vision transformer for ultra-short-term solar radiation forecasting error correction. Scientific Reports, 16.（DOI:10.1038/s41598-026-46558-y）✓
64. Li, H., Wu, W., Chen, W., Zhang, M. (2026). RTI-Net: Physics-informed deep learning for photovoltaic power forecasting. Renewable Energy, 256, 124152. ✓
65. Chen, et al. (2026). Interpretable transformer based intra-day solar forecasting with spatiotemporal satellite and numerical weather prediction inputs. Advances in Applied Energy.（作者全表待核）~
66. Zidoum, H., Ibrahim, H. (2026). A cross-location evaluation of temporal fusion transformers and LSTM for long-term solar PV forecasting in arid environments. Energy Informatics.（DOI:10.1186/s42162-026-00672-3）✓
67. A fully convolutional neural network to interpolate solar irradiation NWP ensemble forecasts. Solar Energy, 2025.（作者待核）~
68. A hybrid XGBoost-LSTM model with physics-informed features and uncertainty quantification for solar power forecasting. Energy Reports, 2026.（作者待核）~
69. Improving cross-site generalisability of vision-based solar forecasting models with physics-informed transfer learning. HAL 预印本, 2024. ~
70. Danner, P., de Meer, H. (2026). Two-Stage Photovoltaic Forecasting: Separating Weather Prediction from Plant-Characteristics. arXiv:2603.04132. ✓
71. Day-Ahead PV Power Prediction Intervals Based on Physical Sensitivity Analysis and Robust Error Decomposition. IET Renewable Power Generation, 2026.（DOI:10.1049/rpg2.70217）✓
72. The added value of combining solar irradiance data and forecasts: A probabilistic benchmarking exercise. Renewable Energy, 2024.（IEA PVPS T16 基准）~
73. Xu, L., Mao, Y. (2024). Evaluation of Two Satellite Surface Solar Radiation Products in the Urban Region in Beijing, China. Remote Sensing, 16(11), 2030.（DOI:10.3390/rs16112030）✓
74. Azam, F., de Miranda, D.R., Schroedter-Homscheidt, M., Saboret, L., Saint-Drenan, Y.-M. (2026). CAMS radiation service v4.6 for solar energy: Evaluation of Himawari based surface solar irradiance products. Remote Sensing of Environment.（rMBE −7.4%–4.9%）✓
75. Lu, C., Qin, Y., Luo, S., et al. (2026). RadNet: an interpretable deep learning model for kilometer resolution solar irradiance estimation over China with Fengyun-4A satellite data. npj Clean Energy.（DOI:10.1038/s44406-026-00023-x）✓
76. Solcast（2024）. Latest forecast accuracy validation（135 个全球站点、1–4 年地面数据）官方验证报告. ✓
77. Terrén-Serrano, G., Deshmukh, R., Martínez-Ramón, M. (2026). Probabilistic day-ahead forecasting of system-level renewable energy and electricity demand. Nature Communications, 17, 3307.（DOI:10.1038/s41467-026-69015-w）✓
78. Lipponen, A., Lezaca, J., Saint-Drenan, Y.-M., Schroedter-Homscheidt, M., Arola, A. (2026). Machine-learning-based approach for solar radiation model uncertainty identification, attribution and bias correction. EGUsphere preprint.（DOI:10.5194/egusphere-2026-2743）✓
79. Carpentieri, A., et al. (2025). Extending intraday solar forecast horizons with deep generative models. Applied Energy, 377, 124186.（DOI:10.1016/j.apenergy.2024.124186）✓
80. Rastgoo, R., Amjady, N., Shah, R., Muyeen, S.M. (2025). A diffusion-based probabilistic ultra-short-term solar power prediction using the sky image sequences. IEEE Access.（DOI:10.1109/access.2025.3600713）✓
81. Nguyen, T.N., Müsgens, F. (2026). A meta-analysis of solar forecasting based on skill score. Journal of Renewable and Sustainable Energy, 18(2), 026102.（DOI:10.1063/5.0300682）✓

---

> 备注：第 11 条与第 27 条为同一文献（Hațegan et al. 2023），在方法谱系中重复出现以便按主题检索；标注 ~ 的条目建议在正式引用前用原文复核页码与作者全名。
