# 超参数与调优

更新 2026-09-11 ｜ 依据 `docs/01_research.md` §3.3 与 §3.4。

## 1 调参协议

- 切分协议见 `docs/01_research.md` §3.1：测试年固定 2025-09-01~2026-08-31，
  训练池 2024-02-01~2025-08-31，月分层五折每折约 20%、覆盖 12 个月；
- 选择准则：τ=0.10/0.50/0.90 三分位平均 pinball；
- 执行脚本 `src/s02_experiment/cv_month_balanced_quantile.py`，逐目标运行；
- 输出 `reports/02_experiment/cv/month_balanced/{ghi,cloud}/{results.csv,summary.md,selection.json}`；
- 最终训练脚本读取 `selection.json` 应用选中配置；文件缺失时退回默认配置并在日志中说明；
- 测试集不参与任何调参或早停。

## 2 v1_ml 搜索空间

LightGBM 五组候选：

| 编号 | 学习率 | num_leaves | max_depth | subsample | colsample |
|---|---|---|---|---|---|
| 0 | 0.05 | 31 | -1 | 0.9 | 0.9 |
| 1 | 0.05 | 63 | 8 | 0.8 | 0.8 |
| 2 | 0.10 | 31 | 8 | 0.9 | 0.8 |
| 3 | 0.10 | 63 | -1 | 0.8 | 0.9 |
| 4 | 0.05 | 127 | 12 | 0.7 | 0.7 |

XGBoost 五组候选：

| 编号 | 学习率 | max_depth | subsample | colsample |
|---|---|---|---|---|
| 0 | 0.05 | 6 | 0.9 | 0.9 |
| 1 | 0.10 | 6 | 0.8 | 0.8 |
| 2 | 0.05 | 8 | 0.8 | 0.7 |
| 3 | 0.10 | 8 | 0.9 | 0.8 |
| 4 | 0.08 | 7 | 0.85 | 0.8 |

Ridge 正则强度候选：1e-5、1e-3、1e-1、1、10、1e3。α 与 LightGBM/XGBoost 使用
**同一批月分层五折**选择（按折内验证 MAE），结果写入 `selection.json` 的
`ridge.alpha`；`linear` 固定为 OLS。其余点预测家族（raw、bias）不参与调参。

v1_ml 中 LightGBM/XGBoost/ridge 的选择准则分别为：前两者用 τ=0.10/0.50/0.90
三分位平均 pinball，ridge 用点预测验证 MAE（它是点模型 + 校准折残差分位）。

## 3 v2–v15 深度模型配置

- 优化器 Adam，学习率 1e-3，批大小 1024；
- PatchTST 与 TimesNet 正式运行批大小 512（两者在 168 小时输入下激活显存占用高）；
- 服务器队列开跑前对 6 个正式结构各做 1 个 smoke 预检，失败即停止；
- 最大轮数 50，早停耐心 4，损失为七分位 pinball；
- 输入长度：full 预算全部模型统一 168 小时；lite 预算 24 小时（仅本地验证）；
- 早停集为训练子集内部按日期排序的尾部 15%（校准折 0 不参与训练与早停），
  数值特征与目标的均值/标准差只用拟合子集（其余 85%）计算；
- 校准折 0 只用于 conformal 校准；
- 随机种子：lite 为 0，full 为 0–4；
- v1_ml（树与线性族）正式结果固定 seed 0；多种子证据由稳健性检验提供
  （XGBoost/LightGBM 各 5 个种子重训，报告 MAE/RMSE 的 mean±SD）；
- 深度模型不做事先的超参网格搜索：结构、学习率、批大小与轮数为固定配置
  （见 `docs/01_research.md` §4），五折 CV 只用于 v1 树/线性族的选择；
  深度模型的多种子证据在稳健性章节对排名靠前的模型补充；
- 网络结构差异见 `docs/01_research.md` §4.1–§4.16。

## 4 校准参数

- 校准折：月分层五折中的固定一折，最终训练使用其余四折；
- 校准方法：conformal 分位修正 $\delta_\tau = Q_\tau(\hat y - y)$，
  输出分位为 $\hat q_\tau + \delta_\tau$；
- 校准后重新计算 mean_pinball、crps_q7_trunc、覆盖率、区间宽度、分位交叉率与 PIT。

## 5 输出位置

| 产物 | 路径 |
|---|---|
| 调参结果 | `reports/02_experiment/cv/month_balanced/{ghi,cloud}/` |
| 最终模型结果 | `reports/03_modeling/vX_{model}/full/quantile/{ghi,cloud}/` |
| 校准结果 | `reports/03_modeling/vX_{model}/full/quantile/{ghi,cloud}/calibrated/` |
| 日志 | `reports/03_modeling/00_logs/` |
