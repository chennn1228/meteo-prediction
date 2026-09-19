# 江苏 GFS–GHI 概率后处理项目

当前协议为 `2.0.0-provisional`。本仓库研究 D+1/D+2/D+3 的 Open-Meteo
位置化 GFS GHI 预报，以及统计、机器学习和深度后处理的增量价值。**目前没有正式结果**：
`project_manifest.yaml` 中的 `official_result_set` 仍为 `null`。既有模型分数、
报告和图片来自旧协议或开发运行，不得用作正式模型排名或论文结论。

## 从哪里开始

1. `project_manifest.yaml`：当前协议、版本、模型、时间切分和结果状态的配置源。
2. `docs/01_research.md`：科学问题与证据边界。
3. `docs/02_project_status.md`：本轮代码状态、验证快照和正式重跑阻断。
4. `docs/04_project_structure.md`：实际目录与旧代码迁移边界。
5. `docs/issues_open.md`：仍须完成的工作。

## 目前可以安全执行

在仓库根目录的 PowerShell 中运行；这些命令**不启动训练，也不生成正式结果**：

```powershell
python -m pytest -q tests
python scripts/01_validate/run.py --mode structural
python scripts/01_validate/run.py --mode official
python scripts/run_stage.py --stage validate
```

`--mode official` 当前应返回 `blocked` 和非零退出码；这表示正式运行门槛在起作用，
不是要求绕过检查。截至 2026-09-20 的本地快照：全量测试 60 项通过（含合成
集成及最小端到端）；结构检查 20/20 通过；正式就绪检查 20/25 通过、5 项阻断。
这不是正式训练或科学结果的验收。

## 正式协议边界

- 正式白天定义为 `solar_elevation > 0`；其他阈值只作敏感性分析。
- 嵌套带隔离滚动验证；三个 expanding inner folds，各自分离拟合、早停和评分。
- 模型选择使用七分位 `mean_pinball`；MAE/RMSE/Bias 仅为 q50/点预测辅助指标。
- 正式结果必须同时满足 `validated` 实现、`official` 执行及正式就绪检查。
- `raw_gfs` 是原始**点**预报，不是凭空构造的概率分布。
- `station_id`、`location_id`、`source_grid_id` 是身份/追踪字段，不能进入模型特征。
- 空间研究对象是 API **实际返回**且通过江苏边界判定的服务点，不是请求坐标。

当前首个外层窗口在既定三内折、隔离和早停设计下不可行；代码会阻断，须先做
逐点、逐 lead 的样本量专项诊断，不能暗中改窗口。一次 0.05° 请求探针得到的
707 个去重返回位置，以及按返回坐标边界筛得的 654 个服务点，都只是**单轮经验枚举**，
不是已收敛、已冻结的江苏 API 服务点全集。现有原始缓存也尚不满足新版 18 变量
和独立小时真值的正式数据门槛。详见 `docs/02_project_status.md`。

`legacy/` 只供历史复现；`archive/` 存放历史文档。网站交接包完全隔离在
`NWP_website_handoff/`，不参与科研执行。`data/`、`secrets/`、大型报告、缓存、
模型权重和网站交接包均不得随科研源码推送到 GitHub。
