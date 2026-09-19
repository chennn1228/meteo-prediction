# 当前仓库结构与迁移边界

按 2026-09-20 实际文件核对。目录名使用可由 Python 导入的 `sNN_职责`，
不使用数字开头的 Python 包名。下表是当前状态，不是已完成迁移的理想图。

```text
project_manifest.yaml       当前协议配置源
README.md / requirements.txt
config/                     站点和变量数据配置
docs/                       当前文档；旧文档已移入 archive/docs/
src/
  s01_core/                 配置、schema、注册和 provenance
  s02_data/                 缓存与清洗合同
  s03_features/             因果特征与折内预处理
  s04_splits/               滚动切分与首外折诊断
  s05_tuning/               候选空间、runner、trial ledger
  s06_models/               实现级别和模型状态门槛
  s07_prediction/           统一逐样本预测 schema、读写器
  s08_calibration/          时间有序加性分位校准、真实时间块 bootstrap
  s09_metrics/              唯一正式确定性/概率/可靠性指标实现
  s10_evaluation/           概率主表、点预测辅助表、分组评价
  s11_interpretation/       开发期解释门槛、SHAP 接口
  s12_spatial/              返回服务点注册、空间层级与密度预检
  s13_visualization/        论文样式与高密度 hexbin 接口
  s14_pipeline/             安全阶段注册与编排
  s15_validation/           结构及正式就绪检查
  s01_data/                 尚存旧数据实现/兼容代码
  s02_experiment/           尚存旧切分兼容代码；month-balanced 已迁 legacy
  s03_models/               尚存过渡训练与深度架构代码；旧 v1 训练已迁 legacy
  s04_evaluation/           旧校准 CLI 已硬阻断；其余旧代码待退出
scripts/
  01_validate/run.py        薄校验入口
  run_stage.py              默认安全校验；其他阶段不自动执行
tests/01_unit/             单元测试
tests/02_integration/      合成协议接口集成测试
tests/03_e2e/              合成预测读写—校准—评价最小端到端测试
legacy/                    旧服务器脚本、绘图/分析与验证代码，禁止正式 import
archive/                   历史文档和不可执行快照
reports/ / figs/           已有历史/开发产物；非正式结果
data/                      本地科研数据，非 GitHub 源码交付
NWP_website_handoff/       独立网站交接包，不参与科研执行
```

`scripts/` 当前只保留薄执行入口；未完成的正式阶段在 `src/s14_pipeline/`
显式拒绝自动运行。`legacy/` 可做单独标注的历史复现，但当前正式模块不得导入它。
旧 `src/s01_data`、`s02_experiment`、`s03_models` 和 `s04_evaluation` 尚未整体
消失，不能把“新增编号目录”误读为全仓迁移完成。

## 主要旧路径 → 新职责

| 旧位置 | 新位置/处置 |
|---|---|
| `src/s01_data/{fetch,clean,features}` | 数据合同到 `s02_data`；特征规则到 `s03_features`；旧实现暂存待逐项迁移 |
| `src/s02_experiment/*` | 滚动切分到 `s04_splits`、调参到 `s05_tuning`；month-balanced 已迁 `legacy/s02_experiment/` |
| `src/s03_models/train/*` | 正式模型应接入 `s06_models` 及统一预测合同；旧 v1 训练已迁 `legacy/s03_models/`，其余过渡实现不得宣称 validated |
| `src/s04_evaluation/calibration/calibrate_quantiles.py` | 旧命令硬阻断；正式校准在 `s08_calibration` |
| `src/s04_evaluation/analysis/*`、`verification/*` | 已迁到 `legacy/s04_evaluation/`；正式计算/绘图分别由 `s09`–`s13` 承担 |
| `scripts/05_server/*`、旧 PowerShell 深训脚本 | 已迁到 `legacy/scripts/`，不得作为正式入口 |
| `scripts/06_validate/validate_project.py` | 历史副本在 `legacy/scripts/`；当前入口 `scripts/01_validate/run.py` |
| `docs/archive/*` | 历史材料在 `archive/docs/`；当前协议以根 manifest 和当前 `docs/` 为准 |

## 正式数据流与状态

```text
manifest/config → API返回服务坐标与数据合同 → 因果特征
→ 外层/内层滚动与折内预处理 → 六组候选 mean pinball 选择
→ 拟合 → 独立早停 → 后期校准 → 独立测试
→ s07 逐样本预测 → s09 指标 / s10 分组评价
→ 开发期解释、空间三层评价、论文图与报告
```

这条链目前是**待完成的正式执行路径**，不是已跑通的正式结果。正式运行要求
`validated + official`，且先通过正式就绪检查；`official_result_set` 当前为空。
`location_id`、请求坐标和各服务坐标只作身份/溯源，其中 API 返回服务坐标是
空间计算依据，不能把请求坐标或身份编码当模型特征。
