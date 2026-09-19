# 协议到代码迁移审计（2026-09-20）

本报告记录本轮**仓库重构与正式重跑前验证**，不是正式实验报告。没有启动正式全量训练、GPU benchmark、多种子或省域正式评价；没有生成新的 official results。`project_manifest.yaml` 的 `official_result_set` 保持 `null`。网站交接包 `NWP_website_handoff/` 未参与此次科研执行，也不纳入源码推送。

## 1. 施工摘要与真实目录

建立了可导入的 `s01`–`s15` 科研模块、单一协议配置读取层、统一预测/指标合同，以及安全的单阶段入口。旧分析、验证和服务器脚本迁入 `legacy/`；旧文档迁入 `archive/`；但旧数据/训练过渡代码仍存在，不能宣称全仓迁移完成。

```text
project_manifest.yaml / README.md / requirements.txt / .gitignore
config/                    站点与数据变量
docs/                      当前研究、状态、结构、问题文档
src/
  s01_core/                配置、合同、模型注册、溯源
  s02_data/                缓存与清洗合同
  s03_features/            issue-time 特征、折内预处理
  s04_splits/              滚动切分与首外折诊断
  s05_tuning/              候选搜索、runner、trial ledger
  s06_models/              实现级别及模型准入门槛
  s07_prediction/          统一逐样本预测合同和读写
  s08_calibration/         有序校准、按日 block bootstrap
  s09_metrics/             点指标、概率指标、可靠性
  s10_evaluation/          概率主表、点指标辅助表、分组评价
  s11_interpretation/      开发期门槛、原生 SHAP 接口
  s12_spatial/             返回服务点、truth gate、三层空间与密度设计
  s13_visualization/       论文样式、hexbin 高密度关系图
  s14_pipeline/            13 阶段安全编排
  s15_validation/          结构、协议、数据、正式就绪校验
  s01_data/ s02_experiment/ s03_models/ s04_evaluation/  过渡兼容代码
scripts/01_validate/       薄校验入口
scripts/run_stage.py       单阶段入口，非校验阶段默认拒绝自动执行
tests/01_unit/ 02_integration/ 03_e2e/
reports/protocol_migration_audit.md
figs/                     既有旧图，非正式新结果
legacy/                   旧协议可执行材料，非正式入口
archive/                  历史文档与不可执行快照
NWP_website_handoff/      冻结且不推送的网站交接包
```

本地另有不推送的 `data/`、`secrets/`、`deploy/`、`.venv/`、`.cache/` 和大型运行报告。旧 `figs/` 已被既有 Git 历史追踪，但本轮没有新增正式图或改动这些既有图片。

## 2. 旧目录迁移与分类

| 原路径 | 当前处置 | 类别与边界 |
|---|---|---|
| `docs/archive/` | `archive/docs/` | archive；不得反推当前协议 |
| `scripts/03_train/run_deep_quantile.ps1` | `legacy/scripts/03_train/`，入口硬阻断 | legacy |
| `scripts/05_server/*` | `legacy/scripts/05_server/` | legacy；旧总脚本保留阻断 |
| `scripts/06_validate/validate_project.py` | `legacy/scripts/06_validate/`；新入口 `scripts/01_validate/run.py` | legacy → current |
| `src/s02_experiment/cv_month_balanced_quantile.py` | `legacy/s02_experiment/`；显式旧协议选项 | legacy，不能进入正式模型选择 |
| `src/s03_models/train/train_quantile_v1.py` | `legacy/s03_models/train/`；显式旧复现选项 | legacy，不能生成正式结果 |
| `src/s03_models/predict/predict_daily.py` | `archive/code/` | archive；原代码未实现，不能当在线推理入口 |
| `src/s04_evaluation/analysis/*`、`verification/*` | `legacy/s04_evaluation/` | legacy；含旧 SHAP/图/指标脚本 |
| `src/s04_evaluation/calibration/calibrate_quantiles.py` | 原路径仅保留受限兼容接口；正式校准在 `s08_calibration` | 过渡；旧 CLI 硬阻断 |
| `src/s01_data/*`、`s02_experiment/*`、`s03_models/*` 尚存文件 | 逐项迁至 `s02`–`s06`，目前保留兼容/原型 | current-transitional；尚非完全正式执行链 |
| 旧 `figs/`、旧大型 `reports/` | 原地留存，标记非正式 | legacy/provisional；不作为新协议证据 |

删除候选：旧 `split_protocol.py`（完成调用方迁移后）、剩余重复的旧特征/指标/绘图实现。没有为了“整洁”直接删除用户历史数据或旧图。

## 3. 协议 → 配置 → 实现 → 测试 → 输出追踪

表中“接口已实现”表示离线合同/代码可调用，**不等于真实数据和正式科学验收完成**。

| 协议要求 | 唯一配置 | 实现 | 验证/输出 | 当前状态 |
|---|---|---|---|---|
| prototype/validated 与 smoke/development/official 分轴；正式双门槛 | `implementation_levels`、`execution_levels`、`official_gate` | `s01_core/provenance.py`、`s06_models/model_status.py`、预测 schema | 单元及合成 E2E 拒绝原型冒充正式 | 门槛已实现；深度 validated 清单为空 |
| 语义模型名与旧 vN 仅溯源 | `models` | `s01_core/registry.py`、预测 schema | 模型注册和 schema 测试 | 接口已实现 |
| 统一日期、purge、七分位、seed、模型、特征、状态 | manifest；`config/` 仅站点/变量 | `s01_core/config_loader.py` 拒绝重复 YAML key；模块读 manifest | 结构校验与配置一致性测试 | 核心合同已实现；旧过渡脚本仍需消除剩余硬编码 |
| 单一白天口径 | `daylight_definition` | `s03_features/engineering.py` | 单元/结构校验 | `solar_elevation > 0` 为正式口径 |
| 数据 18 变量、3 lead、显式 land、旧缓存失效 | `data_sources`、`config/02_variables.yaml` | `s02_data/cache_contract.py`、旧抓取器适配 | 缓存/月份/时序单元测试 | 合同已实现；真实新版数据未再抓取 |
| 请求与各 API 返回坐标分离 | `spatial_design`、`feature_policy` | `s02_data/clean_contract.py`、`s12_spatial/service_registry.py` | 坐标/边界测试 | 合同已实现；注册表未冻结 |
| API 返回服务点、逐步探针收敛 | `spatial_design` | `s12_spatial/service_registry.py` | 单轮 0.05° 经验枚举 | **未完成收敛** |
| 5⊂10⊂15⊂20、五区域配额、最远点 | `spatial_design.training_site_density` 等 | `s12_spatial/density.py`、`levels.py` | 嵌套/配额/覆盖测试 | 算法已实现；无冻结全集和正式收益 |
| 身份字段与未来真值禁入模型 | `feature_policy`、`feature_groups` | `s01_core/schemas.py`、`s03_features`、训练适配 | 单元/结构检查 | 新接口已阻断；旧过渡入口仍需真实数据审计 |
| issue-time 因果、圆周/比值、lag/rolling、折内预处理 | `feature_policy` | `s03_features/{engineering,preprocessing}.py` | 因果/折内合成测试 | 新接口已实现；全变量实数据审计待做 |
| 云量插补只在当前训练折拟合 | `feature_policy.fit_preprocessing_on` | 旧 `cloud_impute.py` 折内适配 | 单元测试 | 适配已做；正式训练链未全接入 |
| nested rolling、独立 early-stop/score、7/10/14d | `validation_protocol` | `s04_splits/{rolling,diagnostics}.py` | 首外折不可行明确阻断 | **正式选择受阻** |
| 七分位 mean pinball 为主；点指标辅助 | `quantiles`、`selection_metric` | `s05_tuning`、`s09_metrics`、`s10_evaluation` | 指标/分组测试 | 数学接口已实现；无正式预测 |
| 每模型六组不同候选、共同三内折 | `tuning_budget`、`tuning_search_spaces` | `s05_tuning/{search_space,runner,trial_ledger}.py` | 候选唯一性/ledger 测试 | Ridge/树/深度原型已接；全部模型及 validated 深度待核验 |
| 深度标准机制、AutoCorrelation lag、PINN 物理约束 | `models`、`pinn` | 旧架构针对性修复、`s06_models` 禁入 | 专项单元测试 | 全部逐模型 validated **未完成**；PINN 仍实验性 |
| 统一逐样本预测与 raw_gfs 点预报 | `prediction_contract` | `s07_prediction` | schema/读写与合成 E2E | 接口已实现 |
| 后期有序校准、真日历日 block bootstrap | `calibration_method` | `s08_calibration` | 符号/时序/时间块测试 | 数学接口已实现；无正式校准集 |
| 唯一正式指标、截尾 CRPS 准确命名、分组 crossing | `selection_metric`、`prediction_contract` | `s09_metrics`、`s10_evaluation` | 概率/点/分组测试 | 已实现；无正式结果 |
| 开发期机制分析后再 SHAP，样本标签同索引 | `feature_groups.selection_evidence` | `s11_interpretation` | 门槛与索引测试 | SHAP 接口已修；消融/置换/稳定性实证未做 |
| 论文统一样式、高密度 hexbin、原生 SHAP | 绘图合同在 `s13_visualization` | `s13_visualization`、`s11_interpretation/s03_shap` | 样式/hexbin 测试 | 基础接口已实现；正式图尚不存在 |
| Level 1/2/3 与密度空间评价 | `spatial_design` | `s12_spatial/{truth_gate,holdout,levels,density}.py` | 合成点集测试 | 预检接口已实现；省域正式评价未执行 |
| 13 阶段、单阶段、默认不训练 | `official_gate` | `s14_pipeline/orchestrator.py`、薄脚本 | 安全默认测试 | 校验可运行；其余阶段因缺正式数据/审核而 fail-closed |
| 强校验、legacy 隔离、最小端到端 | manifest 与数据/预测合同 | `s15_validation`、`tests/` | 见下一节 | 结构通过；正式就绪受阻 |

## 4. 测试与校验

在仓库根目录使用 `D:\anaconda3\python.exe -m pytest -q tests`：**60 项通过**，覆盖单元、纯合成集成与最小端到端预测读写—校准—评价。无网络、GPU 或正式训练。`scripts/01_validate/run.py --mode structural`：**20/20 通过**。同脚本 `--mode official`：**20/25 通过、5 项阻断**，预期非零退出码。结构校验不等于全数据、全部模型或正式科研就绪。后续应补每模型真实适配的集成证据。

## 5. 江苏服务点与首外折审计

已有 0.05° 探针在江苏边界内发出 3997 个请求，得到 707 个去重 API 返回坐标；其中按**返回坐标**判定有 654 个在江苏边界内、53 个在边界外。这只是一轮经验枚举，**没有**进行更细探针、逐轮新增数量或集合收敛检验，不能声称“江苏最终 654/707 服务点”。注册表状态仍为 `empirical_probe_not_frozen`。请求坐标仅留作日志/溯源。

首外折 2024-02-01 至 2024-05-31 共 121 天。当前三次 30 天评分段、14 天早停、两次 10 天隔离已占 124 天，尚未给拟合段留任何时间。`s04_splits` 和正式校验器明确阻断，不自动缩短窗口。逐服务点/lead 的原始、白天、168h 有效及各阶段行数仍须在新版数据上输出，之后由研究者明确决定修改何项协议。

## 6. 未解决事项与正式重跑前清单

1. 决定首外折可行性修订，并基于真实新版数据输出细粒度样本诊断；修订 manifest 和测试，不在代码中暗改。
2. 重新获取 18 变量全时段原始数据，验证版本侧车、月份、小时、lead、issue/target、缺失率和三类 API 返回坐标。
3. 加密请求探针直到江苏**返回服务点集合**收敛，记录每轮步长/新增/差异/边界规则，冻结注册表；完成全时段小时 Himawari truth gate。
4. 逐模型审查架构机制、单位、张量形状、loss/early stop、参数规模与六候选接口；未通过者维持 prototype 或改为机制启发名称；PINN 未通过物理单位与消融前不得正式比较。
5. 完成旧过渡训练代码退役、全部模型到统一预测合同的真实数据最小流程、开发期特征消融/置换/稳定性、校准与空间分组审计；禁止 final test 用于选择。
6. 补齐真实数据的强数据校验凭据、正式结果 provenance 与可复现环境记录。只有所有正式门槛通过，另行批准正式重跑。此次 `official_result_set` 必须继续为空。

Git 提交前应仅暂存源代码、配置、文档及本审计文件；检查 `git diff --cached --name-only` 与 `git diff --cached --check`，确保不含 `data/`、`secrets/`、网站包、权重、缓存、日志和大型运行结果。
