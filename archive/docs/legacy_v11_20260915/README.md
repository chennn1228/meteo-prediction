# 辐照度—云量气象预报订正

江苏 20 个分层站点，GFS 预报 D+1/D+2/D+3 的辐照度（真值 Himawari 卫星反演）与
总云量（参考真值为 ERA5 再分析，非独立观测）条件分位数订正；两个产品独立端到端建模。

> **测试期披露（必读）**：当前测试年 2025-09-01~2026-08-31 中，
> **2025-09~2026-01 曾在早期开发中用于模型比较与误差分析**，该时段不是完全未见样本。
> 正式超参数只由月分层五折验证选择，测试年不参与调参；论文必须披露这一历史暴露。
> 完全干净的验证窗口从 2026-09 起冻结，积累满 12 个月后替换当前测试年。

## 阅读顺序

1. `README.md`（本文件）：入口与运行方式；
2. `docs/01_research.md`：研究问题、数据、实验设计、模型族、指标与流程（方法与结论主文档）；
3. `docs/03_hyperparameters.md`：调参协议、搜索空间与校准参数；
4. `docs/02_project_status.md`：进度、产物、服务器执行顺序与已知风险；
5. `docs/04_project_organization.md`：目录编号规范、命名规则、图件索引、绘图风格与脚本清单；
6. `docs/05_review.md` 与 `literature/00_index.md`：文献综述与本地文献库索引；
7. `docs/06_protocol_changelog.md`：协议变更日志与评审处置；
8. `docs/07_issue_register.md`：全项目问题台账（优先级/修复阶段/是否重训/批次）；
9. `docs/08_website_handoff.md`：网站逻辑架构与对外 agent 交接契约。

文档分工：方法与定稿结论写 `01_research.md`；进度与风险写 `02_project_status.md`；
超参写 `03_hyperparameters.md`；目录/图件/脚本规则写 `04_project_organization.md`。
子目录不再放 README 或说明文件。

## 目录结构

| 路径 | 内容 |
|---|---|
| `config/` | `01_sites.yaml`（20 站点与区域）、`02_variables.yaml`（变量与 kt 阈值） |
| `data/01_raw/{01_gfs,02_era5,03_satellite}/` | 分月原始 JSON |
| `data/02_clean`、`data/03_featured` | 清洗表与特征表（正式 tag=`2024-02_2026-09`） |
| `src/s01_data/{fetch,clean,features}` | 数据三阶段 |
| `src/s02_experiment/` | 切分协议与月分层五折调参 |
| `src/s03_models/{train,predict}` | v1_ml 与 v2–v15 训练/预测 |
| `src/s04_evaluation/{analysis,calibration,verification}` | 出图、校准、稳健性 |
| `scripts/{01_fetch,02_clean,03_train,04_evaluate,05_server}` | 运行入口 |
| `reports/{01_data_audit,02_experiment,03_modeling,04_error_analysis,05_robustness,06_probability}` | 结果 |
| `figs/{01_data_audit,02_site_design,03_modeling,04_error_analysis,05_robustness,06_probability}` | 图件 |
| `docs/`、`literature/` | 文档与文献 |
| `deploy/`、`secrets/`、`tests/` | 部署载荷、凭据、测试占位 |

完整目录树与命名规则见 `docs/04_project_organization.md`。

## 环境

- Python 3.12+，虚拟环境 `.venv`（不入版本控制）；
- 安装：`.venv\Scripts\python.exe -m pip install -r requirements.txt`；
- 服务器训练环境见 `scripts/05_server/`，凭据存 `secrets/`（不上传）。

## 常用命令

Windows（PowerShell）：

```powershell
# 数据三阶段（tag=2024-02_2026-09）
.venv\Scripts\python.exe src\s01_data\fetch\fetch_data.py --site all --start 2024-02-01 --end 2026-09-30
.venv\Scripts\python.exe src\s01_data\clean\clean_data.py --site all --start 2024-02-01 --end 2026-09-30
.venv\Scripts\python.exe src\s01_data\features\features.py --site all --start 2024-02-01 --end 2026-09-30

# 数据门禁与审计图
.venv\Scripts\python.exe src\s04_evaluation\analysis\data_availability.py --tag 2024-02_2026-09
.venv\Scripts\python.exe src\s04_evaluation\analysis\figures_data_audit.py
.venv\Scripts\python.exe src\s04_evaluation\analysis\collinearity_ablation.py
.venv\Scripts\python.exe src\s04_evaluation\analysis\plot_site_figs.py

# 月分层五折调参（正式超参来源）
.venv\Scripts\python.exe src\s02_experiment\cv_month_balanced_quantile.py --target ghi
.venv\Scripts\python.exe src\s02_experiment\cv_month_balanced_quantile.py --target cloud

# v1_ml 与深度模型分位数训练
.venv\Scripts\python.exe src\s03_models\train\train_quantile_v1.py --target ghi
.venv\Scripts\python.exe src\s03_models\train\train_deep.py --target ghi --model mlp --quantile --budget full

# 概率图
.venv\Scripts\python.exe src\s04_evaluation\analysis\plot_quantile_results.py
.venv\Scripts\python.exe src\s04_evaluation\analysis\plot_deep_quantile.py
```

Linux / AutoDL 服务器：

```bash
# 数据三阶段（tag=2024-02_2026-09）
.venv/bin/python src/s01_data/fetch/fetch_data.py --site all --start 2024-02-01 --end 2026-09-30
.venv/bin/python src/s01_data/clean/clean_data.py --site all --start 2024-02-01 --end 2026-09-30
.venv/bin/python src/s01_data/features/features.py --site all --start 2024-02-01 --end 2026-09-30

# 月分层五折调参（正式超参来源）
.venv/bin/python src/s02_experiment/cv_month_balanced_quantile.py --target ghi
.venv/bin/python src/s02_experiment/cv_month_balanced_quantile.py --target cloud

# v1_ml 与深度模型分位数训练
.venv/bin/python src/s03_models/train/train_quantile_v1.py --target ghi
.venv/bin/python src/s03_models/train/train_deep.py --target ghi --model mlp --quantile --budget full --device cuda

# 概率图
.venv/bin/python src/s04_evaluation/analysis/plot_quantile_results.py
.venv/bin/python src/s04_evaluation/analysis/plot_deep_quantile.py

# 服务器全流程（含清场、五折 CV、v1、v2–v15、校准、出图）
bash scripts/05_server/run_server_full.sh
```

## 关键约定

- 测试年固定 2025-09-01~2026-08-31；训练池 2024-02-01~2025-08-31；
  验证为训练池内月分层五折，每折约 20%、覆盖 12 个月；
  **测试年含 2025-09~2026-01 的历史暴露，论文必须披露（见顶部警告）**；
- 正式结论只引用 `full` 预算结果，`lite` 仅本地验证；
  **服务器 full 已跑完并回收**（14 深度×{ghi,cloud} + 校准 + 两目标 `selection.json`）；
  v1_ml 已用修复版 featured 本地重跑（逐样本含 station/time 身份列）；
- **✅ 编号（2026-09-13 M1 已完成）**：文档与磁盘**统一新编号**（v3=cnn、v4=tcn、v5=lstm），
  `reports/03_modeling/` 目录与 `train_deep.py` 已重排；旧→新映射留痕见 `docs/04 §0.3`、`docs/01 §4.3`；
- **服务器端点**：当前 `connect.westb.seetacloud.com:21410`（RTX 4090；早期 westc:37257 实例已弃用）；
  调用 `scripts/05_server/remote.py` 前须 `export MSYS_NO_PATHCONV=1` 且远程命令用单引号；
- 树模型类别特征 one-hot（20 站 + 4 季），禁止裸整数编码；
- PCA 先标准化；概率指标中 `mean_pinball` 为未加权平均 pinball，
  `crps_q7_trunc` 为截尾分位网格积分近似，二者均不得简写为 CRPS；
- 旧协议数字不得进入现行结论；
- 每次任务结束同步更新 `docs/02_project_status.md` 与相关文档。
