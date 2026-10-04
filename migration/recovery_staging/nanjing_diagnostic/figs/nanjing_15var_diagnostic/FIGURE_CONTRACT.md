# 南京单站 15 变量诊断图合同

这些图只证明当前口径南京单站的数据与 CPU 实现状态；不代表 20 站、江苏省域或正式论文结论。
所有模型评分图使用原定五个外折，测试期结果另列，不能混为一组。

| 图 | 一句话结论与证据 | 类型与审稿风险 |
|---|---|---|
| `fig_data_source_timeline` | 南京正式窗口三源月缓存覆盖完整，但可用数值存在真实缺口；按月展示三源白天有效比例和缺失小时数。 | quantitative grid；不可把“文件在”写成“数值全有效”。 |
| `fig_cloud_error_ghi_error_hexbin` | 预报云量偏差与 GHI 偏差有可量化联系；各 lead 独立 hexbin。 | quantitative grid；ERA5 云量仅为补充参考，不能称独立云真值。 |
| `fig_feature_distribution_hexbin` | **Provisional/legacy：不得继续作为数据质量证据。** 该版本使用曾被截到 1.5 的 `kt_fcst`；正式证据已迁移到 `figs/01_data_audit/fig_cloud_kt_hexbin` 的未截断 `kt_raw`。 | 旧处理制造边界；仅保留审计溯源。 |
| `fig_value_ladder` | 单站五外折的七个固定与三个调参 CPU 模型呈现 RMSE 分层；每模型总体与逐折评分。 | quantitative grid；五折不是独立站点，不称省域正式阶梯。 |
| `fig_cpu_overall_pinball` | 三个分位数模型在五外折共同样本上的平均 pinball 和原始分位交叉率一并报告。 | quantitative grid；不得把点预测基线伪装为概率模型。 |
| `fig_cpu_lead_performance` | 十个 CPU 模型的 D+1/2/3 RMSE 明确分开；按相同评分样本计算。 | quantitative grid；不同模型必须共用同一目标时间集合。 |
| `fig_cpu_improvement_vs_raw` | 各外折相对 raw GFS 的 RMSE 变化方向可见；逐折差值，不伪造置信区间。 | quantitative grid；五折季节不同，差异不等于空间泛化。 |
| `fig_tuning_candidates` | 六个预注册候选均在同样的三内折中受评；按模型展示内层 pinball 均值与五外折变异。 | quantitative grid；内折分数只用于选参，不能当外折泛化性能。 |
| `fig_quantile_case` | 用按 raw GFS 日 RMSE 中位数确定的测试日示例展示 D+1 真实观测与经因果校准的 XGBoost 七分位范围。 | quantitative grid；案例不是最佳/最差日，不以单日代表总体性能。 |
| `fig_coverage_reliability` | 南京测试期七个分位数的经验累积覆盖率与理想对角线比较，并显式区分校准前后。 | quantitative grid；同目标的三个 lead 相关，不能当独立站点重复。 |
| `fig_coverage_width` | 经因果校准后三个模型的区间覆盖率和平均宽度同时展示，不能只看覆盖率。 | quantitative grid；只反映南京测试年，区间宽度单位 W m^-2。 |
| `fig_weather_conditional_coverage` | 依签发时可用的预报总云量分层检查 80% 区间覆盖率，并按 D+1/2/3 拆分。 | quantitative grid；天气分层不是独立云真值，稀疏单元须标 n。 |
| `fig_shap_beeswarm` | 对第五外折完整拟合、内折轮数已冻结的 LightGBM 中位数模型作开发期树 SHAP 描述，绝不抽取最终测试特征或标签。 | quantitative grid；SHAP 是模型归因不是因果效应或组消融替代品。 |
| `fig_key_feature_hexbin` | 以预注册的 GFS GHI 与总云量展示五个开发期外折残差密度，揭示剩余误差结构。 | quantitative grid；条件相关不是特征增量证明，缺测只在评分中排除。 |
| `fig_group_ablation` | 在五个外折各自的三个开发期内折，以已选 LightGBM 候选固定超参数重训 Full−组，展示七分位 pinball 增量。 | quantitative grid；单站常量空间组不可识别，不声称完成 Base+cloud/kt 所有对照或正式特征选择。 |
| `fig_group_permutation` | 同样在开发期内折，把同组特征在各 lead 内联合置换三次，展示七分位 pinball 增量和外折变异。 | quantitative grid；置换可能形成非物理组合，相关特征会分摊归因，不能用测试集决定特征。 |
| `fig_gap_sensitivity` | 固定外折评分时间，在 7/10/14 天隔离下各自重新完成六候选三内折选参及全量外折重训，比较概率和点预测性能。 | quantitative grid；7 天小于主隔离推导长度，仅为预注册敏感性，不能取代 10 天正式主口径；三种隔离的评分样本必须一致。 |

绘制后端：Python/matplotlib，仅该后端用于渲染与 QA。目标为约 183 mm 双栏宽、可编辑 SVG 与 PNG 预览；白底、统一单一蓝色系与透明度层级、字体约 7 pt。图旁保留源数据 CSV、样本数、公式、缺测处理和版本信息。无图像裁切、局部增强或统计显著性暗示。
