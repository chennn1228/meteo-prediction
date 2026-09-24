# 图件目录说明

- `01_data_audit/`：当前正式的原始数据质量审计图。
- `nanjing_15var_diagnostic/`：南京单站 15 变量诊断结果；各模块中的 CSV 是对应图的源数据，JSON/Markdown 是结果清单与口径说明。
- `00_preview/`、`02_site_design/` 至 `06_probability/`：此前阶段生成的全项目预览或阶段性图件，保留用于追溯，不与当前正式审计结论混用。

所有绘图脚本统一只导出两种图像格式：

- `SVG`：可编辑矢量图。
- `PNG`：直接查看的高分辨率位图。

不再生成 PDF 或 TIFF。CSV、JSON、Markdown 不是重复图片，分别承担图源数据、清单和方法口径记录。
