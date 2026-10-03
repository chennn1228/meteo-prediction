# Phase H evaluation and figures result

Recorded: 2026-10-03 (Asia/Shanghai)

## Evaluation

- Consolidated deterministic, probabilistic, reliability, grouped, spatial,
  and interpretation logic under `src/nwp/evaluation/`.
- Formal evaluation enforces validated official rows and complete registered
  comparison references.  Grouped lead, site, fold, and season tables are
  produced from prediction columns rather than hard-coded experiment facts.
- Development-gated SHAP, group ablation, and group permutation evidence use
  configured feature groups, reference model, sampling rule, and repeat count.
- Spatial density levels use the configured selector and allowed counts; no
  Nanjing-only or fixed-density workflow is embedded in evaluation code.

## Figures

- Consolidated publication style, relationship-density checks, performance,
  reliability, tuning, case, gap, residual, and mechanism plots under
  `src/nwp/visualization/`.
- Figure generation reads only run metrics/analysis artifacts.  `08_figures`
  is restricted to SVG, PNG, and `figure_index.json`; source CSV/JSON tables
  remain in `06_metrics` or `07_analysis`.
- Every indexed figure records its source table, generation function, caption,
  and status.

## Removal evidence

After imports and tests were migrated, active Python searches found no
references to `s07_prediction`, `s09_metrics`, `s10_evaluation`,
`s11_interpretation`, `s12_spatial`, `s13_visualization`, or any of the seven
Nanjing plot scripts.  The 47 tracked Phase H sources were removed only then
and are marked `COMPLETE_SOURCE_MIGRATED_REMOVED` in `migration/migration_map.csv`.
Their pre-refactor versions remain recoverable from tag
`pre-nwp-refactor-20261002`.

Phase H is covered by the 107-test full-suite result recorded in
`phase_g_workflow.md`.  No old figure or report content was deleted in this
phase; classification and active-tree disposition belong to Phase I/K.
