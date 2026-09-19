"""LEGACY reproduction only: 5-fold month-balanced quantile CV.

Protocol:
  pool = 2024-02~2025-08; test = 2025-09~2026-08 (untouched).
  month_block_folds: each fold validates 20% of days, 4-day blocks, all 12 months.
  Candidate selection metric: 3-quantile mean pinball over tau={0.1,0.5,0.9}
  （未加权平均 pinball，不是 CRPS）；Ridge 的 alpha 用同一批折上的验证 MAE 选。
Output: reports/02_experiment/cv/month_balanced/{target}/{selection.json,summary.md,results.csv}.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config" / "01_sites.yaml").exists())
os.environ.setdefault("MPLCONFIGDIR", str(CODE_ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(CODE_ROOT / "src" / "s03_models" / "train"))
import train_v1 as tv  # noqa: E402
sys.path.insert(0, str(CODE_ROOT / "src" / "s02_experiment"))  # split_protocol
from split_protocol import split_test, month_block_folds  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cv_month_balanced_quantile")

TAUS = [0.10, 0.50, 0.90]
SMOKE = False
LGB_GRID = [
    dict(learning_rate=0.05, num_leaves=31, max_depth=-1, subsample=0.9, colsample_bytree=0.9),
    dict(learning_rate=0.05, num_leaves=63, max_depth=8, subsample=0.8, colsample_bytree=0.8),
    dict(learning_rate=0.1, num_leaves=31, max_depth=8, subsample=0.9, colsample_bytree=0.8),
    dict(learning_rate=0.1, num_leaves=63, max_depth=-1, subsample=0.8, colsample_bytree=0.9),
    dict(learning_rate=0.05, num_leaves=127, max_depth=12, subsample=0.7, colsample_bytree=0.7),
    dict(learning_rate=0.03, num_leaves=63, max_depth=10, subsample=0.9, colsample_bytree=0.9),
]
XGB_GRID = [
    dict(learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9),
    dict(learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8),
    dict(learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.7),
    dict(learning_rate=0.1, max_depth=8, subsample=0.9, colsample_bytree=0.8),
    dict(learning_rate=0.08, max_depth=7, subsample=0.85, colsample_bytree=0.8),
    dict(learning_rate=0.03, max_depth=5, subsample=0.9, colsample_bytree=0.9),
]
RIDGE_GRID = [1e-5, 1e-3, 1e-1, 1.0, 10.0, 1e3]


def pinball(y, q, tau):
    e = y - q
    return float(np.mean(np.maximum(tau * e, (tau - 1) * e)))


def load(target):
    sites = yaml.safe_load((CODE_ROOT / "config" / "01_sites.yaml").read_text(encoding="utf-8"))["sites"]
    tv.STATION_CODE = {s["id"]: i for i, s in enumerate(sites)}
    obs = "ghi_obs_sat" if target == "ghi" else "cloud_cover_obs"
    frames = []
    for s in sites:
        p = CODE_ROOT / "data" / "03_featured" / f"{s['id']}_featured_2024-02_2026-09.parquet"
        d = pd.read_parquet(p, columns=tv.FEATURES_NUM + ["station_id", "target_time_utc", obs])
        d["target_time_utc"] = pd.to_datetime(d["target_time_utc"], utc=True)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    if target == "ghi":
        d = d.query(tv.DAY_FILTER)
    d = d.dropna(subset=[obs]).copy()
    d["season"] = d["target_time_utc"].apply(tv.season_of)
    return d, obs


def fit_quantile(algo, tau, cfg, Xtr, ytr, Xva, yva, cat_cols=None):
    if algo == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(objective="quantile", alpha=tau, metric="quantile",
                              n_estimators=200 if SMOKE else 3000, random_state=0,
                              verbose=-1, **cfg)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              categorical_feature=cat_cols if cat_cols else "auto",
              callbacks=[lgb.early_stopping(20 if SMOKE else 80, verbose=False)])
        return m.predict(Xva, num_iteration=m.best_iteration_)
    import xgboost as xgb
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=tau,
                         n_estimators=200 if SMOKE else 3000, tree_method="hist",
                         early_stopping_rounds=20 if SMOKE else 80,
                         enable_categorical=bool(cat_cols), random_state=0, **cfg)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m.predict(Xva, iteration_range=(0, m.best_iteration + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["ghi", "cloud"], required=True)
    ap.add_argument("--smoke", action="store_true", help="单折 + 单配置 + τ=0.5，仅验证流程")
    ap.add_argument("--allow-legacy", action="store_true",
                    help="确认仅复现旧结果；正式模型选择必须使用 cv_nested_rolling_quantile.py")
    args = ap.parse_args()
    if not args.allow_legacy:
        raise SystemExit("month-balanced protocol is retired; use cv_nested_rolling_quantile.py")
    global SMOKE, TAUS
    SMOKE = bool(args.smoke)
    if SMOKE:
        TAUS = [0.50]
    from sklearn.impute import SimpleImputer

    d, obs = load(args.target)
    pool, test = split_test(d)
    folds = month_block_folds(pool, n_folds=5, block_days=4, seed=0)
    if SMOKE:
        folds = folds[:1]
    rows = []
    for fi, (tr, va) in enumerate(folds):
        Xtr0, Xva0 = tv.tree_matrix(tr), tv.tree_matrix(va)
        imp = SimpleImputer(strategy="median").fit(Xtr0)
        Xtr = pd.DataFrame(imp.transform(Xtr0), columns=Xtr0.columns)
        Xva = pd.DataFrame(imp.transform(Xva0), columns=Xva0.columns)
        ytr, yva = tr[obs].to_numpy(), va[obs].to_numpy()
        for algo, grid in (("lgbm", LGB_GRID[:1] if SMOKE else LGB_GRID),
                           ("xgb", XGB_GRID[:1] if SMOKE else XGB_GRID)):
            for ci, cfg in enumerate(grid):
                losses = []
                for tau in TAUS:
                    pred = fit_quantile(algo, tau, cfg, Xtr, ytr, Xva, yva)
                    losses.append(pinball(yva, pred, tau))
                rows.append(dict(fold=fi, algo=algo, cfg=ci, mean_pinball3=float(np.mean(losses)),
                                 pinball10=losses[0], pinball50=losses[len(losses) // 2],
                                 pinball90=losses[-1],
                                 n_train=len(tr), n_val=len(va)))
                logger.info("fold=%d %s cfg=%d mean_pinball3=%.4f", fi, algo, ci,
                            rows[-1]["mean_pinball3"])
        # Ridge alpha：同一月分层折上的点预测验证 MAE（训练器用点预测 + 校准折残差分位）
        for ai, alpha in enumerate(RIDGE_GRID[:1] if SMOKE else RIDGE_GRID):
            pipe = tv.lin_pipe(alpha, kind="ridge")
            pipe.fit(tr[tv.FEATURES_NUM + tv.CAT_COLS], ytr)
            pred = pipe.predict(va[tv.FEATURES_NUM + tv.CAT_COLS])
            rows.append(dict(fold=fi, algo="ridge", cfg=ai,
                             mae=float(np.mean(np.abs(yva - pred))),
                             rmse=float(np.sqrt(np.mean((yva - pred) ** 2))),
                             n_train=len(tr), n_val=len(va)))
            logger.info("fold=%d ridge alpha=%.0e mae=%.4f", fi, alpha, rows[-1]["mae"])
    res = pd.DataFrame(rows)
    if SMOKE:
        out = CODE_ROOT / "reports" / "02_experiment" / "cv" / "00_smoke" / args.target
    else:
        out = CODE_ROOT / "reports" / "02_experiment" / "cv" / "month_balanced" / args.target
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "results.csv", index=False)

    # 编码消融：默认配置下 one-hot / native / target encoding 的对照
    enc_rows = []
    for fi, (tr, va) in enumerate(folds):
        ytr, yva = tr[obs].to_numpy(), va[obs].to_numpy()
        maps = tv.target_maps(tr, obs)
        for algo in ("lgbm", "xgb"):
            cfg = (LGB_GRID if algo == "lgbm" else XGB_GRID)[0]
            for enc in ("onehot", "native", "target"):
                if enc == "native" and algo == "xgb":
                    continue  # XGBoost native 需要 category dtype，消融仅对 LightGBM 提供
                Xtr_e, cat_cols, _ = tv.tree_matrix_encoded(tr, enc, maps)
                Xva_e, _, _ = tv.tree_matrix_encoded(va, enc, maps)
                if cat_cols:
                    imp = SimpleImputer(strategy="median").fit(Xtr_e[tv.FEATURES_NUM])
                    Xtr_i = pd.concat([pd.DataFrame(imp.transform(Xtr_e[tv.FEATURES_NUM]),
                                                    columns=tv.FEATURES_NUM),
                                       Xtr_e[cat_cols].reset_index(drop=True)], axis=1)
                    Xva_i = pd.concat([pd.DataFrame(imp.transform(Xva_e[tv.FEATURES_NUM]),
                                                    columns=tv.FEATURES_NUM),
                                       Xva_e[cat_cols].reset_index(drop=True)], axis=1)
                else:
                    imp = SimpleImputer(strategy="median").fit(Xtr_e)
                    Xtr_i = pd.DataFrame(imp.transform(Xtr_e), columns=Xtr_e.columns)
                    Xva_i = pd.DataFrame(imp.transform(Xva_e), columns=Xtr_e.columns)
                losses = [pinball(yva, fit_quantile(algo, t, cfg, Xtr_i, ytr, Xva_i, yva,
                                                    cat_cols), t) for t in TAUS]
                enc_rows.append(dict(fold=fi, algo=algo, encoding=enc,
                                     mean_pinball=float(np.mean(losses))))
                logger.info("encoding fold=%d %s %s mean_pinball=%.4f",
                            fi, algo, enc, enc_rows[-1]["mean_pinball"])
    enc_df = pd.DataFrame(enc_rows)
    enc_df.to_csv(out / "encoding_ablation.csv", index=False)

    lines = [f"# month-balanced 5-fold quantile CV ({args.target})",
             "", "- selection metric: mean pinball over tau=0.1/0.5/0.9", ""]
    lines += ["## 编码消融（默认配置）", "| algo | encoding | mean pinball |", "|---|---|---|"]
    for (algo, enc), g in enc_df.groupby(["algo", "encoding"]):
        lines.append(f"| {algo} | {enc} | {g.mean_pinball.mean():.4f} |")
    lines.append("")
    sel = {}
    for algo in ("lgbm", "xgb"):
        g = res[res.algo == algo]
        agg = g.groupby("cfg").mean_pinball3.agg(["mean", "std"]).sort_values("mean")
        selected = LGB_GRID[int(agg.index[0])] if algo == "lgbm" else XGB_GRID[int(agg.index[0])]
        sel[algo] = selected
        lines += [f"## {algo}", "| cfg | mean pinball (0.1/0.5/0.9) | sd |", "|---|---|---|"]
        for ci, r in agg.iterrows():
            lines.append(f"| {int(ci)} | {r['mean']:.4f} | {r['std']:.4f} |")
        lines.append(f"\n**selected cfg {int(agg.index[0])}**\n")
    gr = res[res.algo == "ridge"]
    aggr = gr.groupby("cfg").mae.agg(["mean", "std"]).sort_values("mean")
    sel["ridge"] = {"alpha": RIDGE_GRID[int(aggr.index[0])]}
    lines += ["## ridge", "| cfg | alpha | mean val MAE | sd |", "|---|---|---|---|"]
    for ci, r in aggr.iterrows():
        lines.append(f"| {int(ci)} | {RIDGE_GRID[int(ci)]:.0e} | {r['mean']:.4f} | {r['std']:.4f} |")
    lines.append(f"\n**selected alpha {RIDGE_GRID[int(aggr.index[0])]:.0e}**\n")
    # 编码选择：相对 one-hot 改善 ≥1% 才切换，否则保持 one-hot
    for algo in ("lgbm", "xgb"):
        g = enc_df[enc_df.algo == algo].groupby("encoding").mean_pinball.mean()
        best = g.idxmin() if len(g) else "onehot"
        base = float(g.get("onehot", g.min()))
        chosen = best if float(g[best]) < base * 0.99 else "onehot"
        sel[f"{algo}_encoding"] = chosen
        lines.append(f"- {algo} encoding: one-hot={base:.4f}, best={best}({float(g[best]):.4f}) "
                     f"-> 采用 **{chosen}**（改善 <1% 时保持 one-hot）")
    sel.update({"n_folds": len(folds), "taus": list(TAUS), "smoke": bool(SMOKE)})
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out / ("selection_smoke.json" if SMOKE else "selection.json")).write_text(
        json.dumps(sel, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("done -> %s", out)


if __name__ == "__main__":
    main()
