# -*- coding: utf-8 -*-
"""稳健性验证（正式协议）：LOSO、多种子、Diebold-Mariano。

切分：`src/s02_experiment/split_protocol.py` 的月分层折 0（测试年 2025-09~2026-08）；
特征与编码：`src/s03_models/train/train_v1.py` 的 33 数值特征 + 20 站/4 季 one-hot；
DM 检验：读取正式分位数预测 `reports/03_modeling/v1_ml/full/quantile/{ghi,cloud}/`
       的 `per_lead_predictions.csv`，用 q0.5 作为点预测做模型两两比较。
按项目决定，不做双真值检验。

输出：reports/05_robustness/verification_robustness/{summary.md,loso.csv,multiseed_*.csv,dm.csv}
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config" / "01_sites.yaml").exists())
os.environ.setdefault("MPLCONFIGDIR", str(CODE_ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(CODE_ROOT / "src" / "s03_models" / "train"))
sys.path.insert(0, str(CODE_ROOT / "src" / "s02_experiment"))
import train_v1 as tv  # noqa: E402
from split_protocol import default_split  # noqa: E402


def load(target="ghi"):
    sites = yaml.safe_load((CODE_ROOT / "config" / "01_sites.yaml").read_text(encoding="utf-8"))["sites"]
    tv.STATION_CODE = {s["id"]: i for i, s in enumerate(sites)}
    obs = "ghi_obs_sat" if target == "ghi" else "cloud_cover_obs"
    frames = []
    for s in sites:
        p = (CODE_ROOT / "data" / "03_featured"
             / f"{s['id']}_featured_2024-02_2026-09.parquet")
        d = pd.read_parquet(p, columns=tv.FEATURES_NUM + ["station_id", "target_time_utc", obs])
        d["target_time_utc"] = pd.to_datetime(d["target_time_utc"], utc=True)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    if target == "ghi":
        d = d.query(tv.DAY_FILTER)
    d = d.dropna(subset=[obs]).copy()
    d["season"] = d["target_time_utc"].apply(tv.season_of)
    return d.sort_values("target_time_utc").reset_index(drop=True), obs


def fit_xgb(X, y, Xv, yv, seed=0):
    import xgboost as xgb
    m = xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=2000, learning_rate=0.08,
                         max_depth=7, subsample=0.85, colsample_bytree=0.8, tree_method="hist",
                         eval_metric="mae", random_state=seed, early_stopping_rounds=100)
    m.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    return m


def fit_lgb(X, y, Xv, yv, seed=0):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(objective="regression_l1", metric="mae", n_estimators=2000,
                          learning_rate=0.08, num_leaves=63, max_depth=8, subsample=0.85,
                          colsample_bytree=0.8, random_state=seed, verbose=-1)
    m.fit(X, y, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def metrics(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[mask], yhat[mask]
    e = yhat - y
    return dict(n=int(len(y)), mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))),
                bias=float(np.mean(e)))


def dm_test(e1, e2):
    raw = e1 ** 2 - e2 ** 2
    n = len(raw)
    h = int(max(1, min(n - 2, 4 * (n / 100) ** (2 / 9))))
    dm = float(np.mean(raw))
    d = raw - dm
    g0 = np.mean(d * d)
    ac = sum((1 - l / (h + 1)) * np.mean(d[l:] * d[:-l]) for l in range(1, h + 1))
    var = (g0 + 2 * ac) / n
    if var <= 0:
        return np.nan, np.nan
    stat = dm / np.sqrt(var)
    return stat, 2 * (1 - stats.norm.cdf(abs(stat)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["ghi", "cloud"], default="ghi")
    ap.add_argument("--smoke", action="store_true", help="只跑 2 个站、1 个种子，用于验证流程")
    args = ap.parse_args()

    d, obs = load(args.target)
    tr, va, te = default_split(d)
    sites = sorted(d.station_id.unique())
    if args.smoke:
        sites = sites[:2]
    seeds = [0] if args.smoke else [0, 1, 2, 3, 4]

    out = CODE_ROOT / "reports" / "05_robustness" / "verification_robustness"
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"# 稳健性验证（{args.target}）", ""]
    Xtr, Xva, Xte = tv.tree_matrix(tr), tv.tree_matrix(va), tv.tree_matrix(te)
    ytr, yva, yte = tr[obs].to_numpy(), va[obs].to_numpy(), te[obs].to_numpy()
    lines.append(f"- 切分：train={len(tr)} val={len(va)} test={len(te)}（测试年 2025-09~2026-08）")

    # 1) LOSO（留一站外推）
    rows = []
    for sid in sites:
        keep = tr.station_id != sid
        keepv = va.station_id != sid
        m = fit_xgb(tv.tree_matrix(tr[keep]), tr.loc[keep, obs].to_numpy(),
                    tv.tree_matrix(va[keepv]), va.loc[keepv, obs].to_numpy())
        sub = te[te.station_id == sid]
        p = m.predict(tv.tree_matrix(sub))
        rows.append({"station": sid, **metrics(sub[obs].to_numpy(), p)})
    loso = pd.DataFrame(rows)
    loso.to_csv(out / "loso.csv", index=False)
    lines += ["", "## LOSO（留一站外推，XGBoost L1）",
              f"- MAE 均值 {loso.mae.mean():.2f}，SD {loso.mae.std():.2f}，范围 "
              f"[{loso.mae.min():.2f}, {loso.mae.max():.2f}]",
              f"- RMSE 均值 {loso.rmse.mean():.2f}，SD {loso.rmse.std():.2f}"]

    # 2) 多种子
    for name, fitter in (("xgb", fit_xgb), ("lgbm", fit_lgb)):
        rec = []
        for sd in seeds:
            m = fitter(Xtr, ytr, Xva, yva, seed=sd)
            mm = metrics(yte, m.predict(Xte))
            rec.append((mm["mae"], mm["rmse"]))
        a = np.array(rec)
        pd.DataFrame(rec, columns=["mae", "rmse"]).to_csv(out / f"multiseed_{name}.csv", index=False)
        lines += ["", f"## 多种子 {name}（seeds {seeds[0]}–{seeds[-1]}）",
                  f"- MAE {a[:, 0].mean():.2f} ± {a[:, 0].std():.2f}；"
                  f"RMSE {a[:, 1].mean():.2f} ± {a[:, 1].std():.2f}"]

    # 3) DM：正式 v1 分位数预测的 q0.5 两两比较
    pred_path = (CODE_ROOT / "reports" / "03_modeling" / "v1_ml" / "full" / "quantile"
                 / args.target / "per_lead_predictions.csv")
    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        # 每个 model 一块，行序一致（同一测试样本），用位置对齐。
        blocks = {m: g.reset_index(drop=True) for m, g in pred.groupby("model")}
        lens = {m: len(g) for m, g in blocks.items()}
        yv = blocks["raw"]["y"].to_numpy() if "raw" in blocks else None
        dm_rows = []
        if len(set(lens.values())) == 1 and yv is not None:
            for a, b in (("raw", "lgbm"), ("raw", "xgb"), ("lgbm", "xgb")):
                if a not in blocks or b not in blocks:
                    continue
                st, pv = dm_test(blocks[a]["q0.5"].to_numpy() - yv,
                                 blocks[b]["q0.5"].to_numpy() - yv)
                dm_rows.append({"pair": f"{a} vs {b}", "DM": st, "p": pv,
                                "n": int(lens[a])})
        dmd = pd.DataFrame(dm_rows)
        dmd.to_csv(out / "dm.csv", index=False)
        lines += ["", "## DM 检验（q0.5 平方误差差，HAC 方差）"]
        for _, r in dmd.iterrows():
            lines.append(f"- {r['pair']}: DM={r['DM']:.2f}, p={r['p']:.4f}, n={r['n']}")
    else:
        lines += ["", "## DM 检验", f"- 缺少正式预测文件：{pred_path.relative_to(CODE_ROOT)}"]

    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
