"""Legacy v1 quantile reproduction only; not a current formal entry point.

train 2024-02~2025-02; val 2025-03~2025-08; test 2025-09~2026-08.
quantiles: 0.05,0.10,0.25,0.50,0.75,0.90,0.95.
模型族: raw / bias / linear / ridge / lgbm / xgb（每个模型给出完整分位族）。
输出: reports/03_modeling/v1_ml/full/quantile/{target}/{variant}/。
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
from split_protocol import default_split, split_early_stop  # noqa: E402
from cloud_impute import impute_cloud_forecast  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_quantile_v1")

TAUS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
LGB_CFG = dict(learning_rate=0.05, num_leaves=31, max_depth=-1,
               subsample=0.9, colsample_bytree=0.9)
XGB_CFG = dict(learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9)
RIDGE_ALPHA = 1.0
CROSS = {}
SMOKE = False
ENCODING = {"lgbm": "onehot", "xgb": "onehot"}


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
    if target == "cloud":
        d = impute_cloud_forecast(d, feature_cols=tv.FEATURES_NUM)
    d["season"] = d["target_time_utc"].apply(tv.season_of)
    if SMOKE:
        d = d.sample(frac=0.03, random_state=0)
    return d.sort_values("target_time_utc").reset_index(drop=True), obs


def pinball(y, q, tau):
    e = y - q
    return float(np.mean(np.maximum(tau * e, (tau - 1) * e)))


def metrics(y, qmat):
    out = {"n": int(len(y))}
    for i, t in enumerate(TAUS):
        out[f"pinball_{t:g}"] = pinball(y, qmat[:, i], t)
    out["mean_pinball"] = tv.mean_pinball(y, qmat, TAUS)
    out["crps_q7_trunc"] = tv.crps_quantile(y, qmat, TAUS)
    for nom, lo, hi in ((0.5, 0.25, 0.75), (0.8, 0.10, 0.90), (0.9, 0.05, 0.95)):
        out[f"coverage_{int(nom*100)}"] = float(np.mean((y >= qmat[:, TAUS.index(lo)]) &
                                                         (y <= qmat[:, TAUS.index(hi)])))
        out[f"width_{int(nom*100)}"] = float(np.mean(qmat[:, TAUS.index(hi)] -
                                                     qmat[:, TAUS.index(lo)]))
    q50 = qmat[:, TAUS.index(0.50)]
    out["mae"] = float(np.mean(np.abs(y - q50)))
    out["rmse"] = float(np.sqrt(np.mean((y - q50) ** 2)))
    return out


def residual_quantiles(y, base, taus=TAUS):
    r = np.asarray(y) - np.asarray(base)
    return {t: float(np.quantile(r, t)) for t in taus}


def quantile_models(model, tau, Xtr, ytr, Xva, yva, Xte, cat_cols=None):
    n_est = 200 if SMOKE else 3000
    if model == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(objective="quantile", alpha=tau, metric="quantile",
                              n_estimators=n_est, random_state=0, verbose=-1, **LGB_CFG)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              categorical_feature=cat_cols if cat_cols else "auto",
              callbacks=[lgb.early_stopping(20 if SMOKE else 100, verbose=False)])
        return m.predict(Xte, num_iteration=m.best_iteration_)
    import xgboost as xgb
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=tau,
                         n_estimators=n_est, tree_method="hist",
                         early_stopping_rounds=20 if SMOKE else 100,
                         enable_categorical=bool(cat_cols),
                         random_state=0, **XGB_CFG)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m.predict(Xte, iteration_range=(0, m.best_iteration + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["ghi", "cloud"], required=True)
    ap.add_argument("--smoke", action="store_true", help="3% 抽样 + 200 棵树，仅验证流程")
    ap.add_argument("--encoding", choices=["auto", "onehot", "native", "target"], default="auto",
                    help="树模型编码；auto 读 selection.json，缺省 onehot")
    ap.add_argument("--allow-legacy-reproduction", action="store_true",
                    help="明确仅复现旧结果；输出不得进入正式模型比较")
    args = ap.parse_args()
    if not args.allow_legacy_reproduction:
        raise SystemExit(
            "retired v1 trainer: current protocol uses s04_splits + s05_tuning; "
            "pass --allow-legacy-reproduction only for historical reproduction")
    global SMOKE, ENCODING
    SMOKE = bool(args.smoke)
    from sklearn.impute import SimpleImputer
    sel_path = (CODE_ROOT / "reports" / "02_experiment" / "cv" / "nested_rolling"
                / args.target / "selection.json")
    if sel_path.exists():
        sel = json.loads(sel_path.read_text(encoding="utf-8"))
        if (sel.get("smoke") or sel.get("protocol") != "nested_purged_rolling_origin"
                or int(sel.get("outer_folds", 0)) != 5):
            logger.warning("忽略非正式 nested CV 结果（smoke、协议或折数不符）：%s", sel_path)
        else:
            global LGB_CFG, XGB_CFG, RIDGE_ALPHA
            LGB_CFG = sel.get("lgbm", LGB_CFG)
            XGB_CFG = sel.get("xgb", XGB_CFG)
            RIDGE_ALPHA = float(sel.get("ridge", {}).get("alpha", RIDGE_ALPHA))
            ENCODING = {"lgbm": sel.get("lgbm_encoding", "onehot"),
                        "xgb": sel.get("xgb_encoding", "onehot")}
            logger.info("loaded CV selection: %s", sel_path)
    else:
        logger.warning("nested selection 不存在；当前配置只可用于 smoke/开发，不能生成正式结果：%s",
                       sel_path)
    if args.encoding != "auto":
        ENCODING = {"lgbm": args.encoding, "xgb": args.encoding}
    logger.info("tree encoding: %s", ENCODING)
    d, obs = load(args.target)
    if SMOKE:
        out = (CODE_ROOT / "reports" / "03_modeling" / "00_smoke" / "v1_ml"
               / "quantile" / args.target)
    else:
        out = (CODE_ROOT / "reports" / "03_modeling" / "v1_ml" / "full"
               / "quantile" / args.target)
    out.mkdir(parents=True, exist_ok=True)
    for variant in ("lead_feature", "per_lead"):
        if (out / f"{variant}_results.csv").exists():
            res = pd.read_csv(out / f"{variant}_results.csv")
            lines = [f"# v1_ml quantile ({args.target}/{variant})", "",
                     "pinball family, test 2025-09~2026-08", ""]
            for model, g in res.groupby("model"):
                lines += [f"## {model}", "| unit | n | mean_pinball | crps_q7_trunc | crossing | "
                          "coverage_50 | coverage_90 | width_50 | width_90 | mae | rmse |",
                          "|---|---|---|---|---|---|---|---|---|---|---|"]
                for _, r in g.iterrows():
                    lines.append(f"| {r.unit} | {int(r.n)} | {r.mean_pinball:.3f} | "
                                 f"{r.crps_q7_trunc:.3f} | {r.crossing_rate:.4f} | "
                                 f"{r.coverage_50:.3f} | {r.coverage_90:.3f} | "
                                 f"{r.width_50:.2f} | {r.width_90:.2f} | "
                                 f"{r.mae:.2f} | {r.rmse:.2f} |")
            (out / f"{variant}_summary.md").write_text("\n".join(lines), encoding="utf-8")
            logger.info("skip existing %s", variant)
            continue
        units = [("all", d)] if variant == "lead_feature" else [
            (f"D+{L//24}", d[d.lead_time == L]) for L in (24, 48, 72)]
        frames = []
        for label, u in units:
            tr, va, te = default_split(u)
            # 时间顺序固定为 fit -> 10-day purge -> early-stop ->
            # 10-day purge -> calibration -> final test。
            tr_fit, tr_es = split_early_stop(tr)
            maps = tv.target_maps(tr_fit, obs) if "target" in ENCODING.values() else None

            def build_matrices(encoding):
                Xtr_m, cat_cols, _ = tv.tree_matrix_encoded(tr_fit, encoding, maps)
                Xes_m, _, _ = tv.tree_matrix_encoded(tr_es, encoding, maps)
                Xva_m, _, _ = tv.tree_matrix_encoded(va, encoding, maps)
                Xte_m, _, _ = tv.tree_matrix_encoded(te, encoding, maps)
                if cat_cols:
                    imp = SimpleImputer(strategy="median").fit(Xtr_m[tv.FEATURES_NUM])
                    def prep(X):
                        num = pd.DataFrame(imp.transform(X[tv.FEATURES_NUM]),
                                           columns=tv.FEATURES_NUM)
                        return pd.concat([num, X[cat_cols].reset_index(drop=True)], axis=1)
                else:
                    imp = SimpleImputer(strategy="median").fit(Xtr_m)
                    def prep(X):
                        return pd.DataFrame(imp.transform(X), columns=Xtr_m.columns)
                return prep(Xtr_m), prep(Xes_m), prep(Xva_m), prep(Xte_m), cat_cols

            ytr, yes = tr_fit[obs].to_numpy(), tr_es[obs].to_numpy()
            yva, yte = va[obs].to_numpy(), te[obs].to_numpy()
            ident = {"station_id": te["station_id"].to_numpy(),
                     "target_time_utc": te["target_time_utc"].to_numpy(),
                     "lead_time": te["lead_time"].to_numpy(),
                     "season": te["season"].to_numpy()}
            logger.info("%s/%s: train=%d early-stop=%d calibration=%d test=%d",
                        args.target, label, len(tr_fit), len(tr_es), len(va), len(te))

            # raw / bias q50 + validation residual quantiles（经验分位）
            raw_tr, raw_va, raw_te = (x["cloud_cover_fcst" if args.target == "cloud" else "ghi_fcst"].to_numpy()
                                      for x in (tr_fit, va, te))
            stat = np.median
            bias_map = (tr_fit.assign(_r=raw_tr - ytr).groupby("station_id")["_r"].median()).to_dict()
            bias_va = va.apply(lambda r: r["cloud_cover_fcst" if args.target == "cloud" else "ghi_fcst"]
                               - bias_map[r.station_id], axis=1).to_numpy()
            bias_te = te.apply(lambda r: r["cloud_cover_fcst" if args.target == "cloud" else "ghi_fcst"]
                               - bias_map[r.station_id], axis=1).to_numpy()
            for model, base_va, base_te in (("raw", raw_va, raw_te),
                                            ("bias", bias_va, bias_te)):
                rq = residual_quantiles(yva, base_va)
                q = np.column_stack([base_te + rq[t] for t in TAUS])
                q = np.sort(q, axis=1)
                if args.target == "cloud":
                    q = np.clip(q, 0, 100)
                frames.append(pd.DataFrame({"model": model, "variant": variant, "unit": label,
                                            "y": yte, **ident,
                                            **{f"q{t:g}": q[:, i] for i, t in enumerate(TAUS)}}))
            # linear = OLS；ridge = Ridge(alpha)，α 由月分层五折 CV 选择
            for model in ("linear", "ridge"):
                pipe = (tv.lin_pipe(0.0, kind="ols") if model == "linear"
                        else tv.lin_pipe(RIDGE_ALPHA, kind="ridge"))
                pipe.fit(tr_fit[tv.FEATURES_NUM + tv.CAT_COLS], ytr)
                bv = pipe.predict(va[tv.FEATURES_NUM + tv.CAT_COLS])
                bt = pipe.predict(te[tv.FEATURES_NUM + tv.CAT_COLS])
                rq = residual_quantiles(yva, bv)
                q = np.column_stack([bt + rq[t] for t in TAUS])
                if args.target == "cloud":
                    q = np.clip(q, 0, 100)
                frames.append(pd.DataFrame({"model": model, "variant": variant, "unit": label,
                                            "y": yte, **ident,
                                            **{f"q{t:g}": q[:, i] for i, t in enumerate(TAUS)}}))
            for model in ("lgbm", "xgb"):
                Xtr_m, Xes_m, _, Xte_m, cat_cols = build_matrices(ENCODING[model])
                preds = [quantile_models(model, t, Xtr_m, ytr, Xes_m, yes, Xte_m, cat_cols)
                         for t in TAUS]
                q = np.column_stack(preds)
                CROSS[(model, label)] = tv.crossing_rate(q)
                q = np.sort(q, axis=1)
                if args.target == "cloud":
                    q = np.clip(q, 0, 100)
                frames.append(pd.DataFrame({"model": model, "variant": variant, "unit": label,
                                            "y": yte, **ident,
                                            **{f"q{t:g}": q[:, i] for i, t in enumerate(TAUS)}}))
            logger.info("%s/%s/%s done", args.target, variant, label)
        pred = pd.concat(frames, ignore_index=True)
        pred.to_csv(out / f"{variant}_predictions.csv", index=False)
        rows = []
        for (model, label), g in pred.groupby(["model", "unit"]):
            qmat = g[[f"q{t:g}" for t in TAUS]].to_numpy()
            rows.append(dict(model=model, unit=label,
                             crossing_rate=CROSS.get((model, label), 0.0),
                             **metrics(g.y.to_numpy(), qmat)))
        res = pd.DataFrame(rows)
        res.to_csv(out / f"{variant}_results.csv", index=False)
        lines = [f"# v1_ml quantile ({args.target}/{variant})", "",
                 "pinball family, test 2025-09~2026-08", ""]
        for model, g in res.groupby("model"):
            lines += [f"## {model}", "| unit | n | mean_pinball | crps_q7_trunc | crossing | "
                      "coverage_50 | coverage_90 | width_50 | width_90 | mae | rmse |",
                      "|---|---|---|---|---|---|---|---|---|---|---|"]
            for _, r in g.iterrows():
                lines.append(f"| {r.unit} | {int(r.n)} | {r.mean_pinball:.3f} | "
                             f"{r.crps_q7_trunc:.3f} | {r.crossing_rate:.4f} | "
                             f"{r.coverage_50:.3f} | {r.coverage_90:.3f} | "
                             f"{r.width_50:.2f} | {r.width_90:.2f} | "
                             f"{r.mae:.2f} | {r.rmse:.2f} |")
        (out / f"{variant}_summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("done -> %s", out)


if __name__ == "__main__":
    main()
