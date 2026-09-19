# -*- coding: utf-8 -*-
"""特征共线性消融：33 特征基线 / PCA(>=95%方差) / |r|>0.98 去冗余。

协议（冻结）：
  白天 solar_elevation>10、ghi_obs_sat 非空；
  数据 tag=2024-02_2026-09；
  切分复用 split_protocol.default_split（训练池 80% / 月分层验证 20%，测试年不参与）。

模型：LightGBM / XGBoost，损失 L1(MAE) 与 L2(RMSE)，
      代表配置来自 docs/03_hyperparameters.md（不重调超参，保证路径归因）。
输出：
  reports/01_data_audit/features/collinearity_ablation_valid/{results.csv,summary.md}
  figs/01_data_audit/fig_collinearity_valid.{png,svg}
"""
import logging
import os
import sys
from pathlib import Path

CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config" / "01_sites.yaml").exists())
os.environ.setdefault("MPLCONFIGDIR", str(CODE_ROOT / ".cache" / "matplotlib"))
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(CODE_ROOT / "src" / "s03_models" / "train"))
sys.path.insert(0, str(CODE_ROOT / "src"))
import train_v1 as tv
from s04_evaluation.analysis.plot_common import apply_pub_style, PALETTE, save_pub, panel_label
sys.path.insert(0, str(CODE_ROOT / "src" / "s02_experiment"))
from split_protocol import default_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("collinearity_ablation")

RNG_SAMPLE = 300_000
CORR_TH = 0.98
PCA_VAR = 0.95
OUT_TAG = "_valid"


def load():
    sites = yaml.safe_load((CODE_ROOT / "config" / "01_sites.yaml").read_text(encoding="utf-8"))["sites"]
    tv.STATION_CODE = {s["id"]: i for i, s in enumerate(sites)}
    cols = tv.FEATURES_NUM + ["station_id", "target_time_utc", "ghi_obs_sat"]
    frames = []
    for s in sites:
        p = (CODE_ROOT / "data" / "03_featured"
             / f"{s['id']}_featured_2024-02_2026-09.parquet")
        df = pd.read_parquet(p, columns=cols)
        df["target_time_utc"] = pd.to_datetime(df["target_time_utc"], utc=True)
        frames.append(df)
    d = pd.concat(frames, ignore_index=True)
    d["season"] = d["target_time_utc"].apply(tv.season_of if hasattr(tv, "season_of") else _season_of)
    d = d.query(tv.DAY_FILTER).dropna(subset=["ghi_obs_sat"])
    return d.sort_values("target_time_utc").reset_index(drop=True)


def _season_of(ts):
    m = ts.month
    return ("spring" if m in (3, 4, 5) else "summer" if m in (6, 7, 8)
            else "autumn" if m in (9, 10, 11) else "winter")


def dedup_subset(tr_num, ytr):
    """|r|>0.98 去冗余：按 |r(y)| 降序贪婪保留代表量。"""
    sample = tr_num.sample(n=min(RNG_SAMPLE, len(tr_num)), random_state=0)
    ys = pd.Series(ytr, index=tr_num.index).loc[sample.index]
    rel = sample.corrwith(ys).abs().sort_values(ascending=False)
    corr = sample.corr()
    keep = []
    for name in rel.index:
        if not keep or float(corr.loc[name, keep].abs().max()) <= CORR_TH:
            keep.append(name)
    return keep


def build_sets(tr, va):
    imp = tv.SimpleImputer if hasattr(tv, "SimpleImputer") else None
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    num = tv.FEATURES_NUM
    ytr = tr["ghi_obs_sat"].to_numpy()
    yva = va["ghi_obs_sat"].to_numpy()

    med = SimpleImputer(strategy="median").fit(tr[num])
    Xtr_num = pd.DataFrame(med.transform(tr[num]), columns=num, index=tr.index)
    Xva_num = pd.DataFrame(med.transform(va[num]), columns=num, index=va.index)
    # 类别特征 one-hot（与 train_v1.tree_matrix 一致，避免整数编码引入虚假序关系）
    tm_tr, tm_va = tv.tree_matrix(tr), tv.tree_matrix(va)
    cat_cols = [c for c in tm_tr.columns if c.startswith(("station_", "season_"))]
    cat_tr, cat_va = tm_tr[cat_cols], tm_va[cat_cols]

    sets = {}
    base_name = "base"
    sets[base_name] = pd.concat([Xtr_num, cat_tr], axis=1), pd.concat([Xva_num, cat_va], axis=1)

    keep = dedup_subset(Xtr_num, ytr)
    # 阈值下若没有删除任何特征，dedup 与 base 是同一特征集合；
    # 必须按原顺序组织，否则列顺序差异会在树模型 colsample 下产生伪差异。
    if set(keep) == set(num):
        keep = list(num)
    logger.info("去冗余子集: %d -> %d 特征", len(num), len(keep))
    sets["dedup"] = (pd.concat([Xtr_num[keep], cat_tr], axis=1),
                     pd.concat([Xva_num[keep], cat_va], axis=1))

    pca_sample_idx = Xtr_num.sample(n=min(RNG_SAMPLE, len(Xtr_num)), random_state=0).index
    scaler = StandardScaler().fit(Xtr_num.loc[pca_sample_idx])
    pca_base = scaler.transform(Xtr_num.loc[pca_sample_idx])
    pca_raw = PCA(random_state=0).fit(Xtr_num.loc[pca_sample_idx])
    pca_full = PCA(random_state=0).fit(pca_base)
    sample_corr = Xtr_num.loc[pca_sample_idx].corr()
    corr_pairs = []
    for i, a in enumerate(sample_corr.columns):
        for b in sample_corr.columns[i + 1:]:
            corr_pairs.append((a, b, float(sample_corr.loc[a, b])))
    corr_pairs.sort(key=lambda t: -abs(t[2]))
    ev = np.cumsum(pca_full.explained_variance_ratio_)
    n_pc = int(np.searchsorted(ev, PCA_VAR) + 1)
    pca = PCA(n_components=n_pc, random_state=0).fit(pca_base)
    pc_tr = pd.DataFrame(pca.transform(scaler.transform(Xtr_num)),
                         columns=[f"pc{i+1}" for i in range(n_pc)], index=tr.index)
    pc_va = pd.DataFrame(pca.transform(scaler.transform(Xva_num)),
                         columns=[f"pc{i+1}" for i in range(n_pc)], index=va.index)
    logger.info("PCA>=%.0f%% 方差: %d 个主成分（PC1=%.1f%%, 累计前%d=%.1f%%）",
                PCA_VAR * 100, n_pc, pca.explained_variance_ratio_[0] * 100,
                n_pc, ev[n_pc - 1] * 100)
    sets["pca95"] = pd.concat([pc_tr, cat_tr], axis=1), pd.concat([pc_va, cat_va], axis=1)
    var_rows = []
    for method, pobj in (("raw", pca_raw), ("std", pca_full)):
        cum = np.cumsum(pobj.explained_variance_ratio_)
        for i, (v, c) in enumerate(zip(pobj.explained_variance_ratio_, cum), start=1):
            var_rows.append(dict(method=method, pc=i, var=v, cum=c))
    n_feat_map = {base_name: len(num), "dedup": len(keep), "pca95": n_pc}
    meta = {"n_pc": n_pc, "keep": keep, "var_cum": float(ev[n_pc - 1]),
            "var_rows": var_rows,
            "corr_top": corr_pairs,
            "n_feat_map": n_feat_map,
            "std_pc1": float(pca_full.explained_variance_ratio_[0]),
            "raw_pc1": float(pca_raw.explained_variance_ratio_[0])}
    return sets, ytr, yva, meta


def fit_predict(algo, loss, Xtr, ytr, Xva, yva):
    early = 150
    if algo == "lgbm":
        import lightgbm as lgb
        params = dict(objective="regression_l1" if loss == "mae" else "regression",
                      metric="mae" if loss == "mae" else "rmse",
                      n_estimators=3000, learning_rate=0.05, num_leaves=63, max_depth=8,
                      subsample=0.8, colsample_bytree=0.8, random_state=0, verbose=-1)
        m = lgb.LGBMRegressor(**params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(early, verbose=False)])
        return m.predict(Xva, num_iteration=m.best_iteration_), int(m.best_iteration_)
    import xgboost as xgb
    m = xgb.XGBRegressor(
        objective="reg:absoluteerror" if loss == "mae" else "reg:squarederror",
        eval_metric="mae" if loss == "mae" else "rmse",
        n_estimators=3000, learning_rate=0.08, max_depth=7,
        subsample=0.85, colsample_bytree=0.8, tree_method="hist",
        early_stopping_rounds=early, random_state=0)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m.predict(Xva, iteration_range=(0, m.best_iteration + 1)), int(m.best_iteration)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = load()
    logger.info("样本: %d 行（train pool，测试年 2025-09 起未参与）", len(d))
    tr, va, _ = default_split(d)
    logger.info("月分层折 0：train=%d val=%d", len(tr), len(va))
    sets, ytr, yva, meta = build_sets(tr, va)

    rows = []
    for loss in ("mae", "mse"):
        for algo in ("lgbm", "xgb"):
            for sname, (Xtr, Xva) in sets.items():
                pred, best_iter = fit_predict(algo, loss, Xtr, ytr, Xva, yva)
                m = tv.metrics(yva, pred)
                n_num = meta["n_feat_map"][sname]
                rows.append(dict(algo=algo, loss=loss, feat_set=sname,
                                 n_feat=n_num, n_cols=Xtr.shape[1], n_val=int(m["n"]),
                                 val_mae=m["mae"], val_rmse=m["rmse"],
                                 val_r2=m["r2"], best_iter=best_iter))
                logger.info("%s/%s/%s: val MAE=%.2f RMSE=%.2f (iter=%d)",
                            algo, loss, sname, m["mae"], m["rmse"], best_iter)
    res = pd.DataFrame(rows)
    out = CODE_ROOT / "reports" / "01_data_audit" / "features" / f"collinearity_ablation{OUT_TAG}"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "results.csv", index=False)
    var_df = pd.DataFrame(meta["var_rows"])
    var_df.to_csv(out / "pca_variance_curves.csv", index=False)
    raw_ev = var_df.loc[var_df.method == "raw", "var"].to_numpy()
    pd.DataFrame({"method": ["raw_pca", "std_pca"],
                  "pc1": [meta["raw_pc1"], meta["std_pc1"]],
                  "n_pc_95": [int(np.searchsorted(np.cumsum(raw_ev), PCA_VAR) + 1),
                              meta["n_pc"]]}
                 ).to_csv(out / "pca_artifact_check.csv", index=False)
    pd.DataFrame(meta["corr_top"], columns=["feat_a", "feat_b", "pearson_r"]
                 ).to_csv(out / "corr_top.csv", index=False)

    lines = ["# 特征共线性消融（验证集；测试年不参与）\n",
             "- 切分: split_protocol.default_split（训练池 80%，月分层验证 20%，覆盖 12 个月）",
             f"- 去冗余子集: |Pearson r|>{CORR_TH} 保留代表量，共 {len(meta['keep'])} 个: "
             + ", ".join(meta["keep"]),
             (f"- 去冗余说明: 阈值下未删除任何特征（与 33 特征基线内容相同，不再作为独立对照）"
              if len(meta["keep"]) == len(tv.FEATURES_NUM) else ""),
             f"- PCA: 标准化 ≥{PCA_VAR:.0%} 方差 -> {meta['n_pc']} 个主成分"
             f"（累计 {meta['var_cum']:.1%}）；未标准化 PC1={meta['raw_pc1']:.1%}、"
             f"标准化 PC1={meta['std_pc1']:.1%}",
             "- 模型配置: LGBM(lr=0.05, leaves=63, depth=8) / XGB(lr=0.08, depth=7)，均早停150轮\n"]
    piv = res.pivot_table(index=["loss", "algo"], columns="feat_set",
                          values=["val_mae", "val_rmse"], aggfunc="mean")
    for loss in ("mae", "mse"):
        lines.append(f"## loss={loss}")
        lines.append("| algo | set | n_feat | val MAE | val RMSE | best_iter |")
        lines.append("|---|---|---|---|---|---|")
        sub = res[res["loss"] == loss].sort_values(["algo", "feat_set"])
        for _, r in sub.iterrows():
            lines.append(f"| {r['algo']} | {r['feat_set']} | {int(r['n_feat'])} | "
                         f"{r['val_mae']:.2f} | {r['val_rmse']:.2f} | {int(r['best_iter'])} |")
        lines.append("")
    lines.append("> n_feat=数值特征数（另有 20 站 + 4 季 one-hot 类别列，未计入 n_feat）\n")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": [0.9, 1, 1]})
    g = var_df[var_df.method == "std"].sort_values("pc")
    axes[0].plot(g.pc, g.cum, color=PALETTE["blue_main"], lw=1.8, marker="o", ms=3)
    axes[0].axhline(PCA_VAR, ls="--", color=PALETTE["red_strong"], lw=1)
    axes[0].axvline(meta["n_pc"], ls=":", color=PALETTE["red_strong"], lw=1)
    axes[0].annotate(f"95% at PC{meta['n_pc']}",
                     xy=(meta["n_pc"], PCA_VAR), xytext=(meta["n_pc"] + 1.5, 0.55),
                     color=PALETTE["red_strong"], fontsize=7,
                     arrowprops=dict(arrowstyle="-", color=PALETTE["red_strong"], lw=0.8))
    axes[0].set_xlabel("number of principal components")
    axes[0].set_ylabel("cumulative explained variance")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Standardized PCA")
    panel_label(axes[0], "a")
    base_name = "base"
    order = [base_name]
    if len(meta["keep"]) < len(tv.FEATURES_NUM):
        order.append("dedup")
    order.append("pca95")
    labels = {base_name: "33 features", "dedup": f"dedup({len(meta['keep'])})",
              "pca95": f"PCA 95% ({meta['n_pc']} PCs)"}
    colors = {base_name: PALETTE["blue_main"], "dedup": PALETTE["green_3"],
              "pca95": PALETTE["red_strong"]}
    for ax, metric, title, lab in ((axes[1], "val_mae", "Validation MAE", "b"),
                                   (axes[2], "val_rmse", "Validation RMSE", "c")):
        ax.set_ylabel(metric.replace("val_", "").upper() + " (W m$^{-2}$)")
        for loss, off in (("mae", -0.22), ("mse", 0.22)):
            xs = []
            for i, algo in enumerate(("lgbm", "xgb")):
                for j, sname in enumerate(order):
                    r = res[(res["algo"] == algo) & (res["loss"] == loss) & (res["feat_set"] == sname)]
                    if len(r):
                        x = i * 3 + j + off
                        xs.append((x, algo, sname))
                        ax.bar(x, r.iloc[0][metric], width=0.42, color=colors[sname], alpha=0.9,
                               label=None if (loss != "mae" or i or j) else labels[sname])
            ax.set_title(title)
            ax.set_xticks([(len(order) - 1) / 2 + 3 * k for k in range(2)])
            ax.set_xticklabels(["LightGBM", "XGBoost"])
        panel_label(ax, lab)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s]) for s in order]
    axes[1].legend(handles, [labels[s] for s in order], loc="upper left", frameon=False)
    apply_pub_style(font_size=8)
    if len(meta["keep"]) == len(tv.FEATURES_NUM):
        fig.text(0.5, 0.01,
                 "dedup omitted: |r|>0.98 removed no feature, so it is identical to "
                 "the 33-feature baseline",
                 ha="center", fontsize=7, style="italic")
    fig_dir = CODE_ROOT / "figs" / "01_data_audit"
    save_pub(fig, fig_dir, "fig_collinearity_valid")
    logger.info("done -> %s", out)


def diag():
    """PCA 口径核对：标准化 vs 未标准化；特征两两相关 top 对（不跑模型）。"""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    d = load()
    tr, _, _ = default_split(d)
    num = tv.FEATURES_NUM
    imp = SimpleImputer(strategy="median").fit(tr[num])
    X = pd.DataFrame(imp.transform(tr[num]), columns=num, index=tr.index)
    sample = X.sample(n=min(RNG_SAMPLE, len(X)), random_state=0)

    corr = sample.corr()
    pairs = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            pairs.append((a, b, float(corr.loc[a, b])))
    pairs.sort(key=lambda t: -abs(t[2]))
    n_high = sum(1 for _, _, v in pairs if abs(v) > CORR_TH)

    pca_raw = PCA(random_state=0).fit(sample)
    z = StandardScaler().fit(sample).transform(sample)
    pca_std = PCA(random_state=0).fit(z)
    out = CODE_ROOT / "reports" / "01_data_audit" / "features" / f"collinearity_ablation{OUT_TAG}"
    out.mkdir(parents=True, exist_ok=True)
    var_rows = []
    for method, ev in (("raw", pca_raw.explained_variance_ratio_),
                       ("std", pca_std.explained_variance_ratio_)):
        cum = np.cumsum(ev)
        for i, (v, c) in enumerate(zip(ev, cum), start=1):
            var_rows.append(dict(method=method, pc=i, var=v, cum=c))
    pd.DataFrame(var_rows).to_csv(out / "pca_variance_curves.csv", index=False)
    pd.DataFrame(pairs, columns=["feat_a", "feat_b", "pearson_r"]).to_csv(out / "corr_top.csv", index=False)
    pd.DataFrame({"method": ["raw_pca", "std_pca"],
                  "pc1": [pca_raw.explained_variance_ratio_[0], pca_std.explained_variance_ratio_[0]],
                  "pc1_2_cum": [pca_raw.explained_variance_ratio_[:2].sum(),
                                pca_std.explained_variance_ratio_[:2].sum()],
                  "n_pc_95": [int(np.searchsorted(np.cumsum(pca_raw.explained_variance_ratio_), PCA_VAR) + 1),
                              int(np.searchsorted(np.cumsum(pca_std.explained_variance_ratio_), PCA_VAR) + 1)]}
                ).to_csv(out / "pca_artifact_check.csv", index=False)
    print("top10 |r| pairs:")
    for a, b, v in pairs[:10]:
        print(f"  {a} ~ {b}: {v:.3f}")
    print(f"pairs with |r|>{CORR_TH}: {n_high} / {len(pairs)}")
    print(f"raw(unstd) PCA: PC1={pca_raw.explained_variance_ratio_[0]:.4f}, "
          f"PC1-2={pca_raw.explained_variance_ratio_[:2].sum():.4f}, "
          f"PCs-to-95%={int(np.searchsorted(np.cumsum(pca_raw.explained_variance_ratio_), PCA_VAR) + 1)}")
    print(f"std PCA: PC1={pca_std.explained_variance_ratio_[0]:.4f}, "
          f"PC1-2={pca_std.explained_variance_ratio_[:2].sum():.4f}, "
          f"PCs-to-95%={int(np.searchsorted(np.cumsum(pca_std.explained_variance_ratio_), PCA_VAR) + 1)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--diag":
        diag()
    else:
        main()
