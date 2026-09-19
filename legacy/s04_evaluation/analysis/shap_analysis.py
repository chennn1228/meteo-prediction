# -*- coding: utf-8 -*-
"""Quantile-level / cross-lead / cross-region SHAP + SHAP interaction values (GHI).

Vehicle: LightGBM quantile models (exact TreeExplainer, CPU-fast), trained on the
frozen training pool, explained on the test year (daytime). Deep-model SHAP needs
server weights -> deferred.

Answers:
  1) do drivers differ across quantiles tau (Spearman of importance vectors)?
  2) do drivers shift across lead D+1/2/3 and across regions (station conditioning)?
  3) which feature PAIRS interact strongly (interaction values)?

Outputs: reports/04_error_analysis/shap_*.csv ; figs/04_error_analysis/fig_shap_*.
"""
import sys
from pathlib import Path
CODE_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / "config" / "01_sites.yaml").exists())
sys.path.insert(0, str(CODE_ROOT / "src"))
import numpy as np, pandas as pd, yaml
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import shap
import lightgbm as lgb
from scipy.stats import spearmanr
from s04_evaluation.analysis.plot_common import (
    apply_pub_style, PALETTE, COLORS, save_pub, panel_label)
from s03_models.train import train_v1 as tv
apply_pub_style(font_size=8)

TAU = [0.05,0.10,0.25,0.50,0.75,0.90,0.95]
LEADS = [24,48,72]
TEST_START = pd.Timestamp("2025-09-01", tz="UTC")
N_SHAP = 4000; N_INTER = 1200
rng = np.random.default_rng(0)

# ---- regions ----
cfg = yaml.safe_load(open(CODE_ROOT/"config/01_sites.yaml", encoding="utf-8"))
def find_regions(obj, acc):
    if isinstance(obj, dict):
        for k,v in obj.items():
            if k in ("region","layer","group") and isinstance(v,str): pass
            find_regions(v, acc)
    elif isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict) and "id" in it and ("region" in it or "layer" in it):
                acc[it["id"]] = it.get("region") or it.get("layer")
            find_regions(it, acc)
sreg = {}
find_regions(cfg, sreg)

# ---- data ----
feat = pd.concat([pd.read_parquet(p) for p in sorted(
    (CODE_ROOT/"data/03_featured").glob("*_featured_2024-02_2026-09.parquet"))], ignore_index=True)
feat["target_time_utc"] = pd.to_datetime(feat["target_time_utc"])
ghi = feat[(feat.is_day_fcst==1) & feat["ghi_obs_sat"].notna()].copy()
X = tv.tree_matrix_encoded(ghi, "onehot")[0]
X = X.copy(); X["__y"]=ghi["ghi_obs_sat"].values
X["__lead"]=ghi["lead_time"].values
X["__is_test"]=(ghi["target_time_utc"]>=TEST_START).values
X["__station"]=ghi["station_id"].values
X = X.reset_index(drop=True)
FEATCOLS=[c for c in X.columns if not c.startswith("__")]
def grp(name):
    if name.startswith("station"): return "station(one-hot)"
    if name.startswith("season"): return "season(one-hot)"
    return name
GROUPS = [grp(c) for c in FEATCOLS]
uniq_groups = list(dict.fromkeys(GROUPS))
def group_mean_abs(sv):  # sv: n x F -> per-group mean|shap|
    df=np.abs(sv); out={}
    for g in uniq_groups:
        idx=[i for i,c in enumerate(FEATCOLS) if grp(c)==g]
        out[g]=df[:,idx].sum(axis=1).mean()
    return pd.Series(out)

def fit_explain(mask_tr, mask_te, tau, n=N_SHAP):
    Xt=X.loc[mask_tr, FEATCOLS]; yt=X.loc[mask_tr,"__y"]
    Xe=X.loc[mask_te, FEATCOLS]
    m=lgb.LGBMRegressor(objective="quantile",alpha=tau,num_leaves=31,n_estimators=600,
                        learning_rate=0.05,subsample=0.9,subsample_freq=1,verbose=-1,random_state=0)
    m.fit(Xt,yt)
    idx=rng.choice(np.where(mask_te)[0], size=min(n,int(mask_te.sum())), replace=False)
    Xe_s=X.loc[idx, FEATCOLS]
    sv=shap.TreeExplainer(m).shap_values(Xe_s)
    return sv

# ---- 1) quantile-level importance (avg over leads) ----
print("SHAP per tau ...")
q_imp={}
for tau in TAU:
    acc=[]
    for L in LEADS:
        mt=(X["__lead"]==L)&(~X["__is_test"]); me=(X["__lead"]==L)&(X["__is_test"])
        acc.append(group_mean_abs(fit_explain(mt,me,tau)))
    q_imp[tau]=pd.concat(acc,axis=1).mean(axis=1)
qmat=pd.DataFrame(q_imp)
qmat.to_csv(CODE_ROOT/"reports/04_error_analysis/shap_quantile_importance.csv")
# Spearman between tau importance vectors
sp=pd.DataFrame(index=TAU,columns=TAU,dtype=float)
for a in TAU:
    for b in TAU:
        sp.loc[a,b]=spearmanr(qmat[a],qmat[b]).statistic
print("Spearman(tau=0.05 vs 0.95)=%.3f  (0.5 vs 0.95)=%.3f"%(sp.loc[0.05,0.95],sp.loc[0.5,0.95]))

# ---- 2) lead + region ----
print("SHAP per lead ...")
lead_imp={}
for L in LEADS:
    acc=[]
    for tau in TAU:
        mt=(X["__lead"]==L)&(~X["__is_test"]); me=(X["__lead"]==L)&(X["__is_test"])
        acc.append(group_mean_abs(fit_explain(mt,me,tau)))
    lead_imp[f"D+{L//24}"]=pd.concat(acc,axis=1).mean(axis=1)
lmat=pd.DataFrame(lead_imp); lmat.to_csv(CODE_ROOT/"reports/04_error_analysis/shap_lead_importance.csv")
sp_lead=pd.DataFrame({a:{b:spearmanr(lmat[a],lmat[b]).statistic for b in lmat.columns} for a in lmat.columns})

# region: q0.5 pooled leads, mean|shap| by region
print("SHAP per region ...")
reg_imp={}
Xte=X[X["__is_test"]]; Xtr=X[~X["__is_test"]]
for L in LEADS:
    mt=(X["__lead"]==L)&(~X["__is_test"]); me=(X["__lead"]==L)&(X["__is_test"])
    sv=fit_explain(mt,me,0.5,n=N_SHAP)
    idx=rng.choice(np.where(me)[0],size=min(N_SHAP,int(me.sum())),replace=False)
    reg_of=[sreg.get(X.loc[i,"__station"],"unknown") for i in idx]
    gdf=pd.DataFrame({"g":np.abs(sv).sum(axis=1),"reg":reg_of})
    reg_imp[L]=gdf.groupby("reg")["g"].mean()
rmat=pd.DataFrame(reg_imp); rmat.to_csv(CODE_ROOT/"reports/04_error_analysis/shap_region_importance.csv")

# ---- 3) interaction values (numeric features only, q0.5, D+1) ----
print("SHAP interaction ...")
numcols=[c for c in FEATCOLS if grp(c)==c]  # numeric only
mt=(X["__lead"]==24)&(~X["__is_test"]); me=(X["__lead"]==24)&(X["__is_test"])
Xt=X.loc[mt,numcols]; yt=X.loc[mt,"__y"]
m=lgb.LGBMRegressor(objective="quantile",alpha=0.5,num_leaves=31,n_estimators=600,
                    learning_rate=0.05,subsample=0.9,subsample_freq=1,verbose=-1,random_state=0).fit(Xt,yt)
idx=rng.choice(np.where(me)[0],size=N_INTER,replace=False); Xe_s=X.loc[idx,numcols]
iv=shap.TreeExplainer(m).shap_interaction_values(Xe_s)  # n x F x F
M=np.abs(iv).mean(axis=0); np.fill_diagonal(M,0)
pairs=[]
for i in range(len(numcols)):
    for j in range(i+1,len(numcols)):
        pairs.append((numcols[i],numcols[j],M[i,j]))
ptop=pd.DataFrame(pairs,columns=["f1","f2","mean_abs_interaction"]).sort_values("mean_abs_interaction",ascending=False)
ptop.to_csv(CODE_ROOT/"reports/04_error_analysis/shap_interaction_top.csv",index=False)
# main effect (diagonal) for ratio
main=np.abs(np.diagonal(iv,axis1=1,axis2=2)).mean(axis=0)
print("top interactions:\n", ptop.head(8).to_string(index=False))
print("max main-effect=%.3f  max interaction=%.3f"%(main.max(),M.max()))

# ---- FIG: SHAP quantile heatmap + spearman ----
fig,axes=plt.subplots(1,2,figsize=(7.4,3.4),gridspec_kw={"width_ratios":[2.2,1]})
ax=axes[0]
top=qmat.max(axis=1).sort_values(ascending=False).head(14).index
Q=qmat.loc[top]/qmat.loc[top].values.max()
im=ax.imshow(Q.values,cmap="Blues",aspect="auto",vmin=0,vmax=1)
ax.set_xticks(range(len(TAU))); ax.set_xticklabels([f"{t:g}" for t in TAU],fontsize=6)
ax.set_yticks(range(len(top))); ax.set_yticklabels(top,fontsize=6)
ax.set_xlabel("quantile τ"); ax.set_ylabel("feature (mean |SHAP|, row-normalized)")
plt.colorbar(im,ax=ax,fraction=0.03,pad=0.02)
panel_label(ax,"a")
ax=axes[1]
im=ax.imshow(sp.values,cmap="RdBu_r",vmin=0,vmax=1,aspect="auto")
ax.set_xticks(range(len(TAU))); ax.set_xticklabels([f"{t:g}" for t in TAU],fontsize=6,rotation=90)
ax.set_yticks(range(len(TAU))); ax.set_yticklabels([f"{t:g}" for t in TAU],fontsize=6)
for i in range(len(TAU)):
    for j in range(len(TAU)):
        ax.text(j,i,f"{sp.values[i,j]:.2f}",ha="center",va="center",fontsize=5,
                color="white" if sp.values[i,j]<0.5 else "black")
ax.set_title("Spearman of τ-importance",fontsize=7); plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
panel_label(ax,"b")
fig.tight_layout(pad=0.6)
save_pub(fig,CODE_ROOT/"figs/04_error_analysis","fig_shap_quantile")

# ---- FIG: lead + region ----
fig,axes=plt.subplots(1,2,figsize=(7.2,3.0))
ax=axes[0]
top=lmat.max(axis=1).sort_values(ascending=False).head(10).index
x=np.arange(len(top)); w=0.26
for i,c in enumerate(lmat.columns):
    ax.bar(x+i*w,lmat.loc[top,c]/lmat.loc[top,c].max(),w,label=c,color=COLORS[i],alpha=0.9,edgecolor="none")
ax.set_xticks(x+w); ax.set_xticklabels(top,rotation=45,ha="right",fontsize=6)
ax.set_ylabel("normalized mean |SHAP|"); ax.set_xlabel("feature")
ax.legend(fontsize=6,loc="lower center",bbox_to_anchor=(0.5,1.05),ncol=3,frameon=False)
panel_label(ax,"a")
ax=axes[1]
regs=[r for r in rmat.index if r!="unknown"][:5]
x=np.arange(len(regs)); w=0.8/max(1,len(LEADS))
for i,L in enumerate(LEADS):
    vals=[rmat.loc[r,L] if r in rmat.index else np.nan for r in regs]
    ax.bar(x+i*w,vals,w,label=f"D+{L//24}",color=COLORS[i],alpha=0.9,edgecolor="none")
ax.set_xticks(x+w*1); ax.set_xticklabels([str(r) for r in regs],rotation=45,ha="right",fontsize=6)
ax.set_ylabel("mean |SHAP| (q0.5)"); ax.set_xlabel("region")
ax.legend(fontsize=6,loc="lower center",bbox_to_anchor=(0.5,1.05),ncol=3,frameon=False)
panel_label(ax,"b")
fig.tight_layout(pad=0.6)
save_pub(fig,CODE_ROOT/"figs/04_error_analysis","fig_shap_lead_region")

# ---- FIG: interaction heatmap (top numeric) + top pairs ----
fig,axes=plt.subplots(1,2,figsize=(7.4,3.4),gridspec_kw={"width_ratios":[1.3,1]})
ax=axes[0]
topi=[c for c in ptop.f1.unique()][:10]
ii=[numcols.index(c) for c in topi]
Sm=M[np.ix_(ii,ii)]; Sm=Sm/ (Sm.max() or 1)
im=ax.imshow(Sm,cmap="Reds",aspect="auto")
ax.set_xticks(range(len(topi))); ax.set_xticklabels(topi,rotation=90,fontsize=5.5)
ax.set_yticks(range(len(topi))); ax.set_yticklabels(topi,fontsize=5.5)
ax.set_xlabel("feature"); ax.set_ylabel("feature")
plt.colorbar(im,ax=ax,fraction=0.046,pad=0.03)
panel_label(ax,"a")
ax=axes[1]
tp=ptop.head(10)
ax.barh(range(len(tp)),tp.mean_abs_interaction.values,
        color=PALETTE["blue_main"],alpha=0.9,edgecolor="none")
ax.set_yticks(range(len(tp)))
ax.set_yticklabels([f"{a}×{b}" for a,b in zip(tp.f1,tp.f2)],fontsize=5.5)
ax.invert_yaxis(); ax.set_xlabel("mean |interaction|"); ax.set_ylabel("feature pair")
panel_label(ax,"b")
fig.tight_layout(pad=0.6)
save_pub(fig,CODE_ROOT/"figs/04_error_analysis","fig_shap_interaction")
print("wrote shap csvs + fig_shap_quantile/lead_region/interaction")
print("DONE shap")
