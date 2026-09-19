# -*- coding: utf-8 -*-
"""Local downstream-2: DM test + block-bootstrap conformal + dual Fig.5.

Inputs (per-sample, carry station/time):
  v1 (lgbm/xgb/raw) ghi : reports/03_modeling/v1_ml/full/quantile/ghi/per_lead_predictions.csv
  deep fixed-kt ghi     : reports/05_robustness/kt_sensitivity/fixedkt_top/{v4_tcn,v6_transformer,v7_autoformer,v8_informer}/ghi/test_predictions.csv
  aggregate (all models): reports/03_modeling/*/full/quantile/ghi/results.csv + fixedkt_top results.csv

Outputs:
  reports/05_robustness/dm_test_ghi.csv
  reports/06_probability/conformal_blockbootstrap.csv
  figs/05_robustness/fig_dm_conformal_full
  figs/03_modeling/00_comparison/fig_model_comparison_dual   (main broken-kt + robustness fixed-kt overlay)
"""
import sys, itertools
from pathlib import Path
CODE_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / "config" / "01_sites.yaml").exists())
sys.path.insert(0, str(CODE_ROOT / "src"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from s04_evaluation.analysis.plot_common import (
    apply_pub_style, PALETTE, COLORS, save_pub, panel_label)
apply_pub_style(font_size=8)
R = CODE_ROOT / "reports"
TAU = [0.05,0.10,0.25,0.50,0.75,0.90,0.95]; QC=[f"q{t:g}" for t in TAU]

def leg_above(ax,ncol=2,fs=6,y=1.08):
    ax.legend(fontsize=fs,loc="lower center",bbox_to_anchor=(0.5,y),ncol=ncol,frameon=False)
def xcat(ax,labels,fs=6):
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,rotation=45,ha="right",rotation_mode="anchor",fontsize=fs)

feat = pd.concat([pd.read_parquet(p) for p in sorted(
    (CODE_ROOT/"data/03_featured").glob("*_featured_2024-02_2026-09.parquet"))],ignore_index=True)
feat["target_time_utc"]=pd.to_datetime(feat["target_time_utc"])
sky = feat[["station_id","target_time_utc","lead_time","sky_type"]].drop_duplicates(
    subset=["station_id","target_time_utc","lead_time"])

def load_v1(model):
    d=pd.read_csv(R/"03_modeling/v1_ml/full/quantile/ghi/per_lead_predictions.csv")
    d=d[d.model==model].copy()
    d["lead_time"]=d["unit"].map({"D+1":24,"D+2":48,"D+3":72})
    d["target_time_utc"]=pd.to_datetime(d["target_time_utc"])
    return d[["station_id","target_time_utc","lead_time","y"]+QC]
def load_deep(new):
    f=R/"05_robustness/kt_sensitivity/fixedkt_top"/new/"ghi/test_predictions.csv"
    d=pd.read_csv(f); d["target_time_utc"]=pd.to_datetime(d["target_time_utc"])
    return d[["station_id","target_time_utc","lead_time","y"]+QC]

M={}
M["raw"]=load_v1("raw"); M["lgbm"]=load_v1("lgbm"); M["xgb"]=load_v1("xgb")
for new in ["v4_tcn","v6_transformer","v7_autoformer","v8_informer"]:
    if (R/"05_robustness/kt_sensitivity/fixedkt_top"/new/"ghi/test_predictions.csv").exists():
        M[new]=load_deep(new)
# align all to common key index
keys=None
for k,d in M.items():
    kk=set(map(tuple,d[["station_id","target_time_utc","lead_time"]].values))
    keys = kk if keys is None else (keys & kk)
keys=sorted(keys)
kidx=pd.MultiIndex.from_tuples(keys,names=["station_id","target_time_utc","lead_time"])
def reindex(d):
    d=d.set_index(["station_id","target_time_utc","lead_time"]).loc[kidx].reset_index()
    return d
M={k:reindex(d) for k,d in M.items()}
print("aligned rows:",len(kidx),"models:",list(M))

# ---------- DM test (HAC + Harvey) ----------
def daily_sqerr(d):
    e=(d["q0.5"]-d["y"])**2
    df=pd.DataFrame({"t":d["target_time_utc"],"e":e})
    return df.groupby(df["t"].dt.normalize())["e"].mean().sort_index().values
def dm(x,y,h=1):
    d=x-y; n=len(d); dbar=d.mean()
    gamma0=np.mean((d-dbar)**2)
    gam=[np.mean((d[t:]-dbar)*(d[:-t]-dbar)) for t in range(1,h+1)]
    V=gamma0+2*sum((1-t/(h+1))*g for t,g in zip(range(1,h+1),gam))
    T=dbar/np.sqrt(V/n)
    corr=np.sqrt(n/(n+2*(h-1)))          # Harvey et al. 1997
    return T*corr
def norm_p(t):
    from math import erf,sqrt
    return 2*(1-0.5*(1+erf(abs(t)/sqrt(2))))
pairs=list(itertools.combinations(["raw","lgbm","xgb","v4_tcn","v6_transformer","v7_autoformer","v8_informer"],2))
rows=[]
for a,b in pairs:
    ea,eb=daily_sqerr(M[a]),daily_sqerr(M[b])
    T=dm(ea,eb,h=3); p=norm_p(T)
    rows.append(dict(modelA=a,modelB=b,DM=T,p_raw=p,
                     better=(a if ea.mean()<eb.mean() else b)))
dm_df=pd.DataFrame(rows)
# Benjamini-Hochberg FDR
dm_df=dm_df.sort_values("p_raw").reset_index(drop=True)
m=len(dm_df); dm_df["q_bh"]=[min(1.0,p*m/(i+1)) for i,p in enumerate(dm_df.p_raw)]
dm_df.to_csv(R/"05_robustness/dm_test_ghi.csv",index=False)
print("wrote dm_test_ghi.csv; signif pairs(BH q<0.05):",(dm_df.q_bh<0.05).sum(),"/",m)

# ---------- block-bootstrap conformal (rolling-origin) ----------
SPLIT=pd.Timestamp("2025-11-30",tz="UTC")
tt=pd.Series([k[1] for k in keys]).dt.normalize()
calib=(tt<=SPLIT).values; evalm=~calib
rng=np.random.default_rng(0); BLK=7; B=500
def cov_after(d,delta):
    q=d[QC].values.copy()
    for i in range(7): q[:,i]+=delta[i]
    q=np.sort(q,axis=1)
    y=d["y"].values
    ev=y[evalm]; qq=q[evalm]
    c90=((ev>=qq[:,0])&(ev<=qq[:,6])).mean()
    ov=(d.merge(sky,on=["station_id","target_time_utc","lead_time"],how="left")["sky_type"].values=="overcast")&evalm
    co=((y[ov]>=q[ov,0])&(y[ov]<=q[ov,6])).mean() if ov.sum() else np.nan
    return c90,co
crows=[]
for name,d in M.items():
    y=d["y"].values; q0=d[QC].values
    # raw (no conformal)
    c90r,cor=((y[evalm]>=q0[evalm,0])&(y[evalm]<=q0[evalm,6])).mean(), np.nan
    # split conformal delta
    delta_split=np.array([np.quantile(y[calib]-q0[calib,i],TAU[i]) for i in range(7)])
    # block bootstrap conformal delta (mean over B block-resamples of calib)
    nblk=int(np.ceil(calib.sum()/BLK)); idxc=np.where(calib)[0]
    dbs=np.zeros((B,7))
    for b in range(B):
        starts=rng.integers(0,max(1,idxc.size-BLK),size=nblk)
        samp=np.concatenate([idxc[s:s+BLK] for s in starts])[:idxc.size]
        for i in range(7): dbs[b,i]=np.quantile(y[samp]-q0[samp,i],TAU[i])
    delta_bb=dbs.mean(0)
    c90s,cos_=cov_after(d,delta_split); c90b,cob=cov_after(d,delta_bb)
    ov=(d.merge(sky,on=["station_id","target_time_utc","lead_time"],how="left")["sky_type"].values=="overcast")&evalm
    c90r_ov=((y[ov]>=q0[ov,0])&(y[ov]<=q0[ov,6])).mean() if ov.sum() else np.nan
    crows.append(dict(model=name,cov90_raw=c90r,cov90_split=c90s,cov90_blockbb=c90b,
                      cov90_overcast_raw=c90r_ov,cov90_overcast_split=cos_ if False else np.nan,
                      cov90_overcast_blockbb=cob))
conf=pd.DataFrame(crows)
conf.to_csv(R/"06_probability/conformal_blockbootstrap.csv",index=False)
print("wrote conformal_blockbootstrap.csv")
print(conf.round(3).to_string(index=False))

# ---------- figures: DM + conformal ----------
fig,axes=plt.subplots(1,2,figsize=(7.2,3.0))
ax=axes[0]
sig=dm_df.sort_values("p_raw")
ax.barh(range(len(sig)),-np.log10(sig.p_raw.clip(lower=1e-16)),
        color=[PALETTE["red_strong"] if q<0.05 else PALETTE["neutral_mid"] for q in sig.q_bh])
ax.set_yticks(range(len(sig))); ax.set_yticklabels(
    [f"{r.modelA} vs {r.modelB}" for _,r in sig.iterrows()],fontsize=5.5)
ax.axvline(-np.log10(0.05),color="k",ls="--",lw=0.8)
ax.set_xlabel("-log10 p  (red = BH q<0.05)"); ax.set_ylabel("model pair"); ax.set_xlim(0,None)
ax.set_title("DM test (HAC+Harvey)",fontsize=7); panel_label(ax,"a")
ax=axes[1]
x=np.arange(len(conf)); w=0.26
ax.bar(x-w,conf.cov90_raw,w,label="raw",color=PALETTE["neutral_mid"],edgecolor="none")
ax.bar(x,conf.cov90_split,w,label="split conformal",color=PALETTE["blue_secondary"],edgecolor="none")
ax.bar(x+w,conf.cov90_blockbb,w,label="block-bootstrap",color=PALETTE["blue_main"],edgecolor="none")
ax.axhline(0.9,color=PALETTE["red_strong"],ls="--",lw=0.8)
xcat(ax,conf.model.tolist()); ax.set_ylabel("coverage 90% (eval half)"); ax.set_xlabel("model")
leg_above(ax,ncol=3); panel_label(ax,"b")
fig.tight_layout(pad=0.6)
save_pub(fig,CODE_ROOT/"figs/05_robustness","fig_dm_conformal_full")

# ---------- dual Fig.5: broken-kt main + fixed-kt robustness ----------
def agg_allrow(path):
    if not Path(path).exists(): return None
    r=pd.read_csv(path); a=r[r.group=="all"]
    return None if not len(a) else a.iloc[0]
deep=["v2_mlp","v3_cnn","v4_tcn","v5_lstm","v6_transformer","v7_autoformer","v8_informer",
      "v9_fedformer","v10_itransformer","v11_patchtst","v12_dlinear","v13_timesnet","v14_tsmixer","v15_pinn"]
main=[]
for m in deep:
    a=agg_allrow(R/"03_modeling"/m/"full"/"quantile"/"ghi/results.csv")
    if a is not None: main.append((m,a.mean_pinball,a.rmse))
for vm in ["lgbm","xgb"]:
    d=load_v1(vm); q=d[QC].values; y=d["y"].values
    mp=np.mean([np.maximum(t*(y-q[:,i]),(t-1)*(y-q[:,i])).mean() for i,t in enumerate(TAU)])
    main.append((f"v1_{vm}",mp,np.sqrt(((q[:,3]-y)**2).mean())))
fixmap={"v4_tcn":"v4_tcn","v6_transformer":"v6_transformer","v7_autoformer":"v7_autoformer","v8_informer":"v8_informer"}
fixed={}
for old,new in fixmap.items():
    a=agg_allrow(R/"05_robustness/kt_sensitivity/fixedkt_top"/new/"ghi/results.csv")
    if a is not None: fixed[old]=(a.mean_pinball,a.rmse)
main.sort(key=lambda t:t[1])
names=[t[0] for t in main]; mp=[t[1] for t in main]
fig,ax=plt.subplots(figsize=(7.2,2.8))
x=np.arange(len(names))
ax.bar(x,mp,0.62,color=[PALETTE["green_3"] if n.startswith("v1") else PALETTE["blue_main"] for n in names],
       alpha=0.9,edgecolor="none",label="main (broken-kt)")
for i,n in enumerate(names):
    if n in fixed:
        ax.plot(i,fixed[n][0],marker="D",ms=6,color=PALETTE["red_strong"],zorder=5)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
handles=[Patch(color=PALETTE["green_3"],label="v1 tree/lin"),Patch(color=PALETTE["blue_main"],label="deep"),
         Line2D([0],[0],marker="D",color=PALETTE["red_strong"],ls="none",label="fixed-kt rerun")]
ax.set_xticks(x); ax.set_xticklabels([n.replace("v1_","") for n in names],rotation=45,ha="right",rotation_mode="anchor",fontsize=6)
ax.set_ylabel("mean pinball (GHI)"); ax.set_xlabel("model")
ax.legend(handles=handles,fontsize=6,loc="lower center",bbox_to_anchor=(0.5,1.06),ncol=3,frameon=False)
fig.tight_layout(pad=0.6)
save_pub(fig,CODE_ROOT/"figs/03_modeling/00_comparison","fig_model_comparison_dual")

print("wrote fig_dm_conformal_full and fig_model_comparison_dual")
print("DONE downstream2")
