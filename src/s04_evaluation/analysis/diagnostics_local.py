# -*- coding: utf-8 -*-
"""Local diagnostics: feature-GROUP ablation + residual diagnostics + grouped PIT.

(1) ablation: drop one feature group at a time, retrain LightGBM quantile (pooled
    leads), report mean_pinball / RMSE degradation vs full.
(4) residual: skew/kurt, heteroscedasticity (|res| vs |pred|), residual by
    weather/lead/station, residual vs cloud_cover_change.
(6) grouped PIT: PIT by lead and by weather type, KS per group.

Models used: v1_lgbm (per-sample, has ids) + v4_tcn fixed-kt (per-sample).
"""
import sys
from pathlib import Path
CODE_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / "config" / "01_sites.yaml").exists())
sys.path.insert(0, str(CODE_ROOT / "src"))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import lightgbm as lgb
from s04_evaluation.analysis.plot_common import (
    apply_pub_style, PALETTE, COLORS, save_pub, panel_label)
from s03_models.train import train_v1 as tv
apply_pub_style(font_size=8)

TAU=[0.05,0.10,0.25,0.50,0.75,0.90,0.95]; QC=[f"q{t:g}" for t in TAU]
TEST_START=pd.Timestamp("2025-09-01",tz="UTC")
rng=np.random.default_rng(0)

def leg_above(ax,ncol=2,fs=6,y=1.08):
    ax.legend(fontsize=fs,loc="lower center",bbox_to_anchor=(0.5,y),ncol=ncol,frameon=False)
def xcat(ax,labels,fs=6):
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,rotation=45,ha="right",rotation_mode="anchor",fontsize=fs)

# ---------- data ----------
feat=pd.concat([pd.read_parquet(p) for p in sorted(
    (CODE_ROOT/"data/03_featured").glob("*_featured_2024-02_2026-09.parquet"))],ignore_index=True)
feat["target_time_utc"]=pd.to_datetime(feat["target_time_utc"])
day=feat[(feat.is_day_fcst==1)&feat["ghi_obs_sat"].notna()].copy()
X=tv.tree_matrix_encoded(day,"onehot")[0].copy()
X["__y"]=day["ghi_obs_sat"].values
X["__is_test"]=(day["target_time_utc"]>=TEST_START).values
X=X.reset_index(drop=True)
FN=list(tv.FEATURES_NUM)
GROUPS={
 "radiation":["ghi_fcst","dhi_fcst","dni_fcst","gti_fcst","terrestrial_fcst"],
 "cloud":["cloud_cover_fcst","cloud_cover_change","cloud_cover_roll1","cloud_cover_roll3"],
 "geometry":["solar_elevation","solar_azimuth","ghi_clear_sky","dni_clear_sky"],
 "kt_index":["kt_fcst","kni","ghi_fcst_minus_clear","diffuse_fraction"],
 "met":["temp_fcst","rh_fcst","dewpoint_fcst","wind_speed_fcst","wind_dir_fcst","pressure_fcst","precip_fcst","sunshine_fcst"],
 "temporal_lag":["ghi_fcst_lag1","ghi_fcst_lag2"],
 "time_enc":["hour_local_sin","hour_local_cos","doy_sin","doy_cos","month"],
}
def fit_eval(cols):
    Xtr=X.loc[~X.__is_test,cols]; ytr=X.loc[~X.__is_test,"__y"]
    Xte=X.loc[X.__is_test,cols]; yte=X.loc[X.__is_test,"__y"].values
    qs=[]
    for t in TAU:
        m=lgb.LGBMRegressor(objective="quantile",alpha=t,num_leaves=31,n_estimators=500,
                            learning_rate=0.05,subsample=0.9,subsample_freq=1,verbose=-1,random_state=0)
        m.fit(Xtr,ytr); qs.append(m.predict(Xte))
    Q=np.column_stack(qs)
    mp=np.mean([np.maximum(t*(yte-Q[:,i]),(t-1)*(yte-Q[:,i])).mean() for i,t in enumerate(TAU)])
    rmse=np.sqrt(((Q[:,3]-yte)**2).mean())
    return mp,rmse
print("ablation ...")
base_mp,base_rmse=fit_eval(FN)
rows=[dict(group="FULL",n_feat=len(FN),mean_pinball=round(base_mp,3),rmse=round(base_rmse,2),
           d_pinball_pct=0.0,d_rmse_pct=0.0)]
for g,cols in GROUPS.items():
    keep=[c for c in FN if c not in cols]
    mp,rmse=fit_eval(keep)
    rows.append(dict(group=g,n_feat=len(keep),mean_pinball=round(mp,3),rmse=round(rmse,2),
                     d_pinball_pct=round(100*(mp-base_mp)/base_mp,2),
                     d_rmse_pct=round(100*(rmse-base_rmse)/base_rmse,2)))
abl=pd.DataFrame(rows); abl.to_csv(CODE_ROOT/"reports/04_error_analysis/ablation_feature_groups.csv",index=False)
print(abl.to_string(index=False))

# ---------- residual + grouped PIT (v1_lgbm & tcn) ----------
def load_v1():
    d=pd.read_csv(CODE_ROOT/"reports/03_modeling/v1_ml/full/quantile/ghi/per_lead_predictions.csv")
    d=d[d.model=="lgbm"].copy(); d["lead_time"]=d["unit"].map({"D+1":24,"D+2":48,"D+3":72})
    d["target_time_utc"]=pd.to_datetime(d["target_time_utc"]); return d
def load_tcn():
    d=pd.read_csv(CODE_ROOT/"reports/05_robustness/kt_sensitivity/fixedkt_top/v4_tcn/ghi/test_predictions.csv")
    d["target_time_utc"]=pd.to_datetime(d["target_time_utc"]); return d
sky=feat[["station_id","target_time_utc","lead_time","sky_type","cloud_cover_change"]].drop_duplicates(
    subset=["station_id","target_time_utc","lead_time"])
res_rows=[]; pit_rows=[]
for nm,loader in [("v1_lgbm",load_v1),("v4_tcn",load_tcn)]:
    d=loader().merge(sky,on=["station_id","target_time_utc","lead_time"],how="left")
    d=d[~d.sky_type.eq("night")].copy()
    r=(d["q0.5"]-d["y"]).values; pred=d["q0.5"].values
    het=np.corrcoef(np.abs(r),np.abs(pred))[0,1]
    res_rows.append(dict(model=nm,skew=round(stats.skew(r),2),kurtosis=round(stats.kurtosis(r),2),
                         hetero_corr_absPred=round(het,3)))
    for st in ["clear","partly","overcast"]:
        rr=r[(d.sky_type==st).values]
        res_rows.append(dict(model=nm,scope=st,mean_res=round(rr.mean(),1),mae=round(np.abs(rr).mean(),1)))
    # grouped PIT
    qd=d[QC].values; yv=d["y"].values
    pitv=(qd<yv[:,None]).sum(axis=1)/len(TAU)
    def ks(x): return stats.kstest(x,"uniform").pvalue
    pit_rows.append(dict(model=nm,scope="all",pit_mean=round(pitv.mean(),3),ks_p=round(ks(pitv),4),
                         frac_low=round((pitv<0.1).mean(),3),frac_high=round((pitv>0.9).mean(),3)))
    for L in [24,48,72]:
        m=(d.lead_time==L).values; pit_rows.append(dict(model=nm,scope=f"D+{L//24}",
            pit_mean=round(pitv[m].mean(),3),ks_p=round(ks(pitv[m]),4),
            frac_low=round((pitv[m]<0.1).mean(),3),frac_high=round((pitv[m]>0.9).mean(),3)))
    for st in ["clear","partly","overcast"]:
        m=(d.sky_type==st).values; pit_rows.append(dict(model=nm,scope=st,
            pit_mean=round(pitv[m].mean(),3),ks_p=round(ks(pitv[m]),4),
            frac_low=round((pitv[m]<0.1).mean(),3),frac_high=round((pitv[m]>0.9).mean(),3)))
    # residual vs cloud_cover_change
    cc=d["cloud_cover_change"].values; ok=np.isfinite(cc)
    bins=np.quantile(cc[ok],[0,.2,.4,.6,.8,1.0]); bi=np.clip(np.digitize(cc,bins[1:-1]),0,4)
    for b in range(5):
        m=ok&(bi==b)
        if m.sum()>50: res_rows.append(dict(model=nm,scope=f"cloudchg_q{b+1}",
            mean_res=round(r[m].mean(),1),mae=round(np.abs(r[m]).mean(),1)))
pd.DataFrame(res_rows).to_csv(CODE_ROOT/"reports/04_error_analysis/residual_diagnostics.csv",index=False)
pit=pd.DataFrame(pit_rows); pit.to_csv(CODE_ROOT/"reports/06_probability/pit_grouped.csv",index=False)
print("\nresidual:\n",pd.DataFrame(res_rows).to_string(index=False))
print("\nPIT grouped:\n",pit.to_string(index=False))

# ---------- FIG ablation ----------
fig,axes=plt.subplots(1,2,figsize=(7.2,2.8))
ax=axes[0]; a=abl.iloc[1:]
xcat(ax,a.group.tolist(),fs=5.5)
ax.bar(range(len(a)),a.d_pinball_pct,0.6,color=PALETTE["red_strong"],alpha=0.85,edgecolor="none")
ax.axhline(0,color="k",lw=0.8); ax.set_ylabel("Δ mean_pinball % (drop group)")
ax.set_xlabel("dropped feature group"); panel_label(ax,"a")
ax=axes[1]; xcat(ax,a.group.tolist(),fs=5.5)
ax.bar(range(len(a)),a.d_rmse_pct,0.6,color=PALETTE["blue_main"],alpha=0.85,edgecolor="none")
ax.axhline(0,color="k",lw=0.8); ax.set_ylabel("Δ RMSE %"); ax.set_xlabel("dropped feature group")
panel_label(ax,"b")
fig.tight_layout(pad=0.6); save_pub(fig,CODE_ROOT/"figs/04_error_analysis","fig_ablation_feature_groups")

# ---------- FIG grouped PIT ----------
fig,axes=plt.subplots(1,2,figsize=(7.2,2.8))
for ax,nm in zip(axes,["v1_lgbm","v4_tcn"]):
    sub=pit[pit.model==nm]
    cats=[c for c in sub.scope.tolist() if c!="all"]
    xcat(ax,cats,fs=6)
    ax.bar(range(len(sub)),sub.frac_low,0.4,label="PIT<0.1",color=PALETTE["blue_main"],edgecolor="none")
    ax.bar(np.arange(len(sub))+0.4,sub.frac_high,0.4,label="PIT>0.9",color=PALETTE["red_strong"],edgecolor="none")
    ax.axhline(0.1,color="k",ls="--",lw=0.8)
    ax.set_ylabel("tail mass (ideal=0.1)"); ax.set_xlabel("group"); ax.set_title(nm,fontsize=7)
    leg_above(ax,ncol=2)
fig.tight_layout(pad=0.6); save_pub(fig,CODE_ROOT/"figs/06_probability","fig_pit_grouped")
print("\nwrote ablation/residual/pit csv + fig_ablation_feature_groups + fig_pit_grouped")
print("DONE diagnostics")
