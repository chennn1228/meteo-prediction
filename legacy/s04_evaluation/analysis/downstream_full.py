# -*- coding: utf-8 -*-
"""Downstream per-sample evaluation and formal figures.

Consumes per-sample predictions that carry station/time:
  - v1_ml (lgbm/xgb/raw) ghi  : reports/03_modeling/v1_ml/full/quantile/ghi/per_lead_predictions.csv
  - deep fixed-kt ghi         : reports/05_robustness/kt_sensitivity/fixedkt_top/{v4_tcn,v6_transformer,v7_autoformer,v8_informer}/ghi/test_predictions.csv
  - v1_ml cloud (for cloud-side diagnostics)

Produces:
  reports/06_probability/per_sample_diagnostics.csv   (daytime/segmented/conditional coverage/PIT-KS)
  figs/06_probability/00_comparison/fig_reliability_full   (Fig.6: reliability+PIT+sharpness)
  figs/04_error_analysis/fig_grouped_full                   (Fig.7: grouped MAE + conditional coverage)
  figs/03_modeling/00_comparison/fig_case_studies_full      (Fig.10)

QA style: legends above, both axis labels, 45-deg anchored ticks, filled marks.
"""
import sys
from pathlib import Path
CODE_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / "config" / "01_sites.yaml").exists())
sys.path.insert(0, str(CODE_ROOT / "src"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from s04_evaluation.analysis.plot_common import (
    apply_pub_style, PALETTE, COLORS, save_pub, panel_label)
apply_pub_style(font_size=8)

TAU = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
QC = [f"q{t:g}" for t in TAU]
R = CODE_ROOT / "reports"

def leg_above(ax, ncol=2, fs=6, y=1.08):
    ax.legend(fontsize=fs, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False)
def xcat(ax, labels, fs=6):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=fs)

def pinball_mean(y, q):
    return np.mean([np.maximum(t*(y-q[:,i]), (t-1)*(y-q[:,i])).mean() for i,t in enumerate(TAU)])
def cov(y,q,lo,hi): return float(((y>=q[:,lo])&(y<=q[:,hi])).mean())
def wid(q,lo,hi): return float((q[:,hi]-q[:,lo]).mean())
def reliability(y,q):
    return np.array([(t,(y<=q[:,i]).mean()) for i,t in enumerate(TAU)])
def pit(y,q):
    return (q < y[:,None]).sum(axis=1)/len(TAU)

# ---- featured for sky_type/segment join ----
feat = pd.concat([pd.read_parquet(p) for p in sorted(
    (CODE_ROOT/"data/03_featured").glob("*_featured_2024-02_2026-09.parquet"))], ignore_index=True)
feat["target_time_utc"] = pd.to_datetime(feat["target_time_utc"])
sky = feat[["station_id","target_time_utc","lead_time","sky_type","solar_elevation","month"]].drop_duplicates(
    subset=["station_id","target_time_utc","lead_time"])

# ---- load per-sample models ----
def load_v1(model):
    d = pd.read_csv(R/"03_modeling/v1_ml/full/quantile/ghi/per_lead_predictions.csv")
    d = d[d.model==model].copy()
    d["lead_time"] = d["unit"].map({"D+1":24,"D+2":48,"D+3":72})
    d["target_time_utc"] = pd.to_datetime(d["target_time_utc"])
    return d[["station_id","target_time_utc","lead_time","y"]+QC]
def load_deep(new):
    f = R/"05_robustness/kt_sensitivity/fixedkt_top"/new/"ghi/test_predictions.csv"
    d = pd.read_csv(f); d["target_time_utc"]=pd.to_datetime(d["target_time_utc"])
    return d[["station_id","target_time_utc","lead_time","y"]+QC]

models = {"v1_raw":load_v1("raw"), "v1_lgbm":load_v1("lgbm"), "v1_xgb":load_v1("xgb")}
for new in ["v4_tcn","v6_transformer","v7_autoformer","v8_informer"]:
    f = R/"05_robustness/kt_sensitivity/fixedkt_top"/new/"ghi/test_predictions.csv"
    if f.exists(): models[new.replace("v","V").replace("_"," ")] = load_deep(new)
print("per-sample models:", list(models))

# ---- diagnostics ----
EXPO_END = pd.Timestamp("2026-01-31 23:00", tz="UTC")
rows=[]
for name,d in models.items():
    d = d.merge(sky, on=["station_id","target_time_utc","lead_time"], how="left")
    y=d["y"].values; q=d[QC].values
    day = d["sky_type"].ne("night").values & ~d["sky_type"].isna().values
    dd,qq = y[day], q[day]
    p = pit(dd,qq); ks = stats.kstest(p,"uniform")
    seg = np.where(d["target_time_utc"]<=EXPO_END,"exposed","clean")
    for s in ["all","exposed","clean"]:
        m = np.ones(len(y),bool) if s=="all" else (seg==s)
        rows.append(dict(model=name,scope=s,n=int(m.sum()),
            mean_pinball=pinball_mean(y[m],q[m]),
            rmse=float(np.sqrt(((q[m,3]-y[m])**2).mean())),
            cov90=cov(y[m],q[m],0,6), width90=wid(q[m],0,6),
            ks_p=float(ks.pvalue) if s=="all" else np.nan))
    for st in ["clear","partly","overcast"]:
        m = (d["sky_type"]==st).values
        if m.sum(): rows.append(dict(model=name,scope=f"cov_{st}",n=int(m.sum()),
            cov90=cov(y[m],q[m],0,6), width90=wid(q[m],0,6)))
diag = pd.DataFrame(rows)
(R/"06_probability").mkdir(parents=True, exist_ok=True)
diag.to_csv(R/"06_probability/per_sample_diagnostics.csv", index=False)
print("wrote per_sample_diagnostics.csv")

# ---- Fig.6 reliability / PIT / sharpness ----
show = [k for k in ["v1_lgbm","v1_xgb","V4 tcn","V6 transformer","V7 autoformer","V8 informer"] if k in models]
fig,axes=plt.subplots(1,3,figsize=(7.2,2.6))
ax=axes[0]
for i,k in enumerate(show):
    d=models[k].merge(sky,on=["station_id","target_time_utc","lead_time"],how="left")
    day=~d.sky_type.eq("night").values & ~d.sky_type.isna().values
    rel=reliability(d["y"].values[day], d[QC].values[day])
    ax.plot(rel[:,0],rel[:,1],marker="o",ms=3,lw=1.2,color=COLORS[i%6],label=k.replace("V","v").replace(" ","_"))
ax.plot([0,1],[0,1],"k--",lw=0.8); ax.set_xlabel("nominal τ"); ax.set_ylabel("empirical P(y ≤ q̂_τ)")
leg_above(ax,ncol=2); panel_label(ax,"a")
ax=axes[1]
for i,k in enumerate(show):
    d=models[k].merge(sky,on=["station_id","target_time_utc","lead_time"],how="left")
    day=~d.sky_type.eq("night").values & ~d.sky_type.isna().values
    ax.hist(pit(d["y"].values[day],d[QC].values[day]),bins=10,range=(0,1),histtype="bar",
            alpha=0.5,edgecolor="none",color=COLORS[i%6],label=k.replace("V","v").replace(" ","_"))
ax.set_xlabel("PIT bin"); ax.set_ylabel("count"); leg_above(ax,ncol=2); panel_label(ax,"b")
ax=axes[2]
data=[]
for k in show:
    d=models[k].merge(sky,on=["station_id","target_time_utc","lead_time"],how="left")
    day=~d.sky_type.eq("night").values & ~d.sky_type.isna().values
    data.append(d["q0.95"].values[day]-d["q0.05"].values[day])
bp=ax.boxplot(data,tick_labels=[k.replace("V","v").replace(" ","_") for k in show],patch_artist=True,
              showfliers=False,widths=0.6,medianprops=dict(color="black",lw=1))
for p,c in zip(bp["boxes"],COLORS[:len(show)]): p.set_facecolor(c); p.set_alpha(0.75)
ax.set_ylabel("q95 − q05 width"); ax.set_xlabel("model"); ax.tick_params(axis="x",labelsize=6,rotation=45)
panel_label(ax,"c")
fig.tight_layout(pad=0.6)
save_pub(fig, CODE_ROOT/"figs/06_probability/00_comparison","fig_reliability_full")

# ---- Fig.7 grouped MAE + conditional coverage ----
fig,axes=plt.subplots(1,3,figsize=(7.2,2.6))
k0="v1_lgbm"
d0=models[k0].merge(sky,on=["station_id","target_time_utc","lead_time"],how="left")
d0=d0[~d0.sky_type.eq("night")]; d0["err"]=(d0["q0.5"]-d0["y"]).abs()
specs=[("month","season","season"),("sky_type","sky","weather type")]
mo2seas={12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",
         6:"summer",7:"summer",8:"summer",9:"autumn",10:"autumn",11:"autumn"}
d0["season"]=d0["month"].map(mo2seas)
ax=axes[0]
cats=["spring","summer","autumn","winter"]
for i,lt in enumerate([24,48,72]):
    vals=[d0.loc[(d0.season==c)&(d0.lead_time==lt),"err"].mean() for c in cats]
    ax.bar(np.arange(4)+i*0.26,vals,0.26,label=f"D+{lt//24}",color=COLORS[i],alpha=0.9,edgecolor="none")
xcat(ax,[c.capitalize() for c in cats]); ax.set_ylabel("MAE (W m$^{-2}$)"); ax.set_xlabel("season")
leg_above(ax,ncol=3); panel_label(ax,"a")
ax=axes[1]
cats=["clear","partly","overcast"]
for i,lt in enumerate([24,48,72]):
    vals=[d0.loc[(d0.sky_type==c)&(d0.lead_time==lt),"err"].mean() for c in cats]
    ax.bar(np.arange(3)+i*0.26,vals,0.26,label=f"D+{lt//24}",color=COLORS[i],alpha=0.9,edgecolor="none")
xcat(ax,[c.capitalize() for c in cats]); ax.set_ylabel("MAE"); ax.set_xlabel("weather type")
leg_above(ax,ncol=3); panel_label(ax,"b")
ax=axes[2]
# conditional coverage_90 by sky_type across models
ms=[k for k in show]
cc=[]
for k in ms:
    d=models[k].merge(sky,on=["station_id","target_time_utc","lead_time"],how="left")
    cc.append([cov(d["y"].values[(d.sky_type==s).values],d[QC].values[(d.sky_type==s).values],0,6)
               for s in ["clear","partly","overcast"]])
cc=np.array(cc); x=np.arange(3); w=0.8/max(1,len(ms))
for i,k in enumerate(ms):
    ax.bar(x-(len(ms)/2-i)*w, cc[i], w, label=k.replace("V","v").replace(" ","_"), color=COLORS[i%6],alpha=0.85,edgecolor="none")
ax.axhline(0.9,color=PALETTE["red_strong"],ls="--",lw=0.8)
xcat(ax,[c.capitalize() for c in ["clear","partly","overcast"]]); ax.set_ylabel("coverage 90%"); ax.set_xlabel("weather type")
leg_above(ax,ncol=3,y=1.12); panel_label(ax,"c")
fig.tight_layout(pad=0.6)
save_pub(fig, CODE_ROOT/"figs/04_error_analysis","fig_grouped_full")

# ---- Fig.10 case studies (v1_lgbm) ----
site="nanjing_1"
ds=models["v1_lgbm"].merge(sky[sky.station_id==site],on=["station_id","target_time_utc","lead_time"],how="left")
ds["date"]=ds.target_time_utc.dt.date
dm=ds[ds.lead_time==24].groupby("date")["sky_type"].agg(lambda s:s[s!="night"].mode().iat[0] if (s!="night").any() else "night")
pick={wt:dm[dm==wt].index[len(dm[dm==wt])//2] for wt in ["clear","partly","overcast"] if (dm==wt).any()}
fig,axes=plt.subplots(1,3,figsize=(7.4,2.7))
for ax,(wt,dt) in zip(axes,pick.items()):
    sub=ds[(ds.date==dt)&(ds.lead_time==24)].sort_values("target_time_utc"); t=sub.target_time_utc
    ax.fill_between(t,sub["q0.05"],sub["q0.95"],color=PALETTE["blue_secondary"],alpha=0.22,label="q05–q95")
    ax.fill_between(t,sub["q0.25"],sub["q0.75"],color=PALETTE["blue_main"],alpha=0.4,label="q25–q75")
    ax.plot(t,sub["q0.5"],color=PALETTE["blue_main"],lw=1.2,label="q50")
    ax.plot(t,sub["y"],color=PALETTE["red_strong"],lw=1.0,label="obs")
    ax.set_title(f"{wt} | {dt} | {site}",fontsize=7); ax.set_ylabel("GHI (W m$^{-2}$)"); ax.set_xlabel("hour (UTC)")
    ax.tick_params(labelsize=6); ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
leg_above(axes[0],ncol=2,y=1.16)
for i,ax in enumerate(axes): panel_label(ax,"abc"[i])
fig.tight_layout(pad=0.6)
save_pub(fig, CODE_ROOT/"figs/03_modeling/00_comparison","fig_case_studies_full")

print("wrote figures Fig6/7/10 full")
print("DONE downstream_full")
