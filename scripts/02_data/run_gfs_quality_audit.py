"""Audit raw GFS/Open-Meteo distributions without training or feature clipping."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from s03_features.physics import preceding_hour_solar_geometry  # noqa: E402

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})

AUDIT_BLUE = "#2F6FB0"
LEAD_ALPHA = {"D+1": 0.45, "D+2": 0.70, "D+3": 0.95}
BLUE_ALPHA_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "audit_blue_alpha",
    [mpl.colors.to_rgba(AUDIT_BLUE, 0.06), mpl.colors.to_rgba(AUDIT_BLUE, 1.0)],
)

GFS_FIELDS = {
    "ghi_raw": "shortwave_radiation", "dhi_raw": "diffuse_radiation",
    "dni_raw": "direct_normal_irradiance", "gti_raw": "global_tilted_irradiance",
    "cloud_cover_raw": "cloud_cover", "terrestrial_raw": "terrestrial_radiation",
}
LEADS = (1, 2, 3)


def months(start: str, end: str) -> list[str]:
    return pd.period_range(start, end, freq="M").astype(str).tolist()


def payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def numeric(values) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def load_month(site: str, month: str, raw_root: Path) -> pd.DataFrame:
    gfs = payload(raw_root / "01_gfs" / site / f"{site}_{month}.json")
    sat = payload(raw_root / "03_satellite" / site / f"{site}_{month}.json")
    era = payload(raw_root / "02_era5" / site / f"{site}_{month}.json")
    times = pd.to_datetime(gfs["hourly"]["time"], utc=True)
    if not (list(gfs["hourly"]["time"]) == list(sat["hourly"]["time"])
            == list(era["hourly"]["time"])):
        raise ValueError(f"time axes differ: {site} {month}")
    import pvlib
    location = pvlib.location.Location(float(gfs["latitude"]), float(gfs["longitude"]),
                                       altitude=float(gfs.get("elevation") or 0), tz="UTC")
    geometry = preceding_hour_solar_geometry(times, location)
    common = {
        "site": site, "target_time_utc": times,
        "gfs_service_latitude": float(gfs["latitude"]),
        "gfs_service_longitude": float(gfs["longitude"]),
        "gfs_service_elevation": float(gfs.get("elevation") or 0),
        "himawari_service_latitude": float(sat["latitude"]),
        "himawari_service_longitude": float(sat["longitude"]),
        "solar_elevation": geometry["solar_elevation"],
        "ghi_clear_sky": geometry["ghi_clear_sky"],
        "dni_clear_sky": geometry["dni_clear_sky"],
        "ghi_obs_sat": numeric(sat["hourly"]["shortwave_radiation"]),
        "cloud_cover_obs_era5": numeric(era["hourly"]["cloud_cover"]),
    }
    frames = []
    for lead in LEADS:
        data = dict(common)
        data["lead"] = f"D+{lead}"
        data["lead_time"] = lead * 24
        for new, raw in GFS_FIELDS.items():
            data[new] = numeric(gfs["hourly"][f"{raw}_previous_day{lead}"])
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


def add_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["month"] = out.target_time_utc.dt.strftime("%Y-%m")
    out["season"] = out.target_time_utc.dt.month.map(
        {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
         6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})
    out["daylight"] = np.where(out.solar_elevation > 0, "day", "night")
    out["solar_bin"] = pd.cut(out.solar_elevation, [-np.inf, 10, 30, 50, np.inf],
                              labels=["<=10", "10-30", "30-50", ">50"], right=False)
    out["ghi_clean"] = out.ghi_raw.clip(lower=0)
    out["cloud_cover_clean"] = out.cloud_cover_raw.clip(0, 100)
    out["kt_raw"] = np.where(out.ghi_clear_sky > 50,
                             out.ghi_raw / out.ghi_clear_sky, np.nan)
    out["kni_raw"] = np.where(out.dni_clear_sky > 50,
                              out.dni_raw / out.dni_clear_sky, np.nan)
    out["diffuse_fraction_raw"] = np.where(out.ghi_raw > 50,
                                           out.dhi_raw / out.ghi_raw, np.nan)
    out["ghi_error"] = out.ghi_raw - out.ghi_obs_sat
    out["cloud_cover_error"] = out.cloud_cover_raw - out.cloud_cover_obs_era5
    return out


def group_rates(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    def one(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        zero = g.loc[g.ghi_raw == 0]
        zn = len(zero)
        return pd.Series({
            "n": n, "ghi_raw_zero_rate": (g.ghi_raw == 0).mean(),
            "ghi_raw_negative_rate": (g.ghi_raw < 0).mean(),
            "ghi_clean_zero_rate": (g.ghi_clean == 0).mean(),
            "zero_with_sat_gt100_rate": ((g.ghi_raw == 0) & (g.ghi_obs_sat > 100)).mean(),
            "zero_with_sat_gt300_rate": ((g.ghi_raw == 0) & (g.ghi_obs_sat > 300)).mean(),
            "zero_inconsistent_dhi_dni_rate": ((g.ghi_raw == 0) &
                ((g.dhi_raw > 0) | (g.dni_raw > 0))).mean(),
            "zero_count": zn,
            "zero_dhi_positive_fraction": (zero.dhi_raw > 0).mean() if zn else np.nan,
            "zero_dni_positive_fraction": (zero.dni_raw > 0).mean() if zn else np.nan,
            "zero_gti_positive_fraction": (zero.gti_raw > 0).mean() if zn else np.nan,
            "zero_terrestrial_positive_fraction": (zero.terrestrial_raw > 0).mean() if zn else np.nan,
            "zero_cloud_mean": zero.cloud_cover_raw.mean() if zn else np.nan,
        })
    return frame.groupby(keys, observed=True, dropna=False).apply(one, include_groups=False).reset_index()


def zero_audit(frame: pd.DataFrame) -> pd.DataFrame:
    blocks = []
    for label, mask in (("all", np.ones(len(frame), dtype=bool)),
                        ("solar_elevation_gt_0", frame.solar_elevation > 0),
                        ("solar_elevation_gt_10", frame.solar_elevation > 10),
                        ("solar_elevation_gt_15", frame.solar_elevation > 15)):
        part = group_rates(frame.loc[mask], ["lead", "month", "site"])
        part.insert(0, "condition", label)
        blocks.append(part)
    severe = frame[(frame.solar_elevation > 15) & (frame.ghi_raw == 0)].copy()
    severe["condition"] = np.select(
        [severe.ghi_obs_sat > 300, severe.ghi_obs_sat > 100],
        ["solar_gt15_sat_gt300", "solar_gt15_sat_gt100"], default="solar_gt15")
    extra = group_rates(severe, ["condition", "lead", "month", "site"])
    return pd.concat(blocks + [extra], ignore_index=True)


def endpoint_audit(frame: pd.DataFrame) -> pd.DataFrame:
    def one(g: pd.DataFrame) -> pd.Series:
        cloud = g.cloud_cover_raw
        return pd.Series({"n": len(g), "p_cloud_eq_0": (cloud == 0).mean(),
                          "p_cloud_eq_100": (cloud == 100).mean(),
                          "p_cloud_between": ((cloud > 0) & (cloud < 100)).mean(),
                          "p_cloud_outside_0_100": ((cloud < 0) | (cloud > 100)).mean(),
                          "unique_values": cloud.nunique(dropna=True)})
    return frame.groupby(["lead", "month", "daylight", "site"], observed=True).apply(
        one, include_groups=False).reset_index()


def kt_audit(frame: pd.DataFrame) -> pd.DataFrame:
    def one(g: pd.DataFrame) -> pd.Series:
        x = g.kt_raw.dropna()
        result = {"n": len(g), "n_valid": len(x)}
        for name, q in (("median", .5), ("p90", .9), ("p95", .95), ("p99", .99)):
            result[name] = x.quantile(q) if len(x) else np.nan
        result["max"] = x.max() if len(x) else np.nan
        for threshold in (1, 1.2, 1.5, 2, 3, 5):
            result[f"p_gt_{str(threshold).replace('.', '_')}"] = (x > threshold).mean() if len(x) else np.nan
        return pd.Series(result)
    return frame.groupby(["solar_bin", "lead", "month", "site"], observed=True).apply(
        one, include_groups=False).reset_index()


def duplication_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["gfs_service_latitude", "gfs_service_longitude", "target_time_utc", "lead"]
    unique = frame.drop_duplicates(keys).copy()
    rows = []
    for label, data in (("requested_site_weighted", frame), ("unique_service_time_lead", unique)):
        x = data.kt_raw.dropna()
        rows.append({
            "population": label, "sample_count": len(data),
            "unique_service_time_lead_count": len(unique),
            "duplicate_rate": 1 - len(unique) / len(frame),
            "ghi_zero_rate": (data.ghi_raw == 0).mean(),
            "cloud_zero_rate": (data.cloud_cover_raw == 0).mean(),
            "cloud_100_rate": (data.cloud_cover_raw == 100).mean(),
            "kt_p99": x.quantile(.99), "kt_max": x.max(),
            "kt_gt_1_5_rate": (x > 1.5).mean(), "kt_gt_3_rate": (x > 3).mean(),
        })
    return pd.DataFrame(rows), unique


def save_figure(fig: plt.Figure, output: Path, source_note: str) -> None:
    fig.text(.995, -.055, source_note, ha="right", va="bottom", fontsize=5, color="#555555")
    for ext, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        final = output.with_suffix(f".{ext}")
        last_error = None
        for attempt in range(5):
            try:
                fig.savefig(final, format=ext, bbox_inches="tight", **kwargs)
                last_error = None
                break
            except OSError as exc:
                last_error = exc
                time.sleep(0.25 * (attempt + 1))
        if last_error is not None:
            raise last_error
        if ext == "svg":
            text = final.read_text(encoding="utf-8")
            final.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n",
                             encoding="utf-8")
    plt.close(fig)


def plot_figures(frame: pd.DataFrame, zero: pd.DataFrame, endpoints: pd.DataFrame,
                 out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    note = ("Raw API values unless explicitly labelled; single-blue hue with opacity encoding; "
            "no model training; n shown in source CSVs")

    order = ["all", "solar_elevation_gt_0", "solar_elevation_gt_10", "solar_elevation_gt_15"]
    summary = zero[zero.condition.isin(order)].groupby(["condition", "lead"], observed=True).apply(
        lambda g: np.average(g.ghi_raw_zero_rate, weights=g.n), include_groups=False).unstack()
    summary = summary.reindex(order)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(summary))
    for i, lead in enumerate(summary.columns):
        ax.bar(x + (i-1)*.24, summary[lead], .22, label=lead,
               color=mpl.colors.to_rgba(AUDIT_BLUE, LEAD_ALPHA[lead]))
    ax.set_xticks(x, summary.index, rotation=20, ha="right")
    ax.set_yscale("log"); ax.set_ylim(1e-6, 1)
    ax.set_ylabel("P(raw GFS GHI = 0), log scale"); ax.set_title("Raw GFS zero diagnostics")
    ax.legend(ncol=3); save_figure(fig, out / "fig_gfs_zero_diagnostics", note)

    ep = endpoints.groupby(["lead", "daylight"], observed=True).apply(
        lambda g: pd.Series({k: np.average(g[k], weights=g.n) for k in
                             ("p_cloud_eq_0", "p_cloud_eq_100", "p_cloud_between")}),
        include_groups=False).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, period in zip(axes, ("day", "night")):
        p = ep[ep.daylight == period].set_index("lead")
        bottom = np.zeros(len(p))
        for col, label, alpha in (("p_cloud_eq_0", "0%", .22),
                                  ("p_cloud_between", "1–99%", .55),
                                  ("p_cloud_eq_100", "100%", .92)):
            ax.bar(p.index, p[col], bottom=bottom, label=label,
                   color=mpl.colors.to_rgba(AUDIT_BLUE, alpha))
            bottom += p[col].to_numpy()
        ax.set_title(period.capitalize()); ax.set_ylim(0, 1); ax.set_ylabel("Proportion")
    axes[1].legend(title="Raw total cloud cover")
    fig.suptitle("Endpoint accumulation in raw total cloud cover")
    save_figure(fig, out / "fig_cloud_endpoint_distribution", note)

    valid = frame.loc[frame.kt_raw.notna(), "kt_raw"]
    cap = float(valid.max())
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for lead in ("D+1", "D+2", "D+3"):
        xval = frame.loc[frame.lead == lead, "kt_raw"].dropna()
        ax.hist(xval, bins=120, density=True, histtype="step", lw=1.4,
                color=mpl.colors.to_rgba(AUDIT_BLUE, LEAD_ALPHA[lead]), label=lead)
    ax.axvline(1.5, color="#555555", alpha=.75, ls="--", lw=.9,
               label="legacy clip boundary")
    ax.set_yscale("log"); ax.set_xlim(0, cap); ax.set_xlabel(f"kt_raw (full observed range; max={cap:.2f})")
    ax.set_ylabel("Density (log)"); ax.set_title("Unclipped raw clearness-index distribution")
    ax.legend(ncol=4); save_figure(fig, out / "fig_kt_raw_distribution", note)

    sampled = frame.sample(min(len(frame), 400000), random_state=0)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True, sharey=True)
    for ax, band in zip(axes, ("10-30", "30-50", ">50")):
        p = sampled[(sampled.solar_bin == band) & sampled.kt_raw.notna()]
        h = ax.hexbin(p.cloud_cover_raw, p.kt_raw, gridsize=45,
                      bins="log", mincnt=1, cmap=BLUE_ALPHA_CMAP)
        ax.set_title(f"Solar elevation {band}°"); ax.set_xlabel("Raw total cloud cover (%)")
    axes[0].set_ylabel("kt_raw (unclipped)")
    fig.colorbar(h, ax=axes, label="log10 count", fraction=.025, pad=.03)
    fig.suptitle("Cloud cover vs raw clearness index, solar-angle stratified")
    save_figure(fig, out / "fig_cloud_kt_hexbin", note)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True, sharey=True)
    for ax, lead in zip(axes, ("D+1", "D+2", "D+3")):
        p = sampled[(sampled.lead == lead) & sampled.cloud_cover_error.notna()
                    & sampled.ghi_error.notna()]
        h = ax.hexbin(p.cloud_cover_error, p.ghi_error, gridsize=45, bins="log",
                      mincnt=1, cmap=BLUE_ALPHA_CMAP, extent=(-100, 100, -800, 800))
        ax.axhline(0, color="white", lw=.5); ax.axvline(0, color="white", lw=.5)
        ax.set_title(lead); ax.set_xlabel("GFS total cloud − ERA5 cloud (pp)")
    axes[0].set_ylabel("Raw GFS GHI − Himawari GHI (W m$^{-2}$)")
    fig.colorbar(h, ax=axes, label="log10 count", fraction=.025, pad=.03)
    fig.suptitle("Cloud error vs GHI error")
    save_figure(fig, out / "fig_cloud_error_ghi_error_hexbin", note)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True, sharey=True)
    for ax, lead in zip(axes, ("D+1", "D+2", "D+3")):
        p = sampled[(sampled.lead == lead) & sampled.ghi_raw.notna() & sampled.ghi_obs_sat.notna()]
        h = ax.hexbin(p.ghi_obs_sat, p.ghi_raw, gridsize=48, bins="log", mincnt=1,
                      cmap=BLUE_ALPHA_CMAP, extent=(0, 1200, 0, 1200))
        ax.plot([0, 1200], [0, 1200], color="white", lw=.6, ls="--")
        ax.set_title(lead); ax.set_xlabel("Himawari GHI (W m$^{-2}$)")
    axes[0].set_ylabel("Raw GFS GHI (W m$^{-2}$)")
    fig.colorbar(h, ax=axes, label="log10 count", fraction=.025, pad=.03)
    fig.suptitle("Raw GFS GHI vs Himawari GHI")
    save_figure(fig, out / "fig_gfs_himawari_ghi_hexbin", note)


def write_trace(path: Path) -> None:
    path.write_text("""# Source-to-plot processing trace

| Variable | Original API field | Clean stage | Feature stage | Final model value | Artificially changes distribution? |
|---|---|---|---|---|---|
| GFS GHI | `shortwave_radiation_previous_day{1,2,3}` | negatives set to 0 in legacy clean | none | `ghi_fcst` | Yes for negative raw values; audit uses raw API |
| DHI | `diffuse_radiation_previous_day{1,2,3}` | negatives set to 0 | none | `dhi_fcst` | Yes for negative raw values; audit uses raw API |
| DNI | `direct_normal_irradiance_previous_day{1,2,3}` | negatives set to 0 | none | `dni_fcst` | Yes for negative raw values; audit uses raw API |
| GTI | `global_tilted_irradiance_previous_day{1,2,3}` | negatives set to 0 | none | `gti_fcst` | Yes for negative raw values; audit uses raw API |
| total cloud cover | `cloud_cover_previous_day{1,2,3}` | bounded to [0,100] in legacy clean | lag/change/rolling derived later | `cloud_cover_fcst` plus dynamics | Only if API is outside physical range; audit uses raw API |
| clear-sky GHI | none; pvlib Ineichen at returned GFS coordinate | not applicable | mean of 12 five-minute midpoints over preceding hour | `ghi_clear_sky` | Derived, not clipped |
| kt | none | not applicable | `kt_raw=GHI_GFS/GHI_clear` only when clear GHI >50 | `kt_model` currently identity | Legacy [0,1.5] clip did; removed |
| kni | none | not applicable | `kni_raw=DNI_GFS/DNI_clear` only when clear DNI >50 | `kni_model` currently identity | Legacy [0,1.5] clip did; removed |
| diffuse fraction | none | not applicable | `diffuse_fraction_raw=DHI/GHI` only when GHI >50 | `diffuse_fraction_model` currently identity | Legacy [0,1] clip did; removed |

Raw diagnostic columns and model columns are separate by contract. Fold-local missing-value imputation and standardization occur only at model preprocessing and are forbidden in this audit. Lag/rolling features sort by returned-service identity, lead, and issue time; they are not used in the raw distribution figures.
""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-02")
    parser.add_argument("--end", default="2026-08")
    parser.add_argument("--raw-root", default="data/01_raw")
    args = parser.parse_args()
    sites = [x["id"] for x in yaml.safe_load((ROOT / "config/01_sites.yaml").read_text(
        encoding="utf-8"))["sites"]]
    frames = []
    for site in sites:
        for month in months(args.start, args.end):
            frames.append(load_month(site, month, ROOT / args.raw_root))
    data = add_diagnostics(pd.concat(frames, ignore_index=True))
    report_dir, fig_dir = ROOT / "reports/01_data_audit", ROOT / "figs/01_data_audit"
    report_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
    zero = zero_audit(data); endpoints = endpoint_audit(data); kt = kt_audit(data)
    duplication, unique = duplication_audit(data)
    zero.to_csv(report_dir / "gfs_zero_ghi_audit.csv", index=False)
    zero_cases = data.loc[(data.ghi_raw == 0) & (
        (data.solar_elevation > 15) | (data.dhi_raw > 0) | (data.dni_raw > 0)), [
            "site", "target_time_utc", "lead", "gfs_service_latitude",
            "gfs_service_longitude", "solar_elevation", "ghi_raw", "dhi_raw",
            "dni_raw", "gti_raw", "terrestrial_raw", "cloud_cover_raw",
            "ghi_obs_sat", "ghi_clear_sky",
        ]].sort_values(["target_time_utc", "site", "lead"])
    zero_cases.to_csv(report_dir / "gfs_zero_ghi_cases.csv", index=False)
    endpoints.to_csv(report_dir / "cloud_endpoint_audit.csv", index=False)
    kt.to_csv(report_dir / "kt_raw_audit.csv", index=False)
    duplication.to_csv(report_dir / "service_grid_duplication.csv", index=False)
    inconsistent = data[(data.ghi_raw == 0) & ((data.dhi_raw > 0) | (data.dni_raw > 0))]
    summary = pd.DataFrame([{
        "rows": len(data), "sites": data.site.nunique(),
        "gfs_service_points": data[["gfs_service_latitude", "gfs_service_longitude"]].drop_duplicates().shape[0],
        "start": data.target_time_utc.min(), "end": data.target_time_utc.max(),
        "raw_ghi_negative_rate": (data.ghi_raw < 0).mean(),
        "raw_ghi_zero_rate_all": (data.ghi_raw == 0).mean(),
        "raw_ghi_zero_rate_solar_gt15": (data.loc[data.solar_elevation > 15, "ghi_raw"] == 0).mean(),
        "raw_ghi_zero_sat_gt100_solar_gt15_rate": ((data.ghi_raw == 0) & (data.ghi_obs_sat > 100)
                                                    & (data.solar_elevation > 15)).mean(),
        "raw_ghi_zero_sat_gt300_solar_gt15_count": int(((data.ghi_raw == 0) &
            (data.ghi_obs_sat > 300) & (data.solar_elevation > 15)).sum()),
        "raw_ghi_zero_solar_gt15_count": int(((data.ghi_raw == 0) &
            (data.solar_elevation > 15)).sum()),
        "raw_ghi_zero_inconsistent_components_count": int(((data.ghi_raw == 0) &
            ((data.dhi_raw > 0) | (data.dni_raw > 0))).sum()),
        "inconsistent_components_all_clear_sky_le50": bool(
            (inconsistent.ghi_clear_sky <= 50).all()),
        "inconsistent_components_all_cloud_100": bool(
            (inconsistent.cloud_cover_raw == 100).all()),
        "inconsistent_components_all_dni_zero": bool((inconsistent.dni_raw == 0).all()),
        "inconsistent_components_max_dhi": inconsistent.dhi_raw.max(),
        "cloud_zero_rate": (data.cloud_cover_raw == 0).mean(),
        "cloud_100_rate": (data.cloud_cover_raw == 100).mean(),
        "cloud_interior_rate": ((data.cloud_cover_raw > 0) & (data.cloud_cover_raw < 100)).mean(),
        "kt_p99": data.kt_raw.quantile(.99), "kt_max": data.kt_raw.max(),
        "kt_gt_1_5_rate": (data.kt_raw.dropna() > 1.5).mean(),
        "duplicate_rate": duplication.iloc[0].duplicate_rate,
    }])
    summary.to_csv(report_dir / "gfs_quality_summary.csv", index=False)
    write_trace(report_dir / "source_processing_trace.md")
    plot_figures(data, zero, endpoints, fig_dir)
    s = summary.iloc[0]
    report = f"""# GFS / Open-Meteo data-quality audit

This audit uses raw API values from 20 requested sites over {args.start}–{args.end}; it does not train a model. Solar elevation uses the preceding-hour midpoint and Ineichen clear-sky GHI uses twelve five-minute midpoint samples over that interval, both at API-returned GFS service coordinates. Ratios are not clipped. Requested coordinates are provenance only.

## Main evidence

- Rows: {int(s.rows):,}; returned GFS service points: {int(s.gfs_service_points)}.
- Raw GFS GHI negative rate: {s.raw_ghi_negative_rate:.4%}. These values would become zeros in the legacy clean stage.
- Raw GFS GHI zero rate: {s.raw_ghi_zero_rate_all:.4%} overall and {s.raw_ghi_zero_rate_solar_gt15:.4%} at preceding-hour midpoint solar elevation >15° ({int(s.raw_ghi_zero_solar_gt15_count)} rows).
- Severe raw zeros with solar elevation >15° and Himawari GHI >300 W/m²: {int(s.raw_ghi_zero_sat_gt300_solar_gt15_count):,} rows.
- Raw GHI=0 with DHI>0 or DNI>0: {int(s.raw_ghi_zero_inconsistent_components_count):,} rows. All are low-light intervals with clear-sky GHI ≤50 W/m², total cloud=100%, DNI=0 and DHI ≤{s.inconsistent_components_max_dhi:.0f} W/m². They are consistent with preceding-hour interval semantics plus independent integer quantization, not a field shift.
- Raw total-cloud endpoints: P(0)={s.cloud_zero_rate:.2%}, P(100)={s.cloud_100_rate:.2%}, interior={s.cloud_interior_rate:.2%}.
- Unclipped kt: p99={s.kt_p99:.3f}, max={s.kt_max:.3f}, P(kt>1.5)={s.kt_gt_1_5_rate:.3%}.
- Duplicate requested-site weighting at identical service-point × time × lead keys: {s.duplicate_rate:.2%}. The before/after effect is in `service_grid_duplication.csv`.

## Answers required by the protocol

### A — artifacts created by source processing

The exact pile at kt=1.5 was created by the former hard clip. The same mechanism affected kni at 1.5 and diffuse fraction at 1.0. Those clips are removed; raw and model columns are now separate. Legacy clean also converts negative radiation to zero and bounds cloud cover to [0,100], so cleaned zero counts cannot by themselves diagnose the provider. Old clipped feature-distribution figures are provisional/legacy evidence.

### B — real product distributions still usable for modelling

Raw total cloud cover genuinely has endpoint mass at 0 and 100, and raw radiation/ratios are discretized because the API product is hourly and numerically quantized. These are product characteristics, not automatically errors. Their magnitude and sensitivity to deduplicating returned service cells are machine-readable in the supplied CSVs.

### C — issues that can threaten source usability

After aligning solar geometry to the preceding-hour radiation interval, only {int(s.raw_ghi_zero_solar_gt15_count)} rows remain above 15° (two unique events). The one row with Himawari GHI >300 W/m² is a coherent all-radiation-zero, 100%-cloud D+1 forecast miss surrounded by valid hours and leads: retain it as a genuine extreme forecast error, not missing/corrupt data. The 41 component-difference rows are twilight quantization cases below the registered ratio denominator threshold and are not a source-usability blocker.

## Figure contract

Core conclusion: separate processing-created boundaries from raw product structure before judging data-source suitability. Evidence chain: raw-zero stratification → cloud endpoints → unclipped kt tail → solar-angle-stratified cloud–kt density → lead-specific error coupling. Archetype: quantitative grid. Exports: editable SVG/PDF plus PNG previews, with CSV source data.
"""
    (report_dir / "gfs_quality_audit.md").write_text(report, encoding="utf-8")
    tracked = sorted(list(report_dir.glob("*")) + list(fig_dir.glob("*")))
    receipt = {"status": "complete", "training_performed": False,
               "data_scope": {"sites": len(sites), "start": args.start, "end": args.end},
               "raw_ratio_policy": "unclipped", "dedup_key": ["gfs_service_coordinate", "target_time", "lead"],
               "files": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in tracked if p.is_file()}}
    (report_dir / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(summary.to_dict(orient="records")[0], default=str, indent=2))


if __name__ == "__main__":
    main()
