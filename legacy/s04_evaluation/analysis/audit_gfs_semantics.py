"""Audit local GFS/Himawari temporal semantics without altering source data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "project_manifest.yaml").exists())
REPORT_DIR = ROOT / "reports" / "01_data_audit" / "gfs_semantics"
OFFICIAL_SOURCES = {
    "previous_runs": "https://open-meteo.com/en/docs/previous-runs-api",
    "gfs": "https://open-meteo.com/en/docs/gfs-api",
    "satellite": "https://open-meteo.com/en/docs/satellite-radiation-api",
}


def load_sample(site: str, month: str):
    path = ROOT / "data" / "01_raw" / "01_gfs" / site / f"{site}_{month}.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def run(site="nanjing_1", month="2024-02"):
    path, raw = load_sample(site, month)
    cfg = yaml.safe_load((ROOT / "config" / "02_variables.yaml").read_text(encoding="utf-8"))
    time = pd.to_datetime(raw["hourly"]["time"], utc=True)
    step_hours = pd.Series(time).diff().dropna().dt.total_seconds().to_numpy() / 3600.0
    units = raw["hourly_units"]
    available = sorted(k for k in units if k != "time")
    expected_current = sorted(
        f"{v}_previous_day{lead}"
        for v in cfg["forecast_variables"]
        for lead in cfg["leads"]
    )
    missing_new = sorted(set(expected_current) - set(available))
    result = {
        "status": "provisional",
        "sample_file": str(path.relative_to(ROOT)),
        "requested_product": cfg["models"]["previous_runs"],
        "response_location": {
            "latitude": raw["latitude"], "longitude": raw["longitude"],
            "elevation": raw.get("elevation"), "timezone": raw.get("timezone"),
        },
        "timestamps": len(time),
        "strictly_hourly": bool(len(step_hours) and np.all(step_hours == 1)),
        "minimum_step_hours": float(step_hours.min()),
        "maximum_step_hours": float(step_hours.max()),
        "lead_semantics": {
            "previous_day1": "forecast made 24 h before valid time",
            "previous_day2": "forecast made 48 h before valid time",
            "previous_day3": "forecast made 72 h before valid time",
        },
        "surface_gfs_resolution": "0.11 degree (~13 km); hourly through 120 h",
        "radiation_semantics": "preceding-hour mean",
        "forecast_variable_count_in_sample": len(available) // len(cfg["leads"]),
        "forecast_variable_count_in_current_config": len(cfg["forecast_variables"]),
        "missing_new_protocol_fields": missing_new,
        "official_sources": OFFICIAL_SOURCES,
    }
    return result


def write_report(result):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# GFS temporal-semantics audit", "",
        "> Status: provisional until the three new cloud-level fields are refetched.", "",
        f"- Local sample: `{result['sample_file']}`.",
        f"- Timestamp cadence: **hourly={result['strictly_hourly']}** "
        f"({result['minimum_step_hours']:.0f}--{result['maximum_step_hours']:.0f} h).",
        "- `_previous_day1/2/3` are fixed 24/48/72-hour offsets from valid time.",
        "- GFS surface output is 0.11-degree and hourly through 120 hours; therefore D+1--D+3 are not 3-hourly values interpolated to hourly.",
        "- GHI/DHI/DNI and Himawari shortwave radiation are preceding-hour means; the project uses the same averaging convention.",
        f"- The archived sample contains {result['forecast_variable_count_in_sample']} forecast variables; the new config contains {result['forecast_variable_count_in_current_config']} after adding low/mid/high cloud.",
        f"- New fields absent from the archived sample: {len(result['missing_new_protocol_fields'])}; raw data must be refetched before formal retraining.",
        "", "## Official source URLs", "",
    ]
    lines.extend(f"- {name}: {url}" for name, url in result["official_sources"].items())
    (REPORT_DIR / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="nanjing_1")
    parser.add_argument("--month", default="2024-02")
    args = parser.parse_args()
    result = run(args.site, args.month)
    write_report(result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
