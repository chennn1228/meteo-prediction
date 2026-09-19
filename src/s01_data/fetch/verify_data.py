# -*- coding: utf-8 -*-
"""校验 raw 数据：字段完整性、null 率、关键值域（试点验收用）。

用法：
  python src/s01_data/fetch/verify_data.py --site nanjing --start 2024-01-01 --end 2024-02-29
  python src/s01_data/fetch/verify_data.py --site all --start 2024-01-01 --end 2025-12-31
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config" / "01_sites.yaml").exists())

# raw 数据源目录（docs/04 目录规范：01_gfs / 02_era5 / 03_satellite）
SOURCE_DIRS = {"previous_runs": "01_gfs", "era5": "02_era5", "satellite": "03_satellite"}


def iter_files(data_root, source, site, start, end):
    """按月份扫描 raw 下的 JSON 分片。"""
    cur = start.replace(day=1)
    while cur <= end:
        p = data_root / "01_raw" / SOURCE_DIRS[source] / site / f"{site}_{cur:%Y-%m}.json"
        if p.exists():
            yield p
        cur = (cur.replace(day=1) + dt.timedelta(days=32)).replace(day=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="all")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--sources", default="all",
                        choices=["all", "previous_runs", "satellite", "era5"])
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    import yaml
    cfg_dir = CODE_ROOT / "config"
    sites = yaml.safe_load((cfg_dir / "01_sites.yaml").read_text(encoding="utf-8"))["sites"]
    cfg = yaml.safe_load((cfg_dir / "02_variables.yaml").read_text(encoding="utf-8"))
    if args.site == "all":
        site_ids = [s["id"] for s in sites]
    else:
        site_ids = [args.site]
    sources = ["previous_runs", "satellite", "era5"] if args.sources == "all" else [args.sources]
    data_root = Path(args.data_dir) if args.data_dir else CODE_ROOT / "data"
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    var_map = {
        "previous_runs": [f"{v}_previous_day{n}" for v in cfg["forecast_variables"] for n in cfg["leads"]],
        "satellite": cfg["satellite_variables"],
        "era5": cfg["era5_variables"],
    }
    problems = 0
    for source in sources:
        for site in site_ids:
            files = list(iter_files(data_root, source, site, start, end))
            if not files:
                print(f"[缺失] {source:13s} {site:10s} 无任何文件")
                problems += 1
                continue
            for p in files:
                data = json.loads(p.read_text(encoding="utf-8"))
                hourly = data.get("hourly") or {}
                times = hourly.get("time") or []
                n = len(times)
                print(f"[{source:13s}] {site:10s} {p.name:12s} 时次={n:4d} "
                      f"网格=({data.get('latitude')}, {data.get('longitude')})")
                if n == 0:
                    problems += 1
                    continue
                for var in var_map[source]:
                    vals = hourly.get(var)
                    if vals is None:
                        print(f"    !! 缺字段 {var}")
                        problems += 1
                        continue
                    nulls = sum(1 for v in vals if v is None)
                    non_null = [v for v in vals if v is not None]
                    vmin = min(non_null) if non_null else None
                    vmax = max(non_null) if non_null else None
                    flag = "  << 全 null" if nulls == n else ""
                    print(f"    {var:45s} null={nulls:3d}/{n}{flag}  min={vmin} max={vmax}")
    print("---")
    print("存在问题" if problems else "全部通过")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
