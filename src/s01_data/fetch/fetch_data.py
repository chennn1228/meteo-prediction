# -*- coding: utf-8 -*-
"""Open-Meteo 三源数据抓取（可复用 CLI）。

按站点 × 月份分片抓取，断点续传（已存在的文件自动跳过），
每次请求记录到 data/01_raw/fetch_log.csv（含网格回读坐标）。

用法示例：
  python src/s01_data/fetch/fetch_data.py --site nanjing --start 2024-01-01 --end 2024-02-29
  python src/s01_data/fetch/fetch_data.py --site all --start 2024-01-01 --end 2025-12-31 --workers 4
  python src/s01_data/fetch/fetch_data.py --site all --start 2024-01-01 --end 2025-12-31 --sources previous_runs --force
  python src/s01_data/fetch/fetch_data.py --site nanjing --start 2024-01-01 --end 2025-12-31 --dry-run
"""
import argparse
import calendar
import csv
import datetime as dt
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

# 把项目根目录加入模块搜索路径，使 src.fetch.fetch_client 可导入
CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config" / "01_sites.yaml").exists())
sys.path.insert(0, str(CODE_ROOT))

from src.s01_data.fetch.fetch_client import fetch_json  # noqa: E402
from src.s02_data.cache_contract import cache_metadata, validate_raw_payload  # noqa: E402

# raw 数据源目录（docs/04 目录规范：01_gfs / 02_era5 / 03_satellite）
SOURCE_DIRS = {"previous_runs": "01_gfs", "era5": "02_era5", "satellite": "03_satellite"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("fetch")


def load_cfg():
    cfg_dir = CODE_ROOT / "config"
    with open(cfg_dir / "01_sites.yaml", "r", encoding="utf-8") as fh:
        sites = yaml.safe_load(fh)["sites"]
    with open(cfg_dir / "02_variables.yaml", "r", encoding="utf-8") as fh:
        variables = yaml.safe_load(fh)
    with open(CODE_ROOT / "project_manifest.yaml", "r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)
    return sites, variables, manifest["data_version"], manifest["data_layout"]["root"]


def month_ranges(start: dt.date, end: dt.date):
    """把 [start, end] 切成逐月区间（每片一个请求）。"""
    cur = start
    while cur <= end:
        last_day = calendar.monthrange(cur.year, cur.month)[1]
        m_end = min(dt.date(cur.year, cur.month, last_day), end)
        yield cur, m_end
        cur = m_end + dt.timedelta(days=1)


def source_request(cfg, source, site, s, e):
    """返回 (endpoint, params, 输出文件)。"""
    common = dict(
        latitude=site["lat"],
        longitude=site["lon"],
        start_date=s.isoformat(),
        end_date=e.isoformat(),
        timezone=cfg["timezone"],
        cell_selection="land",
    )
    data_root = DATA_ROOT
    if source == "previous_runs":
        hourly = [
            f"{v}_previous_day{n}"
            for v in cfg["forecast_variables"]
            for n in cfg["leads"]
        ]
        params = {
            **common,
            "hourly": ",".join(hourly),
            "models": cfg["models"]["previous_runs"],
            "tilt": cfg["tilt"],
            "azimuth": cfg["azimuth"],
        }
        endpoint = "https://previous-runs-api.open-meteo.com/v1/forecast"
    elif source == "satellite":
        params = {
            **common,
            "hourly": ",".join(cfg["satellite_variables"]),
            "models": cfg["models"]["satellite"],
            "tilt": cfg["tilt"],
            "azimuth": cfg["azimuth"],
        }
        endpoint = "https://satellite-api.open-meteo.com/v1/archive"
    elif source == "era5":
        params = {
            **common,
            "hourly": ",".join(cfg["era5_variables"]),
            "tilt": cfg["tilt"],
            "azimuth": cfg["azimuth"],
        }
        endpoint = "https://archive-api.open-meteo.com/v1/archive"
    else:
        raise ValueError(f"未知数据源: {source}")
    out_file = data_root / "01_raw" / SOURCE_DIRS[source] / site["id"] / f"{site['id']}_{s:%Y-%m}.json"
    return endpoint, params, out_file


_log_lock = threading.Lock()


def append_log(row):
    log_path = DATA_ROOT / "01_raw" / "fetch_log_v2.csv"
    with _log_lock:
        new = not log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["source", "site", "start", "end", "status",
                            "requested_lat", "requested_lon", "service_lat", "service_lon",
                            "data_version", "bytes", "error"],
            )
            if new:
                writer.writeheader()
            writer.writerow(row)


def fetch_one(cfg, source, site, s, e, force, data_version):
    endpoint, params, out_file = source_request(cfg, source, site, s, e)
    validate = lambda payload: validate_raw_payload(payload, source=source, cfg=cfg, start=s, end=e)
    metadata = cache_metadata(source=source, cfg=cfg, start=s, end=e,
                              params=params, data_version=data_version)
    data = fetch_json(endpoint, params, out_file, force=force,
                      validator=validate, cache_metadata=metadata)
    common = dict(source=source, site=site["id"], start=s.isoformat(), end=e.isoformat(),
                  requested_lat=site["lat"], requested_lon=site["lon"], data_version=data_version)
    if data is None:
        row = dict(**common, status="fail", service_lat="", service_lon="",
                   bytes="", error="缓存无效或请求/合同校验失败")
    else:
        row = dict(**common, status="ok", service_lat=data.get("latitude"),
                   service_lon=data.get("longitude"),
                   bytes=out_file.stat().st_size, error="")
    append_log(row)
    return row


def main():
    parser = argparse.ArgumentParser(description="Open-Meteo 三源数据抓取")
    parser.add_argument("--site", default="all", help="站点 id 或 all")
    parser.add_argument("--start", required=True, help="开始日期 YYYY-MM-DD（含）")
    parser.add_argument("--end", required=True, help="结束日期 YYYY-MM-DD（含）")
    parser.add_argument("--sources", default="all",
                        choices=["all", "previous_runs", "satellite", "era5"])
    parser.add_argument("--workers", type=int, default=4, help="并发数（默认 4）")
    parser.add_argument("--force", action="store_true", help="忽略本地缓存强制重下")
    parser.add_argument("--dry-run", action="store_true", help="只打印任务计划，不实际请求")
    parser.add_argument("--data-dir", default=None, help="数据根目录（默认项目根/data）")
    args = parser.parse_args()

    sites, cfg, data_version, data_root_name = load_cfg()
    global DATA_ROOT
    DATA_ROOT = Path(args.data_dir) if args.data_dir else CODE_ROOT / data_root_name
    if args.site == "all":
        selected = sites
    else:
        selected = [s for s in sites if s["id"] == args.site]
        if not selected:
            logger.error("未知站点: %s（可用: %s）", args.site, [s["id"] for s in sites])
            sys.exit(1)

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    if start > end:
        logger.error("start 晚于 end")
        sys.exit(1)

    sources = ["previous_runs", "satellite", "era5"] if args.sources == "all" else [args.sources]
    tasks = [
        (source, site, s, e)
        for source in sources
        for site in selected
        for s, e in month_ranges(start, end)
    ]
    logger.info("任务计划：%d 个请求（%s × %d 站 × %d 个月）", len(tasks),
                ",".join(sources), len(selected), sum(1 for _ in month_ranges(start, end)))
    if args.dry_run:
        for source, site, s, e in tasks:
            print(f"{source:14s} {site['id']:10s} {s} ~ {e}")
        return

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_one, cfg, source, site, s, e, args.force, data_version): (source, site["id"], s)
            for source, site, s, e in tasks
        }
        for fut in as_completed(futures):
            row = fut.result()
            if row["status"] == "ok":
                ok += 1
            else:
                fail += 1
                logger.error("失败: %s %s %s", row["source"], row["site"], row["start"])
    logger.info("完成：成功 %d，失败 %d（日志: %s）", ok, fail, DATA_ROOT / "01_raw" / "fetch_log_v2.csv")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
