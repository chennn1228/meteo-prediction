# -*- coding: utf-8 -*-
"""Archived unimplemented live-prediction sketch; not an active entry point.

TODO（尚未实现，需按以下步骤补全）：
1. fetch_live_forecast: 调用 Open-Meteo Forecast API（models=gfs_seamless），
   取目标日起报的 D+1/D+2/D+3 逐时变量（与 cfg/02_variables.yaml 的 forecast_variables 一致）。
2. build_features: 逐站按 features.py 的同一逻辑构造 35 个建模特征
   （太阳几何/晴空由 pvlib 本地计算；kt、云量滚动/差分、GHI 滞后按 (station, lead) 分组时序构造），
   必须与训练时的分组顺序完全一致；禁止使用任何真值字段。
3. load_model / predict: 加载训练好的模型（v1-xgb 或 v8），输出订正 GHI。
4. save: 写回按站点/时效/时刻组织的订正结果。
"""
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config" / "01_sites.yaml").exists())
os.environ.setdefault("MPLCONFIGDIR", str(CODE_ROOT / ".cache" / "matplotlib"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("predict_daily")


def load_cfg():
    return yaml.safe_load((CODE_ROOT / "config" / "02_variables.yaml").read_text(encoding="utf-8"))


def fetch_live_forecast(site, start_date, days=3):
    # TODO: 调用 https://api.open-meteo.com/v1/forecast
    raise NotImplementedError("需实现：按 cfg 变量清单抓取最新起报的 D+1/D+2/D+3 逐时数据")


def build_features(raw_df, lat, lon):
    # TODO: 复刻 features.py 的特征构造逻辑
    raise NotImplementedError("需实现：与训练一致的 35 维特征构造（含 lag/rolling 顺序校验）")


def main():
    cfg = load_cfg()
    sites = yaml.safe_load((CODE_ROOT / "config" / "01_sites.yaml").read_text(encoding="utf-8"))["sites"]
    start_date = pd.Timestamp.now(tz="UTC").date()
    for site in sites:
        raw = fetch_live_forecast(site, start_date)
        feat = build_features(raw, site["lat"], site["lon"])
        # TODO: load model and predict
        logger.info("site %s: %d 行特征就绪", site["id"], len(feat))


if __name__ == "__main__":
    main()
