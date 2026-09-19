# -*- coding: utf-8 -*-
"""Open-Meteo 通用抓取客户端：重试退避 + 原子落盘 + 断点续传。

只依赖标准库，可单独复用。
"""
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_RETRIES = 4
DEFAULT_TIMEOUT = 60
USER_AGENT = "solar-irradiance-correction/1.0"


def build_url(endpoint: str, params: dict) -> str:
    """把参数拼进 URL（自动处理编码）。"""
    return endpoint + "?" + urllib.parse.urlencode(params)


def _load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def fetch_json(endpoint, params, out_file, force=False,
               retries=DEFAULT_RETRIES, timeout=DEFAULT_TIMEOUT,
               validator=None, cache_metadata=None):
    """请求 Open-Meteo 并把原始响应落盘为 JSON。

    返回解析后的 dict；重试耗尽后返回 None（不抛异常，由调用方记录失败）。
    有合同的调用仅在 payload 与版本/请求 sidecar 均有效时复用缓存。
    无合同的旧调用保持兼容；正式抓取入口必须提供 validator 和 metadata。
    """
    out_file = Path(out_file)
    meta_path = Path(str(out_file) + ".meta.json")
    if not force and out_file.exists() and out_file.stat().st_size > 0:
        try:
            cached = _load(out_file)
            if validator is not None:
                validator(cached)
            if cache_metadata is not None and _load(meta_path) != cache_metadata:
                raise ValueError("cache sidecar/version/request mismatch")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("缓存无效，重新请求 %s: %s", out_file, exc)
        else:
            logger.info("缓存合同有效，跳过: %s", out_file)
            return cached

    url = build_url(endpoint, params)
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            if data.get("error"):
                raise RuntimeError(f"API 返回错误: {data.get('reason')}")
            if validator is not None:
                validator(data)
            os.makedirs(out_file.parent, exist_ok=True)
            tmp = str(out_file) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(raw)
            os.replace(tmp, out_file)  # 原子替换，避免半截文件
            if cache_metadata is not None:
                meta_tmp = str(meta_path) + ".tmp"
                with open(meta_tmp, "w", encoding="utf-8") as fh:
                    json.dump(cache_metadata, fh, ensure_ascii=False, sort_keys=True)
                os.replace(meta_tmp, meta_path)
            logger.info("已保存 %s（%d bytes）", out_file, len(raw))
            return data
        except Exception as exc:
            last_err = exc
            wait = 2 ** attempt
            logger.warning("第 %d/%d 次失败: %s；%ds 后重试", attempt, retries, exc, wait)
            time.sleep(wait)
    logger.error("重试耗尽: %s（最后错误: %s）", url, last_err)
    return None
