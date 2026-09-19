"""Compatibility entry for the current manifest-backed nested selector.

The previous implementation imported candidate grids and fit functions from
the retired month-balanced module, reused inner scoring as early stopping,
and selected Ridge by MAE. None of those paths are retained here.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "s03_models" / "train"))

from s04_splits.diagnostics import first_outer_sample_report  # noqa: E402
from s04_splits.rolling import inner_folds, outer_folds  # noqa: E402
from s05_tuning.estimators import ridge_fit_predict, tree_fit_predict  # noqa: E402
from s05_tuning.runner import run_trials  # noqa: E402
import train_v1 as tv  # noqa: E402


def load_features(target: str, *, daytime_only: bool = True) -> pd.DataFrame:
    sites = yaml.safe_load((ROOT / "config" / "01_sites.yaml").read_text(encoding="utf-8"))["sites"]
    label = "ghi_obs_sat" if target == "ghi" else "cloud_cover_obs"
    columns = list(dict.fromkeys(tv.FEATURES_NUM + ["station_id", "target_time_utc", label]))
    frames = []
    for site in sites:
        path = ROOT / "data" / "03_featured" / f"{site['id']}_featured_2024-02_2026-09.parquet"
        frames.append(pd.read_parquet(path, columns=columns))
    data = pd.concat(frames, ignore_index=True)
    data["target_time_utc"] = pd.to_datetime(data["target_time_utc"], utc=True)
    if target == "ghi" and daytime_only:
        data = data.query(tv.DAY_FILTER).copy()
    data = data.dropna(subset=[label]).copy()
    data["season"] = data["target_time_utc"].map(tv.season_of)
    data["y"] = data[label]
    return data


def tune_outer(data: pd.DataFrame, model_id: str, *, gap_days: int = 10,
               smoke: bool = False):
    """Return per-outer selections and ledgers, never accessing final test.

    The current first outer prefix is expected to raise an evidence-backed
    InsufficientHistoryError under 3×30-day scoring plus independent early
    stopping. No protocol adjustment is made inside this function.
    """
    adapter = ({"ridge_mos": ridge_fit_predict,
                "lgbm": tree_fit_predict("lgbm"),
                "xgboost": tree_fit_predict("xgboost")})[model_id]
    results = []
    for outer in outer_folds(data, gap_days=gap_days):
        folds = inner_folds(outer, gap_days=gap_days)
        chosen, ledger = run_trials(model_id, outer.fold_id, folds, adapter, smoke=smoke)
        results.append((outer.fold_id, chosen, ledger))
        if smoke:
            break
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Current nested split audit; no training by default")
    parser.add_argument("--target", choices=["ghi", "cloud"], default="ghi")
    parser.add_argument("--gap-days", type=int, choices=[7, 10, 14], default=10)
    parser.add_argument("--diagnose", action="store_true",
                        help="show first outer × inner × station × lead counts")
    args = parser.parse_args()
    if not args.diagnose:
        parser.error("training is disabled in this compatibility entry; use --diagnose")
    report = first_outer_sample_report(
        load_features(args.target, daytime_only=False), gap_days=args.gap_days)
    print(report.to_string(index=False))
    print(f"feasible={int(report.protocol_feasible.sum())}/{len(report)}")


if __name__ == "__main__":
    main()
