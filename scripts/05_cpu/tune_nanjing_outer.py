"""Six candidates × three inner folds on each original Nanjing outer prefix.

Only inner data are supplied to the selector. Artifacts are one-site diagnostic
ledgers, not official model selection or outer-fold performance estimates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").is_file())
sys.path.insert(0, str(ROOT / "src"))

from s01_core.config_loader import load_manifest  # noqa: E402
from s03_features.engineering import formal_daylight_mask  # noqa: E402
from s04_splits.rolling import inner_folds, outer_folds  # noqa: E402
from s05_tuning.estimators import ridge_fit_predict, tree_fit_predict  # noqa: E402
from s05_tuning.runner import IncompleteTrialsError, run_trials  # noqa: E402

MODELS = ("ridge_mos", "lgbm", "xgboost")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eligible(block: pd.DataFrame) -> pd.DataFrame:
    mask = formal_daylight_mask(block) & np.isfinite(pd.to_numeric(block.ghi_obs_sat, errors="coerce"))
    out = block.loc[mask].copy()
    out["y"] = out.ghi_obs_sat.astype(float)
    if out.empty:
        raise ValueError("empty eligible inner block")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", choices=[f"outer_{i}" for i in range(1, 6)])
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--gap-days", type=int, choices=(7, 10, 14), default=10)
    parser.add_argument("--force", action="store_true", help="recompute matching local checkpoint")
    args = parser.parse_args()
    manifest_path = ROOT / "project_manifest.yaml"
    manifest = load_manifest(manifest_path)
    data_root = ROOT / manifest["data_layout"]["root"]
    path = data_root / "03_featured" / "nanjing_1_featured_2024-02_2026-08.parquet"
    feature_receipt = json.loads(path.with_name(path.name + ".receipt.json").read_text(encoding="utf-8"))
    if feature_receipt["data_version"] != manifest["data_version"]:
        raise ValueError("feature/data version mismatch")
    source_sha, manifest_sha = sha256(path), sha256(manifest_path)
    frame = pd.read_parquet(path)
    gap = args.gap_days
    output = data_root / "06_cpu_single_site" / ("tuning" if gap == 10 else f"tuning_gap{gap}")
    output.mkdir(parents=True, exist_ok=True)
    for outer in outer_folds(frame, gap_days=gap):
        if args.outer and outer.fold_id != args.outer:
            continue
        folds = [replace(fold, fit=eligible(fold.fit),
                         early_stop=eligible(fold.early_stop),
                         score=eligible(fold.score)) for fold in inner_folds(outer, gap_days=gap)]
        for model_id in MODELS:
            if args.model and model_id != args.model:
                continue
            destination = output / f"{outer.fold_id}_{model_id}.json"
            if destination.is_file() and not args.force:
                current = json.loads(destination.read_text(encoding="utf-8"))
                if (current.get("status") == "pass" and current.get("source_feature_sha256") == source_sha
                        and current.get("manifest_sha256") == manifest_sha
                        and len(current.get("trials", [])) == 18):
                    print(f"REUSE {outer.fold_id} {model_id}: selected={current['selected_candidate']}", flush=True)
                    continue
            adapter = ridge_fit_predict if model_id == "ridge_mos" else tree_fit_predict(model_id)
            try:
                selected, ledger = run_trials(model_id, outer.fold_id, folds,
                                              adapter, device="cpu", smoke=False,
                                              gap_days=gap)
                status = "pass"
            except IncompleteTrialsError as exc:
                selected, ledger, status = None, exc.ledger, "blocked"
            payload = {
                "status": status,
                "scope": "one_real_site_inner_selection_not_official_benchmark",
                "official_eligible": False,
                "site": "nanjing_1",
                "outer_fold": outer.fold_id,
                "model_id": model_id,
                "gap_days": gap,
                "data_version": manifest["data_version"],
                "source_feature_sha256": source_sha,
                "manifest_sha256": manifest_sha,
                "selected_candidate": selected,
                "inner_fold_counts": [
                    {"inner_fold": fold.fold_id, "fit": len(fold.fit),
                     "early_stop": len(fold.early_stop), "score": len(fold.score)}
                    for fold in folds
                ],
                "trials": [asdict(trial) for trial in ledger],
            }
            destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
            print(f"{status.upper()} {outer.fold_id} {model_id}: selected={selected}; "
                  f"trials={len(ledger)}", flush=True)
            if status != "pass":
                raise RuntimeError(f"{outer.fold_id}/{model_id} incomplete; inspect {destination}")


if __name__ == "__main__":
    main()
