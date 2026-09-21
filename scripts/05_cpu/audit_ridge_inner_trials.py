"""Run the complete preregistered Ridge inner selector on one real site.

Diagnostic only: no outer-validation or final-test rows enter this script.
The selected candidate is not promoted to an official benchmark result.
"""
from __future__ import annotations

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
from s05_tuning.estimators import ridge_fit_predict  # noqa: E402
from s05_tuning.runner import run_trials  # noqa: E402
from s05_tuning.search_space import candidates  # noqa: E402


def daylight_truth(block: pd.DataFrame) -> pd.DataFrame:
    valid = formal_daylight_mask(block) & np.isfinite(pd.to_numeric(block.ghi_obs_sat, errors="coerce"))
    out = block.loc[valid].copy()
    out["y"] = out.ghi_obs_sat.astype(float)
    return out


def main() -> None:
    manifest = load_manifest(ROOT / "project_manifest.yaml")
    data_root = ROOT / manifest["data_layout"]["root"]
    path = data_root / "03_featured" / "nanjing_1_featured_2024-02_2026-08.parquet"
    feature_receipt = json.loads(path.with_name(path.name + ".receipt.json").read_text(encoding="utf-8"))
    if feature_receipt["data_version"] != manifest["data_version"]:
        raise ValueError("wrong data version")
    frame = pd.read_parquet(path)
    outer = outer_folds(frame)[0]
    folds = [replace(fold, fit=daylight_truth(fold.fit),
                     early_stop=daylight_truth(fold.early_stop),
                     score=daylight_truth(fold.score)) for fold in inner_folds(outer)]
    if any(fold.fit.empty or fold.early_stop.empty or fold.score.empty for fold in folds):
        raise ValueError("an inner block has no eligible rows")
    selected, ledger = run_trials("ridge_mos", outer.fold_id, folds,
                                  ridge_fit_predict, device="cpu", smoke=False)
    if len(ledger) != 18 or any(trial.status != "ok" for trial in ledger):
        raise ValueError("six candidates by three inner folds did not all complete")
    receipt = {
        "scope": "one_real_site_all_three_inner_folds_ridge_diagnostic_only",
        "result_status": "diagnostic",
        "official_eligible": False,
        "data_version": manifest["data_version"],
        "site": "nanjing_1",
        "outer_fold": outer.fold_id,
        "candidate_count": len(candidates("ridge_mos")),
        "inner_fold_count": len(folds),
        "selected_candidate_diagnostic_only": selected,
        "candidate_parameters": list(candidates("ridge_mos")),
        "trials": [asdict(trial) for trial in ledger],
    }
    output = data_root / "05_cpu_mini_e2e" / "ridge_inner_trials.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "trials"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
