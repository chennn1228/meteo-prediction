"""Plot complete Nanjing six-candidate, three-inner-fold tuning receipts.

Only inner mean pinball is shown. These points are selection evidence, not
outer-validation/test performance or an official 20-site result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").is_file())
sys.path.insert(0, str(ROOT / "src"))

from s01_core.config_loader import load_manifest  # noqa: E402
from s13_visualization.s01_style.publication import apply_publication_style  # noqa: E402


def main() -> None:
    apply_publication_style()
    manifest = load_manifest(ROOT / "project_manifest.yaml")
    source = ROOT / manifest["data_layout"]["root"] / "06_cpu_single_site" / "tuning"
    rows = []
    for model in ("ridge_mos", "lgbm", "xgboost"):
        for outer_index in range(1, 6):
            payload = json.loads((source / f"outer_{outer_index}_{model}.json").read_text(encoding="utf-8"))
            if payload.get("status") != "pass" or len(payload.get("trials", [])) != 18:
                raise ValueError(f"incomplete tuning ledger for {model}/outer_{outer_index}")
            for trial in payload["trials"]:
                if trial["status"] != "ok" or trial["mean_pinball"] is None:
                    raise ValueError("failed trial cannot enter candidate plot")
                rows.append({"model": model, "outer_fold": trial["outer"],
                             "inner_fold": trial["inner"], "candidate": trial["candidate"] + 1,
                             "mean_pinball": trial["mean_pinball"],
                             "scoring_rows": trial["scoring_rows"]})
    raw = pd.DataFrame(rows)
    if len(raw) != 270:
        raise ValueError("expected 3 models × 5 outer × 6 candidates × 3 inner trials")
    output = ROOT / "figs" / "nanjing_15var_diagnostic" / "03_model_benchmark"
    output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / "fig_tuning_candidates_source.csv", index=False)
    by_outer = raw.groupby(["model", "outer_fold", "candidate"], as_index=False).mean_pinball.mean()
    overall = raw.groupby(["model", "candidate"], as_index=False).mean_pinball.mean()
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 75 / 25.4))
    for axis, model in zip(axes, ("ridge_mos", "lgbm", "xgboost")):
        lines = by_outer.loc[by_outer.model == model]
        for outer, group in lines.groupby("outer_fold"):
            axis.plot(group.candidate, group.mean_pinball, color="#C4CBD0", lw=0.7,
                      marker="o", ms=2.2, alpha=0.9)
        means = overall.loc[overall.model == model].sort_values("candidate")
        axis.plot(means.candidate, means.mean_pinball, color="#245680", lw=1.6,
                  marker="o", ms=3.8)
        chosen = means.loc[means.mean_pinball.idxmin()]
        axis.scatter([chosen.candidate], [chosen.mean_pinball], s=42, marker="D",
                     color="#D4863B", zorder=5)
        axis.set_xticks(np.arange(1, 7))
        axis.set_xlabel("Preregistered candidate")
        axis.set_title(model.replace("_", " "))
        axis.grid(axis="y", color="#D7E2E8", lw=0.5)
    axes[0].set_ylabel("Inner scoring mean pinball (W m$^{-2}$)")
    fig.suptitle("Six candidates × three inner folds on every outer training prefix",
                 x=0.05, ha="left", y=1.01)
    fig.text(0.995, 0.995, "Nanjing only | inner selection, not outer/test performance",
             ha="right", va="top", fontsize=6, color="#8C959A")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.24, top=0.78, wspace=0.30)
    files = []
    for extension in ("svg", "pdf", "png"):
        target = output / f"fig_tuning_candidates.{extension}"
        fig.savefig(target, dpi=300 if extension == "png" else None,
                    bbox_inches="tight", facecolor="white")
        files.append(str(target.relative_to(ROOT)))
    plt.close(fig)
    manifest_path = output.parent / "figure_manifest.json"
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata["figures"]["fig_tuning_candidates"] = {
        "files": files, "source_data": "fig_tuning_candidates_source.csv",
        "n_definition": "270 inner trial scores = 3 models × 5 outer prefixes × 6 candidates × 3 inner folds",
        "statistic": "unweighted mean of 15 inner-fold mean pinball values per candidate; gray lines are per-outer three-fold means",
    }
    metadata["figure_count"] = len(metadata["figures"])
    manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure": "fig_tuning_candidates", "trial_scores": len(raw),
                      "figure_count": metadata["figure_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
