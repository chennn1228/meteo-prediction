"""Plot complete development-inner one-site group ablation/permutation receipts."""
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

SCOPE = "Nanjing only | development inner folds | diagnostic, not 20-site evidence"


def save(fig, output: Path, name: str) -> list[str]:
    fig.text(0.995, 0.995, SCOPE, ha="right", va="top", fontsize=6, color="#8C959A")
    files = []
    for extension in ("svg", "pdf", "png"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=300 if extension == "png" else None,
                    bbox_inches="tight", facecolor="white")
        files.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return files


def load_receipts(folder: Path, groups: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ablation, permutation = [], []
    for outer_index in range(1, 6):
        for inner_index in range(1, 4):
            path = folder / f"outer_{outer_index}_inner_{inner_index}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (payload.get("status") != "pass" or payload.get("official_eligible") is not False
                    or len(payload.get("ablation", [])) != len(groups)
                    or len(payload.get("permutation", [])) != 3 * len(groups)):
                raise ValueError(f"group analysis incomplete: {path}")
            for row in payload["ablation"]:
                ablation.append({"outer_fold": payload["outer_fold"],
                                 "inner_fold": payload["inner_fold"],
                                 "group": row["group"], "score_rows": payload["score_rows"],
                                 "baseline_mean_pinball": payload["baseline_mean_pinball"],
                                 "mean_pinball": row["mean_pinball"],
                                 "delta_vs_full": row["delta_vs_full"]})
            for row in payload["permutation"]:
                permutation.append({"outer_fold": payload["outer_fold"],
                                    "inner_fold": payload["inner_fold"],
                                    "group": row["group"], "repeat": row["repeat"],
                                    "score_rows": payload["score_rows"],
                                    "baseline_mean_pinball": payload["baseline_mean_pinball"],
                                    "mean_pinball": row["mean_pinball"],
                                    "delta_vs_full": row["delta_vs_full"]})
    a, p = pd.DataFrame(ablation), pd.DataFrame(permutation)
    if (len(a) != 15 * len(groups) or len(p) != 15 * len(groups) * 3
            or set(a.group) != set(groups) or set(p.group) != set(groups)):
        raise ValueError("group evidence count/registry mismatch")
    return a, p


def plot_effect(data: pd.DataFrame, groups: list[str], output: Path,
                name: str, title: str, note: str) -> dict:
    # One gray point per outer prefix, after averaging its three inner folds
    # (and the three permutation repeats when applicable).
    by_outer = data.groupby(["group", "outer_fold"], as_index=False).delta_vs_full.mean()
    overall = data.groupby("group", as_index=False).delta_vs_full.mean().set_index("group")
    by_outer.to_csv(output / f"{name}_by_outer.csv", index=False)
    data.to_csv(output / f"{name}_source.csv", index=False)
    def draw(ax, chosen: list[str]) -> None:
        for index, group in enumerate(chosen):
            values = by_outer.loc[by_outer.group == group, "delta_vs_full"].to_numpy(dtype=float)
            center = float(overall.loc[group, "delta_vs_full"])
            ax.plot([values.min(), values.max()], [index, index], color="#9DAAB3", lw=1)
            ax.scatter(values, np.full(len(values), index), color="#9DAAB3", s=11, zorder=3)
            ax.scatter([center], [index], color="#D4863B" if group == "cloud" else "#245680",
                       marker="D", s=38, zorder=4)
        ax.axvline(0, color="#526774", lw=0.8)
        ax.set_yticks(np.arange(len(chosen)), chosen)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#D7E2E8", lw=0.5)

    if name == "fig_group_permutation":
        # Radiation is an order of magnitude larger; use explicitly different
        # linear scales instead of compressing the other group comparisons.
        fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 83 / 25.4),
                                 gridspec_kw={"width_ratios": [1, 2]})
        draw(axes[0], ["radiation"])
        draw(axes[1], [group for group in groups if group != "radiation"])
        axes[0].set_title("Radiation", fontsize=7)
        axes[1].set_title("Other groups (zoomed linear scale)", fontsize=7)
        fig.suptitle(title, x=0.04, ha="left", y=1.01)
        fig.text(0.5, 0.15, "Increase in seven-quantile mean pinball vs full model "
                 "(W m$^{-2}$)", ha="center", fontsize=7)
        fig.text(0.5, 0.055, note, ha="center", fontsize=6, color="#657783")
        fig.subplots_adjust(left=0.11, right=0.97, top=0.77, bottom=0.30, wspace=0.65)
    else:
        fig, ax = plt.subplots(figsize=(183 / 25.4, 83 / 25.4))
        draw(ax, groups)
        ax.set_xlabel("Increase in seven-quantile mean pinball vs full model (W m$^{-2}$)")
        ax.set_title(title, loc="left", pad=11)
        fig.text(0.5, 0.075, note, ha="center", fontsize=6, color="#657783")
        fig.subplots_adjust(left=0.22, right=0.97, top=0.78, bottom=0.25)
    return {"files": save(fig, output, name),
            "source_data": [f"{name}_source.csv", f"{name}_by_outer.csv"],
            "n_definition": "15 inner scoring blocks = 5 nested outer training prefixes × 3 inner folds; outer dots are not independent sites",
            "statistic": "unweighted mean loss difference; positive means disruption/removal worsened pinball; no confidence interval claimed",
            "limitation": note}


def main() -> None:
    apply_publication_style()
    manifest = load_manifest(ROOT / "project_manifest.yaml")
    groups = [group for group in manifest["feature_groups"]
              if group not in {"selection_evidence", "shap_role", "spatial_static"}]
    root = ROOT / manifest["data_layout"]["root"]
    a, p = load_receipts(root / "06_cpu_single_site" / "group_analysis", groups)
    output = ROOT / "figs" / "nanjing_15var_diagnostic" / "05_interpretation"
    output.mkdir(parents=True, exist_ok=True)
    figures = {
        "fig_group_ablation": plot_effect(
            a, groups, output, "fig_group_ablation",
            "Full−group refits inside development folds",
            "Fixed full-model candidate; spatial-static constant at one site; Base+cloud/kt controls still pending"),
        "fig_group_permutation": plot_effect(
            p, groups, output, "fig_group_permutation",
            "Joint group permutation within lead, development folds",
            "Three deterministic repeats; correlated features share effects; not causal importance"),
    }
    path = output.parent / "figure_manifest.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["figures"].update(figures)
    metadata["figure_count"] = len(metadata["figures"])
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figures": list(figures), "total_figures": metadata["figure_count"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
