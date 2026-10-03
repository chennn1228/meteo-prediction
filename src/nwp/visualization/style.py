"""Single source for publication visual defaults and dense relationships."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nwp.core.schema import ContractError


STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "font.size": 7, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "white",
    "savefig.facecolor": "white",
}
PALETTE = {
    "baseline": "#484878", "reference": "#7884B4", "method": "#0F4D92",
    "signal": "#42949E", "gain": "#2E9E44", "loss": "#B64342",
    "neutral": "#767676", "light": "#D8D8D8", "dark": "#272727",
}


@dataclass(frozen=True)
class FigureContract:
    conclusion: str
    evidence: tuple[str, ...]
    source_data: str
    result_status: str
    n_definition: str
    statistical_definition: str

    def validate(self) -> None:
        if not self.conclusion.strip() or not self.evidence:
            raise ContractError("figure needs a conclusion and evidence chain")
        if not self.source_data.strip() or not self.n_definition.strip():
            raise ContractError("figure must identify source data and n")
        if not self.statistical_definition.strip():
            raise ContractError("figure must disclose its statistical definition")
        if self.result_status not in {"diagnostic", "provisional", "official"}:
            raise ContractError("figure result status is invalid")


def apply_publication_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update(STYLE)


def save_figure(fig, stem: Path, contract: FigureContract,
                *, dpi: int = 300) -> tuple[Path, ...]:
    import matplotlib.pyplot as plt
    contract.validate()
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = tuple(stem.with_suffix(f".{suffix}") for suffix in ("svg", "png"))
    for path in paths:
        fig.savefig(path, bbox_inches="tight",
                    dpi=dpi if path.suffix == ".png" else None)
    plt.close(fig)
    return paths


def relationship(ax, x, y, *, xlabel: str, ylabel: str,
                 threshold: int = 200, gridsize: int = 38,
                 show_reference: bool = False):
    xv, yv = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if xv.shape != yv.shape or xv.ndim != 1:
        raise ContractError("relationship inputs must be equal-length vectors")
    valid = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[valid], yv[valid]
    if len(xv) == 0:
        raise ContractError("relationship has no finite observations")
    if len(xv) >= threshold:
        artist = ax.hexbin(xv, yv, gridsize=gridsize, cmap="cividis",
                           mincnt=1, linewidths=0, bins="log")
        ax.figure.colorbar(
            artist, ax=ax, label="count per hexagon (log scale)")
        chart_type = "hexbin"
    else:
        artist = ax.scatter(
            xv, yv, s=9, color=PALETTE["method"], alpha=0.7,
            edgecolors="none")
        chart_type = "scatter"
    if show_reference:
        lo, hi = float(min(xv.min(), yv.min())), float(max(xv.max(), yv.max()))
        ax.plot([lo, hi], [lo, hi], color=PALETTE["neutral"], lw=.8, ls="--")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    return chart_type, artist
