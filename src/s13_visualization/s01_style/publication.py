"""Nature-leaning scientific figure contract and Python-only export settings.

This is a style/validation layer, not a renderer for provisional/legacy results.
Every formal figure must declare its claim, evidence, source and result status.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from s01_core.config_loader import ProtocolError


PALETTE = {
    "baseline": "#484878", "reference": "#7884B4", "method": "#0F4D92",
    "signal": "#42949E", "gain": "#2E9E44", "loss": "#B64342",
    "neutral": "#767676", "light": "#D8D8D8", "dark": "#272727",
}
ARCHETYPES = frozenset({"quantitative_grid", "schematic_led_composite",
                       "image_plate_quant", "asymmetric_mixed_modality"})


@dataclass(frozen=True)
class FigureContract:
    conclusion: str
    evidence: tuple[str, ...]
    archetype: str
    source_data: str
    result_status: str
    n_definition: str
    statistical_definition: str
    width_mm: float = 183.0

    def validate(self) -> None:
        if not self.conclusion.strip() or not self.evidence or not self.source_data.strip():
            raise ProtocolError("figure needs a conclusion, evidence chain and source data")
        if self.archetype not in ARCHETYPES:
            raise ProtocolError(f"unregistered figure archetype: {self.archetype}")
        if self.result_status != "official":
            raise ProtocolError("formal figures cannot display provisional/legacy results")
        if not self.n_definition.strip() or not self.statistical_definition.strip():
            raise ProtocolError("figure must disclose n and statistical definitions")
        if self.width_mm not in (89.0, 183.0):
            raise ProtocolError("figure width must be declared as single/double column")


def apply_publication_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "font.size": 7, "axes.linewidth": 0.8,
        "axes.spines.right": False, "axes.spines.top": False,
        "legend.frameon": False, "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def panel_label(ax, label: str) -> None:
    if len(label) != 1 or not label.islower():
        raise ProtocolError("panel label must be one lowercase letter")
    ax.text(-0.07, 1.02, label, transform=ax.transAxes,
            fontsize=8, fontweight="bold", ha="left", va="bottom")


def save_formal_figure(fig, stem: Path, contract: FigureContract, *, dpi: int = 600) -> tuple[Path, ...]:
    contract.validate()
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = tuple(stem.with_suffix(f".{suffix}") for suffix in ("svg", "png"))
    for path in paths:
        fig.savefig(path, bbox_inches="tight", dpi=dpi if path.suffix == ".png" else None)
    plt.close(fig)
    return paths
