# -*- coding: utf-8 -*-
"""统一出版级绘图规范（Nature-figure Python 轨）。

用法：
  import sys; sys.path.insert(0, <CODE_ROOT>/src)
  from s04_evaluation.analysis.plot_common import apply_pub_style, PALETTE, save_pub, panel_label

规则要点见 docs/visualization.md。
"""
import matplotlib.pyplot as plt

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "neutral_light": "#CFCECE",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
}

COLORS = [PALETTE["blue_main"], PALETTE["blue_secondary"], PALETTE["green_3"],
          PALETTE["red_strong"], PALETTE["teal"], PALETTE["violet"]]


def apply_pub_style(font_size=8, axes_linewidth=0.8, use_tex=False):
    """调用一次后所有图共享样式；SVG 文字保持可编辑。"""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = axes_linewidth
    plt.rcParams["legend.frameon"] = False
    if use_tex:
        plt.rcParams["text.usetex"] = True


def save_pub(fig, out_dir, name, dpi=300, close=True):
    """统一导出：SVG（主）+ PNG（预览）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return out_dir / f"{name}.svg"


def panel_label(ax, label, x=-0.12, y=1.04, fontsize=11, color="black"):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", color=color, ha="left", va="bottom")
