"""Hexagonal continuous-density default for high-density scientific relations."""
from __future__ import annotations

import numpy as np

from s01_core.config_loader import ProtocolError


def relationship(ax, x, y, *, xlabel: str, ylabel: str, threshold: int = 200,
                 gridsize: int = 38, show_reference: bool = False):
    """Use color-blind-readable hexbin for dense points; scatter only for small n."""
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    if xv.shape != yv.shape or xv.ndim != 1:
        raise ProtocolError("relationship inputs must be equal-length vectors")
    valid = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[valid], yv[valid]
    if len(xv) == 0:
        raise ProtocolError("relationship has no finite observations")
    if len(xv) >= threshold:
        artist = ax.hexbin(xv, yv, gridsize=gridsize, cmap="cividis", mincnt=1,
                           linewidths=0, bins="log")
        ax.figure.colorbar(artist, ax=ax, label="count per hexagon (log scale)")
        chart_type = "hexbin"
    else:
        artist = ax.scatter(xv, yv, s=9, color="#0F4D92", alpha=0.7,
                            edgecolors="none")
        chart_type = "scatter"
    if show_reference:
        lo = float(min(xv.min(), yv.min()))
        hi = float(max(xv.max(), yv.max()))
        ax.plot([lo, hi], [lo, hi], color="#767676", lw=0.8, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return chart_type, artist
