"""
Matplotlib styling defaults for earthsciences figures.

Top and right spines are hidden; gridlines are off unless explicitly enabled.
"""

from __future__ import annotations

from importlib.resources import files
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

_STYLE_REGISTERED = False


def use_earthsciences_style() -> None:
    """
    Apply project-wide matplotlib defaults (no top/right spines, no grid).

    Safe to call multiple times. Tutorial and example scripts should call this
    once at startup (e.g. after ``setup_logging()``).
    """
    global _STYLE_REGISTERED
    style_path = files("earthsciences.utils") / "earthsciences.mplstyle"
    plt.style.use(style_path)
    mpl.rcParams.update(
        {
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.top": False,
            "ytick.right": False,
        }
    )
    _STYLE_REGISTERED = True


def clean_plot_style(ax: Any, *, grid: bool = False) -> None:
    """
    Apply earthsciences axis styling.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to style.
    grid : bool, optional
        Show gridlines only when True (default False).
    """
    ax.grid(grid, alpha=0.25, linewidth=0.5) if grid else ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", length=4, width=0.8, color="#555555", labelsize=10)


def setup_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """
    Create a figure and apply ``clean_plot_style`` to every axes.

    Returns
    -------
    fig, axes
        Same convention as ``plt.subplots`` (single Axes if 1×1).
    """
    if figsize is None:
        figsize = (7 * ncols, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

    if nrows == 1 and ncols == 1:
        clean_plot_style(axes)
        return fig, axes

    flat_axes = np.atleast_1d(axes).ravel()
    for ax in flat_axes:
        clean_plot_style(ax)

    return fig, axes
