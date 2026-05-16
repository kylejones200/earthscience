#!/usr/bin/env python3
"""
Plotting utilities for clean, minimalist visualizations.

Based on Edward Tufte's principles:
- Remove chart junk
- Maximize data-ink ratio
- Use descriptive titles instead of redundant labels
"""

import matplotlib.pyplot as plt


def clean_plot_style(ax, remove_grid=True, remove_spines=True):
    """
    Apply clean, minimalist style to a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to style
    remove_grid : bool, optional
        Remove gridlines (default=True)
    remove_spines : bool, optional
        Remove top and right spines (default=True)
    """
    # Remove gridlines unless absolutely needed
    if remove_grid:
        ax.grid(False)

    # Remove top and right spines
    if remove_spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Make bottom and left spines thinner
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["left"].set_linewidth(0.8)

    # Lighter tick marks
    ax.tick_params(axis="both", which="both", length=4, width=0.8, color="#555555", labelsize=10)


def setup_figure(nrows=1, ncols=1, figsize=None, **kwargs):
    """
    Create figure with clean default styling.

    Parameters
    ----------
    nrows, ncols : int
        Number of subplot rows and columns
    figsize : tuple, optional
        Figure size (width, height) in inches
    **kwargs
        Additional arguments passed to plt.subplots()

    Returns
    -------
    fig, axes
        Figure and axes objects
    """
    if figsize is None:
        # Auto-size based on subplots
        width = 7 * ncols
        height = 5 * nrows
        figsize = (width, height)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

    # Apply clean style to all axes
    if nrows == 1 and ncols == 1:
        clean_plot_style(axes)
    elif nrows == 1 or ncols == 1:
        for ax in axes:
            clean_plot_style(ax)
    else:
        for row in axes:
            for ax in row:
                clean_plot_style(ax)

    return fig, axes if (nrows > 1 or ncols > 1) else (fig, axes)
