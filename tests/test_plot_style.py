"""Tests for matplotlib style helpers."""

import matplotlib.pyplot as plt

from earthsciences.utils.plot_style import clean_plot_style, use_earthsciences_style


def test_use_earthsciences_style_disables_grid_and_spines():
    use_earthsciences_style()
    fig, ax = plt.subplots()
    try:
        assert not ax.xaxis._major_tick_kw["gridOn"]
        assert ax.spines["top"].get_visible() is False
        assert ax.spines["right"].get_visible() is False
    finally:
        plt.close(fig)


def test_clean_plot_style_grid_opt_in():
    fig, ax = plt.subplots()
    try:
        clean_plot_style(ax, grid=False)
        assert not ax.xaxis._major_tick_kw["gridOn"]
        clean_plot_style(ax, grid=True)
        assert ax.xaxis._major_tick_kw["gridOn"]
    finally:
        plt.close(fig)
