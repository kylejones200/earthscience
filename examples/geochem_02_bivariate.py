"""
Geochemistry Example 2: Bivariate Analysis and Correlations

This example demonstrates bivariate relationships between elements,
correlation analysis, and scatter plot matrices.

Topics covered:
- Element correlations
- Scatter plots
- Linear regression
- Log-log plots
- Association indices
"""

import logging

import matplotlib.pyplot as plt
import numpy as np

from earthsciences.data import load_stream_sediments
from earthsciences.statistics import (
    bootstrap_confidence_interval,
    correlation,
    linear_regression,
)
from earthsciences.utils.logging_config import log_block, log_section, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

np.random.seed(42)

log_section("GEOCHEMISTRY EXAMPLE 2: Bivariate Analysis")

# %% Load Data
logger.info("1. Loading data...")

elements = ["Cu", "Pb", "Zn", "Ag", "Au", "Mo"]
data = load_stream_sediments(elements)

logger.info("Loaded %s samples with %s elements", len(data), len(elements))

data_clean = data.dropna(subset=elements, thresh=4)
logger.info("After filtering: %s samples", len(data_clean))

# %% Correlation Matrix
logger.info("2. Correlation Analysis")
logger.info("Pearson Correlations (linear relationships):")

correlation_matrix = np.zeros((len(elements), len(elements)))
for i, elem1 in enumerate(elements):
    for j, elem2 in enumerate(elements):
        if elem1 not in data_clean.columns or elem2 not in data_clean.columns:
            correlation_matrix[i, j] = np.nan
            continue

        mask = data_clean[elem1].notna() & data_clean[elem2].notna()
        if mask.sum() < 10:
            correlation_matrix[i, j] = np.nan
            continue

        corr_result = correlation(
            data_clean.loc[mask, elem1].values,
            data_clean.loc[mask, elem2].values,
            method="pearson",
        )
        correlation_matrix[i, j] = corr_result["r"]


def _format_correlation_matrix(labels: list[str], matrix: np.ndarray) -> str:
    lines = ["     " + "".join(f"{label:>7}" for label in labels)]
    for i, label in enumerate(labels):
        row = f"{label:>4} "
        for j in range(len(labels)):
            value = matrix[i, j]
            if np.isnan(value):
                row += f"   {'--':>5}"
            else:
                row += f"  {value:>5.2f}"
        lines.append(row)
    return "\n".join(lines)


log_block(_format_correlation_matrix(elements, correlation_matrix))

# %% Strong Correlations
logger.info("Strong Correlations (|r| > 0.5):")

strong_corrs = []
for i in range(len(elements)):
    for j in range(i + 1, len(elements)):
        if abs(correlation_matrix[i, j]) > 0.5 and not np.isnan(correlation_matrix[i, j]):
            strong_corrs.append((elements[i], elements[j], correlation_matrix[i, j]))

strong_corrs.sort(key=lambda x: abs(x[2]), reverse=True)

for elem1, elem2, r in strong_corrs:
    logger.info("%4s - %-4s: r = %6.3f", elem1, elem2, r)

# %% Detailed Analysis: Cu-Zn Relationship
logger.info("3. Detailed Analysis: Cu-Zn Relationship")

if "Cu" in data_clean.columns and "Zn" in data_clean.columns:
    mask = data_clean["Cu"].notna() & data_clean["Zn"].notna()
    cu = data_clean.loc[mask, "Cu"].values
    zn = data_clean.loc[mask, "Zn"].values

    mask_pos = (cu > 0) & (zn > 0)
    cu_pos = cu[mask_pos]
    zn_pos = zn[mask_pos]

    logger.info("Sample size: %s", len(cu_pos))

    corr_result = correlation(cu_pos, zn_pos, method="pearson")
    logger.info("Pearson correlation: r = %.3f", corr_result["r"])
    logger.info("p-value: %.4e", corr_result["p_value"])
    logger.info("Significant? %s", "Yes" if corr_result["significant"] else "No")

    spear_result = correlation(cu_pos, zn_pos, method="spearman")
    logger.info("Spearman correlation: ρ = %.3f", spear_result["r"])
    logger.info("(rank-based, less sensitive to outliers)")

    reg_result = linear_regression(cu_pos, zn_pos)
    logger.info(
        "Linear Regression: Zn = %.2f + %.4f * Cu",
        reg_result["intercept"],
        reg_result["slope"],
    )
    logger.info("R² = %.3f", reg_result["r_squared"])
    logger.info("p-value: %.4e", reg_result["p_value"])

    logger.info("Bootstrap 95% Confidence Intervals:")
    slope_ci = bootstrap_confidence_interval(
        cu_pos,
        zn_pos,
        statistic=lambda x, y: linear_regression(x, y)["slope"],
        n_iterations=1000,
        confidence=0.95,
    )
    logger.info("Slope: [%.4f, %.4f]", slope_ci["lower"], slope_ci["upper"])

# %% Visualization: Scatter Plot Matrix
logger.info("4. Creating Visualizations...")

plot_elements = ["Cu", "Pb", "Zn", "Mo"]
plot_data = data_clean[[e for e in plot_elements if e in data_clean.columns]].copy()

for col in plot_data.columns:
    plot_data[col] = plot_data[col].replace(0, np.nan)
    plot_data[f"log_{col}"] = np.log10(plot_data[col])

fig = plt.figure(figsize=(14, 14))
fig.suptitle(
    "Geochemical Scatter Plot Matrix (log₁₀ scale)",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

n = len([e for e in plot_elements if e in data_clean.columns])
log_cols = [f"log_{e}" for e in plot_elements if e in data_clean.columns]

for i, col1 in enumerate(log_cols):
    for j, col2 in enumerate(log_cols):
        ax = plt.subplot(n, n, i * n + j + 1)

        if i == j:
            data_plot = plot_data[col1].dropna()
            ax.hist(data_plot, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
            ax.set_ylabel("Frequency" if j == 0 else "", fontsize=9)
            if i == n - 1:
                elem = col1.replace("log_", "")
                ax.set_xlabel(f"log₁₀({elem})", fontsize=10, fontweight="bold")
        else:
            mask = plot_data[col1].notna() & plot_data[col2].notna()
            x = plot_data.loc[mask, col2].values
            y = plot_data.loc[mask, col1].values

            ax.scatter(x, y, alpha=0.3, s=10, c="steelblue", edgecolors="none")

            if len(x) > 10:
                try:
                    reg = linear_regression(x, y)
                    if reg["p_value"] < 0.01:
                        x_line = np.linspace(x.min(), x.max(), 100)
                        y_line = reg["intercept"] + reg["slope"] * x_line
                        ax.plot(x_line, y_line, "r-", linewidth=1.5, alpha=0.7)

                        ax.text(
                            0.05,
                            0.95,
                            f"R²={reg['r_squared']:.2f}",
                            transform=ax.transAxes,
                            fontsize=8,
                            verticalalignment="top",
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                        )
                except (KeyError, TypeError, AttributeError):
                    pass

            if j == 0:
                elem = col1.replace("log_", "")
                ax.set_ylabel(f"log₁₀({elem})", fontsize=10, fontweight="bold")
            else:
                ax.set_ylabel("", fontsize=9)

            if i == n - 1:
                elem = col2.replace("log_", "")
                ax.set_xlabel(f"log₁₀({elem})", fontsize=10, fontweight="bold")
            else:
                ax.set_xlabel("", fontsize=9)

        ax.grid(True, alpha=0.3)
        if i != n - 1:
            ax.set_xticklabels([])
        if j != 0:
            ax.set_yticklabels([])

plt.tight_layout()
plt.savefig("examples/geochem_02_bivariate.png", dpi=150, bbox_inches="tight")
logger.info("Saved: examples/geochem_02_bivariate.png")

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(correlation_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Pearson Correlation (r)", rotation=270, labelpad=20, fontsize=11)

ax.set_xticks(np.arange(len(elements)))
ax.set_yticks(np.arange(len(elements)))
ax.set_xticklabels(elements, fontsize=11)
ax.set_yticklabels(elements, fontsize=11)

for i in range(len(elements)):
    for j in range(len(elements)):
        if not np.isnan(correlation_matrix[i, j]):
            ax.text(
                j,
                i,
                f"{correlation_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

ax.set_title("Element Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("examples/geochem_02_correlation_matrix.png", dpi=150, bbox_inches="tight")
logger.info("Saved: examples/geochem_02_correlation_matrix.png")

log_section("SUMMARY")
logger.info("Correlation analysis reveals element associations:")
logger.info("• Strong correlations suggest common source or similar behavior")
logger.info("• Cu-Zn often correlate in porphyry and VMS deposits")
logger.info("• Log-transformation improves linear relationships")
logger.info("• Rank correlations (Spearman) handle outliers better")
logger.info("Next steps:")
logger.info("• Example 3: Spatial analysis and geochemical mapping")
logger.info("• Example 4: Multivariate analysis (PCA, clustering)")

plt.show()
