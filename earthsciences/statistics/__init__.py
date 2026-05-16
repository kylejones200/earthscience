"""
Statistical analysis recipes (Chapters 3-4)

Univariate and bivariate statistics for earth sciences.
"""

from .bivariate import *
from .distributions import *
from .extreme_values import *
from .hypothesis_tests import *
from .multiple_regression import *
from .resampling import *
from .robust_stats import *
from .univariate import *

__all__ = [
    # Univariate
    "descriptive_stats",
    "percentiles",
    "mode_estimate",
    "skewness",
    "kurtosis",
    # Bivariate
    "correlation",
    "linear_regression",
    "rma_regression",
    "polynomial_fit",
    # Distributions
    "fit_distribution",
    "test_normality",
    "generate_random",
    # Hypothesis tests
    "t_test",
    "chi_square_test",
    # Resampling
    "bootstrap",
    "jackknife",
    "cross_validation",
    # Robust statistics
    "median_absolute_deviation",
    "detect_outliers_iqr",
    "robust_regression",
    # Extreme values
    "fit_gev",
    "return_level",
    "gev_return_level",
    "fit_gpd",
]
