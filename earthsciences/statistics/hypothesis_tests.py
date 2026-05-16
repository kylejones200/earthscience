"""
Hypothesis testing

Statistical tests for earth sciences applications.
"""

import numpy as np
from scipy import stats


def t_test(
    sample1: np.ndarray,
    sample2: np.ndarray | None = None,
    mu: float = 0,
    alternative: str = "two-sided",
) -> dict:
    """
    Perform t-test (one-sample or two-sample).

    Parameters
    ----------
    sample1 : array_like
        First sample
    sample2 : array_like, optional
        Second sample (if None, perform one-sample test)
    mu : float, optional
        Hypothesized mean for one-sample test (default=0)
    alternative : str, optional
        Alternative hypothesis: 'two-sided', 'less', 'greater' (default='two-sided')

    Returns
    -------
    dict
        Dictionary containing:
        - statistic: t-statistic
        - p_value: p-value
        - dof: degrees of freedom

    Examples
    --------
    >>> # One-sample t-test
    >>> sample = np.random.randn(100) + 0.5
    >>> result = t_test(sample, mu=0)
    >>> print(f"t = {result['statistic']:.3f}, p = {result['p_value']:.3f}")
    >>>
    >>> # Two-sample t-test
    >>> sample1 = np.random.randn(50)
    >>> sample2 = np.random.randn(50) + 0.5
    >>> result = t_test(sample1, sample2)
    """

    sample1 = np.asarray(sample1)
    sample1 = sample1[~np.isnan(sample1)]

    if sample2 is None:
        # One-sample t-test
        statistic, p_value = stats.ttest_1samp(sample1, mu, alternative=alternative)
        dof = len(sample1) - 1
    else:
        # Two-sample t-test
        sample2 = np.asarray(sample2)
        sample2 = sample2[~np.isnan(sample2)]
        statistic, p_value = stats.ttest_ind(sample1, sample2, alternative=alternative)
        dof = len(sample1) + len(sample2) - 2

    return {
        "statistic": statistic,
        "p_value": p_value,
        "dof": dof,
        "significant": p_value < 0.05,
    }


def chi_square_test(observed: np.ndarray, expected: np.ndarray | None = None) -> dict:
    """
    Chi-square goodness-of-fit test.

    Parameters
    ----------
    observed : array_like
        Observed frequencies
    expected : array_like, optional
        Expected frequencies (if None, assume uniform distribution)

    Returns
    -------
    dict
        Dictionary containing:
        - statistic: chi-square statistic
        - p_value: p-value
        - dof: degrees of freedom

    Examples
    --------
    >>> # Test if a die is fair
    >>> observed = np.array([15, 18, 12, 20, 14, 21])
    >>> result = chi_square_test(observed)
    >>> print(f"Chi-square = {result['statistic']:.3f}, p = {result['p_value']:.3f}")
    """

    observed = np.asarray(observed)

    if expected is None:
        # Assume uniform distribution
        expected = np.full_like(observed, np.mean(observed), dtype=float)
    else:
        expected = np.asarray(expected, dtype=float)

    # Chi-square test
    statistic, p_value = stats.chisquare(observed, expected)
    dof = len(observed) - 1

    return {
        "statistic": statistic,
        "p_value": p_value,
        "dof": dof,
        "significant": p_value < 0.05,
    }


def f_test(sample1: np.ndarray, sample2: np.ndarray) -> dict:
    """
    F-test for equality of variances.

    Parameters
    ----------
    sample1, sample2 : array_like
        Input samples

    Returns
    -------
    dict
        Dictionary containing:
        - statistic: F-statistic
        - p_value: p-value (two-tailed)
        - dof1, dof2: degrees of freedom

    Examples
    --------
    >>> sample1 = np.random.randn(30)
    >>> sample2 = np.random.randn(40) * 2  # Different variance
    >>> result = f_test(sample1, sample2)
    >>> print(f"F = {result['statistic']:.3f}, p = {result['p_value']:.3f}")
    """
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)

    sample1 = sample1[~np.isnan(sample1)]
    sample2 = sample2[~np.isnan(sample2)]

    var1 = np.var(sample1, ddof=1)
    var2 = np.var(sample2, ddof=1)

    # F-statistic (larger variance in numerator)
    if var1 >= var2:
        f_stat = var1 / var2
        dof1 = len(sample1) - 1
        dof2 = len(sample2) - 1
    else:
        f_stat = var2 / var1
        dof1 = len(sample2) - 1
        dof2 = len(sample1) - 1

    # Two-tailed p-value
    p_value = 2 * min(stats.f.cdf(f_stat, dof1, dof2), 1 - stats.f.cdf(f_stat, dof1, dof2))

    return {
        "statistic": f_stat,
        "p_value": p_value,
        "dof1": dof1,
        "dof2": dof2,
        "significant": p_value < 0.05,
    }


def mann_whitney_test(
    sample1: np.ndarray, sample2: np.ndarray, alternative: str = "two-sided"
) -> dict:
    """
    Mann-Whitney U test (non-parametric alternative to t-test).

    Parameters
    ----------
    sample1, sample2 : array_like
        Input samples
    alternative : str, optional
        Alternative hypothesis: 'two-sided', 'less', 'greater'

    Returns
    -------
    dict
        Test results

    Examples
    --------
    >>> sample1 = np.random.exponential(1, 50)
    >>> sample2 = np.random.exponential(1.5, 50)
    >>> result = mann_whitney_test(sample1, sample2)
    """
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)

    sample1 = sample1[~np.isnan(sample1)]
    sample2 = sample2[~np.isnan(sample2)]

    statistic, p_value = stats.mannwhitneyu(sample1, sample2, alternative=alternative)

    return {
        "statistic": statistic,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def ks_test_2samp(sample1: np.ndarray, sample2: np.ndarray) -> dict:
    """
    Two-sample Kolmogorov-Smirnov test.

    Tests if two samples come from the same distribution.

    Parameters
    ----------
    sample1, sample2 : array_like
        Input samples

    Returns
    -------
    dict
        Test results

    Examples
    --------
    >>> sample1 = np.random.randn(100)
    >>> sample2 = np.random.randn(100) + 0.5
    >>> result = ks_test_2samp(sample1, sample2)
    """
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)

    sample1 = sample1[~np.isnan(sample1)]
    sample2 = sample2[~np.isnan(sample2)]

    statistic, p_value = stats.ks_2samp(sample1, sample2)

    return {
        "statistic": statistic,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def correlation_significance(r: float, n: int, alternative: str = "two-sided") -> float:
    """
    Test significance of correlation coefficient.

    Parameters
    ----------
    r : float
        Correlation coefficient
    n : int
        Sample size
    alternative : str, optional
        Alternative hypothesis: 'two-sided', 'less', 'greater'

    Returns
    -------
    float
        P-value

    Examples
    --------
    >>> r = 0.3
    >>> n = 50
    >>> p = correlation_significance(r, n)
    >>> print(f"Correlation r={r} with n={n}: p = {p:.3f}")
    """
    # Transform r to t-statistic
    t_stat = r * np.sqrt((n - 2) / (1 - r**2))

    # Calculate p-value
    if alternative == "two-sided":
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
    elif alternative == "greater":
        p_value = 1 - stats.t.cdf(t_stat, n - 2)
    elif alternative == "less":
        p_value = stats.t.cdf(t_stat, n - 2)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return p_value


def t_test_1samp(data: np.ndarray, popmean: float = 0, alternative: str = "two-sided") -> dict:
    """One-sample t-test."""
    result = stats.ttest_1samp(data, popmean, alternative=alternative)
    return {"statistic": result.statistic, "p_value": result.pvalue, "pvalue": result.pvalue}


def t_test_2samp(group1: np.ndarray, group2: np.ndarray, equal_var: bool = True) -> dict:
    """Two-sample independent t-test."""
    result = stats.ttest_ind(group1, group2, equal_var=equal_var)
    return {"statistic": result.statistic, "p_value": result.pvalue, "pvalue": result.pvalue}


def t_test_paired(before: np.ndarray, after: np.ndarray) -> dict:
    """Paired t-test."""
    result = stats.ttest_rel(before, after)
    return {"statistic": result.statistic, "p_value": result.pvalue, "pvalue": result.pvalue}


def anova_oneway(*groups) -> dict:
    """One-way ANOVA."""
    result = stats.f_oneway(*groups)
    return {
        "F_statistic": result.statistic,
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "pvalue": result.pvalue,
    }


def chi_square_independence(contingency_table: np.ndarray) -> dict:
    """Chi-square test of independence."""
    result = stats.chi2_contingency(contingency_table)
    return {
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "pvalue": result.pvalue,
        "dof": result.dof,
    }


def ks_test_1samp(data: np.ndarray, distribution: str = "norm") -> dict:
    """One-sample Kolmogorov-Smirnov test."""
    result = stats.kstest(data, distribution)
    return {"statistic": result.statistic, "p_value": result.pvalue, "pvalue": result.pvalue}


def mann_whitney_u(group1: np.ndarray, group2: np.ndarray) -> dict:
    """Mann-Whitney U test."""
    result = stats.mannwhitneyu(group1, group2)
    return {"statistic": result.statistic, "p_value": result.pvalue, "pvalue": result.pvalue}


def wilcoxon_test(before: np.ndarray, after: np.ndarray) -> dict:
    """Wilcoxon signed-rank test."""
    result = stats.wilcoxon(before, after)
    return {"statistic": result.statistic, "p_value": result.pvalue, "pvalue": result.pvalue}


def kruskal_wallis(*groups) -> dict:
    """Kruskal-Wallis H test."""
    result = stats.kruskal(*groups)
    return {
        "statistic": result.statistic,
        "H": result.statistic,
        "p_value": result.pvalue,
        "pvalue": result.pvalue,
    }


def ansari_bradley_test(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Ansari-Bradley test for difference in scale (dispersion) between two samples.

    Non-parametric test for comparing spreads/variances when distributions
    may not be normal. Useful when F-test assumptions are violated.

    Parameters
    ----------
    x, y : array_like
        Two independent samples
    alpha : float, optional
        Significance level (default=0.05)

    Returns
    -------
    dict
        Test results with statistic and p-value

    Notes
    -----
    Null hypothesis: The two samples have equal scale parameters.
    Alternative: The samples have different scale parameters.

    The test is based on ranks of the combined absolute deviations.
    More robust to outliers than the F-test.

    Examples
    --------
    >>> # Same spread
    >>> x = np.random.randn(50)
    >>> y = np.random.randn(50)
    >>> result = ansari_bradley_test(x, y)
    >>> print(f"Same spread - p-value: {result['p_value']:.3f}")
    >>>
    >>> # Different spreads
    >>> x = np.random.randn(50)
    >>> y = np.random.randn(50) * 3  # 3x larger spread
    >>> result = ansari_bradley_test(x, y)
    >>> print(f"Different spread - p-value: {result['p_value']:.3f}")
    """
    from scipy.stats import ansari

    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NaN values
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    if len(x) < 2 or len(y) < 2:
        raise ValueError("Need at least 2 samples in each group")

    # Perform Ansari-Bradley test
    statistic, p_value = ansari(x, y)

    # Calculate sample scales for interpretation
    scale_x = np.std(x, ddof=1)
    scale_y = np.std(y, ddof=1)

    return {
        "statistic": statistic,
        "p_value": p_value,
        "significant": p_value < alpha,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "scale_ratio": scale_y / scale_x if scale_x > 0 else np.inf,
        "test_type": "two-tailed",
        "method": "Ansari-Bradley",
    }
