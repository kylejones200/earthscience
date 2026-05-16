"""
Resampling methods

Bootstrap, jackknife, permutation tests, and cross-validation.
"""

from collections.abc import Callable

import numpy as np


def bootstrap(
    data: np.ndarray,
    statistic: Callable = np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int | None = None,
) -> dict:
    """
    Bootstrap resampling for estimating sampling distribution.

    Parameters
    ----------
    data : array_like
        Input data
    statistic : callable
        Function to compute statistic (default=np.mean)
    n_bootstrap : int
        Number of bootstrap samples (default=1000)
    confidence : float
        Confidence level for CI (default=0.95)
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Dictionary containing:
        - estimate: point estimate
        - confidence_interval or ci: (lower, upper)
        - bootstrap_samples or samples: array of bootstrap statistics
        - standard_error: standard error
    """
    if random_state is not None:
        np.random.seed(random_state)

    data = np.asarray(data)
    n = len(data)

    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic(sample)

    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return {
        "estimate": statistic(data),
        "confidence_interval": (ci_lower, ci_upper),
        "ci": (ci_lower, ci_upper),
        "bootstrap_samples": bootstrap_stats,
        "samples": bootstrap_stats,
        "standard_error": np.std(bootstrap_stats),
    }


def jackknife(data: np.ndarray, statistic: Callable = np.mean) -> dict:
    """
    Jackknife resampling for bias and variance estimation.

    Parameters
    ----------
    data : array_like
        Input data
    statistic : callable
        Function to compute statistic

    Returns
    -------
    dict
        Dictionary containing:
        - estimate: point estimate
        - bias: jackknife bias estimate
        - standard_error or se: standard error
        - jackknife_samples: leave-one-out statistics
    """
    data = np.asarray(data)
    n = len(data)

    full_stat = statistic(data)

    jackknife_stats = np.zeros(n)
    for i in range(n):
        sample = np.delete(data, i)
        jackknife_stats[i] = statistic(sample)

    jackknife_mean = np.mean(jackknife_stats)
    bias = (n - 1) * (jackknife_mean - full_stat)

    variance = ((n - 1) / n) * np.sum((jackknife_stats - jackknife_mean) ** 2)
    se = np.sqrt(variance)

    return {
        "estimate": full_stat,
        "bias": bias,
        "standard_error": se,
        "se": se,
        "jackknife_samples": jackknife_stats,
    }


def permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    statistic: Callable | None = None,
    n_permutations: int = 1000,
    random_state: int | None = None,
) -> dict:
    """
    Permutation test for comparing two groups.

    Parameters
    ----------
    group1, group2 : array_like
        Two groups to compare
    statistic : callable, optional
        Test statistic (default=difference of means)
    n_permutations : int
        Number of permutations (default=1000)
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Dictionary containing:
        - p_value or pvalue: p-value
        - statistic or test_statistic: observed test statistic
        - null_distribution: permutation distribution
    """
    if random_state is not None:
        np.random.seed(random_state)

    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    if statistic is None:

        def statistic(g1, g2):
            return np.mean(g1) - np.mean(g2)

    observed_stat = statistic(group1, group2)

    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    n_total = len(combined)

    perm_stats = np.zeros(n_permutations)
    for i in range(n_permutations):
        permuted = np.random.permutation(combined)
        perm_group1 = permuted[:n1]
        perm_group2 = permuted[n1:]
        perm_stats[i] = statistic(perm_group1, perm_group2)

    p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))

    return {
        "p_value": p_value,
        "pvalue": p_value,
        "statistic": observed_stat,
        "test_statistic": observed_stat,
        "null_distribution": perm_stats,
    }


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    method: str = "kfold",
    model: Callable | None = None,
) -> dict:
    """
    Cross-validation for model evaluation.

    Parameters
    ----------
    X : array_like
        Feature matrix
    y : array_like
        Target values
    k : int
        Number of folds (default=5)
    method : str
        'kfold' or 'loo' (leave-one-out)
    model : callable, optional
        Model function (default=linear regression)

    Returns
    -------
    dict
        Dictionary containing:
        - scores or cv_scores: array of scores
        - mean_score: mean cross-validation score
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)

    if model is None:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        def default_model(X_train, y_train, X_test, y_test):
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return r2_score(y_test, y_pred)

        model = default_model

    if method == "loo":
        k = n

    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_sizes = np.full(k, n // k)
    fold_sizes[: n % k] += 1

    scores = []
    current = 0
    for fold_size in fold_sizes:
        test_indices = indices[current : current + fold_size]
        train_indices = np.concatenate([indices[:current], indices[current + fold_size :]])

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        score = model(X_train, y_train, X_test, y_test)
        scores.append(score)

        current += fold_size

    scores_arr = np.asarray(scores, dtype=float)

    return {"scores": scores_arr, "cv_scores": scores_arr, "mean_score": np.mean(scores_arr)}


def monte_carlo(
    simulation_func: Callable,
    n_simulations: int = 1000,
    confidence: float = 0.95,
    random_state: int | None = None,
) -> dict:
    """
    Monte Carlo simulation.

    Parameters
    ----------
    simulation_func : callable
        Function that returns one simulation result
    n_simulations : int
        Number of simulations (default=1000)
    confidence : float
        Confidence level (default=0.95)
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Dictionary containing:
        - results or simulations: array of simulation results
        - mean or estimate: mean result
        - confidence_interval or ci: confidence interval
        - standard_deviation: standard deviation
    """
    if random_state is not None:
        np.random.seed(random_state)

    results = np.array([simulation_func() for _ in range(n_simulations)])

    alpha = 1 - confidence
    ci_lower = np.percentile(results, 100 * alpha / 2)
    ci_upper = np.percentile(results, 100 * (1 - alpha / 2))

    return {
        "results": results,
        "simulations": results,
        "mean": np.mean(results),
        "estimate": np.mean(results),
        "confidence_interval": (ci_lower, ci_upper),
        "ci": (ci_lower, ci_upper),
        "standard_deviation": np.std(results),
    }
