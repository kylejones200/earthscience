"""
Multiple linear regression framework

Functions for multivariate regression analysis including stepwise selection,
regularization, and diagnostics.
"""

import logging

import numpy as np
from scipy import stats
from sklearn.linear_model import (
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LassoCV,
    LinearRegression,
    Ridge,
    RidgeCV,
)
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


def multiple_linear_regression(
    X: np.ndarray, y: np.ndarray, fit_intercept: bool = True, return_diagnostics: bool = True
) -> dict:
    """
    Multiple linear regression (OLS).

    Fits: y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε

    Parameters
    ----------
    X : array_like
        Predictor variables, shape (n_samples, n_features)
    y : array_like
        Response variable, shape (n_samples,)
    fit_intercept : bool, optional
        Include intercept term (default=True)
    return_diagnostics : bool, optional
        Calculate regression diagnostics (default=True)

    Returns
    -------
    dict
        Regression results including coefficients, R², and diagnostics

    Examples
    --------
    >>> # Simple multiple regression
    >>> X = np.random.randn(100, 3)
    >>> y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(100)*0.5
    >>>
    >>> result = multiple_linear_regression(X, y)
    >>> print(f"R² = {result['r_squared']:.3f}")
    >>> print(f"Coefficients: {result['coefficients']}")
    >>> print(f"F-statistic: {result['f_statistic']:.2f}, p={result['f_pvalue']:.4f}")
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples, n_features = X.shape

    # Fit model
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X, y)

    # Predictions and residuals
    y_pred = model.predict(X)
    residuals = y - y_pred

    # R-squared
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum(residuals**2)
    r_squared = 1 - (ss_residual / ss_total)

    # Adjusted R-squared
    n = len(y)
    p = n_features
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

    # Standard error
    mse = ss_residual / (n - p - 1)
    se = np.sqrt(mse)

    result = {
        "coefficients": model.coef_,
        "intercept": model.intercept_ if fit_intercept else 0.0,
        "predicted": y_pred,
        "residuals": residuals,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "rmse": np.sqrt(mse),
        "model": model,
    }

    if return_diagnostics:
        # F-statistic
        if p > 0:
            f_statistic = (r_squared / p) / ((1 - r_squared) / (n - p - 1))
            f_pvalue = 1 - stats.f.cdf(f_statistic, p, n - p - 1)
        else:
            f_statistic = np.nan
            f_pvalue = np.nan

        # Standard errors of coefficients
        if n > p + 1:
            # Variance-covariance matrix
            X_design = X if not fit_intercept else np.c_[np.ones(n), X]
            try:
                var_covar = mse * np.linalg.inv(X_design.T @ X_design)
                se_coefficients = np.sqrt(np.diag(var_covar))

                if fit_intercept:
                    se_intercept = se_coefficients[0]
                    se_coefficients = se_coefficients[1:]
                else:
                    se_intercept = np.nan

                # t-statistics and p-values
                t_stats = model.coef_ / se_coefficients
                p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))

            except np.linalg.LinAlgError:
                se_coefficients = np.full(p, np.nan)
                se_intercept = np.nan
                t_stats = np.full(p, np.nan)
                p_values = np.full(p, np.nan)
        else:
            se_coefficients = np.full(p, np.nan)
            se_intercept = np.nan
            t_stats = np.full(p, np.nan)
            p_values = np.full(p, np.nan)

        result.update(
            {
                "f_statistic": f_statistic,
                "f_pvalue": f_pvalue,
                "se_coefficients": se_coefficients,
                "se_intercept": se_intercept,
                "t_statistics": t_stats,
                "p_values": p_values,
            }
        )

    return result


def stepwise_regression(
    X: np.ndarray,
    y: np.ndarray,
    direction: str = "forward",
    criterion: str = "aic",
    max_features: int | None = None,
    threshold_in: float = 0.05,
    threshold_out: float = 0.1,
    verbose: bool = False,
) -> dict:
    """
    Stepwise variable selection for regression.

    Parameters
    ----------
    X : array_like
        Predictor variables
    y : array_like
        Response variable
    direction : str, optional
        'forward', 'backward', or 'both' (default='forward')
    criterion : str, optional
        Selection criterion: 'aic', 'bic', or 'pvalue' (default='aic')
    max_features : int, optional
        Maximum number of features to select
    threshold_in : float, optional
        P-value threshold for adding variables (default=0.05)
    threshold_out : float, optional
        P-value threshold for removing variables (default=0.1)
    verbose : bool, optional
        Print progress (default=False)

    Returns
    -------
    dict
        Selected features and model including:
        - selected_features: list of selected feature indices
        - model: trained model on selected features

    Warnings
    --------
    **DATA LEAKAGE PREVENTION**

    Feature selection is a form of model fitting that learns from data.
    **CRITICAL**: Only pass training data, never test data!

    Correct usage:

    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>>
    >>> # Feature selection on training data only
    >>> result = stepwise_regression(X_train, y_train, direction='forward')
    >>> selected_features = result['selected_features']
    >>>
    >>> # Apply same feature selection to test data
    >>> X_test_selected = X_test[:, selected_features]
    >>> y_pred = result['model'].predict(X_test_selected)

    Examples
    --------
    >>> X = np.random.randn(100, 10)
    >>> # Only first 3 features are relevant
    >>> y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(100)*0.5
    >>>
    >>> result = stepwise_regression(X, y, direction='forward', criterion='aic')
    >>> print(f"Selected features: {result['selected_features']}")
    >>> print(f"Final AIC: {result['final_aic']:.2f}")
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n_samples, n_features = X.shape

    if max_features is None:
        max_features = n_features

    # Initialize
    selected: list[int] = []
    remaining = list(range(n_features))

    def calculate_criterion(X_subset, y):
        """Calculate AIC, BIC, or p-value based criterion."""
        model = LinearRegression()
        model.fit(X_subset, y)
        y_pred = model.predict(X_subset)
        residuals = y - y_pred
        rss = np.sum(residuals**2)
        n = len(y)
        k = X_subset.shape[1] + 1  # +1 for intercept

        # Criterion calculation dispatch
        def calc_aic():
            return n * np.log(rss / n) + 2 * k

        def calc_bic():
            return n * np.log(rss / n) + k * np.log(n)

        def calc_pvalue():
            result = multiple_linear_regression(X_subset, y, return_diagnostics=True)
            return -np.log(result["f_pvalue"] + 1e-10)

        criterion_funcs = {
            "aic": calc_aic,
            "bic": calc_bic,
            "pvalue": calc_pvalue,
        }

        if criterion not in criterion_funcs:
            valid_criteria = ", ".join(f"'{c}'" for c in criterion_funcs.keys())
            raise ValueError(f"Unknown criterion '{criterion}'. Valid options: {valid_criteria}")

        return criterion_funcs[criterion]()

    if direction in ["forward", "both"]:
        # Forward selection
        current_score = np.inf

        while len(selected) < max_features and remaining:
            scores_with_candidates = []

            for candidate in remaining:
                test_features = selected + [candidate]
                X_subset = X[:, test_features]
                score = calculate_criterion(X_subset, y)
                scores_with_candidates.append((score, candidate))

            # Find best candidate
            scores_with_candidates.sort()
            best_new_score, best_candidate = scores_with_candidates[0]

            # Check if improvement
            if best_new_score < current_score:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
                current_score = best_new_score

                if verbose:
                    logger.info(f"Added feature {best_candidate}, {criterion}={current_score:.2f}")
            else:
                break

    elif direction == "backward":
        # Start with all features
        selected = list(range(n_features))

        while len(selected) > 1:
            scores_without_features = []

            for feature in selected:
                test_features = [f for f in selected if f != feature]
                X_subset = X[:, test_features]
                score = calculate_criterion(X_subset, y)
                scores_without_features.append((score, feature))

            # Find which removal gives best score
            scores_without_features.sort()
            best_score, worst_feature = scores_without_features[0]

            # Current score
            X_subset = X[:, selected]
            current_score = calculate_criterion(X_subset, y)

            # Remove if improves or doesn't harm much
            if best_score <= current_score:
                selected.remove(worst_feature)

                if verbose:
                    logger.info(f"Removed feature {worst_feature}, {criterion}={best_score:.2f}")
            else:
                break

    # Final model with selected features
    X_selected = X[:, selected]
    final_result = multiple_linear_regression(X_selected, y, return_diagnostics=True)

    return {
        "selected_features": selected,
        "n_features": len(selected),
        "final_aic": calculate_criterion(X_selected, y) if criterion == "aic" else None,
        "model": final_result["model"],
        "coefficients": final_result["coefficients"],
        "r_squared": final_result["r_squared"],
        "adj_r_squared": final_result["adj_r_squared"],
    }


def ridge_regression(
    X: np.ndarray, y: np.ndarray, alpha: float = 1.0, cv: int = 5, auto_alpha: bool = False
) -> dict:
    """
    Ridge regression (L2 regularization).

    Adds penalty: λ * Σβ²ᵢ to prevent overfitting.

    Parameters
    ----------
    X : array_like
        Predictor variables
    y : array_like
        Response variable
    alpha : float, optional
        Regularization strength (default=1.0)
    cv : int, optional
        Cross-validation folds for automatic alpha selection (default=5)
    auto_alpha : bool, optional
        Automatically select optimal alpha via CV (default=False)

    Returns
    -------
    dict
        Ridge regression results

    Warnings
    --------
    **DATA LEAKAGE PREVENTION**

    When auto_alpha=True, hyperparameter selection uses cross-validation.
    **CRITICAL**: Only pass training data to this function, not test data!

    Incorrect usage (causes data leakage):

    >>> # ❌ WRONG - test data leaks into alpha selection
    >>> X_all = np.concatenate([X_train, X_test])
    >>> y_all = np.concatenate([y_train, y_test])
    >>> result = ridge_regression(X_all, y_all, auto_alpha=True)  # Leakage!

    Correct usage (no leakage):

    >>> # ✅ CORRECT - fit on training data only
    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>>
    >>> # Automatic alpha selection on training data only
    >>> result = ridge_regression(X_train, y_train, auto_alpha=True, cv=5)
    >>> optimal_alpha = result['alpha']
    >>>
    >>> # Use trained model to predict test data
    >>> y_pred = result['model'].predict(X_test)
    >>>
    >>> # Or manually fit with the optimal alpha
    >>> from sklearn.linear_model import Ridge
    >>> final_model = Ridge(alpha=optimal_alpha)
    >>> final_model.fit(X_train, y_train)
    >>> y_pred = final_model.predict(X_test)

    Examples
    --------
    >>> X = np.random.randn(100, 20)  # More features than typical
    >>> y = X[:, :3] @ np.array([2, 3, -1]) + np.random.randn(100)*0.5
    >>>
    >>> # Manual alpha
    >>> result = ridge_regression(X, y, alpha=1.0)
    >>> print(f"R² = {result['r_squared']:.3f}")
    >>>
    >>> # Automatic alpha selection
    >>> result_auto = ridge_regression(X, y, auto_alpha=True, cv=5)
    >>> print(f"Optimal alpha: {result_auto['alpha']:.3f}")
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if auto_alpha:
        # Cross-validation to find optimal alpha
        model = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=cv)
        model.fit(X, y)
        alpha_used = model.alpha_
    else:
        model = Ridge(alpha=alpha)
        model.fit(X, y)
        alpha_used = alpha

    # Predictions
    y_pred = model.predict(X)
    residuals = y - y_pred

    # R-squared
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum(residuals**2)
    r_squared = 1 - (ss_residual / ss_total)

    return {
        "coefficients": model.coef_,
        "intercept": model.intercept_,
        "alpha": alpha_used,
        "predicted": y_pred,
        "residuals": residuals,
        "r_squared": r_squared,
        "model": model,
    }


def lasso_regression(
    X: np.ndarray, y: np.ndarray, alpha: float = 1.0, cv: int = 5, auto_alpha: bool = False
) -> dict:
    """
    Lasso regression (L1 regularization).

    Adds penalty: λ * Σ|βᵢ| which performs feature selection
    by driving some coefficients to exactly zero.

    Parameters
    ----------
    X : array_like
        Predictor variables
    y : array_like
        Response variable
    alpha : float, optional
        Regularization strength (default=1.0)
    cv : int, optional
        Cross-validation folds (default=5)
    auto_alpha : bool, optional
        Automatically select optimal alpha (default=False)

    Returns
    -------
    dict
        Lasso regression results including:
        - selected_features: indices of non-zero coefficients
        - n_nonzero_coef: number of selected features

    Warnings
    --------
    **DATA LEAKAGE PREVENTION**

    When auto_alpha=True, both hyperparameter AND feature selection occur.
    **CRITICAL**: Only pass training data, never include test data!

    See ridge_regression() docstring for detailed examples of correct usage.

    Examples
    --------
    >>> X = np.random.randn(100, 20)
    >>> # Only 3 features are relevant
    >>> y = X[:, :3] @ np.array([2, 3, -1]) + np.random.randn(100)*0.5
    >>>
    >>> result = lasso_regression(X, y, auto_alpha=True)
    >>> print(f"Non-zero coefficients: {np.sum(result['coefficients'] != 0)}")
    >>> print(f"R² = {result['r_squared']:.3f}")
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if auto_alpha:
        model = LassoCV(alphas=None, cv=cv, max_iter=10000)
        model.fit(X, y)
        alpha_used = model.alpha_
    else:
        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(X, y)
        alpha_used = alpha

    # Predictions
    y_pred = model.predict(X)
    residuals = y - y_pred

    # R-squared
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum(residuals**2)
    r_squared = 1 - (ss_residual / ss_total)

    # Count non-zero coefficients
    n_nonzero = np.sum(model.coef_ != 0)

    return {
        "coefficients": model.coef_,
        "intercept": model.intercept_,
        "alpha": alpha_used,
        "predicted": y_pred,
        "residuals": residuals,
        "r_squared": r_squared,
        "n_nonzero_coef": n_nonzero,
        "selected_features": np.where(model.coef_ != 0)[0],
        "model": model,
    }


def elastic_net_regression(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    cv: int = 5,
    auto_alpha: bool = False,
) -> dict:
    """
    Elastic Net regression (L1 + L2 regularization).

    Combines Ridge and Lasso: λ₁ * Σ|βᵢ| + λ₂ * Σβ²ᵢ

    Parameters
    ----------
    X : array_like
        Predictor variables
    y : array_like
        Response variable
    alpha : float, optional
        Overall regularization strength (default=1.0)
    l1_ratio : float, optional
        Balance between L1 and L2 (0=Ridge, 1=Lasso) (default=0.5)
    cv : int, optional
        Cross-validation folds (default=5)
    auto_alpha : bool, optional
        Automatically select optimal alpha (default=False)

    Returns
    -------
    dict
        Elastic Net results

    Warnings
    --------
    **DATA LEAKAGE PREVENTION**

    When auto_alpha=True, hyperparameter AND feature selection occur.
    Only pass training data! See ridge_regression() for detailed examples.

    Examples
    --------
    >>> X = np.random.randn(100, 20)
    >>> y = X[:, :3] @ np.array([2, 3, -1]) + np.random.randn(100)*0.5
    >>>
    >>> result = elastic_net_regression(X, y, l1_ratio=0.5, auto_alpha=True)
    >>> print(f"Selected {result['n_nonzero_coef']} features")
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if auto_alpha:
        model = ElasticNetCV(l1_ratio=l1_ratio, cv=cv, max_iter=10000)
        model.fit(X, y)
        alpha_used = model.alpha_
    else:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
        model.fit(X, y)
        alpha_used = alpha

    # Predictions
    y_pred = model.predict(X)
    residuals = y - y_pred

    # R-squared
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum(residuals**2)
    r_squared = 1 - (ss_residual / ss_total)

    n_nonzero = np.sum(model.coef_ != 0)

    return {
        "coefficients": model.coef_,
        "intercept": model.intercept_,
        "alpha": alpha_used,
        "l1_ratio": l1_ratio,
        "predicted": y_pred,
        "residuals": residuals,
        "r_squared": r_squared,
        "n_nonzero_coef": n_nonzero,
        "selected_features": np.where(model.coef_ != 0)[0],
        "model": model,
    }


def polynomial_regression_multivariate(
    X: np.ndarray, y: np.ndarray, degree: int = 2, interaction_only: bool = False
) -> dict:
    """
    Multivariate polynomial regression.

    Creates polynomial and interaction features up to specified degree.

    Parameters
    ----------
    X : array_like
        Predictor variables
    y : array_like
        Response variable
    degree : int, optional
        Polynomial degree (default=2)
    interaction_only : bool, optional
        Only include interaction terms (default=False)

    Returns
    -------
    dict
        Polynomial regression results

    Examples
    --------
    >>> X = np.random.randn(100, 2)
    >>> # Quadratic relationship
    >>> y = 2*X[:, 0]**2 + 3*X[:, 1]**2 + X[:, 0]*X[:, 1] + np.random.randn(100)*0.5
    >>>
    >>> result = polynomial_regression_multivariate(X, y, degree=2)
    >>> print(f"R² = {result['r_squared']:.3f}")
    >>> print(f"Features: {result['n_features']} -> {result['n_poly_features']}")
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Fit model
    result = multiple_linear_regression(X_poly, y, return_diagnostics=True)

    result.update(
        {
            "degree": degree,
            "n_features": X.shape[1],
            "n_poly_features": X_poly.shape[1],
            "feature_names": (
                poly.get_feature_names_out() if hasattr(poly, "get_feature_names_out") else None
            ),
            "poly_transformer": poly,
        }
    )

    return result


def vif_analysis(X: np.ndarray, feature_names: list[str] | None = None) -> dict:
    """
    Variance Inflation Factor (VIF) analysis for multicollinearity detection.

    VIF > 10 indicates high multicollinearity.
    VIF > 5 may be concerning depending on context.

    Parameters
    ----------
    X : array_like
        Predictor variables
    feature_names : list, optional
        Names of features for reporting

    Returns
    -------
    dict
        VIF values for each feature

    Examples
    --------
    >>> X = np.random.randn(100, 5)
    >>> # Add highly correlated feature
    >>> X[:, 4] = X[:, 0] + np.random.randn(100)*0.1
    >>>
    >>> vif = vif_analysis(X, feature_names=['A', 'B', 'C', 'D', 'E'])
    >>> for name, value in zip(vif['feature_names'], vif['vif']):
    ...     print(f"{name}: VIF = {value:.2f}")
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f"X{i}" for i in range(n_features)]

    # Calculate VIF for each feature
    vif_values = []
    for i in range(n_features):
        try:
            vif = variance_inflation_factor(X, i)
        except:
            vif = np.nan
        vif_values.append(vif)

    return {
        "vif": np.array(vif_values),
        "feature_names": feature_names,
        "high_vif_features": [feature_names[i] for i, v in enumerate(vif_values) if v > 10],
    }
