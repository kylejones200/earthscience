"""
Complete geostatistical analysis pipeline.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from earthsciences.utils.logging_config import log_section

logger = logging.getLogger(__name__)


class GeostatsPipeline:
    """
    Complete geostatistical analysis pipeline.

    Executes all steps of a geostatistical workflow from
    data loading through visualization and output generation.

    Parameters
    ----------
    config : AnalysisConfig
        Configuration object with all analysis parameters

    Attributes
    ----------
    data : dict
        Loaded and preprocessed data
    variogram_model : object
        Fitted variogram model
    kriging_result : dict
        Kriging predictions and variance
    validation_result : dict
        Cross-validation metrics
    results : dict
        All analysis results
    """

    def __init__(self, config):
        self.config = config
        self.data = None
        self.data_raw = None
        self.transformer = None
        self.variogram_model = None
        self.variogram_data = None
        self.kriging_result = None
        self.validation_result = None
        self.plots = []
        self.results = {}

    def load_data(self):
        """Load data from input file."""
        log_section("STEP 1: LOADING DATA")

        input_file = Path(self.config.data.input_file).expanduser()

        # Read data file
        if input_file.suffix == ".csv":
            df = pd.read_csv(input_file)
        elif input_file.suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(input_file)
        else:
            raise ValueError(f"Unsupported file format: {input_file.suffix}")

        # Extract columns
        try:
            x = df[self.config.data.x_column].values
            y = df[self.config.data.y_column].values
            z = df[self.config.data.z_column].values
        except KeyError as e:
            raise ValueError(f"Column not found in data file: {e}")

        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        if not mask.all():
            n_removed = (~mask).sum()
            logger.warning(f"  Warning: Removed {n_removed} rows with NaN values")
            x, y, z = x[mask], y[mask], z[mask]

        self.data_raw = {"x": x, "y": y, "z": z}
        self.data = {"x": x.copy(), "y": y.copy(), "z": z.copy()}

        logger.info(f"✓ Loaded {len(x)} data points from {input_file.name}")
        logger.info(f"  X range: [{x.min():.2f}, {x.max():.2f}]")
        logger.info(f"  Y range: [{y.min():.2f}, {y.max():.2f}]")
        logger.info(f"  Z range: [{z.min():.2f}, {z.max():.2f}]")

        self.results["data_stats"] = {
            "n_points": len(x),
            "x_range": (float(x.min()), float(x.max())),
            "y_range": (float(y.min()), float(y.max())),
            "z_range": (float(z.min()), float(z.max())),
            "z_mean": float(np.mean(z)),
            "z_std": float(np.std(z)),
        }

    def preprocess(self):
        """Apply preprocessing steps."""
        if self.config.preprocessing is None:
            return

        log_section("STEP 2: PREPROCESSING")

        z = self.data["z"].copy()
        n_original = len(z)

        # Remove outliers
        if self.config.preprocessing.remove_outliers:
            from earthsciences.statistics import robust_stats

            method = self.config.preprocessing.outlier_method
            threshold = self.config.preprocessing.outlier_threshold

            if method == "iqr":
                q1, q3 = np.percentile(z, [25, 75])
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                mask = (z >= lower) & (z <= upper)
            elif method == "zscore":
                z_scores = np.abs((z - np.mean(z)) / np.std(z))
                mask = z_scores < threshold
            elif method == "mad":
                mad = robust_stats.mad(z)
                median = np.median(z)
                modified_z = 0.6745 * (z - median) / mad
                mask = np.abs(modified_z) < threshold

            n_outliers = (~mask).sum()
            if n_outliers > 0:
                self.data = {k: v[mask] for k, v in self.data.items()}
                z = self.data["z"]
                logger.info(f"✓ Removed {n_outliers} outliers using {method} method")
                logger.info(
                    f"  Retained {len(z)} / {n_original} points ({100*len(z)/n_original:.1f}%)"
                )

        # Transform data
        if self.config.preprocessing.transform:
            transform_name = self.config.preprocessing.transform

            if transform_name == "log":
                if np.any(z <= 0):
                    if self.config.preprocessing.handle_negatives:
                        shift = abs(z.min()) + 1e-6
                        z_transformed = np.log(z + shift)
                        logger.info(f"✓ Applied log transform (shifted by {shift:.6f})")
                        self.transformer = {"type": "log", "shift": shift}
                    else:
                        raise ValueError("Cannot apply log transform to non-positive values")
                else:
                    z_transformed = np.log(z)
                    logger.info("✓ Applied log transform")
                    self.transformer = {"type": "log", "shift": 0}

            elif transform_name == "boxcox":
                from scipy.stats import boxcox

                if np.any(z <= 0):
                    shift = abs(z.min()) + 1e-6
                    z_transformed, lambda_param = boxcox(z + shift)
                    logger.info(
                        f"✓ Applied Box-Cox transform (λ={lambda_param:.3f}, shift={shift:.6f})"
                    )
                    self.transformer = {"type": "boxcox", "lambda": lambda_param, "shift": shift}
                else:
                    z_transformed, lambda_param = boxcox(z)
                    logger.info(f"✓ Applied Box-Cox transform (λ={lambda_param:.3f})")
                    self.transformer = {"type": "boxcox", "lambda": lambda_param, "shift": 0}

            elif transform_name == "normal_score":
                from scipy.stats import norm

                ranks = np.argsort(np.argsort(z))
                uniform = (ranks + 1) / (len(z) + 1)
                z_transformed = norm.ppf(uniform)
                logger.info("✓ Applied normal score transform")
                self.transformer = {"type": "normal_score", "z_original": z.copy()}

            self.data["z_transformed"] = z_transformed
            logger.info(
                f"  Transformed range: [{z_transformed.min():.2f}, {z_transformed.max():.2f}]"
            )

        self.results["preprocessing"] = {
            "n_original": n_original,
            "n_after_preprocessing": len(self.data["z"]),
            "outliers_removed": n_original - len(self.data["z"]),
            "transform": (
                self.config.preprocessing.transform if self.config.preprocessing.transform else None
            ),
        }

    def fit_variogram(self):
        """Compute and fit variogram model."""
        log_section("STEP 3: VARIOGRAM MODELING")

        from earthsciences.spatial import compute_variogram, fit_variogram_model

        # Use transformed data if available
        z = self.data.get("z_transformed", self.data["z"])

        # Compute experimental variogram
        lags, gamma, n_pairs = compute_variogram(
            self.data["x"],
            self.data["y"],
            z,
            n_lags=self.config.variogram.n_lags,
            max_lag=self.config.variogram.max_lag,
        )

        logger.info("✓ Computed experimental variogram")
        logger.info(f"  Lags: {len(lags)}")
        logger.info(f"  Max lag: {lags[-1]:.2f}")
        logger.info(f"  Sill (approximate): {gamma[-1]:.4f}")

        # Fit model
        if self.config.variogram.auto_fit:
            best_model = None
            best_score = np.inf

            for model_name in self.config.variogram.models:
                fit_result = fit_variogram_model(lags, gamma, model=model_name)

                if fit_result["r_squared"] is not None:
                    score = 1 - fit_result["r_squared"]  # Lower is better
                    if score < best_score:
                        best_score = score
                        best_model = fit_result
                        best_model["model_name"] = model_name

            if best_model is None:
                raise ValueError("No variogram model could be fitted")

            self.variogram_model = best_model
            logger.info(f"✓ Best model: {best_model['model_name'].capitalize()}")
            logger.info(f"  Nugget: {best_model.get('nugget', 0):.4f}")
            logger.info(f"  Sill: {best_model.get('sill', 0):.4f}")
            logger.info(f"  Range: {best_model.get('range', 0):.2f}")
            logger.info(f"  R²: {best_model.get('r_squared', 0):.4f}")

        self.variogram_data = {"lags": lags, "gamma": gamma, "n_pairs": n_pairs}

        self.results["variogram"] = {
            "lags": lags.tolist(),
            "gamma": gamma.tolist(),
            "model": self.variogram_model["model_name"] if self.variogram_model else None,
            "parameters": (
                {
                    "nugget": float(self.variogram_model.get("nugget", 0)),
                    "sill": float(self.variogram_model.get("sill", 0)),
                    "range": float(self.variogram_model.get("range", 0)),
                }
                if self.variogram_model
                else None
            ),
        }

    def krige(self):
        """Perform kriging interpolation."""
        log_section("STEP 4: KRIGING INTERPOLATION")

        from earthsciences.spatial import ordinary_kriging
        from earthsciences.spatial.variogram import (
            exponential_model,
            gaussian_model,
            spherical_model,
        )

        # Create prediction grid
        grid_config = self.config.kriging.grid
        x_grid = np.arange(grid_config.x_min, grid_config.x_max, grid_config.resolution)
        y_grid = np.arange(grid_config.y_min, grid_config.y_max, grid_config.resolution)
        X, Y = np.meshgrid(x_grid, y_grid)

        logger.info(f"  Grid size: {X.shape[1]} × {X.shape[0]}")
        logger.info(f"  Total points: {X.size:,}")

        # Use transformed data if available
        z = self.data.get("z_transformed", self.data["z"])

        # Create variogram function from model
        model_name = self.variogram_model.get("model_name", "spherical")
        nugget = self.variogram_model.get("nugget", 0)
        sill = self.variogram_model.get("sill", 1)
        range_param = self.variogram_model.get("range", 1)

        if model_name == "spherical":
            variogram_func = lambda h: spherical_model(h, nugget, sill, range_param)
        elif model_name == "exponential":
            variogram_func = lambda h: exponential_model(h, nugget, sill, range_param)
        elif model_name == "gaussian":
            variogram_func = lambda h: gaussian_model(h, nugget, sill, range_param)
        else:
            variogram_func = lambda h: spherical_model(h, nugget, sill, range_param)

        # Perform kriging
        result = ordinary_kriging(
            self.data["x"], self.data["y"], z, X.ravel(), Y.ravel(), variogram_func
        )

        # Handle different return types
        if isinstance(result, tuple) and len(result) == 3:
            z_pred, z_var, _ = result
        elif isinstance(result, tuple) and len(result) == 2:
            z_pred, z_var = result
        else:
            z_pred = result
            z_var = np.zeros_like(z_pred)

        Z_pred = z_pred.reshape(X.shape)
        Z_var = z_var.reshape(X.shape)

        # Back-transform if needed
        if self.transformer:
            if self.transformer["type"] == "log":
                Z_pred_orig = np.exp(Z_pred) - self.transformer["shift"]
            elif self.transformer["type"] == "boxcox":
                # Inverse Box-Cox
                lambda_param = self.transformer["lambda"]
                shift = self.transformer["shift"]
                if abs(lambda_param) < 1e-10:
                    Z_pred_orig = np.exp(Z_pred) - shift
                else:
                    Z_pred_orig = (Z_pred * lambda_param + 1) ** (1 / lambda_param) - shift
            else:
                Z_pred_orig = Z_pred  # Can't easily back-transform normal score
        else:
            Z_pred_orig = Z_pred

        self.kriging_result = {
            "X": X,
            "Y": Y,
            "Z_pred": Z_pred_orig,
            "Z_pred_transformed": Z_pred,
            "Z_var": Z_var,
        }

        logger.info("✓ Kriging complete")
        logger.info(f"  Prediction range: [{Z_pred_orig.min():.2f}, {Z_pred_orig.max():.2f}]")
        logger.info(f"  Mean prediction: {Z_pred_orig.mean():.2f}")
        logger.info(f"  Mean variance: {Z_var.mean():.4f}")

        self.results["kriging"] = {
            "grid_shape": X.shape,
            "n_predictions": int(X.size),
            "pred_range": (float(Z_pred_orig.min()), float(Z_pred_orig.max())),
            "mean_pred": float(Z_pred_orig.mean()),
            "mean_variance": float(Z_var.mean()),
        }

    def validate(self):
        """Perform cross-validation."""
        if self.config.validation is None or not self.config.validation.cross_validation:
            return

        log_section("STEP 5: CROSS-VALIDATION")

        from sklearn.model_selection import KFold

        from earthsciences.spatial import ordinary_kriging
        from earthsciences.spatial.variogram import (
            exponential_model,
            gaussian_model,
            spherical_model,
        )

        z = self.data.get("z_transformed", self.data["z"])
        n_folds = self.config.validation.n_folds

        # Create variogram function
        model_name = self.variogram_model.get("model_name", "spherical")
        nugget = self.variogram_model.get("nugget", 0)
        sill = self.variogram_model.get("sill", 1)
        range_param = self.variogram_model.get("range", 1)

        if model_name == "spherical":
            variogram_func = lambda h: spherical_model(h, nugget, sill, range_param)
        elif model_name == "exponential":
            variogram_func = lambda h: exponential_model(h, nugget, sill, range_param)
        elif model_name == "gaussian":
            variogram_func = lambda h: gaussian_model(h, nugget, sill, range_param)
        else:
            variogram_func = lambda h: spherical_model(h, nugget, sill, range_param)

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        y_true = []
        y_pred = []

        for fold, (train_idx, test_idx) in enumerate(kfold.split(z), 1):
            # Training data
            x_train = self.data["x"][train_idx]
            y_train = self.data["y"][train_idx]
            z_train = z[train_idx]

            # Test data
            x_test = self.data["x"][test_idx]
            y_test = self.data["y"][test_idx]
            z_test = z[test_idx]

            # Predict
            z_pred_fold = ordinary_kriging(
                x_train, y_train, z_train, x_test, y_test, variogram_func
            )

            y_true.extend(z_test)
            y_pred.extend(z_pred_fold)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Compute metrics
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

        self.validation_result = {
            "y_true": y_true,
            "y_pred": y_pred,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

        logger.info(f"✓ {n_folds}-fold cross-validation complete")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  R²: {r2:.4f}")

        self.results["validation"] = {
            "n_folds": n_folds,
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
        }

    def visualize(self):
        """Create visualizations."""
        if self.config.visualization is None:
            return

        log_section("STEP 6: VISUALIZATION")

        from pathlib import Path

        from earthsciences.utils.plot_style import use_earthsciences_style

        use_earthsciences_style()
        use_clean_style = True

        output_dir = Path(self.config.project.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        for plot_type in self.config.visualization.plots:
            if plot_type == "variogram" and self.variogram_data:
                self._plot_variogram(output_dir, use_clean_style)
            elif plot_type == "kriging_map" and self.kriging_result:
                self._plot_kriging_map(output_dir, use_clean_style)
            elif plot_type == "variance_map" and self.kriging_result:
                self._plot_variance_map(output_dir, use_clean_style)
            elif plot_type == "cross_validation" and self.validation_result:
                self._plot_cross_validation(output_dir, use_clean_style)
            elif plot_type == "histogram":
                self._plot_histogram(output_dir, use_clean_style)

    def _plot_variogram(self, output_dir, use_clean_style):
        """Plot variogram."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        # Experimental variogram
        ax.plot(
            self.variogram_data["lags"],
            self.variogram_data["gamma"],
            "o",
            color="#1976D2",
            markersize=8,
            label="Experimental",
            markeredgecolor="black",
            markeredgewidth=1,
        )

        # Model
        if self.variogram_model:
            lags_fine = np.linspace(0, self.variogram_data["lags"].max(), 100)
            from earthsciences.spatial.variogram import (
                exponential_model,
                gaussian_model,
                spherical_model,
            )

            model_name = self.variogram_model.get("model_name", "spherical")
            nugget = self.variogram_model.get("nugget", 0)
            sill = self.variogram_model.get("sill", 1)
            range_param = self.variogram_model.get("range", 1)

            if model_name == "spherical":
                gamma_model = spherical_model(lags_fine, nugget, sill, range_param)
            elif model_name == "exponential":
                gamma_model = exponential_model(lags_fine, nugget, sill, range_param)
            elif model_name == "gaussian":
                gamma_model = gaussian_model(lags_fine, nugget, sill, range_param)

            ax.plot(
                lags_fine,
                gamma_model,
                "-",
                color="#D32F2F",
                linewidth=2.5,
                label=f"{model_name.capitalize()} model",
            )

        ax.set_title(
            "Variogram: Semivariance vs Lag Distance", fontsize=13, fontweight="bold", pad=15
        )
        ax.legend(frameon=False, fontsize=11)

        if use_clean_style:
            from earthsciences.utils.plot_style import clean_plot_style

            clean_plot_style(ax)

        output_path = output_dir / "variogram.png"
        plt.savefig(output_path, dpi=self.config.visualization.dpi, bbox_inches="tight")
        plt.close()

        logger.info("  ✓ Created variogram plot")
        self.plots.append(str(output_path))

    def _plot_kriging_map(self, output_dir, use_clean_style):
        """Plot kriging predictions."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 10))

        # Kriging predictions
        im = ax.contourf(
            self.kriging_result["X"],
            self.kriging_result["Y"],
            self.kriging_result["Z_pred"],
            levels=self.config.visualization.contour_levels or 20,
            cmap=self.config.visualization.colormap,
        )

        # Data points
        ax.scatter(
            self.data["x"],
            self.data["y"],
            c="white",
            s=20,
            edgecolors="black",
            linewidth=0.5,
            alpha=0.7,
            zorder=5,
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Predicted Value", fontsize=11)

        ax.set_title(
            f"Kriging Interpolation Map: {self.config.project.name}",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )

        if use_clean_style:
            from earthsciences.utils.plot_style import clean_plot_style

            clean_plot_style(ax)

        output_path = output_dir / "kriging_map.png"
        plt.savefig(output_path, dpi=self.config.visualization.dpi, bbox_inches="tight")
        plt.close()

        logger.info("  ✓ Created kriging map")
        self.plots.append(str(output_path))

    def _plot_variance_map(self, output_dir, use_clean_style):
        """Plot kriging variance."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 10))

        im = ax.contourf(
            self.kriging_result["X"],
            self.kriging_result["Y"],
            self.kriging_result["Z_var"],
            levels=20,
            cmap="YlOrRd",
        )

        ax.scatter(
            self.data["x"],
            self.data["y"],
            c="white",
            s=20,
            edgecolors="black",
            linewidth=0.5,
            alpha=0.7,
            zorder=5,
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Kriging Variance", fontsize=11)

        ax.set_title("Kriging Variance (Uncertainty) Map", fontsize=13, fontweight="bold", pad=15)

        if use_clean_style:
            from earthsciences.utils.plot_style import clean_plot_style

            clean_plot_style(ax)

        output_path = output_dir / "variance_map.png"
        plt.savefig(output_path, dpi=self.config.visualization.dpi, bbox_inches="tight")
        plt.close()

        logger.info("  ✓ Created variance map")
        self.plots.append(str(output_path))

    def _plot_cross_validation(self, output_dir, use_clean_style):
        """Plot cross-validation results."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))

        y_true = self.validation_result["y_true"]
        y_pred = self.validation_result["y_pred"]

        ax.scatter(
            y_true, y_pred, alpha=0.6, s=40, color="#1976D2", edgecolors="black", linewidth=0.5
        )

        # 1:1 line
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "--", color="#D32F2F", linewidth=2.5, alpha=0.7, label="1:1 line")

        # Add metrics text
        r2 = self.validation_result["r2"]
        rmse = self.validation_result["rmse"]
        ax.text(
            0.05,
            0.95,
            f"R² = {r2:.3f}\nRMSE = {rmse:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_title(
            "Cross-Validation: Predicted vs Actual Values", fontsize=13, fontweight="bold", pad=15
        )
        ax.legend(frameon=False, fontsize=11)

        if use_clean_style:
            from earthsciences.utils.plot_style import clean_plot_style

            clean_plot_style(ax)

        output_path = output_dir / "cross_validation.png"
        plt.savefig(output_path, dpi=self.config.visualization.dpi, bbox_inches="tight")
        plt.close()

        logger.info("  ✓ Created cross-validation plot")
        self.plots.append(str(output_path))

    def _plot_histogram(self, output_dir, use_clean_style):
        """Plot data histogram."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(
            self.data["z"], bins=30, alpha=0.8, edgecolor="black", linewidth=0.8, color="#64B5F6"
        )

        mean_val = np.mean(self.data["z"])
        median_val = np.median(self.data["z"])

        ax.axvline(
            mean_val, color="#D32F2F", linestyle="--", linewidth=2.5, label=f"Mean: {mean_val:.2f}"
        )
        ax.axvline(
            median_val,
            color="#388E3C",
            linestyle="--",
            linewidth=2.5,
            label=f"Median: {median_val:.2f}",
        )

        ax.set_title(
            f"Data Distribution: {self.config.data.z_column}",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.legend(frameon=False, fontsize=11)

        if use_clean_style:
            from earthsciences.utils.plot_style import clean_plot_style

            clean_plot_style(ax)

        output_path = output_dir / "histogram.png"
        plt.savefig(output_path, dpi=self.config.visualization.dpi, bbox_inches="tight")
        plt.close()

        logger.info("  ✓ Created histogram")
        self.plots.append(str(output_path))

    def save_results(self):
        """Save all results to files."""
        log_section("STEP 7: SAVING RESULTS")

        output_dir = Path(self.config.project.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        if self.config.output.save_predictions and self.kriging_result:
            if "csv" in self.config.output.formats:
                # Flatten grid and save as CSV
                df = pd.DataFrame(
                    {
                        "x": self.kriging_result["X"].ravel(),
                        "y": self.kriging_result["Y"].ravel(),
                        "prediction": self.kriging_result["Z_pred"].ravel(),
                        "variance": self.kriging_result["Z_var"].ravel(),
                    }
                )
                csv_path = output_dir / "predictions.csv"
                df.to_csv(csv_path, index=False)
                logger.info("  ✓ Saved predictions to CSV")

            if "npy" in self.config.output.formats:
                np.save(output_dir / "predictions.npy", self.kriging_result["Z_pred"])
                np.save(output_dir / "variance.npy", self.kriging_result["Z_var"])
                logger.info("  ✓ Saved predictions to NPY")

        # Save report
        if self.config.output.save_report:
            self._generate_report(output_dir / "report.txt")
            logger.info("  ✓ Saved analysis report")

        logger.info(f"\n✓ All results saved to: {output_dir}")

        self.results["output_dir"] = str(output_dir)
        self.results["plots"] = self.plots

    def _generate_report(self, output_path):
        """Generate text report."""
        with open(output_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("GEOSTATISTICAL ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Project: {self.config.project.name}\n")
            if self.config.project.description:
                f.write(f"Description: {self.config.project.description}\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")

            f.write("=" * 70 + "\n")
            f.write("DATA SUMMARY\n")
            f.write("=" * 70 + "\n")
            stats = self.results.get("data_stats", {})
            f.write(f"Number of points: {stats.get('n_points', 'N/A')}\n")
            f.write(f"X range: {stats.get('x_range', 'N/A')}\n")
            f.write(f"Y range: {stats.get('y_range', 'N/A')}\n")
            f.write(f"Z range: {stats.get('z_range', 'N/A')}\n")
            f.write(f"Mean: {stats.get('z_mean', 'N/A'):.4f}\n")
            f.write(f"Std Dev: {stats.get('z_std', 'N/A'):.4f}\n\n")

            if "variogram" in self.results:
                f.write("=" * 70 + "\n")
                f.write("VARIOGRAM MODEL\n")
                f.write("=" * 70 + "\n")
                vario = self.results["variogram"]
                f.write(f"Model: {vario.get('model', 'N/A')}\n")
                if vario.get("parameters"):
                    params = vario["parameters"]
                    f.write(f"Nugget: {params.get('nugget', 0):.4f}\n")
                    f.write(f"Sill: {params.get('sill', 0):.4f}\n")
                    f.write(f"Range: {params.get('range', 0):.2f}\n\n")

            if "validation" in self.results:
                f.write("=" * 70 + "\n")
                f.write("CROSS-VALIDATION RESULTS\n")
                f.write("=" * 70 + "\n")
                val = self.results["validation"]
                f.write(f"Folds: {val.get('n_folds', 'N/A')}\n")
                f.write(f"RMSE: {val.get('rmse', 'N/A'):.4f}\n")
                f.write(f"MAE: {val.get('mae', 'N/A'):.4f}\n")
                f.write(f"R²: {val.get('r2', 'N/A'):.4f}\n\n")

            if "kriging" in self.results:
                f.write("=" * 70 + "\n")
                f.write("KRIGING RESULTS\n")
                f.write("=" * 70 + "\n")
                krig = self.results["kriging"]
                f.write(f"Grid shape: {krig.get('grid_shape', 'N/A')}\n")
                f.write(f"Total predictions: {krig.get('n_predictions', 'N/A'):,}\n")
                f.write(f"Prediction range: {krig.get('pred_range', 'N/A')}\n")
                f.write(f"Mean prediction: {krig.get('mean_pred', 'N/A'):.4f}\n")
                f.write(f"Mean variance: {krig.get('mean_variance', 'N/A'):.4f}\n\n")

    def run_full_pipeline(self):
        """Execute complete pipeline."""
        try:
            self.load_data()
            self.preprocess()
            self.fit_variogram()
            self.krige()
            self.validate()
            self.visualize()
            self.save_results()

            logger.info(f"\n{'='*70}")
            logger.info("✓ ANALYSIS COMPLETE!")
            logger.info(f"{'='*70}\n")

            return self.results

        except Exception as e:
            logger.error("Error during analysis: %s", e)
            raise
