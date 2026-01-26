"""
Configuration schemas using Pydantic for validation.
"""

from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field, validator
from pathlib import Path


class ProjectConfig(BaseModel):
    """Project metadata and output settings."""
    name: str = Field(..., description="Project name")
    output_dir: str = Field(default="./results", description="Output directory path")
    description: Optional[str] = Field(None, description="Project description")
    
    @validator('output_dir')
    def validate_output_dir(cls, v):
        """Ensure output directory path is valid."""
        return str(Path(v).expanduser())


class DataConfig(BaseModel):
    """Data input and column specifications."""
    input_file: str = Field(..., description="Path to input data file")
    x_column: str = Field(..., description="Column name for X coordinates")
    y_column: str = Field(..., description="Column name for Y coordinates")
    z_column: str = Field(..., description="Column name for values")
    
    @validator('input_file')
    def validate_input_file(cls, v):
        """Ensure input file exists."""
        path = Path(v).expanduser()
        if not path.exists():
            raise ValueError(f"Input file not found: {v}")
        return str(path)


class PreprocessingConfig(BaseModel):
    """Data preprocessing options."""
    remove_outliers: bool = Field(default=False, description="Remove outliers")
    outlier_method: Literal['iqr', 'zscore', 'mad'] = Field(
        default='iqr', 
        description="Outlier detection method"
    )
    outlier_threshold: float = Field(default=3.0, description="Outlier threshold")
    transform: Optional[Literal['log', 'boxcox', 'normal_score']] = Field(
        None, 
        description="Data transformation"
    )
    handle_negatives: bool = Field(
        default=True, 
        description="Shift data to handle negative values for log transform"
    )


class VariogramConfig(BaseModel):
    """Variogram computation and fitting settings."""
    n_lags: int = Field(default=15, ge=5, le=50, description="Number of lags")
    max_lag: Optional[float] = Field(None, ge=0, description="Maximum lag distance")
    lag_tolerance: float = Field(default=0.5, ge=0, le=1, description="Lag tolerance")
    models: List[Literal['spherical', 'exponential', 'gaussian', 'linear', 'power']] = Field(
        default=['spherical', 'exponential'],
        description="Variogram models to try"
    )
    auto_fit: bool = Field(default=True, description="Automatic model fitting")
    fit_method: Literal['ols', 'wls', 'ml'] = Field(
        default='wls',
        description="Fitting method (OLS, WLS, ML)"
    )
    direction: Optional[float] = Field(
        None, 
        ge=0, 
        le=360, 
        description="Direction for anisotropy (degrees)"
    )
    tolerance: Optional[float] = Field(
        None, 
        ge=0, 
        le=90, 
        description="Angular tolerance for anisotropy"
    )


class NeighborhoodConfig(BaseModel):
    """Kriging neighborhood search parameters."""
    max_neighbors: int = Field(default=25, ge=3, le=100, description="Maximum neighbors")
    min_neighbors: int = Field(default=3, ge=1, description="Minimum neighbors")
    search_radius: Optional[float] = Field(None, ge=0, description="Search radius")
    use_quadrant_search: bool = Field(
        default=False, 
        description="Use quadrant-based search"
    )


class GridConfig(BaseModel):
    """Prediction grid specification."""
    x_min: float = Field(..., description="Minimum X coordinate")
    x_max: float = Field(..., description="Maximum X coordinate")
    y_min: float = Field(..., description="Minimum Y coordinate")
    y_max: float = Field(..., description="Maximum Y coordinate")
    resolution: float = Field(default=1.0, gt=0, description="Grid resolution")
    
    @validator('x_max')
    def validate_x_range(cls, v, values):
        if 'x_min' in values and v <= values['x_min']:
            raise ValueError("x_max must be greater than x_min")
        return v
    
    @validator('y_max')
    def validate_y_range(cls, v, values):
        if 'y_min' in values and v <= values['y_min']:
            raise ValueError("y_max must be greater than y_min")
        return v


class KrigingConfig(BaseModel):
    """Kriging method and parameters."""
    method: Literal['ordinary', 'simple', 'universal'] = Field(
        default='ordinary',
        description="Kriging method"
    )
    drift_terms: Optional[List[str]] = Field(
        None,
        description="Drift terms for universal kriging (e.g., ['x', 'y', 'x*y'])"
    )
    neighborhood: NeighborhoodConfig = Field(
        default_factory=NeighborhoodConfig,
        description="Neighborhood search parameters"
    )
    grid: GridConfig = Field(..., description="Prediction grid")
    compute_variance: bool = Field(default=True, description="Compute kriging variance")


class ValidationConfig(BaseModel):
    """Cross-validation settings."""
    cross_validation: bool = Field(default=True, description="Perform cross-validation")
    n_folds: int = Field(default=5, ge=2, le=20, description="Number of CV folds")
    method: Literal['kfold', 'loo', 'spatial'] = Field(
        default='kfold',
        description="Cross-validation method"
    )
    metrics: List[Literal['rmse', 'mae', 'r2', 'mse', 'mape']] = Field(
        default=['rmse', 'mae', 'r2'],
        description="Validation metrics"
    )
    stratified: bool = Field(
        default=False,
        description="Use stratified sampling"
    )


class VisualizationConfig(BaseModel):
    """Visualization and plotting settings."""
    style: Literal['minimalist', 'default', 'seaborn'] = Field(
        default='minimalist',
        description="Plot style"
    )
    plots: List[Literal[
        'variogram', 
        'kriging_map', 
        'variance_map',
        'cross_validation', 
        'histogram',
        'qq_plot',
        'scatter_actual_predicted'
    ]] = Field(
        default=['variogram', 'kriging_map', 'cross_validation'],
        description="Plots to generate"
    )
    colormap: str = Field(default='viridis', description="Colormap for maps")
    dpi: int = Field(default=300, ge=72, le=600, description="Output DPI")
    figsize: Optional[List[float]] = Field(
        None,
        description="Figure size [width, height] in inches"
    )
    contour_levels: Optional[int] = Field(
        None,
        ge=5,
        le=50,
        description="Number of contour levels"
    )


class OutputConfig(BaseModel):
    """Output file settings."""
    save_predictions: bool = Field(default=True, description="Save prediction grid")
    save_variance: bool = Field(default=True, description="Save variance grid")
    save_plots: bool = Field(default=True, description="Save plots")
    save_report: bool = Field(default=True, description="Save text report")
    formats: List[Literal['png', 'pdf', 'geotiff', 'csv', 'npy']] = Field(
        default=['png', 'csv'],
        description="Output formats"
    )
    compression: bool = Field(default=False, description="Compress output files")


class AnalysisConfig(BaseModel):
    """Complete analysis configuration."""
    project: ProjectConfig
    data: DataConfig
    preprocessing: Optional[PreprocessingConfig] = Field(
        default_factory=PreprocessingConfig
    )
    variogram: VariogramConfig = Field(default_factory=VariogramConfig)
    kriging: KrigingConfig
    validation: Optional[ValidationConfig] = Field(default_factory=ValidationConfig)
    visualization: Optional[VisualizationConfig] = Field(
        default_factory=VisualizationConfig
    )
    output: OutputConfig = Field(default_factory=OutputConfig)
    
    class Config:
        extra = 'forbid'  # Raise error on unknown fields
        validate_assignment = True
