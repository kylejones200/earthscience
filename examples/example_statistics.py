"""Statistical analysis examples using earthsciences library."""

import numpy as np
import matplotlib.pyplot as plt
import earthsciences as es
from plot_utils import clean_plot_style

np.random.seed(42)


def analyze_grain_sizes():
    grain_sizes = np.random.lognormal(mean=2, sigma=0.5, size=200)
    stats = es.statistics.descriptive_stats(grain_sizes)
    
    print(f"Grain size statistics (n={stats['count']}):")
    print(f"  Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}")
    print(f"  Std: {stats['std']:.2f}, Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    print(f"  Skewness: {stats['skewness']:.2f}, Kurtosis: {stats['kurtosis']:.2f}")
    
    return grain_sizes, stats


def fit_depth_temperature_regression():
    depth = np.linspace(0, 100, 50)
    temperature = 10 + 0.3 * depth + np.random.randn(50) * 2
    
    result = es.statistics.linear_regression(depth, temperature)
    
    print(f"\nDepth-Temperature regression:")
    print(f"  T = {result['intercept']:.2f} + {result['slope']:.3f} × depth")
    print(f"  R² = {result['r_squared']:.3f}, p = {result['p_value']:.6f}")
    
    return depth, temperature, result


def compare_two_samples():
    sample1 = np.random.randn(50) + 10
    sample2 = np.random.randn(50) + 10.5
    
    t_result = es.statistics.t_test(sample1, sample2)
    
    print(f"\nTwo-sample t-test:")
    print(f"  Sample 1: μ = {np.mean(sample1):.2f}")
    print(f"  Sample 2: μ = {np.mean(sample2):.2f}")
    print(f"  t = {t_result['statistic']:.3f}, p = {t_result['p_value']:.4f}")
    print(f"  Significant: {t_result['significant']}")
    
    return sample1, sample2


def bootstrap_confidence_interval(data):
    boot_result = es.statistics.bootstrap(
        data,
        statistic=np.mean,
        n_bootstrap=1000,
        confidence=0.95
    )
    
    ci_lower, ci_upper = boot_result['ci']
    print(f"\nBootstrap CI for mean:")
    print(f"  Estimate: {boot_result['estimate']:.2f} ± {boot_result['standard_error']:.2f}")
    print(f"  95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
    
    return boot_result


def test_distribution_fit(data):
    fit = es.statistics.fit_distribution(data, dist_name='lognorm')
    normality = es.statistics.test_normality(data, method='shapiro')
    
    print(f"\nDistribution fit (Lognormal):")
    print(f"  AIC: {fit['aic']:.2f}, BIC: {fit['bic']:.2f}")
    print(f"  Shapiro-Wilk: W = {normality[0]:.4f}, p = {normality[1]:.4f}")
    print(f"  Normal: {normality[1] >= 0.05}")
    
    return fit, normality


def create_visualizations(grain_sizes, stats, depth, temperature, regression, bootstrap):
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Grain size histogram
        axes[0, 0].hist(grain_sizes, bins=30, alpha=0.8, edgecolor='black', 
                       linewidth=0.8, color='#64B5F6')
        axes[0, 0].axvline(stats['mean'], color='#D32F2F', linestyle='--', 
                          linewidth=2.5, label='Mean')
        axes[0, 0].axvline(stats['median'], color='#388E3C', linestyle='--', 
                          linewidth=2.5, label='Median')
        axes[0, 0].set_title('Grain Size Distribution (n=200)', 
                            fontsize=12, fontweight='bold')
        axes[0, 0].legend(frameon=False)
        clean_plot_style(axes[0, 0])
        
        # Regression plot
        axes[0, 1].scatter(depth, temperature, alpha=0.7, s=50, color='#1976D2', 
                          edgecolors='black', linewidth=0.5)
        axes[0, 1].plot(depth, regression['predicted'], '#D32F2F', linewidth=3)
        axes[0, 1].set_title(f'Temperature vs Depth (R² = {regression["r_squared"]:.3f})', 
                            fontsize=12, fontweight='bold')
        clean_plot_style(axes[0, 1])
        
        # Bootstrap histogram
        axes[1, 0].hist(bootstrap['samples'], bins=50, alpha=0.8, 
                       edgecolor='black', linewidth=0.8, color='#66BB6A')
        axes[1, 0].axvline(bootstrap['estimate'], color='#000000', 
                          linestyle='--', linewidth=2.5, label='Estimate')
        ci_lower, ci_upper = bootstrap['ci']
        axes[1, 0].axvline(ci_lower, color='#D32F2F', linestyle=':', linewidth=2.5)
        axes[1, 0].axvline(ci_upper, color='#D32F2F', linestyle=':', linewidth=2.5, 
                          label='95% CI')
        axes[1, 0].set_title('Bootstrap Distribution of Mean (1000 resamples)', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].legend(frameon=False)
        clean_plot_style(axes[1, 0])
        
        # Q-Q plot
        theoretical, sample = es.statistics.qq_plot_data(grain_sizes, 'lognorm')
        axes[1, 1].scatter(theoretical, sample, alpha=0.6, s=40, color='#9C27B0', 
                          edgecolors='black', linewidth=0.5)
        axes[1, 1].plot([theoretical.min(), theoretical.max()],
                       [theoretical.min(), theoretical.max()], '#D32F2F', 
                       linestyle='--', linewidth=2.5)
        axes[1, 1].set_title('Q-Q Plot: Sample vs Theoretical Lognormal Quantiles', 
                            fontsize=12, fontweight='bold')
        clean_plot_style(axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('statistics_example.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved statistics_example.png")
        plt.close()
        
    except Exception as e:
        print(f"Visualization failed: {e}")


if __name__ == "__main__":
    grain_sizes, stats = analyze_grain_sizes()
    depth, temperature, regression = fit_depth_temperature_regression()
    compare_two_samples()
    bootstrap = bootstrap_confidence_interval(grain_sizes)
    test_distribution_fit(grain_sizes)
    create_visualizations(grain_sizes, stats, depth, temperature, regression, bootstrap)
