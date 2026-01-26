#!/usr/bin/env python3
"""
Example 8: Image Analysis - Remote Sensing and Microscopy

Demonstrates:
- Vegetation indices (NDVI, EVI, SAVI)
- Image enhancement techniques
- Grain size analysis from microscopy
- Shape and fabric analysis
- Watershed segmentation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from earthsciences.imaging import (
    ndvi,
    evi,
    savi,
    histogram_equalization,
    contrast_stretch,
    gamma_correction,
    unsharp_mask,
    grain_size_distribution,
    shape_analysis,
    fabric_analysis
)
from plot_utils import clean_plot_style

print("=" * 70)
print("IMAGE ANALYSIS EXAMPLES")
print("=" * 70)

# =============================================================================
# Example 1: Vegetation Indices from Satellite Imagery
# =============================================================================
print("\n" + "=" * 70)
print("Example 1: Vegetation Indices")
print("=" * 70)

# Simulate satellite image bands (NIR and Red)
# Create a landscape with different vegetation types
np.random.seed(42)
img_size = 256

# Create regions with different vegetation
y, x = np.ogrid[:img_size, :img_size]

# Dense forest (high NIR, low Red)
forest_mask = ((x - 64)**2 + (y - 64)**2) < 50**2
nir_forest = 0.7 + 0.1 * np.random.random((img_size, img_size))
red_forest = 0.1 + 0.05 * np.random.random((img_size, img_size))

# Grassland (moderate NIR and Red)
grass_mask = ((x - 192)**2 + (y - 64)**2) < 50**2
nir_grass = 0.5 + 0.1 * np.random.random((img_size, img_size))
red_grass = 0.2 + 0.05 * np.random.random((img_size, img_size))

# Agricultural field (variable)
crop_mask = ((x - 128)**2 + (y - 192)**2) < 50**2
nir_crop = 0.6 + 0.15 * np.random.random((img_size, img_size))
red_crop = 0.15 + 0.08 * np.random.random((img_size, img_size))

# Bare soil (low NIR, moderate Red)
soil_mask = ((x - 64)**2 + (y - 192)**2) < 40**2
nir_soil = 0.3 + 0.05 * np.random.random((img_size, img_size))
red_soil = 0.25 + 0.05 * np.random.random((img_size, img_size))

# Water (low both)
water_mask = ((x - 192)**2 + (y - 192)**2) < 30**2
nir_water = 0.05 + 0.02 * np.random.random((img_size, img_size))
red_water = 0.03 + 0.02 * np.random.random((img_size, img_size))

# Combine all regions
nir_band = np.ones((img_size, img_size)) * 0.2
red_band = np.ones((img_size, img_size)) * 0.15

nir_band[forest_mask] = nir_forest[forest_mask]
red_band[forest_mask] = red_forest[forest_mask]

nir_band[grass_mask] = nir_grass[grass_mask]
red_band[grass_mask] = red_grass[grass_mask]

nir_band[crop_mask] = nir_crop[crop_mask]
red_band[crop_mask] = red_crop[crop_mask]

nir_band[soil_mask] = nir_soil[soil_mask]
red_band[soil_mask] = red_soil[soil_mask]

nir_band[water_mask] = nir_water[water_mask]
red_band[water_mask] = red_water[water_mask]

# Calculate vegetation indices
ndvi_img = ndvi(nir_band, red_band)
evi_img = evi(nir_band, red_band, nir_band * 0.5)  # Blue band approximation
savi_img = savi(nir_band, red_band, L=0.5)

print(f"\nVegetation index statistics:")
print(f"  NDVI range: {ndvi_img.min():.3f} to {ndvi_img.max():.3f}")
print(f"  EVI range: {evi_img.min():.3f} to {evi_img.max():.3f}")
print(f"  SAVI range: {savi_img.min():.3f} to {savi_img.max():.3f}")

print(f"\nTypical NDVI values:")
print(f"  Water: {ndvi_img[water_mask].mean():.3f}")
print(f"  Bare soil: {ndvi_img[soil_mask].mean():.3f}")
print(f"  Grassland: {ndvi_img[grass_mask].mean():.3f}")
print(f"  Forest: {ndvi_img[forest_mask].mean():.3f}")

# Plot results
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Input bands
im1 = axes[0, 0].imshow(red_band, cmap='Reds', vmin=0, vmax=1)
axes[0, 0].set_title('Red Band', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')
plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

im2 = axes[0, 1].imshow(nir_band, cmap='RdPu', vmin=0, vmax=1)
axes[0, 1].set_title('NIR Band', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')
plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

# False color composite
false_color = np.dstack([nir_band, red_band, red_band*0.5])
axes[0, 2].imshow(false_color)
axes[0, 2].set_title('False Color Composite', fontsize=12, fontweight='bold')
axes[0, 2].axis('off')

# Vegetation indices
im3 = axes[1, 0].imshow(ndvi_img, cmap='RdYlGn', vmin=-0.5, vmax=1)
axes[1, 0].set_title('NDVI (Normalized Difference VI)', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')
plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

im4 = axes[1, 1].imshow(evi_img, cmap='RdYlGn', vmin=-0.5, vmax=2)
axes[1, 1].set_title('EVI (Enhanced VI)', fontsize=12, fontweight='bold')
axes[1, 1].axis('off')
plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

im5 = axes[1, 2].imshow(savi_img, cmap='RdYlGn', vmin=-0.5, vmax=1.5)
axes[1, 2].set_title('SAVI (Soil-Adjusted VI)', fontsize=12, fontweight='bold')
axes[1, 2].axis('off')
plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

plt.suptitle('Vegetation Indices from Satellite Imagery', 
            fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('08_image_vegetation_indices.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved vegetation indices example")
plt.close()

# =============================================================================
# Example 2: Image Enhancement
# =============================================================================
print("\n" + "=" * 70)
print("Example 2: Image Enhancement Techniques")
print("=" * 70)

# Create a low-contrast, blurry image
original_img = np.zeros((256, 256))
y, x = np.ogrid[:256, :256]

# Add several features
for cx, cy, radius, intensity in [(64, 64, 30, 0.8), (192, 64, 25, 0.6),
                                   (128, 192, 35, 0.9), (64, 192, 20, 0.5)]:
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    original_img += intensity * np.exp(-(dist**2) / (2 * radius**2))

# Add noise
original_img += np.random.normal(0, 0.05, original_img.shape)

# Blur the image
original_img = ndimage.gaussian_filter(original_img, sigma=2)

# Reduce contrast
original_img = 0.3 + 0.4 * original_img

print(f"\nOriginal image statistics:")
print(f"  Min: {original_img.min():.3f}")
print(f"  Max: {original_img.max():.3f}")
print(f"  Mean: {original_img.mean():.3f}")
print(f"  Std: {original_img.std():.3f}")

# Apply enhancements
hist_eq = histogram_equalization(original_img)
contrast_st = contrast_stretch(original_img, lower_percentile=2, upper_percentile=98)
gamma_corr = gamma_correction(original_img, gamma=0.7)
sharp = unsharp_mask(original_img, radius=2, amount=1.5)

print(f"\nEnhancement techniques applied:")
print(f"  1. Histogram equalization - Enhances contrast globally")
print(f"  2. Contrast stretch (2-98%) - Linear stretch of histogram")
print(f"  3. Gamma correction (γ=0.7) - Brightens midtones")
print(f"  4. Unsharp mask - Enhances edges and details")

# Plot results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original
im0 = axes[0, 0].imshow(original_img, cmap='gray')
axes[0, 0].set_title('Original (Low Contrast)', fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

# Histogram
axes[0, 1].hist(original_img.ravel(), bins=50, color='#757575', alpha=0.8, edgecolor='none')
axes[0, 1].set_title('Intensity Distribution (Frequency vs Intensity)', 
                    fontsize=11, fontweight='bold')
clean_plot_style(axes[0, 1])

# Contrast stretch
axes[0, 2].imshow(contrast_st, cmap='gray')
axes[0, 2].set_title('Contrast Stretch', fontsize=11, fontweight='bold')
axes[0, 2].axis('off')

# Histogram equalization
axes[1, 0].imshow(hist_eq, cmap='gray')
axes[1, 0].set_title('Histogram Equalization', fontsize=11, fontweight='bold')
axes[1, 0].axis('off')

# Gamma correction
axes[1, 1].imshow(gamma_corr, cmap='gray')
axes[1, 1].set_title('Gamma Correction (γ=0.7)', fontsize=11, fontweight='bold')
axes[1, 1].axis('off')

# Unsharp mask
axes[1, 2].imshow(sharp, cmap='gray')
axes[1, 2].set_title('Unsharp Mask (Sharpened)', fontsize=11, fontweight='bold')
axes[1, 2].axis('off')

plt.suptitle('Image Enhancement Techniques', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('08_image_enhancement.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved image enhancement example")
plt.close()

# =============================================================================
# Example 3: Grain Size Analysis from Thin Section
# =============================================================================
print("\n" + "=" * 70)
print("Example 3: Grain Size Analysis")
print("=" * 70)

# Create synthetic thin section image with grains
grain_img = np.zeros((512, 512), dtype=np.uint8)

# Generate grains with various sizes
np.random.seed(42)
n_grains = 80

for i in range(n_grains):
    # Random position
    cx = np.random.randint(50, 462)
    cy = np.random.randint(50, 462)
    
    # Random size (log-normal distribution typical of sediments)
    size = int(np.random.lognormal(mean=2.5, sigma=0.6))
    size = np.clip(size, 10, 60)
    
    # Random orientation for elliptical grains
    angle = np.random.uniform(0, np.pi)
    aspect = np.random.uniform(0.6, 1.0)
    
    # Create elliptical grain
    y, x = np.ogrid[:512, :512]
    x_rot = (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)
    y_rot = -(x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
    
    grain_mask = (x_rot**2 / (size * aspect)**2 + y_rot**2 / size**2) < 1
    grain_img[grain_mask] = 255

# Analyze grain sizes
result = grain_size_distribution(grain_img, threshold_method='otsu', 
                                min_size=50, return_labels=True)

print(f"\nGrain size analysis:")
print(f"  Number of grains: {result['n_grains']}")
print(f"  Mean diameter: {result['mean_diameter']:.1f} pixels")
print(f"  Median diameter: {result['median_diameter']:.1f} pixels")
print(f"  Std deviation: {result['std_diameter']:.1f} pixels")
print(f"  25th percentile: {result['percentiles']['25th']:.1f} pixels")
print(f"  50th percentile: {result['percentiles']['50th']:.1f} pixels")
print(f"  75th percentile: {result['percentiles']['75th']:.1f} pixels")

# Shape analysis
shape_result = shape_analysis(grain_img, min_size=50)

print(f"\nShape analysis:")
print(f"  Mean circularity: {shape_result['mean_circularity']:.3f}")
print(f"  Mean elongation: {shape_result['mean_elongation']:.3f}")
if 'solidity' in shape_result:
    print(f"  Mean solidity: {np.mean(shape_result['solidity']):.3f}")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

# Original image with labels
labeled = result['labeled_image']
axes[0, 0].imshow(grain_img, cmap='gray')
axes[0, 0].set_title('Original Thin Section Image', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Labeled grains
axes[0, 1].imshow(labeled, cmap='nipy_spectral')
axes[0, 1].set_title(f'Labeled Grains (n={result["n_grains"]})', 
                    fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

# Grain size distribution
axes[1, 0].hist(result['diameters'], bins=30, alpha=0.8, 
               edgecolor='black', linewidth=0.8, color='#64B5F6')
axes[1, 0].axvline(result['mean_diameter'], color='#D32F2F', linestyle='--', 
                  linewidth=2.5, label=f'Mean: {result["mean_diameter"]:.1f}')
axes[1, 0].axvline(result['median_diameter'], color='#388E3C', linestyle='--',
                  linewidth=2.5, label=f'Median: {result["median_diameter"]:.1f}')
axes[1, 0].set_title('Grain Size Distribution: Frequency vs Diameter (pixels)', 
                    fontsize=12, fontweight='bold')
axes[1, 0].legend(frameon=False, fontsize=10)
clean_plot_style(axes[1, 0])

# Shape metrics
metrics = ['circularity', 'elongation', 'solidity']
data = [shape_result['circularity'], 
        shape_result['elongation'],
        shape_result['solidity']]

bp = axes[1, 1].boxplot(data, labels=metrics, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#90CAF9')
    patch.set_edgecolor('black')
    patch.set_linewidth(1)
for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='black', linewidth=1)
axes[1, 1].set_title('Shape Metrics for All Grains', fontsize=12, fontweight='bold')
clean_plot_style(axes[1, 1])

plt.suptitle('Automated Grain Analysis from Thin Section', 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('08_image_grain_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved grain analysis example")
plt.close()

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
Generated plots:
  1. 08_image_vegetation_indices.png - Remote sensing indices
  2. 08_image_enhancement.png - Enhancement techniques
  3. 08_image_grain_analysis.png - Grain size and shape analysis

Key concepts demonstrated:
  • Vegetation indices (NDVI, EVI, SAVI) for multispectral data
  • Image enhancement (histogram eq, contrast, sharpening)
  • Automated grain detection and measurement
  • Shape analysis (circularity, elongation, solidity)
  • Statistical characterization of grain populations

Applications:
  - Remote sensing (vegetation mapping, land cover)
  - Sedimentology (grain size analysis, sorting)
  - Petrography (mineral identification, texture)
  - Soil science (particle size distribution)
  - Paleontology (microfossil analysis)
  - Environmental monitoring (change detection)
""")

print("✓ All image analysis examples complete!")
