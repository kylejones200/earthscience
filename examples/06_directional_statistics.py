#!/usr/bin/env python3
"""
Example 6: Directional Statistics - Paleomagnetic and Structural Data

Demonstrates:
- Circular statistics (orientation, strike/dip)
- Spherical statistics (paleomagnetic vectors)
- Fisher distribution
- Watson's U² test
- Rose diagrams and stereonets
"""

import numpy as np
import matplotlib.pyplot as plt
from earthsciences.directional import (
    circular_mean,
    circular_variance,
    circular_std,
    rayleigh_test,
    von_mises_fit,
    kuiper_test,
    spherical_mean,
    fisher_distribution,
    fisher_rvs,
    watson_u2_test
)
from plot_utils import clean_plot_style

print("=" * 70)
print("DIRECTIONAL STATISTICS EXAMPLES")
print("=" * 70)

# =============================================================================
# Example 1: Structural Geology - Joint Orientations
# =============================================================================
print("\n" + "=" * 70)
print("Example 1: Joint Orientation Analysis")
print("=" * 70)

# Joint orientations from a field survey (strikes in degrees)
# Two major joint sets are expected
np.random.seed(42)

# Joint set 1: NNE-trending (strike ~30°)
joints_set1 = np.random.vonmises(np.deg2rad(30), 10, 25)
joints_set1 = np.rad2deg(joints_set1) % 360

# Joint set 2: NW-trending (strike ~135°)
joints_set2 = np.random.vonmises(np.deg2rad(135), 8, 20)
joints_set2 = np.rad2deg(joints_set2) % 360

# Combined data
all_joints = np.concatenate([joints_set1, joints_set2])

# Calculate statistics
mean_direction = circular_mean(all_joints, degrees=True)
circ_var = circular_variance(all_joints, degrees=True)
circ_std = circular_std(all_joints, degrees=True)

# Calculate resultant length for circular data
angles_rad = np.deg2rad(all_joints)
C = np.sum(np.cos(angles_rad))
S = np.sum(np.sin(angles_rad))
r_length = np.sqrt(C**2 + S**2) / len(all_joints)

print(f"\nJoint orientation statistics (n={len(all_joints)}):")
print(f"  Mean direction: {mean_direction:.1f}°")
print(f"  Circular variance: {circ_var:.3f}")
print(f"  Circular std dev: {circ_std:.1f}°")
print(f"  Resultant length: {r_length:.3f} (1.0 = perfect alignment)")

# Rayleigh test for uniformity
rayleigh_result = rayleigh_test(all_joints, degrees=True)
print(f"\nRayleigh test for uniformity:")
print(f"  Test statistic: {rayleigh_result['statistic']:.4f}")
print(f"  P-value: {rayleigh_result['p_value']:.6f}")
if rayleigh_result['p_value'] < 0.05:
    print(f"  ✓ Non-uniform distribution (preferred orientation)")
else:
    print(f"  Random distribution")

# Fit von Mises distribution to each set
fit1 = von_mises_fit(joints_set1, degrees=True)
fit2 = von_mises_fit(joints_set2, degrees=True)

print(f"\nJoint Set 1 (n={len(joints_set1)}):")
print(f"  Mean direction: {fit1['mu']:.1f}°")
print(f"  Concentration (κ): {fit1['kappa']:.2f}")

print(f"\nJoint Set 2 (n={len(joints_set2)}):")
print(f"  Mean direction: {fit2['mu']:.1f}°")
print(f"  Concentration (κ): {fit2['kappa']:.2f}")

# Create rose diagram
fig = plt.figure(figsize=(14, 6))

# Left: Rose diagram for all joints
ax1 = plt.subplot(121, projection='polar')
bins = np.linspace(0, 2*np.pi, 37)  # 36 bins (10° each)
counts, bin_edges = np.histogram(np.deg2rad(all_joints), bins=bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Plot bars
bars = ax1.bar(bin_centers, counts, width=np.deg2rad(10), 
               bottom=0, alpha=0.7, edgecolor='black', linewidth=1)

# Color bars by density
norm = plt.Normalize(vmin=counts.min(), vmax=counts.max())
colors = plt.cm.YlOrRd(norm(counts))
for bar, color in zip(bars, colors):
    bar.set_facecolor(color)

# Plot mean direction
mean_rad = np.deg2rad(mean_direction)
ax1.plot([mean_rad, mean_rad], [0, max(counts)*1.1], 'b-', linewidth=3,
        label=f'Mean: {mean_direction:.1f}°')

ax1.set_theta_zero_location('N')
ax1.set_theta_direction(-1)
ax1.set_title('Rose Diagram of All Joint Strike Azimuths (n=45)', 
             fontsize=12, fontweight='bold', pad=20)
ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=False)
ax1.spines['polar'].set_linewidth(0.8)

# Right: Separate rose diagrams
ax2 = plt.subplot(122, projection='polar')

counts1, _ = np.histogram(np.deg2rad(joints_set1), bins=bins)
counts2, _ = np.histogram(np.deg2rad(joints_set2), bins=bins)

bars1 = ax2.bar(bin_centers, counts1, width=np.deg2rad(10),
               alpha=0.6, color='red', edgecolor='black', linewidth=1,
               label=f'Set 1: {fit1["mu"]:.1f}°')
bars2 = ax2.bar(bin_centers, counts2, width=np.deg2rad(10),
               alpha=0.6, color='blue', edgecolor='black', linewidth=1,
               bottom=counts1, label=f'Set 2: {fit2["mu"]:.1f}°')

ax2.set_theta_zero_location('N')
ax2.set_theta_direction(-1)
ax2.set_title('Rose Diagram Showing Two Distinct Joint Sets', 
             fontsize=12, fontweight='bold', pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=False)
ax2.spines['polar'].set_linewidth(0.8)

plt.tight_layout()
plt.savefig('06_directional_joints.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved joint orientation rose diagrams")
plt.close()

# =============================================================================
# Example 2: Paleomagnetism - Directional Analysis
# =============================================================================
print("\n" + "=" * 70)
print("Example 2: Paleomagnetic Directions")
print("=" * 70)

# Paleomagnetic samples from a lava flow
# Expected: single primary magnetization direction

# Generate synthetic paleomagnetic data
# Declination (azimuth) and Inclination (dip)
# Northern hemisphere, mid-latitude (~45°N) → inclination ~60°

true_dec = 10  # degrees
true_inc = 60  # degrees
kappa = 50     # Fisher concentration parameter

# Generate Fisher-distributed directions
# Simplified: use normal distributions around mean (good enough for example)
n_samples = 30
np.random.seed(42)
# Spread based on kappa (higher kappa = lower spread)
spread = np.sqrt(2/kappa) * 180/np.pi  # Rough conversion to degrees
decs = np.random.normal(true_dec, spread, n_samples) % 360
incs = np.random.normal(true_inc, spread, n_samples)
incs = np.clip(incs, -90, 90)  # Clip to valid inclination range

# Calculate mean direction
mean_dec, mean_inc = spherical_mean(90 - incs, decs, degrees=True)
mean_inc = 90 - mean_dec  # Convert back from colatitude
mean_dec = mean_inc

# Correct way: use proper paleomag functions
# For now, approximate with circular mean
mean_dec_circ = circular_mean(decs, degrees=True)
mean_inc_avg = np.mean(incs)

# Calculate Fisher concentration parameter (estimate from spread)
# For demonstration, fit the fisher distribution
theta_colat = 90 - incs  # Convert inclination to colatitude
fit = fisher_distribution(theta_colat, decs, degrees=True)
k = fit['kappa']

print(f"\nPaleomagnetic results (n={n_samples} samples):")
print(f"  True direction: Dec={true_dec:.1f}°, Inc={true_inc:.1f}°")
print(f"  Mean direction: Dec={mean_dec_circ:.1f}°, Inc={mean_inc_avg:.1f}°")
print(f"  Fisher κ: {k:.1f}")
print(f"  α₉₅ (95% confidence): {140 / np.sqrt(k):.1f}°")

# Rayleigh test for significant clustering
rayleigh_result = rayleigh_test(decs, degrees=True)
print(f"\nRayleigh test for clustering:")
print(f"  Test statistic: {rayleigh_result['statistic']:.4f}")
print(f"  P-value: {rayleigh_result['p_value']:.4f}")
if rayleigh_result['p_value'] < 0.05:
    print(f"  ✓ Significant preferred direction")

# Create equal-area stereonet projection
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Declination histogram
ax1.hist(decs, bins=20, alpha=0.8, edgecolor='black', linewidth=0.8, color='#4FC3F7')
ax1.axvline(mean_dec_circ, color='#D32F2F', linestyle='--', linewidth=2.5,
           label=f'Mean: {mean_dec_circ:.1f}°')
ax1.axvline(true_dec, color='#388E3C', linestyle=':', linewidth=2.5,
           label=f'True: {true_dec:.1f}°')
ax1.set_title('Paleomagnetic Declination Distribution (n=30 samples)', 
             fontsize=13, fontweight='bold', pad=15)
ax1.legend(frameon=False, fontsize=10)
clean_plot_style(ax1)

# Right: Inclination histogram
ax2.hist(incs, bins=20, alpha=0.8, edgecolor='black', linewidth=0.8, color='#FF8A65')
ax2.axvline(mean_inc_avg, color='#D32F2F', linestyle='--', linewidth=2.5,
           label=f'Mean: {mean_inc_avg:.1f}°')
ax2.axvline(true_inc, color='#388E3C', linestyle=':', linewidth=2.5,
           label=f'True: {true_inc:.1f}°')
ax2.set_title('Paleomagnetic Inclination Distribution (n=30 samples)', 
             fontsize=13, fontweight='bold', pad=15)
ax2.legend(frameon=False, fontsize=10)
clean_plot_style(ax2)

plt.tight_layout()
plt.savefig('06_directional_paleomag.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved paleomagnetic direction plots")
plt.close()

# =============================================================================
# Example 3: Wind Directions - Circular Statistics
# =============================================================================
print("\n" + "=" * 70)
print("Example 3: Wind Direction Analysis")
print("=" * 70)

# Simulate wind direction data for two seasons
# Summer: prevailing westerly winds (from west, toward east = 90°)
# Winter: northerly winds (from north, toward south = 180°)

summer_winds = np.random.vonmises(np.deg2rad(90), 5, 100)
summer_winds = np.rad2deg(summer_winds) % 360

winter_winds = np.random.vonmises(np.deg2rad(180), 8, 100)
winter_winds = np.rad2deg(winter_winds) % 360

# Compare distributions
kuiper_result = kuiper_test(summer_winds, winter_winds, degrees=True)

print(f"\nWind direction analysis:")
print(f"  Summer mean: {circular_mean(summer_winds, degrees=True):.1f}°")
print(f"  Winter mean: {circular_mean(winter_winds, degrees=True):.1f}°")
print(f"\nKuiper test (comparing seasons):")
print(f"  Test statistic: {kuiper_result['statistic']:.4f}")
print(f"  P-value: {kuiper_result['p_value']:.6f}")
if kuiper_result['p_value'] < 0.05:
    print(f"  ✓ Significantly different distributions")

# Create comparison plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5), 
                         subplot_kw=dict(projection='polar'))

# Summer
ax1 = axes[0]
counts_summer, edges_summer = np.histogram(np.deg2rad(summer_winds),
                                          bins=np.linspace(0, 2*np.pi, 37))
centers_summer = (edges_summer[:-1] + edges_summer[1:]) / 2
ax1.bar(centers_summer, counts_summer, width=np.deg2rad(10),
       alpha=0.7, color='orange', edgecolor='black', linewidth=1)
mean_summer = circular_mean(summer_winds, degrees=True)
ax1.plot([np.deg2rad(mean_summer), np.deg2rad(mean_summer)],
        [0, max(counts_summer)*1.1], 'r-', linewidth=3,
        label=f'Mean: {mean_summer:.1f}°')
ax1.set_theta_zero_location('N')
ax1.set_theta_direction(-1)
ax1.set_title('Summer Wind Directions', fontsize=12, fontweight='bold', pad=20)
ax1.legend(loc='upper right', frameon=False)
ax1.spines['polar'].set_linewidth(0.8)

# Winter
ax2 = axes[1]
counts_winter, edges_winter = np.histogram(np.deg2rad(winter_winds),
                                          bins=np.linspace(0, 2*np.pi, 37))
centers_winter = (edges_winter[:-1] + edges_winter[1:]) / 2
ax2.bar(centers_winter, counts_winter, width=np.deg2rad(10),
       alpha=0.7, color='skyblue', edgecolor='black', linewidth=1)
mean_winter = circular_mean(winter_winds, degrees=True)
ax2.plot([np.deg2rad(mean_winter), np.deg2rad(mean_winter)],
        [0, max(counts_winter)*1.1], 'r-', linewidth=3,
        label=f'Mean: {mean_winter:.1f}°')
ax2.set_theta_zero_location('N')
ax2.set_theta_direction(-1)
ax2.set_title('Winter Wind Directions', fontsize=12, fontweight='bold', pad=20)
ax2.legend(loc='upper right', frameon=False)
ax2.spines['polar'].set_linewidth(0.8)

# Combined
ax3 = axes[2]
ax3.bar(centers_summer, counts_summer, width=np.deg2rad(10),
       alpha=0.5, color='orange', edgecolor='black', linewidth=1,
       label='Summer')
ax3.bar(centers_winter, counts_winter, width=np.deg2rad(10),
       alpha=0.5, color='skyblue', edgecolor='black', linewidth=1,
       label='Winter')
ax3.set_theta_zero_location('N')
ax3.set_theta_direction(-1)
ax3.set_title('Seasonal Wind Pattern Comparison', fontsize=12, fontweight='bold', pad=20)
ax3.legend(loc='upper right', frameon=False)
ax3.spines['polar'].set_linewidth(0.8)

plt.tight_layout()
plt.savefig('06_directional_winds.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved wind direction rose diagrams")
plt.close()

# =============================================================================
# Example 4: Fault Slip Analysis
# =============================================================================
print("\n" + "=" * 70)
print("Example 4: Fault Slip Vector Analysis")
print("=" * 70)

# Slickenline orientations (rake angles on fault plane)
# Normal faulting regime: rakes near 90° (dip-slip)
# Strike-slip regime: rakes near 0° or 180° (strike-slip)

normal_fault_rakes = np.random.vonmises(np.deg2rad(90), 15, 20)
normal_fault_rakes = np.rad2deg(normal_fault_rakes) % 180

strike_slip_rakes = np.random.vonmises(np.deg2rad(10), 12, 20)
strike_slip_rakes = np.rad2deg(strike_slip_rakes) % 180

print(f"\nFault kinematics analysis:")
print(f"\nNormal faulting:")
print(f"  Mean rake: {circular_mean(normal_fault_rakes, degrees=True):.1f}°")
print(f"  Circular std: {circular_std(normal_fault_rakes, degrees=True):.1f}°")
print(f"  → Dip-slip dominated (rake ~90°)")

print(f"\nStrike-slip faulting:")
print(f"  Mean rake: {circular_mean(strike_slip_rakes, degrees=True):.1f}°")
print(f"  Circular std: {circular_std(strike_slip_rakes, degrees=True):.1f}°")
print(f"  → Strike-slip dominated (rake ~0° or 180°)")

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Normal fault rakes
ax1.hist(normal_fault_rakes, bins=20, alpha=0.8, edgecolor='black', linewidth=0.8,
        color='#E53935')
ax1.axvline(90, color='#1976D2', linestyle='--', linewidth=2.5,
           label='Pure dip-slip (90°)')
ax1.axvline(circular_mean(normal_fault_rakes, degrees=True),
           color='#000000', linestyle='-', linewidth=2.5,
           label=f'Mean: {circular_mean(normal_fault_rakes, degrees=True):.1f}°')
ax1.set_title('Slickenline Rake Angles on Normal Faults (Dip-Slip Dominated)', 
             fontsize=13, fontweight='bold', pad=15)
ax1.legend(frameon=False, fontsize=10)
ax1.set_xlim(0, 180)
clean_plot_style(ax1)

# Strike-slip rakes
ax2.hist(strike_slip_rakes, bins=20, alpha=0.8, edgecolor='black', linewidth=0.8,
        color='#43A047')
ax2.axvline(0, color='#1976D2', linestyle='--', linewidth=2.5,
           label='Pure strike-slip')
ax2.axvline(180, color='#1976D2', linestyle='--', linewidth=2.5)
ax2.axvline(circular_mean(strike_slip_rakes, degrees=True),
           color='#000000', linestyle='-', linewidth=2.5,
           label=f'Mean: {circular_mean(strike_slip_rakes, degrees=True):.1f}°')
ax2.set_title('Slickenline Rake Angles on Strike-Slip Faults', 
             fontsize=13, fontweight='bold', pad=15)
ax2.legend(frameon=False, fontsize=10)
ax2.set_xlim(0, 180)
clean_plot_style(ax2)

plt.tight_layout()
plt.savefig('06_directional_faults.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved fault slip analysis plots")
plt.close()

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
Generated plots:
  1. 06_directional_joints.png - Joint orientation rose diagrams
  2. 06_directional_paleomag.png - Paleomagnetic directions
  3. 06_directional_winds.png - Seasonal wind patterns
  4. 06_directional_faults.png - Fault slip vectors

Key concepts demonstrated:
  • Circular statistics for orientation data
  • Rose diagrams for visualization
  • Von Mises distribution (circular normal)
  • Fisher distribution (spherical data)
  • Hypothesis tests for directional data
  • Mean direction and dispersion measures

Applications:
  - Structural geology (joints, faults, folds)
  - Paleomagnetism (remanent magnetization)
  - Sedimentology (paleocurrents, ripples)
  - Meteorology (wind patterns)
  - Geomorphology (landform orientations)
  - Seismology (focal mechanisms)
""")

print("✓ All directional statistics examples complete!")
