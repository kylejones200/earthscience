#!/usr/bin/env python3
"""
Example 5: Geochronology - Radiometric Dating

Demonstrates:
- Radioactive decay calculations
- U-Pb dating
- Rb-Sr isochron dating
- Radiocarbon calibration
- Concordia diagrams
"""

import numpy as np
import matplotlib.pyplot as plt
from earthsciences.geochronology import (
    radioactive_decay,
    calculate_age,
    isochron_dating,
    concordia_diagram_age,
    radiocarbon_age,
    DECAY_CONSTANTS
)
from plot_utils import clean_plot_style

# Helper functions for U-Pb dating
def decay_constant_to_halflife(lambda_val):
    """Convert decay constant to half-life"""
    return np.log(2) / lambda_val

def u_pb_age(ratio, system='206Pb_238U'):
    """Calculate age from Pb/U ratio"""
    if system == '206Pb_238U':
        lambda_val = DECAY_CONSTANTS['U238']
    elif system == '207Pb_235U':
        lambda_val = DECAY_CONSTANTS['U235']
    else:
        raise ValueError("System must be '206Pb_238U' or '207Pb_235U'")
    
    # t = (1/λ) * ln(1 + Pb/U)
    age = (1 / lambda_val) * np.log(1 + ratio)
    return age

def rb_sr_isochron_age(rb_sr_ratios, sr_ratios, uncertainties):
    """Calculate Rb-Sr isochron age"""
    return isochron_dating(
        rb_sr_ratios, sr_ratios, uncertainties,
        DECAY_CONSTANTS['Rb87']
    )

print("=" * 70)
print("GEOCHRONOLOGY EXAMPLES")
print("=" * 70)

# =============================================================================
# Example 1: Radioactive Decay Basics
# =============================================================================
print("\n" + "=" * 70)
print("Example 1: Radioactive Decay")
print("=" * 70)

# Calculate how much U-238 remains after 1 billion years
t = 1e9  # years
lambda_238 = DECAY_CONSTANTS['U238']
N0 = 1000  # Initial atoms

N_remaining = radioactive_decay(N0, lambda_238, t)
percent_remaining = (N_remaining / N0) * 100

print(f"\nU-238 decay over 1 billion years:")
print(f"  Initial atoms: {N0}")
print(f"  Time elapsed: {t/1e9:.1f} Ga")
print(f"  Remaining: {N_remaining:.1f} atoms ({percent_remaining:.1f}%)")
print(f"  Decayed: {N0 - N_remaining:.1f} atoms")
print(f"  Half-life: {np.log(2)/lambda_238/1e9:.2f} Ga")

# Plot decay curves for different isotopes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

time = np.linspace(0, 10e9, 1000)  # 0 to 10 Ga

# Common dating systems
systems = {
    'U-238': DECAY_CONSTANTS['U238'],
    'U-235': DECAY_CONSTANTS['U235'],
    'Rb-87': DECAY_CONSTANTS['Rb87'],
    'K-40': DECAY_CONSTANTS['K40']
}

for name, lambda_val in systems.items():
    remaining = radioactive_decay(100, lambda_val, time)
    ax1.plot(time / 1e9, remaining, label=name, linewidth=2.5)

ax1.set_title('Percent Remaining of Common Radioisotopes Over 10 Billion Years', 
              fontsize=13, fontweight='bold', pad=15)
ax1.legend(fontsize=10, frameon=False)
ax1.set_xlim(0, 10)
clean_plot_style(ax1)

# Log scale
for name, lambda_val in systems.items():
    remaining = radioactive_decay(100, lambda_val, time)
    ax2.semilogy(time / 1e9, remaining, label=name, linewidth=2.5)

ax2.set_title('Radioactive Decay on Logarithmic Scale (% Remaining vs Time in Ga)', 
              fontsize=13, fontweight='bold', pad=15)
ax2.legend(fontsize=10, frameon=False)
ax2.set_xlim(0, 10)
clean_plot_style(ax2)

plt.tight_layout()
plt.savefig('05_geochronology_decay.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved decay curves plot")
plt.close()

# =============================================================================
# Example 2: U-Pb Dating of Zircon
# =============================================================================
print("\n" + "=" * 70)
print("Example 2: U-Pb Dating")
print("=" * 70)

# Single grain zircon measurements
# Modern zircon from a volcanic ash bed
pb206_u238_ratio = 0.025  # 206Pb/238U
pb207_u235_ratio = 0.180  # 207Pb/235U

# Calculate ages from both systems
age_206_238 = u_pb_age(pb206_u238_ratio, system='206Pb_238U')
age_207_235 = u_pb_age(pb207_u235_ratio, system='207Pb_235U')

print(f"\nZircon U-Pb dating:")
print(f"  206Pb/238U = {pb206_u238_ratio:.4f} → Age = {age_206_238/1e6:.1f} Ma")
print(f"  207Pb/235U = {pb207_u235_ratio:.4f} → Age = {age_207_235/1e6:.1f} Ma")

# Check concordance
discordance = abs(age_206_238 - age_207_235) / age_206_238 * 100
print(f"  Discordance: {discordance:.1f}%")

if discordance < 5:
    print(f"  ✓ Concordant age: {age_206_238/1e6:.1f} Ma")
else:
    print(f"  ⚠ Discordant - possible Pb loss")

# =============================================================================
# Example 3: Concordia Diagram
# =============================================================================
print("\n" + "=" * 70)
print("Example 3: Concordia Diagram")
print("=" * 70)

# Simulate multiple zircon analyses
np.random.seed(42)
true_age = 500e6  # 500 Ma

# Generate concordant and discordant points
n_points = 10

# Concordant points (on concordia)
ages_concordant = np.random.normal(true_age, 10e6, 5)
pb206_u238_conc = np.exp(DECAY_CONSTANTS['U238'] * ages_concordant) - 1
pb207_u235_conc = np.exp(DECAY_CONSTANTS['U235'] * ages_concordant) - 1

# Discordant points (Pb loss)
ages_discordant = np.random.normal(true_age, 20e6, 5)
pb_loss = np.random.uniform(0.05, 0.25, 5)  # 5-25% Pb loss
pb206_u238_disc = (np.exp(DECAY_CONSTANTS['U238'] * ages_discordant) - 1) * (1 - pb_loss)
pb207_u235_disc = (np.exp(DECAY_CONSTANTS['U235'] * ages_discordant) - 1) * (1 - pb_loss)

# Plot concordia diagram
fig, ax = plt.subplots(figsize=(10, 10))

# Concordia curve
ages_concordia = np.linspace(0, 4000e6, 1000)  # 0 to 4000 Ma
pb206_u238_curve = np.exp(DECAY_CONSTANTS['U238'] * ages_concordia) - 1
pb207_u235_curve = np.exp(DECAY_CONSTANTS['U235'] * ages_concordia) - 1

ax.plot(pb207_u235_curve, pb206_u238_curve, 'k-', linewidth=2.5, 
        label='Concordia', zorder=1)

# Add age markers
for age_ma in [100, 250, 500, 1000, 2000, 3000, 4000]:
    age = age_ma * 1e6
    x = np.exp(DECAY_CONSTANTS['U235'] * age) - 1
    y = np.exp(DECAY_CONSTANTS['U238'] * age) - 1
    ax.plot(x, y, 'ko', markersize=6, zorder=2)
    ax.annotate(f'{age_ma}', (x, y), xytext=(5, 5), 
                textcoords='offset points', fontsize=9)

# Plot data
ax.scatter(pb207_u235_conc, pb206_u238_conc, s=120, c='#2E7D32', 
          marker='o', edgecolors='black', linewidth=1.5, 
          label='Concordant', zorder=3, alpha=0.9)
ax.scatter(pb207_u235_disc, pb206_u238_disc, s=120, c='#C62828', 
          marker='s', edgecolors='black', linewidth=1.5, 
          label='Discordant (Pb loss)', zorder=3, alpha=0.9)

ax.set_title('Zircon U-Pb Concordia Diagram: ²⁰⁶Pb/²³⁸U vs ²⁰⁷Pb/²³⁵U (Ages in Ma)', 
            fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper left', frameon=False)
clean_plot_style(ax)

plt.tight_layout()
plt.savefig('05_geochronology_concordia.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved concordia diagram")
plt.close()

print(f"\nConcordia analysis:")
print(f"  True age: {true_age/1e6:.0f} Ma")
print(f"  Concordant points: {len(pb206_u238_conc)} (mean age: {np.mean(ages_concordant)/1e6:.0f} Ma)")
print(f"  Discordant points: {len(pb206_u238_disc)} (apparent ages too young)")

# =============================================================================
# Example 4: Rb-Sr Isochron Dating
# =============================================================================
print("\n" + "=" * 70)
print("Example 4: Rb-Sr Isochron Dating")
print("=" * 70)

# Whole rock samples from a granite pluton
# True age: 1500 Ma, initial 87Sr/86Sr = 0.705

true_age_rbsr = 1500e6  # years
initial_ratio = 0.705
lambda_rb = DECAY_CONSTANTS['Rb87']

# Different minerals/rocks with different Rb/Sr ratios
rb87_sr86 = np.array([0.5, 1.0, 2.0, 4.0, 6.0])  # 87Rb/86Sr

# Calculate current 87Sr/86Sr ratios
sr87_sr86 = initial_ratio + rb87_sr86 * (np.exp(lambda_rb * true_age_rbsr) - 1)

# Add some analytical uncertainty
sr87_sr86 += np.random.normal(0, 0.0001, len(sr87_sr86))

# Calculate isochron age
result = rb_sr_isochron_age(rb87_sr86, sr87_sr86, np.ones_like(rb87_sr86))

print(f"\nRb-Sr whole rock isochron:")
print(f"  True age: {true_age_rbsr/1e6:.0f} Ma")
print(f"  Calculated age: {result['age']/1e6:.0f} Ma")
print(f"  Initial ⁸⁷Sr/⁸⁶Sr: {result['initial_ratio']:.6f} (true: {initial_ratio:.6f})")
print(f"  MSWD: {result['mswd']:.2f} (good fit if ~1)")

# Plot isochron
fig, ax = plt.subplots(figsize=(10, 7))

# Isochron line
x_line = np.array([0, max(rb87_sr86) * 1.1])
y_line = result['initial_ratio'] + result['slope'] * x_line

# Plot data and line
ax.scatter(rb87_sr86, sr87_sr86, s=140, c='#1565C0', marker='o', 
          edgecolors='black', linewidth=1.5, zorder=5, alpha=0.9,
          label='Whole rock samples')
ax.plot(x_line, y_line, '#D32F2F', linewidth=3, alpha=0.8,
       label=f'Isochron: Age = {result["age"]/1e6:.0f} Ma, MSWD = {result["mswd"]:.2f}')

# Error envelopes (simplified based on MSWD)
error_estimate = 0.002  # Approximate error
ax.fill_between(x_line, 
                y_line - error_estimate,
                y_line + error_estimate,
                alpha=0.15, color='#D32F2F')

ax.set_title('Rb-Sr Isochron: ⁸⁷Sr/⁸⁶Sr vs ⁸⁷Rb/⁸⁶Sr for Granite Whole Rock Samples', 
            fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper left', frameon=False)
clean_plot_style(ax)

# Add text box with results
textstr = f'Initial ⁸⁷Sr/⁸⁶Sr = {result["initial_ratio"]:.6f}'
props = dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.9, edgecolor='none')
ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=11,
       verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('05_geochronology_isochron.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved isochron plot")
plt.close()

# =============================================================================
# Example 5: Radiocarbon Dating
# =============================================================================
print("\n" + "=" * 70)
print("Example 5: Radiocarbon Dating")
print("=" * 70)

# Organic samples with known 14C/12C ratios
# Modern standard (100% modern carbon) = 1.0
# Half-life of C-14 = 5730 years

samples = {
    'Wood from archaeological site': 0.65,  # 65% modern
    'Charcoal from hearth': 0.45,           # 45% modern
    'Bone collagen': 0.35,                  # 35% modern
    'Ancient wood': 0.12                    # 12% modern
}

print("\nRadiocarbon ages:")
print(f"{'Sample':<30} {'¹⁴C/¹²C':<12} {'Age (BP)':<20} {'Cal. Age (BP)':<15}")
print("-" * 80)

for sample_name, fraction_modern in samples.items():
    age_bp, uncertainty_bp = radiocarbon_age(fraction_modern)
    
    # Simple calibration (actual calibration uses calibration curves)
    # This is a rough approximation
    cal_age_bp = age_bp * 1.05  # Simplified correction
    
    print(f"{sample_name:<30} {fraction_modern:.2f}       "
          f"{age_bp:.0f}±{uncertainty_bp:.0f}         {cal_age_bp:.0f}±50")

# Plot radiocarbon decay
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Decay curve
time_years = np.linspace(0, 50000, 1000)
fraction_remaining = np.exp(-DECAY_CONSTANTS['C14'] * time_years)

ax1.plot(time_years, fraction_remaining * 100, '#1976D2', linewidth=3)
ax1.axhline(y=1, color='#D32F2F', linestyle='--', linewidth=2, alpha=0.7,
           label='Detection limit')

# Mark sample ages
for sample_name, fraction_modern in samples.items():
    age, _ = radiocarbon_age(fraction_modern)
    if age < 50000:
        ax1.plot(age, fraction_modern * 100, 'o', color='#E65100', markersize=10, 
                markeredgecolor='black', markeredgewidth=1)

ax1.set_title('Radiocarbon Decay: Percent Modern Carbon Over Time (Years BP)', 
             fontsize=13, fontweight='bold', pad=15)
ax1.set_xlim(0, 50000)
ax1.legend(frameon=False, fontsize=10)
clean_plot_style(ax1)

# Right: Age vs fraction modern
fractions = np.linspace(0.01, 1.0, 100)
ages = np.array([radiocarbon_age(f)[0] for f in fractions])

ax2.plot(fractions * 100, ages / 1000, '#388E3C', linewidth=3)
ax2.set_title('Radiocarbon Age (ka BP) vs Percent Modern Carbon', 
             fontsize=13, fontweight='bold', pad=15)

# Mark useful age ranges with subtle bands
ax2.axhspan(0, 5, alpha=0.08, color='#388E3C')
ax2.axhspan(5, 25, alpha=0.08, color='#1976D2')
ax2.axhspan(25, 50, alpha=0.08, color='#F57C00')

# Add labels for age ranges
ax2.text(95, 2.5, 'Historical', ha='right', va='center', fontsize=9, color='#2E7D32')
ax2.text(95, 15, 'Holocene', ha='right', va='center', fontsize=9, color='#1565C0')
ax2.text(95, 37, 'Late Pleistocene', ha='right', va='center', fontsize=9, color='#E65100')

clean_plot_style(ax2)

plt.tight_layout()
plt.savefig('05_geochronology_radiocarbon.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved radiocarbon plots")
plt.close()

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
Generated plots:
  1. 05_geochronology_decay.png - Radioactive decay curves
  2. 05_geochronology_concordia.png - U-Pb concordia diagram
  3. 05_geochronology_isochron.png - Rb-Sr isochron
  4. 05_geochronology_radiocarbon.png - Radiocarbon decay

Key concepts demonstrated:
  • Radioactive decay follows exponential law
  • U-Pb dating uses concordia to detect disturbance
  • Rb-Sr isochron dating for determining crystallization age
  • Radiocarbon effective for <50,000 year timescales
  • MSWD assesses quality of isochron fit
  • Discordance indicates open system behavior

Applications:
  - Dating igneous crystallization (U-Pb, Rb-Sr)
  - Metamorphic events (mineral isochrons)
  - Archaeological chronology (radiocarbon)
  - Provenance studies (detrital zircon)
  - Tectonic history reconstruction
""")

print("✓ All geochronology examples complete!")
