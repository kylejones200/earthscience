# Minimalist Plot Design - Quick Reference

## Design Philosophy

> "Above all else show the data." - Edward Tufte

The updated plotting style follows Tufte's principles of maximizing the data-ink ratio by removing all non-essential visual elements.

## Quick Style Guide

### ✅ DO
- Use descriptive titles that explain the plot: "Y vs X (units)"
- Remove gridlines (unless absolutely necessary for reading values)
- Remove top and right spines
- Use frameless legends
- Use Material Design colors: `#1976D2` (blue), `#D32F2F` (red), `#388E3C` (green)
- Make data lines prominent (linewidth=2.5-3)
- Let the data be the hero

### ❌ DON'T
- Add gridlines by default
- Use all four spines/borders
- Add redundant axis labels when title is descriptive
- Use boxes/frames around legends
- Use basic HTML colors (red, blue, green)
- Make reference lines heavier than data lines
- Add decorative elements

## Code Template

```python
from plot_utils import clean_plot_style
import matplotlib.pyplot as plt

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot data with prominent styling
ax.plot(x, y, color='#1976D2', linewidth=2.5, label='Data')

# Descriptive title (eliminates need for axis labels)
ax.set_title('Temperature vs Depth in Borehole (°C and meters)', 
             fontsize=13, fontweight='bold', pad=15)

# Add legend without frame
ax.legend(frameon=False, fontsize=11)

# Apply clean styling (removes gridlines, top/right spines)
clean_plot_style(ax)

# Save
plt.tight_layout()
plt.savefig('output.png', dpi=300, bbox_inches='tight')
```

## Special Cases

### When to Keep Gridlines
For log-scale plots where gridlines help reading values:
```python
clean_plot_style(ax, remove_grid=False)
ax.grid(True, alpha=0.15, which='both', linewidth=0.5)
```

### Polar Plots (Rose Diagrams)
```python
ax = plt.subplot(111, projection='polar')
# ... plot data ...
ax.spines['polar'].set_linewidth(0.8)  # Thinner spine
```

## Color Palette

Primary colors (Material Design):
- Blue: `#1976D2` - Main data
- Red: `#D32F2F` - Important markers, reference lines
- Green: `#388E3C` - Secondary markers, positive values
- Gray: `#424242` - Text, neutral data

Light variants for fills:
- Light Blue: `#64B5F6`, `#90CAF9`
- Light Green: `#66BB6A`
- Light Gray: `#757575`

## Examples

All example files demonstrate the clean style:
- `examples/05_geochronology.py` - 4 plots
- `examples/06_directional_statistics.py` - 4 plots
- `examples/07_timeseries_analysis.py` - 4 plots
- `examples/08_image_analysis.py` - 3 plots
- `examples/example_statistics.py` - 1 plot (4 panels)

Run any example to see the minimalist design in action!

## Benefits

✓ Professional, publication-ready appearance  
✓ Easier to read - less visual clutter  
✓ Better focus on the actual data  
✓ Consistent style across all plots  
✓ Modern, clean aesthetic  

---

*For detailed documentation, see `PLOT_STYLE_UPDATES.md`*
