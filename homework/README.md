# EESC2200 Homework 3 Solutions

## Overview
This folder contains solutions to Homework 3 on Radioactive Decay and Geochronology, solved using the **earthsciences** Python library.

## Files

### Input
- **`hw03.pdf`** - Original homework assignment (PDF format)
- **`hw03.txt`** - Extracted text from homework PDF

### Solutions
- **`hw03_solutions.md`** - Complete written solutions with detailed explanations
- **`solve_hw03.py`** - Python script that performs all calculations
- **`hw03_isochron_plot.png`** - Generated Rb-Sr isochron diagram

## Problem Summary

### Problem 1: Radioactive Decay - Nuclides
Analysis of Lanthanide/Rare Earth element isotopes (requires Chart of Nuclides reference material).

### Problem 2: Radioactive Decay - Heat Production
Calculations of heat generation from ²³⁵U decay:
- **Half-life calculation**: 704 million years
- **Number of half-lives**: 6.48 since Earth formation
- **Alpha particles**: 7 in the decay chain
- **Mass defect**: 0.0498 AMU
- **Energy released**: 46.4 MeV per atom
- **Heat at 4 Ga**: 1.45 × 10²⁶ cal/Ma
- **Ocean temperature rise**: 103.8°C/Ma (if all heat transferred instantaneously)

### Problem 3: Geochronology - Dating
#### U-Pb Dating Results:
| Sample | ²⁰⁶Pb/²³⁸U Ratio | Age (Ma) |
|--------|------------------|----------|
| Dike V | 0.03151 ± 0.0001 | 200.0 ± 0.6 |
| Dike W | 0.003107 ± 0.000015 | 20.0 ± 0.1 |

#### Rb-Sr Isochron Results:
| Sample | Age (Ma) | Initial ⁸⁷Sr/⁸⁶Sr | Method |
|--------|----------|-------------------|--------|
| Whole Rock (A,B,D,G) | 2000 | 0.703 | Isochron |
| Rock B Minerals | 200 | 0.711 | Mineral isochron |
| Rock G Minerals | 200 | 0.729 | Mineral isochron |

### Problem 4: Geological Interpretation
Comprehensive synthesis of the isotopic ages to constrain geological events:
1. **~2000 Ma**: Initial volcanic crystallization
2. **~200 Ma**: Regional metamorphism and Dike V intrusion
3. **~20 Ma**: Post-metamorphic Dike W intrusion

## Running the Solution

To reproduce the calculations and generate the plot:

```bash
cd /Users/k.jones/Desktop/earth\ science/homework
python solve_hw03.py
```

### Requirements
The script uses:
- NumPy
- Matplotlib
- SciPy
- earthsciences library (parent directory)

### Output
Running the script will:
1. Print all calculations to the console
2. Generate `hw03_isochron_plot.png` - Rb-Sr isochron diagram showing:
   - Whole rock data (blue circles)
   - Rock B minerals (red squares)
   - Rock G minerals (green triangles)
   - Three isochron lines with calculated ages

## Key Features

The solution demonstrates use of the earthsciences library for:
- Radioactive decay calculations
- Geochronology (U-Pb and Rb-Sr dating)
- Isochron age determination
- Error propagation
- Geological interpretation

## Visualization

The isochron plot (`hw03_isochron_plot.png`) shows:
- **Whole Rock Isochron**: 2000 Ma (blue line) - records initial crystallization
- **Rock B Mineral Isochron**: 200 Ma (red dashed line) - records metamorphism
- **Rock G Mineral Isochron**: 200 Ma (green dash-dot line) - records metamorphism
- All three isochrons plotted with data points on the same diagram

## Scientific Context

This homework demonstrates fundamental concepts in:
- **Nuclear Physics**: Radioactive decay, mass-energy equivalence
- **Geochronology**: U-Pb and Rb-Sr dating systems
- **Metamorphic Petrology**: Isotopic resetting, closure temperatures
- **Tectonics**: Orogenic cycles, continental assembly

The calculated ages tell a story of:
- Ancient volcanic rocks (2000 Ma)
- Later metamorphism (200 Ma)
- Recent magmatic intrusion (20 Ma)

This is typical of continental collision zones where old rocks are reworked during mountain-building events.

---

**Solutions prepared using the earthsciences Python library**  
*Demonstrating practical applications of computational geoscience*
