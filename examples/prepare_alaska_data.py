#!/usr/bin/env python3
"""
Prepare Alaska geochemistry data for config-driven examples.

This script merges the chemistry and geology data from AGDB4 to create
CSV files ready for geostatistical analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Data directory
data_dir = Path("data/AGDB4_text")
output_dir = Path("data/processed")
output_dir.mkdir(exist_ok=True)

print("="*70)
print("ALASKA GEOCHEMISTRY DATA PREPARATION")
print("="*70)

# Load geology data (has coordinates)
print("\n1. Loading geology data...")
geol = pd.read_csv(data_dir / "Geol_DeDuped.txt", 
                   sep=",", 
                   quotechar='"',
                   encoding='latin-1',  # Handle special characters
                   low_memory=False)

print(f"   Loaded {len(geol)} geological samples")
print(f"   Columns: {geol.columns.tolist()[:10]}...")

# Keep only samples with valid coordinates
geol = geol.dropna(subset=['LATITUDE', 'LONGITUDE'])
print(f"   Samples with coordinates: {len(geol)}")

# Load chemistry data
print("\n2. Loading chemistry data...")
chem_files = [
    "Chem_A_Br.txt",
    "Chem_C_Gd.txt", 
    "Chem_Ge_Os.txt",
    "Chem_P_Te.txt",
    "Chem_Th_Zr.txt"
]

chem_dfs = []
for file in chem_files:
    df = pd.read_csv(data_dir / file, sep=",", quotechar='"', 
                    encoding='latin-1', low_memory=False)
    chem_dfs.append(df)
    print(f"   Loaded {file}: {len(df)} measurements")

chem = pd.concat(chem_dfs, ignore_index=True)
print(f"\n   Total chemistry measurements: {len(chem)}")

# Filter for specific elements of interest
print("\n3. Extracting key elements...")

elements_of_interest = {
    'Au': 'GOLD',
    'Cu': 'COPPER', 
    'Pb': 'LEAD',
    'Zn': 'ZINC',
    'Ag': 'SILVER',
    'Mo': 'MOLYBDENUM',
    'As': 'ARSENIC',
    'Sb': 'ANTIMONY'
}

# Pivot chemistry data to wide format
print("   Pivoting to wide format...")

# Filter for best values only
chem_best = chem[chem['DATA_VALUE'].notna()].copy()

# Create element datasets
datasets = {}

for elem, name in elements_of_interest.items():
    print(f"\n4. Processing {name} ({elem})...")
    
    # Filter for this element
    elem_data = chem_best[chem_best['SPECIES'] == elem].copy()
    
    if len(elem_data) == 0:
        print(f"   No data found for {elem}")
        continue
    
    print(f"   Found {len(elem_data)} measurements")
    
    # Convert units to ppm
    elem_data['VALUE_PPM'] = elem_data['DATA_VALUE']
    
    # Handle percentage units (convert to ppm)
    pct_mask = elem_data['UNITS'] == 'pct'
    if pct_mask.any():
        elem_data.loc[pct_mask, 'VALUE_PPM'] = elem_data.loc[pct_mask, 'DATA_VALUE'] * 10000
        print(f"   Converted {pct_mask.sum()} percent values to ppm")
    
    # Group by sample and take mean (in case of duplicates)
    elem_summary = elem_data.groupby('DDPD_ID').agg({
        'VALUE_PPM': 'mean',
        'PARAMETER': 'first',
        'ANALYTIC_METHOD': 'first'
    }).reset_index()
    
    # Merge with geology data
    merged = geol.merge(elem_summary, on='DDPD_ID', how='inner')
    
    print(f"   Merged samples with coordinates: {len(merged)}")
    
    if len(merged) < 50:
        print(f"   ⚠ Too few samples for {elem}, skipping...")
        continue
    
    # Clean data
    merged = merged.dropna(subset=['LATITUDE', 'LONGITUDE', 'VALUE_PPM'])
    
    # Remove zeros and negatives
    merged = merged[merged['VALUE_PPM'] > 0]
    
    print(f"   Clean samples (>0): {len(merged)}")
    
    if len(merged) < 50:
        print(f"   ⚠ Too few clean samples for {elem}, skipping...")
        continue
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'SAMPLE_ID': merged['SPL_ID'],
        'LONGITUDE': merged['LONGITUDE'],
        'LATITUDE': merged['LATITUDE'],
        f'{elem}_PPM': merged['VALUE_PPM'],
        'MATERIAL': merged.get('MATERIAL', ''),
        'QUAD': merged.get('QUAD', '')
    })
    
    # Save to CSV
    output_file = output_dir / f"alaska_{elem.lower()}.csv"
    output_df.to_csv(output_file, index=False)
    
    datasets[elem] = {
        'file': output_file,
        'count': len(output_df),
        'stats': {
            'min': output_df[f'{elem}_PPM'].min(),
            'max': output_df[f'{elem}_PPM'].max(),
            'mean': output_df[f'{elem}_PPM'].mean(),
            'median': output_df[f'{elem}_PPM'].median()
        }
    }
    
    print(f"   ✓ Saved to {output_file.name}")
    print(f"   Stats: min={datasets[elem]['stats']['min']:.3f}, "
          f"max={datasets[elem]['stats']['max']:.3f}, "
          f"median={datasets[elem]['stats']['median']:.3f}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

for elem, info in datasets.items():
    print(f"\n{elem} ({elements_of_interest[elem]}):")
    print(f"  File: {info['file'].name}")
    print(f"  Samples: {info['count']}")
    print(f"  Range: {info['stats']['min']:.3f} - {info['stats']['max']:.3f} ppm")
    print(f"  Median: {info['stats']['median']:.3f} ppm")

print(f"\n✓ Prepared {len(datasets)} element datasets")
print(f"✓ Files saved to: {output_dir}/")
