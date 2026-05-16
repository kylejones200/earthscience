# Geochronology case studies

End-to-end workflows for radiometric dating, isochron regression, and geological interpretation using **earthsciences**.

## Metamorphic volcanic terrane

**Script:** `geochronology_metamorphic_terrane.py`
**Figure:** `geochronology_metamorphic_terrane_isochron.png`

### Geological setting

A package of metamorphosed volcanic layers (A–G) is intruded by two crosscutting dikes (V and W). Isotopic data constrain:

| Event | Method | Age (Ma) | Interpretation |
|-------|--------|----------|----------------|
| Protolith crystallization | Rb–Sr whole-rock isochron | ~1500 | Primary magmatic age |
| Peak metamorphism + Dike V | U–Pb zircon; Rb–Sr minerals | ~200 | Thermal resetting |
| Post-orogenic intrusion | U–Pb zircon (Dike W) | ~20 | Younger magmatic pulse |

### What the example covers

1. **Radiogenic heat** — Mass–energy balance for ²³⁵U decay; crustal heat budget vs. hypothetical ocean heating.
2. **U–Pb geochronology** — ²⁰⁶Pb/²³⁸U ages for zircon with analytical uncertainty propagation.
3. **Rb–Sr isochrons** — Whole-rock and mineral separates via `isochron_dating()`.
4. **Multi-isochron diagram** — Whole-rock vs. mineral isochrons on one plot.
5. **Synthesis** — Integrating discordant ages into a tectonic timeline.

### Key library functions

```python
from earthsciences.geochronology import (
    DECAY_CONSTANTS,
    calculate_age,
    isochron_dating,
)

# U-Pb age from 206Pb/238U
age_years = calculate_age(1.0, pb206_u238_ratio, DECAY_CONSTANTS["U238"])

# Rb-Sr isochron
result = isochron_dating(rb_sr, sr_ratio, np.ones(n), DECAY_CONSTANTS["Rb87"])
print(f"Age: {result['age'] / 1e6:.1f} Ma")
print(f"Initial ratio: {result['initial_ratio']:.6f}")
```

### Run

```bash
cd examples
uv run python geochronology_metamorphic_terrane.py
```

### Scientific takeaways

- **Closure temperature:** Mineral isochrons date cooling through Rb–Sr closure (~300–500°C for biotite/muscovite), not the instant of metamorphism.
- **Whole-rock closure:** Bulk samples can preserve protolith ages when metamorphism did not homogenize Sr at the hand-specimen scale.
- **Crosscutting relations:** Younger dikes bracket the minimum age of structures they cut.
- **Concordant systems:** Agreement between U–Pb (Dike V) and mineral Rb–Sr (Rock G) strengthens the ~200 Ma event interpretation.

---

## Additional geochronology examples

**`tutorials/05_geochronology.py`** — Synthetic tutorials for decay curves, concordia diagrams, generic isochrons, and radiocarbon calibration.

```bash
cd examples/tutorials
uv run python 05_geochronology.py
```

Outputs: `05_geochronology_decay.png`, `05_geochronology_concordia.png`, `05_geochronology_isochron.png`, `05_geochronology_radiocarbon.png`.
