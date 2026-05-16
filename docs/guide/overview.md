# Package overview

| Subpackage | Purpose |
|------------|---------|
| `earthsciences.statistics` | Descriptive stats, regression, hypothesis tests, extreme values |
| `earthsciences.timeseries` | Spectral analysis, filtering, wavelets |
| `earthsciences.spatial` | Variograms, kriging, interpolation, spatial autocorrelation |
| `earthsciences.multivariate` | PCA, clustering, classification |
| `earthsciences.directional` | Circular and spherical statistics |
| `earthsciences.geochronology` | Radiometric dating, decay systems |
| `earthsciences.geophysics` | Gravity, magnetics, seismic |
| `earthsciences.imaging` | Remote sensing indices, grain analysis |
| `earthsciences.workflows` | YAML-driven geostatistics pipelines |

Import the top-level package or submodules:

```python
import earthsciences as es
from earthsciences.spatial import ordinary_kriging, compute_variogram
```
