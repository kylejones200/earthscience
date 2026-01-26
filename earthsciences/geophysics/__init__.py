"""
Geophysical modeling and corrections

Gravity, magnetic, and seismic data processing and modeling.
"""

from .gravity import (
    G,
    EARTH_RADIUS,
    GAMMA,
    bouguer_correction,
    free_air_correction,
    terrain_correction
)

from .magnetics import (
    magnetic_anomaly_sphere,
    reduction_to_pole
)

from .seismic import (
    nmo_correction,
    stacking_velocity_analysis,
    eikonal_traveltime
)

__all__ = [
    'G',
    'EARTH_RADIUS',
    'GAMMA',
    'bouguer_correction',
    'free_air_correction',
    'terrain_correction',
    'magnetic_anomaly_sphere',
    'reduction_to_pole',
    'nmo_correction',
    'stacking_velocity_analysis',
    'eikonal_traveltime',
]
