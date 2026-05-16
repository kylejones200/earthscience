# Scientific reference tests

Golden-value tests for core geoscience algorithms. Each test documents its
reference (scipy, closed-form variogram limits, hand-solved kriging system, or
synthetic isochron with known decay constant).

Tolerances are tight (`rtol` 1e-9 to 1e-6) because inputs are analytic, not
noisy field data.
