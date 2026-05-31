"""Frozen parameters with MixedDataFitter — sparse dict API.

Demonstrates four scenarios:

1. Simple 2-component mix (Gaussian and Gamma):
   Freeze sigma in the Gaussian and loc in the Gamma.

2. Large 6-component mix (Line + 3×Gaussian + 2×Laplace):
   Only 1 of the 6 components has a frozen parameter — shows how the sparse dict keeps the call clean regardless of how many models are in the mix.

3. Loc/scale auto-padding in MixedDataFitter:
   Passing a pn_par-length inner mask for a loc/scale distribution; the loc parameter is auto-padded as False (unfrozen).

4. Partial freeze — Line + Gaussian + Gaussian:
   The Line (idx 0) is fully frozen, Gaussian 1 (idx 1) is entirely free, and only mu of Gaussian 2 (idx 2) is frozen.
   The sparse dict only lists the two components that need freezing; Gaussian 1 is simply omitted.
"""

import numpy as np
from matplotlib import pyplot as plt

from pymultifit import EPSILON, GAMMA, GAUSSIAN, LAPLACE, LINE
from pymultifit.fitters import MixedDataFitter
from pymultifit.generators import multiple_models, multi_gamma, multi_gaussian

rng = np.random.default_rng(1)

# ---------------------------------------------------------------------------
# 1. Gaussian + Gamma — freeze sigma in Gaussian (idx 0), loc in Gamma (idx 1)
# ---------------------------------------------------------------------------
x = np.linspace(EPSILON, 15, 1000)

# True params — Gaussian: (amp, mu, sigma), Gamma: (amp, shape, scale, loc)
gauss_true = (5.0, 4.0, 1.2)
gamma_true = (3.0, 3.0, 1.5, 0.0)

y = (
    multi_gaussian(x, params=[gauss_true], noise_level=0.0)
    + multi_gamma(x, params=[gamma_true], noise_level=0.0)
    + rng.normal(0, 0.05, len(x))
)

fitter = MixedDataFitter(x, y, model_list=[GAUSSIAN, GAMMA])

fitter.fit(
    p0=[gauss_true, gamma_true],
    frozen={0: [False, False, True], 1: [False, False, False, True]},  # freeze sigma in Gaussian  # freeze loc in Gamma
)
print("=== Gaussian + Gamma — sigma & loc frozen ===")
print("Fitted params:", fitter.params)

fig, ax = plt.subplots(figsize=(10, 5))
fitter.plotter.plot_fit(
    show_individuals=True, axis=ax, plot_title="MixedDataFitter — sigma(Gaussian) & loc(Gamma) frozen", is_scatter=True
)
fig.tight_layout()

# ---------------------------------------------------------------------------
# 2. Large 6-component mix — only component 3 (Laplace) has frozen params
# ---------------------------------------------------------------------------
x2 = np.linspace(-50, 50, 10_000)

params_true = [
    (-0.1, 5),  # line: (slope, intercept)
    (20, -20, 2),  # gaussian: (amp, mu, sigma)
    (4, -5.5, 10),  # gaussian
    (5, -1, 0.5),  # laplace: (amp, mu, scale)
    (10, 3, 1),  # laplace
    (4, 15, 3),  # gaussian
]
model_order = [LINE, GAUSSIAN, GAUSSIAN, LAPLACE, LAPLACE, GAUSSIAN]

y2 = multiple_models(x=x2, params=params_true, model_list=model_order, noise_level=0.1)

fitter2 = MixedDataFitter(x2, y2, model_list=model_order)

# Only freeze mu in the first Laplace (component index 3) — all others free
fitter2.fit(
    p0=[(-0.1, 5), (18, -20, 1), (3, -5, 8), (4, -1, 0.5), (8, 3, 1), (3, 15, 2)],
    frozen={3: [False, True, False]},  # freeze mu of first Laplace
)
print("\n=== 6-component mix — only Laplace[0] mu frozen ===")
print("Fitted params:", fitter2.params)

fig2, ax2 = plt.subplots(figsize=(12, 5))
fitter2.plotter.plot_fit(
    show_individuals=True,
    axis=ax2,
    plot_title="MixedDataFitter — Laplace[0] mu frozen (sparse: 1 of 6 components)",
    is_scatter=True,
)
fig2.tight_layout()

# ------------------------------------------------------------------------------
# 3. Loc/scale auto-padding — passing pn_par-length mask for the Gamma component
# ------------------------------------------------------------------------------
# model_list = [GAUSSIAN, GAMMA]
# Gamma has n_par=4, pn_par=3 → inner mask of length 3 is valid; loc auto-False
fitter3 = MixedDataFitter(x, y, model_list=[GAUSSIAN, GAMMA])

fitter3.fit(
    p0=[gauss_true, gamma_true], frozen={1: [False, False, False]}  # pn_par-length mask → loc auto-padded as False
)
print("\n=== Gaussian + Gamma — pn_par mask for Gamma (loc auto-unfrozen) ===")
print("Fitted params:", fitter3.params)

# Same but freeze all primary params in Gamma — only loc is free
fitter3.fit(
    p0=[gauss_true, gamma_true], frozen={1: [True, True, True]}  # pn_par-length mask → loc auto-padded as False (free)
)
print("\n=== Gaussian + Gamma — Gamma primary params frozen, only loc free ===")
print("Fitted params:", fitter3.params)

# ---------------------------------------------------------------------------
# 4. Partial freeze — Line + Gaussian + Gaussian
#    - Line (idx 0): both parameters frozen → {0: [True, True]}
#    - Gaussian (idx 1): entirely free → not listed in dict
#    - Gaussian (idx 2): only mu frozen → {2: [False, True, False]}
# ---------------------------------------------------------------------------
x3 = np.linspace(-20, 20, 2000)

line_true = (0.5, 2.0)  # (slope, intercept)
gauss1_true = (10.0, -8.0, 2.0)  # (amp, mu, sigma) — free
gauss2_true = (7.0, 5.0, 1.5)  # (amp, mu, sigma) — mu frozen

params3 = [line_true, gauss1_true, gauss2_true]
model_order3 = [LINE, GAUSSIAN, GAUSSIAN]

y3 = multiple_models(x=x3, params=params3, model_list=model_order3, noise_level=0.2)

fitter4 = MixedDataFitter(x3, y3, model_list=model_order3)

fitter4.fit(
    p0=[(0.5, 1), gauss1_true, gauss2_true],
    frozen={
        0: [True, False],  # freeze both Line params
        # idx 1 (Gaussian 1) omitted — fully free
        2: [False, True, False],  # freeze only mu of Gaussian 2
    },
)
print("\n=== Line + Gaussian + Gaussian — Line frozen, Gaussian 1 free, Gaussian 2 mu frozen ===")
print("Fitted params:", fitter4.params)

fig3, ax3 = plt.subplots(figsize=(12, 5))
fitter4.plotter.plot_fit(
    show_individuals=True,
    axis=ax3,
    plot_title="Line (frozen) + Gaussian (free) + Gaussian (mu frozen)",
    is_scatter=True,
)
fig3.tight_layout()

plt.show()
