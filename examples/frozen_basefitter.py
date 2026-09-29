"""Frozen parameters with BaseFitter — non-loc/scale and loc/scale distributions.

Demonstrates three scenarios:

1. GaussianFitter (n_par=3, no secondary params):
   Freeze the mean (mu) so only amplitude and sigma are optimised.

2. GammaFitter (n_par=4, pn_par=3, sn_par={'loc': 0.0}):
   a) Freeze loc using the full n_par mask.
   b) Freeze loc using the shorter pn_par mask (auto-padded).

3. ExponentialFitter (n_par=3, pn_par=2, sn_par={'loc': 0.0}):
   Freeze loc at zero using the shorter pn_par mask (auto-padded).
"""

import numpy as np
from matplotlib import pyplot as plt

from pymultifit import EPSILON
from pymultifit.fitters import GaussianFitter, GammaFitter, ExponentialFitter
from pymultifit.generators import multi_gaussian, multi_gamma, multi_exponential

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# 1. GaussianFitter — freeze mu of each component
# ---------------------------------------------------------------------------
# True params: (amplitude, mu, sigma)
gauss_params = [(10, -5, 1.5), (8, 0, 2.0), (6, 6, 1.0)]
x_gauss = np.linspace(-12, 12, 800)
y_gauss = multi_gaussian(x_gauss, params=gauss_params, noise_level=0.15)

gauss_fitter = GaussianFitter(x_values=x_gauss, y_values=y_gauss)

# frozen=[False, True, False] freezes mu (index 1) in every component
gauss_fitter.fit(p0=[(10, -5, 1.5), (8, 0, 2.0), (6, 6, 1.0)], frozen=[False, True, False])
print("=== GaussianFitter — mu frozen ===")
print("Fitted params:", gauss_fitter.params)

# ---------------------------------------------------------------------------
# 2a. GammaFitter — freeze loc using the full n_par mask
# ---------------------------------------------------------------------------
# True params: (amplitude, shape, scale, loc)
gamma_params = [(3, 3, 1.5, 0.0), (2, 6, 0.8, 0.0)]
x_gamma = np.linspace(EPSILON, 20, 800)
y_gamma = multi_gamma(x_gamma, params=gamma_params, noise_level=0.05)

gamma_fitter = GammaFitter(x_values=x_gamma, y_values=y_gamma)

# n_par=4 mask — explicitly freeze the 4th parameter (loc)
gamma_fitter.fit(p0=[(3, 3, 1.5, 0.0), (2, 6, 0.8, 0.0)], frozen=[False, False, False, True])
print("\n=== GammaFitter — loc frozen (n_par mask) ===")
print("Fitted params:", gamma_fitter.params)

# ---------------------------------------------------------------------------
# 2b. GammaFitter — freeze loc using the shorter pn_par mask (auto-padded)
# ---------------------------------------------------------------------------
# pn_par=3: passing 3 bools auto-pads a False for loc
gamma_fitter.fit(
    p0=[(3, 3, 1.5, 0.0), (2, 6, 0.8, 0.0)], frozen=[False, False, False]  # equivalent to [False, False, False, False]
)
print("\n=== GammaFitter — pn_par mask (loc auto-unfrozen) ===")
print("Fitted params:", gamma_fitter.params)

# Freeze *all* primary params — only loc is free (uncommon but valid)
gamma_fitter.fit(p0=[(3, 3, 1.5, 0.0), (2, 6, 0.8, 0.0)], frozen=[True, True, True])  # loc auto-padded as False
print("\n=== GammaFitter — pn_par mask (only loc free) ===")
print("Fitted params:", gamma_fitter.params)

# ---------------------------------------------------------------------------
# 3. ExponentialFitter — freeze loc at zero using pn_par mask
# ---------------------------------------------------------------------------
# True params: (amplitude, rate, loc)
exp_params = [(5, 0.8, 0.0), (3, 1.5, 0.0)]
x_exp = np.linspace(EPSILON, 8, 600)
y_exp = multi_exponential(x_exp, params=exp_params, noise_level=0.05)

exp_fitter = ExponentialFitter(x_values=x_exp, y_values=y_exp)

# pn_par=2: passing 2 bools auto-pads False for loc
exp_fitter.fit(
    p0=[(5, 0.8), (3, 1.5)],  # omitting loc — BaseFitter fills default 0.0
    frozen=[False, False],  # loc auto-padded as False (free)
)
print("\n=== ExponentialFitter — pn_par mask, loc free ===")
print("Fitted params:", exp_fitter.params)

exp_fitter.fit(p0=[(5, 0.8, 0.0), (3, 1.5, 0.0)], frozen=[False, False, True])  # full n_par mask — loc frozen at 0.0
print("\n=== ExponentialFitter — n_par mask, loc frozen ===")
print("Fitted params:", exp_fitter.params)

# ---------------------------------------------------------------------------
# Quick plot — Gaussian frozen-mu example
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
gauss_fitter.plotter.plot_fit(show_individuals=True, axis=ax, plot_title="GaussianFitter — mu frozen", is_scatter=True)
fig.tight_layout()
plt.show()
