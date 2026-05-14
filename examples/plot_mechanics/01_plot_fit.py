"""plot_fit() — composite fit overlaid on raw data.

Demonstrates:
  - Basic fit plot with default labels
  - Custom axis / label arguments
  - show_individuals=True  (dashed per-component lines)
  - Same mechanics work for MixedDataFitter via the plotter property
"""

import numpy as np
from matplotlib import pyplot as plt

from pymultifit import GAUSSIAN, LAPLACE, LINE
from pymultifit.fitters import GaussianFitter, MixedDataFitter
from pymultifit.generators import multi_gaussian, multiple_models

# -- data ----------------------------------------------------------------------
rng = np.random.default_rng(0)

params_g = [(20, -20, 2), (4, -5.5, 10), (5, -1, 0.5), (10, 3, 1), (4, 15, 3)]
x_g = np.linspace(-35, 35, 1500)
y_g = multi_gaussian(x_g, params=params_g, noise_level=0.2)

params_m = [(-0.1, 3), (8, -15, 3), (6, 5, 2), (4, 20, 4)]
x_m = np.linspace(-35, 35, 2000)
y_m = multiple_models(x_m, params=params_m, model_list=[LINE, GAUSSIAN, LAPLACE, GAUSSIAN], noise_level=0.1)

# -- fit -----------------------------------------------------------------------
gf = GaussianFitter(x_g, y_g)
gf.fit([(10, -18, 1), (4, -5.5, 10), (5, -1, 0.5), (10, 3, 1), (4, 15, 3)])

mf = MixedDataFitter(x_m, y_m, model_list=[LINE, GAUSSIAN, LAPLACE, GAUSSIAN])
mf.fit([(0, 2), (6, -15, 2), (4, 5, 1), (3, 20, 3)])

# -- Example 1: default labels -----------------------------------------------
fig1, ax1 = plt.subplots(figsize=(12, 5))
gf.plotter.plot_fit(axis=ax1)
ax1.set_title("GaussianFitter — default labels")
plt.tight_layout()

# -- Example 2: custom labels + show_individuals -----------------------------
fig2, ax2 = plt.subplots(figsize=(12, 5))
gf.plotter.plot_fit(
    show_individuals=True,
    x_label="X data",
    y_label="Amplitude",
    plot_title="5-component Gaussian fit",
    data_label="Observations",
    fit_label="Composite fit",
    axis=ax2,
)
plt.tight_layout()

# -- Example 3 : MixedDataFitter with individuals -----------------------------
fig3, ax3 = plt.subplots(figsize=(12, 5))
mf.plotter.plot_fit(
    show_individuals=True,
    x_label="X",
    y_label="Y",
    plot_title="Mixed model fit  (Line + Gaussian + Laplace + Gaussian)",
    axis=ax3,
)
plt.tight_layout()

plt.show()
