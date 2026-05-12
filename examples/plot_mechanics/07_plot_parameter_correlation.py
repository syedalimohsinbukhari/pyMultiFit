"""plot_parameter_correlation() — heatmap of fitted-parameter correlations.

Demonstrates:
  - Auto-generated labels (p1, p2, …) for single-model fitters
  - Custom parameter labels
  - Model-aware auto-labels for MixedDataFitter
  - How to read the heatmap: values near ±1 signal identifiability issues

The correlation matrix is derived directly from the covariance matrix returned
by scipy.optimize.curve_fit, so no extra computation is required.
"""

import numpy as np
from matplotlib import pyplot as plt

from pymultifit import GAUSSIAN, LINE
from pymultifit.fitters import GaussianFitter, MixedDataFitter
from pymultifit.generators import multi_gaussian, multiple_models

# -- data & fits ---------------------------------------------------------------
# 3-component Gaussian
params_g = [(20, -15, 2), (10, 0, 5), (8, 12, 3)]
x_g = np.linspace(-25, 25, 1200)
y_g = multi_gaussian(x_g, params=params_g, noise_level=0.2)

gf = GaussianFitter(x_g, y_g)
gf.fit([(16, -14, 1.5), (8, 0, 4), (6, 11, 2)])

# Mixed: Line + 2 Gaussians
params_m = [(0.05, 2), (12, -10, 3), (7, 8, 4)]
x_m = np.linspace(-20, 20, 1200)
y_m = multiple_models(x_m, params=params_m, model_list=[LINE, GAUSSIAN, GAUSSIAN], noise_level=0.15)

mf = MixedDataFitter(x_m, y_m, model_list=[LINE, GAUSSIAN, GAUSSIAN])
mf.fit([(0.03, 1.5), (10, -9, 2.5), (5, 7, 3)])

# -- Example 1 : auto-labels (p1, p2, …) --------------------------------------
fig1, ax1 = plt.subplots(figsize=(7, 7))
gf.plotter.plot_parameter_correlation(axis=ax1)
ax1.set_title("3-component Gaussian — auto labels")
plt.tight_layout()

# -- Example 2 : physics-motivated custom labels -------------------------------
# For GaussianFitter each component has (amplitude, mu, sigma)
custom_labels = ["A₁", "μ₁", "σ₁", "A₂", "μ₂", "σ₂", "A₃", "μ₃", "σ₃"]

fig2, ax2 = plt.subplots(figsize=(8, 8))
gf.plotter.plot_parameter_correlation(param_labels=custom_labels, axis=ax2)
ax2.set_title("3-component Gaussian — physics labels")
plt.tight_layout()

# -- Example 3 : MixedDataFitter — model-aware auto-labels --------------------
# Labels are auto-generated as "Line_1_p1", "Gaussian_1_p1", etc.
fig3, ax3 = plt.subplots(figsize=(8, 8))
mf.plotter.plot_parameter_correlation(axis=ax3)
ax3.set_title("MixedDataFitter  (Line + 2×Gaussian) — model-aware labels")
plt.tight_layout()

print("MixedDataFitter auto-labels:", mf.plotter._default_param_labels())

plt.show()
