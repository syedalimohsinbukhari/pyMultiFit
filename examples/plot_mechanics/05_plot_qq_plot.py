"""plot_qq_plot() — normality check on fit residuals.

Demonstrates:
  - Q-Q plot of residuals vs theoretical normal distribution
  - The Pearson r value in the legend as a quick normality gauge
  - Comparing a well-fitted vs poorly-fitted model
"""

import numpy as np
from matplotlib import pyplot as plt

from pymultifit.fitters import GaussianFitter, LaplaceFitter
from pymultifit.generators import multi_gaussian
from pymultifit.plot import qq_compare

# -- data ----------------------------------------------------------------------
# Gaussian signal — residuals should be approximately normal
params = [(15, 0, 3), (8, -10, 2)]
x = np.linspace(-20, 20, 1000)
y_gaussian = multi_gaussian(x, params=params, noise_level=0.3)

# -- Example 1: good Gaussian fit → residuals close to normal -----------------
gf = GaussianFitter(x, y_gaussian)
gf.fit([(12, 0, 2.5), (7, -9, 1.5)])

fig1, ax1 = plt.subplots(figsize=(6, 6))
gf.plotter.plot_qq_plot("Q-Q plot — GaussianFitter (expect r ≈ 1)", axis=ax1)
plt.tight_layout()

# -- Example 2 : wrong model family → residuals depart from normality ----------
# Fit a Laplace model to Gaussian-shaped data; residuals will be skewed
lf = LaplaceFitter(x_values=x, y_values=y_gaussian)
lf.fit([(15, 0, 3), (8, -10, 2)])

lf.plotter.plot_qq_plot("Q-Q plot — LaplaceFitter on Gaussian data (expect r < 1)")
plt.tight_layout()

# -- Example 3 : side-by-side comparison --------------------------------------
fig3, (ax_l, ax_r) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
gf.plotter.plot_qq_plot(plot_title="Gaussian fit (correct model)", axis=ax_l)
lf.plotter.plot_qq_plot(plot_title="Laplace fit (wrong model)", axis=ax_r)
fig3.suptitle("Q-Q plots — model comparison", fontsize=13)
plt.tight_layout()

# -- Example 4 : side-by-side auto-comparison ---------------------------------
fig4, (ax_l, ax_r) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
qq_compare(fitter_left=gf, fitter_right=lf, axes=(ax_l, ax_r)) # works with and without provided axes

fig4.tight_layout()

plt.show()
