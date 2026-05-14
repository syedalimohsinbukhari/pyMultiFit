"""plot_fit_and_residuals() — combined two-panel figure.

Demonstrates:
  - Fit panel (3 parts) stacked above residuals panel (1 part), shared x-axis
  - show_individuals in the top panel
  - Unpacking the (fig, (ax_fit, ax_res)) return value for further customization
  - MixedDataFitter version
"""

import numpy as np
from matplotlib import pyplot as plt

from pymultifit import GAUSSIAN, LAPLACE, LINE
from pymultifit.fitters import GaussianFitter, MixedDataFitter
from pymultifit.generators import multi_gaussian, multiple_models

# -- data & fits ---------------------------------------------------------------
params_g = [(20, -20, 2), (4, -5.5, 10), (5, -1, 0.5), (10, 3, 1), (4, 15, 3)]
x_g = np.linspace(-35, 35, 1500)
y_g = multi_gaussian(x_g, params=params_g, noise_level=0.2)

params_m = [(-0.1, 3), (8, -15, 3), (6, 5, 2), (4, 20, 4)]
x_m = np.linspace(-35, 35, 2000)
y_m = multiple_models(x_m, params=params_m, model_list=[LINE, GAUSSIAN, LAPLACE, GAUSSIAN], noise_level=0.1)

gf = GaussianFitter(x_g, y_g)
gf.fit([(10, -18, 1), (4, -5.5, 10), (5, -1, 0.5), (10, 3, 1), (4, 15, 3)])

mf = MixedDataFitter(x_m, y_m, model_list=[LINE, GAUSSIAN, LAPLACE, GAUSSIAN])
mf.fit([(0, 2), (6, -15, 2), (4, 5, 1), (3, 20, 3)])

# -- Example 1 : GaussianFitter, default --------------------------------------
fig1, (ax_fit1, ax_res1) = gf.plotter.plot_fit_and_residuals()

# -- Example 2 : GaussianFitter, show individuals + custom labels --------------
fig2, (ax_fit2, ax_res2) = gf.plotter.plot_fit_and_residuals(
    show_individuals=True,
    x_label="X data",
    y_label="Amplitude",
    plot_title="5-component Gaussian fit",
    data_label="Observations",
    fit_label="Composite fit",
)
# further customisation of the returned axes
ax_res2.set_ylim(-1.5, 1.5)

# -- Example 3 : MixedDataFitter -----------------------------------------------
fig3, (ax_fit3, ax_res3) = mf.plotter.plot_fit_and_residuals(
    show_individuals=True, x_label="X", y_label="Y", plot_title="Mixed model — Line + Gaussian + Laplace + Gaussian"
)

plt.show()
