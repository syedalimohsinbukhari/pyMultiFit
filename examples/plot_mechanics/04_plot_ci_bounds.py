"""plot_confidence_intervals() — bootstrap confidence intervals.

Demonstrates:
  - Single CI level (95%)
  - Multiple CI levels with graduated alpha transparency
  - overall_ci vs. individual_ci modes
  - Fast multivariate normal sampling from covariance
  - One random generator for the whole script: reproducible, and every call draws new samples

The intervals are computed once, into a ``ConfidenceIntervals`` object that knows its levels and the parts it holds, and the
plotter only draws it.
"""

import numpy as np
from matplotlib import pyplot as plt

from pymultifit.fitters import GaussianFitter
from pymultifit.generators import multi_gaussian

# -- data & fit ----------------------------------------------------------------
rng = np.random.default_rng(42)  # created once and passed on, a new default_rng(42) per call would repeat the draws

params = [(10, -5, 2), (8, 5, 3)]
x = np.linspace(-15, 15, 500)
y = multi_gaussian(x, params=params, noise_level=0.7, rng_engine=rng)

fitter = GaussianFitter(x, y)
fitter.fit([(8, -4, 1.5), (6, 4, 2)])

# -- Example 1: Overall CI at 95% with plot_confidence_intervals() -----------------------
print("Running bootstrap (Example 1)…")
ci_95 = fitter.confidence_intervals(ci_levels=95, n_bootstrap=200, rng_engine=rng)

fig1, ax1 = plt.subplots(figsize=(10, 5))
fitter.plotter.plot_fit(axis=ax1)
fitter.plotter.plot_confidence_intervals(ci_95, axis=ax1)
ax1.set_title("Overall 95% bootstrap CI")
plt.tight_layout()

# -- Example 2: Nested CI bands with graduated transparency (68/90/95%) --------
print("Running bootstrap (Example 2)…")
ci_multi = fitter.confidence_intervals(ci_levels=[68, 90, 95], rng_engine=rng)

fig2, ax2 = plt.subplots(figsize=(10, 5))
fitter.plotter.plot_fit(axis=ax2, x_label="X", y_label="Y", data_label="Data", plot_title="Fit")
fitter.plotter.plot_confidence_intervals(ci_multi, axis=ax2)
ax2.set_title("Nested bootstrap CI bands (68/90/95%) — Note graduated transparency")
plt.tight_layout()

# -- Example 3: Individual component CIs ---------------------------------------
print("Running bootstrap (Example 3)…")

ci_ind = fitter.confidence_intervals(
    ci_levels=95, n_bootstrap=200, overall_ci=False, individual_ci=True, rng_engine=rng
)

ax3 = fitter.plotter.plot_confidence_intervals(ci_ind)
fitter.plotter.plot_fit(show_individuals=True, axis=ax3)
ax3.set_title("95% CI per individual component")
plt.tight_layout()

plt.show()
