"""plot_prediction_intervals() — where new observations are expected to fall.

Demonstrates:
  - Single PI level
  - Multiple PI levels as nested bands (widest first)
  - Comparison with bootstrap CI to visualise the difference

Prediction intervals are *wider* than confidence intervals because they must
account for both the uncertainty in the model parameters *and* the irreducible
scatter of individual observations.
"""

import numpy as np
from matplotlib import pyplot as plt

from pymultifit.fitters import GaussianFitter
from pymultifit.generators import multi_gaussian

# -- data & fit ----------------------------------------------------------------
params = [(15, 0, 4), (8, -12, 2)]
x = np.linspace(-22, 22, 800)
y = multi_gaussian(x, params=params, noise_level=0.5)

fitter = GaussianFitter(x, y)
fitter.fit([(12, 0, 3), (7, -11, 1.5)])

# -- Example 1 : single PI level -----------------------------------------------
fig1, ax1 = plt.subplots(figsize=(10, 5))
fitter.plotter.plot_prediction_intervals(pi_level=95, axis=ax1)
ax1.set_title("95 % Prediction Interval")
plt.tight_layout()

# -- Example 2 : nested PI bands -----------------------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 5))
fitter.plotter.plot_prediction_intervals(pi_level=[68, 90, 95], axis=ax2)
ax2.set_title("Nested Prediction Interval bands  (68 / 90 / 95 %)")
plt.tight_layout()

# -- Example 3: PI vs CI side-by-side -----------------------------------------
print("Running bootstrap for CI comparison…")
results_ci = fitter.ci_bounds(
    ci_levels=95,
    n_bootstrap=300,
    overall_ci=True,
    individual_ci=False,
    seed=0
)

fig3, (ax_pi, ax_ci) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

fitter.plotter.plot_prediction_intervals(pi_level=95, axis=ax_pi)
ax_pi.set_title("95% Prediction Interval\n(accounts for observation scatter)")

fitter.plotter.plot_fit(axis=ax_ci)
fitter.plotter.plot_ci_bounds(
    results=results_ci,
    ci_levels=95,
    overall_ci=True,
    individual_ci=False,
    axis=ax_ci
)
ax_ci.set_title("95% Bootstrap CI\n(uncertainty of mean response only)")

fig3.suptitle("PI vs CI — PI is always wider", fontsize=13)
plt.tight_layout()

plt.show()
