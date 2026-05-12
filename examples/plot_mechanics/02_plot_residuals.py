"""plot_residuals() — residual panel after fitting.

Demonstrates:
  - Standalone residual plot with the zero-reference line
  - get_residuals() for numeric inspection
  - Custom label arguments
"""

import numpy as np
from matplotlib import pyplot as plt

from pymultifit.fitters import GaussianFitter
from pymultifit.generators import multi_gaussian

# -- data & fit ----------------------------------------------------------------
params = [(20, -20, 2), (4, -5.5, 10), (5, -1, 0.5), (10, 3, 1), (4, 15, 3)]
x = np.linspace(-35, 35, 1500)
y = multi_gaussian(x, params=params, noise_level=0.2)

fitter = GaussianFitter(x, y)
fitter.fit([(10, -18, 1), (4, -5.5, 10), (5, -1, 0.5), (10, 3, 1), (4, 15, 3)])

# -- numeric inspection --------------------------------------------------------
residuals = fitter.get_residuals()
print("Residual statistics")
print(f"  mean  : {np.mean(residuals):+.6f}")
print(f"  std   : {np.std(residuals):.6f}")
print(f"  min   : {np.min(residuals):+.6f}")
print(f"  max   : {np.max(residuals):+.6f}")

# -- Example 1 : default residual plot ----------------------------------------
fig1, ax1 = plt.subplots(figsize=(12, 4))
fitter.plotter.plot_residuals(axis=ax1)
plt.tight_layout()

# -- Example 2 : custom labels ------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(12, 4))
fitter.plotter.plot_residuals(
    x_label="X data", y_label="y − ŷ", plot_title="Residuals of 5-component Gaussian fit", axis=ax2
)
plt.tight_layout()

plt.show()
