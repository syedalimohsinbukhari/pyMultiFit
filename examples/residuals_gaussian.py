"""Example demonstrating residual plotting with GaussianFitter.

This example shows how to use the new residual functions:
1. get_residuals() - Get residual values
2. plot_residuals() - Plot only residuals
3. plot_fit_and_residuals() - Combined plot of fit and residuals
"""

import numpy as np
from matplotlib import pyplot as plt

from src.pymultifit.fitters import GaussianFitter
from src.pymultifit.generators import multi_gaussian

# Generate multi-modal Gaussian data with noise
params = [(20, -20, 2), (4, -5.5, 10), (5, -1, 0.5), (10, 3, 1), (4, 15, 3)]
x = np.linspace(-35, 35, 1500)
noise_level = 0.2
y = multi_gaussian(x, params=params, noise_level=noise_level)

# Create fitter and perform fitting
fitter = GaussianFitter(x_values=x, y_values=y)
guess = [(10, -18, 1), (4, -5.5, 10), (5, -1, 0.5), (10, 3, 1), (4, 15, 3)]
fitter.fit(guess)

# Example 1: Get residuals as array
residuals = fitter.get_residuals()
print(f"Residual statistics:")
print(f"  Mean: {np.mean(residuals):.6f}")
print(f"  Std:  {np.std(residuals):.6f}")
print(f"  Min:  {np.min(residuals):.6f}")
print(f"  Max:  {np.max(residuals):.6f}")

# Example 2: Plot only residuals
fig1, ax1 = plt.subplots(1, 1, figsize=(12, 4))
fitter.plotter.plot_residuals(x_label="X data", y_label="Residuals", plot_title="Residuals of Gaussian Fit", axis=ax1)
plt.tight_layout()
plt.savefig("example_gaussian_residuals_only.png", dpi=150, bbox_inches="tight")
print("\nSaved: example_gaussian_residuals_only.png")

# Example 3: Combined plot of fit and residuals
fig2, (ax_fit, ax_res) = fitter.plotter.plot_fit_and_residuals(
    show_individuals=True,
    x_label="X data",
    y_label="Y data",
    plot_title="Multi-Gaussian Fit with Residuals",
    data_label="Data",
)
plt.savefig("example_gaussian_fit_and_residuals.png", dpi=150, bbox_inches="tight")
print("Saved: example_gaussian_fit_and_residuals.png")

plt.show()
