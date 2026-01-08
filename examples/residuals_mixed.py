"""Example demonstrating residual plotting with MixedDataFitter.

This example shows how to use the new residual functions with mixed models:
1. get_residuals() - Get residual values
2. plot_residuals() - Plot only residuals
3. plot_fit_and_residuals() - Combined plot of fit and residuals
"""

import numpy as np
from matplotlib import pyplot as plt

from src.pymultifit import GAUSSIAN, LAPLACE, LINE
from src.pymultifit.fitters import MixedDataFitter
from src.pymultifit.generators import multiple_models

# Generate data from multiple different model types
x = np.linspace(start=-50, stop=50, num=10_000)
noise_level = 0.1

# Parameters: [line, gaussian, gaussian, laplace, laplace, gaussian]
params = [(-0.1, 5), (20, -20, 2), (4, -5.5, 10), (5, -1, 0.5), (10, 3, 1), (4, 15, 3)]

y = multiple_models(
    x=x,
    params=params,
    model_list=[LINE, GAUSSIAN, GAUSSIAN, LAPLACE, LAPLACE, GAUSSIAN],
    noise_level=noise_level
)

# Create fitter with mixed models
guess = [(0, 2), (1, -20, 1), (1, -5, 5), (3, -1, 0.5), (7, 2, 1), (1, 15, 2)]

fitter = MixedDataFitter(
    x_values=x,
    y_values=y,
    model_list=['line'] + ['gaussian'] * 2 + ['laplace'] * 2 + ['gaussian']
)

# Perform fitting
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
fitter.plot_residuals(
    x_label='X data',
    y_label='Residuals',
    title='Residuals of Mixed Model Fit',
    axis=ax1
)
plt.tight_layout()
plt.savefig('example_mixed_residuals_only.png', dpi=150, bbox_inches='tight')
print("\nSaved: example_mixed_residuals_only.png")

# Example 3: Combined plot of fit and residuals
fig2, (ax_fit, ax_res) = fitter.plot_fit_and_residuals(
    show_individuals=True,
    x_label='X data',
    y_label='Y data',
    title='Mixed Model Fit (Line + Gaussians + Laplace)',
    data_label='Data'
)
plt.savefig('example_mixed_fit_and_residuals.png', dpi=150, bbox_inches='tight')
print("Saved: example_mixed_fit_and_residuals.png")

plt.show()
