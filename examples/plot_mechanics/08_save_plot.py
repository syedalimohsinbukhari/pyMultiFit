"""save_plot() — save any figure with format auto-detection.

Demonstrates:
  - Format inferred from file extension (.png / .pdf / .svg / .eps)
  - Automatic .png fallback when no extension is given
  - Saving a specific Figure object vs the current active figure
  - Passing extra keyword arguments to matplotlib's savefig
  - ValueError raised for unsupported extensions
"""

import numpy as np
from matplotlib import pyplot as plt

from pymultifit.fitters import GaussianFitter
from pymultifit.generators import multi_gaussian

# -- data & fit ----------------------------------------------------------------
params = [(15, -8, 2), (10, 5, 3)]
x = np.linspace(-18, 18, 800)
y = multi_gaussian(x, params=params, noise_level=0.25)

fitter = GaussianFitter(x, y)
fitter.fit([(12, -7, 1.5), (8, 4, 2.5)])

# -- Example 1: save as PNG (explicit extension) ------------------------------
fig1, ax1 = plt.subplots(figsize=(10, 5))
fitter.plotter.plot_fit(show_individuals=True, axis=ax1)
path1 = fitter.plotter.save_plot("fit_result.png", figure=fig1)
print(f"PNG  saved → {path1}")

# -- Example 2: save as PDF ---------------------------------------------------
fig2, (ax_fit, ax_res) = fitter.plotter.plot_fit_and_residuals(show_individuals=True)
path2 = fitter.plotter.save_plot("fit_and_residuals.pdf", figure=fig2)
print(f"PDF  saved → {path2}")

# -- Example 3: save as SVG (vector, ideal for publications) -----------------
fig3, ax3 = plt.subplots(figsize=(8, 6))
fitter.plotter.plot_parameter_correlation(axis=ax3)
path3 = fitter.plotter.save_plot("param_correlation.svg", figure=fig3)
print(f"SVG  saved → {path3}")

# -- Example 4: no extension → .png appended automatically -------------------
fig4, ax4 = plt.subplots(figsize=(6, 6))
fitter.plotter.plot_qq_plot(axis=ax4)
path4 = fitter.plotter.save_plot("qq_plot_no_ext", figure=fig4)
print(f"Auto saved → {path4}  (.png appended)")

# -- Example 5: extra savefig kwargs (transparent background) ----------------
fig5, ax5 = plt.subplots(figsize=(10, 5))
fitter.plotter.plot_prediction_intervals(pi_level=[90, 95], axis=ax5)
path5 = fitter.plotter.save_plot("pi_transparent.png", figure=fig5, transparent=True, dpi=200)
print(f"Transparent PNG saved → {path5}  (dpi=200)")

# -- Example 6: unsupported extension raises ValueError ----------------------
try:
    fitter.plotter.save_plot("output.xyz", figure=fig1)
except ValueError as exc:
    print(f"\nExpected error for .xyz: {exc}")

plt.show()
