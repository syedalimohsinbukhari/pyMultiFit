"""Created on Jul 18 14:05:42 2024"""

import numpy as np

from src.pymultifit import SkewedNormal
from src.pymultifit.backend import generate_multi_skewed_normal_data

# Generate sample data
np.random.seed(42)
x = np.linspace(-15, 15, 500)
params = [(12, 2, -7, 1), (3, -4, 2, 2), (4, 2, 7, 1.5)]
y = generate_multi_skewed_normal_data(x, params, noise_level=0.2)

# Fit the data
fitter = SkewedNormal(n_fits=3, x_values=x, y_values=y)
initial_guesses = [4, 6, -6, 0.5, 3, -4, 2, 2, 4, 2, 1, 1.5]
fitter.fit(initial_guesses)
print(fitter.get_value_error_pair(only_values=True))

# Plot the result
plotter = fitter.plot_fit(show_individual=True, auto_label=True)
plotter.show()
