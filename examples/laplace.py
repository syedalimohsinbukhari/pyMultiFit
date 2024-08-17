"""Created on Aug 17 21:48:07 2024"""

import numpy as np

from src.pymultifit.fitters import LaplaceFitter
from src.pymultifit.generators import generate_multi_laplace_data

params = [(10, -20, 2), (4, -5.5, 10), (5, -1, 0.5), (10, 3, 1), (4, 15, 3)]

x = np.linspace(-35, 35, 1500)

noise_level = 0.2
y = generate_multi_laplace_data(x, params=params, noise_level=noise_level)

fitter = LaplaceFitter(n_fits=5, x_values=x, y_values=y)

guess = [(5, -18, 1), (4, -5.5, 10), (5, -1, 0.5), (10, 3, 1), (4, 15, 3)]

fitter.fit(guess)

plotter = fitter.plot_fit(show_individual=True, auto_label=True)
plotter.show()
