"""Created on Jul 18 01:11:32 2024"""

import numpy as np

from pymultifit.fitters import GaussianFitter
from pymultifit.fitters.multi_generators import generate_multi_gaussian_data

params = [(20, -20, 2), (4, -5.5, 10), (5, -1, 0.5), (10, 3, 1), (4, 15, 3)]

x = np.linspace(-35, 35, 1500)

noise_level = 0.2
y = generate_multi_gaussian_data(x, params, noise_level=noise_level)

fitter = GaussianFitter(5, x, y)

guess = [(10, -18, 1), (4, -5.5, 10), (5, -1, 0.5), (10, 3, 1), (4, 15, 3)]

fitter.fit(guess)

fitter.get_fit_values()

plotter = fitter.plot_fit(True, auto_label=True)
plotter.show()
