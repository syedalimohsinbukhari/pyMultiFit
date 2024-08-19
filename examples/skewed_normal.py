"""Created on Jul 18 14:05:42 2024"""

import numpy as np

from pymultifit.fitters import SkewedNormalFitter
from pymultifit.generators import generate_multi_skewed_normal_data

params = [(12, 2, -7, 1), (3, -4, 2, 2), (4, 2, 7, 1.5)]

x = np.linspace(-15, 15, 10_000)

noise_level = 0.1
y = generate_multi_skewed_normal_data(x, params=params, noise_level=noise_level)

fitter = SkewedNormalFitter(n_fits=3, x_values=x, y_values=y)

guess = [(4, 6, -6, 0.5), (3, -4, 2, 2), (4, 2, 7, 1.5)]

fitter.fit(guess)

plotter = fitter.plot_fit(show_individual=True, auto_label=True)
plotter.show()
