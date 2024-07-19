"""Created on Jul 18 14:05:42 2024"""

import numpy as np

from src.pymultifit import SkewedNormal
from src.pymultifit.backend import generate_multi_skewed_normal_data

params = [(12, 2, -7, 1), (3, -4, 2, 2), (4, 2, 7, 1.5)]

x = np.linspace(-15, 15, 500)

noise_level = 0.2
y = generate_multi_skewed_normal_data(x, params, noise_level=noise_level)

skn = SkewedNormal(n_fits=3, x_values=x, y_values=y)

skn_guess = [(4, 6, -6, 0.5), (3, -4, 2, 2), (4, 2, 1, 1.5)]

skn.fit(skn_guess)

plotter = skn.plot_fit(show_individual=True, auto_label=True)
plotter.show()
