"""Created on Jul 19 23:15:21 2024"""

import numpy as np
from pymultifit import LogNormal
from pymultifit.backend import generate_multi_log_normal_data

params = [(5, 3, 1), (6, 30, 0.5)]

x = np.linspace(0.001, 100, 2000)

noise_level = 0.2
y = generate_multi_log_normal_data(x, params, noise_level=noise_level, exact_mean=True)

fitter = LogNormal.from_exact_mean(2, x, y, 5000)

guess = [(5, 3, 0.5), (6, 30, 0.5)]

fitter.fit(guess)

plotter = fitter.plot_fit(True, auto_label=True)
plotter.show()
