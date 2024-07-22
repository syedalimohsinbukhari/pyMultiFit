"""Created on Jul 19 23:15:21 2024"""

import numpy as np

from src.pymultifit import LogNormal
from src.pymultifit.backend.utilities import generate_multi_log_normal_data

params = [(5, 1, 1), (3, 30, 0.5), (2, 50, 0.2)]

x = np.linspace(0.001, 100, 2000)

noise_level = 0.2
y = generate_multi_log_normal_data(x, params, noise_level=noise_level, exact_mean=True)

fitter = LogNormal.from_exact_mean(3, x, y)

guess = [(5, 1, 1), (3, 20, 0.5), (2, 50, 1)]

fitter.fit(guess)

plotter = fitter.plot_fit(True, auto_label=True)
plotter.show()
