"""Created on Jul 19 23:15:21 2024"""

import numpy as np

from pymultifit.fitters import LogNormalFitter
from pymultifit.fitters.multi_generators import generate_multi_log_normal_data

params = [(15, 1, 1), (3, 2, 0.2), (20, 4, 0.2)]

x = np.linspace(0.001, 100, 2000)

noise_level = 0.1
y = generate_multi_log_normal_data(x, params, noise_level=noise_level)

fitter = LogNormalFitter(3, x, y)

guess = [(5, 1, 1), (3, 2, 0.2), (2, 4, 0.2)]

fitter.fit(guess)

plotter = fitter.plot_fit(True, auto_label=True)
plotter.show()
