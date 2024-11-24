"""Created on Jul 19 23:15:21 2024"""

import numpy as np

from src.pymultifit.fitters import LogNormalFitter
from src.pymultifit.generators import generate_multi_log_normal_data

# taken from https://stackoverflow.com/a/19141711
EPSILON = np.finfo(float).eps  # 2.220446049250313e-16

params = [(15, 1, 1), (3, 2, 0.2), (20, 4, 0.1)]

x = np.linspace(EPSILON, 100, 2000)

noise_level = 0.2
y = generate_multi_log_normal_data(x, params=params, noise_level=noise_level)

fitter = LogNormalFitter(n_fits=3, x_values=x, y_values=y)

guess = [(10, 1, 1), (3, 2, 0.2), (10, 4, 0.1)]

fitter.fit(guess)

plotter = fitter.plot_fit(show_individual=True, x_label='X_data', y_label='Y_data', title='XY_plot', data_label='XY_data')
plotter.show()
