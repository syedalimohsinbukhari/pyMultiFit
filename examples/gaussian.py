"""Created on Jul 18 01:11:32 2024"""

import numpy as np
from matplotlib import pyplot as plt

from src.pymultifit.fitters import GaussianFitter
from src.pymultifit.generators import multi_gaussian

params = [(20, -20, 2), (4, -5.5, 10), (5, -1, 0.5), (10, 3, 1), (4, 15, 3)]

x = np.linspace(-35, 35, 1500)

noise_level = 0.2
y = multi_gaussian(x, params=params, noise_level=noise_level)

fitter = GaussianFitter(n_fits=5, x_values=x, y_values=y)

guess = [(10, -18, 1), (4, -5.5, 10), (5, -1, 0.5), (10, 3, 1), (4, 15, 3)]

fitter.fit(guess)

plotter = fitter.plot_fit(show_individual=True, x_label='X_data', y_label='Y_data', title='XY_plot', data_label='XY_data')
plt.show()
