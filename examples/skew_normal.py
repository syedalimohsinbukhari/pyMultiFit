"""Created on Jul 18 14:05:42 2024"""

import matplotlib.pyplot as plt
import numpy as np

from pymultifit.fitters import SkewNormalFitter
from pymultifit.generators import multi_skew_normal

params = [(3, 4, -7, 3), (3, -5, -2, 8), (4, 2, 7, 2)]

x = np.linspace(-30, 15, 10_000)

noise_level = 0.1
y = multi_skew_normal(x, params=params, noise_level=noise_level)

fitter = SkewNormalFitter(x_values=x, y_values=y)

guess = [(2, 1, -6, 0.5), (3, -2, -2, 5), (4, 1, 7, 1.5)]

fitter.fit(guess)
plotter = fitter.plot_fit(show_individual=True, x_label='X_data', y_label='Y_data', title='XY_plot', data_label='XY_data')
plt.show()
