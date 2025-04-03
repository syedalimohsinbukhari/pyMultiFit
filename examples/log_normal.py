"""Created on Jul 19 23:15:21 2024"""

import matplotlib.pyplot as plt
import numpy as np

from src.pymultifit import EPSILON
from src.pymultifit.fitters import LogNormalFitter
from src.pymultifit.generators import multi_log_normal

params = [(15, 1, 1), (3, 2, 0.2), (20, 4, 0.1)]

x = np.linspace(EPSILON, 10, 2000)

noise_level = 0.2
y = multi_log_normal(x, params=params, noise_level=noise_level)

fitter = LogNormalFitter(x_values=x, y_values=y)

guess = [(12, np.log(1), 1), (3, np.log(2), 0.2), (15, np.log(4), 0.1)]

fitter.fit(guess)

plotter = fitter.plot_fit(x_label='X_data', y_label='Y_data', data_label='XY_data', title='XY_plot')
plt.show()
