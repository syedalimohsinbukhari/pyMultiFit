"""Created on Jul 19 23:15:21 2024"""

import matplotlib.pyplot as plt
import numpy as np

from pymultifit import EPSILON
from pymultifit.fitters import LogNormalFitter
from pymultifit.generators import multi_log_normal

params = [(15, 1, 1), (3, 2, 0.2), (20, 4, 0.1)]

x = np.linspace(EPSILON, 10, 2000)

noise_level = 0.2
y = multi_log_normal(x, params=params, noise_level=noise_level)

fitter = LogNormalFitter(x_values=x, y_values=y)

guess = [(12, np.log(1), 1), (3, np.log(2), 0.2), (15, np.log(4), 0.1)]

fitter.fit(guess)

f, ax = plt.subplots(1, 1, figsize=(12, 6))
plotter = fitter.plot_fit(
    show_individuals=True, x_label="X_data", y_label="Y_data", title="XY_plot", data_label="XY_data", axis=ax
)
plt.show()
