"""Created on Jul 18 14:05:42 2024"""

import matplotlib.pyplot as plt
import numpy as np

from pymultifit.fitters import SkewNormalFitter
from pymultifit.generators import multi_skew_normal

params = [(3, -10, -10, 3), (3, 5, -2, 8), (4, 2, -7, 2)]

x = np.linspace(-30, 25, 10_000)

noise_level = 0.1
y = multi_skew_normal(x, params=params, noise_level=noise_level)

fitter = SkewNormalFitter(x_values=x, y_values=y)

guess = [(2, -5, -8, 1), (3, 4, -2, 5), (4, 1, -7, 1.5)]

fitter.fit(guess)
f, ax = plt.subplots(1, 1, figsize=(12, 6))

plotter = fitter.plot_fit(x_label='X_data', y_label='Y_data', data_label='XY_data', title='XY_plot', axis=ax)
plt.show()
