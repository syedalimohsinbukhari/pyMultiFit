"""Created on Jul 18 14:05:42 2024"""

import matplotlib.pyplot as plt
import numpy as np

from src.pymultifit.fitters import SkewedNormalFitter
from src.pymultifit.generators import generate_multi_skewed_normal_data

params = [(12, 2, -7, 1), (3, -4, 2, 2), (4, 2, 7, 1.5)]

x = np.linspace(-15, 15, 10_000)

noise_level = 1e-2
y = generate_multi_skewed_normal_data(x, params=params, noise_level=noise_level)

fitter = SkewedNormalFitter(n_fits=3, x_values=x, y_values=y)

guess = [(4, 6, -6, 0.5), (3, -4, 2, 2), (4, 2, 7, 1.5)]

fitter.fit(guess)

plotter = fitter.plot_fit(show_individual=True, x_label='X_data', y_label='Y_data', title='XY_plot', data_label='XY_data')
plt.show()
