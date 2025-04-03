"""Created on Nov 24 02:45:12 2024"""

import numpy as np
from matplotlib import pyplot as plt

from pymultifit import SKEW_NORMAL, LAPLACE, GAUSSIAN, LINE
from pymultifit.fitters import MixedDataFitter
from pymultifit.generators import multiple_models

x = np.linspace(start=-50, stop=50, num=10_000)
noise_level = 0.1

params = [(-0.1, 5), (20, -20, 2), (4, -5.5, 10, 3), (5, -1, 0.5), (10, 3, 1), (4, 15, 3)]

y = multiple_models(x=x, params=params, model_list=[LINE, GAUSSIAN, SKEW_NORMAL, LAPLACE, LAPLACE, GAUSSIAN],
                    noise_level=noise_level)

guess = [(0, 2), (1, -20, 1), (1, -5, 10, 1), (3, -1, 0.5), (7, 2, 1), (1, 15, 2)]

fitter = MixedDataFitter(x_values=x, y_values=y,
                         model_list=[LINE, GAUSSIAN, SKEW_NORMAL] + [LAPLACE] * 2 + [GAUSSIAN])

fitter.fit(guess)

f, ax = plt.subplots(1, 1, figsize=(8, 6))
plotter = fitter.plot_fit(show_individuals=True, x_label='X_data', y_label='Y_data', data_label='XY_data',
                          title='XY_plot', axis=ax, ci=3)
plt.show()
