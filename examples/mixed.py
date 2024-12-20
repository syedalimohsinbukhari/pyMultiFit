"""Created on Nov 24 02:45:12 2024"""

import numpy as np
from matplotlib import pyplot as plt

from src.pymultifit.fitters import MixedDataFitter
from src.pymultifit.generators import multiple_models

x = np.linspace(start=-50, stop=50, num=10_000)
noise_level = 0.1

params = [(-0.1, 5), (20, -20, 2), (4, -5.5, 10), (5, -1, 0.5), (10, 3, 1), (4, 15, 3)]

y = multiple_models(x=x, params=params, model_list=['line', 'gaussian', 'gaussian', 'laplace', 'laplace', 'gaussian'], noise_level=noise_level)

fitter = MixedDataFitter(x_values=x, y_values=y, model_list=['line'] + ['gaussian'] * 2 + ['laplace'] * 2 + ['gaussian'])
guess = [(0, 0), (1, -20, 1), (1, -5, 5), (1, -1, 0.5), (1, 2, 1), (1, 15, 2)]

fitter.fit(guess)

plotter = fitter.plot_fit(show_individuals=True, x_label='X_data', y_label='Y_data', title='XY_plot', data_label='XY_data')
plt.show()
