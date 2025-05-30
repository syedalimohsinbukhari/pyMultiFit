"""Created on Dec 27 11:40:37 2024"""

import numpy as np
from matplotlib import pyplot as plt

from src.pymultifit.fitters import GammaFitter
from src.pymultifit.generators import multi_gamma
from src.pymultifit import EPSILON

params = [(2, 2, 1, 5), (4, 6, 2, 1), (1, 3, 2, 9)]
x = np.linspace(EPSILON, 15, 1000)

noise_level = 0.05
y = multi_gamma(x, params=params, noise_level=noise_level)

fitter = GammaFitter(x, y)

guess = [(2, 2, 1, 3), (3, 5, 1, 1), (1, 2, 1, 8)]

fitter.fit(p0=guess)

fitter.plot_fit(show_individuals=True, x_label='X_data', y_label='Y_data', title='XY_plot', data_label='XY_data')
plt.show()
