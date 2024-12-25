"""Created on Jul 18 14:05:42 2024"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skewnorm

from pymultifit.fitters import SkewNormalFitter
from pymultifit.generators import multi_skewed_normal

params = [(2, 1, -7, 1), (3, 3, -2, 1), (4, 2, 7, 1)]

x = np.linspace(-15, 15, 10_000)

y = multi_skewed_normal(x, params=params)

y2 = np.zeros_like(x, dtype=float)
for par in params:
    y2 += par[0] * skewnorm.pdf(x, a=par[1], loc=par[2], scale=par[3]) * (np.sqrt(2 * np.pi * par[3]**2) / (2 * par[3]))

fitter = SkewNormalFitter(n_fits=3, x_values=x, y_values=y)

guess = [(2, 1, -6, 0.5), (3, 1, 2, 2), (4, 1, 7, 1.5)]

fitter.fit(guess)
f, ax = plt.subplots(1, 1, figsize=(8, 4))
plotter = fitter.plot_fit(show_individual=True, x_label='X_data', y_label='Y_data', title='XY_plot', data_label='XY_data', axis=ax)
ax.plot(x, y2, 'k--')
plt.show()
