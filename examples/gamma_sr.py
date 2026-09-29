"""Created on Dec 27 11:40:37 2024"""

import numpy as np
from matplotlib import pyplot as plt

from pymultifit import EPSILON
from pymultifit.fitters import GammaFitter
from pymultifit.generators import multi_gamma

params = [(2, 2, 1, 5), (4, 6, 2, 1), (1, 3, 2, 9)]

x = np.linspace(EPSILON, 30, 1000)

noise_level = 0.05
y = multi_gamma(x, params=params, noise_level=noise_level)

fitter = GammaFitter(x, y)

guess = [(2, 2, 1, 3), (3, 5, 1.5, 1), (1, 2, 1, 8)]

fitter.fit(p0=guess)

f, ax = plt.subplots(1, 1, figsize=(12, 6))
fitter.dry_run(axis=ax)

f2, ax = fitter.plotter.plot_fit_and_residuals(show_individuals=True)
f2.tight_layout()
plt.show()
