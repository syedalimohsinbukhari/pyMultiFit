"""Created on Jul 18 01:11:32 2024"""

import numpy as np

from src.pymultifit import Gaussian
from src.pymultifit.backend import generate_multi_gaussian_data

params = [(5, -1, 0.5), (20, -20, 2), (4, -5.5, 10), (10, 3, 1), (4, 15, 3)]

x = np.linspace(-35, 35, 1500)

noise_level = 0.2
y = generate_multi_gaussian_data(x, params, noise_level=noise_level)

mg = Gaussian(5, x, y)

mg_guess = [(5, -1, 0.5), (10, -18, 1), (4, -5.5, 10), (10, 3, 1), (4, 15, 3)]

mg.fit(mg_guess)

plotter = mg.plot_fit(True, auto_label=True)
plotter.show()
