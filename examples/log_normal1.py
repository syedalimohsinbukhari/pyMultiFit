"""Created on Jul 19 23:15:21 2024"""

import numpy as np

from src.pymultifit import LogNormal
from src.pymultifit.backend.utilities import generate_multi_log_normal_data

params = [(5, 1, 1), (3, 2, 0.2), (2, 4, 0.2)]

x = np.linspace(0.001, 100, 2000)

noise_level = 0.2
y = generate_multi_log_normal_data(x, params, noise_level=noise_level)

lNorm = LogNormal(3, x, y)

ln_guess = [(5, 1, 1), (3, 2, 0.2), (2, 4, 0.2)]

lNorm.fit(ln_guess)

plotter = lNorm.plot_fit(True, auto_label=True)
plotter.show()
