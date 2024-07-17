"""Created on Jul 18 01:11:32 2024"""

import matplotlib.pyplot as plt
import numpy as np

from src.pymultifit.backend.utilities import generate_multi_gaussian_data
from src.pymultifit.gaussian import MultiGaussian

params = [(5, -1, 0.5), (4, -5.5, 10), (10, 3, 1), (4, 15, 3)]

x = np.linspace(-35, 35, 1500)

noise_level = 0.2
y = generate_multi_gaussian_data(x, params, noise_level=noise_level)

mg = MultiGaussian(4, x, y)

mg_guess = [5, -1, 0.5,
            4, -5.5, 10,
            10, 3, 1,
            4, 15, 3]

mg.fit(mg_guess)

f, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

plotter = mg.plot_fit(True, auto_label=True, ax=ax[0])
