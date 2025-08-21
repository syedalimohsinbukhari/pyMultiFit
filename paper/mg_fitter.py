"""Created on Aug 21 14:32:17 2025"""

import matplotlib.pyplot as plt
import numpy as np

from pymultifit.fitters import GaussianFitter
from pymultifit.generators import multi_gaussian

# x-data
x = np.linspace(-10, 10, 10_000)

# parameters for the 3-component gaussian model
amp = np.array([4, 2, 6])
mu = np.array([-3, 0, 6])
std = np.array([1, 1, 0.3])

# stacking the parameters into a single array
params = np.column_stack([amp, mu, std])

# data generation
mg_data = multi_gaussian(x, params, noise_level=0.2, normalize=False)

# guess for the parameters
# note that the order of the parameters must match the order of the parameters in the distribution class
amp_guess = np.array([3, 1, 4])
mu_guess = np.array([-2, 0, 5])
std_guess = np.array([1, 0.5, 0.5])

# stacking the parameters into a single array
params_guess = np.column_stack([amp_guess, mu_guess, std_guess])

# initializing fitter with the data
mg_fitter = GaussianFitter(x, mg_data)

# fitting the model
mg_fitter.fit(params_guess)

# plotting the fitted model
mg_fitter.plot_fit(show_individuals=True)
plt.savefig('./mg_fit_paper.png', dpi=300)
plt.show()
