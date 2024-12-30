"""Created on Dec 29 10:23:45 2024"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import arcsine

from pymultifit.distributions import ArcSineDistribution

x_values = np.linspace(start=0, stop=1, num=500)

y_multifit = ArcSineDistribution(normalize=True)
y_scipy = arcsine

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x_values), label='Scipy ArcSine')
ax[0].plot(x_values, y_multifit.pdf(x_values), 'k:', label='pyMultiFit ArcSine')
ax[0].set_ylabel('f(x)')

ax[1].plot(x_values, y_scipy.cdf(x_values), label='Scipy ArcSine')
ax[1].plot(x_values, y_multifit.cdf(x_values), 'k:', label='pyMultiFit ArcSine')
ax[1].set_ylabel('F(x)')

for i in ax:
    i.set_xlabel('X')
    i.legend()
plt.tight_layout()
