"""Created on Jan 02 16:12:23 2025"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skewnorm

from pymultifit.distributions import SkewNormalDistribution

x_values = np.linspace(start=-10, stop=10, num=500)

y_multifit = SkewNormalDistribution(normalize=True)
y_scipy = skewnorm

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values, a=1), label='Scipy Skew Normal')
ax[0].plot(x_values, y_multifit.pdf(x_values), 'k:', label='pyMultiFit Skew Normal')
ax[0].set_ylabel('f(x)')

ax[1].plot(x_values, y_scipy.cdf(x=x_values, a=1), label='Scipy Skew Normal')
ax[1].plot(x_values, y_multifit.cdf(x_values), 'k:', label='pyMultiFit Skew Normal')
ax[1].set_ylabel('F(x)')

f.suptitle('Skew Normal(1, 0, 1)')

for i in ax:
    i.set_xlabel('X')
    i.legend()
plt.tight_layout()
plt.savefig('./../../images/skew_norm_example1.png')

y_multifit = SkewNormalDistribution(shape=3, scale=3, location=-3, normalize=True)

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values, a=3, loc=-3, scale=3), label='Scipy translated Skew Normal')
ax[0].plot(x_values, y_multifit.pdf(x_values), 'k:', label='pyMultiFit translated Skew Normal')
ax[0].set_ylabel('f(x)')

ax[1].plot(x_values, y_scipy.cdf(x=x_values, a=3, loc=-3, scale=3), label='Scipy translated Skew Normal')
ax[1].plot(x_values, y_multifit.cdf(x_values), 'k:', label='pyMultiFit translated Skew Normal')
ax[1].set_ylabel('F(x)')

f.suptitle(r'Skew Normal(3, -3, 3)')

for i in ax:
    i.set_xlabel('X')
    i.legend()
plt.tight_layout()
plt.savefig('./../../images/skew_norm_example2.png')
plt.show()
