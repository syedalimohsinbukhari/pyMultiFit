"""Created on Dec 30 10:21:37 2024"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon

from pymultifit.distributions import ExponentialDistribution

x_values = np.linspace(start=0, stop=5, num=500)

y_multifit = ExponentialDistribution(scale=1.5, normalize=True)
y_scipy = expon

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values, scale=1 / 1.5), label='Scipy Exponential')
ax[0].plot(x_values, y_multifit.pdf(x_values), 'k:', label='pyMultiFit Exponential')
ax[0].set_ylabel('f(x)')

ax[1].plot(x_values, y_scipy.cdf(x=x_values, scale=1 / 1.5), label='Scipy Exponential')
ax[1].plot(x_values, y_multifit.cdf(x_values), 'k:', label='pyMultiFit Exponential')
ax[1].set_ylabel('F(x)')

f.suptitle('Exponential(1.5)')

for i in ax:
    i.set_xlabel('X')
    i.legend()
plt.tight_layout()
# plt.savefig('./../../images/expon_example1.png')

y_multifit = ExponentialDistribution(scale=1.5, loc=3, normalize=True)

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values, scale=1 / 1.5, loc=3), label='Scipy translated Exponential')
ax[0].plot(x_values, y_multifit.pdf(x_values), 'k:', label='pyMultiFit translated Exponential')
ax[0].set_ylabel('f(x)')

ax[1].plot(x_values, y_scipy.cdf(x=x_values, scale=1 / 1.5, loc=3), label='Scipy translated Exponential')
ax[1].plot(x_values, y_multifit.cdf(x_values), 'k:', label='pyMultiFit translated Exponential')
ax[1].set_ylabel('F(x)')

f.suptitle(r'Exponential(1.5, 3)')

for i in ax:
    i.set_xlabel('X')
    i.legend()
plt.tight_layout()
# plt.savefig('./../../images/expon_example2.png')
