"""Created on Dec 30 10:21:37 2024"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import halfnorm

from pymultifit.distributions import HalfNormalDistribution

x_values = np.linspace(start=-10, stop=10, num=500)

y_multifit = HalfNormalDistribution(normalize=True)
y_scipy = halfnorm

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values), label='Scipy Half Normal')
ax[0].plot(x_values, y_multifit.pdf(x_values), 'k:', label='pyMultiFit Half Normal')
ax[0].set_ylabel('f(x)')

ax[1].plot(x_values, y_scipy.cdf(x=x_values), label='Scipy Half Normal')
ax[1].plot(x_values, y_multifit.cdf(x_values), 'k:', label='pyMultiFit Half Normal')
ax[1].set_ylabel('F(x)')

f.suptitle('Half Normal(1)')

for i in ax:
    i.set_xlabel('X')
    i.legend()
plt.tight_layout()
plt.savefig('./../../images/half_normal_example1.png')

y_multifit = HalfNormalDistribution(loc=3, scale=3, normalize=True)

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values, scale=3, loc=3), label='Scipy translated Half Normal')
ax[0].plot(x_values, y_multifit.pdf(x_values), 'k:', label='pyMultiFit translated Half Normal')
ax[0].set_ylabel('f(x)')

ax[1].plot(x_values, y_scipy.cdf(x=x_values, scale=3, loc=3), label='Scipy translated Half Normal')
ax[1].plot(x_values, y_multifit.cdf(x_values), 'k:', label='pyMultiFit translated Half Normal')
ax[1].set_ylabel('F(x)')

f.suptitle(r'Half Normal(3, 3)')

for i in ax:
    i.set_xlabel('X')
    i.legend()
plt.tight_layout()
plt.savefig('./../../images/half_normal_example2.png')
plt.show()
