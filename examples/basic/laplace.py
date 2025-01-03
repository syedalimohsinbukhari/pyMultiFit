"""Created on Jan 02 14:25:43 2025"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import laplace

from pymultifit.distributions import LaplaceDistribution

x_values = np.linspace(start=-10, stop=10, num=500)

y_multifit = LaplaceDistribution(normalize=True)
y_scipy = laplace

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values), label='Scipy Laplace')
ax[0].plot(x_values, y_multifit.pdf(x_values), 'k:', label='pyMultiFit Laplace')
ax[0].set_ylabel('f(x)')

ax[1].plot(x_values, y_scipy.cdf(x=x_values), label='Scipy Laplace')
ax[1].plot(x_values, y_multifit.cdf(x_values), 'k:', label='pyMultiFit Laplace')
ax[1].set_ylabel('F(x)')

f.suptitle('Laplace(0, 1)')

for i in ax:
    i.set_xlabel('X')
    i.legend()
plt.tight_layout()
plt.savefig('./../../images/laplace_example1.png')

y_multifit = LaplaceDistribution(mean=3, diversity=2, normalize=True)

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values, loc=3, scale=2), label='Scipy translated Laplace')
ax[0].plot(x_values, y_multifit.pdf(x_values), 'k:', label='pyMultiFit translated Laplace')
ax[0].set_ylabel('f(x)')

ax[1].plot(x_values, y_scipy.cdf(x=x_values, loc=3, scale=2), label='Scipy translated Laplace')
ax[1].plot(x_values, y_multifit.cdf(x_values), 'k:', label='pyMultiFit translated Laplace')
ax[1].set_ylabel('F(x)')

f.suptitle(r'Folded Normal(3, 2)')

for i in ax:
    i.set_xlabel('X')
    i.legend()
plt.tight_layout()
plt.savefig('./../../images/laplace_example2.png')
plt.show()
