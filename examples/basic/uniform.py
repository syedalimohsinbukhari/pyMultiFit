"""Created on Jan 03 10:38:39 2025"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform

from pymultifit.distributions import UniformDistribution

x_values = np.linspace(start=-10, stop=10, num=500)

y_multifit = UniformDistribution(normalize=True)
y_scipy = uniform

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values), label='Scipy Uniform')
ax[0].plot(x_values, y_multifit.pdf(x_values), 'k:', label='pyMultiFit Uniform')
ax[0].set_ylabel('f(x)')

ax[1].plot(x_values, y_scipy.cdf(x=x_values), label='Scipy Uniform')
ax[1].plot(x_values, y_multifit.cdf(x_values), 'k:', label='pyMultiFit Uniform')
ax[1].set_ylabel('F(x)')

f.suptitle('Uniform(0, 1)')

for i in ax:
    i.set_xlabel('X')
    i.legend()
plt.tight_layout()
plt.savefig('./../../images/uniform_example1.png')

y_multifit = UniformDistribution(low=3, high=5, normalize=True)

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values, loc=3, scale=5), label='Scipy translated Uniform')
ax[0].plot(x_values, y_multifit.pdf(x_values), 'k:', label='pyMultiFit translated Uniform')
ax[0].set_ylabel('f(x)')

ax[1].plot(x_values, y_scipy.cdf(x=x_values, loc=3, scale=5), label='Scipy translated Uniform')
ax[1].plot(x_values, y_multifit.cdf(x_values), 'k:', label='pyMultiFit translated Uniform')
ax[1].set_ylabel('F(x)')

f.suptitle(r'Uniform(3, 5)')

for i in ax:
    i.set_xlabel('X')
    i.legend()
plt.tight_layout()
plt.savefig('./../../images/uniform_example2.png')
plt.show()
