"""Created on Jan 31 16:00:47 2025"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gennorm

from pymultifit.distributions import SymmetricGeneralizedNormalDistribution

x_values = np.linspace(start=-10, stop=10, num=500)

y_multifit = SymmetricGeneralizedNormalDistribution(normalize=True)
y_scipy = gennorm

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values, beta=1), label='Scipy GenNorm')
ax[0].plot(x_values, y_multifit.pdf(x_values), 'k:', label='pyMultiFit GenNorm')
ax[0].set_ylabel('f(x)')

ax[1].plot(x_values, y_scipy.cdf(x=x_values, beta=1), label='Scipy GenNorm')
ax[1].plot(x_values, y_multifit.cdf(x_values), 'k:', label='pyMultiFit GenNorm')
ax[1].set_ylabel('F(x)')

f.suptitle('GenNorm(1, 0, 1)')

for i in ax:
    i.set_xlabel('X')
    i.legend()
plt.tight_layout()
plt.savefig('./../../images/gen_norm_example1.png')

y_multifit = SymmetricGeneralizedNormalDistribution(shape=2, loc=-3, scale=5, normalize=True)

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values, beta=2, loc=-3, scale=5), label='Scipy translated GenNorm')
ax[0].plot(x_values, y_multifit.pdf(x_values), 'k:', label='pyMultiFit translated GenNorm')
ax[0].set_ylabel('f(x)')

ax[1].plot(x_values, y_scipy.cdf(x=x_values, beta=2, loc=-3, scale=5), label='Scipy translated GenNorm')
ax[1].plot(x_values, y_multifit.cdf(x_values), 'k:', label='pyMultiFit translated GenNorm')
ax[1].set_ylabel('F(x)')

f.suptitle(r'GenNorm(2, -3, 5)')

for i in ax:
    i.set_xlabel('X')
    i.legend()
plt.tight_layout()
plt.savefig('./../../images/gen_norm_example2.png')
plt.show()
