"""Created on Jan 02 13:05:00 2025"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma

from pymultifit.distributions import GammaDistributionSR

x_values = np.linspace(start=0, stop=5, num=500)

y_multifit = GammaDistributionSR(shape=1.5, normalize=True)
y_scipy = gamma

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values, a=1.5), label='Scipy Gamma SR')
ax[0].plot(x_values, y_multifit.pdf(x_values), 'k:', label='pyMultiFit Gamma SR')
ax[0].set_ylabel('f(x)')

ax[1].plot(x_values, y_scipy.cdf(x=x_values, a=1.5), label='Scipy Gamma SR')
ax[1].plot(x_values, y_multifit.cdf(x_values), 'k:', label='pyMultiFit Gamma SR')
ax[1].set_ylabel('F(x)')

f.suptitle('GammaSR(1.5)')

for i in ax:
    i.set_xlabel('X')
    i.legend()
plt.tight_layout()
# plt.savefig('./../../images/gammaSR_example1.png')

y_multifit = GammaDistributionSR(shape=1.5, rate=0.2, loc=3, normalize=True)

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values, a=1.5, scale=1 / 0.2, loc=3), label='Scipy translated Gamma SR')
ax[0].plot(x_values, y_multifit.pdf(x_values), 'k:', label='pyMultiFit translated Gamma SR')
ax[0].set_ylabel('f(x)')

ax[1].plot(x_values, y_scipy.cdf(x=x_values, a=1.5, scale=1 / 0.2, loc=3), label='Scipy translated Gamma SR')
ax[1].plot(x_values, y_multifit.cdf(x_values), 'k:', label='pyMultiFit translated Gamma SR')
ax[1].set_ylabel('F(x)')

f.suptitle(r'Gamma SR(1.5, 0.2, 3)')

for i in ax:
    i.set_xlabel('X')
    i.legend()
plt.tight_layout()
# plt.savefig('./../../images/gammaSR_example2.png')