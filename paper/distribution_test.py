"""Created on Aug 21 14:35:18 2025"""

import numpy as np
import scipy.stats as ss
from matplotlib import pyplot as plt

from pymultifit.distributions import GaussianDistribution

x = np.linspace(-15, 15, 100)

custom_distribution = GaussianDistribution(mu=-1, std=4, normalize=True).pdf(x)

scipy_distribution = ss.norm(loc=-1, scale=4).pdf(x)
custom_distribution_with_scipy = GaussianDistribution.from_scipy_params(loc=-1, scale=4).pdf(x)

f, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(x, custom_distribution, label='custom distribution\nw/o\nscipy parametrization')
ax[0].plot(x, scipy_distribution, label='scipy distribution')
ax[0].set_xlabel('x')
ax[0].set_ylabel('pdf')
ax[0].set_title('Gaussian Distribution')
ax[0].grid(True, alpha=0.5, ls='--')
ax[0].legend(loc='best')

ax[1].plot(x, custom_distribution_with_scipy, label='custom distribution\nw\nscipy parametrization')
ax[1].plot(x, scipy_distribution, label='scipy distribution')
ax[1].set_xlabel('x')
ax[1].set_ylabel('pdf')
ax[1].set_title('Gaussian Distribution')
ax[1].grid(True, alpha=0.5, ls='--')
ax[1].legend(loc='best')

plt.tight_layout()
plt.savefig('./distribution_test.png')

plt.show()
