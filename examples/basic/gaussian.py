"""Created on Jan 02 13:39:29 2025"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from pymultifit.distributions import GaussianDistribution

x_values = np.linspace(start=-10, stop=10, num=500)

y_multifit = GaussianDistribution(normalize=True)
y_scipy = norm

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values), label="Scipy Gaussian")
ax[0].plot(x_values, y_multifit.pdf(x_values), "k:", label="pyMultiFit Gaussian")
ax[0].set_ylabel("f(x)")

ax[1].plot(x_values, y_scipy.cdf(x=x_values), label="Scipy Gaussian")
ax[1].plot(x_values, y_multifit.cdf(x_values), "k:", label="pyMultiFit Gaussian")
ax[1].set_ylabel("F(x)")

f.suptitle("Gaussian(0, 1)")

for i in ax:
    i.set_xlabel("X")
    i.legend()
plt.tight_layout()
plt.savefig("./../../images/gaussian_example1.png")

y_multifit = GaussianDistribution(mu=3, std=2, normalize=True)

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values, loc=3, scale=2), label="Scipy translated Gaussian")
ax[0].plot(x_values, y_multifit.pdf(x_values), "k:", label="pyMultiFit translated Gaussian")
ax[0].set_ylabel("f(x)")

ax[1].plot(x_values, y_scipy.cdf(x=x_values, loc=3, scale=2), label="Scipy translated Gaussian")
ax[1].plot(x_values, y_multifit.cdf(x_values), "k:", label="pyMultiFit translated Gaussian")
ax[1].set_ylabel("F(x)")

f.suptitle(r"Gaussian(3, 2)")

for i in ax:
    i.set_xlabel("X")
    i.legend()
plt.tight_layout()
plt.savefig("./../../images/gaussian_example2.png")
plt.show()
