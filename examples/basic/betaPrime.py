"""Created on Dec 29 11:05:26 2024"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import betaprime

from pymultifit.distributions import BetaPrimeDistribution

x_values = np.linspace(start=0, stop=10, num=500)

y_multifit = BetaPrimeDistribution.from_scipy_params(2, 30)
y_scipy = betaprime(2, 30)

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x_values), label="Scipy BetaPrime")
ax[0].plot(x_values, y_multifit.pdf(x_values), "k:", label="pyMultiFit BetaPrime")
ax[0].set_ylabel("f(x)")

ax[1].plot(x_values, y_scipy.cdf(x_values), label="Scipy BetaPrime")
ax[1].plot(x_values, y_multifit.cdf(x_values), "k:", label="pyMultiFit BetaPrime")
ax[1].set_ylabel("F(x)")

f.suptitle("BetaPrime(2, 30)")

for i in ax:
    i.set_xlabel("X")
    i.legend()
plt.tight_layout()
plt.savefig("./../../images/beta_prime1.png")
plt.close()

y_multifit = BetaPrimeDistribution.from_scipy_params(2, 30, loc=3, scale=5)
y_scipy = betaprime(2, 30, loc=3, scale=5)

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x_values), label="Scipy BetaPrime scaled")
ax[0].plot(x_values, y_multifit.pdf(x_values), "k:", label="pyMultiFit BetaPrime scaled")
ax[0].set_ylabel("f(x)")

ax[1].plot(x_values, y_scipy.cdf(x_values), label="Scipy BetaPrime scaled")
ax[1].plot(x_values, y_multifit.cdf(x_values), "k:", label="pyMultiFit BetaPrime scaled")
ax[1].set_ylabel("F(x)")

f.suptitle("BetaPrime(2, 30, 5, 3)")

for i in ax:
    i.set_xlabel("X")
    i.legend()
plt.tight_layout()
plt.savefig("./../../images/beta_prime2.png")
plt.close()
