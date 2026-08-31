"""Created on Dec 29 10:26:55 2024"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

from pymultifit.distributions import BetaDistribution

x_values = np.linspace(start=0, stop=1, num=500)

y_multifit = BetaDistribution(alpha=2, beta=30, normalize=True)
y_scipy = beta

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values, a=2, b=30), label="Scipy Beta")
ax[0].plot(x_values, y_multifit.pdf(x_values), "k:", label="pyMultiFit Beta")
ax[0].set_ylabel("f(x)")

ax[1].plot(x_values, y_scipy.cdf(x=x_values, a=2, b=30), label="Scipy Beta")
ax[1].plot(x_values, y_multifit.cdf(x_values), "k:", label="pyMultiFit Beta")
ax[1].set_ylabel("F(x)")

f.suptitle("Beta(2, 30)")

for i in ax:
    i.set_xlabel("X")
    i.legend()
plt.tight_layout()
