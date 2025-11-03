"""Created on Dec 30 10:21:37 2024"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import foldnorm

from pymultifit.distributions import FoldedNormalDistribution

x_values = np.linspace(start=-10, stop=10, num=500)

y_multifit = FoldedNormalDistribution(normalize=True)
y_scipy = foldnorm

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values, c=0), label="Scipy Folded Normal")
ax[0].plot(x_values, y_multifit.pdf(x_values), "k:", label="pyMultiFit Folded Normal")
ax[0].set_ylabel("f(x)")

ax[1].plot(x_values, y_scipy.cdf(x=x_values, c=0), label="Scipy Folded Normal")
ax[1].plot(x_values, y_multifit.cdf(x_values), "k:", label="pyMultiFit Folded Normal")
ax[1].set_ylabel("F(x)")

f.suptitle("Folded Normal(0, 1)")

for i in ax:
    i.set_xlabel("X")
    i.legend()
plt.tight_layout()
plt.savefig("./../../images/folded_normal_example1.png")

y_multifit = FoldedNormalDistribution(mu=2, sigma=3, loc=3, normalize=True)

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values, c=2, scale=3, loc=3), label="Scipy translated Folded Normal")
ax[0].plot(x_values, y_multifit.pdf(x_values), "k:", label="pyMultiFit translated Folded Normal")
ax[0].set_ylabel("f(x)")

ax[1].plot(x_values, y_scipy.cdf(x=x_values, c=2, scale=3, loc=3), label="Scipy translated Folded Normal")
ax[1].plot(x_values, y_multifit.cdf(x_values), "k:", label="pyMultiFit translated Folded Normal")
ax[1].set_ylabel("F(x)")

f.suptitle(r"Folded Normal(2, 3, 3)")

for i in ax:
    i.set_xlabel("X")
    i.legend()
plt.tight_layout()
plt.savefig("./../../images/folded_normal_example2.png")
# plt.show()
