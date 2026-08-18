"""Created on Aug 12 11:26:00 2026"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon

from pymultifit.distributions.generalized import QExponentialDistribution

x_values1 = np.linspace(start=0, stop=10, num=500)

y_multifit1 = QExponentialDistribution(q=1.0, rate=1.0, loc=0.0, normalize=True)
y_scipy1 = expon

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values1, y_scipy1.pdf(x=x_values1), label="Scipy Exponential")
ax[0].plot(x_values1, y_multifit1.pdf(x_values1), "k:", label="pyMultiFit QExponential")
ax[0].set_ylabel("f(x)")

ax[1].plot(x_values1, y_scipy1.cdf(x=x_values1), label="Scipy Exponential")
ax[1].plot(x_values1, y_multifit1.cdf(x_values1), "k:", label="pyMultiFit QExponential")
ax[1].set_ylabel("F(x)")

f.suptitle("QExponential(1.0, 1.0, 0.0)")

for i in ax:
    i.set_xlabel("X")
    i.legend()
plt.tight_layout()
plt.savefig("./../../images/q_exponential_example1.png")

x_values2 = np.linspace(start=3, stop=13, num=500)

y_multifit2 = QExponentialDistribution(q=1.0, rate=0.5, loc=3.0, normalize=True)

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values2, expon.pdf(x=x_values2, loc=3.0, scale=2.0), label="Scipy translated Exponential")
ax[0].plot(x_values2, y_multifit2.pdf(x_values2), "k:", label="pyMultiFit translated QExponential")
ax[0].set_ylabel("f(x)")

ax[1].plot(x_values2, expon.cdf(x=x_values2, loc=3.0, scale=2.0), label="Scipy translated Exponential")
ax[1].plot(x_values2, y_multifit2.cdf(x_values2), "k:", label="pyMultiFit translated QExponential")
ax[1].set_ylabel("F(x)")

f.suptitle(r"QExponential(1.0, 0.5, 3.0)")

for i in ax:
    i.set_xlabel("X")
    i.legend()
plt.tight_layout()
plt.savefig("./../../images/q_exponential_example2.png")
plt.show()