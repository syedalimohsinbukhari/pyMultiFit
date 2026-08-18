"""Created on Aug 12 11:26:00 2026"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon

from pymultifit.distributions.generalized import QExponentialDistribution

x_values = np.linspace(start=0, stop=10, num=500)

y_multifit = QExponentialDistribution(normalize=True)
y_scipy = expon

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values), label="Scipy Exponential")
ax[0].plot(x_values, y_multifit.pdf(x_values), "k:", label="pyMultiFit QExponential")
ax[0].set_ylabel("f(x)")

ax[1].plot(x_values, y_scipy.cdf(x=x_values), label="Scipy Exponential")
ax[1].plot(x_values, y_multifit.cdf(x_values), "k:", label="pyMultiFit QExponential")
ax[1].set_ylabel("F(x)")

f.suptitle("QExponential(1, 1)")

for i in ax:
    i.set_xlabel("X")
    i.legend()
plt.tight_layout()
plt.savefig("./../../images/q_exponential_example1.png")

y_multifit = QExponentialDistribution(q=1.5, rate=0.5, loc=3, normalize=True)

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values, loc=3, scale=2), label="Scipy translated Exponential")
ax[0].plot(x_values, y_multifit.pdf(x_values), "k:", label="pyMultiFit translated QExponential")
ax[0].set_ylabel("f(x)")

ax[1].plot(x_values, y_scipy.cdf(x=x_values, loc=3, scale=2), label="Scipy translated Exponential")
ax[1].plot(x_values, y_multifit.cdf(x_values), "k:", label="pyMultiFit translated QExponential")
ax[1].set_ylabel("F(x)")

f.suptitle(r"QExponential(q=1.5, rate=0.5, loc=3)")

for i in ax:
    i.set_xlabel("X")
    i.legend()
plt.tight_layout()
plt.savefig("./../../images/q_exponential_example2.png")
plt.show()