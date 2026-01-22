"""Created on Jan 22 13:30:00 2026"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import johnsonsu

from pymultifit.distributions import JohnsonSUDistribution

x_values = np.linspace(start=-5, stop=5, num=500)

# Example 1: Standard parameters
y_multifit = JohnsonSUDistribution(normalize=True)
y_scipy = johnsonsu(a=0, b=1)

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values), label="Scipy Johnson SU")
ax[0].plot(x_values, y_multifit.pdf(x_values), "k:", label="pyMultiFit Johnson SU")
ax[0].set_ylabel("f(x)")

ax[1].plot(x_values, y_scipy.cdf(x=x_values), label="Scipy Johnson SU")
ax[1].plot(x_values, y_multifit.cdf(x_values), "k:", label="pyMultiFit Johnson SU")
ax[1].set_ylabel("F(x)")

f.suptitle(r"Johnson SU($\gamma=0$, $\delta=1$, $\xi=0$, $\lambda=1$)")

for i in ax:
    i.set_xlabel("X")
    i.legend()
plt.tight_layout()
plt.savefig("./../../images/johnsonSU_example1.png")

# Example 2: Custom parameters
gamma, delta = 2.0, 1.5
xi, lambda_ = 1.0, 2.0

y_multifit = JohnsonSUDistribution(gamma=gamma, delta=delta, xi=xi, lambda_=lambda_, normalize=True)
y_scipy = johnsonsu(a=gamma, b=delta, loc=xi, scale=lambda_)

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values), label="Scipy Johnson SU")
ax[0].plot(x_values, y_multifit.pdf(x_values), "k:", label="pyMultiFit Johnson SU")
ax[0].set_ylabel("f(x)")

ax[1].plot(x_values, y_scipy.cdf(x=x_values), label="Scipy Johnson SU")
ax[1].plot(x_values, y_multifit.cdf(x_values), "k:", label="pyMultiFit Johnson SU")
ax[1].set_ylabel("F(x)")

f.suptitle(rf"Johnson SU($\gamma={gamma}$, $\delta={delta}$, $\xi={xi}$, $\lambda={lambda_}$)")

for i in ax:
    i.set_xlabel("X")
    i.legend()
plt.tight_layout()
plt.savefig("./../../images/johnsonSU_example2.png")
plt.show()
