"""Created on Jan 22 13:30:00 2026"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import johnsonsu

from pymultifit.distributions import JohnsonSUDistribution

x_values = np.linspace(start=-5, stop=5, num=500)

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

y_multifit = JohnsonSUDistribution(gamma=2, delta=1.5, xi=1, lambda_=2, normalize=True)

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, johnsonsu(a=2, b=1.5, loc=1, scale=2).pdf(x=x_values), label="Scipy Johnson SU")
ax[0].plot(x_values, y_multifit.pdf(x_values), "k:", label="pyMultiFit Johnson SU")
ax[0].set_ylabel("f(x)")

ax[1].plot(x_values, johnsonsu(a=2, b=1.5, loc=1, scale=2).cdf(x=x_values), label="Scipy Johnson SU")
ax[1].plot(x_values, y_multifit.cdf(x_values), "k:", label="pyMultiFit Johnson SU")
ax[1].set_ylabel("F(x)")

f.suptitle(rf"Johnson SU($\gamma={2}$, $\delta={1.5}$, $\xi={1}$, $\lambda={2}$)")

for i in ax:
    i.set_xlabel("X")
    i.legend()
plt.tight_layout()
plt.savefig("./../../images/johnsonSU_example2.png")
plt.show()
