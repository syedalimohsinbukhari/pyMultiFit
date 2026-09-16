"""Created on Aug 20 10:48:38 2026"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import invgamma

from pymultifit.distributions import ScaledInverseChiSquareDistribution

x_values = np.linspace(start=0.01, stop=10, num=500)

y_multifit = ScaledInverseChiSquareDistribution(df=3, normalize=True)
y_scipy = invgamma

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x_values, a=3 / 2.0, scale=1 / 2.0), label="Scipy InvGamma")
ax[0].plot(x_values, y_multifit.pdf(x_values), "k:", label="pyMultiFit ScaledInvChi2")
ax[0].set_ylabel("f(x)")

ax[1].plot(x_values, y_scipy.cdf(x_values, a=3 / 2.0, scale=1 / 2.0), label="Scipy InvGamma")
ax[1].plot(x_values, y_multifit.cdf(x_values), "k:", label="pyMultiFit ScaledInvChi2")
ax[1].set_ylabel("F(x)")

f.suptitle(r"ScaledInverseChiSquare($\nu$=3)")

for i in ax:
    i.set_xlabel("X")
    i.legend()
plt.tight_layout()
plt.savefig("./../../images/scaled_inv_chi2_example1.png")

y_multifit = ScaledInverseChiSquareDistribution(df=3, loc=3, scale=2, normalize=True)

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x_values, a=3 / 2.0, loc=3, scale=1), label="Scipy translated InvGamma")
ax[0].plot(x_values, y_multifit.pdf(x_values), "k:", label="pyMultiFit translated ScaledInvChi2")
ax[0].set_ylabel("f(x)")

ax[1].plot(x_values, y_scipy.cdf(x_values, a=3 / 2.0, loc=3, scale=1), label="Scipy translated InvGamma")
ax[1].plot(x_values, y_multifit.cdf(x_values), "k:", label="pyMultiFit translated ScaledInvChi2")
ax[1].set_ylabel("F(x)")

f.suptitle(r"ScaledInverseChiSquare($\nu$=3, $\mu$=3, $s^2$=2)")

for i in ax:
    i.set_xlabel("X")
    i.legend()
plt.tight_layout()
plt.savefig("./../../images/scaled_inv_chi2_example2.png")
plt.show()
