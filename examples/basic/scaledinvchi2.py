"""Created on Aug 20 10:48:38 2026"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import invgamma

from pymultifit.distributions import ScaledInverseChiSquareDistribution

x_values = np.linspace(start=0.01, stop=10, num=500)

df1, loc1, scale1 = 3.0, 0.0, 1.0
a1 = df1 / 2.0
scipy_scale1 = scale1 / 2

y_multifit = ScaledInverseChiSquareDistribution(df=df1, loc=loc1, scale=scale1, normalize=True)
y_scipy = invgamma

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x_values, a=a1, loc=loc1, scale=scipy_scale1), label="Scipy InvGamma")
ax[0].plot(x_values, y_multifit.pdf(x_values), "k:", label="pyMultiFit ScaledInvChi2")
ax[0].set_ylabel("f(x)")

ax[1].plot(x_values, y_scipy.cdf(x_values, a=a1, loc=loc1, scale=scipy_scale1), label="Scipy InvGamma")
ax[1].plot(x_values, y_multifit.cdf(x_values), "k:", label="pyMultiFit ScaledInvChi2")
ax[1].set_ylabel("F(x)")

f.suptitle("ScaledInverseChiSquare(df=3, scale=1.0, loc=0)")

for i in ax:
    i.set_xlabel("X")
    i.legend()
plt.tight_layout()
plt.savefig("./../../images/scaled_inv_chi2_example1.png")

df2, loc2, scale2 = 3.0, 3.0, 1.0
a2 = df2 / 2.0
scipy_scale2 = scale2 / 2.0

y_multifit = ScaledInverseChiSquareDistribution(df=df2, loc=loc2, scale=scale2, normalize=True)

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x_values, a=a2, loc=loc2, scale=scipy_scale2), label="Scipy translated InvGamma")
ax[0].plot(x_values, y_multifit.pdf(x_values), "k:", label="pyMultiFit translated ScaledInvChi2")
ax[0].set_ylabel("f(x)")

ax[1].plot(x_values, y_scipy.cdf(x_values, a=a2, loc=loc2, scale=scipy_scale2), label="Scipy translated InvGamma")
ax[1].plot(x_values, y_multifit.cdf(x_values), "k:", label="pyMultiFit translated ScaledInvChi2")
ax[1].set_ylabel("F(x)")

f.suptitle("ScaledInverseChiSquare(df=3, scale=1.0, loc=3)")

for i in ax:
    i.set_xlabel("X")
    i.legend()
plt.tight_layout()
plt.savefig("./../../images/scaled_inv_chi2_example2.png")
plt.show()
