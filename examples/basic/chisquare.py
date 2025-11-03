"""Created on Dec 30 09:51:12 2024"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from pymultifit.distributions import ChiSquareDistribution

x_values = np.linspace(start=0, stop=5, num=500)

y_multifit = ChiSquareDistribution(degree_of_freedom=1, normalize=True)
y_scipy = chi2

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values, df=1), label="Scipy Chi2")
ax[0].plot(x_values, y_multifit.pdf(x_values), "k:", label="pyMultiFit Chi2")
ax[0].set_ylabel("f(x)")

ax[1].plot(x_values, y_scipy.cdf(x=x_values, df=1), label="Scipy Chi2")
ax[1].plot(x_values, y_multifit.cdf(x_values), "k:", label="pyMultiFit Chi2")
ax[1].set_ylabel("F(x)")

f.suptitle(r"$\chi^2$(1)")

for i in ax:
    i.set_xlabel("X")
    i.legend()
plt.tight_layout()
# plt.savefig('./../../images/chi2_example1.png')

y_multifit = ChiSquareDistribution(degree_of_freedom=1, loc=3, normalize=True)

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values, df=1, loc=3), label="Scipy translated Chi2")
ax[0].plot(x_values, y_multifit.pdf(x_values), "k:", label="pyMultiFit translated Chi2")
ax[0].set_ylabel("f(x)")

ax[1].plot(x_values, y_scipy.cdf(x=x_values, df=1, loc=3), label="Scipy translated Chi2")
ax[1].plot(x_values, y_multifit.cdf(x_values), "k:", label="pyMultiFit translated Chi2")
ax[1].set_ylabel("F(x)")

f.suptitle(r"$\chi^2$(1, loc=3)")

for i in ax:
    i.set_xlabel("X")
    i.legend()
plt.tight_layout()
# plt.savefig('./../../images/chi2_example2.png')
