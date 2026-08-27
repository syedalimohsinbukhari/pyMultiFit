"""Created on Aug 20 10:42:20 2026"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from pymultifit.distributions.generalized import StudentsTDistribution

x_values = np.linspace(start=-10, stop=10, num=500)

y_multifit = StudentsTDistribution(v=3, loc=0.0, scale=1.0, normalize=True)
y_scipy = t

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values, df=3, loc=0.0, scale=1.0), label="Scipy StudentsT (df=3)")
ax[0].plot(x_values, y_multifit.pdf(x_values), "k:", label="pyMultiFit StudentsT (v=3)")
ax[0].set_ylabel("f(x)")

ax[1].plot(x_values, y_scipy.cdf(x=x_values, df=3, loc=0.0, scale=1.0), label="Scipy StudentsT (df=3)")
ax[1].plot(x_values, y_multifit.cdf(x_values), "k:", label="pyMultiFit StudentsT (v=3)")
ax[1].set_ylabel("F(x)")

f.suptitle("StudentsT(v=3, loc=0, scale=1)")

for i in ax:
    i.set_xlabel("X")
    i.legend()
plt.tight_layout()
plt.savefig("./../../images/students_T_example1.png")

y_multifit = StudentsTDistribution(v=3, loc=3.0, scale=1.5, normalize=True)

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x=x_values, df=3, loc=3.0, scale=1.5), label="Scipy translated StudentsT (df=3)")
ax[0].plot(x_values, y_multifit.pdf(x_values), "k:", label="pyMultiFit translated StudentsT (v=3)")
ax[0].set_ylabel("f(x)")

ax[1].plot(x_values, y_scipy.cdf(x=x_values, df=3, loc=3.0, scale=1.5), label="Scipy translated StudentsT (df=3)")
ax[1].plot(x_values, y_multifit.cdf(x_values), "k:", label="pyMultiFit translated StudentsT (v=3)")
ax[1].set_ylabel("F(x)")

f.suptitle("StudentsT(v=3, loc=3, scale=1.5)")

for i in ax:
    i.set_xlabel("X")
    i.legend()
plt.tight_layout()
plt.savefig("./../../images/students_T_example2.png")
plt.show()