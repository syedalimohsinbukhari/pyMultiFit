"""Created on Aug 12 11:26:00 2026"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import genpareto

from pymultifit.distributions.generalized import QExponentialDistribution
from pymultifit.distributions.utilities_d import q_exp_to_gen_pareto

x_values = np.linspace(start=-10, stop=10, num=500)

y_multifit = QExponentialDistribution(normalize=True)
y_scipy = genpareto

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x_values, **q_exp_to_gen_pareto(1, 1, 0)), label="Scipy GenPareto")
ax[0].plot(x_values, y_multifit.pdf(x_values), "k:", label="pyMultiFit QExponential")
ax[0].set_ylabel("f(x)")

ax[1].plot(x_values, y_scipy.cdf(x_values, **q_exp_to_gen_pareto(1, 1, 0)), label="Scipy GenPareto")
ax[1].plot(x_values, y_multifit.cdf(x_values), "k:", label="pyMultiFit QExponential")
ax[1].set_ylabel("F(x)")

f.suptitle("QExponential(1, 1, 0)")

for i in ax:
    i.set_xlabel("X")
    i.legend()
plt.tight_layout()
plt.savefig("./../../images/q_exponential_example1.png")

y_multifit = QExponentialDistribution(q=1.5, rate=1.3, loc=-3.3, normalize=True)

f, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x_values, y_scipy.pdf(x_values, **q_exp_to_gen_pareto(1.5, 1.3, -3.3)),
           label="Scipy translated GenPareto")
ax[0].plot(x_values, y_multifit.pdf(x_values), "k:", label="pyMultiFit translated QExponential")
ax[0].set_ylabel("f(x)")

ax[1].plot(x_values, y_scipy.cdf(x_values, **q_exp_to_gen_pareto(1.5, 1.3, -3.3)),
           label="Scipy translated GenPareto")
ax[1].plot(x_values, y_multifit.cdf(x_values), "k:", label="pyMultiFit translated QExponential")
ax[1].set_ylabel("F(x)")

f.suptitle("QExponential(1.5, 1.3, -3.3)")

for i in ax:
    i.set_xlabel("X")
    i.legend()
plt.tight_layout()
plt.savefig("./../../images/q_exponential_example2.png")
plt.show()
