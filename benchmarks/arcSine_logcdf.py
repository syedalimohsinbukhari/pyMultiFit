"""Created on Jan 31 23:22:29 2025"""

import numpy as np

from functions import plot_all_variations


def log_cdf_version1(y):
    return np.log(2 / np.pi) + np.log(np.arcsin(np.sqrt(y)))


def log_cdf_version2(y):
    return np.log(2 / np.pi) + np.log(np.arcsin(y**0.5))


def log_cdf_version3(y):
    return np.log(2) - np.log(np.pi) + np.log(np.arcsin(np.sqrt(y)))


def log_cdf_version4(y):
    return np.log(2) - np.log(np.pi) + np.log(np.arcsin(y**0.5))


test_values = np.linspace(start=0.001, stop=0.999, num=10**5)
func_list = [log_cdf_version1, log_cdf_version2, log_cdf_version3, log_cdf_version4]

latex_annotations = [r"$\log\left(\dfrac{2}{\pi}\right) + \log(\arcsin(\sqrt{y}))$",
                     r"$\log\left(\dfrac{2}{\pi}\right) + \log(\arcsin(y^{0.5}))$",
                     r"$\log(2) - \log(\pi) + \log(\arcsin(\sqrt{y}))$",
                     r"$\log(2) - \log(\pi) + \log(\arcsin(y^{0.5}))$"]

plot_all_variations(distribution_name='arcSine_logcdf',
                    functions=func_list, values=test_values, latex_annotations=latex_annotations)
