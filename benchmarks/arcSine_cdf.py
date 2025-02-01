"""Created on Jan 31 23:22:15 2025"""

import numpy as np

from functions import plot_all_variations


def cdf_version1(y):
    return (2 / np.pi) * np.arcsin(np.sqrt(y))


def cdf_version2(y):
    return (2 / np.pi) * np.arcsin(y**0.5)


def cdf_version3(y):
    return (2 * np.arcsin(np.sqrt(y))) * np.pi**-1


def cdf_version4(y):
    return (2 * np.arcsin(y**0.5)) * np.pi**-1


test_values = np.linspace(start=0.001, stop=0.999, num=10**4)
func_list = [cdf_version1, cdf_version2, cdf_version3, cdf_version4]

latex_annotations = [r"$\left(\dfrac{2}{\pi}\right)\arcsin(\sqrt{y})$",
                     r"$\left(\dfrac{2}{\pi}\right)\arcsin(y^{0.5})$",
                     r"$\left[2 \arcsin(\sqrt{y})\right]\pi^{-1}$",
                     r"$\left[2 \arcsin(y^{0.5})\right]\pi^{-1}$"]

plot_all_variations(distribution_name='arcSine_cdf',
                    functions=func_list, values=test_values, latex_annotations=latex_annotations)
