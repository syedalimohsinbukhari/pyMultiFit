"""Created on Jan 31 22:50:44 2025"""

import numpy as np

from functions import plot_all_variations


def pdf_version1(y):
    return 1 / (np.pi * np.sqrt(y * (1 - y)))


def pdf_version2(y):
    return 1 / (np.pi * (y * (1 - y))**0.5)


def pdf_version3(y):
    return 1 / (np.pi * np.sqrt(y - y**2))


def pdf_version4(y):
    return 1 / (np.pi * (y - y**2)**0.5)


test_values = np.linspace(start=0.001, stop=0.999, num=10**4)
func_list = [pdf_version1, pdf_version2, pdf_version3, pdf_version4]

latex_annotations = [r"$\dfrac{1}{\pi\sqrt{y(1-y)}}$",
                     r"$\dfrac{1}{\pi(y(1-y))^{0.5}}$",
                     r"$\dfrac{1}{\pi\sqrt{y-y^2}}$",
                     r"$\dfrac{1}{\pi(y-y^2)^{0.5}}$"]

plot_all_variations(distribution_name='arcSine_pdf',
                    functions=func_list, values=test_values, latex_annotations=latex_annotations)
