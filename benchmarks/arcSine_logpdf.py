"""Created on Jan 31 22:21:08 2025"""

import numpy as np

from functions import plot_all_variations
from pymultifit import EPSILON


def log_pdf_version1(y):
    return -np.log(np.pi * np.sqrt(y * (1 - y)))


def log_pdf_version2(y):
    return -np.log(np.pi) - np.log(np.sqrt(y * (1 - y)))


def log_pdf_version3(y):
    return -np.log(np.pi) - 0.5 * (np.log(y) + np.log(1 - y))


def log_pdf_version4(y):
    return -np.log(np.pi) - np.log(np.sqrt(y - y**2))


def log_pdf_version5(y):
    return -np.log(np.pi) - 0.5 * (np.log(y - y**2))


def log_pdf_version6(y):
    return -(np.log(np.pi) + 0.5 * (np.log(y - y**2)))


test_values = np.linspace(start=np.sqrt(EPSILON), stop=1 - np.sqrt(EPSILON), num=10**4)
func_list = [log_pdf_version1, log_pdf_version2, log_pdf_version3, log_pdf_version4,
             log_pdf_version5, log_pdf_version6]

latex_annotations = [r"$-\ln(\pi \sqrt{y(1 - y)})$",
                     r"$-\ln(\pi) - \ln(\sqrt{y(1 - y)})$",
                     r"$-\ln(\pi) - 0.5 (\ln(y) + \log(1 - y))$",
                     r"$-\ln(\pi) - \ln(\sqrt{y - y^2})$",
                     r"$-\ln(\pi) - 0.5 (\ln(y - y^2))$",
                     r"$-(\ln(\pi) + 0.5 \ln(y - y^2))$"]

plot_all_variations(distribution_name='arcSine_logpdf',
                    functions=func_list, values=test_values, latex_annotations=latex_annotations)
