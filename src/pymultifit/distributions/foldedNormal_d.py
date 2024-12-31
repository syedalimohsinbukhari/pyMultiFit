"""Created on Dec 04 03:42:42 2024"""

from typing import Any, Dict

import numpy as np
from scipy.special import erf

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import folded_normal_cdf_, folded_normal_pdf_


class FoldedNormalDistribution(BaseDistribution):
    r"""
    Class for FoldedNormal distribution.

    :param amplitude: The amplitude of the PDF. Defaults to 1.0. Ignored if **normalize** is ``True``.
    :type amplitude: float, optional

    :param mean: The mean parameter, :math:`\mu`. Defaults to 0.0.
    :type mean: float, optional

    :param sigma: The standard deviation parameter, :math:`\sigma`. Defaults to 1.0.
    :type sigma: float, optional

    :param normalize: If ``True``, the distribution is normalized so that the total area under the PDF equals 1. Defaults to ``False``.
    :type normalize: bool, optional

    :raise NegativeAmplitudeError: If the provided value of amplitude is negative.
    :raise NegativeStandardDeviationError: If the provided value of standard deviation is negative.
    """

    def __init__(self, amplitude: float = 1.0, mean: float = 0.0, sigma: float = 1., loc: float = 0.0, normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        self.amplitude = 1. if normalize else amplitude
        self.mean = mean
        self.sigma = sigma
        self.loc = loc

        self.norm = normalize

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return folded_normal_pdf_(x=x, amplitude=self.amplitude, mean=self.mean, sigma=self.sigma, loc=self.loc, normalize=self.norm)

    def cdf(self, x: np.array) -> np.array:
        return folded_normal_cdf_(x=x, amplitude=self.amplitude, mean=self.mean, sigma=self.sigma, loc=self.loc, normalize=self.norm)

    def stats(self) -> Dict[str, Any]:
        mean_, std_ = self.mean, self.sigma

        f1 = std_ * np.sqrt(2 / np.pi) * np.exp(-mean_**2 / (2 * std_**2))
        f2 = mean_ * erf(mean_ / (np.sqrt(2 * np.pi)))

        mu_y = f1 + f2
        var_y = mean_**2 + std_**2 - mu_y**2

        return {'mean': mu_y + self.loc,
                'variance': var_y}
