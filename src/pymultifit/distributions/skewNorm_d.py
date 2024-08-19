"""Created on Aug 03 21:35:28 2024"""

from typing import Dict

import numpy as np
from scipy.stats import skewnorm

from ._backend import BaseDistribution


# TODO:
#   See if normalization factor can be used to un-normalize the pdf.

class SkewedNormalDistribution(BaseDistribution):
    """Class for Skewed Normal distribution."""

    def __init__(self, shape: float = 1., location: float = 0., scale: float = 1.):
        self.shape = shape
        self.location = location
        self.scale = scale

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return skewnorm(self.shape, loc=self.location, scale=self.scale).pdf(x)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return skewnorm(self.shape, loc=self.location, scale=self.scale).cdf(x)

    def stats(self) -> Dict[str, float]:
        alpha, omega, epsilon = self.shape, self.scale, self.location
        delta = alpha / np.sqrt(1 + alpha**2)

        mean = epsilon + omega * delta * np.sqrt(2 / np.pi)

        def _m0(alpha_):
            m0 = np.sqrt(2 / np.pi) * delta
            m0 -= ((1 - np.pi / 4) * (np.sqrt(2 / np.pi) * delta)**3) / (1 - (2 / np.pi) * delta**2)
            m0 -= (2 * np.pi / abs(alpha_)) * np.exp(-(2 * np.pi / abs(alpha_))) * np.sign(alpha_)
            return m0

        mode = epsilon + omega * _m0(alpha)
        variance = omega**2 * (1 - (2 * delta**2 / np.pi))

        return {'mean': mean, 'mode': mode, 'variance': variance}

# def _skew_normal(x: np.ndarray,
#                  shape: float = 0., location: float = 0., scale: float = 1.) -> np.ndarray:
#     """
#     Compute the Skew-Normal distribution probability density function (PDF).
#
#     The Skew-Normal PDF is given by:
#     f(x) = (2 / sigma * sqrt(2 * pi)) * phi((x - mu) / sigma) * Phi(alpha * (x - mu) / sigma)
#     where:
#     - phi is the standard normal PDF.
#     - Phi is the standard normal cumulative distribution function (CDF).
#
#     Parameters
#     ----------
#     x : np.ndarray
#         The input values at which to evaluate the Skew-Normal PDF.
#     location : float
#         The location parameter (mean) of the Skew-Normal distribution.
#     scale : float
#         The scale parameter (standard deviation) of the Skew-Normal distribution.
#     shape : float
#         The shape parameter that controls the skewness of the distribution.
#
#     Returns
#     -------
#     np.ndarray
#         The probability density function values for the input values.
#
#     Raises
#     ------
#     ValueError
#         If `scale` is non-positive.
#
#     Notes
#     -----
#     - The Skew-Normal distribution is a generalization of the normal distribution that allows for skewness.
#     - The input `x` can be any real number, but `scale` must be positive.
#     """
#     if scale <= 0:
#         raise ValueError("scale must be positive.")
#
#     z = (x - location) / scale
#     gd = GaussianDistribution()
#     phi_z = gd.pdf(z)
#
#     def _phi(z_value):
#         return gd.cdf(z_value)
#
#     return 2 * scale * phi_z * _phi(shape * z)
