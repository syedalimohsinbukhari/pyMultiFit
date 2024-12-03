"""Created on Aug 14 01:28:13 2024"""

from typing import Dict

import numpy as np
from scipy.special import gamma, gammainc

from .backend import BaseDistribution


class GammaDistributionSR(BaseDistribution):
    """Class for Gamma distribution."""

    def __init__(self, amplitude: float = 1., shape: float = 1., rate: float = 1., normalize: bool = False):
        self.amplitude = 1. if normalize else amplitude
        self.shape = shape
        self.rate = rate

        self.norm = normalize

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return gamma_(x, amplitude=self.amplitude, alpha=self.shape, beta=self.rate, normalize=self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return gammainc(self.shape, self.rate * x)

    def stats(self) -> Dict[str, float]:
        a, b = self.shape, self.rate

        mean_ = a / b
        mode_ = (a - 1) / b if a >= 1 else 0
        variance_ = a / b**2

        return {'mean': mean_,
                'mode': mode_,
                'variance': variance_}


class GammaDistributionSS(GammaDistributionSR):
    def __init__(self, amplitude: float = 1., shape: float = 1., scale: float = 1., normalize: bool = False):
        super().__init__(amplitude=amplitude, shape=shape, rate=1 / scale, normalize=normalize)


def gamma_(x: np.ndarray,
           amplitude: float = 1., alpha: float = 1., beta: float = 1.,
           normalize: bool = False) -> np.ndarray:
    """
    Computes the Gamma distribution PDF for given parameters.

    Parameters
    ----------
    x : np.ndarray
        Values at which to evaluate the PDF.
    amplitude : float
        The scaling factor for the distribution. Defaults to 1.
    alpha : float
        The shape parameter of the Gamma distribution. Defaults to 1.
    beta : float
        The rate parameter of the Gamma distribution. Defaults to 1.
    normalize : bool
        Whether to normalize the distribution (i.e., set amplitude to 1). Defaults to True.

    Returns
    -------
    np.ndarray
        The probability density function evaluated at `x`.
    """
    numerator = x**(alpha - 1) * np.exp(-beta * x)

    if normalize:
        normalization_factor = gamma(alpha) / beta**alpha
        amplitude = 1
    else:
        normalization_factor = 1

    return amplitude * (numerator / normalization_factor)
