"""Created on Aug 14 00:45:37 2024"""

from typing import Dict

import numpy as np
from scipy.special import beta as beta_function, betainc

from .backend import BaseDistribution


class BetaDistribution(BaseDistribution):
    """Class for Beta distribution."""

    def __init__(self, amplitude: float = 1., alpha: float = 1., beta: float = 1., normalize: bool = False):
        self.amplitude = 1. if normalize else amplitude
        self.alpha = alpha
        self.beta = beta

        self.norm = normalize

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return _beta(x, amplitude=self.amplitude, alpha=self.alpha, beta=self.beta, normalize=self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return betainc(self.alpha, self.beta, x)

    def stats(self) -> Dict[str, float]:
        a, b = self.alpha, self.beta

        mean_ = a / (a + b)
        variance_ = (a * b) / ((a + b)**2 * (a + b + 1))

        return {'mean': mean_,
                'variance': variance_}


def _beta(x: np.ndarray,
          amplitude: float = 1., alpha: float = 1., beta: float = 1.,
          normalize: bool = False) -> np.ndarray:
    """
    Compute the beta probability density function (PDF).

    Parameters
    ----------
    x : np.ndarray
        The input array for which to compute the PDF.
    amplitude : float
        The amplitude to apply to the PDF. Default is 1.
    alpha : float
        The alpha (shape) parameter of the beta distribution. Default is 1.
    beta : float
        The beta (shape) parameter of the beta distribution. Default is 1.
    normalize : bool
        If True, the PDF is normalized using the beta function. Default is True.

    Returns
    -------
    np.ndarray
        The probability density function values for the given input.
    """
    numerator = x**(alpha - 1) * (1 - x)**(beta - 1)

    if normalize:
        normalization_factor = beta_function(alpha, beta)
        amplitude = 1.0
    else:
        normalization_factor = 1.0

    return amplitude * (numerator / normalization_factor)
