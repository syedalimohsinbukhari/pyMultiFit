"""Created on Aug 14 01:28:13 2024"""

from typing import Any, Dict, Optional

import numpy as np
from scipy.special import gamma, gammainc

from ._backend import BaseDistribution


class GammaDistribution(BaseDistribution):

    def __init__(self,
                 amplitude: Optional[float] = 1.,
                 alpha: Optional[float] = 1.,
                 beta: Optional[float] = 1.,
                 normalize: bool = True):
        self.alpha = alpha
        self.beta = beta
        self.norm = normalize

        self.amplitude = 1 if normalize else amplitude

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return _gamma(x, self.amplitude, self.alpha, self.beta, self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return gammainc(self.alpha, self.beta * x)

    def stats(self) -> Dict[str, Any]:
        a, b = self.alpha, self.beta

        mean_ = a / b
        mode_ = (a - 1) / b if a >= 1 else 0
        variance_ = a / b**2

        return {'mean': mean_,
                'mode': mode_,
                'variance': variance_}


def _gamma(x: np.ndarray,
           amplitude: Optional[float] = 1.0,
           alpha: Optional[float] = 1.0,
           beta: Optional[float] = 1.0,
           normalize: bool = True) -> np.ndarray:
    """
    Computes the Gamma distribution PDF for given parameters.

    Parameters
    ----------
    x : np.ndarray
        Values at which to evaluate the PDF.
    amplitude : float, optional
        The scaling factor for the distribution. Defaults to 1.
    alpha : float, optional
        The shape parameter of the Gamma distribution. Defaults to 1.
    beta : float, optional
        The rate parameter of the Gamma distribution. Defaults to 1.
    normalize : bool, optional
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
