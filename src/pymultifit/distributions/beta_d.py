"""Created on Aug 14 00:45:37 2024"""

from typing import Any, Dict

import numpy as np
from scipy.special import beta as beta_, betainc

from . import oFloat
from ._backend import BaseDistribution


class BetaDistribution(BaseDistribution):
    """Class for Beta distribution."""

    def __init__(self,
                 alpha: oFloat = 1.,
                 beta: oFloat = 1.):
        self.alpha = alpha
        self.beta = beta

        self.norm = True
        self.amplitude = 1.0

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return _beta(x, amplitude=self.amplitude, alpha=self.alpha, beta=self.beta, normalize=self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    @classmethod
    def with_amplitude(cls, amplitude: oFloat = 1., alpha: oFloat = 1., beta: oFloat = 1.):
        """
        Create an instance with a specified amplitude, without normalization.

        Parameters
        ----------
        amplitude : float, optional
            The amplitude to apply to the PDF. Defaults to 1.
        alpha : float, optional
            The alpha parameter of the beta distribution. Defaults to 1.
        beta : float, optional
            The beta parameter of the beta distribution. Defaults to 1.

        Returns
        -------
        BetaDistribution
            An instance of BetaDistribution with the specified amplitude and parameters.
        """
        instance = cls(alpha=alpha, beta=beta)
        instance.amplitude = amplitude
        instance.norm = False
        return instance

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return betainc(self.alpha, self.beta, x)

    def stats(self) -> Dict[str, Any]:
        a, b = self.alpha, self.beta

        mean_ = a / (a + b)

        variance_ = (a * b) / ((a + b)**2 * (a + b + 1))

        return {'mean': mean_,
                'variance': variance_}


def _beta(x: np.ndarray,
          amplitude: oFloat = 1.,
          alpha: oFloat = 1.,
          beta: oFloat = 1.,
          normalize: bool = True) -> np.ndarray:
    """
    Compute the beta probability density function (PDF).

    Parameters
    ----------
    x : np.ndarray
        The input array for which to compute the PDF.
    amplitude : float, optional
        The amplitude to apply to the PDF. Default is 1.
    alpha : float, optional
        The alpha (shape) parameter of the beta distribution. Default is 1.
    beta : float, optional
        The beta (shape) parameter of the beta distribution. Default is 1.
    normalize : bool, optional
        If True, the PDF is normalized using the beta function. Default is True.

    Returns
    -------
    np.ndarray
        The probability density function values for the given input.
    """
    numerator = x**(alpha - 1) * (1 - x)**(beta - 1)

    if normalize:
        normalization_factor = beta_(alpha, beta)
        amplitude = 1.0
    else:
        normalization_factor = 1.0

    return amplitude * (numerator / normalization_factor)
