"""Created on Aug 14 01:28:13 2024"""

from typing import Dict

import numpy as np
from scipy.special import gamma, gammainc

from ._backend import BaseDistribution


class GammaDistribution(BaseDistribution):
    """Class for Gamma distribution."""

    def __init__(self, alpha: float = 1., beta: float = 1.):
        self.alpha = alpha
        self.beta = beta

        self.amplitude = 1
        self.norm = True

    @classmethod
    def with_amplitude(cls, amplitude: float = 1., alpha: float = 1., beta: float = 1.):
        """
        Create an instance with a specified amplitude, without normalization.

        Parameters
        ----------
        amplitude : float
            The amplitude to apply to the PDF. Defaults to 1.
        alpha : float
            The alpha parameter of the gamma distribution. Defaults to 1.
        beta : float
            The beta parameter of the gamma distribution. Defaults to 1.

        Returns
        -------
        GammaDistribution
            An instance of GammaDistribution with the specified amplitude and parameters.
        """
        instance = cls(alpha=alpha, beta=beta)
        instance.amplitude = amplitude
        instance.norm = False
        return instance

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return gamma_(x, amplitude=self.amplitude, alpha=self.alpha, beta=self.beta, normalize=self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return gammainc(self.alpha, self.beta * x)

    def stats(self) -> Dict[str, float]:
        a, b = self.alpha, self.beta

        mean_ = a / b
        mode_ = (a - 1) / b if a >= 1 else 0
        variance_ = a / b**2

        return {'mean': mean_,
                'mode': mode_,
                'variance': variance_}


def gamma_(x: np.ndarray,
           amplitude: float = 1., alpha: float = 1., beta: float = 1.,
           normalize: bool = True) -> np.ndarray:
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
