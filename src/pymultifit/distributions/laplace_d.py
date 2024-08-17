"""Created on Aug 03 21:12:13 2024"""

from typing import Any, Dict

import numpy as np

from . import oFloat
from ._backend import BaseDistribution


class LaplaceDistribution(BaseDistribution):
    """Class for Laplace distribution."""

    def __init__(self,
                 mean: oFloat = 0,
                 diversity: oFloat = 1):
        self.mu = mean
        self.b = diversity

        self.norm = True
        self.amplitude = 1

    @classmethod
    def with_amplitude(cls, amplitude: oFloat = 1., mean: oFloat = 0., diversity: oFloat = 1.):
        """
        Create an instance with a specified amplitude, without normalization.

        Parameters
        ----------
        amplitude : float, optional
            The amplitude (scale) of the distribution. Defaults to 1.
        mean : float, optional
            The mean (location parameter) of the distribution. Defaults to 0.
        diversity : float, optional
            The diversity (scale parameter) of the distribution. Defaults to 1.

        Returns
        -------
        LaplaceDistribution
            An instance of LaplaceDistribution with specified amplitude.
        """
        instance = cls(mean=mean, diversity=diversity)
        instance.amplitude = amplitude
        instance.norm = False

        return instance

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return _laplace(x, amplitude=self.amplitude, mu=self.mu, b=self.b, normalize=self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        def _cdf1(x_):
            return 0.5 * np.exp((x_ - self.mu) / self.b)

        def _cdf2(x_):
            return 1 - 0.5 * np.exp(-(x_ - self.mu) / self.b)

        # Use piecewise function to handle different branches for x <= mu and x > mu
        return np.piecewise(x, [x <= self.mu, x > self.mu], [_cdf1, _cdf2])

    def stats(self) -> Dict[str, Any]:
        mean_, b_ = self.mu, self.b

        return {
            'mean': mean_,
            'median': mean_,
            'mode': mean_,
            'variance': 2 * b_**2
        }


def _laplace(x: np.ndarray,
             amplitude: oFloat = 1.,
             mu: oFloat = 0.,
             b: oFloat = 1.,
             normalize: bool = True) -> np.ndarray:
    """Compute the Laplace distribution's probability density function (PDF).

    Parameters
    ----------
    x : np.ndarray
        Points at which to evaluate the PDF.
    amplitude : float, optional
        The amplitude (scale) of the distribution. Defaults to 1.
    mu : float, optional
        The mean (location parameter) of the distribution. Defaults to 0.
    b : float, optional
        The diversity (scale parameter) of the distribution. Defaults to 1.
    normalize : bool, optional
        Whether to normalize the PDF. Defaults to True.

    Returns
    -------
    np.ndarray
        The PDF values at the given points.
    """
    exponent_factor = abs(x - mu) / b

    if normalize:
        normalization_factor = 2 * b
        amplitude = 1
    else:
        normalization_factor = 1

    return amplitude * (np.exp(-exponent_factor) / normalization_factor)
