"""Created on Aug 03 21:12:13 2024"""

from typing import Dict

import numpy as np

from ._backend import BaseDistribution


class LaplaceDistribution(BaseDistribution):
    """Class for Laplace distribution."""

    def __init__(self, mean: float = 0, diversity: float = 1):
        self.mu = mean
        self.b = diversity

        self.norm = True
        self.amplitude = 1

    @classmethod
    def with_amplitude(cls, amplitude: float = 1., mean: float = 0., diversity: float = 1.):
        """
        Create an instance with a specified amplitude, without normalization.

        Parameters
        ----------
        amplitude : float
            The amplitude (scale) of the distribution. Defaults to 1.
        mean : float
            The mean (location parameter) of the distribution. Defaults to 0.
        diversity : float
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
        return laplace_(x, amplitude=self.amplitude, mean=self.mu, diversity=self.b, normalize=self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        def _cdf1(x_):
            return 0.5 * np.exp((x_ - self.mu) / self.b)

        def _cdf2(x_):
            return 1 - 0.5 * np.exp(-(x_ - self.mu) / self.b)

        return np.piecewise(x, [x <= self.mu, x > self.mu], [_cdf1, _cdf2])

    def stats(self) -> Dict[str, float]:
        mean_, b_ = self.mu, self.b

        return {'mean': mean_,
                'median': mean_,
                'mode': mean_,
                'variance': 2 * b_**2}


def laplace_(x: np.ndarray,
             amplitude: float = 1., mean: float = 0., diversity: float = 1.,
             normalize: bool = True) -> np.ndarray:
    """Compute the Laplace distribution's probability density function (PDF).

    Parameters
    ----------
    x : np.ndarray
        Points at which to evaluate the PDF.
    amplitude : float
        The amplitude (scale) of the distribution. Defaults to 1.
    mean : float
        The mean (location parameter) of the distribution. Defaults to 0.
    diversity : float
        The diversity (scale parameter) of the distribution. Defaults to 1.
    normalize : bool, optional
        Whether to normalize the PDF. Defaults to True.

    Returns
    -------
    np.ndarray
        The PDF values at the given points.
    """
    exponent_factor = abs(x - mean) / diversity

    if normalize:
        normalization_factor = 2 * diversity
        amplitude = 1
    else:
        normalization_factor = 1

    return amplitude * (np.exp(-exponent_factor) / normalization_factor)
