"""Created on Aug 03 21:12:13 2024"""

from typing import Any, Dict, Optional

import numpy as np

from ._backend import BaseDistribution


class LaplaceDistribution(BaseDistribution):
    """Class for representing a Laplace distribution.

    Parameters
    ----------
    amplitude : float, optional
        The amplitude (scale) of the distribution. Defaults to 1.
    mean : float, optional
        The mean (location parameter) of the distribution. Defaults to 0.
    diversity : float, optional
        The diversity (scale parameter) of the distribution, also known as `b`. Defaults to 1.
    normalize : bool, optional
        Whether to normalize the distribution so that the area under the curve equals 1. Defaults to True.
    """

    def __init__(self,
                 amplitude: Optional[float] = 1,
                 mean: Optional[float] = 0,
                 diversity: Optional[float] = 1,
                 normalize: bool = True):
        self.mu = mean
        self.b = diversity
        self.norm = normalize

        # Set amplitude to 1 if normalized, else use the given amplitude
        self.amplitude = 1 if normalize else amplitude

    @classmethod
    def with_amplitude(cls, amplitude=1, mean=0, diversity=1):
        """Alternative constructor to create a LaplaceDistribution instance without normalization.

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
        return cls(amplitude, mean, diversity, False)

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        """Private method to compute the probability density function (PDF) of the Laplace distribution.

        Parameters
        ----------
        x : np.ndarray
            Points at which to evaluate the PDF.

        Returns
        -------
        np.ndarray
            The PDF values at the given points.
        """
        return _laplace(x, self.amplitude, self.mu, self.b, self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute the probability density function (PDF) of the Laplace distribution.

        Parameters
        ----------
        x : np.ndarray
            Points at which to evaluate the PDF.

        Returns
        -------
        np.ndarray
            The PDF values at the given points.
        """
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Compute the cumulative distribution function (CDF) of the Laplace distribution.

        Parameters
        ----------
        x : np.ndarray
            Points at which to evaluate the CDF.

        Returns
        -------
        np.ndarray
            The CDF values at the given points.
        """

        def _cdf1(x_):
            return 0.5 * np.exp((x_ - self.mu) / self.b)

        def _cdf2(x_):
            return 1 - 0.5 * np.exp(-(x_ - self.mu) / self.b)

        # Use piecewise function to handle different branches for x <= mu and x > mu
        return np.piecewise(x, [x <= self.mu, x > self.mu], [_cdf1, _cdf2])

    def stats(self) -> Dict[str, Any]:
        """Compute statistical properties of the Laplace distribution.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the mean, median, mode, and variance of the distribution.
        """
        mean_, b_ = self.mu, self.b

        return {
            'mean': mean_,
            'median': mean_,
            'mode': mean_,
            'variance': 2 * b_**2
        }


def _laplace(x: np.ndarray,
             amplitude: Optional[float] = 1.,
             mu: Optional[float] = 0.,
             b: Optional[float] = 1.,
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
