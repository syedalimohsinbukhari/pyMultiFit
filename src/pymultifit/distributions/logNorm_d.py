"""Created on Aug 03 21:02:45 2024"""

from typing import Any, Dict, Optional

import numpy as np
from scipy.special import erf

from ._backend import BaseDistribution


class LogNormalDistribution(BaseDistribution):
    """Class for representing a Log-Normal distribution.

    Parameters
    ----------
    amplitude : float, optional
        The amplitude (scale) of the distribution. Defaults to 1.
    mean : float, optional
        The mean of the logarithm of the distribution (i.e., mean of the underlying normal distribution in log-space).
        Defaults to 0.
    standard_deviation : float, optional
        The standard deviation of the logarithm of the distribution (i.e., std. dev. of normal distribution in
        log-space). Defaults to 1.
    normalize : bool, optional
        Whether to normalize the distribution so that the area under the curve equals 1. Defaults to True.
    """

    def __init__(self,
                 amplitude: Optional[float] = 1,
                 mean: Optional[float] = 0,
                 standard_deviation: Optional[float] = 1,
                 normalize: bool = True):
        self.mean = mean
        self.std_ = standard_deviation
        self.norm = normalize

        # Set amplitude to 1 if normalized, else use the given amplitude
        self.amplitude = 1 if normalize else amplitude

    @classmethod
    def with_amplitude(cls, amplitude: Optional[float] = 1, mean: Optional[float] = 0,
                       standard_deviation: Optional[float] = 1):
        """Alternative constructor to create a LogNormalDistribution instance without normalization.

        Parameters
        ----------
        amplitude : float, optional
            The amplitude (scale) of the distribution. Defaults to 1.
        mean : float, optional
            The mean of the logarithm of the distribution. Defaults to 0.
        standard_deviation : float, optional
            The standard deviation of the logarithm of the distribution. Defaults to 1.

        Returns
        -------
        LogNormalDistribution
            An instance of LogNormalDistribution with specified amplitude.
        """
        return cls(amplitude, mean, standard_deviation, False)

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        """Private method to compute the probability density function (PDF) of the Log-Normal distribution.

        Parameters
        ----------
        x : np.ndarray
            Points at which to evaluate the PDF. Must be positive.

        Returns
        -------
        np.ndarray
            The PDF values at the given points.
        """
        return _log_normal(x, self.amplitude, self.mean, self.std_, self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute the probability density function (PDF) of the Log-Normal distribution.

        Parameters
        ----------
        x : np.ndarray
            Points at which to evaluate the PDF. Must be positive.

        Returns
        -------
        np.ndarray
            The PDF values at the given points.
        """
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Compute the cumulative distribution function (CDF) of the Log-Normal distribution.

        Parameters
        ----------
        x : np.ndarray
            Points at which to evaluate the CDF. Must be positive.

        Returns
        -------
        np.ndarray
            The CDF values at the given points.
        """
        num_ = np.log(x) - self.mean
        den_ = self.std_ * np.sqrt(2)
        return 0.5 * (1 + erf(num_ / den_))

    def stats(self) -> Dict[str, Any]:
        """Compute statistical properties of the Log-Normal distribution.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the mean, median, mode, and variance of the distribution.
        """
        mean_ = np.exp(self.mean + (self.std_**2 / 2))
        median_ = np.exp(self.mean)
        mode_ = np.exp(self.mean - self.std_**2)
        variance_ = (np.exp(self.std_**2) - 1) * np.exp(2 * self.mean + self.std_**2)

        return {
            'mean': mean_,
            'median': median_,
            'mode': mode_,
            'variance': variance_
        }


def _log_normal(x: np.ndarray,
                amplitude: Optional[float] = 1,
                mu: Optional[float] = 0,
                sigma: Optional[float] = 1,
                normalize: bool = True) -> np.ndarray:
    """
    Compute the Log-Normal distribution probability density function (PDF).

    The Log-Normal PDF is given by:

    f(x) = (1 / (x * sigma * sqrt(2 * pi))) * exp(- (log(x) - mu)**2 / (2 * sigma**2))

    Parameters
    ----------
    x : np.ndarray
        The input values at which to evaluate the Log-Normal PDF. Must be positive.
    amplitude : float, optional
        The amplitude (scale) of the distribution. Defaults to 1.
    mu : float, optional
        The mean of the logarithm of the distribution (i.e., mu of the normal distribution in log-space). Defaults to 0.
    sigma : float, optional
        The standard deviation of the logarithm of the distribution (i.e., sigma of the normal distribution in log-space). Defaults to 1.
    normalize : bool, optional
        If True, the function returns the normalized value of the PDF. Defaults to True.

    Returns
    -------
    np.ndarray
        The probability density function values for the input values.

    Raises
    ------
    ValueError
        If any value in `x` is less than or equal to zero.

    Notes
    -----
    The input `x` must be positive because the logarithm of zero or negative numbers is undefined.
    """
    if np.any(x <= 0):
        raise ValueError("x must be positive for the log-normal distribution.")

    exponent_factor = (np.log(x) - mu)**2 / (2 * sigma**2)

    if normalize:
        normalization_factor = sigma * x * np.sqrt(2 * np.pi)
        amplitude = 1
    else:
        normalization_factor = 1

    return amplitude * np.exp(-exponent_factor) / normalization_factor
