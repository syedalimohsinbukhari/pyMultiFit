"""Created on Aug 03 21:02:45 2024"""

from typing import Dict

import numpy as np
from scipy.special import erf

from ._backend import BaseDistribution


class LogNormalDistribution(BaseDistribution):
    """Class for Log-Normal distribution."""

    def __init__(self, mean: float = 0., standard_deviation: float = 1.):
        self.mean = mean
        self.std_ = standard_deviation

        self.norm = True
        self.amplitude = 1

    @classmethod
    def with_amplitude(cls, amplitude: float = 1., mean: float = 0., standard_deviation: float = 1.):
        """
        Create an instance with a specified amplitude, without normalization.

        Parameters
        ----------
        amplitude : float
            The amplitude (scale) of the distribution. Defaults to 1.
        mean : float
            The mean of the logarithm of the distribution. Defaults to 0.
        standard_deviation : float
            The standard deviation of the logarithm of the distribution. Defaults to 1.

        Returns
        -------
        LogNormalDistribution
            An instance of LogNormalDistribution with specified amplitude.
        """
        instance = cls(mean=mean, standard_deviation=standard_deviation)
        instance.amplitude = amplitude
        instance.norm = False

        return instance

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return log_normal_(x, amplitude=self.amplitude, mean=self.mean, standard_deviation=self.std_, normalize=self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        num_ = np.log(x) - self.mean
        den_ = self.std_ * np.sqrt(2)
        return 0.5 * (1 + erf(num_ / den_))

    def stats(self) -> Dict[str, float]:
        mean_ = np.exp(self.mean + (self.std_**2 / 2))
        median_ = np.exp(self.mean)
        mode_ = np.exp(self.mean - self.std_**2)
        variance_ = (np.exp(self.std_**2) - 1) * np.exp(2 * self.mean + self.std_**2)

        return {'mean': mean_,
                'median': median_,
                'mode': mode_,
                'variance': variance_}


def log_normal_(x: np.ndarray,
                amplitude: float = 1., mean: float = 0., standard_deviation: float = 1.,
                normalize: bool = True) -> np.ndarray:
    """
    Compute the Log-Normal distribution probability density function (PDF).

    The Log-Normal PDF is given by:

    f(x) = (1 / (x * sigma * sqrt(2 * pi))) * exp(- (log(x) - mu)**2 / (2 * sigma**2))

    Parameters
    ----------
    x : np.ndarray
        The input values at which to evaluate the Log-Normal PDF. Must be positive.
    amplitude : float
        The amplitude (scale) of the distribution. Defaults to 1.
    mean : float
        The mean of the logarithm of the distribution (i.e., mu of the normal distribution in log-space). Defaults to 0.
    standard_deviation : float
        The standard deviation of the logarithm of the distribution (i.e., sigma of the normal distribution in
        log-space). Defaults to 1.
    normalize : bool
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

    exponent_factor = (np.log(x) - mean)**2 / (2 * standard_deviation**2)

    if normalize:
        normalization_factor = standard_deviation * x * np.sqrt(2 * np.pi)
        amplitude = 1
    else:
        normalization_factor = 1

    return amplitude * np.exp(-exponent_factor) / normalization_factor
