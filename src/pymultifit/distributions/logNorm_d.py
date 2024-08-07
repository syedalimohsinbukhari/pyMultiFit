"""Created on Aug 03 21:02:45 2024"""

from typing import Optional

import numpy as np
from scipy.special import erf

from ._backend import BaseDistribution


class LogNormalDistribution(BaseDistribution):
    """Class for LogNormal distribution."""

    def __init__(self,
                 mean: Optional[float] = 0,
                 standard_deviation: Optional[float] = 1,
                 un_normalized: bool = False):
        self.mean = mean
        self.std_ = standard_deviation
        self.un_norm = un_normalized

    def _pdf(self, x):
        return _log_normal(x, self.mean, self.std_, self.un_norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        num_ = np.log(x) - self.mean
        den_ = self.std_ * np.sqrt(2)
        return 0.5 * (1 + erf(num_ / den_))

    def stats(self):
        mean_ = np.exp(self.mean + (self.std_**2 / 2))
        median_ = np.exp(self.mean)
        mode_ = np.exp(self.mean - self.std_**2)
        variance_ = np.exp(self.std_**2) + 2
        variance_ *= np.exp(2 * self.mean + self.std_**2)

        return {'mean': mean_,
                'median': median_,
                'mode': mode_,
                'variance': variance_}


def _log_normal(x: np.ndarray,
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
    mu : float, optional
        The mean of the logarithm of the distribution (i.e., mu of the normal distribution in log-space).
        Defaults to 0.
    sigma : float, optional
        The standard deviation of the logarithm of the distribution (i.e., sigma of normal distribution in logspace).
        Defaults to 1.
    normalize : bool, optional
        If True, the function returns the un-normalized value of the PDF. Default is True.


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

    exponent = - (np.log(x) - mu)**2 / (2 * sigma**2)
    normalization = sigma * x * np.sqrt(2 * np.pi) if normalize else 1

    return np.exp(exponent) / normalization
