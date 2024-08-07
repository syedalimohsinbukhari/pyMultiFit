"""Created on Aug 03 20:07:50 2024"""

from typing import Optional

import numpy as np
from scipy.special import erf

from ._backend import BaseDistribution


class GaussianDistribution(BaseDistribution):
    """Class for Gaussian distribution."""

    def __init__(self,
                 mean: Optional[float] = 0,
                 standard_deviation: Optional[float] = 1,
                 normalize: bool = True):
        self.mean = mean
        self.std_ = standard_deviation
        self.norm = normalize

    @classmethod
    def from_standard_notation(cls, mu, sigma, normalize: bool = True):
        """Initialize the distribution class from standard mathematical notation as parameters."""
        return cls(mean=mu, standard_deviation=sigma, normalize=normalize)

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return _gaussian(x, self.mean, self.std_, self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        num_ = x - self.mean
        den_ = self.std_ * np.sqrt(2)
        return 0.5 * (1 + erf(num_ / den_))

    def stats(self):
        mean_, std_ = self.mean, self.std_
        return {'mean': mean_,
                'median': mean_,
                'mode': mean_,
                'variance': std_**2}


def _gaussian(x: np.ndarray,
              mu: Optional[float] = 0,
              sigma: Optional[float] = 1,
              normalize: bool = True) -> np.ndarray:
    """
    Compute the Gaussian (Normal) distribution probability density function (PDF).

    The Gaussian PDF is given by:
    f(x) = (1 / (sigma * sqrt(2 * pi))) * exp(- (x - mu)**2 / (2 * sigma**2))

    Parameters
    ----------
    x : np.ndarray
        The input values at which to evaluate the Gaussian PDF.
    mu : float, optional
        The mean of the Gaussian distribution. Defaults to 0.
    sigma : float, optional
        The standard deviation of the Gaussian distribution. Defaults to 1.
    normalize : bool, optional
        If True, the function returns the normalized value of the PDF else the PDF is not normalized. Default is True.


    Returns
    -------
    np.ndarray
        The probability density function values for the input values.

    Notes
    -----
    The input `x` should be a NumPy array. If `x` is a scalar, it will be treated as a single-element array.
    """
    exponent = - (x - mu)**2 / (2 * sigma**2)
    normalization = sigma * np.sqrt(2 * np.pi) if normalize else 1

    return np.exp(exponent) / normalization
