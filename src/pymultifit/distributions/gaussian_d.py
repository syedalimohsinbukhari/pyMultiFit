"""Created on Aug 03 20:07:50 2024"""

from typing import Dict

import numpy as np
from scipy.special import erf

from .backend import BaseDistribution


class GaussianDistribution(BaseDistribution):
    """Class for Gaussian distribution."""

    def __init__(self, amplitude: float = 1.0, mean: float = 0., standard_deviation: float = 1., normalize: bool = False):
        self.amplitude = 1. if normalize else amplitude
        self.mean = mean
        self.std_ = standard_deviation

        self.norm = normalize

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return gaussian_(x, amplitude=self.amplitude, mu=self.mean, sigma=self.std_, normalize=self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        num_ = x - self.mean
        den_ = self.std_ * np.sqrt(2)
        return 0.5 * (1 + erf(num_ / den_))

    def stats(self) -> Dict[str, float]:
        mean_, std_ = self.mean, self.std_
        return {'mean': mean_,
                'median': mean_,
                'mode': mean_,
                'variance': std_**2}


def gaussian_(x: np.ndarray,
              amplitude: float = 1., mu: float = 0., sigma: float = 1.,
              normalize: bool = False) -> np.ndarray:
    """
    Compute the Gaussian (Normal) distribution probability density function (PDF).

    The Gaussian PDF is given by:
    f(x) = (1 / (sigma * sqrt(2 * pi))) * exp(- (x - mu)**2 / (2 * sigma**2))

    Parameters
    ----------
    x : np.ndarray
        The input values at which to evaluate the Gaussian PDF.
    amplitude: float
        The amplitude of the Gaussian distribution. Defaults to 1.
    mu : float
        The mean of the Gaussian distribution. Defaults to 0.
    sigma : float
        The standard deviation of the Gaussian distribution. Defaults to 1.
    normalize : bool
        If True, the function returns the normalized value of the PDF else the PDF is not normalized. Default is True.


    Returns
    -------
    np.ndarray
        The probability density function values for the input values.

    Notes
    -----
    The input `x` should be a NumPy array. If `x` is a scalar, it will be treated as a single-element array.
    """
    exponent_factor = (x - mu)**2 / (2 * sigma**2)

    if normalize:
        normalization_factor = sigma * np.sqrt(2 * np.pi)
        amplitude = 1
    else:
        normalization_factor = 1

    return amplitude * (np.exp(-exponent_factor) / normalization_factor)
