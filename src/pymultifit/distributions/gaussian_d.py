"""Created on Aug 03 20:07:50 2024"""

from typing import Any, Dict, Optional

import numpy as np
from scipy.special import erf

from ._backend import BaseDistribution


class GaussianDistribution(BaseDistribution):
    """Class for Gaussian distribution."""

    def __init__(self,
                 amplitude: Optional[float] = 1,
                 mean: Optional[float] = 0,
                 standard_deviation: Optional[float] = 1,
                 normalize: bool = True):
        self.mean = mean
        self.std_ = standard_deviation
        self.norm = normalize

        self.amplitude = 1 if normalize else amplitude

    @classmethod
    def with_amplitude(cls, amplitude=1, mu=0, standard_deviation=1):
        return cls(amplitude, mu, standard_deviation, False)

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return _gaussian(x, self.amplitude, self.mean, self.std_, self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        num_ = x - self.mean
        den_ = self.std_ * np.sqrt(2)
        return 0.5 * (1 + erf(num_ / den_))

    def stats(self) -> Dict[str, Any]:
        mean_, std_ = self.mean, self.std_
        return {'mean': mean_,
                'median': mean_,
                'mode': mean_,
                'variance': std_**2}


def _gaussian(x: np.ndarray,
              amplitude: Optional[float] = 1.,
              mu: Optional[float] = 0.,
              sigma: Optional[float] = 1.,
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
    exponent_factor = (x - mu)**2 / (2 * sigma**2)

    if normalize:
        normalization_factor = sigma * np.sqrt(2 * np.pi)
        amplitude = 1
    else:
        normalization_factor = 1

    return amplitude * (np.exp(-exponent_factor) / normalization_factor)
