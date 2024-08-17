"""Created on Aug 03 20:07:50 2024"""

from typing import Any, Dict

import numpy as np
from scipy.special import erf

from . import oFloat
from ._backend import BaseDistribution


class GaussianDistribution(BaseDistribution):
    """Class for Gaussian distribution."""

    def __init__(self,
                 mean: oFloat = 0.,
                 standard_deviation: oFloat = 1.):
        self.mean = mean
        self.std_ = standard_deviation

        self.norm = True
        self.amplitude = 1.

    @classmethod
    def with_amplitude(cls, amplitude: oFloat = 1., mean: oFloat = 0., standard_deviation: oFloat = 1.):
        """
        Create an instance with a specified amplitude, without normalization.

        Parameters
        ----------
        amplitude : float, optional
            The amplitude to apply to the PDF. Defaults to 1.
        mean : float, optional
            The mean of the normal distribution. Defaults to 0.
        standard_deviation : float, optional
            The standard deviation of the normal distribution. Defaults to 1.

        Returns
        -------
        GaussianDistribution
            An instance of GaussianDistribution with the specified amplitude and parameters.
        """
        instance = cls(mean=mean, standard_deviation=standard_deviation)
        instance.amplitude = amplitude
        instance.norm = False

        return instance

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return _gaussian(x, amplitude=self.amplitude, mu=self.mean, sigma=self.std_, normalize=self.norm)

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
              amplitude: oFloat = 1.,
              mu: oFloat = 0.,
              sigma: oFloat = 1.,
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
