"""Created on Aug 14 01:28:13 2024"""

from typing import Dict

import numpy as np
from scipy.special import gamma, gammainc

from .backend import BaseDistribution


class GammaDistributionSR(BaseDistribution):
    """Class for Gamma distribution."""

    def __init__(self, amplitude: float = 1., shape: float = 1., rate: float = 1., normalize: bool = False):
        self.amplitude = 1. if normalize else amplitude
        self.shape = shape
        self.rate = rate

        self.norm = normalize

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return gamma_sr_(x, amplitude=self.amplitude, shape=self.shape, rate=self.rate, normalize=self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return gammainc(self.shape, self.rate * x)

    def stats(self) -> Dict[str, float]:
        a, b = self.shape, self.rate

        mean_ = a / b
        mode_ = (a - 1) / b if a >= 1 else 0
        variance_ = a / b**2

        return {'mean': mean_,
                'mode': mode_,
                'variance': variance_}


class GammaDistributionSS(GammaDistributionSR):
    def __init__(self, amplitude: float = 1., shape: float = 1., scale: float = 1., normalize: bool = False):
        super().__init__(amplitude=amplitude, shape=shape, rate=1 / scale, normalize=normalize)


def gamma_sr_(x: np.ndarray,
              amplitude: float = 1., shape: float = 1., rate: float = 1.,
              normalize: bool = False) -> np.ndarray:
    """
    Computes the Gamma distribution PDF for given parameters.

    Parameters
    ----------
    x : np.ndarray
        Values at which to evaluate the PDF.
    amplitude : float
        The scaling factor for the distribution. Defaults to 1.
    shape : float
        The shape parameter of the Gamma distribution. Defaults to 1.
    rate : float
        The rate parameter of the Gamma distribution. Defaults to 1.
    normalize : bool
        Whether to normalize the distribution (i.e., set amplitude to 1). Defaults to True.

    Returns
    -------
    np.ndarray
        The probability density function evaluated at `x`.
    """
    numerator = x**(shape - 1) * np.exp(-rate * x)

    if normalize:
        normalization_factor = gamma(shape) / rate**shape
        amplitude = 1
    else:
        normalization_factor = 1

    return amplitude * (numerator / normalization_factor)


def gamma_ss_(x: np.ndarray,
              amplitude: float = 1., shape: float = 1., scale: float = 1.,
              normalize: bool = False) -> np.ndarray:
    """
    Compute the Gamma distribution using the shape and scale parameterization.

    This function wraps the Gamma distribution parameterized by `shape` and `rate` and provides an interface for  `shape` and `scale`.
    The relationship between scale and rate is defined as:
    .. math::
        \text{rate} = \frac{1}{\text{scale}}

    Parameters
    ----------
    x : np.ndarray
        The input values at which to evaluate the Gamma distribution.
    amplitude : float, optional
        The amplitude of the distribution. Defaults to 1. Ignored if `normalize` is set to True.
    shape : float, optional
        The shape parameter (`k`) of the Gamma distribution. Defaults to 1.
    scale : float, optional
        The scale parameter (`\theta`) of the Gamma distribution. Defaults to 1. The rate parameter is computed as `1 / scale`.
    normalize : bool, optional
        If True, the distribution will be normalized so that the PDF is at most 1. Defaults to False.

    Returns
    -------
    np.ndarray
        The computed Gamma distribution values for the input `x`.

    Notes
    -----
    The Gamma distribution parameterized by shape and scale is defined as:
    .. math::
        f(x; k, \theta) = \frac{x^{k-1} e^{-x / \theta}}{\theta^k \Gamma(k)}

    where:
        - `k` is the shape parameter
        - `\theta` is the scale parameter
        - `\Gamma(k)` is the Gamma function.

    This function computes the equivalent Gamma distribution using the relationship between scale (`\theta`) and rate (`\beta`) where:
    .. math::
        \beta = \frac{1}{\theta}.
    """
    return gamma_sr_(x, amplitude=amplitude, shape=shape, rate=1 / scale, normalize=normalize)
