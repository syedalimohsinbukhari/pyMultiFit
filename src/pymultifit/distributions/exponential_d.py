"""Created on Nov 30 10:49:49 2024"""

from typing import Any, Dict

import numpy as np

from .backend import BaseDistribution


class ExponentialDistribution(BaseDistribution):
    """Class for Exponential distribution."""

    def __init__(self, amplitude: float = 1., scale: float = 1., normalize: bool = False):
        self.amplitude = 1. if normalize else amplitude
        self.scale = scale

        self.norm = normalize

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return exponential_(x, amplitude=self.amplitude, scale=self.scale, normalize=self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return np.where(x < 0, 0.0, 1 - np.exp(-self.scale * x))

    def stats(self) -> Dict[str, Any]:
        return {'mean': 1 / self.scale,
                'median': np.log(2) / self.scale,
                'mode': 0,
                'variance': 1 / self.scale**2}


def exponential_(x: np.ndarray,
                 amplitude: float = 1., scale: float = 1.,
                 normalize: bool = False) -> np.ndarray:
    """
    Compute the Exponential distribution probability density function (PDF).

    The Exponential PDF is given by:
    f(x; lambda) = lambda * exp(-lambda * x), for x >= 0, otherwise 0.

    Parameters
    ----------
    x : np.ndarray
        The input values at which to evaluate the Exponential PDF. Should be non-negative.
    amplitude : float
        The amplitude of the exponential distribution. Defaults to 1.
    scale : float
        The scale parameter (lambda) of the exponential distribution. Defaults to 1.
    normalize : bool
        If True, the function normalizes the PDF. Otherwise, the amplitude scales the PDF. Defaults to True.

    Returns
    -------
    np.ndarray
        The probability density function values for the input values. For values of `x < 0`, the PDF is 0.

    Notes
    -----
    The input `x` should be a NumPy array. If `x` is a scalar, it will be treated as a single-element array.
    """
    if np.any(x <= 0):
        raise ValueError("x must be positive for the log-normal distribution.")

    if normalize:
        amplitude = 1

    pdf = np.where(x >= 0, amplitude * scale * np.exp(-scale * x), 0.0)

    return pdf
