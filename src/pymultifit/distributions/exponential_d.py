"""Created on Nov 30 10:49:49 2024"""

import numpy as np

from .gamma_d import gamma_sr_, GammaDistributionSR


class ExponentialDistribution(GammaDistributionSR):
    """Class for Exponential distribution."""

    def __init__(self, amplitude: float = 1., scale: float = 1., normalize: bool = False):
        super().__init__(amplitude=amplitude, shape=1., rate=scale, normalize=normalize)


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
    return gamma_sr_(x, amplitude=amplitude, shape=1., rate=scale, normalize=normalize)
