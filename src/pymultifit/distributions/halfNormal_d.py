"""Created on Dec 04 03:57:18 2024"""

import numpy as np

from . import folded_half_normal_, FoldedHalfNormalDistribution


class HalfNormalDistribution(FoldedHalfNormalDistribution):
    """A class for half folded normal distribution."""

    def __init__(self, amplitude: float = 1.0, scale: float = 1.0, normalize: bool = False):
        super().__init__(amplitude=amplitude, mean=0, standard_deviation=scale, normalize=normalize)


def half_normal_(x: np.ndarray,
                 amplitude: float = 1.0, scale: float = 1.0, normalize: bool = False) -> np.ndarray:
    """
    Compute the half-normal distribution.

    The half-normal distribution is a special case of the folded half-normal distribution where the mean (`mu`)  is set to 0.
    It describes the absolute value of a standard normal variable scaled by a specified factor.

    Parameters
    ----------
    x : np.ndarray
        The input values at which to evaluate the half-normal distribution.
    amplitude : float, optional
        The amplitude of the distribution. Defaults to 1. Ignored if `normalize` is set to True.
    scale : float, optional
        The scale parameter (`sigma`) of the distribution, corresponding to the standard deviation of the original normal distribution.
        Defaults to 1.0.
    normalize : bool, optional
        If True, the distribution will be normalized so that the PDF is at most 1. Defaults to False.

    Returns
    -------
    np.ndarray
        The computed half-normal distribution values for the input `x`.

    Notes
    -----
    The half-normal distribution is defined as the absolute value of a normal distribution with zero mean and specified standard deviation.
    The probability density function (PDF) is:

    .. math::
        f(x; \\sigma) = \\sqrt{2/\\pi} \\frac{1}{\\sigma} e^{-x^2 / (2\\sigma^2)}

    where `x >= 0` and `\\sigma` is the scale parameter.

    The half-normal distribution is a special case of the folded half-normal distribution with `mu = 0`.
    """
    return folded_half_normal_(x, amplitude=amplitude, mu=0, sigma=scale, normalize=normalize)
