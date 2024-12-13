"""Created on Dec 11 20:40:15 2024"""

from typing import Dict

import numpy as np

from .backend import BaseDistribution, errorHandling as erH


class UniformDistribution(BaseDistribution):
    def __init__(self, amplitude: float = 1.0, low: float = 0.0, high: float = 1.0, normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif high < low:
            raise erH.InvalidUniformParameters()
        self.amplitude = 1 if normalize else amplitude
        self.low = low
        self.high = high

        self.norm = normalize

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return uniform_(x=x, amplitude=self.amplitude, low=self.low, high=self.high, normalize=self.norm)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        cdf_values = np.zeros_like(x, dtype=float)
        within_bounds = (x >= self.low) & (x <= self.high)
        cdf_values[within_bounds] = (x[within_bounds] - self.low) / (self.high - self.low)  # Compute CDF for bounds
        cdf_values[x > self.high] = 1  # Set values for x > high to 1
        return cdf_values

    def stats(self) -> Dict[str, float]:
        mean_ = 0.5 * (self.low + self.high)
        median_ = mean_
        variance_ = (1 / 12.) * (self.high - self.low)**2

        return {'mean': mean_,
                'median': median_,
                'variance': variance_}


def uniform_(x: np.ndarray,
             amplitude: float = 1.0, low: float = 0.0, high: float = 1.0,
             normalize: bool = False) -> np.ndarray:
    """
    Compute the Uniform distribution probability density function (PDF).

    The Uniform PDF is given by:
    f(x) = amplitude / (high - low) for x ∈ [low, high]
           0 otherwise

    Parameters
    ----------
    x : np.ndarray
        The input values at which to evaluate the Uniform PDF.
    low : float
        The lower bound of the Uniform distribution. Defaults to 0.
    high : float
        The upper bound of the Uniform distribution. Defaults to 1.
    amplitude : float
        The amplitude of the Uniform distribution. Defaults to 1.
    normalize : bool
        If True, the function returns the normalized PDF (amplitude = 1 / (high - low)).
        Defaults to False.

    Returns
    -------
    np.ndarray
        The probability density function values for the input values.

    Notes
    -----
    - The input `x` should be a NumPy array. If `x` is a scalar, it will be treated as a single-element array.
    - If `normalize=True`, the amplitude is overridden to ensure the PDF integrates to 1.
    """
    if low >= high:
        raise ValueError("`low` must be less than `high`.")

    if normalize:
        amplitude = 1.0

    # Compute the PDF values
    pdf_values = np.where((x >= low) & (x <= high), amplitude / (high - low), 0.0)

    return pdf_values
