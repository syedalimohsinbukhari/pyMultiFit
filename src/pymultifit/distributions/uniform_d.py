"""Created on Dec 11 20:40:15 2024"""

from typing import Dict

import numpy as np

from .backend import BaseDistribution, errorHandling as erH
from .utilities import uniform_


class UniformDistribution(BaseDistribution):
    """Class for Uniform Distribution."""

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
        cdf_values[x > self.high] = 1
        return cdf_values

    def stats(self) -> Dict[str, float]:
        mean_ = 0.5 * (self.low + self.high)
        median_ = mean_
        variance_ = (1 / 12.) * (self.high - self.low)**2

        return {'mean': mean_,
                'median': median_,
                'variance': variance_}
