"""Created on Dec 11 20:40:15 2024"""

from typing import Dict

import numpy as np

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import uniform_cdf_, uniform_pdf_


class UniformDistribution(BaseDistribution):
    """Class for Uniform Distribution."""

    def __init__(self, amplitude: float = 1.0, low: float = 0.0, high: float = 1.0, normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        self.amplitude = 1 if normalize else amplitude
        self.low = low
        self.high = high

        self.norm = normalize

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return uniform_pdf_(x=x, amplitude=self.amplitude, low=self.low, high=self.high, normalize=self.norm)

    def _cdf(self, x: np.array) -> np.array:
        return uniform_cdf_(x=x, amplitude=self.amplitude, low=self.low, high=self.high, normalize=self.norm)

    def stats(self) -> Dict[str, float]:
        low, high = self.low, self.low + self.high

        if low == high:
            return {'mean': np.nan,
                    'median': np.nan,
                    'variance': np.nan}

        mean_ = 0.5 * (low + high)
        median_ = mean_
        variance_ = (1 / 12.) * (high - low)**2

        return {'mean': mean_,
                'median': median_,
                'variance': variance_}
