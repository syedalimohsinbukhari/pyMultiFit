"""Created on Aug 03 21:12:13 2024"""

from typing import Dict

import numpy as np

from .backend import BaseDistribution, errorHandling as erH
from .utilities import laplace_


class LaplaceDistribution(BaseDistribution):
    """Class for Laplace distribution."""

    def __init__(self, amplitude: float = 1., mean: float = 0, diversity: float = 1, normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif diversity <= 0:
            raise erH.NegativeScaleError('diversity')
        self.amplitude = 1. if normalize else amplitude
        self.mu = mean
        self.b = diversity

        self.norm = normalize

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return laplace_(x, amplitude=self.amplitude, mean=self.mu, diversity=self.b, normalize=self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        def _cdf1(x_):
            return 0.5 * np.exp((x_ - self.mu) / self.b)

        def _cdf2(x_):
            return 1 - 0.5 * np.exp(-(x_ - self.mu) / self.b)

        # to ensure equality with scipy, had to break down the output with empty array so that sorting is not needed.
        result = np.empty_like(x, dtype=np.float64)

        mask_leq = x <= self.mu
        result[mask_leq] = _cdf1(x[mask_leq])
        result[~mask_leq] = _cdf2(x[~mask_leq])

        return result

    def stats(self) -> Dict[str, float]:
        mean_, b_ = self.mu, self.b

        return {'mean': mean_,
                'median': mean_,
                'mode': mean_,
                'variance': 2 * b_**2}
