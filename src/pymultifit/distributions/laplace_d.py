"""Created on Aug 03 21:12:13 2024"""

from typing import Dict

import numpy as np

from .backend import BaseDistribution, errorHandling as erH
from .utilities import laplace_cdf_, laplace_pdf_


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
        return laplace_pdf_(x, amplitude=self.amplitude, mean=self.mu, diversity=self.b, normalize=self.norm)

    def _cdf(self, x: np.array) -> np.array:
        return laplace_cdf_(x, amplitude=self.amplitude, mean=self.mu, diversity=self.b, normalize=self.norm)

    def stats(self) -> Dict[str, float]:
        mean_, b_ = self.mu, self.b

        return {'mean': mean_,
                'median': mean_,
                'mode': mean_,
                'variance': 2 * b_**2}
