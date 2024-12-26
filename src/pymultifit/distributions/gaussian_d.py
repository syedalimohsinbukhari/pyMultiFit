"""Created on Aug 03 20:07:50 2024"""

from typing import Dict

import numpy as np

from .backend import BaseDistribution, errorHandling as erH
from .utilities import gaussian_cdf_, gaussian_pdf_


class GaussianDistribution(BaseDistribution):
    """Class for Gaussian distribution."""

    def __init__(self, amplitude: float = 1.0, mean: float = 0., std: float = 1., normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif std <= 0:
            raise erH.NegativeStandardDeviationError()

        self.amplitude = 1. if normalize else amplitude
        self.mean = mean
        self.std_ = std
        self.norm = normalize

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return gaussian_pdf_(x, amplitude=self.amplitude, mean=self.mean, std=self.std_, normalize=self.norm)

    def _cdf(self, x: np.array) -> np.array:
        return gaussian_cdf_(x, amplitude=self.amplitude, mean=self.mean, std=self.std_, normalize=self.norm)

    def stats(self) -> Dict[str, float]:
        mean_, std_ = self.mean, self.std_
        return {'mean': mean_,
                'median': mean_,
                'mode': mean_,
                'variance': std_**2}
