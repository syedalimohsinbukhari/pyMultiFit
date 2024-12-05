"""Created on Aug 03 20:07:50 2024"""

from typing import Dict

import numpy as np
from scipy.special import erf

from .backend import BaseDistribution
from .utilities import gaussian_


class GaussianDistribution(BaseDistribution):
    """Class for Gaussian distribution."""

    def __init__(self, amplitude: float = 1.0, mean: float = 0., standard_deviation: float = 1., normalize: bool = False):
        self.amplitude = 1. if normalize else amplitude
        self.mean = mean
        self.std_ = standard_deviation

        self.norm = normalize

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return gaussian_(x, amplitude=self.amplitude, mu=self.mean, sigma=self.std_, normalize=self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        num_ = x - self.mean
        den_ = self.std_ * np.sqrt(2)
        return 0.5 * (1 + erf(num_ / den_))

    def stats(self) -> Dict[str, float]:
        mean_, std_ = self.mean, self.std_
        return {'mean': mean_,
                'median': mean_,
                'mode': mean_,
                'variance': std_**2}
