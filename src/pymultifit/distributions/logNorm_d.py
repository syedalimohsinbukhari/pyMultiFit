"""Created on Aug 03 21:02:45 2024"""

from typing import Dict

import numpy as np
from scipy.special import erf

from .backend import BaseDistribution
from .utilities import log_normal_


class LogNormalDistribution(BaseDistribution):
    """Class for Log-Normal distribution."""

    def __init__(self, amplitude: float = 1., mean: float = 0., standard_deviation: float = 1., normalize: bool = False):
        self.amplitude = 1. if normalize else amplitude
        self.mean = mean
        self.std_ = standard_deviation

        self.norm = normalize

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return log_normal_(x, amplitude=self.amplitude, mean=self.mean, standard_deviation=self.std_, normalize=self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        num_ = np.log(x) - self.mean
        den_ = self.std_ * np.sqrt(2)
        return 0.5 * (1 + erf(num_ / den_))

    def stats(self) -> Dict[str, float]:
        mean_ = np.exp(self.mean + (self.std_**2 / 2))
        median_ = np.exp(self.mean)
        mode_ = np.exp(self.mean - self.std_**2)
        variance_ = (np.exp(self.std_**2) - 1) * np.exp(2 * self.mean + self.std_**2)

        return {'mean': mean_,
                'median': median_,
                'mode': mode_,
                'variance': variance_}
