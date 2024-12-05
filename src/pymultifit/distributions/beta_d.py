"""Created on Aug 14 00:45:37 2024"""

from typing import Dict

import numpy as np
from scipy.special import betainc

from .backend import BaseDistribution
from .utilities import beta_


class BetaDistribution(BaseDistribution):
    """Class for Beta distribution."""

    def __init__(self, amplitude: float = 1., alpha: float = 1., beta: float = 1., normalize: bool = False):
        self.amplitude = 1. if normalize else amplitude
        self.alpha = alpha
        self.beta = beta

        self.norm = normalize

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return beta_(x, amplitude=self.amplitude, alpha=self.alpha, beta=self.beta, normalize=self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return betainc(self.alpha, self.beta, x)

    def stats(self) -> Dict[str, float]:
        a, b = self.alpha, self.beta

        mean_ = a / (a + b)
        variance_ = (a * b) / ((a + b)**2 * (a + b + 1))

        return {'mean': mean_,
                'variance': variance_}
