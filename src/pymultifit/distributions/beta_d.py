"""Created on Aug 14 00:45:37 2024"""

from typing import Dict

import numpy as np
from scipy.special import betainc, betaincinv

from .backend import BaseDistribution, errorHandling as erH
from .utilities import beta_


class BetaDistribution(BaseDistribution):
    """Class for Beta distribution."""

    def __init__(self, amplitude: float = 1., alpha: float = 1., beta: float = 1., normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif alpha <= 0:
            raise erH.NegativeAlphaError()
        elif beta <= 0:
            raise erH.NegativeBetaError()
        self.amplitude = 1. if normalize else amplitude
        self.alpha = alpha
        self.beta = beta

        self.norm = normalize

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return beta_(x, amplitude=self.amplitude, alpha=self.alpha, beta=self.beta, normalize=self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x, dtype=float)
        mask_ = np.logical_and(x > 0, x < 1)
        y[mask_] = self._pdf(x[mask_])

        # hack to match beta distribution at x = 0 and x = 1
        y[np.logical_or(x == 0, x == 1)] = np.inf
        return y

    def cdf(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x, dtype=float)
        mask_ = np.logical_and(x > 0, x < 1)
        y[mask_] = betainc(self.alpha, self.beta, x[mask_])

        # hack to match scipy beta cdf at x >= 1
        y[x >= 1] = 1
        return y

    def stats(self) -> Dict[str, float]:
        a, b = self.alpha, self.beta

        mean_ = a / (a + b)
        median_ = betaincinv(a, b, 0.5)
        mode_ = []
        if np.logical_and(a > 1, b > 1):
            mode_ = (a - 1) / (a + b - 2)

        variance_ = (a * b) / ((a + b)**2 * (a + b + 1))

        return {'mean': mean_,
                'median': median_,
                'mode': mode_,
                'variance': variance_}
