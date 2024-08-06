"""Created on Aug 03 21:12:13 2024"""

from typing import Optional

import numpy as np

from ._backend import BaseDistribution


class LaplaceDistribution(BaseDistribution):
    """Class for Laplace distribution."""

    def __init__(self,
                 mean: Optional[float] = 0,
                 diversity: Optional[float] = 1,
                 un_normalized: bool = False):
        self.mu = mean
        self.b = diversity
        self.un_norm = un_normalized

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return _laplace(x, self.mu, self.b, self.un_norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        def _cdf1(x_):
            return 0.5 * np.exp((x_ - self.mu) / self.b)

        def _cdf2(x_):
            return 1 - 0.5 * np.exp(-(x_ - self.mu) / self.b)

        return np.piecewise(x, [x <= self.mu, x > self.mu], [_cdf1, _cdf2])


def _laplace(x: np.ndarray, mu: Optional[float] = 0, b: Optional[float] = 1,
             un_normalized: bool = False) -> np.ndarray:
    exponent = (abs(x - mu)) / b
    exponent = np.exp(-exponent)

    normalization = 2 * b if not un_normalized else 1

    return exponent / normalization
