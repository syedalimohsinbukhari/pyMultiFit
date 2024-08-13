"""Created on Aug 14 00:45:37 2024"""

from typing import Any, Dict, Optional

import numpy as np
from scipy.special import beta as beta_, betainc

from ._backend import BaseDistribution


class BetaDistribution(BaseDistribution):

    def __init__(self,
                 amplitude: Optional[float] = 1.,
                 alpha: Optional[float] = 1.,
                 beta: Optional[float] = 1.,
                 normalize: bool = True):
        self.alpha = alpha
        self.beta = beta
        self.norm = normalize

        self.amplitude = 1 if normalize else amplitude

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return _beta(x, self.amplitude, self.alpha, self.beta, self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return betainc(self.alpha, self.beta, x)

    def stats(self) -> Dict[str, Any]:
        a, b = self.alpha, self.beta

        mean_ = a / (a + b)

        variance_ = (a * b)
        variance_ /= (a + b)**2 * (a + b + 1)

        return {'mean': mean_,
                'variance': variance_}


def _beta(x: np.ndarray,
          amplitude: Optional[float] = 1.,
          alpha: Optional[float] = 1.,
          beta: Optional[float] = 1.,
          normalize: bool = True):
    numerator = x**(alpha - 1) * (1 - x)**(beta - 1)

    if normalize:
        normalization_factor = beta_(alpha, beta)
        amplitude = 1
    else:
        normalization_factor = 1

    return amplitude * (numerator / normalization_factor)
