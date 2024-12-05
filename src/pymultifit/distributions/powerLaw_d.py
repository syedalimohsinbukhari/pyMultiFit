"""Created on Nov 10 00:12:52 2024"""

from typing import Any, Dict

import numpy as np

from .backend import BaseDistribution
from .utilities import power_law_


class PowerLawDistribution(BaseDistribution):

    def __init__(self, amplitude: float = 1.0, alpha: float = -1, normalize: bool = False):
        self.amplitude = amplitude
        self.alpha = alpha

        self.norm = normalize

    def _pdf(self, x: np.ndarray):
        return power_law_(x, amplitude=self.amplitude, alpha=self.alpha, normalize=self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass

    def stats(self) -> Dict[str, Any]:
        pass
