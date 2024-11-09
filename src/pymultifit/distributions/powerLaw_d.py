"""Created on Nov 10 00:12:52 2024"""

from typing import Any, Dict

import numpy as np

from ._backend import BaseDistribution


class PowerLawDistribution(BaseDistribution):
    
    def __init__(self, alpha):
        self.alpha = alpha
        
        self.norm = True
        self.amplitude = 1.
    
    @classmethod
    def with_amplitude(cls, amplitude: float = 1, alpha: float = -1):
        instance = cls(alpha=alpha)
        instance.amplitude = amplitude
        instance.norm = False
        
        return instance
    
    def _pdf(self, x: np.ndarray):
        return power_law_(x, amplitude=self.amplitude, alpha=self.alpha, normalize=self.norm)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._pdf(x)
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass
    
    def stats(self) -> Dict[str, Any]:
        pass


def power_law_(x: np.ndarray, amplitude: float = 1, alpha: float = -1, normalize: bool = True):
    return amplitude * x**alpha


powerLawWA = PowerLawDistribution.with_amplitude
