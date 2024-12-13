"""Created on Aug 14 01:28:13 2024"""

from typing import Dict

import numpy as np
from scipy.special import gammainc

from .backend import BaseDistribution, errorHandling as erH
from .utilities import gamma_sr_


class GammaDistributionSR(BaseDistribution):
    """Class for Gamma distribution."""

    def __init__(self, amplitude: float = 1., shape: float = 1., rate: float = 1., normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif shape <= 0:
            raise erH.NegativeShapeError()
        elif rate <= 0:
            raise erH.NegativeRateError()
        self.amplitude = 1. if normalize else amplitude
        self.shape = shape
        self.rate = rate

        self.norm = normalize

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return gamma_sr_(x, amplitude=self.amplitude, shape=self.shape, rate=self.rate, normalize=self.norm)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.nan_to_num(x=self._pdf(x), copy=False, nan=0)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return np.nan_to_num(x=gammainc(self.shape, self.rate * x), copy=False, nan=0)

    def stats(self) -> Dict[str, float]:
        a, b = self.shape, self.rate

        mean_ = a / b
        mode_ = (a - 1) / b if a >= 1 else 0
        variance_ = a / b**2

        return {'mean': mean_,
                'mode': mode_,
                'variance': variance_}


class GammaDistributionSS(GammaDistributionSR):
    def __init__(self, amplitude: float = 1., shape: float = 1., scale: float = 1., normalize: bool = False):
        self.scale = scale
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif shape <= 0:
            raise erH.NegativeShapeError()
        elif scale <= 0:
            raise erH.NegativeScaleError()
        super().__init__(amplitude=amplitude, shape=shape, rate=1 / self.scale, normalize=normalize)
