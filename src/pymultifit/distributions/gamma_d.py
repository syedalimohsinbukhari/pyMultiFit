"""Created on Aug 14 01:28:13 2024"""

from typing import Dict

import numpy as np

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import gamma_sr_cdf_, gamma_sr_pdf_


class GammaDistributionSR(BaseDistribution):
    """Class for Gamma distribution with shape and rate parameters."""

    def __init__(self,
                 amplitude: float = 1.0, shape: float = 1.0, rate: float = 1.0,
                 loc: float = 0.0, normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif shape <= 0:
            raise erH.NegativeShapeError()
        elif rate <= 0:
            raise erH.NegativeRateError()
        self.amplitude = 1. if normalize else amplitude
        self.shape = shape
        self.rate = rate
        self.loc = loc

        self.norm = normalize

    def _pdf(self, x: np.array) -> np.array:
        return gamma_sr_pdf_(x, amplitude=self.amplitude, alpha=self.shape, lambda_=self.rate, loc=self.loc, normalize=self.norm)

    def _cdf(self, x: np.array) -> np.array:
        return gamma_sr_cdf_(x, amplitude=self.amplitude, alpha=self.shape, lambda_=self.rate, normalize=self.norm)

    def stats(self) -> Dict[str, float]:
        a, b = self.shape, self.rate

        mean_ = a / b
        mode_ = (a - 1) / b if a >= 1 else 0
        variance_ = a / b**2

        return {'mean': mean_,
                'mode': mode_,
                'variance': variance_}


class GammaDistributionSS(GammaDistributionSR):
    """Class for Gamma distribution with shape and scale parameters."""

    def __init__(self, amplitude: float = 1.0, shape: float = 1.0, scale: float = 1.0, loc: float = 0.0, normalize: bool = False):
        self.scale = scale
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif shape <= 0:
            raise erH.NegativeShapeError()
        elif scale <= 0:
            raise erH.NegativeScaleError()
        super().__init__(amplitude=amplitude, shape=shape, rate=1 / self.scale, loc=loc, normalize=normalize)
