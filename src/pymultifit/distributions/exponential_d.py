"""Created on Nov 30 10:49:49 2024"""

from typing import Dict

import numpy as np

from .backend.errorHandling import NegativeAmplitudeError, NegativeScaleError
from .gamma_d import GammaDistributionSR


class ExponentialDistribution(GammaDistributionSR):
    """Class for Exponential distribution."""

    def __init__(self, amplitude: float = 1., scale: float = 1., normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise NegativeAmplitudeError()
        elif scale <= 0:
            raise NegativeScaleError()
        self.scale = scale
        super().__init__(amplitude=amplitude, shape=1., rate=scale, normalize=normalize)

    def stats(self) -> Dict[str, float]:
        stats_ = super().stats()
        stats_['median'] = np.log(2) / self.scale

        return stats_
