"""Created on Nov 30 10:49:49 2024"""

from .backend.errorHandling import NegativeAmplitudeError, NegativeScaleError
from .gamma_d import GammaDistributionSR


class ExponentialDistribution(GammaDistributionSR):
    """Class for Exponential distribution."""

    def __init__(self, amplitude: float = 1., scale: float = 1., normalize: bool = False):
        if amplitude < 0:
            raise NegativeAmplitudeError()
        elif scale < 0:
            raise NegativeScaleError()
        super().__init__(amplitude=amplitude, shape=1., rate=scale, normalize=normalize)
