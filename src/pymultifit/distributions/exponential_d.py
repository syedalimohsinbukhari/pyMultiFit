"""Created on Nov 30 10:49:49 2024"""

from .gamma_d import GammaDistributionSR


class ExponentialDistribution(GammaDistributionSR):
    """Class for Exponential distribution."""

    def __init__(self, amplitude: float = 1., scale: float = 1., normalize: bool = False):
        super().__init__(amplitude=amplitude, shape=1., rate=scale, normalize=normalize)
