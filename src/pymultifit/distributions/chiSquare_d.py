"""Created on Dec 03 17:37:05 2024"""

from .backend import errorHandling as erH
from .gamma_d import GammaDistributionSS


class ChiSquareDistribution(GammaDistributionSS):
    """Class for chi-squared distribution."""

    def __init__(self, amplitude: float = 1., degree_of_freedom: float = 1., normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif not isinstance(degree_of_freedom, int) or degree_of_freedom <= 0:
            raise erH.DegreeOfFreedomError()
        super().__init__(amplitude=amplitude, shape=degree_of_freedom / 2., scale=2., normalize=normalize)
