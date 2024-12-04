"""Created on Dec 03 17:37:05 2024"""

from .gamma_d import GammaDistributionSS


class ChiSquareDistribution(GammaDistributionSS):
    """Class for chi-squared distribution."""

    def __init__(self, amplitude: float = 1., degree_of_freedom: float = 1., normalize: bool = False):
        super().__init__(amplitude=amplitude, shape=degree_of_freedom / 2., scale=2., normalize=normalize)
