"""Created on Dec 03 17:37:05 2024"""

from typing import Dict

from .backend import errorHandling as erH
from .gamma_d import GammaDistributionSR


class ChiSquareDistribution(GammaDistributionSR):
    """Class for ChiSquare distribution."""

    def __init__(self, amplitude: float = 1., degree_of_freedom: int = 1, normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif not isinstance(degree_of_freedom, int) or degree_of_freedom <= 0:
            raise erH.DegreeOfFreedomError()
        self.dof = degree_of_freedom
        super().__init__(amplitude=amplitude, shape=degree_of_freedom / 2., rate=0.5, normalize=normalize)

    def stats(self) -> Dict[str, float]:
        stat_ = super().stats()
        f1 = 9 * self.dof
        f1 = 1 - (2 / f1)
        f1 = self.dof * f1**3

        stat_['median'] = f1

        return stat_
