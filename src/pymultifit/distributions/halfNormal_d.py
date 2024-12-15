"""Created on Dec 04 03:57:18 2024"""

from .backend import errorHandling as erH
from .foldedNormal_d import FoldedNormalDistribution


class NormalDistribution(FoldedNormalDistribution):
    """A class for half normal distribution."""

    def __init__(self, amplitude: float = 1.0, scale: float = 1.0, normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif scale <= 0:
            raise erH.NegativeScaleError()
        super().__init__(amplitude=amplitude, mean=0, sigma=scale, normalize=normalize)
