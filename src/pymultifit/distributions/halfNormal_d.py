"""Created on Dec 04 03:57:18 2024"""

from .backend.errorHandling import NegativeAmplitudeError, NegativeScaleError
from .foldedHalfNormal_d import FoldedHalfNormalDistribution


class HalfNormalDistribution(FoldedHalfNormalDistribution):
    """A class for half normal distribution."""

    def __init__(self, amplitude: float = 1.0, scale: float = 1.0, normalize: bool = False):
        if not normalize and amplitude < 0:
            raise NegativeAmplitudeError()
        elif scale < 0:
            raise NegativeScaleError()
        super().__init__(amplitude=amplitude, mean=0, variance=scale, normalize=normalize)
