"""Created on Dec 04 03:57:18 2024"""

from .foldedHalfNormal_d import FoldedHalfNormalDistribution


class HalfNormalDistribution(FoldedHalfNormalDistribution):
    """A class for half normal distribution."""

    def __init__(self, amplitude: float = 1.0, scale: float = 1.0, normalize: bool = False):
        super().__init__(amplitude=amplitude, mean=0, variance=scale, normalize=normalize)
