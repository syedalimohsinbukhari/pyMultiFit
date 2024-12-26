"""Created on Aug 14 02:02:42 2024"""

from . import BetaDistribution


class ArcSineDistribution(BetaDistribution):
    """Class for ArcSine distribution."""

    def __init__(self, amplitude: float = 1., loc: float = 0.0, scale: float = 1.0, normalize: bool = False):
        self.amplitude = 1 if normalize else amplitude
        self.loc = loc
        self.scale = scale

        self.norm = normalize
        super().__init__(amplitude=self.amplitude, alpha=0.5, beta=0.5, loc=self.loc, scale=self.scale, normalize=self.norm)
