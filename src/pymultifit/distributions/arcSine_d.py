"""Created on Aug 14 02:02:42 2024"""

from .beta_d import BetaDistribution


class ArcSineDistribution(BetaDistribution):

    def __init__(self, normalize: bool = True):
        super().__init__(1, 0.5, 0.5, normalize)

        self.norm = normalize
