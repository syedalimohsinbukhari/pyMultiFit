"""Created on Nov 30 10:49:49 2024"""

from typing import Dict

import numpy as np

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import exponential_cdf_, exponential_pdf_


class ExponentialDistribution(BaseDistribution):
    """Class for Exponential distribution."""

    def __init__(self, amplitude: float = 1.0, scale: float = 1.0, loc: float = 0.0, normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif scale <= 0:
            raise erH.NegativeScaleError()
        self.amplitude = 1 if normalize else amplitude
        self.scale = scale
        self.loc = loc

        self.norm = normalize

    def _pdf(self, x: np.array) -> np.array:
        return exponential_pdf_(x=x, amplitude=self.amplitude, scale=self.scale, loc=self.loc, normalize=self.norm)

    def _cdf(self, x: np.array) -> np.array:
        return exponential_cdf_(x=x, amplitude=self.amplitude, scale=self.scale, loc=self.loc, normalize=self.norm)

    def stats(self) -> Dict[str, float]:
        lambda_ = self.scale

        return {'mean': 1 / lambda_ + self.loc,
                'median': np.log(2) / lambda_ + self.loc,
                'mode': 0 + self.loc,
                'variance': 1 / lambda_**2}
