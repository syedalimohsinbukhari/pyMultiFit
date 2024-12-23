r"""
:strong:`pymultifit.distributions.arcSine_d.`
"""

from typing import Any, Dict

import numpy as np

from .backend import BaseDistribution
from .utilities import arc_sine_cdf_, arc_sine_logpdf_, arc_sine_pdf_


class ArcSineDistribution(BaseDistribution):
    """Class for ArcSine distribution."""

    def __init__(self, amplitude: float = 1., loc: float = 0.0, scale: float = 1.0, normalize: bool = False):
        self.amplitude = 1 if normalize else amplitude
        self.loc = loc
        self.scale = scale

        self.norm = normalize

    def _pdf(self, x: np.array) -> np.array:
        return arc_sine_pdf_(x=x, amplitude=self.amplitude, loc=self.loc, scale=self.scale, normalize=self.norm)

    def _cdf(self, x: np.array) -> np.array:
        return arc_sine_cdf_(x=x, loc=self.loc, scale=self.scale)

    def logpdf(self, x: np.array) -> np.array:
        return arc_sine_logpdf_(x=x, amplitude=self.amplitude, loc=self.loc, scale=self.scale, normalize=self.norm)

    def stats(self) -> Dict[str, Any]:
        mean_ = 0.5
        median_ = 0.5
        variance_ = 1 / 8

        return {'mean': mean_,
                'median': median_,
                'model': [],
                'variance': variance_}
