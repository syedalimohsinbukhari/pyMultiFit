"""Created on Aug 14 00:45:37 2024"""

from typing import Dict

import numpy as np
from scipy.special import betaincinv

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import beta_cdf_, beta_pdf_


class BetaDistribution(BaseDistribution):
    """Class for Beta distribution."""

    def __init__(self,
                 amplitude: float = 1.0, alpha: float = 1.0, beta: float = 1.0,
                 loc: float = 0.0, scale: float = 1.0, normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif alpha <= 0:
            raise erH.NegativeAlphaError()
        elif beta <= 0:
            raise erH.NegativeBetaError()
        self.amplitude = 1. if normalize else amplitude
        self.alpha = alpha
        self.beta = beta
        self.loc = loc
        self.scale = scale

        self.norm = normalize

    def pdf(self, x: np.ndarray) -> np.ndarray:
        if self.scale > 0:
            return beta_pdf_(x=x, amplitude=self.amplitude, alpha=self.alpha, beta=self.beta, loc=self.loc, scale=self.scale, normalize=self.norm)
        else:
            return np.full(shape=x.shape, fill_value=np.nan)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        if self.scale > 0:
            return beta_cdf_(x=x, alpha=self.alpha, beta=self.beta, loc=self.loc, scale=self.scale)
        else:
            return np.full(shape=x.shape, fill_value=np.nan)

    # def logpdf(self, x: np.array) -> np.array:
    #     return beta_logpdf_(x=x, alpha=self.alpha, beta=self.beta, loc=self.loc, scale=self.scale, normalize=self.norm)

    def stats(self) -> Dict[str, float]:
        a, b = self.alpha, self.beta

        mean_ = a / (a + b)
        median_ = betaincinv(a, b, 0.5)
        mode_ = []
        if np.logical_and(a > 1, b > 1):
            mode_ = (a - 1) / (a + b - 2)

        variance_ = (a * b) / ((a + b)**2 * (a + b + 1))

        return {'mean': mean_,
                'median': median_,
                'mode': mode_,
                'variance': variance_}
