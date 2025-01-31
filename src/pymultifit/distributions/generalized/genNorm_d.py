"""Created on Jan 29 15:42:23 2025"""

from typing import Dict

import numpy as np
from scipy.special import gamma

from ..backend import BaseDistribution
from ..utilities_d import gen_sym_normal_pdf_, gen_sym_normal_cdf_


class SymmetricGeneralizedNormalDistribution(BaseDistribution):

    def __init__(self, amplitude: float = 1.0, shape: float = 1.0, loc: float = 0.0, scale: float = 1.0,
                 normalize: bool = False):
        self.amplitude = 1. if normalize else amplitude
        self.mu = loc
        self.alpha = scale
        self.beta = shape

        self.norm = normalize

    @classmethod
    def scipy_like(cls, beta, loc: float = 0.0, scale: float = 1.0):
        instance = cls(shape=beta, loc=loc, scale=scale, normalize=True)
        return instance

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return gen_sym_normal_pdf_(x, amplitude=self.amplitude, loc=self.mu, scale=self.alpha, shape=self.beta,
                                   normalize=self.norm)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return gen_sym_normal_cdf_(x, amplitude=self.amplitude, loc=self.mu, scale=self.alpha, shape=self.beta,
                                   normalize=self.norm)

    def stats(self) -> Dict[str, float]:
        mean_ = self.mu
        median_ = self.mu
        mode_ = self.mu
        variance_ = self.alpha**2 * gamma(3 / self.beta)
        variance_ /= gamma(1 / self.beta)

        return {'mean': mean_,
                'median': median_,
                'mode': mode_,
                'variance': variance_,
                'std': np.sqrt(variance_)}
