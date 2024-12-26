"""Created on Aug 03 21:35:28 2024"""

from typing import Dict

import numpy as np

from .backend import BaseDistribution
from .utilities_d import skew_normal_cdf_, skew_normal_pdf_


class SkewNormalDistribution(BaseDistribution):
    """Class for SkewNormal distribution."""

    def __init__(self, amplitude: float = 1.0, shape: float = 1., location: float = 0., scale: float = 1., normalize: bool = False):
        self.amplitude = 1 if normalize else amplitude
        self.shape = shape
        self.location = location
        self.scale = scale

        self.norm = normalize

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return skew_normal_pdf_(x=x, amplitude=self.amplitude, shape=self.shape, loc=self.location, scale=self.scale, normalize=self.norm)

    def _cdf(self, x: np.array) -> np.array:
        return skew_normal_cdf_(x=x, amplitude=self.amplitude, shape=self.shape, loc=self.location, scale=self.scale, normalize=self.norm)

    def stats(self) -> Dict[str, float]:
        alpha, omega, epsilon = self.shape, self.scale, self.location
        delta = alpha / np.sqrt(1 + alpha**2)

        mean = epsilon + omega * delta * np.sqrt(2 / np.pi)

        def _m0(alpha_):
            m0 = np.sqrt(2 / np.pi) * delta
            m0 -= ((1 - np.pi / 4) * (np.sqrt(2 / np.pi) * delta)**3) / (1 - (2 / np.pi) * delta**2)
            m0 -= (2 * np.pi / abs(alpha_)) * np.exp(-(2 * np.pi / abs(alpha_))) * np.sign(alpha_)
            return m0

        mode = epsilon + omega * _m0(alpha)
        variance = omega**2 * (1 - (2 * delta**2 / np.pi))

        return {'mean': mean, 'mode': mode, 'variance': variance}
