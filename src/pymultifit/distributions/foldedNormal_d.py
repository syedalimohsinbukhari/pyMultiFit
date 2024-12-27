"""Created on Dec 04 03:42:42 2024"""

from typing import Any, Dict

import numpy as np
from scipy.special import erf

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import folded_normal_cdf_, folded_normal_pdf_


class FoldedNormalDistribution(BaseDistribution):
    """Class for FoldedNormal distribution."""

    def __init__(self, amplitude: float = 1.0, mean: float = 0.0, sigma: float = 1., normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif sigma <= 0:
            raise erH.NegativeStandardDeviationError()
        self.amplitude = 1. if normalize else amplitude
        self.mean = mean
        self.sigma = sigma

        self.c = abs(mean) / sigma
        self.var_ = sigma**2
        self.norm = normalize

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return folded_normal_pdf_(x=x, amplitude=self.amplitude, mu=self.mean, variance=self.var_, normalize=self.norm)

    def cdf(self, x: np.array) -> np.array:
        return folded_normal_cdf_(x=x, amplitude=self.amplitude, mu=self.mean, variance=self.var_, normalize=self.norm)

    def stats(self) -> Dict[str, Any]:
        mean_, std_ = self.mean, np.sqrt(self.var_)

        f1 = std_ * np.sqrt(2 / np.pi) * np.exp(-mean_**2 / (2 * std_**2))
        f2 = mean_ * erf(mean_ / (np.sqrt(2 * np.pi)))

        mu_y = f1 + f2
        var_y = mean_**2 + std_**2 - mu_y**2

        return {'mean': mu_y,
                'variance': var_y}
