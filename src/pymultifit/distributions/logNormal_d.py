"""Created on Aug 03 21:02:45 2024"""

from typing import Dict

import numpy as np

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import log_normal_cdf_, log_normal_pdf_


class LogNormalDistribution(BaseDistribution):
    r"""
    Class for LogNormal distribution.

    .. note::
        .. list-table::
           :header-rows: 1
           :class: centered-table

           * - :obj:`scipy.stats.lognorm`
             - :class:`~pymultifit.distributions.logNormal_d.LogNormalDistribution`
           * - **shape**
             - **std**
           * - **loc**
             - **loc**
           * - **scale**
             - **mean**
    """

    def __init__(self, amplitude: float = 1., mean: float = 0.0, std: float = 1.0, loc: float = 0.0, normalize: bool = False):
        if not normalize and amplitude <= 0:
            raise erH.NegativeAmplitudeError()
        elif std <= 0:
            raise erH.NegativeStandardDeviationError()
        self.amplitude = 1. if normalize else amplitude
        self.mean = np.log(mean)
        self.std = std
        self.loc = loc

        self.norm = normalize

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return log_normal_pdf_(x, amplitude=self.amplitude, mean=self.mean, std=self.std, loc=self.loc, normalize=self.norm)

    def cdf(self, x: np.array) -> np.array:
        return log_normal_cdf_(x, amplitude=self.amplitude, mean=self.mean, std=self.std, loc=self.loc, normalize=self.norm)

    def stats(self) -> Dict[str, float]:
        mean_ = np.exp(self.mean + (self.std**2 / 2))
        median_ = np.exp(self.mean)
        mode_ = np.exp(self.mean - self.std**2)
        variance_ = (np.exp(self.std**2) - 1) * np.exp(2 * self.mean + self.std**2)

        return {'mean': mean_,
                'median': median_,
                'mode': mode_,
                'variance': variance_}
