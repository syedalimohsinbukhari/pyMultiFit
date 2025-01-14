"""Created on Aug 10 23:37:54 2024"""

import numpy as np

from .backend import BaseDistribution, line


class Line(BaseDistribution):

    def __init__(self, slope: float = 1.0, intercept: float = 1.0, normalize: bool = False):
        self.slope = slope
        self.intercept = intercept

        self.norm = normalize

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return line(x, slope=self.slope, intercept=self.intercept)

    def cdf(self, x: np.ndarray):
        pass

    def stats(self):
        pass
