"""Created on Nov 02 18:49:12 2025"""

from typing import Dict

import numpy as np

from .backend import BaseDistribution
from .utilities_d import johnsonSU_pdf, johnsonSU_log_pdf_, johnsonSU_cdf_, johnsonSU_log_cdf_
from .. import OneDArray


class JohnsonSUDistribution(BaseDistribution):

    def __init__(
        self,
        amplitude: float = 1.0,
        gamma: float = 0.0,
        delta: float = 1.0,
        xi: float = 0.0,
        lambda_: float = 1.0,
        normalize: bool = False,
    ):
        self.amplitude = 1.0 if normalize else amplitude
        self.gamma = gamma
        self.delta = delta
        self.xi = xi
        self.lambda_ = lambda_

        self.norm = normalize

    @classmethod
    def from_scipy_params(cls, a, b, loc: float = 0.0, scale: float = 1.0) -> "JohnsonSUDistribution":
        return cls(gamma=a, delta=b, xi=loc, lambda_=scale, normalize=True)

    def pdf(self, x: OneDArray) -> OneDArray:
        return johnsonSU_pdf(
            x,
            amplitude=self.amplitude,
            gamma=self.gamma,
            delta=self.delta,
            xi=self.xi,
            lambda_=self.lambda_,
            normalize=self.norm,
        )

    def logpdf(self, x: OneDArray) -> OneDArray:
        return johnsonSU_log_pdf_(
            x,
            amplitude=self.amplitude,
            gamma=self.gamma,
            delta=self.delta,
            xi=self.xi,
            lambda_=self.lambda_,
            normalize=self.norm,
        )

    def cdf(self, x: OneDArray) -> OneDArray:
        return johnsonSU_cdf_(
            x,
            amplitude=self.amplitude,
            gamma=self.gamma,
            delta=self.delta,
            xi=self.xi,
            lambda_=self.lambda_,
            normalize=self.norm,
        )

    def logcdf(self, x: OneDArray) -> OneDArray:
        return johnsonSU_log_cdf_(
            x,
            amplitude=self.amplitude,
            gamma=self.gamma,
            delta=self.delta,
            xi=self.xi,
            lambda_=self.lambda_,
            normalize=self.norm,
        )

    def stats(self) -> Dict[str, float]:
        a, b = self.gamma, self.delta
        s, l_ = self.lambda_, self.xi

        mean_ = l_ - s * np.exp(1 / (2 * b**2)) * np.sinh(a / b)

        median_ = l_ + s * np.sinh(-a / b)

        v1 = np.exp(b**-2) * np.cosh(2 * a / b) + 1
        v2 = np.exp(b**-2) - 1
        variance_ = s**2 / 2 * v1 * v2

        return {"mean": mean_, "median": median_, "variance": variance_, "std": np.sqrt(variance_)}
