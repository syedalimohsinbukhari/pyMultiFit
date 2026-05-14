"""Created on Nov 02 18:49:12 2025"""

from __future__ import annotations

from numpy import cosh, expm1, sinh

from .backend import BaseDistribution, errorHandling as erH
from .utilities_d import johnsonSU_cdf_, johnsonSU_log_cdf_, johnsonSU_log_pdf_, johnsonSU_pdf_
from .. import EXP, SQRT, NAN_DICT
from ..typing import ArrayLike, NDArray


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
        if amplitude <= 0 and not normalize:
            raise erH.NegativeAmplitudeError()

        self.amplitude = 1.0 if normalize else amplitude
        self.gamma = gamma
        self.delta = delta
        self.xi = xi
        self.lambda_ = lambda_

        self.norm = normalize

    @classmethod
    def from_scipy_params(cls, a: float, b: float, loc: float = 0.0, scale: float = 1.0) -> "JohnsonSUDistribution":
        return cls(gamma=a, delta=b, xi=loc, lambda_=scale, normalize=True)

    def pdf(self, x: ArrayLike) -> NDArray:
        return johnsonSU_pdf_(
            x,
            amplitude=self.amplitude,
            gamma=self.gamma,
            delta=self.delta,
            xi=self.xi,
            lambda_=self.lambda_,
            normalize=self.norm,
        )

    def logpdf(self, x: ArrayLike) -> NDArray:
        return johnsonSU_log_pdf_(
            x,
            amplitude=self.amplitude,
            gamma=self.gamma,
            delta=self.delta,
            xi=self.xi,
            lambda_=self.lambda_,
            normalize=self.norm,
        )

    def cdf(self, x: ArrayLike) -> NDArray:
        return johnsonSU_cdf_(
            x,
            amplitude=self.amplitude,
            gamma=self.gamma,
            delta=self.delta,
            xi=self.xi,
            lambda_=self.lambda_,
            normalize=self.norm,
        )

    def logcdf(self, x: ArrayLike) -> NDArray:
        return johnsonSU_log_cdf_(
            x,
            amplitude=self.amplitude,
            gamma=self.gamma,
            delta=self.delta,
            xi=self.xi,
            lambda_=self.lambda_,
            normalize=self.norm,
        )

    def stats(self) -> dict[str, float]:
        a, b = self.gamma, self.delta
        s, l_ = self.lambda_, self.xi

        if any(param <= 0 for param in (b, s)):
            return NAN_DICT

        mean_ = l_ - s * EXP(1 / (2 * b ** 2)) * sinh(a / b)

        median_ = l_ + s * sinh(-a / b)

        v1 = EXP(b ** -2) * cosh(2 * a / b) + 1
        v2 = expm1(b ** -2)
        variance_ = s ** 2 / 2 * v1 * v2

        return {"mean": mean_, "median": median_, "variance": variance_, "std": SQRT(variance_)}
