"""Created on Feb 02 03:46:43 2025"""

from ..backend import BaseDistribution, errorHandling as erH
from ..utilities_d import (
    scaled_inv_chi_square_cdf_,
    scaled_inv_chi_square_log_cdf_,
    scaled_inv_chi_square_log_pdf_,
    scaled_inv_chi_square_pdf_,
)
from ... import md_scipy_like, SQRT, INF, NAN_DICT
from ...typing import ArrayLike, NDArray


class ScaledInverseChiSquareDistribution(BaseDistribution):
    def __init__(
        self, amplitude: float = 1.0, df: float = 1.0, scale: float = 1.0, loc: float = 0.0, normalize: bool = False
    ):
        if not normalize and amplitude < 0:
            raise erH.NegativeAmplitudeError()

        self.amplitude = 1 if normalize else amplitude
        self.df = df
        self.scale = scale
        self.tau2 = scale / df

        self.loc = loc
        self.norm = normalize

    @classmethod
    @md_scipy_like("1.0.7")
    def scipy_like(cls, a: float, loc: float = 0.0, scale=1.0):
        return cls(df=a, loc=loc, scale=scale, normalize=True)

    @classmethod
    def from_scipy_params(cls, a: float, loc: float = 0.0, scale=1.0):
        return cls(df=a, loc=loc, scale=scale, normalize=True)

    def pdf(self, x: ArrayLike) -> NDArray:
        return scaled_inv_chi_square_pdf_(
            x, amplitude=self.amplitude, df=self.df, scale=self.scale, loc=self.loc, normalize=self.norm
        )

    def logpdf(self, x: ArrayLike) -> NDArray:
        return scaled_inv_chi_square_log_pdf_(
            x, amplitude=self.amplitude, df=self.df, scale=self.scale, loc=self.loc, normalize=self.norm
        )

    def cdf(self, x: ArrayLike) -> NDArray:
        return scaled_inv_chi_square_cdf_(
            x, amplitude=self.amplitude, df=self.df, scale=self.scale, loc=self.loc, normalize=self.norm
        )

    def logcdf(self, x: ArrayLike) -> NDArray:
        return scaled_inv_chi_square_log_cdf_(
            x, amplitude=self.amplitude, df=self.df, loc=self.loc, scale=self.scale, normalize=self.norm
        )

    def stats(self) -> dict[str, float]:
        v, tau2, loc = self.df, self.tau2, self.loc

        if any(param <= 0 for param in (v, tau2)):
            return NAN_DICT

        mean_ = (v * tau2) / (v - 2)
        mode_ = (v * tau2) / (v + 2)
        variance_ = (2 * v ** 2 * tau2 ** 2) / ((v - 2) ** 2 * (v - 4))

        return {
            "mean": mean_ + loc if v > 2 else INF,
            "mode": mode_ + loc,
            "variance": variance_ if v > 4 else INF,
            "std": SQRT(variance_) if v > 4 else INF,
        }
